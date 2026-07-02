// ============================================================================
// phase1.cuh —— SM90 (H100) 稀疏注意力"第一阶段"kernel
// ============================================================================
//
// 【这段代码解决什么实际问题？】
//
// 大模型推理分两步：预填充（一次性处理用户输入的所有 token）和解码（逐字生成）。
// 稀疏注意力是预填充阶段的一种加速技术：不把所有历史 token 都参与注意力计算，
// 而是用一个 topk 索引数组 `indices` 指定"只关心最重要的 K 个 token"。
//
// 但"最重要"的 K 个 token 是怎么算出来的？这就需要两步：
//   - phase1：用 Q（查询）和被选中的 K/V（keys/values）做注意力，得到部分 softmax 结果。
//             这些部分结果最后会和 phase2 的结果合并，得到最终的 attention 输出。
//   - phase2：处理剩下的（未被 topk 选中的）token，把结果合并进来。
//
// 这个文件就是 phase1。它的输入是 Q、KV、topk 索引；输出是部分 O（attention 输出）、
// max_logits、lse（log-sum-exp，用于 phase2 在线 softmax 的合并）。
//
// 【为什么需要这么复杂的结构？】
//
// GPU 的计算单元叫 SM（Streaming Multiprocessor，可以类比成"工厂的一条产线"）。
// H100 (SM90) 一个 SM 里有 4 个 warpgroup（"产线上的工作组"，每个 128 线程）。
// 这个 kernel 一个 block 启动了 3 个 warpgroup（共 384 线程），分工如下：
//
//   ┌──────────────────────────────────────────────────────────────────┐
//   │  WG0 (消费者)：算左半边 QK^T，左半边 PV，左半边 O                 │
//   │  WG1 (消费者)：算右半边 QK^T，右半边 PV，右半边 O                 │
//   │  Producer WG (生产者)：从 global memory 异步加载 K/V 到 shared   │
//   │                        memory，喂给 WG0/WG1 用                    │
//   └──────────────────────────────────────────────────────────────────┘
//
// 两个消费者 warpgroup 各自算一半，最后把结果合并。生产者只负责搬数据不算数。
//
// 【关键概念第一次出现时的通俗解释】
//
// * TMA (Tensor Memory Accelerator)：H100 新增的硬件单元，专门用来在 global memory
//   和 shared memory 之间异步搬数据。CPU/GPU 代码只下一条"搬这个 tensor"的指令，
//   TMA 硬件自己完成搬运，不占用计算资源。可以类比成"全自动传送带"。
//
// * cp.async：Ampere (SM80) 引入的异步拷贝指令，比 TMA 早一代。这个 kernel 用它
//   从 global memory 加载 K/V 到 shared memory（生产者 warpgroup 干的事）。
//
// * Warpgroup (WG)：128 个线程组成一个 warpgroup，是 H100 上 MMA (Matrix Multiply-
//   Add) 指令的最小调度单位。一个 WG 一次可以算一个 64xN 的 GEMM。
//
// * Shared memory (smem, 共享内存)：SM 内部的快速缓存，所有线程共享。可以类比成
//   "产线上的公共工作台"，比 global memory (主仓库) 快几十倍。
//
// * Register (寄存器)：每个线程私有的最快存储，类比"每个工人的双手"。
//
// * Barrier (屏障)：线程同步原语。一种"约定信号"——一群线程都到达后才能继续，
//   否则等待。这个文件里有两种：
//     - transac_bar_t (mbar)：transaction barrier，可以附带"传输字节数"信息，
//       TMA/cp.async 完成时会自动 arrive，消费者 wait 时就知道数据已就绪。
//     - NamedBarrier：命名屏障，给指定的线程组（如 WG0 或 WG1）同步用。
//
// * Online softmax：传统 softmax 需要先遍历一遍求 max，再遍历一遍求 exp 和 sum。
//   "在线"版本可以一边接收新数据一边更新 max 和 sum，最后归一化。这样能流式处理
//   K/V，不用全装进内存。
//
// * 双缓冲 (Double Buffering)：准备两块共享内存缓冲区（buf0 和 buf1）。消费者在
//   用 buf0 算的时候，生产者同时往 buf1 装下一批数据。下一轮反过来。这样计算和
//   搬运重叠，GPU 不会"等数据"。
//
// * rP/rS/rO/rM/rL：前缀 r 表示 register tensor（存在寄存器里的张量）。
//   - rP = QK^T 的原始结果（logits）
//   - rS = softmax 后的 P（中间结果，用于 PV GEMM）
//   - rO = 输出 O 的累加器
//   - rM = max logits（每行一个，用于 online softmax 的 rescale）
//   - rL = sum of exp（每行一个，最后用于归一化 O）
//
// 【整体流程图】
//
//   Host (CPU) 调用 run() → 设置 TMA 描述符 → 启动 kernel (cluster launch)
//                              ↓
//   Device (GPU) 每个 block 执行 devfunc()：
//     ├─ WG0: 算 QK^T 左半边 → online softmax → 算 PV 左半边 → store O 左半边
//     ├─ WG1: 算 QK^T 右半边 → online softmax → 算 PV 右半边 → store O 右半边
//     └─ Producer: 用 cp.async 把 K/V 从 global 搬到 shared，通过 mbar 通知 WG0/WG1
//
// ============================================================================

#pragma once

#include "config.h"

#include "utils.h"
#include "../../helpers.h"

namespace sm90::fwd {

using namespace cute;

// ----------------------------------------------------------------------------
// 辅助函数 1/3：st_global_cs_128
// ----------------------------------------------------------------------------
// 【解决什么问题】需要把 4 个 float（共 16 字节）一次性写到 global memory，
// 并且要用特定的 cache 策略（cs = streaming，提示硬件"这个数据只写一次不读，
// 别污染 L2 cache"）。普通 store 指令做不到 128-bit 一次性写 + streaming 策略，
// 所以用 PTX 内联汇编（inline assembly）直接发指令。
//
// 【参数】4 个 float 值 + 目标地址。一次写 16 字节。
// 【PTX 解释】st.weak = 弱顺序写（不强制全局内存顺序）; global = 写到 global
// memory; cs = cache-streaming（不进 L2）; v4.f32 = 4 个 float 向量写。
CUTE_DEVICE void st_global_cs_128(float f0, float f1, float f2, float f3, void *dst_ptr) {
    asm volatile("st.weak.global.cs.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 :
                 : "l"(dst_ptr),                              // "l" = 64-bit 指针
                   "f"(f0), "f"(f1), "f"(f2), "f"(f3)          // "f" = float 寄存器
                );
}

// ----------------------------------------------------------------------------
// 辅助函数 2/3：__shfl_xor_sync_float2
// ----------------------------------------------------------------------------
// 【解决什么问题】warp 内线程间通信。CUDA 的 __shfl_xor_sync 可以让线程 i 和
// 线程 i^offset 交换一个 32-bit 值（异或交换，butterfly 模式）。但我们要交换
// float2（64-bit，两个 float），原语只支持 32-bit，所以把 float2 重新解释成
// long long，一次交换 64-bit。
//
// 【用在哪】online softmax 时，需要把每个线程算出的"行 max"在 warp 内做规约
// （取最大值）。用 xor 0x1 配对相邻线程，xor 0x2 配对隔一个线程，两次后整个
// 4 线程小组就拿到共同的最大值。
CUTE_DEVICE
float2 __shfl_xor_sync_float2(
    uint32_t mask, float2 value, int offset
) {
    float2 res;
    // 把 float2 (8 字节) 当 long long (8 字节) 处理，做 64-bit 异或交换
    *reinterpret_cast<long long*>(&res) = __shfl_xor_sync(
        mask,
        *reinterpret_cast<long long*>(&value),
        offset
    );
    return res;
}

// ----------------------------------------------------------------------------
// 辅助函数 3/3：tma_bulk_reduce_add
// ----------------------------------------------------------------------------
// 【解决什么问题】多个 block 算同一个输出位置时，需要把它们的中间结果累加。
// 普通做法是：每个 block 写自己的结果到 global memory，再启动一个 reduce kernel
// 累加。这样要多一次内存往返。
//
// 这个函数用 H100 的 TMA 硬件直接做"原子加"：把 shared memory 里的数据加到
// global memory 上，由 TMA 硬件完成，不需要额外的 reduce kernel。
//
// 【用在哪】phase1 的输出 O 会被多个 block 写到同一位置（不同 block 处理不同
// 的 topk 块），用这个指令做 in-place 累加。
//
// 【PTX 解释】cp.reduce.async.bulk = TMA 异步批量归约拷贝; global.shared::cta
// = 从 smem 到 gmem; add.f32 = 用 float 加法归约; bulk_group = 配套 barrier
// 同步用。
CUTE_DEVICE
void tma_bulk_reduce_add(void const* src_ptr, void* dst_ptr, int32_t store_bytes) {
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(src_ptr);  // smem 地址转成 uint 句柄
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                     :
                     : "l"(dst_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
}

// ============================================================================
// devfunc —— 整个 kernel 的"设备端入口"
// ============================================================================
//
// 【这个函数做什么】被每个 thread block 调用一次。一个 block 处理"一个序列的
// 一组 Q 头"（具体是 s_q_idx 序列的 q_h_idx 这一批 B_H=64 个头）。block 内部
// 384 个线程分成 3 个 warpgroup，分别走 WG0 / WG1 / Producer 三条分支。
//
// 【模板参数】
//   - D_QK：Q/K 的维度（512 或 576，对应不同的模型架构）
//   - HAVE_TOPK_LENGTH：每个序列的 topk 长度是否可变（true 时从 topk_length 数组读）
//
// 【条件编译】只在 SM90 (H100, __CUDA_ARCH__ == 900) 上编译。CLION/VSCODE 宏
// 是为了让 IDE 不报错（IDE 不知道目标 arch）。
//
template<int D_QK, bool HAVE_TOPK_LENGTH>
template<typename TMAParams>
__device__ void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::devfunc(const SparseAttnFwdParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // ====== 步骤 1：确定本 block / 本线程的身份 ======
    // blockIdx.x 编码了两层信息：(序列号 s_q_idx, 头组号 q_h_idx)
    // 把 blockIdx.x 拆开：低 bits 是头组号，高 bits 是序列号
    const int q_h_idx = blockIdx.x % (params.h_q/B_H);   // 本 block 处理哪一组 B_H 个 Q 头
    const int s_q_idx = blockIdx.x / (params.h_q/B_H);   // 本 block 处理哪个序列
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();  // 0/1/2：本线程属于哪个 WG
    const int warp_idx = cutlass::canonical_warp_idx_sync();        // 0..11：warp 编号（每 WG 4 个 warp）
    const int idx_in_warpgroup = threadIdx.x % 128;                 // 0..127：本线程在 WG 内的编号

    // ====== 步骤 2：建立 shared memory 视图 ======
    // extern __shared__ 是动态共享内存——大小由 host 端在 launch 时指定。
    // SharedMemoryPlan 定义了这块内存的布局（Q、O、K、S 缓冲区、barrier 等）。
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    // sQ / sO / sS0 / sS1 都是 shared memory 上的 tensor 视图（CuTe tensor），
    // 给后面 GEMM 用。sQ 是 Q 的 smem 副本，sO 是输出 O 的 smem 暂存区，
    // sS0/sS1 是两个 WG 之间交换 rS（softmax 后的 P）的中转站。
    Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
    Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data()), SmemLayoutO{});
    // 注意 sS0 的特殊处理：V3.2 模型 (D_QK=576) 复用 sK0 的 RoPE 部分内存省空间；
    // MODEL1 (D_QK=512) 用独立的 s[1] 缓冲区。
    Tensor sS0 = make_tensor(make_smem_ptr(D_QK == 576 ? plan.k[0].data()+64*512 : plan.s[1].data()), SmemLayoutS{});    // Overlap with sK0's RoPE part for V3.2
    Tensor sS1 = make_tensor(make_smem_ptr(plan.s[0].data()), SmemLayoutS{});

    // ====== 步骤 3：单线程初始化 barrier（只用 warp 0 的一个线程做） ======
    if (warp_idx == 0 && elect_one_sync()) {
        // 预取 TMA 描述符到 L1 cache，后面真正发 TMA 拷贝时少一次访存延迟
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_O);

        // 初始化所有 mbar（transaction barrier）。参数是 arrival count——多少次
        // arrive 后屏障解锁。例如 init(128) 表示要 128 次 arrive（一个 WG 128 线程，
        // 每个线程 arrive 一次，或者 cp.async 完成时 arrive 一次 + 人工 arrive 补足）。
        plan.bar_q.init(1);                                    // Q 加载完成屏障，1 次 TMA arrive
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            plan.bar_k0_free[i].init(128);                     // sK 缓冲区 i 空闲了（消费者用完）
            plan.bar_k0_ready[i].init(128);                    // sK 缓冲区 i 数据已就绪（生产者装完）
            plan.bar_k1_free[i].init(128);
            plan.bar_k1_ready[i].init(128);
        }
        plan.bar_is_kv_valid_ready.init(16);                   // 16 = 1 个 warp（4 线程 × 4 次 arrive？）
        fence_barrier_init();                                  // 让上面的 init 在所有线程可见前不往后走
    }

    __syncthreads();

    // ====== 步骤 4：算本 block 要处理几个 topk 块 ======
    // topk_length 是每个序列独立的"实际选了多少个 token"（如果支持变长）。
    // num_topk_blocks = 一共要处理几个 B_TOPK=64 大小的块。
    const int topk_length = HAVE_TOPK_LENGTH ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int num_topk_blocks = HAVE_TOPK_LENGTH ? ku::ceil_div(topk_length, (int)B_TOPK) : (int)((unsigned int)params.topk/(unsigned int)B_TOPK);

    // ==========================================================================
    // 消费者分支（WG0 + WG1）
    // ==========================================================================
    // 这两个 warpgroup 干计算活：算 QK^T、online softmax、PV、写回 O。
    // 每个 WG 各算一半（D_V/2 = 256 维），最后合并。
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
        // 给消费者 WG 分配 216 个寄存器/线程（多寄存器 = 多算力，因为要装 rO/rP/rS）
        cutlass::arch::warpgroup_reg_alloc<216>();

        // ====== 步骤 5：用 TMA 异步加载 Q 到 shared memory ======
        // 只让一个线程发 TMA 指令（TMA 是单线程发起、硬件完成）。
        if (warp_idx == 0 && elect_one_sync()) {
            // gQ 是 Q 在 global memory 的视图，切成本 block 需要的那一片
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H>, Int<D_Q>>{}
            )(_, _, q_h_idx, _0{});
            // 发 TMA 拷贝：gQ → sQ，附带 barrier（拷完会 arrive bar_q）
            // EVICT_FIRST 提示：Q 只用一次，搬完别留在 L2 cache 占地方
            launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
            // 告诉 bar_q "预计要收到 B_H*D_Q*sizeof(bf16) 字节"，凑够才解锁
            plan.bar_q.arrive_and_expect_tx(B_H*D_Q*sizeof(bf16));
        }

        // ====== 步骤 6：初始化寄存器累加器 ======
        // rM[r] = 第 r 行的"历史 max logits"（online softmax 用）
        // rL[r] = 第 r 行的"历史 sum(exp)"（online softmax 用）
        // 初始化为 -1e30 和 0，意味着"还没看到任何 token"
        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL}; // Meaning: the `max_logits` used for O / rL calculation
        float rL[2] = {0.0f, 0.0f};
        // rO = 输出 O 的累加器（在寄存器里累加 PV 的结果）
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<D_V/2>>{});
        // rP = QK^T 的结果（每个线程持有 B_H x B_TOPK 的一部分）
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
        // rS = softmax(P) 后存 bf16，作为下一次 PV GEMM 的输入 A
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<B_TOPK>>{}));
        cute::fill(rO, 0.0f);

        // 等 Q 加载完成（消费者要用 Q 算 QK^T）
        plan.bar_q.wait(0);

        // barrier phase：双缓冲用 phase 翻转（0/1 交替），避免 ABA 问题
        // 每过一轮翻转一次，wait 时带 phase 参数能区分"这次 arrive"还是"上次 arrive"
        bool cur_bar_wait_phase = 0;

        // 用空 struct 当类型 tag，编译期区分 WG0 / WG1（让 if constexpr 在编译期分支）
        struct Warpgroup0 {};
        struct Warpgroup1 {};

        // ==========================================================================
        // Lambda 1/5：qkt_gemm_one_tile —— 算一块 QK^T
        // ==========================================================================
        // 做一次 sQ @ sK^T 的 GEMM，结果累加到 rP。
        // 一个 tile = 64 行 Q x 64 列 K（B_H x 64 的一块 K，B_TOPK=64 个 token）
        // sK 的位置由 IS_WG1 决定：WG0 用 plan.k[0]，WG1 用 plan.k[1]（两个 WG 用不同的 smem 缓冲区）
        // clear_accum=true 时清空 rP 再算（每轮新 block 的第一次）
        auto qkt_gemm_one_tile = [&](auto warpgroup_idx, int tile_idx, bool clear_accum) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            TiledMMA tiled_mma_QK = TiledMMA_QK{};
            // sQ_tile：sQ 沿 D_Q 方向切片，取第 tile_idx 个 64 维片段
            Tensor sQ_tile = flat_divide(sQ, Tile<Int<B_H>, Int<64>>{})(_, _, _0{}, tile_idx);
            // sK_tile：plan.k[IS_WG1] 里第 tile_idx 个 64-token 片段
            Tensor sK_tile = make_tensor(make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx*B_TOPK*64), SmemLayoutKTiles<1>{});
            // gemm_ss = shared-shared GEMM，结果在寄存器 rP
            gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP, idx_in_warpgroup);
        };

        // ==========================================================================
        // Lambda 2/5：mask_rP —— 把无效 token 的 QK^T 结果置为 -INF
        // ==========================================================================
        // 【为什么要 mask】topk 索引里可能有 -1（padding，凑齐 B_TOPK 整数倍），
        // 或者 HAVE_TOPK_LENGTH 时实际有效 token 数小于 topk。这些位置的 K 是垃圾数据，
        // QK^T 算出的 rP 没意义，必须置为 -INF，softmax 后才是 0，不影响结果。
        // is_kv_valid[][] 是生产者 WG 在加载时填好的标记数组。
        auto mask_rP = [&](auto warpgroup_idx) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            // 等生产者把 is_kv_valid 写好
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            // 每个线程负责 rP 里的若干元素，根据 is_kv_valid 把无效位置设成 -INF
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    int col = 8*(i/4) + (idx_in_warpgroup%4)*2;
                    if (!plan.is_kv_valid[IS_WG1][col]) rP(i) = -INFINITY;
                    if (!plan.is_kv_valid[IS_WG1][col+1]) rP(i+1) = -INFINITY;
                }
            }
        };

        // ==========================================================================
        // Lambda 3/5：online_softmax_and_rescale_o —— 在线 softmax + 重缩放 O
        // ==========================================================================
        // 这是整个 kernel 最核心的算法。
        //
        // 【问题】传统 softmax 需要先看所有数据求 max，再看一遍求 exp 和 sum。
        // 但我们是流式处理 K/V（一次来 64 个 token），不能存所有 P。
        //
        // 【在线 softmax 公式】
        //   假设前面已经看到 m_old (max), l_old (sum exp), O_old (累加输出)
        //   现在新来一批，max 是 m_cur，要算新的 m_new = max(m_old, m_cur)
        //   则：l_new = l_old * exp(m_old - m_new) + sum(exp(p_i - m_new))
        //        O_new = O_old * exp(m_old - m_new) + sum(p_i * exp(p_i - m_new) * v_i)
        //   也就是 O 和 l 都要乘以一个"重缩放因子" exp(m_old - m_new)。
        //
        // 【WG0 vs WG1 的差异】
        //   - WG0：old_max 来自自己的 rM（上一轮的 max）
        //   - WG1：old_max 来自 sM（WG0 写到 smem 给 WG1 读）
        //   两个 WG 算的是 Q 的不同列，max 必须一致才能合并，所以 WG0 先算完写 sM，WG1 读。
        auto online_softmax_and_rescale_o = [&](auto warpgroup_idx) {
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            const float scale = params.sm_scale_div_log2;   // QK^T 通常要除 sqrt(d_k)，提前算好
            float r_sM[2];
            if constexpr (IS_WG1) {
                // WG1 从 smem 读 WG0 写的 old_max
                *(float2*)r_sM = plan.sM[idx_in_warpgroup/4];
            }
            float new_maxs[2];
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                // ----- 第 1 步：求本线程负责部分的行 max -----
                float cur_max = -INFINITY;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    cur_max = max(cur_max, max(rP(i), rP(i+1)));
                }
                // warp 内 4 个线程做 butterfly reduce（xor 1 配对相邻，xor 2 配对隔一个）
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
                cur_max *= scale;

                // ----- 第 2 步：算 new_max = max(old_max, cur_max) -----
                // For WG1, old_max comes from sM (written by WG0); for WG0, old_max comes from rM (read by WG0 from sM in the last round)
                new_maxs[row_idx] = max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);

                // ----- 第 3 步：重缩放 O -----
                // O 之前是基于 old_max 累加的，现在 max 变大了，要乘 exp(old_max - new_max) 缩小
                float scale_for_o = exp2f(rM[row_idx]-new_maxs[row_idx]);
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rO); i += 4) {
                    rO(i) *= scale_for_o;
                    rO(i+1) *= scale_for_o;
                }

                // ----- 第 4 步：算 rS = softmax(P) = exp(P - new_max) -----
                // 同时累加 cur_sum = sum(exp(p - new_max))
                float cur_sum = 0;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    rP(i) = exp2f(rP(i)*scale - new_maxs[row_idx]);
                    rP(i+1) = exp2f(rP(i+1)*scale - new_maxs[row_idx]);
                    rS(i) = (bf16)rP(i);       // rS 转 bf16，给 PV GEMM 用
                    rS(i+1) = (bf16)rP(i+1);
                    cur_sum += rP(i) + rP(i+1);
                }
                // 更新 rL：旧 l * scale_for_o + 新增 sum
                rL[row_idx] = rL[row_idx]*scale_for_o + cur_sum;
            }
            __syncwarp();
            // 每 4 个线程里第 0 个把 new_maxs 写到 sM，给 WG1 用（WG0 自己也更新 rM）
            if (idx_in_warpgroup%4 == 0) {
                plan.sM[idx_in_warpgroup/4] = *(float2*)new_maxs;
            }
            rM[0] = new_maxs[0];
            rM[1] = new_maxs[1];
        };

        // ==========================================================================
        // Lambda 4/5：reduce_L —— 跨 WG0/WG1 规约 rL
        // ==========================================================================
        // 【为什么要 reduce】最终输出 O = O / l（softmax 归一化）。
        // 但 rL 是每个 WG 各自累加的，只覆盖一半 V 维度。要合并两个 WG 的 rL 才是完整 l。
        // 流程：warp 内 butterfly reduce → 写到 smem → 跨 WG 用 NamedBarrier 同步 → 读对方 WG 的 l → 加起来。
        auto reduce_L = [&]() {
            // Reduce L
            // For example, thread 0 reduces with thread 1, 2, and 3, as well as thread 128, 129, 130, and 131
            // 步骤 1：warp 内 4 线程 butterfly reduce（thread 0 配对 1, 2, 3）
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
            // 步骤 2：每 4 个线程的第 0 个把 rL 写到 sL
            if (idx_in_warpgroup%4 == 0)
                plan.sL[threadIdx.x/4] = *(float2*)(rL);
            // 步骤 3：256 个线程（WG0+WG1）在 NamedBarrier 上同步
            NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
            // 步骤 4：读"对偶线程"（threadIdx.x^32）的 sL，加到自己 rL 上
            float2 peer_L = plan.sL[(threadIdx.x/4)^32];
            rL[0] += peer_L.x;
            rL[1] += peer_L.y;
        };

        // ==========================================================================
        // Lambda 5/5：store_O —— 把累加器 rO 写回 global memory
        // ==========================================================================
        // 做三件事：
        //   1) 用 rL 归一化 rO（O = O / l）
        //   2) rO (float) → bf16，从寄存器写到 shared memory (sO)
        //   3) 用 TMA 把 sO 写到 global memory 的 O 张量
        // 还要处理 attn_sink（注意力下沉，一种稳定数值的技术：往 l 里加一个常数项）。
        auto store_O = [&]() {
            // ----- 第 1 步：算归一化因子 scale = 1/l -----
            // attn_sink 项：如果设了 attn_sink，l 里多加一项 exp(attn_sink - m)
            // 如果 rL==0（这行没有任何有效 token），scale=0 让输出为 0
            float scale_factors[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[q_h_idx*B_H + get_AorC_row_idx(i, idx_in_warpgroup)]*CUDART_L2E_F;
                scale_factors[i] = 1.0f / (rL[i] + exp2f(attn_sink - rM[i]));
                if (rL[i] == 0.0f)
                    scale_factors[i] = 0.0f;    // The output should be 0 whatever attn_sink is
            }

            // ----- 第 2 步：准备 sO 的 smem 视图和 STSM 地址 -----
            // STSM (Store-to-Shared-Matrix) 是 H100 的指令，让 warp 把寄存器数据
            // 直接写到 smem 的矩阵布局，比普通 store 快。
            Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data() + warpgroup_idx*B_H*(D_V/2)), SmemLayoutOTiles<4>{});
            bf16* stsm_addrs[4];
            int stsm_row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%16);
            CUTE_UNROLL
            for (int i = 0; i < 64/16; ++i) {
                stsm_addrs[i] = &sO(stsm_row, (idx_in_warpgroup%32/16*8) + 16*i);
            }
            // 只让 warp 的第 0 线程发 TMA store
            bool s2g_pred = warp_idx%4 == 0 && elect_one_sync();

            // 等所有 PV GEMM 完成
            warpgroup_wait<0>();
            // 把 rO 按 tile 顺序写到 smem 再到 gmem
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < (D_V/2)/64; tile_idx += 1) {
                // ----- 第 3 步：rO (float) × scale → bf16 -----
                constexpr int NUM_ELEMS_EACH_TILE = B_H*64 / 128;   // 64: tile size, 128: warpgroup size
                bf16 cur_rOb[NUM_ELEMS_EACH_TILE];
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_EACH_TILE; ++i) {
                    cur_rOb[i] = (bf16)(rO(tile_idx*NUM_ELEMS_EACH_TILE + i) * scale_factors[i%4>=2]);
                }
                // ----- 第 4 步：R → S（寄存器 → 共享内存）用 STSM 指令 -----
                CUTE_UNROLL
                for (int i = 0; i < 64/16; ++i) {
                    SM90_U32x4_STSM_N::copy(
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 0),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 2),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 4),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 6),
                        *reinterpret_cast<uint128_t*>(stsm_addrs[i] + tile_idx*(B_H*64))
                    );
                }
                // STSM 是异步的，fence 让后面的 WG 间同步看到写完
                fence_view_async_shared();
                // WG 内 128 线程同步（保证这个 tile 的 sO 写完，TMA 才发）
                NamedBarrier::arrive_and_wait(128, warpgroup_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
                // ----- 第 5 步：S → G（共享内存 → global memory）用 TMA -----
                if (s2g_pred) {
                    int g_tile_idx = warpgroup_idx*4 + tile_idx;
                    SM90_TMA_STORE_3D::copy(
                        &tma_params.tensor_map_O,
                        plan.q_o.o.data() + g_tile_idx*(B_H*64),
                        g_tile_idx*64,
                        q_h_idx*B_H,
                        s_q_idx
                    );
                }
            }
            // 通知 TMA store 队列：本 block 的 store 都发完了
            cute::tma_store_arrive();
        };


        // ==========================================================================
        // WG0 主循环
        // ==========================================================================
        // WG0 负责"左半边"计算：
        //   - 算 Q @ K^T 的左半（前 4-5 个 64-d tile）
        //   - PV 用本 WG 自己的 rS（local P）算左半 V
        //   - 还要算 PV 用 WG1 的 rS（remote P，通过 sS1 传过来）的左半 V
        //   - 两个 PV 加起来形成完整的左半 O
        if (warpgroup_idx == 0) {
            // Warpgroup 0

            // ----- Lambda：等 sK 左半就绪 + 算 4 个 QK^T tile -----
            // 左半 = D_Q 的前 4 个 64 维 tile (tile 0/1/2/3)
            // 第一次（block_idx==0）clear_accum=true 清空 rP，之后累加
            auto pipelined_wait_and_qkt_gemm_l = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[0].wait(cur_bar_wait_phase);  // 等生产者装满 sK buf 0
                qkt_gemm_one_tile(Warpgroup0{}, 0, true);       // 第一个 tile 清空 rP
                qkt_gemm_one_tile(Warpgroup0{}, 1, false);      // 后续 tile 累加到 rP
                qkt_gemm_one_tile(Warpgroup0{}, 2, false);
                qkt_gemm_one_tile(Warpgroup0{}, 3, false);
                warpgroup_commit_batch();                       // 提交这批 GEMM（异步执行）
            };

            // ----- Lambda：等 sK 右半就绪 + 算右半 QK^T -----
            // 右半 = D_Q 的后 4-5 个 64 维 tile (tile 4/5/6/7[, 8])
            // D_QK=576 时多一个 tile 8（576 = 9*64，而 512 = 8*64）
            auto pipelined_wait_and_qkt_gemm_r = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[1].wait(cur_bar_wait_phase);  // 等生产者装满 sK buf 1
                qkt_gemm_one_tile(Warpgroup0{}, 4, false);
                qkt_gemm_one_tile(Warpgroup0{}, 5, false);
                qkt_gemm_one_tile(Warpgroup0{}, 6, false);
                qkt_gemm_one_tile(Warpgroup0{}, 7, false);
                if constexpr (D_QK == 576) {
                    qkt_gemm_one_tile(Warpgroup0{}, 8, false);  // V3.2 模型多一个 tile
                }
                warpgroup_commit_batch();
            };

            // ----- Lambda：用 scales 重缩放 rS -----
            // 当 WG0 收到 WG1 的新 sM 后，rS 也要相应 rescale（保持数值一致）
            auto scale_rS = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row*2; i < size(rP); i += 4) {
                        rS(i) = (bf16)(rP(i) * scales[row]);
                        rS(i+1) = (bf16)(rP(i+1) * scales[row]);
                    }
                }
            };

            // ----- Lambda：用 scales 重缩放 rO 和 rL -----
            auto rescale_rO = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row*2; i < size(rO); i += 4) {
                        rO(i) *= scales[row];
                        rO(i+1) *= scales[row];
                    }
                    rL[row] *= scales[row];
                }
            };

            // ====== 主循环：每次处理 2 个 topk 块（双缓冲） ======
            // 每次迭代处理 block_idx 和 block_idx+1 两个块。
            // 这两个块轮流用 sK buf 0 和 buf 1，实现"算 buf 0 时装 buf 1"。
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                // sV0l / sV1l = buf 0 / buf 1 的 V 视图（K 转置后就是 V，因为 MLA K=V）
                Tensor sV0l = make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTilesTransposed<4>{});
                Tensor sV1l = make_tensor(make_smem_ptr(plan.k[1].data()), SmemLayoutKTilesTransposed<4>{});

                if (block_idx == 0) {
                    // 第一轮特殊处理：先发起 QK^T GEMM，否则循环里没东西可算
                    // NOTE: We put this code here to avoid register spilling
                    pipelined_wait_and_qkt_gemm_l();
                    pipelined_wait_and_qkt_gemm_r();
                    warpgroup_wait<0>();   // 等 GEMM 完成
                }

                // ----- 步骤 A：mask + online softmax，告诉 WG1 我算完了 -----
                // Online softmax, inform WG1
                mask_rP(Warpgroup0{});


                online_softmax_and_rescale_o(Warpgroup0{});
                // 通知 WG1："本 WG 的 bunch 0 准备好了"（sM 已写）
                NamedBarrier::arrive(256, NamedBarriers::wg0_bunch_0_ready);

                // ----- 步骤 B：用本 WG 的 rS 算左半 PV (rO += rS @ sV0l) -----
                // gemm_rs = register-shared GEMM（A 在寄存器 rS，B 在 smem sV0l）
                // Issue rO0 += rS0 @ sV0l
                gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV0l, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                // buf 0 用完了，告诉生产者"可以装下一批 K/V 进 buf 0 了"
                // Mark V0L as free
                warpgroup_wait<0>();
                plan.bar_k0_free[0].arrive();

                // ----- 步骤 C：等 WG1 的 sM，rescale rS，写到 sS0 给 WG1 用 -----
                // Wait for new sM, scale rS, save, inform WG1
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
                float new_rM[2], scale_factors[2];
                *(float2*)new_rM = plan.sM[idx_in_warpgroup/4];   // 读 WG1 写的 new_max
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    scale_factors[i] = exp2f(rM[i] - new_rM[i]);  // rS 要乘这个因子才和 WG1 的 max 对齐
                    rM[i] = new_rM[i];
                }
                scale_rS(scale_factors);
                // 把 rS 存到 sS0（用 STSM 指令），让 WG1 能读到（WG1 用 sS0 算右半 PV）
                save_rS_to_sS(rS, sS0, idx_in_warpgroup);
                fence_view_async_shared();
                NamedBarrier::arrive(256, NamedBarriers::wg0_s0_ready);  // 告诉 WG1 "sS0 就绪"

                // ----- 步骤 D：等 WG1 写完 sS1，用它算右半 PV -----
                // Wait for sS1
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_s1_ready);

                // 先 rescale rO 和 rL（对齐 WG1 的 max），再用 sS1 算 PV
                // Rescale rO0, Issue rO0 += sS1 @ sV1L
                rescale_rO(scale_factors);
                gemm_ss(false, TiledMMA_PV_RemoteP{}, sS1, sV1l, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                // 翻转 phase（双缓冲的下一轮用另一个 phase）
                cur_bar_wait_phase ^= 1;

                if (block_idx+2 < num_topk_blocks) {
                    // 还有下一轮，提前发起下一轮的 QK^T（生产者已经在装下一批 K/V）
                    // Launch the next QK^T GEMM
                    pipelined_wait_and_qkt_gemm_l();

                    // buf 1 用完了，告诉生产者
                    // Mark V1L as free
                    warpgroup_wait<1>();
                    plan.bar_k1_free[0].arrive();
                    pipelined_wait_and_qkt_gemm_r();

                    // Wait for rP0 = sQ @ sK0
                    warpgroup_wait<0>();
                } else {
                    // 最后一轮：buf 1 用完直接释放，不再发新 QK^T
                    // Mark V1L as free
                    warpgroup_wait<0>();
                    plan.bar_k1_free[0].arrive();
                }
            }

            reduce_L();
            store_O();
        } else {
            // ==========================================================================
            // WG1 主循环（与 WG0 镜像对称）
            // ==========================================================================
            // WG1 负责"右半边"计算：
            //   - 算 Q @ K^T 的右半（与 WG0 用不同的 sK 缓冲区）
            //   - PV 用本 WG 的 rS 算右半 V
            //   - 还要算 PV 用 WG0 的 rS（remote P，通过 sS0 传过来）的右半 V
            //
            // 【为什么 WG1 的循环结构比 WG0 简单】
            // WG0 要先算 sM 给 WG1 读，所以 WG0 的循环里有"先算 sM 再通知 WG1"的步骤。
            // WG1 反过来：先等 WG0 的 sM，再算自己的 sM（合并 max）写回去给 WG0。
            // 所以 WG1 的 online_softmax 比较被动——读 sM、合并、再写 sM。
            // Warpgroup 1

            // ----- Lambda：等 sK 就绪 + 算全部 QK^T tile（右半先，左半后） -----
            // 注意顺序：WG1 先算右半（tile 4-8），再算左半（tile 0-3），与 WG0 相反。
            // 这是为了让两个 WG 用不同的 sK 缓冲区并发算，避免 smem 冲突。
            auto pipelined_wait_and_qkt_gemm = [&]() __attribute__((always_inline)) {
                plan.bar_k1_ready[1].wait(cur_bar_wait_phase);   // 等右半 sK 就绪
                qkt_gemm_one_tile(Warpgroup1{}, 4, true);        // 第一个 tile 清空 rP
                qkt_gemm_one_tile(Warpgroup1{}, 5, false);
                qkt_gemm_one_tile(Warpgroup1{}, 6, false);
                qkt_gemm_one_tile(Warpgroup1{}, 7, false);
                if constexpr (D_QK == 576) {
                    qkt_gemm_one_tile(Warpgroup1{}, 8, false);   // V3.2 多一个 tile
                }
                plan.bar_k1_ready[0].wait(cur_bar_wait_phase);   // 等左半 sK 就绪
                qkt_gemm_one_tile(Warpgroup1{}, 0, false);
                qkt_gemm_one_tile(Warpgroup1{}, 1, false);
                qkt_gemm_one_tile(Warpgroup1{}, 2, false);
                qkt_gemm_one_tile(Warpgroup1{}, 3, false);
                warpgroup_commit_batch();
            };

            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                // sV0r / sV1r = buf 0 / buf 1 的"右半 V"视图（D_V/2=256 维那部分）
                // 注意偏移 +64*256：跳过左半 256 维，只取右半 256 维
                Tensor sV0r = make_tensor(make_smem_ptr(plan.k[0].data()+64*256), SmemLayoutKTilesTransposed<4>{});
                Tensor sV1r = make_tensor(make_smem_ptr(plan.k[1].data()+64*256), SmemLayoutKTilesTransposed<4>{});

                // ----- 步骤 A：算 QK^T，等 GEMM 完成 -----
                // Issue rP1 = sQ @ sK1, and wait
                pipelined_wait_and_qkt_gemm();
                warpgroup_wait<0>();

                // ----- 步骤 B：mask 无效 token -----
                mask_rP(Warpgroup1{});


                // ----- 步骤 C：等 WG0 的 sM，做 online softmax，通知 WG0 -----
                // WG1 必须等 WG0 算完 sM（因为 WG1 要把 WG0 的 max 和自己的 max 合并）
                // Wait for WG0 (for sM), online softmax, Notify WG0 (sM ready)
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
                online_softmax_and_rescale_o(Warpgroup1{});
                NamedBarrier::arrive(256, NamedBarriers::wg1_bunch_0_ready);


                // ----- 步骤 D：用本 WG 的 rS 算右半 PV (rO += rS @ sV1R) -----
                // Issue rO1 += rS1 @ sV1R
                gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV1r, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                // ----- 步骤 E：把 rS 写到 sS1（给 WG0 用），等 WG0 的 sS0，算 remote PV -----
                // 这里先把 rS 存到 sS1 再 wait，是为了重叠"存 rS"和"等 WG0"的时间
                // Wait for WG0 (for sS0), Issue rO1 += rS0 @ sV0R
                save_rS_to_sS(rS, sS1, idx_in_warpgroup);   // Put it here is faster
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_s0_ready);
                gemm_ss(false, TiledMMA_PV_RemoteP{}, sS0, sV0r, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                // ----- 步骤 F：通知 WG0 "sS1 就绪"，等 GEMM 完，释放 sV 缓冲区 -----
                // Save rS1, inform WG0
                fence_view_async_shared();
                NamedBarrier::arrive(256, NamedBarriers::wg1_s1_ready);

                // 等右半 PV GEMM 完成，告诉生产者 buf 1 可以重新装
                // Wait for GEMM, and inform that sV1R is free
                warpgroup_wait<1>();
                plan.bar_k1_free[1].arrive();

                // 等左半 PV GEMM 完成，告诉生产者 buf 0 可以重新装
                // Wait for GEMM, and inform that sV0R is free
                warpgroup_wait<0>();
                plan.bar_k0_free[1].arrive();

                cur_bar_wait_phase ^= 1;
            }

            // 收尾：跨 WG 归约 rL + 写回 O
            reduce_L();
            store_O();

            // ----- 额外步骤：WG1 还要写 max_logits 和 lse 到 global memory -----
            // 这两个值用于 phase2 在线 softmax 合并（phase2 要把 phase1 的部分结果合并进去，
            // 必须知道 phase1 的 max 和 lse 才能正确归一化）。
            // 【为什么是 WG1 写而不是 WG0】因为两个 WG 共享同一份 final_max_logits/final_lse
            // smem 缓冲区，让 WG1 写可以避免额外的同步。WG0 的 reduce_L/store_O 完成后
            // 已经把数据写到 smem，WG1 在自己的 reduce_L/store_O 后再写一次就完整了。
            // Save lse
            if (idx_in_warpgroup%4 == 0) {
                for (int row = 0; row < 2; ++row) {
                    int real_row = get_AorC_row_idx(row, idx_in_warpgroup);
                    bool is_no_valid_tokens = rL[row] == 0.0f;
                    // rM 存的是 log2 域的 max，转回自然 log 域要乘 ln2
                    // lse = log(sum exp) + max = log(rL) + rM*ln2
                    plan.final_max_logits[real_row] = is_no_valid_tokens ? -INFINITY : rM[row]*CUDART_LN2_F;
                    plan.final_lse[real_row] = is_no_valid_tokens ? +INFINITY : logf(rL[row]) + rM[row]*CUDART_LN2_F;
                }
                fence_view_async_shared();
            }

            // 等 WG1 内所有线程把 final_max_logits / final_lse 写完
            NamedBarrier::arrive_and_wait(128, NamedBarriers::warpgroup1_sync);
            // 只让一个线程发 BULK_COPY（s2g），把 smem 的两个数组拷到 global memory
            if (idx_in_warpgroup == 0) {
                int g_offset = s_q_idx*params.h_q + q_h_idx*B_H;
                SM90_BULK_COPY_S2G::copy(plan.final_max_logits, params.max_logits + g_offset, B_H*sizeof(float));
                SM90_BULK_COPY_S2G::copy(plan.final_lse, params.lse + g_offset, B_H*sizeof(float));
                cute::tma_store_arrive();
            }
        }
    } else {
        // ==========================================================================
        // 生产者 warpgroup（Producer）
        // ==========================================================================
        // 【这组人做什么】不算任何 GEMM，只干一件事：把 K/V 从 global memory 异步加载
        // 到 shared memory，喂给 WG0/WG1 用。它是"传送带工人"，不参与计算。
        //
        // 【为什么要单独一个 WG 干这个】如果让 WG0/WG1 自己发 cp.async 指令搬数据，
        // 它们的寄存器会被搬运逻辑占用，没法专心算 GEMM。把搬运分离出来后，
        // WG0/WG1 可以连续算 GEMM 不被打断，吞吐量高得多。这叫"软件流水线"。
        //
        // 【寄存器分配】生产者只需要 72 个寄存器/线程（消费者要 216 个）。
        // warpgroup_reg_dealloc 把寄存器让给消费者 WG，让它们能装下大块 rO/rP/rS。
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<72>();

        // ====== 步骤 1：把 128 个线程分成 16 组，每组负责一段 K/V ======
        // GROUP_SIZE=8 表示一组 8 个线程，NUM_GROUPS=16 表示共 16 组。
        // 每组负责 B_TOPK/16 = 4 行 token，每行 64 维。
        constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/GROUP_SIZE;
        constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
        int idx_in_group = idx_in_warpgroup % GROUP_SIZE;   // 组内编号 0..7
        int group_idx = idx_in_warpgroup / GROUP_SIZE;      // 组编号 0..15
        // gIndices 指向本序列的 topk 索引数组（每个元素是被选中的 token 编号）
        int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q;   // [topk]

        // 本线程在 smem 里负责的 K/V 起始地址（每个线程管 8 个 bf16 = 16 字节）
        bf16* my_sKV_base = &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles<1>{})(group_idx, idx_in_group*8));
        // 本线程在 global memory 里负责的 K/V 起始地址
        // 注意 stride_kv_s_kv 是相邻 token 的字节距离（可能是负数，表示逆序存储）
        bf16* my_gKV_base = params.kv + idx_in_group*8;

        // ====== 步骤 2：定义加载 token 索引的 lambda ======
        // 把 topk 索引数组里的 token 编号读出来，预先乘上 stride（后面用偏移就快了）
        // 同时检查每个 token 是否有效（>=0 且 < 序列长度；如果支持变长 topk_length 还要 < topk_length）
        int64_t token_indices[2][NUM_ROWS_PER_GROUP];   // [buf_idx][local_row]：两个缓冲区的 token 偏移
        bool is_token_valid[2][NUM_ROWS_PER_GROUP];
        auto load_token_indices = [&](int block_idx) {
            CUTE_UNROLL
            for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                    int offs = (block_idx+buf_idx)*B_TOPK + local_row*NUM_GROUPS + group_idx;
                    int t = __ldg(gIndices + offs);    // __ldg 走只读 cache，比普通 load 快
                    // 预乘 stride，后面 copy_tiles 直接加偏移即可（避免在 hot loop 里乘）
                    token_indices[buf_idx][local_row] = t*(int64_t)params.stride_kv_s_kv;   // We mult it with params.stride_kv_s_kv here since it's faster
                    bool is_cur_token_valid = t >= 0 && t < params.s_kv;
                    if constexpr (HAVE_TOPK_LENGTH) {
                        is_cur_token_valid &= offs < topk_length;
                    }
                    is_token_valid[buf_idx][local_row] = is_cur_token_valid;
                }
            }
        };

        // ====== 步骤 3：定义 copy_tiles lambda（实际的 cp.async 搬运） ======
        // cache_policy = evict_last：提示 L2 cache "这些 K/V 数据马上还要用，尽量保留"
        int64_t cache_policy = createpolicy_evict_last();
        auto copy_tiles = [&](int block_idx, int buf_idx, int tile_start, int tile_end) {
            // Copy some K/V tiles from global memory to shared memory
            // A tile has a shape of 64 (B_TOPK) x 64
            // `buf_idx` is the index of the shared memory buffer, 0 or 1
            // `tile_idx` is the index of the tile to load, from 0 to D_K/64-1 = 8
            CUTE_UNROLL
            for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                int64_t token_index = token_indices[buf_idx][local_row];
                CUTE_UNROLL
                for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                    // cp.async 把 gmem 的一块拷到 smem，不阻塞线程
                    // 256B 对齐：每次拷 256 字节（32 个 bf16），效率最高
                    cp_async_cacheglobal_l2_prefetch_256B(
                        my_gKV_base + token_index + tile_idx*64,                          // 源：gmem 里的 K/V 数据
                        my_sKV_base + (buf_idx*B_TOPK*D_K + tile_idx*(B_TOPK*64) + local_row*NUM_GROUPS*64),  // 目的：smem 缓冲区
                        is_token_valid[buf_idx][local_row],                               // 如果 token 无效，跳过拷贝（smem 里是垃圾，后面 mask 掉）
                        cache_policy
                    );
                }
            }
        };

        // ====== 步骤 4：定义 commit_to_mbar lambda（搬运完成通知） ======
        // cp.async 是异步的——发指令后立即返回，数据没到 smem 也能继续执行。
        // 消费者怎么知道数据到了？通过 mbar（transaction barrier）：
        //   - 每次 cp.async 完成一定字节，硬件自动 arrive mbar 一次（带字节数）
        //   - 这里再 arrive 一次"补足"arrive count（cpasync_barrier_arrive_noinc 不带字节数）
        //   - 消费者 wait mbar 时，arrive count 到了就解锁
        auto commit_to_mbar = [&](transac_bar_t &bar) {
            cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
        };

        int cur_bar_wait_phase = 1;   // 注意初始值是 1（消费者是 0），双方在第一轮 wait 后会同步

        // ====== 步骤 5：主循环——双缓冲加载 K/V ======
        // 每轮加载 4 个 tile 块：V0L, V1R, V0R, V1L
        //   - L = 左半 D 维（前 4 个 64-d tile，0..3）
        //   - R = 右半 D 维（后 4-5 个 64-d tile，4..D_K/64-1）
        //   - V0/V1 = 两个 buf（双缓冲）
        // 这 4 个块覆盖了 2 个 topk 块 × 2 个 buf × 2 个 D 半边，正好对应消费者每轮的需求。
        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
            load_token_indices(block_idx);

            // V0L：buf 0 装第 block_idx 个 topk 块的左半 D
            // 先等消费者说"buf 0 空闲了"（上一轮用完），再装
            plan.bar_k0_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 0, 4);
            commit_to_mbar(plan.bar_k0_ready[0]);   // 告诉消费者"buf 0 左半就绪"

            // V1R：buf 1 装第 block_idx+1 个 topk 块的右半 D
            plan.bar_k1_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 4, D_K/64);
            commit_to_mbar(plan.bar_k1_ready[1]);

            // V0R：buf 0 装第 block_idx 个 topk 块的右半 D
            plan.bar_k0_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 4, D_K/64);
            commit_to_mbar(plan.bar_k0_ready[1]);

            // V1L：buf 1 装第 block_idx+1 个 topk 块的左半 D
            plan.bar_k1_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 0, 4);
            commit_to_mbar(plan.bar_k1_ready[0]);

            // ----- 写 is_kv_valid 掩码数组（给消费者 mask 用） -----
            // Valid mask
            // NOTE: V1R's finish implies maskings of the last round have finished
            // （上一轮的 mask 已经被消费者用完，可以安全覆盖）
            if (idx_in_group == 0) {   // 每组只让 1 个线程写，避免冲突
                CUTE_UNROLL
                for (int buf_idx = 0; buf_idx < 2; ++buf_idx)
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
                        plan.is_kv_valid[buf_idx][local_row*NUM_GROUPS+group_idx] = is_token_valid[buf_idx][local_row];
                plan.bar_is_kv_valid_ready.arrive();   // 告诉消费者"mask 准备好了"
            }

            cur_bar_wait_phase ^= 1;
        }
    }


#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}

// ============================================================================
// sparse_attn_fwd_kernel —— CUDA kernel 入口函数
// ============================================================================
// 【这个函数做什么】CUDA kernel 的入口。每个 thread block 启动一次，调用
// Kernel::devfunc 干活。
//
// 【关键属性解释】
//   - __global__：表示这是 device kernel，从 host 调用、在 GPU 上执行
//   - __launch_bounds__(NUM_THREADS, 1, 1)：
//       * NUM_THREADS=384：每 block 384 线程（3 个 WG × 128）
//       * 第一个 1：每 SM 最多驻留 1 个 block（因为寄存器/smem 占用大）
//       * 第二个 1：cluster size=1（不用 cluster，单 block 即可）
//     显式告诉编译器资源约束，让编译器更好地分配寄存器。
//   - __grid_constant__：const 引用参数，告诉编译器"这个参数在整个 grid 内不变，
//     所有线程看到的值一样"，编译器可以把它放在常量内存/寄存器里优化访问。
//
// 【模板参数】
//   - Kernel：具体 kernel 类（KernelTemplate<512/576, true/false>）
//   - TMAParams：TMA 描述符的类型（包含 Q 的 TMA 和 O 的 tensor map）
template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

// ============================================================================
// KernelTemplate::run —— host 端的"启动器"函数
// ============================================================================
// 【这个函数做什么】在 host (CPU) 上调用，负责：
//   1) 检查参数合法性
//   2) 创建 TMA 描述符（告诉 TMA 硬件 Q 和 O 张量的形状、布局）
//   3) 设置 shared memory 大小
//   4) 启动 kernel
//
// 【TMA 描述符是什么】TMA 硬件不会自己认 tensor——它需要 host 提前用
// cuTensorMapEncodeTiled 算一个 128 字节的"描述符"，告诉它：
//   - 数据起始地址、形状、stride
//   - 每次搬的"盒子"大小（box_size，比如 64×B_H×1）
//   - swizzle 模式（128B swizzle 是 smem 访问优化的内存布局变换）
//   - L2 promotion、OOB fill 等策略
// 描述符算好后，device 端发 TMA 指令时只传这个描述符 + 偏移，硬件自己解码。
template<int D_QK, bool HAVE_TOPK_LENGTH>
void KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(const SparseAttnFwdParams &params) {
    // ====== 步骤 1：参数合法性检查 ======
    KU_ASSERT(params.h_kv == 1);                         // 稀疏注意力只支持 MQA（h_kv=1）
    KU_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundry checkings
    KU_ASSERT(params.topk > 0);
    KU_ASSERT(params.h_q % B_H == 0);                    // Q 头数必须是 B_H 的整数倍

    // ====== 步骤 2：构造 Q 的 TMA 描述符 ======
    // shape_Q = (h_q, d_qk, s_q)：头数 × 维度 × 序列数
    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)   // stride：head 维、dim 维（连续，stride=1）、seq 维
            )
        ),
        SmemLayoutQ{}    // smem 里的布局，影响 TMA 怎么摆放数据
    );

    // ====== 步骤 3：构造 O 的 TMA 描述符（用 CUDA Driver API 直接调 cuTensorMapEncodeTiled） ======
    // O 是 3D 张量：[D_V, h_q, s_q]
    CUtensorMap tensor_map_O;
    {
        uint64_t size[3] = {D_V, (unsigned long)params.h_q, (unsigned long)params.s_q};
        uint64_t stride[2] = {D_V*sizeof(bf16), D_V*params.h_q*sizeof(bf16)};   // 维度 stride（字节）
        uint32_t box_size[3] = {64, B_H, 1};    // 一次 TMA store 的"盒子"：64×64×1 个 bf16
        uint32_t elem_stride[3] = {1, 1, 1};    // 元素间无 stride
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_O,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,                                  // 3 维张量
            params.out,                         // gmem 起始地址
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,   // 不交织
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,          // 128B swizzle（smem 访问优化）
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, // 不做 L2 promotion
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   // 越界不填充
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // ====== 步骤 4：打包 TMA 参数 ======
    TmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q,
        tensor_map_O
    };
    auto kernel = &sparse_attn_fwd_kernel<KernelTemplate<D_QK, HAVE_TOPK_LENGTH>, decltype(tma_params)>;

    // ====== 步骤 5：设置 shared memory 大小 ======
    // SharedMemoryPlan 是动态共享内存（extern __shared__），这里告诉运行时要多少字节。
    // H100 一个 block 最多 228KB smem，本 kernel 用得满满的。
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // ====== 步骤 6：构造 launch 参数并启动 kernel ======
    cutlass::ClusterLaunchParams launch_params = {
        // grid 维度：block 总数 = (h_q/B_H) * s_q（每个 block 处理一组 B_H 个头 × 一个序列）
        // 注意 s_q 放在第一维，因为 s_q 可能超过 65535（gridDim.y/z 的上限）
        dim3((params.h_q/B_H)*params.s_q, 1, 1),    // NOTE: We put s_q on the first dim since it can be larger than 65536 (the maximum size of griddim.y and griddim.z)
        dim3(NUM_THREADS, 1, 1),                     // block 维度：384 线程
        dim3(1, 1, 1),                               // cluster 维度：1（不用 cluster）
        smem_size,                                   // 动态 smem 大小
        params.stream                                // CUDA stream
    };
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    );
    KU_CHECK_KERNEL_LAUNCH();
}

// ============================================================================
// run_fwd_phase1_kernel —— 外部调用入口（被 fwd.cu 调用）
// ============================================================================
// 这是个薄封装，让 fwd.cu 不用知道模板参数 D_QK/HAVE_TOPK_LENGTH 的细节。
// fwd.cu 根据 params.d_qk 和 params.topk_length 是否非空，分派到对应的模板实例。
template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params) {
    KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(params);
}

}
