// ============================================================================
// helpers.h —— SM90 (H100) kernel 通用工具函数库
// ============================================================================
//
// 【这个文件解决什么实际问题？】
//
// 写 CUDA kernel 时，有一堆"小事"反复出现：
//   - 异步搬数据（cp.async）：从 global memory 拷一块到 shared memory，不阻塞线程
//   - 控制 L2 cache 策略：有些数据想留在 cache 里反复用，有些用完就丢
//   - 算 WGMMA 矩阵乘法：H100 的 warpgroup-level GEMM 指令很强大但难用，要包一层
//   - 查"我现在在哪个 SM"：分布式调度时需要
//   - 访问"邻居 SM"的共享内存（cluster 模式下）：跨 SM 通信用
//   - 用 TMA 拷贝数据：H100 新引入的硬件搬运单元
//
// 这些工具函数本身不实现业务逻辑，但它们把底层 PTX 内联汇编、CuTe 模板魔法
// 封装成一行就能调的 C++ 函数，让 kernel 主体（phase1.cuh 等）保持简洁。
//
// 【为什么大部分用 PTX 内联汇编】
//
// PTX (Parallel Thread Execution) 是 NVIDIA GPU 的"汇编语言"。C++ 编译器
// 不一定知道最新的 H100 指令（比如 cp.async 的 L2 预取 hint），所以要用
// `asm volatile("...")` 直接写 PTX。每条 PTX 指令对应一条硬件指令，最直接。
//
// 【关键概念第一次出现的解释】
//
// * cp.async (copy asynchronous)：Ampere (SM80) 引入的异步拷贝指令。线程发
//   一条指令后立刻往下执行，硬件在后台把数据从 global memory 搬到 shared
//   memory，搬完会通过 barrier 通知线程。可以类比成"快递下单后就干别的去，
//   到货了再回来取"。
//
// * L2 cache policy：L2 cache 是 SM 共享的快速缓存（比 global memory 快几十倍）。
//   默认情况下硬件用启发式决定哪些数据留在 cache。我们可以用 createpolicy
//   指令主动控制：
//     - evict_last：尽量保留（"用完最后再踢"，给热点数据用）
//     - evict_first：尽快踢出（"用完就丢"，给只读一次的数据用，避免污染 cache）
//
// * WGMMA (Warpgroup Matrix Multiply-Accumulate)：H100 的张量核心指令，
//   一个 warpgroup (128 线程) 一条指令可以算 64xN 的大矩阵乘法。C (累加器)
//   分布在 128 个线程的寄存器里，每个线程持有不连续的"片段"——拿线程号和
//   本地索引算全局行列号要用 get_AorC_row_idx / get_AorC_col_idx。
//
// * TMA (Tensor Memory Accelerator)：H100 新增的硬件单元，专门搬 tensor。
//   线程发一条指令告诉 TMA "按这个描述符搬这块 tensor"，硬件自己完成搬运，
//   不占用计算资源。比 cp.async 更高效，但需要提前算 tensor map 描述符。
//
// * Cluster：H100 引入的"SM 组"概念，多个 SM 组成一个 cluster，可以共享
//   共享内存、同步。`get_peer_addr` 用来访问同 cluster 内其他 SM 的共享内存。
//
// ============================================================================

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace sm90 {

// ----------------------------------------------------------------------------
// cp.async 拷贝（无 cache hint 版本）
// ----------------------------------------------------------------------------
// 【做什么】从 global memory 拷 16 字节到 shared memory，异步、不阻塞线程。
// 同时让 L2 cache 顺手预取 256 字节（L2::256B hint），让后续相邻地址的访问更快。
//
// 【参数】src = 源地址（global memory）; dst = 目的地址（shared memory）
//
// 【PTX 解释】
//   cp.async = 异步拷贝
//   cg       = cache-global（走 L2 cache，比 ca 模式省带宽）
//   shared.global = 从 gmem 拷到 smem
//   L2::256B = 顺手在 L2 里预取 256 字节的 cache line
//   %2 = 字节数，这里是编译期常量 16（"n" 约束）
//
// 【为什么没有 arrive barrier】这个函数只发指令，不通知"搬完了"——
// 通知由调用方做（cpasync_barrier_arrive_noinc）。这样多条 cp.async 可以批量
// 发出去，最后统一通知一次，减少同步开销。
__forceinline__ __device__ void cp_async_cacheglobal_l2_prefetch_256B(const void* src, void* dst) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n"
        :: "r"(dst_addr),
           "l"(src),
           "n"(16));
}

// ----------------------------------------------------------------------------
// cp.async 拷贝（带谓词 + cache hint 版本）
// ----------------------------------------------------------------------------
// 【和上面那个的差别】多了两个参数：
//   - pred：谓词，false 时拷 0 字节（相当于跳过这次拷贝）。用于处理边界情况，
//           比如 token 无效时不想让 cp.async 拷垃圾数据，传 pred=false 就行。
//   - cache_policy：来自 createpolicy_evict_last/first 的返回值，告诉 L2 cache
//                   "这块数据用完尽量保留 / 用完尽快踢出"。
//
// 【PTX 解释】L2::cache_hint = 带 cache 策略的版本，多一个 64-bit 策略寄存器。
__forceinline__ __device__ void cp_async_cacheglobal_l2_prefetch_256B(const void* src, void* dst, bool pred, int64_t cache_policy) {
    uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
    asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2, %3;\n"
        :: "r"(dst_addr),
           "l"(src),
           "r"(pred?16:0),       // 谓词转成字节数：true 拷 16 字节，false 拷 0 字节
           "l"(cache_policy));
}

// ----------------------------------------------------------------------------
// createpolicy_evict_last —— "用完最后再踢"cache 策略
// ----------------------------------------------------------------------------
// 【做什么】生成一个 64-bit 的 cache 策略句柄，传给 cp.async 的 cache_hint 参数，
// 让那块数据在 L2 cache 里尽量保留（"evict last" = 最后再淘汰）。
//
// 【什么时候用】K/V 数据会被反复读多次（消费者 WG 每轮都用），希望留在 cache 里。
//
// 【PTX 解释】createpolicy.fractional = 创建一个 fractional（这里 1.0 = 全量）
// 策略; L2::evict_last = 策略类型; b64 = 64-bit 句柄。
__forceinline__ __device__ int64_t createpolicy_evict_last() {
    int64_t res;
    asm volatile(
        "createpolicy.fractional.L2::evict_last.b64 %0, 1.0; \n\t"
        : "=l"(res)
        :
    );
    return res;
}

// ----------------------------------------------------------------------------
// createpolicy_evict_first —— "用完就丢"cache 策略
// ----------------------------------------------------------------------------
// 【做什么】和上面相反：告诉 L2 cache "这块数据只用一次，用完尽快踢出"。
//
// 【什么时候用】Q 数据每轮只读一次，不希望它占着 cache 把 K/V 挤出去。
__forceinline__ __device__ int64_t createpolicy_evict_first() {
    int64_t res;
    asm volatile(
        "createpolicy.fractional.L2::evict_first.b64 %0, 1.0; \n\t"
        : "=l"(res)
        :
    );
    return res;
}


// ----------------------------------------------------------------------------
// get_AorC_row_idx —— 把"本地行号"转成"全局行号"
// ----------------------------------------------------------------------------
// 【背景】WGMMA 算 64xN 矩阵乘时，64 个线程一组共同持有 C 矩阵（也叫 fragment C）。
// 每个线程拿到的不是连续的行，而是按特定规则分布的"片段"。
// 比如线程 0 持有第 0,8 行，线程 1 持有第 1,9 行，等等（具体见 PTX 文档）。
//
// 【做什么】我们用 local_row_idx (0/1) 表示"本线程持有的第几行"，这个函数算出
// 它对应矩阵 C 的全局第几行。在 phase1.cuh 的 reduce_L / store_O 里要用，
// 比如读 attn_sink[q_h_idx*B_H + get_AorC_row_idx(row, idx_in_warpgroup)]。
//
// 【参数】
//   local_row_idx：0 或 1（本线程持有的两行中的第几行）
//   idx_in_warpgroup：本线程在 warpgroup 内的编号 (0..127)
//
// 【公式拆解】（参见 https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a）
//   (idx_in_warpgroup/32)*16   ：前 32 线程在 0-15 行，后 32 线程在 16-31 行 ...
//   + local_row_idx*8          ：本线程持有的两行相隔 8
//   + (idx_in_warpgroup%32/4)  ：warp 内每 4 线程一组，组间行号差 1
//
// In the layout of fragment A and fragment C during WGMMA, the data each thread holds resides in two particular rows. This function converts the local_row_idx (0~2) to the actual row_idx
// You may refer to this link for the detailed layout: https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    int row_idx = (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
    return row_idx;
}

// ----------------------------------------------------------------------------
// get_AorC_col_idx —— 把"本地元素编号"转成"全局列号"
// ----------------------------------------------------------------------------
// 【做什么】和上面类似，但转的是列号。本线程持有的 fragment A/C 元素 local_elem_idx
// 对应矩阵的全局第几列。
//
// 【公式拆解】
//   8*(local_elem_idx/4)    ：每 4 个元素为一组，组间列号差 8
//   + (idx_in_warpgroup%4)*2：warp 内 4 线程一组，组内线程列号差 2
//   + (local_elem_idx&1)    ：组内两个元素列号差 1
__forceinline__ __device__ int get_AorC_col_idx(int local_elem_idx, int idx_in_warpgroup) {
    int col_idx = 8*(local_elem_idx/4) + (idx_in_warpgroup%4)*2 + (local_elem_idx&1);
    return col_idx;
}

// ----------------------------------------------------------------------------
// gemm —— 通用 WGMMA 封装（最灵活的版本）
// ----------------------------------------------------------------------------
// 【做什么】封装 H100 的 WGMMA 指令，算 C += A @ B（或 C = A @ B）。
// A、B 可以在寄存器或 shared memory，由 TiledMma 模板参数决定。
//
// 【为什么需要封装】直接调 cute::gemm 不够——WGMMA 是异步的，前后要配
// warpgroup_arrive / warpgroup_commit_batch / warpgroup_wait 才正确，
// 还要 warpgroup_fence_operand 防止编译器把寄存器重排破坏 WGMMA 假设的数据布局。
// 这些细节每次手写容易出错，封装起来更安全。
//
// 【模板参数】
//   - zero_init：true = 清空 C 再算（C = A @ B）; false = 累加（C += A @ B）
//   - wg_wait：>=0 表示立即等 GEMM 完成（同步）; <0 表示不等（异步，后面自己 wait）
//   - arrive：是否发 warpgroup_arrive（开始一批 WGMMA 的信号）
//   - commit：是否发 warpgroup_commit_batch（结束一批 WGMMA 的提交信号）
//
// 【Is_RS 判断】WGMMA 有两种形式：
//   - SS：A 和 B 都在 shared memory（用描述符访问）
//   - RS：A 在寄存器（register），B 在 shared memory
//   通过看 FrgTypeA 是不是 DescriptorIterator 来区分。RS 模式下 A 在寄存器，
//   要额外的 fence 保护。
//
// Adapted from https://github.com/Dao-AILab/flash-attention/blob/cdaf2de6e95cb05400959b5ab984f66e4c7df317/hopper/utils.h
// * Copyright (c) 2024, Tri Dao.
template <bool zero_init=false, int wg_wait=0, bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA, Tensor1 const &tCrB, Tensor2 &tCrC) {
    using namespace cute;
    constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;       // 第一块：清空 C（D=0）
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;    // 后续块：累加（D=1）
        }
    } else {
        // cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        // Unroll the K mode manually to set scale D to 1
        CUTLASS_PRAGMA_UNROLL
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        }
    }
    if constexpr (commit) {
        warpgroup_commit_batch();
    }
    if constexpr (wg_wait >= 0) { warpgroup_wait<wg_wait>(); }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) { cute::warpgroup_fence_operand(const_cast<Tensor0 &>(tCrA)); }
}

// ----------------------------------------------------------------------------
// gemm_ss —— SS 模式 GEMM 的简化封装（A、B 都在 shared memory）
// ----------------------------------------------------------------------------
// 【和上面 gemm 的差别】专门给 SS 模式用，少了 arrive/commit/wait 的开关，
// 一律 arrive + 不显式 wait（异步，调用者自己 wait）。代码更短，调用更简单。
//
// 【参数】clear_accum = true 时清空 C 再算，否则累加
// 【流程】partition_fragment_A/B 切出每个线程负责的片段 → 循环 K 维做 WGMMA
// A simpler version of gemm
template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_ss(bool clear_accum, TiledMma tiled_mma, Tensor0 const &sA, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sA_frag = thr_mma.partition_fragment_A(sA);    // 切出本线程的 A 片段
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);    // 切出本线程的 B 片段
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));  // K 维大小必须匹配

    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;     // 后续 K 块累加
    }
    warpgroup_fence_operand(rC_frag);
}

// ----------------------------------------------------------------------------
// gemm_rs —— RS 模式 GEMM 的简化封装（A 在寄存器，B 在 shared memory）
// ----------------------------------------------------------------------------
// 【和 gemm_ss 的差别】A 在寄存器（rA_frag），所以要给 A 也加 fence 保护。
// 这个模式用于"A 已经在寄存器里准备好了"的场景，省去从 smem 加载 A 的步骤。
// 比如 phase1.cuh 里用本 WG 算出的 rS（softmax 后的 P）直接当 A 算 PV。
template <typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_rs(bool clear_accum, TiledMma tiled_mma, Tensor0 rA_frag, Tensor1 const &sB, Tensor2 &rC_frag, int idx_in_warpgroup) {
    using namespace cute;
    ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
    Tensor sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(rA_frag) == size<2>(sB_frag));

    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
    warpgroup_fence_operand(rC_frag);
    warpgroup_arrive();
    tiled_mma.accumulate_ = clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(rA_frag); ++k) {
        cute::gemm(tiled_mma, rA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    warpgroup_fence_operand(rC_frag);
    warpgroup_fence_operand(const_cast<Tensor0 &>(rA_frag));
}


// ----------------------------------------------------------------------------
// get_sm_id —— 查"我现在跑在哪个 SM 上"
// ----------------------------------------------------------------------------
// 【做什么】返回当前线程所在 SM 的编号（0..num_sms-1）。
//
// 【什么时候用】某些调度逻辑需要按 SM 分配任务（比如不同 SM 处理不同 batch），
// 这时要在 kernel 里查 SM id。普通 CUDA 编程里 blockIdx 已经够用，但跨 cluster
// 协作或细粒度调度时需要直接拿 SM id。
//
// 【PTX 解释】%smid 是个特殊寄存器，存当前 SM 编号。mov.u32 把它读到普通寄存器。
__forceinline__ __device__ uint32_t get_sm_id() {
    uint32_t ret;
    asm("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

// ----------------------------------------------------------------------------
// get_peer_addr —— 算"邻居 SM"共享内存的对应地址
// ----------------------------------------------------------------------------
// 【做什么】给定本 SM 的 shared memory 地址 p，返回"对偶 SM"上对应位置的地址。
// 用于 cluster 模式下跨 SM 共享 smem。
//
// 【PEER_ADDR_MASK 的含义】H100 cluster 模式下，同一 cluster 内两个 SM 的
// smem 地址通过 XOR 这个掩码互相对应。把本地地址 XOR 掩码就得到对端地址。
// 16777216 = 2^24 = 16MB，这是 H100 一个 SM 的 smem 大小（实际可能因 GPU 型号
// 不同而异，所以注释说 "Not sure if this number is the same on all GPUs"）。
//
// 【什么时候用】cluster 模式下，两个 SM 协作算同一个任务，需要互相读对方的 smem。
// 用这个函数算对端地址，然后直接 load/store 就行（硬件会路由到对端 SM）。
static constexpr int PEER_ADDR_MASK = 16777216; // peer_addr = my_addr ^ PEER_ADDR_MASK. Not sure if this number is the same on all GPUs.
template<typename T>
CUTE_DEVICE
T* get_peer_addr(const T* p) {
    return (T*)((int64_t)(p) ^ PEER_ADDR_MASK);
}

// ----------------------------------------------------------------------------
// launch_tma_copy —— 用 TMA 拷贝一个 tensor
// ----------------------------------------------------------------------------
// 【做什么】用 H100 的 TMA 硬件把 src 拷到 dst，附带 barrier 通知。
// 调用方传一个 ClusterTransactionBarrier，TMA 拷完会自动 arrive 它，调用方
// 后面 wait 这个 barrier 就知道数据到了。
//
// 【参数】
//   - tma_copy：TMA 拷贝对象（用 cute::make_tma_copy 创建，里面装了 tensor map）
//   - src / dst：源和目的 tensor（CuTe tensor 视图）
//   - bar：同步用的 barrier，TMA 完成时自动 arrive
//   - cache_hint：cache 策略（EVICT_NORMAL / EVICT_FIRST / EVICT_LAST）
//
// 【流程】get_slice(0) 拿到线程 0 的 TMA 视图 → partition_S/D 切出源/目的的
// 子块 → cute::copy 发 TMA 指令（with(bar) 让拷贝完成时 arrive barrier）。
//
// 【为什么只 get_slice(0)】TMA 是单线程发起的（一个代表线程发指令，硬件完成搬运），
// 不像 cp.async 可以多线程并行发。所以这里固定用线程 0 的视角。
template<
    typename TMA,
    typename Tensor0,
    typename Tensor1
>
CUTE_DEVICE
void launch_tma_copy(
    const TMA &tma_copy,
    Tensor0 src,
    Tensor1 dst,
    cutlass::arch::ClusterTransactionBarrier &bar,
    const cute::TMA::CacheHintSm90 &cache_hint = cute::TMA::CacheHintSm90::EVICT_NORMAL
) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(bar), 0, cache_hint),
        thr_tma.partition_S(src),
        thr_tma.partition_D(dst)
    );
}

}
