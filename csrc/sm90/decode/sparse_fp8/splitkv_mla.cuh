// ============================================================================
// splitkv_mla.cuh —— SM90 (H100) FP8 稀疏解码 MLA kernel（带 Split-KV）
// ============================================================================
//
// 【这段代码解决什么实际问题？】
//
// 大模型"解码"阶段：每次生成一个字，要用当前 token 的 Q 去查所有历史 K/V，
// 算 attention 得到下一个 token 的 logits。问题是历史 K/V 可能很长（几万 token），
// 一次性算完很慢。
//
// FlashMLA 的优化思路：
//   1. 稀疏（Sparse）：不让所有历史 token 都参与，只挑 topk 个最相关的。
//   2. FP8 量化：K/V 用 8-bit 浮点存（fp8），节省一半显存和带宽。
//   3. Split-KV：把 topk 个 token 切成多段，让多个 SM 并行算不同段，最后合并。
//   4. MLA (Multi-head Latent Attention)：DeepSeek 模型的注意力机制，K/V 维度
//      特殊（512 维 nope + 64 维 rope），head 数 64 或 128。
//
// 这个 kernel 就是上面 4 个优化的集大成者。
//
// 【3 个 warpgroup 分工】（和 phase1.cuh 类似但不同）
//
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │  WG0 (消费者)：算 QK^T → online softmax → 算左半 PV → 写回 O         │
//   │  WG1 (消费者)：算右半 PV（用 WG0 的 rS 通过 sS 共享）→ 写回 O 右半    │
//   │  Producer WG (生产者)：从 gmem 读 FP8 K/V → 反量化成 bf16 → 写 smem   │
//   │                        （如果 CLUSTER_SIZE=2，还要异步写到对端 SM）   │
//   └──────────────────────────────────────────────────────────────────────┘
//
// 【V3.2 vs MODEL1 两种模型架构】
//
//   - V3.2 (DeepSeek V3.2)：HEAD_DIM_K=576（512 nope + 64 rope），
//     每 128 维 4 个 fp32 scale，KV cache 行大小 656 字节。
//   - MODEL1：HEAD_DIM_K=512（448 nope + 64 rope），每 64 维 8 个 fp8_e8m0 scale，
//     支持 extra KV cache（额外的索引池），行大小 = 448 + 2*64 + 8 = 584 字节。
//
// 【Split-KV 是什么】
//
// 一个序列的 topk 可能很大（比如 1024），一个 SM 算太慢。Split-KV 把 topk 切成
// 几段，每段由一个 partition (blockIdx.z) 算。每个 partition 算出"部分 O"和
// "部分 lse"，后面由 combine kernel 合并。这样多个 SM 并行处理同一序列，加速
// 长序列解码。代价是要多一次合并开销，所以只在 topk 大时才划算。
//
// 【关键概念第一次出现时的解释】
//
// * FP8 (e4m3 / e8m0)：8-bit 浮点数。e4m3 = 4 位指数 + 3 位尾数（精度高范围小），
//   e8m0 = 8 位指数无尾数（只能表示 2 的幂，专门用作 scale）。MODEL1 用 e8m0
//   作 scale，比 fp32 scale 省空间。
//
// * 反量化（Dequantize）：FP8 数据不能直接参与 GEMM（WGMMA 不支持 fp8 输入到
//   bf16 累加器），要先转成 bf16。生产者 WG 干的就是这事——读 fp8，乘 scale，
//   写 bf16 到 smem。
//
// * MLA (Multi-head Latent Attention)：DeepSeek 的注意力变体。K/V 不是直接存的，
//   而是存一个低维"潜变量"，用时再投影回高维。但本 kernel 直接收投影后的 K/V，
//   所以代码里看不到投影逻辑，只关心 K/V 的特殊维度布局。
//
// * RoPE (Rotary Position Embedding)：旋转位置编码，作用在 K 和 Q 的 rope 部分。
//   nope 部分不旋转。本 kernel 把 K 拆成 nope (512/448 维) + rope (64 维) 分别处理。
//
// * Cluster (H100 新概念)：多个 SM 组成一个 cluster，可以共享 smem、同步。
//   CLUSTER_SIZE=2 时两个 SM 协作，一个 SM 反量化完通过 st_async 写到对端 SM 的
//   smem，对端不用再算一遍——共享劳动成果。
//
// * Split-KV 的 lse 合并：每个 partition 算出局部 O 和 lse，combine kernel 用
//   lse 做加权平均得到最终 O。所以本 kernel 不直接输出最终 O（split 情况下），
//   而是输出 "O × sum(exp)" 和 "lse = log(sum(exp)) + max"，留给 combine 合并。
//
// * PDL (Programmatic Dependent Launch)：H100 新特性，前一个 kernel 快结束时
//   就开始下一个 kernel，省等待时间。本文件注释里说有编译器 bug，禁用了。
//
// ============================================================================

#pragma once

#include "splitkv_mla.h"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "utils.h"
#include "components/dequant.h"
#include "components/helpers.h"
#include "config.h"
using namespace cute;

namespace sm90::decode::sparse_fp8 {

// 初始值用 -1e30 而不是 -INFINITY，防止 (-inf) - (-inf) = NaN 的数值问题
// 【为什么不用 -INFINITY】如果用 -inf，后面做 exp2f(old_max - new_max) 时
// 会出现 (-inf) - (-inf) = NaN，整个 softmax 就废了。用 -1e30 既能表示
// "极小"，又不会触发 NaN。
static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using fp8_e8m0 = __nv_fp8_e8m0;   // 8 位浮点，只有指数位无尾数（专门用作 scale）

// ----------------------------------------------------------------------------
// scale_softmax —— 在线 softmax + mask + 重缩放 O
// ----------------------------------------------------------------------------
// 【这段代码解决什么实际问题？】
//
// 在 attention 里，QK^T 算出来后要做 softmax 才能乘 V。但是：
//   1. K/V 是流式分块读进来的（不能一次性全装下），所以 softmax 也要"流式"做——
//      每来一块新 K/V，就要把当前累积的 max/sum 更新一次，并相应调整之前算出的 O。
//   2. topk 里可能有 padding（topk_length 不满 64 个有效 token），这些位置不能
//      参与 softmax，否则会把概率"漏"到无效位置上。
//   3. WG0 和 WG1 共享同一组 max，所以 rescale 因子要让 WG1 也知道。
//
// 【"在线 softmax" 是什么？】（详见 phase1.cuh 文件头注释）
//
// 普通 softmax：一次性看到所有数据，max → exp → sum → div。
// 在线 softmax：流式处理，每来一块新数据：
//   1. 求本块行 max → cur_max
//   2. new_max = max(old_max, cur_max)
//   3. 旧的 O 和 L 都要乘 exp(old_max - new_max)，缩小到新 max 下
//   4. 用新数据算 exp(P - new_max)，累加到 L，乘 V 累加到 O
// 这样不用等所有数据到齐才能开始算，可以边读边算。
//
// 【为什么要把 scale 因子写到 sScale？】
//
//   WG0 算的是 O 的"左半"（前 256 维），WG1 算"右半"（后 256 维）。
//   但两个 WG 共享同一组 max（因为是同一行 Q 对同一批 K 的 attention）。
//   所以 WG0 更新 max 后，WG1 也要按同样的因子 rescale 自己累加的 rO。
//   WG0 把 scale_for_olds 写到 smem 的 sScale 数组，WG1 读出来用。
//
// 【参数含义】
//   - rP：QK^T 结果（寄存器，输入，会被修改成 softmax 后的值）
//   - rS：softmax 后的 P 转 bf16（输出，给后面的 PV GEMM 当 A 矩阵用）
//   - rO：累加器（输入输出，会 rescale；存的是 O = sum(P·V) 的部分和）
//   - scale_softmax_log2：1/sqrt(d_k) 的 log2 域值（QK^T 通常要除 sqrt(d_k)）
//   - sScale：smem 数组，存 rescale 因子给 WG1
//   - rM/rL：每行一个 max / sum（输入输出，会被更新）
//   - is_kv_valid[]：本 block 内每个 token 是否有效（padding 或越界的置 false）
//   - block_idx：当前块编号（这里没用到，保留参数为了将来扩展）
//   - idx_in_warpgroup：本线程在 WG 内编号（用于 shfl reduce 和写 sScale）
template<
    typename Tensor0,
    typename Tensor1,
    typename Tensor2
>
__forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup
) {
    // 每个线程负责 2 行（BLOCK_M=64 行被 32 个线程分，每线程 2 行）
    float scale_for_olds[2];   // 两行的 rescale 因子，最后写到 sScale 给 WG1
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        // 取出本线程负责的"第 local_row_idx 行"片段
        // rP/rS/rO 是三维的 (row, col, ...)，这里固定 row，展开后变成一维好遍历
        Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

        // ----- 第 1 步：mask 无效 token（置 -INF），求本线程部分行 max -----
        // 【为什么 mask】无效 token 不能参与 softmax，否则它的"得分"会被当成
        // 一个真实的概率值，导致最终概率分布不对。
        float cur_max = -INFINITY;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            // 复杂的索引公式：把"片段内编号 i"映射回"原矩阵列号"，查 is_kv_valid
            // 这个索引和 WGMMA 的 fragment layout 有关，看起来奇怪但是对的
            if (!is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2])
                cur_rP(i) = -INFINITY;   // 无效位置分数设为 -INF，softmax 后变 0
            cur_max = max(cur_max, cur_rP(i));
        }
        // warp 内 4 线程 butterfly reduce（求 warp 内 4 线程的 max）
        // 【为什么是 butterfly】每步 xor 一个不同的位，1→2→4→8 线程配对，
        // log2(4)=2 步就能 reduce 完，比串行 reduce 快得多
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));  // 配对相邻线程
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));  // 配对隔一个线程

        // ----- 第 2 步：算 new_max = max(old_max, cur_max)，求 rescale 因子 -----
        // 【为什么乘 scale_softmax_log2】QK^T 的结果要除 sqrt(d_k) 防止数值过大，
        // 在 log2 域做就是乘 log2(1/sqrt(d_k))，等价但更快（exp2 比 exp 快）
        cur_max *= scale_softmax_log2;
        float old_max = rM[local_row_idx];
        rM[local_row_idx] = max(cur_max, old_max);   // 更新到新的 max
        // 【关键公式】rescale 因子 = exp2(old_max - new_max)
        // 如果 new_max > old_max，因子 < 1，旧 O 要缩小
        // 如果 new_max == old_max，因子 = 1，旧 O 不变
        float scale_for_old = exp2f(old_max - rM[local_row_idx]);
        scale_for_olds[local_row_idx] = scale_for_old;

        // ----- 第 3 步：rO *= scale_for_old（旧 O 缩到新 max 下）-----
        // 【为什么】之前用 old_max 算的 O = sum(exp(P - old_max) · V)
        // 现在要用 new_max，所以 O 要乘 exp(old_max - new_max) 调整
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rO); ++i) {
            cur_rO(i) *= scale_for_old;
        }

        // ----- 第 4 步：rS = softmax(P) = exp(P - new_max)，累加 sum 到 rL -----
        float cur_sum = 0;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            // exp2(P*log2(scale) - new_max) = exp(P*scale - new_max) 在数学上等价
            // 但用 log2 域计算更快（GPU 的 exp2 指令比 exp 快）
            rP(i) = exp2f(cur_rP(i)*scale_softmax_log2 - rM[local_row_idx]);
            cur_rS(i) = (bf16)cur_rP(i);    // 转 bf16 给后面的 PV GEMM 用（WGMMA 要 bf16 输入）
            cur_sum += cur_rP(i);
        }

        // rL 也要 rescale（旧 l 缩到新 max 下）再加新 sum
        // 【为什么】L = sum(exp(P - max))，max 变了 L 也要相应调整
        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
    // 每 4 个线程的第 0 个把 scale_for_olds 写到 sScale，给 WG1 读
    // 【为什么 idx_in_warpgroup%4 == 0】因为 4 线程共享一组 max/scale，
    // 只需要 1 个线程写就够，避免重复写
    if (idx_in_warpgroup%4 == 0)
        *(float2*)(sScale + 2*(idx_in_warpgroup/4)) = *(float2*)(scale_for_olds);
}

// ============================================================================
// devfunc —— kernel 的设备端入口
// ============================================================================
//
// 【这个函数做什么】每个 thread block 调用一次。blockIdx 编码了三层信息：
//   - blockIdx.x = head_block_idx：第几组 64 个 Q 头（NUM_HEADS=64 时只有 0，
//                  NUM_HEADS=128 时有 0/1，由两个 cluster 成员分别处理）
//   - blockIdx.y = s_q_idx：第几个 query token（解码时通常是 1）
//   - blockIdx.z = partition_idx：Split-KV 的第几个分片
//
// 一个 block 内 384 线程分 3 个 WG：
//   - WG0：算 QK^T + 左半 PV
//   - WG1：算右半 PV
//   - Producer：FP8 反量化 + 加载 K/V 到 smem
//
// 【模板参数】
//   - MODEL_TYPE：V32 (DeepSeek V3.2) 或 MODEL1（另一种模型架构）
//   - NUM_HEADS：64 或 128（MLA 头数）
//   - TMAParams：TMA 描述符类型
template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    // ====== 步骤 1：解析 blockIdx 和 threadIdx，确定本线程的身份 ======
    // 【为什么要先确定身份】GPU 上成千上万个线程同时跑同一段代码（SIMT 模型），
    // 每个线程必须根据自己的 blockIdx/threadIdx 知道"我是谁、要处理哪块数据"，
    // 这样才能正确地分工——否则所有线程做一样的事，就出错了。
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;  // 哪一组 64 个头
    const int s_q_idx = blockIdx.y;                                 // 第几个 query token（解码时通常 0）
    const int partition_idx = blockIdx.z;                           // Split-KV 第几片（blockIdx.z）
    // cluster 内 CTA 编号（CLUSTER_SIZE=2 时 0/1，对应两个 SM 协作）
    // 【为什么用 head_block_idx % 2】NUM_HEADS=128 时有两个 head_block，正好
    // 用它来区分 cluster 内的两个 CTA；NUM_HEADS=64 时 CLUSTER_SIZE=1，直接是 0
    const int idx_in_cluster = CLUSTER_SIZE == 1 ? 0 : head_block_idx % 2;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();  // 0/1/2：本线程属哪个 WG
    // 【三个 WG 分工】WG0=算 QK^T + 左半 PV，WG1=算右半 PV，WG2(else)=生产者读 K/V
    const int idx_in_warpgroup = threadIdx.x % 128;                 // WG 内编号 0..127
    const int warp_idx = cutlass::canonical_warp_idx_sync();        // 0..11：warp 编号

    // ====== 步骤 2：定义 shared memory tensors ======
    // 【什么是 shared memory】GPU 上的高速缓存，比全局显存快 10-100 倍。
    // 同一个 block 内的线程可以共享。这里把 smem 划分成多个区域：
    //   sQ：存 Q 矩阵（head_block_idx 这一组 64 个头的 Q）
    //   sK：存 K 矩阵（topk 选出的 64 个 token 的 K，双缓冲 NUM_K_BUFS=2）
    //   sS：存 softmax 后的 P 矩阵（WG0 算完给 WG1 用）
    //   sOBuf/sOAccumBuf：存输出 O 的临时缓冲（epilogue 用）
    //   sM/sL/sScale/sOScale：存 softmax 的 max/sum/scale 等中间值
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;
    
    // ====== 步骤 3：预取 TMA 描述符 ======
    // 【什么是 TMA】Tensor Memory Accelerator，H100 新硬件，专门做"成块数据搬运"，
    // 比手动用 cp.async 快得多。这里先预取 TMA 描述符（描述张量的形状/stride），
    // 后面真正搬数据时直接用描述符，不用再查表。
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
    }
    
    // ====== 步骤 4：初始化 TMA barriers ======
    // 【什么是 barrier】同步原语。这里的 barrier 是"transactional barrier"——
    // 不光能同步，还能计数（有多少字节到达 smem）。生产者写完 smem 后 arrive，
    // 消费者 wait 到达后才能读。这样实现生产-消费的流水线。
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_q.init(1);    // Q 的 barrier：1 个生产者 arrive 即可
        if constexpr (CLUSTER_SIZE == 2) {
            // 【CLUSTER_SIZE=2 时的 barrier 配置】
            // bar_k_local_ready：本地 SM 的 K 缓冲就绪，128 个线程 arrive（整个 WG2）
            // bar_k_remote_ready：对端 SM 的 K 缓冲就绪，1 个 arrive（st_async 写完）
            // bar_k_avail：K 缓冲可被生产者复用，4 个 arrive（2 个 WG 消费者各 2 个 arrive）
            CUTE_UNROLL
            for (int i = 0; i < NUM_K_BUFS; ++i) {
                plan.bar_k_local_ready[i].init(128);
                plan.bar_k_remote_ready[i].init(1);
                plan.bar_k_avail[i].init(4);
            }
        } else {
            // 【CLUSTER_SIZE=1 时】没有 remote barrier，avail 需要 256 个 arrive
            // （WG0 和 WG1 各 128 线程，但实际只用部分线程 arrive）
            CUTE_UNROLL
            for (int i = 0; i < NUM_K_BUFS; ++i) {
                plan.bar_k_local_ready[i].init(128);
                plan.bar_k_avail[i].init(256);
            }
        }
        cutlass::arch::fence_barrier_init();   // 确保 barrier 初始化对其他线程可见
    }
    ku::barrier_cluster_arrive_relaxed();   // cluster 内所有 CTA 同步一下

    int bar_phase_k = 0; // Don't use array here to prevent using local memory
    // 【为什么不用数组】数组下标访问会让编译器把变量放到 local memory（慢），
    // 用标量 bar_phase_k 编译器会放在寄存器里，快得多。
    // 【什么是 phase】barrier 的"相位"，每次 arrive 后翻转，wait 时要匹配当前 phase。

    // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // Don't use PDL because of compiler bugs!
    // cudaGridDependencySynchronize();
    // 【为什么注释掉 PDL】PDL 是 H100 新特性，可以让前一个 kernel 快结束时
    // 就开始下一个 kernel，省等待时间。但作者发现编译器有 bug，禁用了。

    // ====== 步骤 5：读取调度元数据，确定本 block 要处理哪些 batch ======
    // 【什么是 sched_meta】调度器预先算好的"分工方案"——告诉每个 partition：
    //   - begin_req_idx / end_req_idx：本 partition 要处理 batch 的哪个范围
    //   - begin_block_idx / end_block_idx：每个 batch 的 topk block 范围
    //   - begin_split_idx：split 索引（用于合并时定位）
    //   - is_first/last_req_splitted：首尾 batch 是否被切分到多个 partition
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

    // 【为什么 return】如果 begin_req_idx 已经超出 batch 数，说明这个 partition
    // 没活干（多余的 SM），直接退出避免越界访问。
    if (sched_meta.begin_req_idx >= params.b) return;

    // ====== 步骤 6：用 TMA 异步加载 Q 到 smem ======
    // 【为什么用 TMA】Q 矩阵不大（BLOCK_M × HEAD_DIM_K × bf16 ≈ 64×576×2=72KB），
    // 一次加载到 smem，后面反复用。TMA 比手动 cp.async 快得多。
    // 【为什么只要 warp_idx==0 && elect_one】TMA 是 block 级操作，
    // 只需要一个线程发起，其他线程会通过 barrier 等待。
    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ = flat_divide(
            tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, head_block_idx, _0{});
        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
        // arrive_and_expect_tx：告诉 barrier "我期望收到 BLOCK_M*HEAD_DIM_K*sizeof(bf16) 字节"
        // 收到这么多字节后 barrier 才会释放，让消费者读 sQ
        plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
    }

    ku::barrier_cluster_wait_acquire();   // 等 cluster 内所有 CTA 都到达这一步

    // ====== 步骤 7：定义 lambda 获取每个 batch 的处理范围 ======
    // 【为什么需要这个 lambda】一个 partition 可能要处理多个 batch，
    // 每个 batch 的 topk_length/extra_topk_length 可能不同，需要动态查询。
    struct MainloopArgs {
        int start_block_idx, end_block_idx;  // 本 batch 的 topk block 范围 [start, end)
        bool is_no_split;                    // 本 batch 是否没有被切分（不需要合并）

        // The following fields are only valid for MODEL1
        // 【为什么只有 MODEL1】V3.2 的 topk 是固定的，MODEL1 支持动态 topk_length
        // 和额外的 extra KV cache（索引池），所以需要这些额外字段
        int topk_length, extra_topk_length, num_orig_kv_blocks;
    };
    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            // V3.2：topk 固定，直接用全局参数
            total_topk_padded = params.topk;
        } else {
            // MODEL1：topk_length 可能每个 batch 不同，要查表
            // 【__ldg】read-only cache load，比普通 load 快，不走 L1
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            // 【为什么 ceil 到 TOPK_BLOCK_SIZE】topk 不一定是 64 的倍数，但要按 block 处理，
            // 所以向上取整。无效位置后面用 is_kv_valid=false mask 掉
            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
            args.topk_length = topk_length;
            args.extra_topk_length = extra_topk_length;
            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
        }

        // 【关键：确定本 batch 的 block 范围】
        // 如果是 begin_req_idx：用 sched_meta.begin_block_idx（可能是非零，因为被前一个 partition 切走了一部分）
        // 如果是 end_req_idx：用 sched_meta.end_block_idx（可能小于总数，因为切给下一个 partition 了）
        // 中间的 batch：完整处理 [0, total_topk_padded/TOPK_BLOCK_SIZE)
        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
        // 【is_no_split 的含义】如果 batch 被 split 了，输出到 o_accum/lse_accum（待合并）；
        // 如果没 split，直接输出到 out/lse（最终结果）
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        return args;
    };

    // ==========================================================================
    // WG0：算 QK^T → softmax → 算左半 PV → 写回 O（head_dim_v 的前 256 维）
    // ==========================================================================
    // 【WG0 的核心职责】
    //   1. 用 Q 和 K 算 QK^T（attention 分数）
    //   2. 做 online softmax（和 scale_softmax 函数配合）
    //   3. 用 softmax 后的 S 和 V 算 PV（attention 输出的左半 256 维）
    //   4. 把 softmax 的 rescale 因子写到 sScale，给 WG1 用
    //   5. 把 softmax 后的 S 写到 sS，给 WG1 算右半 PV 用
    if (warpgroup_idx == 0) {
        // 【寄存器分配】WG0 要 192 个寄存器/线程（算 QK^T 和 PV 都要大量寄存器）
        // warpgroup_reg_alloc 告诉编译器给这个 WG 多分寄存器
        cutlass::arch::warpgroup_reg_alloc<192>();

        // 【TiledMMA 是什么】CUTLASS 的"矩阵乘法分块描述器"——告诉 WGMMA 硬件
        // 怎么把大矩阵乘法切成小块分给 warpgroup 内的 128 个线程
        TiledMMA tiled_mma_QK = TiledMMA_QK{};       // QK^T 的 MMA：64x64x16，Q/K 都在 smem
        ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);  // 本线程负责哪一片
        TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{}; // PV 的 MMA：64x256x16，S 在寄存器 V 在 smem
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

        // 【寄存器中的累加器】rO/rP/rS 都是寄存器里的 tensor，不占 smem
        float rL[2], rM[2];   // 每行一个 L（sum）和 M（max），2 行
        // rO：PV 的累加器，shape (BLOCK_M, HEAD_DIM_V/2) = (64, 256)，存 O 的左半
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        // rP：QK^T 的累加器，shape (BLOCK_M, TOPK_BLOCK_SIZE) = (64, 64)
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        // rS：softmax 后的 P（转 bf16），给 PV GEMM 当 A 矩阵用
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

        // 【attn_sink 是什么】attention 的"吸收槽"——对于完全不相关的 Q 和 K，
        // 给一个最低的分数（-INF），防止 softmax 后概率分布过于平摊。
        // 这里读入每个 head 的 attn_sink 值（转 log2 域：* CUDART_L2E_F）
        float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
        if (params.attn_sink != nullptr) {
            for (int i = 0; i < 2; ++i) {
                int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
                rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
            }
        }

        // ====== 主循环：遍历本 partition 负责的所有 batch ======
        // 【为什么 #pragma unroll 1】不让编译器展开循环——展开会让代码膨胀，
        // 寄存器压力大，反而慢。每个 batch 独立处理，没有展开收益。
        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);

            // 每个 batch 开始前，重置累加器
            rL[0] = rL[1] = 0.0f;
            rM[0] = rM[1] = MAX_INIT_VAL;
            cute::fill(rO, 0.);

            // 等 Q 的 TMA 加载完成（双缓冲，phase 根据 batch_idx 翻转）
            plan.bar_q.wait((sched_meta.begin_req_idx-batch_idx)&1);

            // ====== 内层循环：遍历本 batch 的所有 topk block ======
            // 【为什么 CUTE_NO_UNROLL】不让编译器展开——每次迭代都有 barrier 同步，
            // 展开会让 barrier 顺序错乱
            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                // 双缓冲：交替使用 buf 0 和 buf 1，让生产者和消费者并行
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
                Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});

                // 等 K 缓冲就绪（生产者 WG2 已经把 K 反量化好写到 smem）
                plan.bar_k_local_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                if constexpr (CLUSTER_SIZE == 2) {
                    plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);  // 等对端 SM 也准备好
                }

                // ====== 步骤 A：算 QK^T ======
                // gemm<true, -1>：true=用 K 做主矩阵，-1=累加到 rP（-1 表示不缩放，直接累加）
                gemm<true, -1>(
                    tiled_mma_QK,
                    thr_mma_QK.partition_fragment_A(sQ),
                    thr_mma_QK.partition_fragment_B(sK),
                    rP
                );

                bar_phase_k ^= 1<<buf_idx;   // 翻转 phase，下次 wait 用新 phase

                cute::warpgroup_wait<0>();   // 等 WGMMA 完成，rP 才能读

                // ====== 步骤 B：做 softmax ======
                // 【为什么除了第一个 block 都要等】sScale 和 sS 是 WG0 写、WG1 读的共享资源。
                // 上一轮 WG1 可能还在读 sScale/sS，必须等它读完（sScale_and_sS_free）才能覆盖。
                if (block_idx != args.start_block_idx)
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);  // Make sure that sScale and sS is free

                // 调用前面的 scale_softmax 函数：mask + online softmax + rescale rO
                // Since in our case TOPK_BLOCK_SIZE == BLOCK_M, so we only need to do OOB checking for the last 2 blocks
                scale_softmax(rP, rS, rO, params.sm_scale_div_log2, sScale, rM, rL, plan.is_kv_valid[buf_idx], block_idx, idx_in_warpgroup);

                // ====== 步骤 C：把 softmax 后的 S 写到 smem，给 WG1 算右半 PV ======
                // save_rPb_to_sP：用 stmatrix 指令把寄存器里的 rS 搬到 smem sS
                save_rPb_to_sP(rS, sS, idx_in_warpgroup);
                // fence_view_async_shared：确保异步写操作对其他线程可见
                fence_view_async_shared();

                // ====== 步骤 D：算 O += S @ V（左半 256 维）======
                // gemm<false, -1>：false=S 在寄存器，-1=累加到 rO
                gemm<false, -1>(
                    tiled_mma_PV,
                    rS,
                    thr_mma_PV.partition_fragment_B(sV),
                    rO
                );

                // 通知 WG1：sScale 和 sS 已经准备好了，可以读
                NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

                cute::warpgroup_wait<0>();   // 等 PV 的 WGMMA 完成

                // 通知生产者：本 buf 的 K/V 我用完了，可以覆盖写新数据了
                if constexpr (CLUSTER_SIZE == 2) {
                    // 【为什么 arrive 两次】cluster 模式下 bar_k_avail 需要 4 个 arrive，
                    // WG0 这里 arrive 2 次（idx 32 和 64 是 WG0 内的两个 warp 代表）
                    plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                    plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                } else {
                    plan.bar_k_avail[buf_idx].arrive();
                }
            }

            // ====== 预取下一个 batch 的 Q ======
            // 【为什么在这里预取】当前 batch 的计算快结束了，提前发起下一个 batch 的
            // Q 加载，可以和当前 batch 的最后几个 block 计算重叠，省等待时间。
            if (threadIdx.x/32 == 0 && elect_one_sync()) {
                if (batch_idx != sched_meta.end_req_idx) {
                    Tensor gQ = flat_divide(
                        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx+1),
                        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
                    )(_, _, head_block_idx, _0{});
                    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
                    plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
                } else {
                    // This kernel is followed by the combine kernel, so we signal PDL here
                    // 【PDL 触发】通知下一个 kernel（combine）可以开始了
                    cudaTriggerProgrammaticLaunchCompletion();
                }
            }

            // ====== 步骤 E：跨 warp 归约 L 和 M ======
            // 【为什么需要归约】每个 warp 算了部分 L/M，需要加起来得到整行的 L/M。
            // 用 butterfly reduce（shfl_xor）2 步归约 4 个线程的部分和
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

            // 把归约后的 L/M 写到 smem，给 epilogue（写回阶段）用
            if (idx_in_warpgroup%4 == 0) {
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    sL[row] = rL[i];
                    sM[row] = rM[i];
                }
            }

            // ====== 步骤 F：算 o_scales（最终输出 O 的归一化因子）======
            // 【o_scales 是什么】O = sum(P·V) 还没归一化，要除以 L 才是 attention 输出。
            // o_scales = 1/L，后面 store_o 时会乘上。
            // 【为什么分 is_no_split】
            //   - is_no_split=true：最终输出，要考虑 attn_sink（吸收槽）
            //     o_scales = 1/(L + exp(attn_sink - M))，attn_sink 项把"完全不相关"的概率也加进来
            //   - is_no_split=false：输出到 o_accum，后面还要合并，先不归一化 attn_sink
            //     o_scales = 1/L
            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                if (args.is_no_split) {
                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i] + exp2f(rAttn_sink[i] - rM[i]));
                } else {
                    o_scales[i] = rL[i] == 0.0f ? 0.0f : __fdividef(1.0f, rL[i]);
                }
                if (idx_in_warpgroup%4 == 0) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    plan.sOScale[row] = o_scales[i];   // 写到 smem 给 WG1 也用（WG1 算右半 O）
                }
            }

            // ====== 步骤 G：同步点——WG0 等 WG1 用完 oBuf，WG1 等 WG0 写好 sL ======
            // This is a synchronization point for warpgroup 0/1.
            // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
            // Warpgroup 1 should wait wg 0 for sL to be ready
            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            // 防 0 除：L=0 时改成 1（虽然结果没意义，但不会 NaN）
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i)
                rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];

            int start_head_idx = head_block_idx*BLOCK_M;
            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);  // 处理 head 数不是 64 倍数的边界
            // ====== 步骤 H：写回输出 ======
            // 【分两种情况】
            //   - is_no_split=true：直接写到最终输出 out/lse
            //   - is_no_split=false：写到 o_accum/lse_accum，后面由 combine kernel 合并
            if (args.is_no_split) {
                // 直接写最终输出 O（bf16）和 lse（log sum exp）
                bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_h_q, _1{})
                ));
                float* gSoftmaxLse = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + start_head_idx;	// (BLOCK_M) : (1)

                // store_o 把 rO 从寄存器搬到 smem 再用 TMA 写到 gmem
                store_o<true>(rO, gO, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                // 写 lse = log(L) + M*log2(e)（从 log2 域转回自然域）
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
                }

                cute::tma_store_wait<0>();   // 等 TMA 写完成
            } else {
                // 写到 o_accum/lse_accum（待 combine kernel 合并）
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
                float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_accum_h_q, _1{})
                ));
                store_o<false>(rO, gOAccum, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
                }

                cute::tma_store_wait<0>();
            }
            
            sync_all_threads_in_cluster();
        }
    // ==========================================================================
    // WG1：算右半 PV（HEAD_DIM_V 的后 256 维）
    // ==========================================================================
    // 【WG1 的核心职责】
    //   1. 从 smem 读 WG0 写好的 sS（softmax 后的 P）
    //   2. 从 smem 读 sScale，rescale 自己的 rO（和 WG0 保持一致）
    //   3. 用 S 和 V 的右半算 PV，累加到 rO
    //   4. 写回 O 的右半
    // 【为什么 WG0 和 WG1 分算左右半】HEAD_DIM_V=512 太大，一个 WG 算不下，
    // 拆成两半各算 256 维，可以并行加速。
    } else if (warpgroup_idx == 1) {
        // 【寄存器分配】WG1 只算 PV（不算 QK^T），需要的寄存器少，让出 32 个给 WG0
        cutlass::arch::warpgroup_reg_dealloc<160>();

        // 【RemoteP 的含义】P（softmax 后的 S）不在自己的寄存器，在 smem（remote），
        // 所以用 SS 类型的 MMA（A 和 B 都在 smem）
        TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        // rO：累加器，shape (BLOCK_M, HEAD_DIM_V/2) = (64, 256)，存 O 的右半
        Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});

        #pragma unroll 1
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            cute::fill(rO, 0.);

            CUTE_NO_UNROLL
            for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;
                Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

                // 等 WG0 把 sScale 和 sS 写好
                NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

                // ====== 用 sScale rescale 自己的 rO ======
                // 【为什么 WG1 也要 rescale】WG0 更新 max 后，旧的 O 要缩到新 max 下。
                // WG1 的 rO 和 WG0 的 rO 用同一组 max，所以也要乘同样的因子。
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup/4)*2);
                CUTE_UNROLL
                for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
                    CUTE_UNROLL
                    for (int i = 0; i < size(cur_rO); ++i) {
                        cur_rO(i) *= cur_scales[local_row_idx];
                    }
                }
                
                // ====== 算 O += S @ V（右半 256 维）======
                // S 从 smem（sS），V 从 smem（sV）
                gemm<false, -1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sS),
                    thr_mma_PV.partition_fragment_B(sV),
                    rO
                );
                cute::warpgroup_wait<0>();

                // 通知生产者：本 buf 的 K/V 用完了
                if constexpr (CLUSTER_SIZE == 2) {
                    // 【为什么 arrive 两次】cluster 模式下 bar_k_avail 需要 4 个 arrive，
                    // WG1 这里 arrive 2 次（idx 32 和 64 是 WG1 内的两个 warp 代表）
                    plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                    plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                } else {
                    plan.bar_k_avail[buf_idx].arrive();
                }

                // 通知 WG0：sScale 和 sS 我读完了，可以覆盖（除了最后一个 block）
                if (block_idx != args.end_block_idx-1)
                    NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);   // Tell WG0 that sScale and sS are available
            }

            // 等 WG0 算好 sL（用于 o_scales）并让出 oBuf
            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            // 从 smem 读 o_scales（WG0 算好写到 sOScale 的）
            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = get_AorC_row_idx(i, idx_in_warpgroup);
                o_scales[i] = plan.sOScale[row];
            }
                
            int start_head_idx = head_block_idx*BLOCK_M;
            int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
            // 写回右半 O（和 WG0 写左半的逻辑一样，只是 head_dim 维度的偏移不同）
            if (args.is_no_split) {
                bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_h_q, _1{})
                ));

                store_o<true>(rO, gO, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                cute::tma_store_wait<0>();
            } else {
                int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.stride_o_accum_h_q, _1{})
                ));
                store_o<false>(rO, gOAccum, sOBuf, sOAccumBuf, plan, o_scales, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                cute::tma_store_wait<0>();
            }

            sync_all_threads_in_cluster();   // 等 cluster 内所有 CTA 完成本 batch
        }
    // ==========================================================================
    // WG2（else 分支）：生产者——从 gmem 读 FP8 K/V，反量化成 bf16 写到 smem
    // ==========================================================================
    // 【WG2 的核心职责】
    //   1. 从 gmem 读 FP8 格式的 K/V（节省带宽）
    //   2. 用 scale 反量化成 bf16（WGMMA 只吃 bf16 不吃 fp8）
    //   3. 写到 smem 给 WG0/WG1 消费
    //   4. 如果 CLUSTER_SIZE=2，还要异步写到对端 SM 的 smem（共享劳动成果）
    } else {
        // Producer warpgroup
        // 【寄存器分配】生产者不需要太多寄存器，让出更多给 WG0/WG1
        cutlass::arch::warpgroup_reg_dealloc<152>();

        static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
        // 【NUM_TOKENS_PER_THREAD】每个线程一次处理几个 token
        //   - CLUSTER_SIZE=1：2 个（一个 CTA 处理 64 个 token，128 线程×2=256，要 2 轮？实际是 32 token/轮×2 轮=64）
        //   - CLUSTER_SIZE=2：1 个（两个 CTA 各处理 32 个 token，128 线程×1=128，要 1 轮？实际是 32 token/轮×1 轮=32）
        static constexpr int NUM_TOKENS_PER_THREAD = CLUSTER_SIZE == 1 ? 2 : 1;
        static constexpr int NUM_TOKENS_PER_ROUND = 32; // If head is 128, each CTA is responsible for dequantizing 32 tokens (1 rounds); if head is 64, each CTA is responsible for dequantizing 64 tokens (2 rounds)
        int warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);  // 本线程在哪个 warp（用 shfl 广播，避免本地除法）
        int lane_idx = idx_in_warpgroup % 32;   // warp 内编号 0..31
        int my_token_idx_base = warp_idx*8 + lane_idx%8;  // 本线程负责的 token 基地址
        
        // ====== 主循环：遍历本 partition 负责的所有 batch ======
        CUTE_NO_UNROLL
        for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
            MainloopArgs args = get_cur_req_info(batch_idx);
            // gIndices：topk 选出的 token 在 KV cache 中的全局索引
            int* gIndices = params.indices + batch_idx*params.stride_indices_b + s_q_idx*params.stride_indices_s_q; // (topk) : (1)
            // gExtraIndices：MODEL1 额外索引池（extra KV cache）的 token 索引
            int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q; // (extra_topk) : (1)

            // 预取第一个 block 的 token 索引（后面边算边预取下一个 block）
            int nxt_token_indexs[NUM_TOKENS_PER_THREAD];
            CUTE_UNROLL
            for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                if (MODEL_TYPE == ModelType::V32 || args.start_block_idx < args.num_orig_kv_blocks)
                    nxt_token_indexs[round] = __ldg(gIndices + args.start_block_idx*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + round*NUM_TOKENS_PER_ROUND + my_token_idx_base);
            }

            // 【用空 struct 当"标签类型"】C++ 模板技巧——用类型来区分"原始 block"
            // 还是"extra block"，让编译器在编译期生成不同的代码分支，比运行时 if 快
            struct IsOrigBlock {};
            struct IsExtraBlock {};

            struct IsFirstExtraBlock {};
            struct IsNotFirstExtraBlock {};
            // process_one_block：处理一个 topk block（64 个 token）的反量化
            auto process_one_block = [&](int block_idx, auto is_extra_block_t, auto is_first_extra_block_t) {
                static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
                static constexpr bool IS_FIRST_EXTRA_BLOCK = std::is_same_v<decltype(is_first_extra_block_t), IsFirstExtraBlock>;
                int buf_idx = (block_idx-args.start_block_idx) % NUM_K_BUFS;

                // 根据 IS_EXTRA_BLOCK 选择用原始 KV cache 还是 extra KV cache 的参数
                int* indices_base;
                int page_block_size;
                int64_t k_block_stride, k_row_stride;
                fp8* k_ptr;
                if constexpr (!IS_EXTRA_BLOCK) {
                    // 原始 KV cache
                    indices_base = gIndices + (block_idx)*TOPK_BLOCK_SIZE;
                    page_block_size = params.page_block_size;
                    k_block_stride = params.stride_kv_block;
                    k_row_stride = params.stride_kv_row;
                    k_ptr = (fp8*)params.kv;
                } else {
                    // extra KV cache（索引池）
                    indices_base = gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE;
                    page_block_size = params.extra_page_block_size;
                    k_block_stride = params.stride_extra_kv_block;
                    k_row_stride = params.stride_extra_kv_row;
                    k_ptr = (fp8*)params.extra_kv;
                }
                [[maybe_unused]] int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
                [[maybe_unused]] int rel_block_idx = IS_EXTRA_BLOCK ? (block_idx - args.num_orig_kv_blocks) : block_idx;
                // peer_bar_k_remote_ready：对端 SM 的 barrier 地址（cluster 模式下用）
                transac_bar_t* peer_bar_k_remote_ready = get_peer_addr(&(plan.bar_k_remote_ready[buf_idx]));

                CUTE_UNROLL
                for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
                    int my_token_idx = my_token_idx_base + round*NUM_TOKENS_PER_ROUND;
                    // sK_nope_base：本线程要写的 smem 中 nope 部分的基地址
                    bf16* sK_nope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*16)*TOPK_BLOCK_SIZE;
                    // sK_nope_peer_base：对端 SM 的对应地址（cluster 模式下 st_async 写过去）
                    bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);

                    // ====== 取出本 token 在 KV cache 中的全局索引 ======
                    int token_index;
                    if constexpr (!IS_EXTRA_BLOCK) {
                        // 原始 block：用预取的索引，并预取下一个 block 的索引
                        token_index = nxt_token_indexs[round];
                        if (block_idx+1 != (MODEL_TYPE == ModelType::V32 ? args.end_block_idx : args.num_orig_kv_blocks))
                            nxt_token_indexs[round] = __ldg(gIndices + (block_idx+1)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                    } else {
                        // extra block：第一个 extra block 要现场读索引（之前没预取），
                        // 后续 extra block 用预取的索引
                        if constexpr (IS_FIRST_EXTRA_BLOCK) {
                            token_index = __ldg(gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                        } else {
                            token_index = nxt_token_indexs[round];
                        }
                        if (block_idx+1 != args.end_block_idx)
                            nxt_token_indexs[round] = __ldg(gExtraIndices + (block_idx+1-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx);
                    }

                    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
                        // For MODEL1, we need to check whether the token_index is within topk_length
                        // 【为什么检查】MODEL1 的 topk_length 可能不是 64 的倍数，
                        // padding 位置的索引可能是 INT_MAX 等无效值，直接用会越界访问
                        if (rel_block_idx*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx >= topk_length) {
                            token_index = -1;   // 标记为无效，后面用 0 填充
                        }
                    }

                    // 把全局 token 索引拆成 (block_index, rel_idx_in_block)
                    // 【为什么用 uint32_t 除法】比 int 除法快好几倍
                    // 【为什么 token_index==-1 时 block_index=0】避免越界访问，
                    // 反正后面会用 scale=0 把数据清零
                    int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);   // Use uint32_t division and mod to improve performance
                    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;   // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size < page_block_size, so there will be no illegal-memory-access error

                    // ====== 读取 scale（反量化用）======
                    // 【V3.2 和 MODEL1 的 scale 布局不同】
                    //   V3.2：4 个 fp32 scale，跟在 512 维 fp8 K 后面
                    //   MODEL1：8 个 fp8_e8m0 scale，存在单独的 scale 区
                    fp8* gK_base;
                    bf16 scales[NUM_SCALES];
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        static_assert(NUM_SCALES == 4);
                        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*k_row_stride;
                        float scales_float[NUM_SCALES];
                        // load_128b_from_gmem：一次读 16 字节（4 个 float），带 L1/L2 hint
                        // EVICT_LAST：尽量留在 L1（马上还要用 scale 反量化）
                        *(float4*)(scales_float) = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>((float*)(gK_base+HEAD_DIM_NOPE));
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; ++i) {
                            scales[i] = (bf16)scales_float[i];   // fp32 → bf16（后面反量化用 bf16 乘）
                        }
                    } else {
                        static_assert(NUM_SCALES == 8);
                        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));
                        // MODEL1 的 scale 区：在 page block 末尾，每个 token 8 个 fp8_e8m0
                        fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16)) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));
                        fp8_e8m0 scales_e8m0[NUM_SCALES];
                        *(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);   // 一次读 8 字节（8 个 e8m0）
                        // e8m0 → bf16 转换（2 个一组转换，更快）
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; i += 2) {
                            *(__nv_bfloat162_raw*)(scales+i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0+i));
                        }
                    }

                    // 等 K 缓冲可用（消费者 WG0/WG1 用完了，通知生产者可以覆盖）
                    if (round == 0) {
                        plan.bar_k_avail[buf_idx].wait((bar_phase_k>>buf_idx&1)^1);
                    }

                    // CLUSTER_SIZE=2 时，通知对端 SM：我要给你写数据了，预计多少字节
                    if (CLUSTER_SIZE == 2 && round == 0 && idx_in_warpgroup == 0) {
                        plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx((TOPK_BLOCK_SIZE/2)*(HEAD_DIM_NOPE+HEAD_DIM_ROPE)*sizeof(bf16));
                    }

                    // ====== 读 FP8 K 的 nope 部分，反量化成 bf16 写到 smem ======
                    fp8* gK_nope = gK_base + (lane_idx/8)*16;
                    if (token_index == -1) {
                        // 无效 token：scale 清零，后面反量化结果就是 0
                        CUTE_UNROLL
                        for (int i = 0; i < NUM_SCALES; ++i)
                            scales[i] = (bf16)0.0f;
                    }
                    // nope 部分按 64 维一组处理（V3.2 是 512/64=8 组，MODEL1 是 448/64=7 组）
                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE/64; dim_idx += 1) {
                        // 一次读 16 个 fp8（16 字节），带 L2 prefetch hint B256（预取 256 字节）
                        fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_nope + dim_idx*64);   // We use EVICT_LAST here since gK_base may not be aligned to 32B (for V3.2) and the performance is the best among all cache hints (for MODEL1)
                        // 【为什么 scale 索引不同】
                        //   V3.2：每 128 维用 1 个 scale（NUM_SCALES=4，512/128=4）
                        //   MODEL1：每 64 维用 1 个 scale（NUM_SCALES=8，448/64=7，+1 padding）
                        bf16 scale = scales[MODEL_TYPE == ModelType::V32 ? dim_idx/2 : dim_idx];
                        // 反量化并把 8 个 bf16 写到 smem 的 lambda
                        auto dequant_and_save_bf16x8 = [&](const fp8x8 &data, int offset) {
                            int smem_offset = (dim_idx*64 + offset) * TOPK_BLOCK_SIZE;
                            // cvt_fp8x8_bf16x8：PTX 指令把 8 个 fp8 转成 8 个 bf16，同时乘上 scale
                            bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(data, __bfloat162bfloat162(*(__nv_bfloat16*)(&scale)));
                            // 写到本地 SM 的 smem
                            *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                            if constexpr (CLUSTER_SIZE == 2) {
                                // st_async_128b：异步写到对端 SM 的 smem（cluster 共享）
                                st_async_128b(sK_nope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                            }
                        };
                        // 无效 token：fp8 数据清零，反量化后就是 0
                        if (token_index == -1)
                            *(uint128_t*)(&cur_fp8x16) = uint128_t();
                        // 把 16 个 fp8 拆成两个 8，分别反量化
                        dequant_and_save_bf16x8(cur_fp8x16.lo, 0);
                        dequant_and_save_bf16x8(cur_fp8x16.hi, 8);
                    }

                    // ====== 读取 rope 部分（64 维 bf16，不需要反量化）======
                    // 【为什么 rope 不需要反量化】rope 部分本来就是 bf16 存的（旋转位置编码
                    // 的数学特性决定的），只有 nope 部分是 fp8 量化的
                    bf16* gK_rope;
                    if constexpr (MODEL_TYPE == ModelType::V32) {
                        // V3.2：rope 在 nope + scale 之后
                        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE+NUM_SCALES*sizeof(float)) + (lane_idx/8)*8;
                    } else {
                        // MODEL1：rope 紧跟在 nope 之后
                        gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE) + (lane_idx/8)*8;
                    }
                    bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*8)*TOPK_BLOCK_SIZE;
                    bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

                    // rope 按 32 维一组处理（64/32=2 组）
                    CUTE_UNROLL
                    for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
                        // 直接读 8 个 bf16（16 字节），不需要反量化
                        bf16x8 cur_bf16x8 = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx*32);
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            // NOTE We do not need to mask the RoPE part for V3.2 since it isn't involved in the SV gemm
                            // 【为什么 V3.2 不用清零 rope】V3.2 的 rope 部分不参与 SV 计算，
                            // 即使有垃圾数据也无所谓。MODEL1 的 rope 会参与计算，所以要清零
                        } else {
                            if (token_index == -1)
                                *(uint128_t*)(&cur_bf16x8) = uint128_t();
                        }
                        int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
                        *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                        if constexpr (CLUSTER_SIZE == 2) {
                            st_async_128b(sK_rope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                        }
                    }
                }

                // fence_view_async_shared：确保所有异步 smem 写操作完成
                fence_view_async_shared();

                // ====== 写 is_kv_valid 数组（给 scale_softmax 用）======
                if (idx_in_warpgroup < 32) {
                    // We put this after fence_view_async_shared() since this won't be read by async proxy
                    // is_kv_valid 是 bool 数组，scale_softmax 用它 mask 无效 token
                    auto is_index_valid = [&](int index, int offset_within_thread) -> bool {
                        if constexpr (MODEL_TYPE == ModelType::V32) {
                            return index != -1;
                        } else {
                            // MODEL1 还要检查是否在 topk_length 范围内
                            return index != -1 && rel_block_idx*TOPK_BLOCK_SIZE + lane_idx*2 + offset_within_thread < topk_length;
                        }
                    };
                    int2 indices = __ldg((int2*)(indices_base + lane_idx*2));   // 一次读 2 个 int
                    *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx*2]) = {
                        is_index_valid(indices.x, 0),
                        is_index_valid(indices.y, 1)
                    };
                }

                // 通知消费者：K 缓冲准备好了，可以读了
                plan.bar_k_local_ready[buf_idx].arrive();
                bar_phase_k ^= 1 << buf_idx;   // 翻转 phase
            };

            // ====== 遍历所有 block 调用 process_one_block ======
            if constexpr (MODEL_TYPE == ModelType::V32) {
                // V3.2：只有原始 KV cache，直接循环
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
                }
            } else {
                // MODEL1：先处理原始 block，再处理 extra block
                // 【为什么 extra block 要分两段】第一个 extra block 要现场读索引
                // （IS_FIRST_EXTRA_BLOCK），后续 extra block 用预取的索引
                CUTE_NO_UNROLL
                for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
                    process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
                }

                if (args.num_orig_kv_blocks < args.end_block_idx) {
                    process_one_block(max(args.start_block_idx, args.num_orig_kv_blocks), IsExtraBlock{}, IsFirstExtraBlock{});
                }
                CUTE_NO_UNROLL
                for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks)+1; block_idx < args.end_block_idx; ++block_idx) {
                    process_one_block(block_idx, IsExtraBlock{}, IsNotFirstExtraBlock{});
                }
            }

            sync_all_threads_in_cluster();   // 等 cluster 内所有 CTA 完成本 batch
        }
    }
#else
    // 【为什么有这个 #else】只编译到 SM90（H100）架构。其他架构报错——
    // 这个 kernel 用了 WGMMA、TMA 等 H100 专有指令，老 GPU 跑不了。
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif

}

// ============================================================================
// flash_fwd_splitkv_mla_fp8_sparse_kernel —— 实际的 CUDA kernel 入口
// ============================================================================
// 【__launch_bounds__】告诉编译器：每个 block 最多 NUM_THREADS=384 个线程，
// 最少 1 个 block/SM，cluster 大小 CLUSTER_SIZE。这帮编译器优化寄存器分配。
template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
    // __grid_constant__：告诉编译器 params/tma_params 在整个 kernel 生命周期内不变，
    // 可以放进常量缓存，访问比普通 gmem 参数快得多
    Kernel::devfunc(params, tma_params);
}

// ============================================================================
// run —— host 端入口：参数检查、TMA 描述符构造、kernel 启动
// ============================================================================
template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    // ====== 参数检查（KU_ASSERT 在 release 模式下可能被去掉）======
    // 【为什么 assert】在 host 端尽早发现问题，比在 device 端崩溃好排查
    KU_ASSERT(params.h_kv == 1);                        // MLA 的 KV head 数固定为 1
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);      // topk 必须是 64 的倍数
    KU_ASSERT(params.d_qk == HEAD_DIM_K);               // Q/K 维度必须匹配
    KU_ASSERT(params.d_v == HEAD_DIM_V);                // V 维度必须匹配
    KU_ASSERT(params.h_q % BLOCK_M == 0);               // head 数必须是 64 的倍数
    if constexpr (MODEL_TYPE == ModelType::MODEL1) {
        // MODEL1：每个 token = 448 字节 fp8 + 64*2 字节 bf16 rope + 8 字节 scale = 584 字节
        // 【为什么要求 contiguous】kernel 里用 load_128b_from_gmem 一次读 16 字节，
        // 如果 row stride 不等于实际数据大小，会有 padding 垃圾数据
        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
        if (params.extra_kv != nullptr) {
            KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in MODEL1");  // Each block must be contiguous
        }
    } else {
        // V3.2：不支持 extra KV cache 和动态 topk_length
        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
        // V3.2 每个 token = 512 字节 fp8 + 4*4 字节 fp32 scale + 64*2 字节 bf16 rope = 656 字节
        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
    }

    // ====== 构造 Q 的 TMA 描述符 ======
    // 【TMA 描述符是什么】一个 128 字节的结构，描述张量的基地址、形状、stride、
    // swizzle 模式等。TMA 硬件根据它自动做地址计算和数据搬运。
    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
            )
        ),
        SmemLayoutQ{}   // 告诉 TMA：数据在 smem 里要按 SmemLayoutQ 的 swizzle 布局
    );

    // ====== 手动构造 O 的 5D TMA 描述符 ======
    // 【为什么要手动构造】CUTLASS 的 make_tma_copy 不支持 5D，
    // 但 O 的输出布局需要 5D（batch × s_q × head × head_dim/sw × sw）
    CUtensorMap tensor_map_o;
    {
        // Here we manually construct TMA descriptor to store O, in order to leverage 5D TMA
        uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
        uint64_t stride[4] = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
        uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW, 1, 1};
        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_o,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            5,
            params.out,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            // 【swizzle 模式】根据 OBUF_SW 选不同的 swizzle——这是 smem 的字节重排技巧，
            // 可以避免 bank conflict，让 WGMMA 读得更快
            OBUF_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B :
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    // 把 Q 的 shape、Q 的 TMA、O 的 TMA 打包成 tma_params，传给 kernel
    TmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q,
        tensor_map_o
    };
    // mla_kernel：函数指针，指向特化后的 kernel
    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, decltype(tma_params)>;

    // ====== 设置 smem 大小并启动 kernel ======
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    // 【为什么要 setAttribute】默认 smem 上限是 48KB，这个 kernel 要 ~150KB，
    // 必须动态申请扩大。cudaFuncSetAttribute 告诉运行时"我要用更多 smem"
    KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // NOTE Don't use PDL because of potential compiler bugs!
    // 【被注释掉的 PDL 启动代码】正常应该用 cudaLaunchKernelEx 启动并开启 PDL，
    // 但因为编译器 bug，改用下面的 cutlass::launch_kernel_on_cluster
    // cudaLaunchAttribute mla_kernel_attributes[1];
    // mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    // cudaLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_k, params.num_sm_parts),
    //     dim3(NUM_THREADS, 1, 1),
    //     smem_size,
    //     stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    // ====== 启动配置 ======
    // grid: (NUM_M_BLOCKS, s_q, num_sm_parts)
    //   - NUM_M_BLOCKS: head_block 数（64 或 128 head → 1 或 2）
    //   - s_q: query token 数（解码时通常 1）
    //   - num_sm_parts: Split-KV 的 partition 数（由调度器决定）
    // block: (NUM_THREADS=384, 1, 1)
    // cluster: (CLUSTER_SIZE, 1, 1) —— 1 或 2 个 CTA 协作
    cutlass::ClusterLaunchParams launch_params = {
        dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        dim3(CLUSTER_SIZE, 1, 1),
        smem_size,
        params.stream
    };
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)mla_kernel, params, tma_params
    );
    KU_CHECK_KERNEL_LAUNCH();   // 检查 kernel 启动是否成功（不检查执行结果）
}

// ============================================================================
// run_flash_splitkv_mla_fp8_sparse_kernel —— 外部调用入口
// ============================================================================
// 根据 MODEL_TYPE 和 NUM_HEADS 模板参数，调用对应的 KernelTemplate::run。
// Python 层通过 pybind 调用这个函数。
template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
