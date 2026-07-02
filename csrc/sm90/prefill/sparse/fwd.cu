// ============================================================================
// fwd.cu —— SM90 稀疏注意力前向 kernel 的"分派入口"
// ============================================================================
//
// 【这段代码解决什么实际问题？】
//
// 真正干活的 kernel 在 phase1.cuh 里，它是一个模板：
//
//     template<int D_QK, bool HAVE_TOPK_LENGTH>
//     void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);
//
// 模板参数 D_QK（Q/K 维度）和 HAVE_TOPK_LENGTH（是否支持变长 topk）不同，
// 会编译出"不同的 kernel 函数"。运行时该用哪一个？要根据用户实际传的参数决定：
//
//   - D_QK 取值：512 (MODEL1 模型) 或 576 (V3.2 模型) → 看 params.d_qk
//   - HAVE_TOPK_LENGTH：true 表示每个序列的 topk 数可变（params.topk_length != nullptr）
//                       false 表示所有序列用同一个 topk（params.topk_length == nullptr）
//
// 两个维度组合出 4 种情况，对应 4 个模板实例。本文件就是根据运行时参数选一个实例调用。
//
// 【为什么不让编译器自动选】
//
// 模板参数是编译期决定的，而 params.d_qk 是运行时才知道的值。C++ 不能在运行时
// "动态挑模板参数"——只能写 if-else 把 4 种情况都列出来，由运行时分支选择。
// 这种写法在 CUDA 项目里很常见，叫"模板特化分派"（template specialization dispatch）。
//
// 【这 4 个分支最后都调到同一个 phase1.cuh 里的代码吗】
//
// 不完全是。phase1.cuh 里的 `KernelTemplate<D_QK, HAVE_TOPK_LENGTH>` 是模板类，
// 不同模板参数会编译出 4 份独立的机器码。它们共享同一份源代码，但编译器会根据
// 模板参数做不同的优化（比如 D_QK=576 比 512 多算一个 tile，编译器会把那个
// 多出的 tile 算法直接 inline 进循环；HAVE_TOPK_LENGTH=true 会多一段读 topk_length
// 数组的逻辑）。
//
// 【整体调用链】
//
//   用户 Python 代码
//        ↓
//   flash_mla.cuda.sparse_prefill_fwd(...)   ← PYBIND11 绑定的 C++ 函数
//        ↓
//   构造 SparseAttnFwdParams（在 csrc/api/sparse_fwd.h 里）
//        ↓
//   sm90::run_fwd_kernel(params)             ← 本文件的入口
//        ↓
//   根据 d_qk / topk_length 分派到 4 个实例之一
//        ↓
//   sm90::fwd::run_fwd_phase1_kernel<512/576, true/false>(params)
//        ↓
//   KernelTemplate<D_QK, HAVE_TOPK_LENGTH>::run(params)
//        ↓ （在 phase1.cuh 里）
//   设置 TMA 描述符 → cluster launch → kernel 在 GPU 上跑
//
// ============================================================================

#include "fwd.h"

#include <stdexcept>

#include "phase1.h"

namespace sm90 {

// ----------------------------------------------------------------------------
// run_fwd_kernel —— 分派函数
// ----------------------------------------------------------------------------
// 【输入】params：所有 kernel 需要的参数打包成一个结构体（见 csrc/params.h 的
//                  SparseAttnFwdParams）。包含 Q/KV 数据指针、topk 索引、
//                  输出 O/max_logits/lse 指针、各种 stride、scale 等。
// 【输出】无（kernel 直接写到 params.out / params.max_logits / params.lse）
//
// 【为什么把分派逻辑放在单独的 .cu 文件里】
//   - fwd.h 只声明 run_fwd_kernel，不暴露模板细节，外部调用者不用关心 D_QK
//   - 模板实例化（instantiate）放在 instantiations/ 目录下的 .cu 文件，避免重复编译
//   - 本文件做"运行时分派"，是连接上层 API 和底层 kernel 的桥梁
void run_fwd_kernel(const SparseAttnFwdParams& params) {
    // 判断是否支持变长 topk：topk_length != nullptr 表示每个序列有自己的 topk 数
    // （比如序列 A 选 64 个 token，序列 B 选 128 个）。nullptr 表示所有序列用同一个
    // params.topk（更简单，也更快——少读一次 gmem）。
    const bool have_topk_length = params.topk_length != nullptr;

    // ====== 分派矩阵：2×2 = 4 种情况 ======
    //
    //                  │ HAVE_TOPK_LENGTH=false │ HAVE_TOPK_LENGTH=true
    //   ───────────────┼──────────────────────┼──────────────────────
    //   d_qk = 512     │  <512, false>         │  <512, true>
    //   d_qk = 576     │  <576, false>         │  <576, true>
    //
    // 对应 instantiations/ 目录下的 4 个 .cu 文件，每个文件 instantiate 一个实例：
    //   - phase1_k512.cu          → <512, false>
    //   - phase1_k512_topklen.cu  → <512, true>
    //   - phase1_k576.cu          → <576, false>
    //   - phase1_k576_topklen.cu  → <576, true>
    //
    // 为什么在分开的文件里 instantiate？因为模板实例化很慢（每个实例都要编译一遍
    // 整个 kernel），分开后可以并行编译，加链接速度。
    if (params.d_qk == 512) {
        if (have_topk_length) {
            sm90::fwd::run_fwd_phase1_kernel<512, true>(params);
        } else {
            sm90::fwd::run_fwd_phase1_kernel<512, false>(params);
        }
    } else if (params.d_qk == 576) {
        if (have_topk_length) {
            sm90::fwd::run_fwd_phase1_kernel<576, true>(params);
        } else {
            sm90::fwd::run_fwd_phase1_kernel<576, false>(params);
        }
    } else {
        // 如果 d_qk 不是 512 也不是 576，说明用户传错了参数（或未来加了新模型维度）。
        // 抛异常让上层 Python 代码看到清晰错误，比让 kernel 跑出错好排查。
        throw std::runtime_error("Unsupported d_qk value in sparse attention fwd kernel");
    }
}

}  // namespace sm90
