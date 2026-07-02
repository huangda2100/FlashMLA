// ============================================================================
// v32_persistent_h64.cu —— V3.2 模型 + 64 头的模板实例化文件
// ============================================================================
//
// 【这个文件解决什么实际问题？】
//
// splitkv_mla.cuh 里把 kernel 写成了"函数模板"：
//
//   template<ModelType MODEL_TYPE, int NUM_HEADS>
//   void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);
//
// 模板本身只是"代码蓝图"——编译器看到模板时不会生成机器码，因为不知道
// MODEL_TYPE 和 NUM_HEADS 具体是什么值。必须有人"告诉编译器：请用
// MODEL_TYPE=V32、NUM_HEADS=64 这组参数生成一份真正的函数"，这个函数才会
// 被编译成 GPU 上能跑的机器码。这一行 `template void ...<V32, 64>(...)`
// 就是这个"告诉"的动作，专业术语叫"模板显式实例化"（Explicit Instantiation）。
//
// 【为什么不在头文件里直接实例化】
//
// 如果在 splitkv_mla.cuh 里实例化，每个 #include 它的 .cu 文件都会生成
// 一份相同的函数，链接时报"重复定义"错误。所以单独放一个 .cu 文件，全局
// 只实例化一次，其他地方调用时链接到这份就行。
//
// 【为什么有 4 个 instantiation 文件】
//
// 项目支持 2 种模型 × 2 种 head 数 = 4 种组合：
//   - v32_persistent_h64.cu    : V3.2 + 64 头  ← 本文件
//   - v32_persistent_h128.cu   : V3.2 + 128 头
//   - model1_persistent_h64.cu : MODEL1 + 64 头
//   - model1_persistent_h128.cu: MODEL1 + 128 头
// Python 层根据用户传的 model_type 和 num_heads 选对应的那份调用。
//
// 【为什么文件名带 "persistent"】
//
// "Persistent kernel"（持久化 kernel）是一种优化技术——kernel 启动后不
// 立刻退出，而是循环处理多个 batch 的任务，省去反复启动 kernel 的开销。
// 本项目的 split-KV 解码 kernel 用了这种技术，所以文件名带上 persistent。
//
// 【为什么 .cu 不是 .cuh】
//
// .cuh 是头文件（被别人 #include），.cu 是"编译单元"（自己被编译成机器码）。
// 实例化必须放在 .cu 里——只有被编译器独立编译时才会真正生成函数代码。
//
// ============================================================================

#include "../splitkv_mla.cuh"   // 引入模板声明

namespace sm90::decode::sparse_fp8 {

// 模板显式实例化：请编译器用 <V32, 64> 这组参数生成一份函数代码
//   - ModelType::V32：DeepSeek V3.2 模型，HEAD_DIM_K=576（512 nope + 64 rope）
//   - 64：NUM_HEADS=64，Q 有 64 个注意力头
// 编译后这个文件会提供一份 run_flash_splitkv_mla_fp8_sparse_kernel<V32, 64>
// 的符号，其他 .cu 文件调用时链接到这份即可。
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, 64>(const SparseAttnDecodeParams &params);

}
