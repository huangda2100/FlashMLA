// ============================================================================
// 这个文件解决什么问题？
//
// 回想 Python 的调用链：
//   flash_mla_with_kvcache(indices=...)
//     → flash_mla_cuda.sparse_decode_fwd(...)
//       → sparse_attn_decode_interface(...)  ← 就是这个文件的主函数
//
// 这个文件是"稀疏解码注意力"从 Python 进入 C++ 后的第一站。它负责：
//   1. 检查参数对不对（维度、数据类型、设备是否匹配）
//   2. 判断你的 GPU 型号（H100 还是 B200）
//   3. 根据 head 数等配置，挑选最合适的 CUDA kernel
//   4. 组装参数，启动 kernel，合并结果，返回给 Python
//
// 可以把它理解成一个"工厂调度中心"——来了订单（稀疏注意力请求），
// 先验货（检查参数），再分配产线（选 kernel），最后打包发货（返回结果）。
// ============================================================================
#pragma once

// ---------- 头文件引用 ----------
// 每个 #include 都相当于从别的文件"借"来一些定义。
// 就像做饭前先把各种调料从柜子里拿出来放在台面上。
#include "common.h"   // 基类 ImplBase、架构检测 Arch 等通用工具

#include "params.h"   // 参数结构体（SparseAttnDecodeParams）的定义

// 下面是各个具体的 CUDA kernel 实现，按 GPU 架构和功能分类存放
#include "sm90/decode/sparse_fp8/splitkv_mla.h"          // H100 (SM90) 的稀疏解码 kernel
#include "sm100/decode/head64/kernel.h"                   // B200 (SM100) 64-head kernel
#include "sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h" // B200 128-head kernel
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h" // 调度方案生成工具
#include "smxx/decode/combine/combine.h"                  // Split-KV 结果合并工具

// ============================================================================
// DecodeFeatures —— 功能开关枚举
//
// 这个枚举定义了"稀疏解码 kernel 可能支持的各种功能"。
// 每个具体的 kernel 实现（后面那 4 个类）会声明"我支持哪些功能"。
// 调用时，系统会检查所需功能和实现的支持列表是否匹配。
//
// 就像点外卖——你勾选"加香菜"（功能），系统只给你看支持香菜的店家（实现）。
// ============================================================================
enum class DecodeFeatures : int {
    HEAD_64,              // 64 个 query head
    HEAD_128,             // 128 个 query head

    HEAD_DIM_576,         // 每个 head 的维度 = 576（DeepSeek V3.2 用）
    HEAD_DIM_512,         // 每个 head 的维度 = 512（另一种模型用）

    V32_KVCACHE_FORMAT,   // DeepSeek V3.2 格式的 KV Cache
    MODEL1_KVCACHE_FORMAT,// 另一种模型格式的 KV Cache

    ATTN_SINK,            // 支持 Attention Sink（稳定长上下文注意力）
    TOPK_LENGTH,          // 不同 query 可以有不同的 topk 值
    EXTRA_KVCACHE,        // 支持额外 KV Cache
    EXTRA_TOPK_LENGTH     // 额外 KV Cache 的 topk 长度
};

// ============================================================================
// DecodeImplMeta —— 每个 kernel 实现的"参数模板"
//
// 不同 GPU、不同 head 数，需要不同的调度参数。这个结构体定义了
// 每个实现必须提供的 3 个参数，用于指导 Split-KV 调度器如何工作。
// ============================================================================
struct DecodeImplMeta {
    int num_sm_parts;               // 把 GPU 的 SM（计算单元）分成几组？
    int fixed_overhead_num_blocks;  // 每个 SM 固定预留的 KV 块数（开销）
    int block_size_topk;            // 每个调度块一次处理几个 topk？
};

// ============================================================================
// DecodeImplBase —— 所有解码实现的"统一接口合同"
//
// 定义了所有解码实现（后面那 4 个类）必须遵守的规则：
//   1. 参数类型固定为 SparseAttnDecodeParams
//   2. 功能列表类型固定为 DecodeFeatures
//   3. 必须实现 get_meta() —— 告诉调用者自己的调度参数
//
// virtual ... = 0 的意思是"纯虚函数"——子类必须自己实现。
// 就像公司规定"所有部门都要做季度汇报"，但每个部门自己决定怎么汇报。
// ============================================================================
class DecodeImplBase : public ImplBase<
    SparseAttnDecodeParams,
    DecodeFeatures
> {
public:
    virtual DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

// ============================================================================
// Decode_Sm90_Impl —— 适用于 H100 GPU（SM90 架构）的稀疏解码实现
//
// SM90 = NVIDIA Hopper 架构（H100 GPU）。这个实现用统一的 kernel
// 覆盖所有配置——不管 64 head 还是 128 head，都用一个函数处理。
//
// 支持的功能列表：几乎全部（除了 HEAD_64、HEAD_128 都支持）
// ============================================================================
class Decode_Sm90_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            // num_sm_parts = SM总数 / query数 / (head数/64)
            // 例：132个SM / 1个query / (64头/64=1) = 132
            // 例：132个SM / 4个query / (128头/64=2) = 132/4/2 = 16
            std::max(arch.num_sms / s_q / (h_q/64), 1),
            5,   // 每个 SM 固定预留 5 个 KV 块的开销
            64   // 每次调度处理 64 个 topk
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        // 通过宏展开成模板参数，在编译期确定具体调用的 kernel
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
                // 调用 SM90 的稀疏 FP8 解码 kernel（splitkv 版本）
                sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>(params);
            });
        });
    }
};

// ============================================================================
// Decode_Sm100_Head64_Impl —— 适用于 B200 GPU（SM100 架构），64 head
//
// SM100 = NVIDIA Blackwell 架构（B200 GPU）。这个实现是 SM100 上专门
// 为 64 个 head 优化的 kernel。如果用户需要 128 head，有另外两个路径：
//   1. Head64x2：跑两次 64-head kernel（下面那个类）
//   2. Head128：用专门的 128-head kernel（再下面那个类）
// ============================================================================
class Decode_Sm100_Head64_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,       // 只支持 64 head
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            // 和 SM90 比，这里没有除以 (h_q/64)——因为 SM100 的 64-head kernel
            // 本身只处理 64 head，不需要根据 head 数调整分组。
            // 分组更细，并行度更高。
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            // 直接调用 SM100 的 64-head kernel
            sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE>(params);
        });
    }
};

// ============================================================================
// Decode_Sm100_Head64x2_Impl —— 跑两次 64-head kernel 实现 128 head
//
// 最巧妙的设计：SM100 的原生 kernel 只支持 64 head。当用户需要 128 head
// 时，把 128 个 head 分成两组（0-63 和 64-127），每次都调用 64-head kernel，
// 但把 Q / out / lse 等指针偏移到对应的 head 位置。
//
// 就像一台机器一次只能切 64 块蛋糕，你要切 128 块——那就切两次。
// 这是比专门写一个 128-head kernel 更快的方案（kernel 复用）。
// ============================================================================
class Decode_Sm100_Head64x2_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,      // ← 这个实现声明支持 HEAD_128（虽然是跑两次 64）
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            // 分两轮跑：
            // 第1轮：start_head_idx=0,  处理 head 0~63
            // 第2轮：start_head_idx=64, 处理 head 64~127
            for (int start_head_idx = 0; start_head_idx < 128; start_head_idx += 64) {
                // 复制参数，然后修改指针偏移到当前要处理的 head 位置
                SparseAttnDecodeParams cur_params = params;
                cur_params.q += start_head_idx * params.stride_q_h_q;     // Q 指针移到对应 head
                if (cur_params.attn_sink) {
                    cur_params.attn_sink += start_head_idx;                 // attn_sink 也偏移
                }
                cur_params.lse += start_head_idx;                          // lse 也偏移
                cur_params.out += start_head_idx * params.stride_o_h_q;    // 输出指针也偏移
                cur_params.lse_accum += start_head_idx;                    // 中间结果也偏移
                cur_params.o_accum += start_head_idx * params.stride_o_accum_h_q;
                cur_params.h_q = 64;   // 告诉 kernel 只处理 64 个 head
                // 调用 64-head kernel。cur_params 里的指针已经偏好了，
                // kernel 只管算自己的 64 head，不需要知道外面有 128 head。
                sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE>(cur_params);
            }
        });
    }
};

// ============================================================================
// Decode_Sm100_Head128_Impl —— SM100 上的专用 128-head kernel
//
// 这个实现走的是另一条路径：调用 sm100/prefill/ 目录下的 128-head kernel
// （虽然是 prefill 目录下的代码，但以 Decode 模式调用它）。
//
// 限制：不支持 V32_KVCACHE_FORMAT（只支持 MODEL1 格式的 KV Cache）。
// 如果用户需要 V32 格式 + 128 head，就只能走 Head64x2 路径。
// ============================================================================
class Decode_Sm100_Head128_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            // 和 Head64 的分组公式比，这里多除以了 2：
            // 因为 128 head 的工作量是 64 head 的两倍，每个 SM 组分到的
            // 工作更多，所以分组更少（num_sm_parts 更小）。
            std::max(arch.num_sms / s_q / 2, 1),
            3,   // 固定开销更少（3 vs 5），因为 128 的 kernel 更高效
            64   // 和 64-head 一样，每次调度处理 64 个 topk
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        // 调用 SM100 prefill 目录下的 128-head kernel（以 DecodeWithSplitKV 模式运行）
        sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(params);
    }
};

// ============================================================================
// sparse_attn_decode_interface —— 稀疏解码的 C++ 主入口
//
// 这是 Python 调用 flash_mla_cuda.sparse_decode_fwd 时实际进入的 C++ 函数。
// 整个流程可以分成 9 个步骤：
//
//   步骤 1：提取维度信息（从 tensor 中读出 batch、head 数等）
//   步骤 2：参数合法性检查（维度、数据类型、设备一致性）
//   步骤 3：分配输出 tensor（out 和 lse）
//   步骤 4：构建功能需求列表（需要哪些 DecodeFeatures）
//   步骤 5：选择具体实现类（根据 GPU 型号 + head 数）
//   步骤 6：组装参数结构体 SparseAttnDecodeParams
//   步骤 7：调度元数据懒初始化
//   步骤 8：分配 Split-KV 中间缓冲区
//   步骤 9：执行 kernel + 合并结果 + 返回
// ============================================================================
static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor &q,   // [b, s_q, h_q, d_qk]  —— query
    const at::Tensor &kv,   // [num_blocks, page_block_size, h_k, d_qk] —— KV Cache（FP8 格式）
    const at::Tensor &indices,    // [b, s_q, topk]  —— 每个 query 要关注的 KV 块索引
    const std::optional<at::Tensor> &topk_length,   // [b, s_q]  —— 每个 query 实际有效的 topk 数
    const std::optional<at::Tensor> &attn_sink, // [h_q]  —— Attention Sink 参数
    std::optional<at::Tensor> &tile_scheduler_metadata,   // 调度元数据（存/取）
    std::optional<at::Tensor> &num_splits,                // split 信息（存/取）
    const std::optional<at::Tensor> &extra_kv,            // 额外 KV Cache（可选）
    const std::optional<at::Tensor> &extra_indices,       // 额外 KV 的索引（可选）
    const std::optional<at::Tensor> &extra_topk_length,   // 额外 KV 的 topk 长度（可选）
    int d_v,
    float sm_scale
) {
    using bf16 = cutlass::bfloat16_t;

    // ====== 步骤 1：提取维度信息 ======
    Arch arch = Arch();

    // 检查 tensor 的维度数量是否正确（比如 q 必须是 4 维）
    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    // 从 tensor 的形状中读出各种维度值
    int b = q.size(0);    // batch_size：这次同时处理几个序列？
    int s_q = q.size(1);  // seq_len_q：每个序列的 Q 长度（解码时通常是 1）
    int h_q = q.size(2);  // num_heads_q：Q 有几个"头"？
    int d_qk = q.size(3); // head_dim：每个 head 的向量维度（576 或 512）
    int num_blocks = kv.size(0);     // KV Cache 有多少个 block？
    int page_block_size = kv.size(1);// 每个 block 存几个 token？
    int h_kv = kv.size(2);           // K 有几个 head？（稀疏模式下必须是 1，即 MQA）
    int topk = indices.size(2);      // 每个 query 关注几个 token？

    // 检查可选参数是否传了
    bool have_topk_length = topk_length.has_value();
    bool have_extra_kcache = extra_kv.has_value();
    bool have_extra_topk_length = extra_topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    // 如果有额外 KV Cache，读出它的维度
    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (have_extra_kcache) {
        extra_num_blocks = extra_kv->size(0);
        extra_page_block_size = extra_kv->size(1);
    }
    if (extra_indices.has_value()) {
        extra_topk = extra_indices->size(-1);
    }

    // ====== 步骤 2：参数合法性检查 ======
    // 这些 TORCH_CHECK 就像安检——东西不合规就拦住报错。
    // 在 C++ 层拦截，比跑到 GPU kernel 里崩溃好排查得多。

    // 基本维度不能为零
    TORCH_CHECK(b > 0);
    TORCH_CHECK(s_q > 0);
    TORCH_CHECK(h_q > 0);
    // 稀疏解码目前只支持 MQA（Multi-Query Attention），即 1 个 KV head
    TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
    // head 维度只能为 576 或 512
    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Only head_size_k == 576 or 512 is supported for sparse decoding");
    TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
    TORCH_CHECK(topk > 0);

    // 额外 KV Cache 和额外索引必须同时传或同时不传
    if (have_extra_kcache) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
    }

    // 检查所有 tensor 是否都在同一个 GPU 上
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);

    // 检查数据类型是否正确
    KU_CHECK_DTYPE(q, torch::kBFloat16);  // Q 必须是 bfloat16
    // KV Cache 必须是 FP8 格式（float8_e4m3fn、int8 或 uint8）
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn, int8 or uint8");
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 || extra_kv->dtype() == torch::kUInt8, "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);     // 索引是 int32
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32); // Attention Sink 是 float32
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);

    // 检查内存布局（是否连续）
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);

    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);

    // 检查形状是否完全匹配
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    {
        // 计算 KV Cache 中每个 token 占多少字节
        int bytes_per_token;
        if (d_qk == 576 && d_v == 512) {
            // V3.2 格式：512 字节 FP8 NoPE + 4 个 scale (16 字节) + 64 个 bf16 RoPE (128 字节)
            bytes_per_token = 512 + 64*2 + (512/128)*4;
        } else if (d_qk == 512 && d_v == 512) {
            // MODEL1 格式
            bytes_per_token = 448 + 64*2 + (448/64)*1 + 1;
        } else {
            TORCH_CHECK(false, "Unsupported head sizes for is_fp8_kvcache == True");
        }
        KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token);
        KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token);
        // 整个 block 必须是连续的内存
        TORCH_CHECK(kv.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for kv cache");
        if (extra_kv.has_value()) {
            TORCH_CHECK(extra_kv->stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
        }
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);

    // ====== 步骤 3：分配输出 tensor ======
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = torch::empty({b, s_q, h_q, d_v}, opts); // 注意力输出 [b, s_q, h_q, d_v]
    at::Tensor lse = torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat)); // log-sum-exp [b, s_q, h_q]

    // 确定模型类型（不同模型使用不同的 KV Cache 格式）
    ModelType model_type;
    if (d_qk == 576) {
        model_type = ModelType::V32;       // DeepSeek V3.2
    } else if (d_qk == 512) {
        model_type = ModelType::MODEL1;     // 另一种模型
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }

    // ====== 步骤 4：构建"功能需求列表" ======
    // 把用户的需求（head 数、维度、格式等）整理成 features 列表。
    // 后面实现类的 run() 函数会用这个列表和自己声明的 DECLARE_SUPPORTED_FEATURES
    // 对比，确保选到的实现确实支持所有需要的功能。
    std::vector<DecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(DecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(DecodeFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        features.push_back(DecodeFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        features.push_back(DecodeFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (model_type == ModelType::V32) {
        features.push_back(DecodeFeatures::V32_KVCACHE_FORMAT);
    } else if (model_type == ModelType::MODEL1) {
        features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
    } else {
        TORCH_CHECK(false, "Unsupported model type: ", (int)model_type);
    }
    if (have_attn_sink) {
        features.push_back(DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
    }

    // ====== 步骤 5：选择具体实现类 ======
    // 根据 GPU 型号和 head 数，从 4 个实现类中选一个。
    // 选择逻辑：
    //
    //               ┌── h_q=64 ──→ Decode_Sm100_Head64_Impl
    //               │
    //  GPU是SM100? ─┤              ┌── d_qk=576 ──→ Head64x2（跑两次 64）
    //               └── h_q=128 ──┤
    //                              └── d_qk=512 ──→ Head128（专用 kernel）
    //
    //  GPU是SM90? ──→ Decode_Sm90_Impl（统一的 kernel，支持所有配置）
    //
    //  都不匹配 ──→ 报错"不支持的架构"
    DecodeImplBase* impl;
    if (arch.is_sm100f()) {
        if (h_q == 64) {
            impl = new Decode_Sm100_Head64_Impl();
        } else if (h_q == 128) {
            if (d_qk == 576) {
                impl = new Decode_Sm100_Head64x2_Impl();
            } else if (d_qk == 512) {
                impl = new Decode_Sm100_Head128_Impl();
            } else {
                TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
            }
        } else {
            TORCH_CHECK(false, "Unsupported h_q: ", h_q);
        }
    } else if (arch.is_sm90a()) {
        impl = new Decode_Sm90_Impl();
    } else {
        TORCH_CHECK(false, "Unsupported architecture for sparse decode fwd");
    }

    // ====== 步骤 6：获取调度参数 + 组装参数结构体 ======

    // 从选好的实现类中获取调度参数（分组数、开销等）
    DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

    // 把所有参数打包成一个结构体，传给 kernel
    // data_ptr() 获取 tensor 在 GPU 上的内存地址 — 就像拿到档案柜的钥匙
    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,  // softmax 缩放系数
        num_blocks, page_block_size, topk,
        model_type,

        (bf16*)q.data_ptr(),            // Q 的 GPU 地址
        (bf16*)kv.data_ptr(),           // KV Cache 的 GPU 地址
        (int*)indices.data_ptr(),       // 索引的 GPU 地址
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),         // 输出 lse 的地址
        (bf16*)out.data_ptr(),          // 输出 out 的地址

        extra_num_blocks, extra_page_block_size, extra_topk,
        ku::get_optional_tensor_ptr<bf16>(extra_kv),
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        // stride（步长）信息——告诉 kernel tensor 在内存中如何排列
        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        // 额外 KV 的 stride（如果有的话）
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        at::cuda::getCurrentCUDAStream().stream()  // 当前 CUDA 流
    };

    // ====== 步骤 7：调度元数据的懒初始化 ======
    // 和 Python 层的 FlashMLASchedMeta 懒初始化对应——
    // 第一次调用时生成调度方案，后续调用复用。
    at::Tensor o_accum, lse_accum;
    if (!tile_scheduler_metadata.has_value()) {
        // 第一次调用：分配调度元数据 tensor
        tile_scheduler_metadata = torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, opts.dtype(torch::kInt32));
        num_splits = torch::empty({b+1}, opts.dtype(torch::kInt32));
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);

        // 调用专门的 CUDA kernel 来生成调度方案
        GetDecodeSchedMetaParams get_sched_meta_params = {
            b, s_q,
            impl_meta.block_size_topk,
            impl_meta.fixed_overhead_num_blocks,
            topk,
            extra_topk,
            ku::get_optional_tensor_ptr<int>(topk_length),
            ku::get_optional_tensor_ptr<int>(extra_topk_length),
            nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->data_ptr<int>(),
            impl_meta.num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
    }
    // 把调度元数据的指针塞进 params，kernel 运行时通过 params 找到它
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/sizeof(int));
    KU_CHECK_SHAPE(num_splits, b+1);
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.num_sm_parts = impl_meta.num_sm_parts;

    // ====== 步骤 8：分配 Split-KV 中间缓冲区 ======
    // Split-KV 是什么？当序列很长时，把 KV 分成多块，让不同 SM 各自算一块，
    // 最后合并结果。lse_accum 和 o_accum 就是每个 SM 的"中间计算结果"。
    // 就像多人一起做一张试卷——每人负责一页，写在自己的草稿纸上，最后汇总。
    const int total_num_splits = b + impl_meta.num_sm_parts;
    lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
    o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
    KU_CHECK_CONTIGUOUS(lse_accum);
    KU_CHECK_CONTIGUOUS(o_accum);
    params.lse_accum = lse_accum.data_ptr<float>();
    params.o_accum = o_accum.data_ptr<float>();
    params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
    params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
    params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
    params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
    params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));

    // ====== 步骤 9：执行 kernel → 合并结果 → 返回 ======

    // 9a：启动 CUDA kernel，每个 SM 算自己的部分
    impl->run(params, features);

    // 9b：把各 SM 的中间结果合并成最终结果
    // combine kernel 通过"加权平均"把 lse_accum 和 o_accum 合并成最终的 out 和 lse
    CombineParams combine_params = {
        b, s_q, h_q, d_v,

        params.lse,          // 最终 lse 输出的地址
        params.out,          // 最终 out 输出的地址
        params.stride_lse_b, params.stride_lse_s_q,
        params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

        params.lse_accum,    // 各 SM 的中间 lse
        params.o_accum,      // 各 SM 的中间 out
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,  // 调度元数据（告诉 combine 各 SM 怎么分的）
        params.num_splits_ptr,
        params.num_sm_parts,

        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    // 清理实现类
    delete impl;

    // 返回结果给 Python 层
    // lse.transpose(1,2) 把 lse 的维度从 [b, s_q, h_q] 转成 [b, h_q, s_q]
    // 匹配 Python 层期望的格式
    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}
