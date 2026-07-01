# FlashMLA Python 接口全面解析

> 本文逐函数分析 `flash_mla/flash_mla_interface.py`，讲清楚每个函数的作用、为什么需要它、解决了什么问题，以及对应的 C++ 函数做了什么。

---

## 一、全景：这个文件的位置

```
用户代码 (Python)
    ↓
flash_mla/__init__.py          ← 导出名字
    ↓
flash_mla/flash_mla_interface.py  ← ← 我们在这（Python 接口层）
    ↓  PYBIND11
flash_mla.cuda (cuda.so)       ← C++ 编译产物
    ↓
csrc/api/api.cpp               ← C++ API 入口
    ↓
csrc/api/sparse_fwd.h 等       ← C++ 调度层
    ↓
csrc/sm90/.../*.cuh            ← CUDA Kernel（真正在 GPU 上跑的代码）
```

**这个文件的核心角色：** 它是一道"翻译门面"。用户看到的全是友好的 Python 函数，内部在合适时机调用 C++/CUDA 完成真正的 GPU 计算。

---

## 二、数据类：FlashMLASchedMeta

```python
@dataclasses.dataclass
class FlashMLASchedMeta:
    class Config:
        b: int                    # batch size
        s_q: int                  # query 序列长度
        h_q: int                  # query head 数量
        page_block_size: int      # page 大小（每个 block 包含多少个 token）
        h_k: int                  # key/value head 数量
        causal: bool              # 是否 causal mask
        is_fp8_kvcache: bool      # KV cache 是否为 FP8 格式
        topk: Optional[int]       # 稀疏 attention 的 topk
        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False
    config: Optional[Config] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None  # 调度元数据
    num_splits: Optional[torch.Tensor] = None                # split 数量
```

### 为什么需要它？

这是 FlashMLA 解码阶段的**"调度管家"**。

解码阶段要在 GPU 上反复执行 `flash_mla_with_kvcache`（每生成一个 token 调用一次）。每次调用都需要知道"当前 batch 的 KV cache 怎么分块、每个 block 由哪个 SM 处理"。这些信息在第一次调用时确定，后续复用。

FlashMLASchedMeta 就是用来**缓存这些调度信息**的对象——第一次调用时自动生成，后续调用直接复用，避免重复计算。

---

## 三、`get_mla_metadata()`

```python
def get_mla_metadata(*args, **kwargs) -> Tuple[FlashMLASchedMeta, None]:
    return FlashMLASchedMeta(), None
```

### 作用

创建并返回一个**空的** `FlashMLASchedMeta` 对象。

### 为什么需要这个函数？

**历史兼容性。** 老版本的 FlashMLA 需要用户手动提供一些元数据才能开始解码循环。新版本把调度元数据的生成逻辑移到了 `flash_mla_with_kvcache` 内部（懒初始化），但为了不破坏老代码，保留了 `get_mla_metadata()` 这个接口。

现在它做的事情就是把一个空壳 `FlashMLASchedMeta` 对象返回给你，真正的初始化发生在你第一次调用 `flash_mla_with_kvcache` 时。

### 典型用法

```python
# 解码循环之前创建
tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

# 然后在解码循环中反复使用
for i in range(num_layers):
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, ...
    )
```

### 返回值

返回 `(FlashMLASchedMeta(), None)`——一个元组，第一个元素是调度管家，第二个元素是历史遗留的 `None`。

---

## 四、`flash_mla_with_kvcache()`（核心函数）

```python
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
```

### 它是做什么的？

这是 FlashMLA **最核心的函数**——在**解码阶段**（每次生成一个 token）执行 MLA 注意力计算，返回注意力结果 `out` 和 log-sum-exp `lse`。

### 为什么需要这个函数？

大模型推理时，解码阶段是一个 token 一个 token 生成的。每次生成新 token 时，需要：

1. 用当前 token 的 Query 去和已经缓存的所有 Key/Value 做注意力计算
2. 得到注意力输出，用于预测下一个 token

这个函数实现的就是**第 1 步**——给定一个 Query 和已有的 KV Cache，计算出注意力结果。

### 两种模式

这个函数内部有两种模式，由 `indices` 参数决定：

#### 模式 A：密集注意力（Dense）

```python
# indices 为 None → 密集注意力
block_table ≠ None, cache_seqlens ≠ None
```

密集模式下，Query 和 KV Cache 中**所有** token 计算注意力。这是标准的 Attention 模式。

#### 模式 B：稀疏注意力（Sparse）

```python
# indices 不为 None → 稀疏注意力
indices ≠ None, is_fp8_kvcache = True
```

稀疏模式下，Query 只和 `indices` 中指定的 token 子集计算注意力。这是 DeepSeek V3.2 的 DSA（DeepSeek Sparse Attention）机制，用于大幅降低长上下文下的计算量。

### 关键代码解析

#### 1. 参数提取

```python
topk = indices_in_kvcache.shape[-1] if indices_in_kvcache is not None else None
```

如果 `indices` 存在，获取其最后一维大小（即每个 query 关注的 token 数量）。

#### 2. 第一次调用时初始化调度元数据

```python
if not sched_meta.have_initialized:
    # 参数校验
    if indices_in_kvcache is not None:
        assert not causal, "causal must be False when sparse attention"
    
    # 记录本次调用的所有配置
    sched_meta.have_initialized = True
    sched_meta.config = FlashMLASchedMeta.Config(
        q.shape[0], q.shape[1], q.shape[2],
        k_cache.shape[1], k_cache.shape[2],
        causal, is_fp8_kvcache, topk,
        extra_k_page_block_size, extra_topk,
    )
else:
    # 后续调用校验参数是否一致
    assert sched_meta.config.b == q.shape[0], "..."
    # ... 更多一致性检查
```

**为什么第一次调用时做初始化？** 因为此时才知道 tensor 的实际 shape。后续调用如果 shape 变了会报错——这就是 `FlashMLASchedMeta` 的一致性检查。

#### 3. 路由到不同的 C++ 函数

```python
if topk is not None:
    # 稀疏注意力 → 调用 C++ 的 sparse_decode_fwd
    out, lse, new_tile_scheduler_metadata, new_num_splits = \
        flash_mla_cuda.sparse_decode_fwd(q, k_cache, indices_in_kvcache, ...)
else:
    # 密集注意力 → 调用 C++ 的 dense_decode_fwd
    out, lse, new_tile_scheduler_metadata, new_num_splits = \
        flash_mla_cuda.dense_decode_fwd(q, k_cache, head_dim_v, cache_seqlens, ...)
```

这是一个**路由**——Python 层根据参数决定调用哪个 C++ kernel，然后更新调度元数据。

#### 4. 更新调度元数据

```python
sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
sched_meta.num_splits = new_num_splits
```

C++ 函数内部可能会更新调度元数据（比如 Split-KV 的 split 数量），这些更新被写回到 Python 对象中供后续调用复用。

### 对应的 C++ 函数

#### 稀疏分支：`sparse_decode_fwd` → C++ `sparse_attn_decode_interface`

定义在 `csrc/api/sparse_decode.h`：

```cpp
static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor &q,        // [b, s_q, h_q, d_qk]
    const at::Tensor &kv,       // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor &indices,  // [b, s_q, topk]
    ...
)
```

**这个 C++ 函数做了什么：**

1. **参数校验**：检查 tensor shape、dtype、device 是否正确
2. **选择实现**：根据 GPU 型号（SM90/SM100）和参数（head 数、head dim），选择最合适的 CUDA kernel
3. **准备调度**：如果 `tile_scheduler_metadata` 还没分配，调用 `run_get_decoding_sched_meta_kernel` 计算调度信息
4. **分配中间缓存**：为 Split-KV 分配 `lse_accum` 和 `o_accum` 缓冲区
5. **执行 Split-KV**：调用选中的实现类的 `run()` 方法，启动 CUDA kernel 在 GPU 上计算
6. **合并结果**：调用 `run_flash_mla_combine_kernel` 把 Split-KV 的多个分段结果合并成最终输出

流程图：

```
Python 调用 flash_mla_cuda.sparse_decode_fwd(...)
  ↓
C++ sparse_attn_decode_interface:
  ├─ 校验 shape/dtype/device
  ├─ 校验架构（SM90 or SM100）
  ├─ 选择实现类 (Decode_Sm90_Impl / Decode_Sm100_Head64_Impl / ...)
  ├─ 分配/复用调度元数据
  ├─ 启动 get_decoding_sched_meta kernel (CPU 端调度规划)
  ├─ 分配 SplitKV 中间缓冲区
  ├─ 启动 splitkv MLA kernel (GPU 上真正计算注意力)    ← 主要计算
  └─ 启动 combine kernel (GPU 上合并所有 split 结果)    ← 结果合并
  ↓
返回 (out, lse, tile_scheduler_metadata, num_splits) 给 Python
```

#### 密集分支：`dense_decode_fwd` → C++ `dense_attn_decode_interface`

定义在 `csrc/api/dense_decode.h`：

```cpp
static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
dense_attn_decode_interface(
    at::Tensor &q,
    const at::Tensor &kcache,
    const int head_size_v,
    const at::Tensor &seqlens_k,    // 每个序列的实际长度
    const at::Tensor &block_table,  // 页面映射表
    const float softmax_scale,
    bool is_causal,
    ...
)
```

**和稀疏版本的关键不同：**

- 需要 `seqlens_k`（每个序列在 KV Cache 中的实际长度，因为不同序列长度不同）
- 需要 `block_table`（逻辑页面到物理页面的映射，因为 KV Cache 使用分页管理）
- 支持 causal mask（因果遮罩）
- 目前只支持 SM90 架构（H100/H800）

---

## 五、`flash_mla_sparse_fwd()`

```python
def flash_mla_sparse_fwd(
    q: torch.Tensor,       # [s_q, h_q, d_qk], bfloat16
    kv: torch.Tensor,      # [s_kv, h_kv, d_qk], bfloat16
    indices: torch.Tensor, # [s_q, h_kv, topk], int32
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

### 它是做什么的？

这是**预填充阶段**（prefill）的稀疏注意力函数。和 `flash_mla_with_kvcache`（解码）不同，预填充是一次性处理整个 prompt 序列的所有 token。

### 预填充 vs 解码

| 维度 | 预填充 (Prefill) | 解码 (Decoding) |
|------|---------|----------|
| 处理方式 | 一次处理整个 prompt | 一次一个 token |
| 计算特点 | 计算密集型（Q 和 K 都是长序列） | 访存密集型（Q 短，K 长） |
| API 函数 | `flash_mla_sparse_fwd` | `flash_mla_with_kvcache` |
| 输入格式 | Q/KV 全量传入 | Q 是当前 token，K 通过 cache 传入 |
| batch 维度 | **不支持** batch（需手动 reshape） | 支持 batch |

### 为什么需要这个函数？

大模型推理分为两个阶段：
1. **Prefill（预填充）：** 用户输入 prompt，模型一次性处理整个 prompt，生成第一个 token。这个阶段需要计算 prompt 中所有 token 之间的注意力。
2. **Decoding（解码）：** 基于已生成的 token，逐个生成后续 token。

`flash_mla_sparse_fwd` 就是负责**第一阶段**的稀疏注意力计算。稀疏意味着只计算选中 token 子集之间的注意力，而不是全量的。

### 返回值

```python
return (output, max_logits, lse)
```

- `output`: `[s_q, h_q, d_v]`，注意力计算结果
- `max_logits`: `[s_q, h_q]`，每个 query 的 attention score 最大值（用于数值稳定）
- `lse`: `[s_q, h_q]`，log-sum-exp（用于后续的 Split-KV 合并）

### 对应的 C++ 函数

```python
# Python
results = flash_mla_cuda.sparse_prefill_fwd(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
```

→ `csrc/api/sparse_fwd.h` 中的 `sparse_attn_prefill_interface`

**C++ 函数做了什么：**

```cpp
static std::vector<at::Tensor> sparse_attn_prefill_interface(...) {
    // 1. 获取当前 GPU 架构
    Arch arch = Arch();
    
    // 2. 参数校验（shape、dtype、device）
    KU_CHECK_NDIM(q, 3);
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    // ...
    
    // 3. 打包参数到 SparseAttnFwdParams 结构体
    SparseAttnFwdParams params = { s_q, s_kv, h_q, h_kv, d_qk, d_v, topk, ... };
    
    // 4. 收集需要的 features
    std::vector<FwdFeatures> required_features;
    if (h_q == 64) required_features.push_back(FwdFeatures::HEAD_64);
    if (d_qk == 576) required_features.push_back(FwdFeatures::HEAD_DIM_576);
    // ...
    
    // 5. 根据架构选择实现
    if (is_sm90a) {        // H100/H800
        Fwd_Sm90_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else if (is_sm100f) {  // B200
        if (h_q == 64) {
            Fwd_Sm100_Head64_Impl ...
        } else if (h_q == 128) {
            // 根据 topk 大小选择优化版本
            if (topk <= 1280) 使用 Fwd_Sm100_Head128_Small_TopK_Impl
            else              使用 Fwd_Sm100_Head128_Impl
        }
    }
    
    return {out, max_logits, lse};
}
```

**最关键的设计：基于 Feature 的调度器**

每个实现类声明自己支持的 feature 组合：
```cpp
class Fwd_Sm90_Impl : public FwdImplBase {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_64,
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        FwdFeatures::TOPK_LENGTH
    )
};
```

运行时自动检查：当前实现是否支持用户请求的所有 feature。如果不支持就报错，给出清晰的错误信息。

---

## 六、SM100 Dense MHA 相关函数

以下四个函数是一个整体，用于在 **SM100 (B200) 上执行标准的密集 Multi-Head Attention**（注意：不是 MQA，是真的 MHA，支持多个 KV head）。

### 6.1 `_flash_attn_varlen_forward`

```python
def _flash_attn_varlen_forward(
    q, k, v, cu_seqlens_qo, cu_seqlens_kv,
    max_seqlen_qo, max_seqlen_kv,
    out, lse,
    causal=False, softmax_scale=None, is_varlen=True,
) -> Tuple[Tensor, Tensor]:
```

**作用：** SM100 上的密集 MHA 前向计算，支持变长序列（varlen）。

**为什么是 varlen？** 在大模型推理中，一个 batch 里的多个序列长度可能不同。varlen 格式通过 `cu_seqlens`（累积序列长度数组）来表示每个序列的起始和结束位置，避免了 padding 浪费。

**C++ 调用：**
```python
flash_mla_cuda.dense_prefill_fwd(workspace_buffer, q, k, v, cu_seqlens_qo, ...)
```

→ `csrc/sm100/prefill/dense/` 下的 CUTLASS 实现（基于 NVIDIA CUTLASS 库的 MHA kernel）。

### 6.2 `_flash_attn_varlen_backward`

```python
def _flash_attn_varlen_backward(
    do, q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv,
    max_seqlen_qo, max_seqlen_kv,
    dq, dk, dv,
    causal=False, softmax_scale=None, is_varlen=True,
) -> Tuple[Tensor, Tensor, Tensor]:
```

**作用：** SM100 上密集 MHA 的反向传播。计算 loss 对 Q/K/V 的梯度。

**为什么需要手动计算 workspace？**
```python
workspace_bytes  = 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc
workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2           # sum_OdO and scaled_lse
```

反向传播需要额外的中间缓冲区来累加梯度。FlashMLA 根据序列长度和 head 数量精确计算需要的 workspace 大小，而不是每次都分配固定大小的缓冲区。

**C++ 调用：**
```python
flash_mla_cuda.dense_prefill_bwd(workspace_buffer, do, q, k, v, out, lse, ...)
```

→ `csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu`（CUTLASS 反向 MHA kernel）。

### 6.3 `FlashAttnVarlenFunc`

```python
class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(ctx, q, k, v, cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv, ...):
        # 调用 _flash_attn_varlen_forward
        out, lse = _flash_attn_varlen_forward(...)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        return out, lse

    def backward(ctx, do, dlse):
        # 调用 _flash_attn_varlen_backward
        dq, dk, dv = _flash_attn_varlen_backward(...)
        return dq, dk, dv, None, None, ...
```

**为什么需要它？** PyTorch 的 `torch.autograd.Function` 允许你自定义操作的前向和反向传播。FlashMLA 的 SM100 dense prefill 需要支持训练（不仅仅是推理），所以要实现完整的 autograd 函数。

`ctx.save_for_backward` 保存了前向计算的中间结果（Q、K、V、out、lse），这些在反向传播中需要用到。

### 6.4 三个公开 API

```python
def flash_attn_varlen_func(q, k, v, ...):
    return FlashAttnVarlenFunc.apply(q, k, v, ...)

def flash_attn_varlen_qkvpacked_func(qkv, ...):   # QKV 打包在一起的快捷方式
    head_dim_qk = ...
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk],        # 切片出 Q
        qkv[:, :, head_dim_qk:2*hd],    # 切片出 K
        qkv[:, :, 2*hd:],               # 切片出 V
        ...
    )

def flash_attn_varlen_kvpacked_func(q, kv, ...):   # KV 打包在一起的快捷方式
    return FlashAttnVarlenFunc.apply(
        q,
        kv[:, :, :head_dim_qk],         # 切片出 K
        kv[:, :, head_dim_qk:],         # 切片出 V
        ...
    )
```

**为什么有三种？** 兼容性。`flash_attn` 库（Dao-AILab 的 FlashAttention）提供了类似的 API。FlashMLA 提供的这三个函数签名和 `flash_attn` 一致，方便用户从 `flash_attn` 迁移过来时只需要改 import 路径，不需要改调用代码。

**`flash_attn_varlen_qkvpacked_func` 适用场景：** Q、K、V 已经拼成一个大的 tensor（常见于 Transformer 的 self-attention 层）。

**`flash_attn_varlen_kvpacked_func` 适用场景：** K 和 V 拼在一起（常见于 cross-attention 或 decoder 的 KV cache）。

---

## 七、函数对比总结

| 函数 | 阶段 | 注意力类型 | 架构 | 支持训练？ | C++ 入口 |
|------|------|-----------|------|:-------:|---------|
| `flash_mla_with_kvcache` (dense) | Decoding | Dense MLA (MQA) | SM90 | ❌ | `dense_attn_decode_interface` |
| `flash_mla_with_kvcache` (sparse) | Decoding | Sparse MLA (MQA) | SM90+SM100 | ❌ | `sparse_attn_decode_interface` |
| `flash_mla_sparse_fwd` | Prefill | Sparse MLA (MQA) | SM90+SM100 | ❌ | `sparse_attn_prefill_interface` |
| `flash_attn_varlen_func` | Prefill | Dense MHA | SM100 | ✅ | `dense_prefill_fwd/bwd` |

---

## 八、学习建议

要理解这个文件，建议按这个顺序走读：

1. **先看 `flash_mla_sparse_fwd`** — 最简单、最干净。没有 split-kv、没有调度元数据、没有初始化逻辑。输入 → C++ 调用 → 输出。
2. **再看 `flash_mla_with_kvcache`** — 复杂得多。理解 FlashMLASchedMeta 的设计意图（懒初始化 + 一致性检查），理解 dense/sparse 两条分支。
3. **看对应的 C++ 函数** — `sparse_attn_prefill_interface` 和 `sparse_attn_decode_interface`。理解参数校验、架构调度、SplitKV 的三段式（调度 → 计算 → 合并）。
4. **最后看 SM100 dense MHA 部分** — 这是和前面完全不同的东西（标准的 MHA 前向+反向，基于 CUTLASS）。理解 autograd.Function 的 forward/backward。
