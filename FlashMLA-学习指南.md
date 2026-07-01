# FlashMLA 学习指南

> 用问题驱动的方式，从外到内彻底掌握这个项目。

---

## 一、项目全景概览

在学习具体代码之前，先建立宏观认知。FlashMLA 是 DeepSeek 为自家大模型（DeepSeek-V3 / V3.2）开发的 **MLA（多头潜在注意力）专用 CUDA 加速库**。它不是通用的 Attention 库（如 FlashAttention），而是深度定制、极致优化的专属内核。

---

## 二、分层学习路径

建议按**从高层到底层**的顺序递进学习，每层都带着核心问题去理解。

---

### 第一层：问题域（为什么存在）

> 理解 FlashMLA 要解决的根本问题。

| 问题 | 说明 |
|------|------|
| **Q1: MLA 和标准 MHA 有什么不同？** | 标准 Multi-Head Attention 的 K/V 是没有压缩的；MLA 通过低秩投影将 KV 压缩到低维 latent space。为什么这样做？好处是什么？（KV Cache 大幅减小、推理成本降低） |
| **Q2: 为什么 DeepSeek 不能直接用 FlashAttention？** | FlashAttention 是为标准 MHA 设计的。MLA 使用了非标准的 MQA 模式（`head_dim_k=576`, `head_dim_v=512`），标准的 FlashAttention 无法支持这种形状。 |
| **Q3: "Dense" 和 "Sparse" 分别指什么？** | Dense = 所有 token 都参与注意力计算（标准做法）。Sparse = 只计算选定的 token 子集，在 DeepSeek V3.2 中由 DSA（DeepSeek Sparse Attention）机制筛选。为什么需要稀疏？——长上下文下的计算量控制。 |
| **Q4: FP8 KV Cache 解决了什么问题？** | 显存带宽是 Decoding 阶段的主要瓶颈。将 KV Cache 量化为 FP8 存储（配合 per-block 的 scale factor）、反量化回 BF16 计算，在不明显损失精度的情况下降低显存和带宽消耗。 |

**需要读的文件：**
- `README.md` — 全局背景
- `flash_mla/flash_mla_interface.py` — Python API 接口定义和参数文档
- `docs/20250422-new-kernel-deep-dive.md` — 内核深度解读
- `docs/20250929-hopper-fp8-sparse-deep-dive.md` — FP8 稀疏解码深度解读

---

### 第二层：Python API 层（用户视角）

> 理解用户是怎么调用 FlashMLA 的，API 的设计意图是什么。

| 问题 | 说明 |
|------|------|
| **Q5: 为什么 Decoding 阶段需要先调用 `get_mla_metadata()`？** | 这不是简单的初始化——它返回一个 `FlashMLASchedMeta` 对象，用于 tile scheduler。注意：`get_mla_metadata()` 现在返回空对象，元数据在第一次调用 `flash_mla_with_kvcache` 时懒初始化。为什么需要这种设计？（因为实际调度元数据依赖于第一次调用时传入的具体 tensor shape） |
| **Q6: `flash_mla_with_kvcache` 的参数是如何设计的？** | 它同时支持 dense 和 sparse 两种模式。当 `indices` 不为 None 时走 sparse 路径。仔细看参数校验逻辑：有哪些参数是互斥的？为什么 block_table 在 sparse 模式下可以为 None？ |
| **Q7: Prefill 和 Decoding 的 API 为什么不同？** | Prefill 使用 `flash_mla_sparse_fwd`，Decoding 使用 `flash_mla_with_kvcache`。两者在计算模式上有什么本质区别？（Prefill 一次处理整个 prompt 序列，Decoding 每次处理一个 token。） |
| **Q8: SM100 Dense Prefill 为什么提供了 `flash_attn_varlen_func` 等兼容接口？** | 这些接口试图兼容 `flash_attn` 库的函数签名，降低了迁移成本。但底层内核完全不同。为什么需要 `torch.autograd.Function`？因为要实现反向传播。 |

**核心文件：**
- `flash_mla/__init__.py` — 导出的公共 API
- `flash_mla/flash_mla_interface.py` — Python 接口实现（重点）
- `tests/test_flash_mla_dense_decoding.py` — 密集解码测试
- `tests/test_flash_mla_sparse_decoding.py` — 稀疏解码测试
- `tests/test_flash_mla_sparse_prefill.py` — 稀疏预填充测试
- `tests/test_fmha_sm100.py` — SM100 MHA 测试
- `tests/ref.py` — PyTorch 参考实现（**强烈推荐**，用于理解正确的数学语义）
- `tests/quant.py` — FP8 量化/反量化细节

---

### 第三层：C++/CUDA API 层（桥接层）

> 理解 Python 和 CUDA 内核之间是如何连接的。

| 问题 | 说明 |
|------|------|
| **Q9: API 层是如何将 Python 调用转发到 CUDA 内核的？** | 看 `csrc/api/api.cpp` 和对应的头文件 `sparse_decode.h`, `dense_decode.h`, `sparse_fwd.h`, `dense_fwd.h`。每个 API 函数如何解析 PyTorch tensor，调用对应的 CUDA kernel launch。 |
| **Q10: 传统 CUDA extension 的编译流程是怎样的？** | 看 `setup.py`。它使用了 `torch.utils.cpp_extension.CUDAExtension`。注意编译选项：`--use_fast_math`、`--expt-relaxed-constexpr`、`-lineinfo` 等。NVCC 版本检测和 SM 架构控制。 |
| **Q11: `csrc/utils.h` 和 `csrc/params.h` 是什么作用？** | 看这些公共头文件定义了哪些通用工具函数和参数结构体。 |

**核心文件：**
- `csrc/api/api.cpp` — 主入口，所有 CUDA kernel 的 launcher
- `csrc/api/common.h` — API 公共头文件
- `csrc/api/dense_decode.h` / `sparse_decode.h` / `dense_fwd.h` / `sparse_fwd.h`
- `setup.py` — 编译配置
- `csrc/defines.h` / `params.h` / `utils.h` — 公共定义

---

### 第四层：通用工具层（跨架构基础设施）

> 理解项目中自建的 kernel 工具库。

| 问题 | 说明 |
|------|------|
| **Q12: `kerutils` 这个子项目是什么？** | 它包含了 SM80/SM90/SM100 架构共用的 CUDA 工具函数。看 `device/sm90/helpers.cuh` 和 `device/sm100/helpers.cuh`：有哪些架构相关的差异？TMA（Tensor Memory Accelerator）相关代码在哪里？ |
| **Q13: `csrc/smxx/` 目录中的 "xx" 是什么意思？** | "smxx" 表示架构无关或跨架构共享的代码。例如 `decode/get_decoding_sched_meta/` 和 `decode/combine/` 在不同架构之间共享。 |

**核心文件：**
- `csrc/kerutils/include/kerutils/` — 工具库根目录
- `csrc/kerutils/include/kerutils/device/sm90/helpers.cuh`
- `csrc/kerutils/include/kerutils/device/sm100/helpers.cuh`
- `csrc/kerutils/include/kerutils/device/device.cuh`
- `csrc/smxx/decode/combine/` — Combine kernel（解码结果合并）
- `csrc/smxx/decode/get_decoding_sched_meta/` — 调度元数据生成

---

### 第五层：SM90（H100/H800）内核层

> 理解 Hopper 架构上的密集解码和稀疏解码/预填充实现。

| 问题 | 说明 |
|------|------|
| **Q14: Split-KV 解码内核的结构是什么？** | 看 `csrc/sm90/decode/dense/splitkv_mla.cuh` 和 `splitkv_mla.h`。Split-KV 是一种将 KV 序列分段并行计算的策略。为什么需要 Split-KV？（当 KV 序列很长时，单个 thread block 无法一次处理完，需要分段计算后合并。） |
| **Q15: FP8 稀疏解码内核 (`sparse_fp8`) 相比密集解码多了什么？** | 看 `sparse_fp8/splitkv_mla.cuh`。FP8 反量化逻辑在哪？Sparse indexing 如何映射到显存地址？`components/dequant.h` 中的反量化实现细节。 |
| **Q16: 稀疏预填充内核 (`sm90/prefill/sparse/`) 的 Phase1 是什么？** | 为什么叫 Phase1？和 Phase2 是什么关系？看 `phase1.cuh` 和 `config.h`。Phase1 是稀疏前向的第一步——在 GPU 上做稀疏 attention 的在线 softmax。 |
| **Q17: Instantiations 目录中的各种 .cu 文件是做什么用的？** | 每个 `instantiations/xxx.cu` 是一个编译单元，用于显式模板实例化，避免头文件中的所有模板代码被重复编译。为什么需要按 `head_dim`（k512 / k576）和 `topk` 粒度分开实例化？ |

**核心文件：**
- `csrc/sm90/decode/dense/splitkv_mla.cuh` — 密集解码 SplitKV 主逻辑
- `csrc/sm90/decode/dense/config.h` — 密集解码配置
- `csrc/sm90/decode/dense/traits.h` — 数据类型 traits
- `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` — 稀疏解码 SplitKV 主逻辑
- `csrc/sm90/decode/sparse_fp8/splitkv_mla.h`
- `csrc/sm90/decode/sparse_fp8/config.h`
- `csrc/sm90/decode/sparse_fp8/components/` — 子组件（dequant 等）
- `csrc/sm90/prefill/sparse/fwd.cu` / `fwd.h` — 稀疏预填充入口
- `csrc/sm90/prefill/sparse/phase1.cuh` — Phase1 实现
- `csrc/sm90/prefill/sparse/config.h`
- `csrc/sm90/helpers.h` — SM90 架构辅助函数

---

### 第六层：SM100（B200）内核层

> 理解 Blackwell 架构上的优化（支持 MHA Dense Prefill 的前向+反向）。

| 问题 | 说明 |
|------|------|
| **Q18: SM100 和 SM90 的内核在架构利用上有何不同？** | SM100（Blackwell）相比 SM90（Hopper）新增了哪些硬件特性？内核如何利用 TMA、第四代 Tensor Core 等新特性？看 `csrc/sm100/helpers.h`。 |
| **Q19: Dense MHA Prefill 的前向和反向实现是如何组织的？** | 看 `fmha_cutlass_fwd_sm100.cuh` 和 `fmha_cutlass_bwd_sm100.cuh`。为什么 SM100 的 dense 实现基于 CUTLASS？反向传播的 softmax 梯度如何计算？ |
| **Q20: SM100 上的稀疏解码实现有什么特点？** | 看 `csrc/sm100/decode/head64/` 和 `csrc/sm100/decode/head128/`。为什么 head 维度不同需要分开实现？注意 README.md 中的说明。 |
| **Q21: `fwd_for_small_topk` 是什么优化？** | 当 topk 较小时，可以使用不同的内核实现以获得更好的性能。和标准 sparse prefill 有何区别？ |

**核心文件：**
- `csrc/sm100/helpers.h` — SM100 辅助函数
- `csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh` — Dense MHA 前向
- `csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cuh` — Dense MHA 反向
- `csrc/sm100/prefill/dense/interface.h` — Dense MHA 接口
- `csrc/sm100/prefill/sparse/fwd/head64/` / `head128/` — 稀疏预填充
- `csrc/sm100/prefill/sparse/fwd_for_small_topk/` — 小 topk 优化
- `csrc/sm100/decode/head64/` / `head128/` — 稀疏解码

---

### 第七层：Benchmark 和测试（性能验证）

> 理解项目如何验证正确性和度量性能。

| 问题 | 说明 |
|------|------|
| **Q22: 测试框架是如何组织正确性测试和性能测试的？** | 看 `tests/test_flash_mla_sparse_decoding.py` 中的 `gen_testcase()`。正确性测试对照 `ref.py` 中的 PyTorch 参考实现。性能测试用 `kernelkit` 做 profiling。 |
| **Q23: 如何理解 kernel 的 Compute/Memory 比率？** | 计算 bound 和 memory bound 的区别。FlashMLA 在 H800 上密集解码：memory-bound 配置下达 3000 GB/s，compute-bound 配置下达 660 TFLOPS。如何区分这两种场景？ |
| **Q24: Benchmark 中的 TFlops 和 GB/s 是怎么算出来的？** | 看 `lib.py` 中的 `count_flop_and_mem_vol_for_decode()`。Flop 计数包括了哪些运算（matmul + softmax 等）？内存访问量包括了哪些数据（Q、K、V、中间结果、输出）？ |

**核心文件：**
- `tests/test_flash_mla_dense_decoding.py`
- `tests/test_flash_mla_sparse_decoding.py`
- `tests/test_flash_mla_sparse_prefill.py`
- `tests/test_fmha_sm100.py`
- `tests/ref.py` — 参考实现（学习 attention 数学语义）
- `tests/lib.py` — 测试辅助函数
- `benchmark/bench_flash_mla.py` — 性能基准
- `benchmark/visualize.py` — 性能可视化

---

## 三、进阶学习路线

如果你希望深入 GPU 编程和 Attention 机制的实现细节：

### 阶段 1：Attention 机制本身
1. 先理解标准 Attention 的数学定义：$O = \text{softmax}(QK^T / \sqrt{d}) V$
2. 理解 FlashAttention 的 online softmax（tiling 分块计算）
3. 理解 MLA 的 MQA 模式：为什么 `head_dim_k=576` 但 `head_dim_v=512`？
4. 阅读 DeepSeek V3 / V3.2 论文中关于 MLA 的附录

### 阶段 2：CUDA 优化技巧
1. **Split-KV**：为什么要分段？如何合并结果？
2. **TMA（Tensor Memory Accelerator）**：SM90/SM100 的异步数据拷贝机制
3. **FP8 量化**：per-block scale factor 的设计、反量化的精度控制
4. **Warp Specialization**：Hopper 架构上 producer-consumer 的 warp 分工
5. **Persistent Kernel**：解码 kernel 为什么用 persistent 模式？
6. **Tile Scheduler**：`get_decoding_sched_meta` 计算了什么调度信息？

### 阶段 3：代码走读顺序（建议）

```
第一步 (宏观理解):
  README.md
  → flash_mla/flash_mla_interface.py
  → tests/ref.py
  → tests/test_flash_mla_sparse_prefill.py

第二步 (API 桥接):
  → csrc/api/api.cpp
  → csrc/api/sparse_fwd.h / sparse_decode.h / dense_decode.h / dense_fwd.h
  → setup.py

第三步 (共享基础设施):
  → csrc/params.h / defines.h
  → csrc/kerutils/include/kerutils/device/sm90/helpers.cuh
  → csrc/smxx/decode/combine/combine.cuh
  → csrc/smxx/decode/get_decoding_sched_meta/

第四步 (SM90 密集解码):
  → csrc/sm90/decode/dense/config.h
  → csrc/sm90/decode/dense/splitkv_mla.cuh
  → csrc/sm90/decode/dense/traits.h

第五步 (SM90 稀疏解码):
  → csrc/sm90/decode/sparse_fp8/config.h
  → csrc/sm90/decode/sparse_fp8/components/dequant.h
  → csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh

第六步 (SM90 稀疏预填充):
  → csrc/sm90/prefill/sparse/config.h
  → csrc/sm90/prefill/sparse/phase1.cuh
  → csrc/sm90/prefill/sparse/fwd.h

第七步 (SM100 内核):
  → csrc/sm100/helpers.h
  → csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh
  → csrc/sm100/prefill/sparse/fwd/head128/phase1.cuh
  → csrc/sm100/decode/head64/kernel.cuh
```

---

## 四、评估自己是否掌握了

当你能回答以下问题时，说明你对 FlashMLA 有了比较深入的理解：

1. **MLA 和标准 MHA 的数学差异是什么？为什么 KV Cache 能大幅减少？**
2. **Online softmax（tiled softmax）为什么是 FlashAttention 的核心创新？**
3. **Split-KV 策略在解码 kernel 中是怎么工作的？合并阶段为什么需要 log-sum-exp？**
4. **FP8 KV Cache 的格式为什么是 656 Bytes（512 FP8 + 16 scales + 128 BF16 RoPE）？**
5. **Prefill 和 Decoding 的 kernel 在 tile 策略上有什么本质不同？**
6. **Sparse attention 的 indices 编码方式——为什么 page block index 和 page block offset 要编码成一个整数？**
7. **SM90 和 SM100 的 kernel 实现分别利用了哪些架构特性？**
8. **如果一个 kernel 是 memory-bound，你可以从哪些角度优化？如果是 compute-bound 呢？**
