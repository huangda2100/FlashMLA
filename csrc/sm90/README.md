# csrc/sm90 —— H100 (SM90) 专用的 MLA attention CUDA kernel

## 这个目录解决什么问题？

FlashMLA 是 DeepSeek V3.2 模型的注意力（MLA, Multi-head Latent Attention）高性能实现。
大模型推理分两个阶段，每个阶段的计算特点完全不同，所以要用不同的 kernel：

| 阶段 | 做什么 | 计算特点 | 对应子目录 |
|------|--------|----------|------------|
| **预填充 (Prefill)** | 一次性处理用户输入的所有 token | Q/K/V 都是批量数据，算一次完事 | `prefill/` |
| **解码 (Decode)** | 逐字生成回复，每次 1 个 token | Q 只有 1 个，K/V 要从缓存读，反复调用 | `decode/` |

每个阶段还分两种注意力模式：

| 模式 | 含义 | 数据格式 | 对应子目录 |
|------|------|----------|------------|
| **Dense (稠密)** | 所有历史 token 都参与 attention | K/V 用 bf16/fp16 存 | `decode/dense/` |
| **Sparse (稀疏)** | 只挑 topk 个最相关的 token 参与 | K/V 用 FP8 量化存，配 topk 索引 | `decode/sparse_fp8/`、`prefill/sparse/` |

所以 `csrc/sm90` 下面有 3 个 kernel 子目录 + 1 个通用工具文件：

```
csrc/sm90/
├── helpers.h                    # 通用工具：cp.async、WGMMA、TMA、cluster 等封装
├── prefill/sparse/              # 预填充阶段的稀疏注意力 kernel
├── decode/dense/                # 解码阶段的稠密注意力 kernel
└── decode/sparse_fp8/           # 解码阶段的稀疏 + FP8 量化注意力 kernel
```

## 为什么叫 "sm90"

SM90 是 NVIDIA H100 GPU 的计算能力版本号。这个目录下的代码**只能在 H100 上跑**——
它们用了 H100 独有的硬件特性：

- **WGMMA** (Warpgroup Matrix Multiply-Accumulate)：一条指令算 64×N 大矩阵乘
- **TMA** (Tensor Memory Accelerator)：硬件数据搬运单元，不占计算资源
- **Cluster**：多个 SM 组成一组，可以共享 shared memory、跨 SM 通信
- **async barrier**：带字节计数的同步原语，能边搬数据边同步

老 GPU（A100/V100）没有这些硬件，跑不动这些 kernel。

## 各子目录详解

### `helpers.h` —— 通用工具函数库

把底层 PTX 内联汇编、CuTe 模板魔法封装成一行就能调的 C++ 函数，让 kernel 主体保持简洁：
- 异步数据搬运（cp.async、TMA）
- L2 cache 策略控制（evict_last / evict_first）
- WGMMA 矩阵乘法封装
- Cluster 模式下访问邻居 SM 的 shared memory
- 查询"我现在在哪个 SM"

### `prefill/sparse/` —— 预填充阶段稀疏注意力

**核心问题**：用户输入一长段 prompt 后，要一次性算完所有 token 的 attention。但稀疏
模式只挑 topk 个最相关的 K/V 参与，省计算量。

**两阶段架构**：
- `phase1.cuh`：处理被 topk 选中的 K/V，输出部分 O / max_logits / lse
- `phase2`（在其他地方）：处理剩余 token，和 phase1 结果合并

**3 个 warpgroup 分工**（共 384 线程）：
- WG0：算左半 QK^T + 左半 PV
- WG1：算右半 QK^T + 右半 PV
- Producer WG：异步加载 K/V 到 shared memory

**模板实例化**（`instantiations/`）：按 Q/K 维度 (576 或 512) × 是否支持变长 topk (true/false) 组合，共 4 份：
- `phase1_k576.cu` —— V3.2 模型，固定 topk
- `phase1_k576_topklen.cu` —— V3.2 模型，变长 topk
- `phase1_k512.cu` —— MODEL1 模型，固定 topk
- `phase1_k512_topklen.cu` —— MODEL1 模型，变长 topk

`fwd.cu` 是运行时入口，根据 `params.d_qk` 和 `params.topk_length` 选对应的实例调用。

### `decode/dense/` —— 解码阶段稠密注意力

**核心问题**：解码时每次生成 1 个 token，要用当前 Q 去查所有历史 K/V。历史 K/V 可能
很长（几万 token），用 **Split-KV** 把 K/V 切成多段，多个 SM 并行算不同段，最后合并。

**数据格式**：K/V 用 bf16 或 fp16 存（没量化）。

**模板实例化**（`instantiations/`）：按数据类型 (bf16/fp16) 分：
- `bf16.cu`
- `fp16.cu`

**关键文件**：
- `splitkv_mla.cuh` —— 主 kernel 实现
- `splitkv_mla.h` —— 对外声明
- `config.h` —— 静态配置（HEAD_DIM_K=576, HEAD_DIM_V=512 等）
- `traits.h` —— 类型 traits，根据 InputT 选 WGMMA 配置

### `decode/sparse_fp8/` —— 解码阶段稀疏 + FP8 量化注意力

**核心问题**：解码 + 稀疏 + FP8 三个优化叠加，是性能最高的解码 kernel。

**三个优化叠加**：
1. **稀疏**：只算 topk 个最相关的 token（和 prefill/sparse 一样）
2. **FP8 量化**：K/V 用 8-bit 浮点存，省一半显存和带宽
3. **Split-KV**：多个 SM 并行处理同一序列的不同 K/V 段，加速长序列解码

**3 个 warpgroup 分工**（和 prefill/sparse 类似但不同）：
- WG0：算 QK^T + 左半 PV + 写 sScale/sS 给 WG1
- WG1：算右半 PV（用 WG0 共享的 sS）
- Producer WG：读 FP8 K/V，反量化成 bf16 写到 smem（CLUSTER_SIZE=2 时还要异步写到对端 SM）

**两种模型架构支持**：
- **V3.2 (DeepSeek V3.2)**：HEAD_DIM_K=576（512 nope + 64 rope），4 个 fp32 scale
- **MODEL1**：HEAD_DIM_K=512（448 nope + 64 rope），8 个 fp8_e8m0 scale，支持动态 topk_length 和 extra KV cache

**模板实例化**（`instantiations/`）：2 模型 × 2 head 数 = 4 份：
- `v32_persistent_h64.cu` —— V3.2 + 64 头
- `v32_persistent_h128.cu` —— V3.2 + 128 头
- `model1_persistent_h64.cu` —— MODEL1 + 64 头
- `model1_persistent_h128.cu` —— MODEL1 + 128 头

文件名带 "persistent" 因为用了 **Persistent kernel** 优化——kernel 启动后不退出，
循环处理多个 batch 任务，省去反复启动 kernel 的开销。

**子目录 `components/`**：FP8 反量化相关的小工具：
- `dequant.h` —— FP8 → bf16 反量化函数（PTX 指令封装）
- `config.h` —— FP8 相关常量（NUM_SCALES、NUM_BYTES_PER_TOKEN 等）
- `helpers.h` —— 局部辅助函数

## 整体调用链

```
用户 Python 代码
     ↓
flash_mla.cuda.sparse_prefill_fwd(...) / decode_mla(...)  ← PYBIND11 绑定
     ↓
csrc/flash_mla_interface.cpp  ← 选择 SM90 / 非 SM90 实现
     ↓
sm90::fwd::run_fwd_phase1_kernel(...)              (prefill 阶段)
sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel(...)  (decode + sparse + fp8)
sm90::run_flash_splitkv_mla_kernel(...)            (decode + dense)
     ↓
本目录下的 kernel（最终在 H100 GPU 上执行）
```

## 术语速查表

| 术语 | 含义 |
|------|------|
| **MLA** | Multi-head Latent Attention，DeepSeek 模型的注意力变体 |
| **SM** | Streaming Multiprocessor，GPU 的计算单元（类比 CPU 核心） |
| **Warpgroup (WG)** | 128 线程组成的组，H100 WGMMA 的最小调度单位 |
| **WGMMA** | Warpgroup Matrix Multiply-Accumulate，H100 张量核心指令 |
| **TMA** | Tensor Memory Accelerator，H100 硬件数据搬运单元 |
| **Cluster** | H100 的 SM 组概念，组内可共享 shared memory |
| **smem** | Shared memory，SM 内部高速缓存 |
| **Split-KV** | 把 K/V 切成多段并行算，最后合并的优化 |
| **topk** | 稀疏注意力只选 topk 个最相关的 token |
| **FP8** | 8-bit 浮点数（e4m3 / e8m0），省一半显存 |
| **nope / rope** | MLA 把 K 拆成 nope（不旋转）和 rope（旋转位置编码）两部分 |
| **lse** | log-sum-exp，在线 softmax 的中间状态，用于合并 |
| **PDL** | Programmatic Dependent Launch，H100 kernel 提前启动优化（本项目因编译器 bug 禁用） |

## 编译注意事项

- 只能在 H100 (SM90) 及以上 GPU 上编译运行
- 依赖 CUTLASS、CuTe 库
- 实例化文件（`instantiations/*.cu`）必须独立编译，避免重复定义的链接错误
- 每个实例化文件对应一种模板参数组合，Python 层根据模型类型/head 数动态选择
