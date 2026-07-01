# FlashMLA FP8 稀疏解码内核深度解读：中英对照 + 背景讲解

> 逐行分析 `docs/20250929-hopper-fp8-sparse-deep-dive.md`，理解 FlashMLA 的 FP8 稀疏解码内核优化原理。

---

## 一、背景：这篇文章在说什么？

2025.09.29，DeepSeek 随 DeepSeek-V3.2 发布，将上下文长度从 64K 翻倍到 **128K tokens**。这给 GPU 显存带来了巨大压力——单次请求的 KV Cache 就需要 8.72 GiB。为了解决这个问题，DeepSeek 引入了 **FP8 KV Cache**（将 KV Cache 量化为 FP8 格式存储），并开发了配套的高性能稀疏解码内核。

这篇文章就是 DeepSeek 团队写的 FP8 稀疏解码内核深度解读，涵盖了：
- FP8 KV Cache 的格式设计
- 时钟周期的理论分析（为什么反量化会成为瓶颈）
- Crossover 技术（如何在 CTA 之间共享反量化结果）
- Distributed Shared Memory 的实现
- 性能数据

---

## 二、逐行翻译 + 讲解

---

### 原文标题

```
A Deep Dive Into The Flash MLA FP8 Decoding Kernel on Hopper
```

> **翻译：** Hopper 架构上 Flash MLA FP8 解码内核深度解析

---

### 第一段：引言

**原文：**

```markdown
With the release of DeepSeek-V3.2, we have doubled the context length of our models from 64K tokens
to 128K tokens. This puts significant pressure on GPU memory (a single request with 128K tokens
requires a KVCache of size $576 \times 2 \times 62 \times 128 \times 1024 = 8.72\ \mathrm{GiB}$), which
can lead to out-of-memory (OOM) errors or under-utilized GPUs due to small batch sizes. To address
this, we introduced FP8 KVCache for DeepSeek-V3.2.
```

> **翻译：** 随着 DeepSeek-V3.2 的发布，我们将模型的上下文长度从 64K tokens 翻倍到了 128K tokens。这给 GPU 显存带来了巨大压力（一个 128K tokens 的请求就需要大小为 $576 \times 2 \times 62 \times 128 \times 1024 = 8.72\ \mathrm{GiB}$ 的 KV Cache），这可能导致显存溢出（OOM）或因为 batch size 过小而导致 GPU 利用率不足。为了解决这个问题，我们为 DeepSeek-V3.2 引入了 FP8 KV Cache。

**[讲解] 这 8.72 GiB 是怎么算出来的？**

计算分解：
- **576** = head_dim_k（K 的 head 维度，MLA 特有）
- **× 2** = K 和 V 两份（KV Cache 需要缓存 K 和 V）
- **× 62** = Transformer 层数（DeepSeek V3 有 62 层 attention layer）
- **× 128** = head 数量
- **× 1024** = 128K tokens / 128（这里的 128 是 page block size？不对，等等）

重新看公式：$576 \times 2 \times 62 \times 128 \times 1024$

这其实是：576（每 token 每层的 KV 字节数，即 head_dim_k）× 2（K 和 V）× 62（层数）× 128K（tokens）= 9.3 × 10^9 bytes ≈ 8.72 GiB。注意这里已经隐含了 "每 token × 层数" 的逻辑，用了 128K 而不是 1024 —— 所以 1024 其实是 128 × 1024 / 128？不对。

让我重新解读：$576 \times 2 \times 62 \times 128 \times 1024$ 中：
- $576$ = 每 token 每条 attention 的 K 维度（head_dim）
- $2$ = K 和 V 两份
- $62$ = Transformer 层数
- $128$ = 128K tokens 中的 128
- $1024$ = 128 × 1024 / 128... 

实际上更合理的分解：576（head_dim_k）× 2（K+V）× 62（层数）× 128K（tokens数，用 128×1024 表示）= 8.72 GiB。即 576 × 2 × 62 = 71,424 bytes per token across all layers，乘以 131,072（128K）= 9,361,686,528 bytes ≈ 8.72 GiB。

8.72 GiB 是什么概念？一张 H800 是 80 GiB 显存，单请求就要吃掉 8.72 GiB，如果 batch size = 8 就直接 70 GiB 用完了——还有模型权重和其他中间结果的空间吗？所以 128K 上下文实际上限制了 batch size，导致 GPU 利用率下降。

**FP8 量化**就是要把这 8.72 GiB 降下来——FP8 相比 BF16 只用一半空间，加上分块量化的额外开销，能降到约 BF16 的 1/2 左右。

**原文：**

```markdown
However, writing a high-performance decoding kernel is challenging due to the need for dequantization
and its sparse memory access patterns. In this blog, we share the story behind our new FP8 sparse
decoding kernel for Hopper GPUs. We will first explain our FP8 KVCache format, then provide a
theoretical analysis of clock cycles, and finally detail the techniques used in our new kernel.
```

> **翻译：** 然而，编写高性能的解码内核是很有挑战性的，因为需要反量化处理以及稀疏的内存访问模式。在这篇博客中，我们将分享新的 Hopper GPU FP8 稀疏解码内核背后的故事。我们首先解释 FP8 KV Cache 的格式，然后进行时钟周期的理论分析，最后详细介绍新内核所使用的技术。

**[讲解] 稀疏解码的两个挑战**

1. **反量化开销**：FP8 存、BF16 算——每次读取 KV Cache 都要做 FP8→BF16 的转换。这个转换很慢（下面会分析）。
2. **稀疏内存访问模式**：稀疏 attention 只访问部分 token，而且这些 token 在显存中不是连续分布的。这破坏了 GPU 的合并内存访问（coalesced memory access），导致带宽利用率下降。

---

### 第二段：FP8 KV Cache 格式

**原文：**

```markdown
## The FP8 KVCache Format
```

> **翻译：** FP8 KV Cache 格式

**原文：**

```markdown
Recall that the decoding phase of the Multi-head Latent Attention (MLA) algorithm operates similarly
to Multi-Query Attention (MQA), with 128 query heads and 1 key head, where `head_dim_k = 576` and
`head_dim_v = 512` respectively. To reduce the size of the KVCache while maintaining accuracy, we
use a fine-grained quantization method. Specifically, we apply tile-level quantization (with a tile
size of $1 \times 128$) to the first 512 elements in each token's KV Cache. This results in 512
`float8_e4m3` values and 4 `float32` scale factors. For the remaining 64 elements (the RoPE part),
we do not apply quantization as they are sensitive to precision loss. Therefore, in GPU memory, each
token's KVCache occupies 656 bytes, consisting of 512 `float8_e4m3`s, 4 `float32`s, and 64
`bfloat16`s.
```

> **翻译：** 回顾一下，多头潜在注意力（MLA）算法的解码阶段与多查询注意力（MQA）类似，有 128 个 query head 和 1 个 key head，其中 `head_dim_k = 576` 和 `head_dim_v = 512`。为了在保持精度的同时减少 KV Cache 的大小，我们使用细粒度的量化方法。具体来说，我们对每个 token 的 KV Cache 中前 512 个元素应用 tile 级量化（tile 大小为 $1 \times 128$）。这产生了 512 个 `float8_e4m3` 值和 4 个 `float32` 缩放因子。对于剩余的 64 个元素（RoPE 部分），由于它们对精度损失敏感，我们不进行量化。因此，在 GPU 显存中，每个 token 的 KV Cache 占用 656 字节，由 512 个 `float8_e4m3`、4 个 `float32` 和 64 个 `bfloat16` 组成。

**[讲解] 656 字节的构成**

```
每个 token 的 KV Cache = 656 字节
├── 512 bytes: float8_e4m3 值（量化的 NoPE 部分）
│    含义：head_dim_k = 576，其中前 512 个元素是 NoPE（无位置编码）
│          用 FP8 存储 = 512 × 1 byte = 512 bytes
├── 16 bytes: float32 缩放因子（4 个 scale factor）
│    原因：tile 大小 1×128，512 ÷ 128 = 4 个 tile
│          每个 tile 一个 float32 scale = 4 × 4 bytes = 16 bytes
└── 128 bytes: bfloat16 值（未量化的 RoPE 部分）
     含义：head_dim_k 中剩余的 64 个元素是 RoPE（旋转位置编码）
           对精度敏感，保持 BF16 = 64 × 2 bytes = 128 bytes
```

**为什么 RoPE 部分不能量化？** RoPE（旋转位置编码）通过在 attention score 中注入位置信息来工作。RoPE 值的微小变化会直接影响 token 之间的相对位置关系，对精度非常敏感。实验表明确实无法量化而不损失质量。

**为什么 tile 大小选 1×128？** 这是精度和存储开销之间的权衡。tile 越小，量化越精确（每组值更少，scale 更贴近局部分布），但 scale factor 的存储开销越大。128 是一个常用的平衡点。

**原文：**

```markdown
Inside the kernel, we first dequantize the 512 `float8_e4m3` values into 512 `bfloat16`s. We then
concatenate them with the 64 original `bfloat16` values from the RoPE part. Finally, we perform the
MQA calculation using matrix multiplication-add (MMA) operations in `bfloat16` precision (i.e., the
inputs to the MMAs are in `bfloat16` and the outputs are in `float32`. This applies to both the QK
gemm and the attention-score-V gemm).
```

> **翻译：** 在内核内部，我们首先将 512 个 `float8_e4m3` 值反量化为 512 个 `bfloat16`。然后将它们与来自 RoPE 部分的 64 个原始 `bfloat16` 值拼接起来。最后，我们使用 `bfloat16` 精度的矩阵乘加（MMA）操作执行 MQA 计算（即：MMA 的输入是 `bfloat16`，输出是 `float32`。这同时适用于 QK 矩阵乘和 attention-score-V 矩阵乘）。

**[讲解] 为什么输入是 BF16 但输出是 FP32？**

这是 GPU 数值计算中的标准实践：
- **输入用 BF16**：节省显存带宽（BF16 是 2 字节，FP32 是 4 字节）
- **中间计算用 FP32**：避免精度损失。注意力计算中累加结果（如 score×V 的加权和）如果只用 BF16，累加误差会很大
- **输出也用 FP32**（或再量化回 BF16）：保证下游计算的精度

这就像记账：收据用简写记（BF16），但算总账时用精确数字（FP32）。

---

### 第三段：时钟周期理论分析

**原文：**

```markdown
## Theoretical Analysis of Clock Cycles
```

> **翻译：** 时钟周期理论分析

**原文：**

```markdown
The main challenge is that Tensor Cores (which handle MMA calculations) are extremely fast, while the
dequantization process, performed on CUDA Cores, struggles to keep up.
```

> **翻译：** 主要挑战在于，Tensor Core（负责 MMA 计算）非常快，而在 CUDA Core 上执行的反量化过程难以跟上。

**[讲解] Tensor Core vs CUDA Core 的速度差异**

这是理解整个优化关键的**核心矛盾**：
- **Tensor Core**：专用硬件，专门为矩阵乘（MMA）设计。每时钟周期可完成大量 FLOPs。
- **CUDA Core**：通用计算单元，处理转换、比较、加减等操作。做类型转换比 Tensor Core 慢得多。

在传统 BF16 解码中，数据从显存读到 Tensor Core 直接可用。但在 FP8 解码中，数据需要先经过 CUDA Core 反量化，再送到 Tensor Core——这个"中间人"环节成为了瓶颈。

**原文：**

```markdown
The basic unit on an NVIDIA GPU is the Stream Multiprocessor (SM). You can think of each SM as an
independent core on the GPU. For simplicity, let's focus on a single SM. Each SM can process 4096
MMA Flops per clock cycle (calculated as `989 TFlops / 1830 MHz / 132 SMs` on H800). In our kernel,
each CTA runs on one SM, and each SM is only mapped to one CTA. If we assign each CTA (CUDA Thread
Block) to process 64 query heads, it only requires $64 \times (576+512) \times 2 / 4096 \approx 34$
cycles for MMA operations per K/V token.
```

> **翻译：** NVIDIA GPU 的基本单元是流多处理器（SM）。你可以把每个 SM 想象成 GPU 上的一个独立内核。为简单起见，我们只关注单个 SM。每个 SM 每时钟周期可以处理 4096 个 MMA FLOPs（计算公式为 H800 上的 `989 TFlops / 1830 MHz / 132 SMs`）。在我们的内核中，每个 CTA 运行在一个 SM 上，每个 SM 只映射到一个 CTA。如果我们让每个 CTA 处理 64 个 query head，那么每个 K/V token 的 MMA 操作只需要 $64 \times (576+512) \times 2 / 4096 \approx 34$ 个时钟周期。

**[讲解] 数字验证**

- H800 理论峰值：989 TFlops（FP16/BF16 Tensor Core）
- H800 时钟频率：1830 MHz
- H800 SM 数量：132
- 每 SM 每周期 MMA FLOPs：989 × 10¹² / (1830 × 10⁶ × 132) ≈ 4096

每个 K/V token 的 MMA 操作量（处理 64 个 query head）：
- Q×K：64 × 576（矩阵乘，64 个 head 各 × head_dim_k 576）
- Score×V：64 × 512（加权求和，64 个 head 各 × head_dim_v 512）
- 每次矩阵乘有 2 个操作（乘+加）：64 × (576 + 512) × 2 = 139,264 FLOPs
- 时钟周期：139,264 / 4096 ≈ 34 cycles

这个数字**非常小**——34 个时钟周期对一个 token 的注意力计算来说极快。问题是反量化用了多少周期？

**原文：**

```markdown
However, because the H800 cannot directly cast `float8_e4m3` to `bfloat16`, dequantizing the KVCache
for one token requires the following steps:
1.  Convert `float8_e4m3` to `half`
2.  Convert `half` to `float32`
3.  Convert `float32` to `bfloat16`
4.  Multiply the converted `bfloat16` value by the `float32` scale factor
```

> **翻译：** 然而，由于 H800 不能直接将 `float8_e4m3` 转换为 `bfloat16`，反量化一个 token 的 KV Cache 需要以下步骤：
> 1. 将 `float8_e4m3` 转换为 `half`（FP16）
> 2. 将 `half` 转换为 `float32`
> 3. 将 `float32` 转换为 `bfloat16`
> 4. 将转换后的 `bfloat16` 值乘以 `float32` 缩放因子

**[讲解] 为什么 H800 不能直接 FP8→BF16？**

NVIDIA Hopper 架构的 Tensor Core 支持 FP8 输入做矩阵乘，但**CUDA Core 的指令集**不支持直接的 FP8→BF16 转换指令。CUDA Core 只支持：
- 直接的 FP8→FP16（通过 `__half2half2` + 类型转换）
- 直接的 FP16→FP32
- 直接的 FP32→BF16

所以这条链路是：`FP8 → FP16 → FP32 → BF16 → ×scale`，一步一步，一个都不能少。

**原文：**

```markdown
According to NVIDIA's documentation, we need at least $(\frac{1}{64} + \frac{1}{64} + \frac{1}{16} +
\frac{1}{256}) \times 512 \approx 50$ cycles for dequantizing each token! This is significantly more
than the 34 cycles required for the MMA operations, meaning the kernel is **dequantization-bound**.
If left unaddressed, dequantization would become the performance bottleneck, leaving the powerful
Tensor Cores underutilized.
```

> **翻译：** 根据 NVIDIA 的文档，反量化每个 token 至少需要 $(\frac{1}{64} + \frac{1}{64} + \frac{1}{16} + \frac{1}{256}) \times 512 \approx 50$ 个时钟周期！这显著多于 MMA 操作所需的 34 个周期，意味着内核是**反量化瓶颈型**。如果不解决，反量化将成为性能瓶颈，让强大的 Tensor Core 得不到充分利用。

**[讲解] 50 个周期的计算拆解**

公式中每个分数代表**每元素操作所需的周期数**（吞吐量的倒数）：

| 操作 | 每元素周期数 | 含义 |
|------|:-----------:|------|
| FP8→FP16 | 1/64 | 每周期可处理 64 个元素 |
| FP16→FP32 | 1/64 | 同上 |
| FP32→BF16 | 1/16 | 每周期可处理 16 个元素（慢一些） |
| BF16×FP32 | 1/256 | 每周期可处理 256 个元素 |

乘以 512（需要反量化的元素数）：
$(1/64 + 1/64 + 1/16 + 1/256) \times 512 = (0.015625 + 0.015625 + 0.0625 + 0.00390625) \times 512 \approx 0.09765625 \times 512 \approx 50$ cycles

**核心矛盾：**
- MMA（Tensor Core）：**34 cycles/token**
- 反量化（CUDA Core）：**50 cycles/token**

反量化比实际计算还慢！如果不解决，Tensor Core 有 50 - 34 = 16 个周期的空闲时间——利用率只有 34/50 = 68%。

---

### 第四段：Crossover 技术

**原文：**

```markdown
## Crossover
```

> **翻译：** 交叉（Crossover）

**原文：**

```markdown
Before we continue, it's important to note a key fact: every query head within the same query token
attends to the same key heads, because this is Multi-Query Attention (MQA).
```

> **翻译：** 在继续之前，先注意一个关键事实：同一个 query token 内的每个 query head 都与相同的 key head 计算注意力，因为这是多查询注意力（MQA）。

**[讲解] MQA 的关键特性**

在 MQA 中，所有 query head **共享同一组 K 和 V**。也就是说：
- Head 0 的 Q × K 和 Head 1 的 Q × K 用的是**同一个 K**（但 Q 不同）
- 因此，K 只需要从显存读取一次，就可以被所有 head 共用

在 FlashMLA 的语境中，128 个 query head 共享 1 个 key head（h_kv = 1）。所以每个 token 的 KV Cache 只需要缓存一份。

这个事实引出了一个关键的优化思路：**如果多个 CTA 处理同一个 query token 的不同 head，它们可以共享反量化后的 KV 数据**。

**原文：**

```markdown
Recall that each CTA processes 64 query heads, while DeepSeek-V3.2 has a total of 128 query heads.
If we can find a way to "share" the dequantized K/V values between two CTAs that are processing
different sets of query heads, then each CTA would only need to dequantize **half** of the KV cache
– which is fantastic! We call this method "crossover", since the idea was actually inspired by
Chromosomal crossover during Meiosis.
```

> **翻译：** 回顾一下，每个 CTA 处理 64 个 query head，而 DeepSeek-V3.2 总共有 128 个 query head。如果我们能找到一种方法，在处理不同 query head 的两个 CTA 之间"共享"反量化后的 K/V 值，那么每个 CTA 就只需要反量化 KV Cache 的**一半**——这太棒了！我们称这种方法为"crossover"，因为这个想法实际上受到减数分裂中染色体交叉的启发。

**[讲解] Crossover 的核心思想**

```
无 crossover（naive 方案）：
  CTA0 (head 0-63):  反量化全部 512 个 FP8 → BF16  → MMA
  CTA1 (head 64-127): 反量化全部 512 个 FP8 → BF16  → MMA
  → 同一个 KV Cache 被反量化了两次！浪费！

有 crossover：
  CTA0 (head 0-63):  反量化前半 256 个 FP8 → BF16  → 共享给 CTA1  → MMA with full KV
  CTA1 (head 64-127): 反量化后半 256 个 FP8 → BF16  → 共享给 CTA0  → MMA with full KV
  → 每个 CTA 只反量化一半，通过共享得到另一半
```

**为什么叫 Crossover？** 这和生物减数分裂中的染色体交叉（Chromosomal crossover）非常相似——两条染色体交换对应片段，双方都获得对方的部分基因。两个 CTA 各自反量化一半 KV，然后交换，双方都获得完整的 KV。

---

### 第五段：Distributed Shared Memory

**原文：**

```markdown
## Distributed Shared Memory to the Rescue
```

> **翻译：** 分布式共享内存来救援

**原文：**

```markdown
Distributed Shared Memory (DSM) is a new feature introduced with the Hopper architecture, alongside
the CTA Cluster (thread block cluster). CTAs within the same cluster can directly access each other's
shared memory. For more details, you can refer to NVIDIA Hopper Architecture In-Depth.
```

> **翻译：** 分布式共享内存（DSM）是 Hopper 架构引入的新特性，与 CTA Cluster（线程块集群）一起推出。同一集群内的 CTA 可以直接访问彼此的共享内存。更多细节请参考 NVIDIA 官方博客。

**[讲解] 什么是 CTA Cluster 和 DSM？**

传统 GPU 架构中，不同的 CTA（CUDA Thread Block）是相互隔离的：
- CTA A 和 CTA B 各自有私有的共享内存
- CTA A 不能直接访问 CTA B 的共享内存
- 数据只能通过全局内存或 L2 缓存交换——**很慢**

Hopper 引入的 **CTA Cluster** 和 **Distributed Shared Memory**：
- 你可以把 2 个（或更多）CTA 组成一个 Cluster
- Cluster 内的 CTA 可以把共享内存映射为一个"集群级共享内存池"
- CTA A 可以**直接读 CTA B 的共享内存**，不需要经过全局内存

这正是 Crossover 所需的底层支持。

**原文：**

```markdown
Here is how we use it: We launch CTAs in clusters of size 2. Each CTA within a cluster is responsible
for 64 query heads from the same query token. Each CTA performs the following steps:
1.  Loads *half* of the quantized K/V from global memory. We use a wide `__ldg` load with a width
    of 128 bits to improve performance.
2.  Dequantizes its assigned half on the CUDA Cores.
3.  Stores the dequantized K/V into its own shared memory.
4.  Simultaneously uses `st.async` to write the dequantized K/V into the shared memory of the other
    CTA in the cluster.
```

> **翻译：** 以下是我们的使用方法：我们以大小为 2 的 Cluster 启动 CTA。Cluster 中的每个 CTA 负责来自同一个 query token 的 64 个 query head。每个 CTA 执行以下步骤：
> 1. 从全局内存加载**一半**的量化的 K/V。我们使用宽度为 128 位的 `__ldg` 加载指令来提高性能。
> 2. 在 CUDA Core 上反量化其分配的一半。
> 3. 将反量化后的 K/V 存储到自己的共享内存中。
> 4. 同时使用 `st.async` 将反量化后的 K/V 写入集群中另一个 CTA 的共享内存。

**[讲解] 四步流程详解**

```
Global Memory (HBM)           CTA0 Shared Mem      CTA1 Shared Mem
┌───────────────────┐
│ FP8 K Cache       │
│ ┌──────┬──────┐   │        ┌─────────────┐      ┌─────────────┐
│ │ Half │ Half │   │  step1 │             │      │             │
│ │  0   │  1   │   │ ────→  │  K Half 0   │      │             │
│ └──────┴──────┘   │  ldg   │  (FP8)      │      │             │
│                   │        │             │      │             │
│                   │  step2 │  ──dequant──│      │             │
│                   │        │  K Half 0   │      │             │
│                   │        │  (BF16)     │      │             │
│                   │  step3 │             │      │             │
│                   │        │  store      │      │             │
│                   │  step4 │  ──────────────→  │  K Half 0   │
│                   │        │  st.async   │      │  (BF16)     │
│                   │        │             │      │             │
└───────────────────┘        └─────────────┘      └─────────────┘
```

关键的优化点：
- **`__ldg` 128-bit 加载**：一次加载 16 字节（4 个 FP8 元素），提高显存带宽利用率。稀疏访问模式下，合并访问困难，宽加载指令可以最大化每次访存的有效字节数。
- **`st.async` 异步存储**：写入对方共享内存是异步的，不阻塞当前 CTA 继续执行后续指令。
- **每 CTA 只处理一半反量化**：原来 50 周期 → 现在约 25 周期（因为只处理一半）。

**原文：**

```markdown
For synchronization between these operations, we rely on the cluster transaction barrier, another
powerful programming primitive available in CTA Clusters. After the data exchange is complete, each
CTA has the *full* set of dequantized K and V values available in its own shared memory, which it
can then use to perform the MMA operations.
```

> **翻译：** 对于这些操作之间的同步，我们依赖集群事务屏障（cluster transaction barrier），这是 CTA Cluster 中可用的另一个强大的编程原语。数据交换完成后，每个 CTA 在其自己的共享内存中都拥有**完整的**反量化后的 K 和 V 值，然后可以用来执行 MMA 操作。

**[讲解] Cluster Transaction Barrier**

传统的 `__syncthreads()` 只同步**同一个 CTA** 内的线程。而 cluster transaction barrier（通过 `__cluster_barrier_wait()` 等指令）可以同步**同一 Cluster 内的所有 CTA**。

在 Crossover 流程中：
1. 两个 CTA 各自做自己的反量化 + `st.async`（没有同步依赖）
2. 遇到 cluster barrier：两个 CTA 都等在这里
3. 当双方都到达 barrier 时，`st.async` 的写入保证已经完成
4. 此时每个 CTA 的共享内存中都有完整的 K/V（自己的那一半 + 对方写过来的那一半）
5. 继续执行 MMA 操作

```
       CTA0                     CTA1
         │                        │
   反量化前半                 反量化后半
   写对方共享内存             写对方共享内存
         │                        │
    ──── cluster barrier ────
         │                        │
   共享内存中已有            共享内存中已有
   完整 K/V                  完整 K/V
         │                        │
   开始 MMA                 开始 MMA
```

---

### 第六段：性能

**原文：**

```markdown
## Performance
```

> **翻译：** 性能

**原文：**

```markdown
Using these techniques, we achieved 410 TFLOPS in a compute-bound configuration
(batch_size=128, num_heads=128, s_q=2, topk=2048) on H800 SXM5 GPUs. This is a significant
improvement over the 250 TFLOPS achieved by our previous FP8 sparse decoding kernel without the
crossover technique.
```

> **翻译：** 使用这些技术，我们在 H800 SXM5 GPU 上实现了 compute-bound 配置（batch_size=128, num_heads=128, s_q=2, topk=2048）下 410 TFLOPS 的性能。相比之前没有使用 crossover 技术的 FP8 稀疏解码内核（250 TFLOPS），这是一个显著的提升。

**[讲解] 410 vs 250 = 1.64 倍的提升**

Crossover 技术的收益是**巨大**的：
- 没有 crossover：250 TFLOPS（反量化是瓶颈，Tensor Core 利用率低）
- 有 crossover：410 TFLOPS（反量化开销被分摊到两个 CTA，不再是瓶颈）

提升幅度 64%，远超通常的 5-15% 优化。这说明 Crossover 确实解决了核心瓶颈，而不是只做了边际优化。

**原文：**

```markdown
Although this number is still below the 640 TFLOPS peak of our previous bfloat16 dense decoding
kernel, one reason is that it's a **sparse** kernel, and its topk is only 2048. With a smaller topk,
the relative overhead of the kernel's prologue and epilogue becomes larger compared with dense
decoding with long context length. If we set topk to a larger value, such as 32768, this kernel can
achieve up to 460 TFLOPS.
```

> **翻译：** 虽然这个数字仍然低于之前 bfloat16 密集解码内核的 640 TFLOPS 峰值，一个原因是这是一个**稀疏**内核，而它的 topk 只有 2048。当 topk 较小时，内核的 prologue 和 epilogue 的相对开销会变大（相比于长上下文的密集解码）。如果将 topk 设置为更大的值（如 32768），这个内核可以达到 460 TFLOPS。

**[讲解] 为什么稀疏内核跑不过密集内核？**

稀疏内核和密集内核的一个关键区别：**计算量不同**。

- 密集内核：对所有 KV token 计算注意力（假设 32K 个 token）
- 稀疏内核：只对 topk 个 token 计算注意力（如 topk=2048，只有 1/16 的计算量）

但 **prologue（初始化、元数据加载）和 epilogue（结果写入、后处理）的开销是固定的**，和 topk 无关。所以 topk 越小，这些固定开销占比越大，TFlops 就越低。

当 topk 从 2048 增加到 32768（16 倍），计算量也增加了 16 倍，但固定开销不变，所以 TFlops 可以上升到 460（但仍是稀疏的，因为 32768 < full sequence length）。

**原文：**

```markdown
From another perspective, the execution time of this kernel in the configuration mentioned above is
comparable to that of the dense decoding kernel when the sequence length is around 3000. When the
sequence length exceeds 3000, the performance advantage of our new kernel becomes even more
significant. This also highlights the effectiveness of our DeepSeek Sparse Attention algorithm.
```

> **翻译：** 从另一个角度看，在上述配置下，这个内核的执行时间与序列长度约为 3000 时的密集解码内核相当。当序列长度超过 3000 时，我们的新内核的性能优势变得更加显著。这也凸显了 DeepSeek 稀疏注意力算法的有效性。

**[讲解] 稀疏注意力的实际意义**

这个对比非常直观：
- **序列长度 ≤ 3000**：稀疏和密集速度差不多（稀疏的额外开销抵消了节省的计算量）
- **序列长度 > 3000**：稀疏比密集快（节省的计算量超过了额外开销）
- **序列长度 = 128K**：稀疏快得多（密集要处理 128K 个 token，稀疏只处理 topk=2048）

DeepSeek Sparse Attention（DSA）之所以有效，是因为它通过某种筛选机制（基于之前的 attention pattern 或启发式规则），从 128K 个 token 中选出最相关的 2048 个——维持了质量，但计算量降低到 1/64。

---

## 三、总结

| 技术点 | 解决的问题 | 具体方法 | 效果 |
|--------|-----------|---------|------|
| **FP8 KV Cache** | 128K 上下文显存不足 | tile 级量化（1×128），RoPE 部分不量化 | 656 bytes/token（vs BF16 的 1152） |
| **反量化瓶颈分析** | 定位性能瓶颈 | 精确计算时钟周期开销 | 发现反量化(50cyc) > MMA(34cyc) |
| **Crossover** | 反量化成为瓶颈 | 两个 CTA 各反量化一半后交换 | 250→410 TFLOPS（+64%） |
| **Distributed Shared Memory** | CTA 间数据交换慢 | Hopper CTA Cluster + DSM 直接共享 | 避免全局内存读写延迟 |
| **Cluster Transaction Barrier** | 同步两个 CTA 的共享 | 集群级同步原语 | 无竞态条件的数据交换 |
| **128-bit ldg 加载** | 稀疏访问的带宽利用率 | 宽加载指令 | 提高全局内存读取效率 |
