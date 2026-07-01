# FlashMLA Kernel 深度解读：中英对照 + 背景讲解

> 逐行分析 `docs/20250422-new-kernel-deep-dive.md`，理解 FlashMLA 密集解码内核的优化原理。

---

## 一、背景：这篇文章在说什么？

FlashMLA 在 2025.04.22 发布了一个**新版本的内核**，性能相比旧版提升了 5%~15%，在 H800 上密集解码达到了 **660 TFLOPS**（上一版是 580 TFLOPS）。

这篇文章就是 DeepSeek 团队写的**技术深度解读**，解释了新内核的优化思路和具体实现。文章不长，但信息密度很高，涉及：计算 bound vs 访存 bound 分析、online softmax、warpgroup 调度、TMA 拷贝、Split-KV 等技术。

---

## 二、逐行翻译 + 讲解

---

### 原文标题

```
A Deep-Dive Into the New Flash MLA Kernel
```

> **翻译：** 深入解析新版 Flash MLA 内核

---

### 第一段：引言

**原文：**

```markdown
In the previous version of the Flash MLA kernel, we have achieved impressive performance:
3000 GB/s in memory-intensive settings and 580 TFlops in compute-bound settings.
Now, we're pushing these numbers even further, reaching up to 660 TFlops.
```

> **翻译：** 在 Flash MLA 内核的上一版本中，我们已经取得了令人瞩目的性能：在访存密集型场景下达 3000 GB/s，在计算密集型场景下达 580 TFlops。现在，我们将这些数字进一步推高，达到了 660 TFlops。

**原文：**

```markdown
In this blog, we present a deep dive into the new kernel, explaining the optimizations and techniques
behind this performance boost. We'll first explain why the MLA kernel is compute-bound despite being a
decoding-stage attention kernel, then discuss our high-level kernel schedule design, and finally cover
the technical details of the new kernel.
```

> **翻译：** 在这篇博客中，我们将深入解析新内核，解释这次性能提升背后的优化和技术。我们首先分析 MLA 解码内核为什么是计算密集型的（虽然解码阶段的注意力内核通常被认为是访存密集型的），然后讨论我们的高层次 kernel 调度设计，最后介绍新内核的技术细节。

**[讲解] 为什么这里要强调 "解码内核是 compute-bound"？**

通常来说，大模型的**解码阶段**是访存密集型的，因为解码时 Q 只有 1 个 token，而 KV Cache 很大——主要瓶颈在于把 KV Cache 从显存搬到计算单元。但 FlashMLA 这里的情况不同：

- DeepSeek V3 的解码不使用 Tensor Parallel（TP=1），所以 `h_q = 128`（128 个 query head）
- 128 个 query head 意味着 Q 的规模扩大了 128 倍，计算量也因此大幅上升
- 计算量（FLOPs）和访存量（bytes）的比值超过了 H800 的平衡点，所以变成了 compute-bound

这是理解整个优化的**前提**——优化 compute-bound kernel 和优化 memory-bound kernel 的策略完全不同。

---

### 第二段：算法理论分析

**原文：**

```markdown
## A Theoretical Analysis of the MLA Algorithm
```

> **翻译：** MLA 算法的理论分析

**原文：**

```markdown
GPU kernels can be classified as either compute-bound (limited by floating-point operations per second,
FLOPs) or memory-bound (limited by memory bandwidth). To identify the kernel's bottleneck, we calculate
the ratio of FLOPs to memory bandwidth (FLOPs/byte) and compare it with the GPU's capacity.
```

> **翻译：** GPU 内核可以分为两类：计算密集型（受浮点运算速度 FLOPs 限制）或访存密集型（受显存带宽限制）。要确定内核的瓶颈，我们计算 FLOPs 和访存量的比值（每字节对应的计算量），然后与 GPU 的能力进行比较。

**[讲解] Compute-bound vs Memory-bound**

想象你在做饭：
- **Compute-bound** = 你是切菜大师傅，刀工极快，但砧板太小一次放不了太多菜。厨师等着菜送来（CPU 等着数据）。
- **Memory-bound** = 砧板上堆满了菜，但你的刀不够快。瓶颈在于处理速度，不是等菜。

在 GPU 上：
- **Compute-bound**：GPU 的计算单元（Tensor Core）忙到 100%，内存带宽还有富余。问题在于计算不够快。
- **Memory-bound**：内存带宽已经用满，但计算单元经常空闲（在等数据）。问题在于数据搬得不够快。

**原文：**

```markdown
Assume the number of q heads is $h_q$, the number of q tokens per request is $s_q$ (should be 1 if MTP /
speculative decoding is disabled), the number of kv tokens per request is $s_k\ (s_k \gg h_q s_q)$, and
the head dimensions of K and V are $d_k$ and $d_v$ respectively. The number of FLOPs is roughly
$2 (h_q s_q \cdot d_k \cdot s_k + h_q s_q \cdot s_k \cdot d_v) = 2 h_q s_q s_k (d_k+d_v)$,
and the memory access volume (in bytes) is
$\mathop{\text{sizeof}}(\text{bfloat16}) \times (h_q s_q d_k + s_k d_k + h_q s_q d_v) \approx 2s_k d_k$.
Thus, the compute-memory ratio is $h_q s_q \cdot \frac{d_k+d_v}{d_k} \approx 2 h_q s_q$.
```

> **翻译：** 假设 query head 数量为 $h_q$，每个请求的 query token 数为 $s_q$（如果 MTP/推测解码被禁用则为 1），每个请求的 KV token 数为 $s_k$（$s_k \gg h_q s_q$），K 和 V 的 head 维度分别为 $d_k$ 和 $d_v$。FLOPs 数大约为 $2 (h_q s_q \cdot d_k \cdot s_k + h_q s_q \cdot s_k \cdot d_v) = 2 h_q s_q s_k (d_k+d_v)$，访存量（以字节计）大约为 $\text{sizeof}(\text{bfloat16}) \times (h_q s_q d_k + s_k d_k + h_q s_q d_v) \approx 2s_k d_k$。因此，计算-访存比为 $h_q s_q \cdot \frac{d_k+d_v}{d_k} \approx 2 h_q s_q$。

**[讲解] 这个数学推导在说什么？**

这段推导核心是计算**每字节访存对应多少次浮点运算**（计算-访存比）。

- **FLOPs（计算量）：** 注意力计算有两个矩阵乘：Q×K（score）和 score×V（输出）。每次矩阵乘大约 $2 \times \text{输入1大小} \times \text{输入2大小}$ 次浮点运算。代入推导得到：$2 h_q s_q s_k(d_k + d_v)$

- **访存量（Memory）：** 需要从显存读取 Q、K、V，写出输出 O。由于解码阶段 $s_k$ 远大于其他维度，KV Cache 是主要访存对象：大约 $2 s_k d_k$ 字节（BF16 格式）。

- **比值：** $\frac{\text{FLOPs}}{\text{Memory}} \approx 2 h_q s_q$

关键洞察：当 $h_q = 128$（DeepSeek V3 的解码设置）、$s_q = 1$（推测解码关闭）时，$2 h_q s_q = 256$。这是一个非常大的比值——据此判断 kernel 是 compute-bound。

**原文：**

```markdown
An NVIDIA H800 SXM5 GPU has a peak memory bandwidth of 3.35 TB/s and peak FLOPs of 990 TFlops.
However, due to throttling (reducing to ~1600 MHz in our case), the practical peak FLOPs drops to
~865 TFlops. Therefore, when $h_qs_q \ge \frac{1}{2} \cdot \frac{865}{3.35} = 128$, the kernel is
compute-bound; otherwise, it's memory-bound.
```

> **翻译：** NVIDIA H800 SXM5 GPU 的峰值显存带宽为 3.35 TB/s，峰值算力为 990 TFlops。但由于降频（在实际运行中降到了约 1600 MHz），实际峰值算力约为 865 TFlops。因此，当 $h_q s_q \ge \frac{1}{2} \cdot \frac{865}{3.35} = 128$ 时，内核是计算密集型的；否则是访存密集型的。

**[讲解] "Roof line" 分析**

这里在做一个经典的 roof-line 分析：

- GPU 的内存带宽：3.35 TB/s → 每纳秒可以搬 3350 字节
- GPU 的计算能力：865 TFlops → 每纳秒可以做 865 万亿次浮点运算
- **平衡点：** 865 / 3.35 ≈ 258 FLOPs/byte。通俗说：每搬 1 个字节到 GPU，GPU 需要做 258 次浮点运算才能刚好用完算力。

但如果只是简单的注意力计算，还要考虑中间的 softmax（不是纯矩阵乘），所以有个 1/2 的经验因子。平衡点是 129。

当 $h_q s_q = 128 \ge 129$ 时，计算量访存量比 256 超过 GPU 容量，所以是计算密集型。

**原文：**

```markdown
According to the overview of DeepSeek's Online Inference System, we don't use Tensor Parallel for
decoding instances, meaning $h_q$ is 128 and the kernel is compute-bound. Thus, we need to optimize
the kernel for compute-bound settings.
```

> **翻译：** 根据 DeepSeek 在线推理系统的概述，解码实例不使用 Tensor Parallel，意味着 $h_q = 128$，内核是计算密集型的。因此，我们需要针对计算密集型场景来优化内核。

**[讲解] 为什么不用 Tensor Parallel 反而让 kernel 变成了 compute-bound？**

Tensor Parallel 会把 head 分摊到多个 GPU 上。如果不使用 Tensor Parallel，所有 128 个 head 都在一张 GPU 上计算——Q 数据的规模是 128 × head_dim，足够大，使得矩阵乘的计算密度高于访存。

这是反直觉的：通常解码被认为必然是 memory-bound——但 DeepSeek V3 的 MLA 解码因为 head 数多，变成了 compute-bound。这个判断决定了所有后续的优化方向。

---

### 第三段：新内核的高层设计

**原文：**

```markdown
## High-Level Design of the New Kernel
```

> **翻译：** 新内核的高层设计

**原文：**

```markdown
To fully utilize GPU compute resources, we need to overlap CUDA Core operations with Tensor Core
operations and memory access with computation, keeping the Tensor Core constantly busy. This requires
redesigning the kernel's "schedule."
```

> **翻译：** 为了充分利用 GPU 的计算资源，我们需要将 CUDA Core 操作与 Tensor Core 操作重叠，同时将内存访问与计算重叠，让 Tensor Core 始终处于忙碌状态。这需要重新设计内核的"调度"。

**[讲解] CUDA Core vs Tensor Core**

- **Tensor Core**：NVIDIA GPU 上的专用矩阵乘加速器。处理 GEMM（通用矩阵乘）时极快。
- **CUDA Core**：传统的 GPU 计算单元，处理非矩阵运算（加法、比较、指数、softmax 等）。

在注意力计算中，交替出现：
1. Tensor Core 做矩阵乘（Q×K、score×V）
2. CUDA Core 做 softmax、rescale 等

优化目标是：**Tensor Core 永远不要等 CUDA Core**，反之亦然。理想的调度是 CUDA Core 在处理步骤 N 的 softmax 时，Tensor Core 同时在计算步骤 N+1 的矩阵乘。

**原文：**

```markdown
FlashAttention-3's paper introduces ping-pong scheduling and intra-warpgroup GEMM-softmax pipelining
to overlap block-wise matmul and CUDA Core operations. However, these techniques can't be directly
applied here due to resource constraints. The output matrix (scaled and accumulated during each
mainloop round, similar to FlashAttention's algorithm) must be stored in registers due to WGMMA
instruction requirements. Each $64 \times 512$ output matrix occupies 32,768 32-bit registers. With
only 65,536 32-bit registers per SM, we can store only one output matrix per SM. This eliminates the
possibility of having two output matrices and letting them use CUDA Core and Tensor Core in a
interleaved manner. We need to find another clever way to overlap CUDA Core and Tensor Core
computation.
```

> **翻译：** FlashAttention-3 的论文引入了 ping-pong 调度和 warpgroup 内部的 GEMM-softmax 流水线，以重叠分块的矩阵乘和 CUDA Core 操作。然而，这些技术由于资源限制无法直接在这里应用。输出矩阵（在每一轮主循环中缩放和累积，类似于 FlashAttention 算法）必须存储在寄存器中，这是 WGMMA 指令的要求。每个 $64 \times 512$ 的输出矩阵占用 32,768 个 32 位寄存器。每个 SM 只有 65,536 个 32 位寄存器，所以每个 SM 只能存储一个输出矩阵。这就排除了使用两个输出矩阵交替进行 CUDA Core 和 Tensor Core 操作的可能性。我们需要找到另一种巧妙的方法来重叠 CUDA Core 和 Tensor Core 计算。

**[讲解] 寄存器资源限制**

这里的关键限制是**寄存器**：

- H100 SM 有 65,536 个 32 位寄存器
- 输出矩阵的大小是 64×512 = 32,768 个元素
- WGMMA（warpgroup 矩阵乘加指令）要求操作数在寄存器中
- 所以一个输出矩阵用掉了一半的寄存器

FlashAttention-3 的 ping-pong 调度需要两个输出矩阵（相当于两倍寄存器），这在 H100 上做不到。传统的 ping-pong 方案是：Tensor Core 在处理矩阵 A 时，CUDA Core 在处理矩阵 B。但这里只有一个输出矩阵的寄存器空间。

**原文：**

```markdown
(You might pause here to ponder - perhaps you can find a better solution than ours!)
```

> **翻译：** （你可以在这里停下来思考一下——也许你能找到比我们更好的解决方案！）

**原文：**

```markdown
Our solution involves an additional mathematical transformation beyond FlashAttention's online softmax
and accumulation approach. In each step, we take two KV blocks (called $K_0$, $K_1$, $V_0$, and $V_1$).
Since the output matrix occupies 32,768 registers (too many for one warpgroup), we split it vertically
into $O_L$ and $O_R$ (each $64 \times 256$). We similarly split $V_0$ and $V_1$ into $V_{0L}$, $V_{0R}$,
$V_{1L}$, and $V_{1R}$ (each $64 \times 256$). The output matrix is then computed as follows:
```

> **翻译：** 我们的解决方案涉及一个超越 FlashAttention 的 online softmax 和累积方法的额外数学变换。在每个步骤中，我们选取两个 KV 块（称为 $K_0$、$K_1$、$V_0$、$V_1$）。由于输出矩阵占用 32,768 个寄存器（对于一个 warpgroup 来说太多了），我们将其垂直拆分为 $O_L$ 和 $O_R$（各 $64 \times 256$）。我们同样将 $V_0$ 和 $V_1$ 拆分为 $V_{0L}$、$V_{0R}$、$V_{1L}$、$V_{1R}$（各 $64 \times 256$）。然后按如下方式计算输出矩阵：

**[讲解] Seesaw 调度的核心思想**

核心创新是把输出矩阵**竖着劈成两半**（左半边和右半边），分配到两个不同的 warpgroup 上：
- **Warpgroup 0** 负责左半边 $O_L$（64×256）
- **Warpgroup 1** 负责右半边 $O_R$（64×256）

每个 warpgroup 只负责 16,384 个寄存器（原来的一半），这样就有剩余寄存器做其他事情。

同时，V 矩阵也竖着劈成两半：$V_{0L}$、$V_{0R}$。这样 warpgroup 0 只和 $V_{0L}$ 交互，warpgroup 1 只和 $V_{0R}$ 交互——它们可以并行工作。

**原文：**

```markdown
0. Maintain a running max $m$ (initialized to $-\infty$, shared between the two warpgroups) and output
   matrices $\vec o_L, \vec o_R$ (initialized to 0).
1. [0] Compute $\vec p_0 = \vec q K_0^\intercal / qk\_scale$.
2. [1] Compute $\vec p_1 = \vec q K_1^\intercal / qk\_scale$.
3. [0] Compute $mp_0 = \max(\vec p_0)$, $m\_new_0 = \max(m, mp_0)$, and
       $scale_0 = \exp(m\_new_0 - m)$. Update $m \gets m\_new_0$.
4. [0] Perform softmax on $\vec p_0$: $\vec p_0 \gets \exp(\vec p_0 - m\_new_0)$.
5. [0] Update $\vec o_L \gets \vec o_L \cdot scale_0 + \vec p_0 V_{0L}$.
6. [1] Compute $mp_1 = \max(\vec p_1)$, $m\_new_1 = \max(m, mp_1)$, and
       $scale_1 = \exp(m\_new_1 - m)$. Update $m \gets m\_new_1$.
7. [1] Perform softmax on $\vec p_1$: $\vec p_1 \gets \exp(\vec p_1 - m\_new_1)$.
8. [1] Update $\vec o_R \gets \vec o_R \cdot (scale_0 \cdot scale_1) + \vec p_1 V_{1R}$.
9. [0] Update $\vec p_0 \gets \vec p_0 \cdot scale_1$.
10. [1] Update $\vec o_R \gets \vec o_R + \vec p_0 V_{0R}$.
11. [0] Update $\vec o_L \gets \vec o_L \cdot scale_1 + \vec p_1 V_{1L}$.
```

> **翻译：**
> 0. 维护一个全局的运行最大值 $m$（初始化为 $-\infty$，在两个 warpgroup 之间共享）和输出矩阵 $\vec o_L, \vec o_R$（初始化为 0）。
> 1. [warpgroup 0] 计算 $\vec p_0 = \vec q K_0^\intercal / qk\_scale$
> 2. [warpgroup 1] 计算 $\vec p_1 = \vec q K_1^\intercal / qk\_scale$
> 3. [warpgroup 0] 计算 $mp_0 = \max(\vec p_0)$，$m\_new_0 = \max(m, mp_0)$，$scale_0 = \exp(m\_new_0 - m)$。更新 $m \gets m\_new_0$。
> 4. [warpgroup 0] 对 $\vec p_0$ 执行 softmax：$\vec p_0 \gets \exp(\vec p_0 - m\_new_0)$
> 5. [warpgroup 0] 更新 $\vec o_L \gets \vec o_L \cdot scale_0 + \vec p_0 V_{0L}$
> 6. [warpgroup 1] 计算 $mp_1 = \max(\vec p_1)$，$m\_new_1 = \max(m, mp_1)$，$scale_1 = \exp(m\_new_1 - m)$。更新 $m \gets m\_new_1$。
> 7. [warpgroup 1] 对 $\vec p_1$ 执行 softmax：$\vec p_1 \gets \exp(\vec p_1 - m\_new_1)$
> 8. [warpgroup 1] 更新 $\vec o_R \gets \vec o_R \cdot (scale_0 \cdot scale_1) + \vec p_1 V_{1R}$
> 9. [warpgroup 0] 更新 $\vec p_0 \gets \vec p_0 \cdot scale_1$
> 10. [warpgroup 1] 更新 $\vec o_R \gets \vec o_R + \vec p_0 V_{0R}$
> 11. [warpgroup 0] 更新 $\vec o_L \gets \vec o_L \cdot scale_1 + \vec p_1 V_{1L}$

> **注释：** 为简单起见，我们假设只有一个 query head，所以 $\vec q$ 和 $\vec o$ 是向量。方括号中的数字表示执行操作的 warpgroup。假设 $\vec o_L$ 位于 warpgroup 0 的寄存器中，$\vec o_R$ 位于 warpgroup 1 的寄存器中。

**[讲解] Seesaw 调度详解**

这个调度被称为"**跷跷板（seesaw）调度**"——两个 warpgroup 交替执行操作，像跷跷板一样此起彼伏。

**核心挑战：** 两个 KV 块（K₀、K₁）的 attention score（p₀、p₁）的 softmax 是彼此依赖的。p₁ 的 max 可能比 p₀ 大，这会改变 p₀ 的 softmax 值。标准做法是 wait-and-see——等两个 score 都算出来再一起做 softmax——但这会引入同步开销。

**Seesaw 调度如何解决：** 它允许两个 warpgroup 共享一个全局的 $m$（running max），通过**逐步更新** $m$ 来处理依赖：

1. WG0 先处理 K₀：算 score、算 local max（mp₀）、更新全局 max、做局部 softmax、更新 O_L（用当前的 $scale_0$）
2. WG1 处理 K₁：算 score、算 local max（mp₁）、发现 mp₁ 更大 → 更新全局 max 为 mp₁、重新计算 $scale_1$
3. **关键步骤 8-11：** 因为全局 max 从 mp₀ 变成了 mp₁，之前 WG0 对 p₀ 做的 softmax 是基于旧 max 的，需要**修正**。这就是步骤 9 做的——用 $scale_1$ 去缩放 p₀。同时步骤 8、11 用 $scale_1$ 去缩放 O_R 和 O_L。

**直觉类比：** 想象你和朋友合买彩票。你算出中了 100 块（mp₀），朋友算出中了 200 块（mp₁）。实际所有人中了 200 块。你之前算的"你的奖金分成"需要根据新的最大值重新调整。

**这种调度的优势：**
- Tensor Core（做 GEMM 矩阵乘）和 CUDA Core（做 max/softmax/rescale）的工作在两个 warpgroup 之间自然交替，避免了等待
- 每个 warpgroup 的寄存器占用量减半，留出了做其他优化的空间
- 不需要 FlashAttention-3 的双缓冲区 ping-pong 调度

**原文：**

```markdown
This schedule can be viewed as a "ping-pong" variant using one output matrix—we call it "seesaw"
scheduling. It's mathematically equivalent to FlashAttention's online softmax algorithm. This schedule
allows us to overlap CUDA Core and Tensor Core operations by interleaving the two warpgroups,
and also allows us to overlap memory access with computation since we can launch the corresponding
Tensor Memory Accelerator (TMA) instructions right after data is no longer needed.
```

> **翻译：** 这种调度可以看作是一种使用单个输出矩阵的"乒乓"变体——我们称之为"跷跷板"调度。它在数学上等价于 FlashAttention 的 online softmax 算法。这种调度允许我们通过交错两个 warpgroup 来重叠 CUDA Core 和 Tensor Core 操作，同时还允许我们将内存访问与计算重叠，因为我们可以在数据不再需要时立即启动相应的 TMA 指令。

**原文：**

```markdown
The complete schedule is shown below (remember that in MLA, $K$ and $V$ are the same with different names):
```

> **翻译：** 完整的调度如下图所示（记住在 MLA 中，K 和 V 是同一个东西的不同名称）：

**[讲解] "K 和 V 是同一个东西"是什么意思？**

MLA（Multi-head Latent Attention）中，K 和 V 是从同一个低维潜在表示投影出来的。在实际计算中，K 和 V 共享相同的底层数据，只是计算注意力分数时用 K 的角色，计算加权求和时用 V 的角色。这就是 MLA 节省显存的核心——不需要分别缓存 K 和 V，只需要缓存一个低维的 latent 表示。

**原文附图：**

```
![MLA Kernel Sched](assets/MLA%20Kernel%20Sched.drawio.svg)
```

> **翻译：** 图片名称为"MLA Kernel 调度图"，是一个 SVG 示意图，展示了 seesaw 调度的时间序列。

**[讲解] 图片说明**

图片在 `docs/assets/MLA Kernel Sched.drawio.svg`，建议在浏览器中打开查看。图中应该展示了：
- 两个 warpgroup（WG0、WG1）在同一 timeline 上交替执行的操作
- GEMM（矩阵乘）和 softmax/CUDA Core 操作的交错
- TMA 数据拷贝操作的时机
- 每一步的依赖关系

---

### 第四段：技术细节讨论

**原文：**

```markdown
## Discussion of Technical Details
```

> **翻译：** 技术细节讨论

**原文：**

```markdown
This section covers technical details of the new kernel.
```

> **翻译：** 本节介绍新内核的技术细节。

**原文：**

```markdown
First, although the kernel targets compute-bound scenarios (where memory bandwidth isn't the bottleneck),
we can't ignore memory latency. If the data is not ready when we want to use it, we have to wait.
To solve this problem, we employ the following techniques:
```

> **翻译：** 首先，虽然内核针对的是计算密集型场景（带宽不是瓶颈），但我们仍然不能忽略内存延迟。如果数据在我们想要使用时还没有准备好，我们就必须等待。为了解决这个问题，我们采用了以下技术：

**[讲解] 带宽不是瓶颈 ≠ 延迟可以忽略**

这是一个重要的概念区分：
- **内存带宽（Bandwidth）：** 数据搬移的"吞吐量"——单位时间能搬多少数据。在 compute-bound 场景下，这是够用的。
- **内存延迟（Latency）：** 一次数据请求到数据就绪的"响应时间"。即使带宽够用，但如果你请求数据后必须等几百个周期才能用上，计算单元还是会空闲。

所以优化目标变成了：**尽可能提前发起数据请求**，让数据在需要时就绪。"隐藏延迟"是 GPU 优化中永恒的主题。

**原文：**

```markdown
- **Fine-grained TMA copy - GEMM pipelining:** For a $64 \times 576$ K block, we launch 9 TMA copies
  (each moving a $64 \times 64$ block). GEMM operations begin as soon as each TMA copy completes
  (When the first TMA copy is done, we can start the first GEMM operation, and so on), improving
  memory latency tolerance.
- **Cache hints:** Using `cute::TMA::CacheHintSm90::EVICT_FIRST` for TMA copies improves L2 cache hit
  rates, as shown by experiments.
```

> **翻译：**
> - **细粒度 TMA 拷贝 - GEMM 流水线化：** 对于一个 $64 \times 576$ 的 K 块，我们启动 9 个 TMA 拷贝（每个搬运一个 $64 \times 64$ 的块）。GEMM 操作在每次 TMA 拷贝完成时立即开始（当第一个 TMA 拷贝完成后，我们就可以开始第一个 GEMM 操作，以此类推），提高了对内存延迟的容忍度。
> - **缓存提示：** 对 TMA 拷贝使用 `cute::TMA::CacheHintSm90::EVICT_FIRST` 可以提高 L2 缓存命中率，实验已证实这一点。

**[讲解] 细粒度 TMA pipelining**

传统做法：等一整块 K（64×576）全部从显存搬到共享内存，再开始矩阵乘——等待时间长。

FlashMLA 的做法：把 K 切成 9 个 64×64 的小块（576 ÷ 64 = 9）。每搬完一个小块就立即开始计算对应的部分矩阵乘。这样：
- 第 1 个小块搬完 → 开始计算第 1 个部分乘
- 第 2 个小块搬完 → 开始计算第 2 个部分乘（同时第 1 个可能还在算）
- ...

结果：等待被"摊平"到整个计算过程中，而不是集中在开始阶段。这也利用了 TMA（Tensor Memory Accelerator）的异步拷贝能力——计算和拷贝可以同时进行。

**[讲解] 缓存提示 EVICT_FIRST**

`EVICT_FIRST` 是 CUTLASS 中的一个缓存策略，告诉 GPU 的 L2 缓存："这块数据用完就可以优先驱逐，不用留着。"TMA 拷贝操作涉及 KV Cache 数据——这些数据量非常大（整个 KV Cache 可能几十 GB），即使 L2 缓存有 50MB 也装不下。与其让旧数据占着 L2 缓存浪费空间，不如主动标记可驱逐，让更重要的数据（如 Q、中间结果）有更多缓存空间。

**原文：**

```markdown
These optimizations achieve up to 80% Tensor Core utilization (of the throttled theoretical peak) and
3 TB/s memory bandwidth on an H800 SXM5 GPU. While slightly slower (~2%) than the old ping-pong
buffer version in memory-bound settings, this is acceptable.
```

> **翻译：** 这些优化在 H800 SXM5 GPU 上实现了高达 80% 的 Tensor Core 利用率（相对于降频后的理论峰值）和 3 TB/s 的显存带宽。虽然在访存密集型配置下比老版本的 ping-pong 缓冲区版本略慢（约 2%），但这是可以接受的。

**[讲解] 这个数字意味着什么？**

- **80% Tensor Core 利用率：** Tensor Core 的理论峰值是 865 TFlops（降频后），80% 利用率 ≈ 690 TFlops。实际报告 660 TFlops（考虑到其他开销），非常接近理论极限。
- **3 TB/s：** H800 的理论带宽是 3.35 TB/s，达到了约 90% 的带宽利用率。
- **"比老版本慢 2%"的权衡：** 在 memory-bound 配置下（h_q × s_q 较小），新内核因为 seesaw 调度有额外的管理开销，所以比旧版本的 ping-pong 缓冲区方案略慢。但这不是重点——新内核的主要优化是针对 compute-bound 场景的。

**原文：**

```markdown
Other performance improvements include:
- **Programmatic Dependent Launch.** We use programmatic dependent launch to overlap `splitkv_mla`
  and `combine` kernels.
- **Tile Scheduler.** We implement a tile scheduler to allocate jobs (requests and blocks) to SMs.
  This ensures a balanced load across SMs.
```

> **翻译：** 其他性能改进包括：
> - **程序化依赖启动：** 我们使用程序化依赖启动来重叠 `splitkv_mla` 和 `combine` 内核的执行。
> - **Tile 调度器：** 我们实现了一个 tile 调度器来将任务（请求和块）分配给 SMs，确保 SM 之间的负载均衡。

**[讲解] 程序化依赖启动（Programmatic Dependent Launch）**

这是 Hopper 架构引入的特性，允许一个 GPU kernel 在另一个 kernel 完成后立即自动启动，**不需要 CPU 介入**。传统做法是 CPU 发射 kernel A → CPU 等待完成 → CPU 发射 kernel B，每次切换都有 CPU-GPU 通信延迟。

在 FlashMLA 中，`splitkv_mla`（分段计算注意力）和 `combine`（合并结果）是两个 kernel。使用 PDL 后，combine kernel 在 splitkv_mla 完成后自动在 GPU 上启动，省去了 CPU-GPU 之间的"往返"延迟。

**[讲解] Tile 调度器**

在解码阶段，每个 SM 需要处理多个请求和多个 KV 块。不同请求的序列长度不同，导致计算量不同。如果分配不均匀，有些 SM 忙死、有些 SM 闲死（load imbalance）。

Tile scheduler 的作用就是：**预先计算好每个 SM 应该处理哪些请求的哪些 KV 块**，确保负载均衡。这也是 `FlashMLASchedMeta` 中 `tile_scheduler_metadata` 和 `num_splits` 的来源。

---

### 第五段：致谢与引用

**原文：**

```markdown
## Acknowledgements

FlashMLA's algorithm and scheduling are inspired by FlashAttention, Flash-Decoding, and CUTLASS,
as well as many projects behind them. We thank the authors for their great work.
```

> **翻译：** FlashMLA 的算法和调度受到 FlashAttention、Flash-Decoding 和 CUTLASS 以及它们背后的许多项目的启发。我们感谢作者的杰出工作。

**原文：**

```markdown
## Citation

@misc{flashmla2025,
      title={FlashMLA: Efficient MLA decoding kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```

> **翻译：** 引用格式如上。

---

## 三、总结：这篇文章的关键技术点

| 技术点 | 解决的问题 | 具体方法 |
|--------|-----------|---------|
| **Compute-bound 分析** | 判断优化方向 | 用 roof-line 模型计算 FLOPs/memory 比，确定瓶颈 |
| **Seesaw 调度** | 寄存器不足，无法用 ping-pong | 竖劈输出矩阵，两个 warpgroup 交替执行 |
| **Online softmax 修正** | 两个 KV 块的 softmax 依赖 | 全局 running max + 滞后修正（scale 调整） |
| **细粒度 TMA pipelining** | 内存延迟 | 64×64 粒度异步拷贝，做到就开算 |
| **缓存提示 EVICT_FIRST** | 大 KV Cache 污染 L2 | 标记可驱逐，提高命中率 |
| **程序化依赖启动** | kernel launch 延迟 | GPU 自动启动下一个 kernel |
| **Tile 调度器** | SM 负载不均 | 预分配请求和块到各个 SM |
