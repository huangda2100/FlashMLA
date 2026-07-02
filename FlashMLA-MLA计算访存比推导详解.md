# MLA 计算访存比推导详解：$\frac{\text{FLOPs}}{\text{Memory}} \approx 2 h_q s_q$

> 这篇文档回答一个问题：**DeepSeek 官方博客里那句 $\frac{\text{FLOPs}}{\text{Memory}} \approx 2 h_q s_q$ 是怎么推出来的？每个符号、每个数字都对得上什么？**
>
> 面向零基础读者。会矩阵乘就行，不需要懂 GPU、不需要懂注意力。

---

## 一、问题背景：为什么需要这个推导？

DeepSeek 在 FlashMLA 博客里要回答一个问题：**MLA 解码内核到底是"算力不够"还是"带宽不够"？**

判断方法叫 **roof-line 分析**：

> 算一下内核的"计算-访存比"，然后和 GPU 的"平衡点"比一比。
>
> - 比值高于平衡点 → 算力不够 → **计算密集型**
> - 比值低于平衡点 → 带宽不够 → **访存密集型**

**这个推导就是在算 MLA 解码内核的"计算-访存比"**。推出来的结果是 $2 h_q s_q$——一个非常简洁的公式，可以直接读出哪些场景是计算密集型。

---

## 二、先认识 5 个符号

| 符号 | 含义 | 通俗解释 |
|------|------|---------|
| $h_q$ | query head 数量 | "问答"里"问答者"的数量。每个 head 都是一个独立的注意力计算单元 |
| $s_q$ | 每个请求的 query token 数 | 这次问几个 token。普通解码 = 1，开推测解码/MTP 可能 > 1 |
| $s_k$ | KV Cache 的 token 数 | 历史记录有多长。解码时这个很大（几千~几万） |
| $d_k$ | K 的 head 维度 | 每个 K 向量多少维 |
| $d_v$ | V 的 head 维度 | 每个 V 向量多少维（MLA 中一般 $d_k = d_v$） |

**MLA 解码的典型配置**（DeepSeek V3）：
- $h_q = 128$（不用 Tensor Parallel，128 个 head 全在一张卡上）
- $s_q = 1$（关掉推测解码）
- $s_k$ = 几千到几万
- $d_k = d_v = 512$（MLA 的 latent 维度）

**关键前提**：$s_k \gg h_q s_q$。即"KV Cache 长度远大于 query 数量"。这个前提在解码时几乎总成立，后面化简会用到。

---

## 三、MLA 解码在算什么？—— 注意力的两个矩阵乘

注意力计算的标准公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

中间虽然还有 softmax、scale 等操作，但**主要计算量来自两个矩阵乘**：

```
步骤 1: P = Q × K^T     →  得到 score 矩阵 P
步骤 2: O = P × V       →  得到最终输出 O
```

用图形表示（省略 head 维度 $d$，以单个 head 为例）：

```
       K^T              V
       (s_k × d)        (s_k × d)
        ↑                ↑
Q ──────┼──→ P           │
(s_q×d) │  (s_q×s_k)     │
        │                │
        └────────────────┴──→ O
                            (s_q×d)
```

**这是化简的核心骨架：两个矩阵乘。** 后面所有推导都从这两个乘法来。

---

## 四、第一步：算 FLOPs（计算量）

### 4.1 矩阵乘的 FLOPs 公式

矩阵乘 `C = A @ B`，其中 A 是 $M \times K$，B 是 $K \times N$：

- 结果 C 有 $M \times N$ 个元素
- 每个元素要算 $K$ 次乘法 + $K-1$ 次加法 ≈ $2K$ 次浮点运算
- 总 FLOPs = $M \times N \times 2K = 2 M K N$

**口诀**：`2 × 三个维度的乘积`。

### 4.2 把两个矩阵乘的 FLOPs 算出来

**第一个矩阵乘：$P = Q \times K^\top$**

Q 的形状：$h_q \times s_q \times d_k$（head 数 × query 数 × head 维）
$K^\top$ 的形状：$h_q \times d_k \times s_k$（head 数 × head 维 × KV 数）

把 head 数 $h_q$ 当成 batch 维度，单个 head 的矩阵乘是：
- $M = s_q$，$K = d_k$，$N = s_k$
- 单 head FLOPs = $2 \cdot s_q \cdot d_k \cdot s_k$
- 所有 head 总 FLOPs = $h_q \cdot 2 s_q d_k s_k = 2 h_q s_q d_k s_k$

**第二个矩阵乘：$O = P \times V$**

P 的形状：$h_q \times s_q \times s_k$
V 的形状：$h_q \times s_k \times d_v$

单 head：
- $M = s_q$，$K = s_k$，$N = d_v$
- 单 head FLOPs = $2 \cdot s_q \cdot s_k \cdot d_v$
- 所有 head 总 FLOPs = $h_q \cdot 2 s_q s_k d_v = 2 h_q s_q s_k d_v$

### 4.3 两个加起来

$$
\text{FLOPs} = \underbrace{2 h_q s_q d_k s_k}_{\text{Q×K}^\top} + \underbrace{2 h_q s_q s_k d_v}_{\text{P×V}} = 2 h_q s_q s_k (d_k + d_v)
$$

**这就是原文里的公式**：$2 (h_q s_q \cdot d_k \cdot s_k + h_q s_q \cdot s_k \cdot d_v) = 2 h_q s_q s_k (d_k + d_v)$

提取公因子 $h_q s_q s_k$ 后剩 $(d_k + d_v)$，把"乘法分配律"反过来用而已。

### 4.4 直觉理解

> **算力消耗 = 2 × head 数 × query 数 × KV 数 × (K维度 + V维度)**

每个 query token 都要和每个 KV token 算一次相似度（Q×K），再用这个相似度加权 V（P×V）。所以**计算量正比于 $h_q \times s_q \times s_k$**——head 越多、query 越多、KV 越长，算得越多。

---

## 五、第二步：算访存量（Memory，单位 bytes）

### 5.1 哪些张量要从显存搬到 SM？

GPU 计算前，输入数据必须从显存（HBM）搬到计算单元（SM 的共享内存/寄存器）。要搬的张量：

| 张量 | 形状 | 搬运字节数（BF16 = 2 字节/元素） | 说明 |
|------|------|-------------------------------|------|
| Q | $h_q \times s_q \times d_k$ | $2 \cdot h_q s_q d_k$ | 读 |
| K（latent） | $1 \times s_k \times d_k$ | $2 \cdot s_k d_k$ | 读，$h_k=1$ |
| V | — | **0** | **不单独搬运**，从 latent 在 SM 内投影出来 |
| O（输出） | $h_q \times s_q \times d_v$ | $2 \cdot h_q s_q d_v$ | 写回显存 |

> **MLA 的关键特性**：K 和 V 共享同一个 latent 表示，所以**从显存搬运时只搬一份 latent**（习惯上叫 `k_cache`，但实际是 K 和 V 共享的底层数据）。V 是在 SM 内部从 latent 投影出来的——这是**计算**（一次矩阵乘），不是**访存**。
>
> 所以"V 的搬运字节数 = 0"——它已经被 latent 那一项算过了。

总访存量（字节）：

$$
\text{Memory} = 2 \cdot (\underbrace{h_q s_q d_k}_{\text{Q 读}} + \underbrace{s_k d_k}_{\text{KV Cache 读（latent）}} + \underbrace{h_q s_q d_v}_{\text{O 写}})
$$

> **重要澄清**：博客原文公式 $\text{sizeof}(\text{bfloat16}) \times (h_q s_q d_k + s_k d_k + h_q s_q d_v)$ 里的第三项 $h_q s_q d_v$ 是 **O 的写入量**，不是 V 的搬运量。V 在 MLA 中不单独搬运。
>
> 这一块容易混淆——$h_q s_q d_v$ 的形状对应 $h_q \times s_q \times d_v$，正好是 O 的形状，而不是 V 的存储形状（V 在 MLA 里和 K 共享 $s_k \times d_k$ 的 latent）。详见 [FlashMLA-KVCache一体与搬运量详解.md](FlashMLA-KVCache一体与搬运量详解.md)。

### 5.2 关键化简：$s_k \gg h_q s_q$

回到那个前提：**$s_k \gg h_q s_q$**（KV Cache 远长于 query 数量）。

举几个数字感受一下：
- $s_k = 4096$（典型短对话）
- $h_q s_q = 128 \times 1 = 128$（DeepSeek V3 解码）
- $s_k / (h_q s_q) = 32$ 倍

把这个比例代入访存量公式：

| 项 | 含义 | 值 | 量级（以 $s_k = 4096, h_q s_q = 128, d_k = d_v = 512$ 估算） |
|---|------|----|------|
| $h_q s_q d_k$ | Q 读 | $128 \cdot 512$ | 65,536 |
| $s_k d_k$ | KV Cache 读（latent，K/V 共享） | $4096 \cdot 512$ | **2,097,152** ← 主导项 |
| $h_q s_q d_v$ | O 写 | $128 \cdot 512$ | 65,536 |
| V 搬运 | （已含在 latent 里） | 0 | 0 |

**KV Cache 的搬运量是其他项的 32 倍**，所以可以近似为：

$$
\text{Memory} \approx 2 \cdot s_k d_k \quad \text{（BF16）}
$$

**这就是原文的近似**：$\text{sizeof}(\text{bfloat16}) \times (\ldots) \approx 2 s_k d_k$。前面的 2 是 BF16 每元素 2 字节，$s_k d_k$ 是 KV Cache 的元素数。

### 5.3 直觉理解

> **访存消耗 ≈ 2 × KV Cache 长度 × K 维度**

解码时 Q 只有 1 个 token，输出也只有一个 token，都很小。**几乎所有的访存都在搬 KV Cache**——这就是为什么大家总觉得解码是"访存密集型"。

---

## 六、第三步：算比值

把前两步的结果相除：

$$
\frac{\text{FLOPs}}{\text{Memory}} = \frac{2 h_q s_q s_k (d_k + d_v)}{2 s_k d_k}
$$

### 6.1 逐步化简

**第 1 步**：分子分母约掉 2：

$$
= \frac{h_q s_q s_k (d_k + d_v)}{s_k d_k}
$$

**第 2 步**：分子分母约掉 $s_k$：

$$
= \frac{h_q s_q (d_k + d_v)}{d_k}
$$

**第 3 步**：拆开括号：

$$
= h_q s_q \cdot \frac{d_k + d_v}{d_k}
$$

**第 4 步**：MLA 中 $d_k = d_v$（K 和 V 共享同一个 latent 表示，维度相同），所以 $\frac{d_k + d_v}{d_k} = \frac{2 d_k}{d_k} = 2$：

$$
= h_q s_q \cdot 2 = 2 h_q s_q
$$

### 6.2 最终结果

$$
\boxed{\frac{\text{FLOPs}}{\text{Memory}} \approx 2 h_q s_q}
$$

**单位**：FLOPs/byte（每搬 1 字节做多少次浮点运算）。

---

## 七、这个公式在说什么？（直觉解释）

把公式拆开看：

### 7.1 分子里的 2：来自两个矩阵乘

注意力有两次 GEMM：Q×K 和 P×V。每次 GEMM 的 FLOPs 系数都是 2（乘加各一次），但**只有一次 KV 搬运**（K 和 V 共享 latent）。

所以分子比分母**多了一个 2**——这个 2 是"两次矩阵乘对一次搬运"的倍数。

### 7.2 $h_q s_q$：来自"多个 head × 多个 query"的复用

每个 KV 元素被**所有 head 和所有 query 共享**——它们都要用同一份 KV Cache 算注意力。

- $h_q$ 个 head 都要用同一个 KV → 计算量 ×$h_q$，搬运量不增
- $s_q$ 个 query 都要用同一个 KV → 计算量 ×$s_q$，搬运量不增

所以比值正比于 $h_q s_q$：**共享 KV 的"消费者"越多，算得越多但搬得不变，比值越高**。

### 7.3 $s_k$ 和 $d_k$ 都消失了：因为它们在分子分母同阶

- 增大 $s_k$：FLOPs 涨 $s_k$ 倍，搬运也涨 $s_k$ 倍，比值不变
- 增大 $d_k$：FLOPs 涨 $d_k$，搬运也涨 $d_k$，比值不变

**直觉**：KV Cache 越长或维度越大，"算的活"和"搬的活"**同比例增加**，所以算-搬比不变。

这就是 MLA 解码算-搬比公式这么简洁的根本原因——**KV 相关的维度都被约掉了**。

---

## 八、代入数字验证

DeepSeek V3 解码配置：$h_q = 128, s_q = 1, d_k = d_v = 512, s_k = 4096$。

### 8.1 算 FLOPs

$$
\text{FLOPs} = 2 \cdot 128 \cdot 1 \cdot 4096 \cdot (512 + 512) = 2 \cdot 128 \cdot 4096 \cdot 1024
$$

$= 2 \cdot 128 \cdot 4096 \cdot 1024 = 1,073,741,824 \approx 1.07 \times 10^9$ FLOPs

### 8.2 算 Memory

$$
\text{Memory} \approx 2 \cdot s_k d_k = 2 \cdot 4096 \cdot 512 = 4,194,304 \text{ bytes} \approx 4.19 \text{ MB}
$$

### 8.3 算比值

$$
\frac{\text{FLOPs}}{\text{Memory}} = \frac{1.07 \times 10^9}{4.19 \times 10^6} \approx 256 \text{ FLOPs/byte}
$$

### 8.4 用简化公式

$$
2 h_q s_q = 2 \cdot 128 \cdot 1 = 256 \text{ FLOPs/byte}
$$

**两个结果一致**，验证了简化公式 $2 h_q s_q$ 的正确性。

---

## 九、这个比值如何用来判断 compute-bound？

### 9.1 GPU 的平衡点

H800（降频后）：
- 算力 = 865 TFlops = $865 \times 10^{12}$ FLOPs/s
- 带宽 = 3.35 TB/s = $3.35 \times 10^{12}$ bytes/s
- **平衡点** = $865 / 3.35 \approx 258$ FLOPs/byte

考虑注意力中 softmax 不是矩阵乘的损失，经验上加 1/2 因子：
- **实际门槛** ≈ 129 FLOPs/byte

### 9.2 比较

$$
\frac{\text{FLOPs}}{\text{Memory}} = 256 \quad \text{vs} \quad \text{门槛} = 129
$$

$$
256 \ge 129 \quad \Rightarrow \quad \text{compute-bound} \checkmark
$$

### 9.3 临界点

让 $2 h_q s_q = 129$：

$$
h_q s_q = 64.5 \approx 64
$$

也就是说：
- $h_q s_q \ge 65$ → MLA 解码是计算密集型
- $h_q s_q \le 64$ → MLA 解码是访存密集型

DeepSeek V3 的 $h_q s_q = 128$ 远超 65，所以是稳定的计算密集型。

**原文写的临界值是 $h_q s_q \ge 128$，对应于"不加 1/2 因子"的理论平衡点 258——保守一点的判断。** 两种说法都对，差别只在用了哪个门槛。

---

## 十、为什么这个推导重要？

### 10.1 它决定了所有后续优化方向

| 比值判断 | 优化目标 | FlashMLA 的具体做法 |
|---------|---------|---------------------|
| **计算密集型**（实际） | 让 Tensor Core 满载 | Seesaw 调度、warpgroup 交错、TMA pipelining |
| 访存密集型（假设） | 提高带宽利用率 | 增大 L2 命中率、减少搬运、用更紧凑的数据格式 |

**如果一开始判断错了，所有优化都白做**。这就是为什么 DeepSeek 在博客开头花一大段做这个推导——它是所有后续技术决策的基石。

### 10.2 它解释了"反直觉"的现象

普通认知：解码 = 访存密集型（Q 太小，主要在搬 KV Cache）。

MLA 的反直觉：因为 $h_q = 128$ 个 head 共享同一份 KV Cache，每个 head 都要算一遍 → 计算量放大 128 倍，搬运量不变 → 比值顶过平衡点。

**这个推导把"反直觉"量化了**：$h_q s_q = 128$ 直接代入 $2 h_q s_q$ 得 256，一眼能看出超过门槛 129。

### 10.3 它给出了"什么时候 MLA 会变访存密集型"

从公式 $2 h_q s_q$ 直接读出：

- **用 Tensor Parallel**：$h_q$ 被 TP 切分到多卡，单卡 $h_q$ 变小 → 比值变小 → 可能变访存密集型
- **开推测解码/MTP**：$s_q$ 变大（一次算多个 token）→ 比值变大 → 更计算密集
- **小模型/少 head**：$h_q$ 小 → 比值小 → 访存密集型

**一个公式预测了三种场景的瓶颈位置**——这就是数学推导的价值。

---

## 十一、常见误区与 FAQ

### Q1：为什么 K 和 V 共享同一份内存？

这是 MLA（Multi-head Latent Attention）的核心创新。传统 attention 里 K 和 V 是分别缓存的，占两份显存。MLA 把 K 和 V 都从一个低维 latent 向量投影出来，所以只缓存 latent → 实际存储只有一份。

这个推导里 $d_k = d_v$、访存只算一份 $s_k d_k$，都来源于此。如果换成传统 MHA（K 和 V 分开缓存），推导会不一样。

### Q2：softmax 的 FLOPs 去哪了？

严格说注意力里除了两个 GEMM，还有 softmax（指数、求和、归一化）。这些不是矩阵乘，Tensor Core 帮不上忙。

但 DeepSeek 在推导里**故意忽略了 softmax 的 FLOPs**，原因有二：
1. softmax 的 FLOPs 远小于两个 GEMM（GEMM 是 $O(s_q \cdot s_k \cdot d)$，softmax 是 $O(s_q \cdot s_k)$）
2. 推导的目的是判断**算力 vs 带宽**的瓶颈位置，softmax 的非矩阵运算部分让算力"打折"，这通过 1/2 因子近似处理（平衡点 258 → 门槛 129）

### Q3：输出 O 的写入为什么被忽略了？

O 的形状是 $h_q \times s_q \times d_v$，写入量是 $h_q s_q d_v$ 字节。在 $s_k \gg h_q s_q$ 的前提下：

$$
\frac{h_q s_q d_v}{s_k d_k} = \frac{h_q s_q}{s_k} \cdot \frac{d_v}{d_k} \ll 1
$$

所以 O 的写入量在总访存中占比很小，可以忽略。但在 $s_k$ 不大（比如预填充阶段）时这个近似就不成立了——这也是为什么这个推导只适用于解码阶段。

### Q4：BF16 的 2 字节哪里来的？

BF16（Brain Float 16）是 16 位浮点数 = 2 字节。原文里 $\text{sizeof}(\text{bfloat16}) = 2$。

如果用 FP8（1 字节），分母会减半，比值会翻倍——这是为什么 FP8 能把更多 kernel 推向计算密集型。FlashMLA 的 FP8 版本就是利用这一点。

### Q5：$d_k = d_v$ 总是成立吗？

在 MLA 中成立（K 和 V 同源），但在传统 MHA、GQA、MQA 中**不一定**。如果 $d_k \neq d_v$，公式变成：

$$
\frac{\text{FLOPs}}{\text{Memory}} \approx h_q s_q \cdot \frac{d_k + d_v}{d_k}
$$

无法进一步化简为 $2 h_q s_q$。MLA 的"对称性"（$d_k = d_v$）是公式简洁的原因。

### Q6：为什么不用 FP32 算 FLOPs？

GPU 的 Tensor Core 主要加速低精度（BF16/FP16/FP8）。FP32 计算慢 4-16 倍，现代大模型推理几乎不用 FP32。访存也是按 BF16 算——BF16 是大模型推理的"默认精度"。

---

## 十二、推导流程图（一图回顾）

```
   ┌─────────────────────────────────────────────────────┐
   │  MLA 注意力的两个矩阵乘：                             │
   │    P = Q × K^T    和    O = P × V                    │
   └─────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                            ▼
   ┌────────────────────┐      ┌──────────────────────┐
   │  FLOPs (计算量)     │      │  Memory (访存量)      │
   │                    │      │                      │
   │  Q×K^T: 2·hq·sq·dk·sk │      │  Q:        hq·sq·dk  │
   │  P×V:   2·hq·sq·sk·dv │      │  K (latent): sk·dk  │ ← 主导项（K/V 共享）
   │  合计:               │      │  V:         0        │ ← 从 latent 投影，不单独搬运
   │  2·hq·sq·sk·(dk+dv) │      │  O (写入): hq·sq·dv  │
   └────────────────────┘      │  合计 (BF16):         │
                               │  ≈ 2·sk·dk            │
                               └──────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │  比值 = FLOPs / Memory                                │
   │      = 2·hq·sq·sk·(dk+dv) / (2·sk·dk)                │
   │      = hq·sq·(dk+dv)/dk        ← 约掉 2 和 sk        │
   │      = hq·sq·2                 ← MLA 中 dk=dv        │
   │      = 2·hq·sq                                        │
   └─────────────────────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │  代入 hq=128, sq=1:                                   │
   │    比值 = 2·128·1 = 256 FLOPs/byte                   │
   │                                                       │
   │  H800 门槛（含 1/2 因子）= 129 FLOPs/byte             │
   │                                                       │
   │  256 > 129  →  compute-bound ✓                       │
   └─────────────────────────────────────────────────────┘
```

---

## 十三、一句话总结

> **公式 $\frac{\text{FLOPs}}{\text{Memory}} \approx 2 h_q s_q$ 的含义：**
>
> - **2** = 两个矩阵乘（Q×K 和 P×V）共用一份 KV 搬运
> - **$h_q s_q$** = 多个 head × 多个 query 共享同一份 KV Cache，每多一个消费者就多一份计算但搬运不变
> - **$s_k$ 和 $d_k$ 被约掉了** = KV 长度和维度的变化对算-搬比无影响（算的活和搬的活同比例增减）
> - **MLA 中 $d_k = d_v$** 让公式进一步简化为 $2 h_q s_q$
>
> 代入 DeepSeek V3 解码配置 $h_q = 128, s_q = 1$ 得 256 FLOPs/byte，超过 H800 门槛 129 → compute-bound。

---

## 附：关键公式速查

| 公式 | 含义 |
|------|------|
| $\text{FLOPs} = 2 h_q s_q s_k (d_k + d_v)$ | 注意力的总浮点运算量 |
| $\text{Memory} \approx 2 s_k d_k$ | BF16 下 KV Cache 主导的访存量 |
| $\frac{\text{FLOPs}}{\text{Memory}} \approx 2 h_q s_q$ | MLA 解码的计算-访存比（MLA 中 $d_k = d_v$） |
| $\text{平衡点} = \frac{\text{GPU算力}}{\text{GPU带宽}}$ | H800 上 ≈ 258 FLOPs/byte |
| $\text{门槛} = \frac{1}{2} \cdot \text{平衡点}$ | 含 softmax 损失，≈ 129 FLOPs/byte |

---

## 附：进一步阅读

- [FlashMLA-计算访存比详解.md](FlashMLA-计算访存比详解.md) —— 计算访存比的入门讲解
- [FlashMLA-Kernel深度解读.md](FlashMLA-Kernel深度解读.md) —— FlashMLA 整体技术解读
- 原博客：`docs/20250422-new-kernel-deep-dive.md`
