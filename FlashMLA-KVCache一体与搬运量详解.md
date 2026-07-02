# MLA 里 K 和 V 是一体的，为什么搬运量要分开算？

> 这篇文档回答一个非常尖锐的问题：
>
> **"为什么算访存量时，K 的搬运字节数是 $2 \cdot s_k d_k$，而 V 的搬运字节数是 $h_q s_q d_v$？两者差了一个 $h_q s_q$ 因子。我们常说的 KV Cache，K 和 V 不是一体的吗？"**
>
> 这个问题里藏着一个常见的误解。本文把它彻底讲清楚。

---

## 一、先把误解钉死：博客原文公式是什么？

### 1.1 用户问题的前提有误

用户问：
> K 的搬运字节数是 $2 \cdot s_k d_k$，V 的搬运字节数是 $2 \cdot h_q s_q d_v$

**但 DeepSeek 博客原文的公式不是这样的**。原文写的是：

$$
\text{Memory} = \text{sizeof}(\text{bfloat16}) \times \underbrace{(h_q s_q d_k}_{\text{Q}} + \underbrace{s_k d_k}_{\text{K}} + \underbrace{h_q s_q d_v}_{\text{V 或 O}})
$$

注意：
- **没有 $2 \cdot h_q s_q d_v$ 这一项**（前面没有 2）
- $h_q s_q d_k$ 这一项是 **Q** 的搬运量，**不是 K**
- K 的搬运量是 $s_k d_k$，前面乘 sizeof(bfloat16)=2 得到 $2 s_k d_k$ 字节
- V 的搬运量在原文公式里没显式出现——它被 $s_k d_k$ **吸收了**（原因下面讲）

### 1.2 用户混淆了什么？

用户的问题里其实把三件事搅在一起了：

1. **Q 的搬运量** $h_q s_q d_k$ —— 被误读成 V 的搬运量
2. **K 的搬运量** $s_k d_k$ —— 这个是对的
3. **V 的搬运量** —— 实际上和 K 共享同一块内存，所以是 0（或者说已经被 K 那一项算过了）
4. **O 的写入量** $h_q s_q d_v$ —— 这个量级和 Q 相同

**真正的问题应该是**：

> K 和 V 共享同一份存储（KV Cache），那 V 的搬运是不是就不用算了？为什么公式里还有个 $h_q s_q d_v$？这一项到底是 V 还是 O？

下面正式回答。

---

## 二、MLA 里 K 和 V 到底是不是一体的？

### 2.1 答案：是的，K 和 V 共享同一份底层存储

**MLA（Multi-head Latent Attention）的核心创新**就是：把 K 和 V 都从一个低维的 latent 向量投影出来，所以**只缓存 latent**，不分别缓存 K 和 V。

看 FlashMLA 的接口就清楚（`flash_mla/flash_mla_interface.py`）：

```python
def flash_mla_with_kvcache(
    q: (batch_size, seq_len_q, num_heads_q, head_dim),
    k_cache: (num_blocks, page_block_size, num_heads_k, head_dim),
    ...
)
```

**注意**：接口里只有 `k_cache`，**没有 `v_cache`**。这就是 MLA 的特征——KV Cache 实际上只是一块叫 `k_cache` 的内存，里面存的是 latent 表示。

### 2.2 传统 MHA vs MLA 的对比

**传统 MHA**（标准多头注意力）：

```
KV Cache 内存布局：
┌────────────────────────────┐
│  K: [s_k, h_k, d_k]         │  ← 独立存储
├────────────────────────────┤
│  V: [s_k, h_k, d_v]         │  ← 独立存储
└────────────────────────────┘

总搬运量 = K搬运 + V搬运 = 2·s_k·h_k·d_k + 2·s_k·h_k·d_v
                                   （bytes，BF16）
```

**MLA**（DeepSeek V3）：

```
KV Cache 内存布局：
┌────────────────────────────┐
│  latent: [s_k, 1, d_k]      │  ← 共享存储，h_k=1
│  （K 和 V 都从这块投影出来）  │
└────────────────────────────┘

总搬运量 = latent搬运 = 2·s_k·d_k
                            （bytes，BF16）
```

**关键数字**：传统 MHA 的 KV Cache 是 `2·s_k·h_k·d_k`（h_k 个 head 各存一份 K 和 V），MLA 的 KV Cache 只有 `2·s_k·d_k`（h_k=1，且 K/V 共享）。

**所以 K 和 V 的搬运量是同一份**——这就是博客公式里只有 $s_k d_k$ 一项（不带 $h_k$ 也不带 V）的原因。

### 2.3 一个生活比喻

**传统 MHA**：
- 档案柜里每个档案袋装两份文件：一份"索引"（K）+ 一份"内容"（V）
- 每个档案袋要存两份 → 占空间
- 取档案时要取两份 → 搬运量大

**MLA**：
- 档案柜里每个档案袋只装一份"压缩卷宗"（latent）
- 要用的时候，根据需要从压缩卷宗"投影"出 K 或 V（按需解压）
- 每个档案袋只存一份 → 省空间
- 取档案只取一份 → 搬运量小

**MLA 的本质：用"按需解压"换"存储节省"**。你只搬运 latent，K 和 V 在计算单元里临时投影出来。

---

## 三、那 $h_q s_q d_v$ 这一项到底是什么？

### 3.1 它是 O（输出）的写入量，不是 V 的搬运量

博客公式 $\text{sizeof}(\text{bfloat16}) \times (\ldots + h_q s_q d_v)$ 里的最后一项 $h_q s_q d_v$，**最合理的解释是 O 的写入量**：

| 张量 | 形状 | 搬运字节数（BF16） |
|------|------|------------------|
| Q（读） | $h_q \times s_q \times d_k$ | $2 \cdot h_q s_q d_k$ |
| K（读，KV Cache 共享） | $s_k \times d_k$ | $2 \cdot s_k d_k$ |
| V（读，与 K 共享 latent） | （已含在 K 里） | 0 |
| O（写） | $h_q \times s_q \times d_v$ | $2 \cdot h_q s_q d_v$ |

**所以 $h_q s_q d_v$ 是输出 O 的写入量**，不是 V 的搬运量。

### 3.2 为什么 V 的搬运量是 0？

在 MLA 里，V 不是预先存储的——它从 latent 投影出来。投影发生在**计算单元**（SM 的寄存器/共享内存）里，不需要从显存搬。

具体过程：

```
1. 从显存搬 latent（= K 的存储）到 SM         ← 搬运量：2·s_k·d_k
2. 在 SM 里把 latent 投影成 K（用于 Q×K^T）   ← 不需要访存，纯计算
3. 在 SM 里把 latent 投影成 V（用于 P×V）     ← 不需要访存，纯计算
4. 算出 O，写回显存                            ← 搬运量：2·h_q·s_q·d_v
```

**步骤 2 和 3 是矩阵乘法（投影）**，不是访存——它们消耗 FLOPs，不消耗带宽。

### 3.3 博客公式为什么这么写？

博客公式 $\text{Memory} = 2 \times (h_q s_q d_k + s_k d_k + h_q s_q d_v)$ 实际上是：

$$
\text{Memory} = \underbrace{2 h_q s_q d_k}_{\text{Q 读}} + \underbrace{2 s_k d_k}_{\text{KV Cache 读（K+V 共享）}} + \underbrace{2 h_q s_q d_v}_{\text{O 写}}
$$

**注意原文里 V 的搬运量没单独列**——因为它已经被 $s_k d_k$ 吸收了。这就是"K 和 V 一体"在公式里的体现。

---

## 四、为什么用户会觉得 V 的搬运量是 $h_q s_q d_v$？

这个误解很常见，源于几个混淆：

### 4.1 混淆 1：把 O 的形状误认为 V 的搬运量

O 的形状是 $h_q \times s_q \times d_v$——和 Q 的形状对称（Q 是 $h_q \times s_q \times d_k$）。看到公式里有 $h_q s_q d_v$，很容易直觉认为"这是 V 的搬运量"，但**它其实是 O 的写入量**。

### 4.2 混淆 2：把传统 MHA 的 V 搬运量套到 MLA 上

传统 MHA 里 V 的搬运量是 $s_k \cdot h_k \cdot d_v$（每个 token 每个 head 一份 V）。但 MLA 里 $h_k = 1$ 且 V 共享 latent，所以这个公式不适用。

### 4.3 混淆 3：把 V 的"计算量"误认为"搬运量"

V 参与 $P \times V$ 这个矩阵乘，计算量是 $2 h_q s_q s_k d_v$ FLOPs。**但 V 的搬运量不是 $h_q s_q d_v$**——V 在 SM 里被多次复用，搬运量按"原始数据大小"算，不按"参与计算的次数"算。

### 4.4 用一张表澄清所有混淆

| 量 | 公式 | 在 MLA 里 |
|----|------|----------|
| V 的**搬运量** | 0（V 共享 latent，不单独搬） | ✓ |
| V 的**计算量**（参与 P×V） | $2 h_q s_q s_k d_v$ FLOPs | ✓ |
| O 的**写入量** | $2 h_q s_q d_v$ bytes | ✓ |
| K 的**搬运量**（= latent 搬运） | $2 s_k d_k$ bytes | ✓ |

**这四件事完全不同，不要混在一起**。

---

## 五、回到原问题：那 K 和 V 是不是一体的？

### 5.1 答案：是一体的，但要分清"搬运"和"使用"

**K 和 V 在"搬运"层面是一体的**：
- 共享同一块显存（latent）
- 从显存搬到 SM 只搬一次
- 搬运量 = $2 s_k d_k$（不是两份）

**K 和 V 在"使用"层面是分开的**：
- 算 Q×K^T 时，把 latent 投影成 K
- 算 P×V 时，把 latent 投影成 V
- 两次投影是两次矩阵乘（计算量），不是两次搬运

### 5.2 一个生活比喻

想象你搬家：

- **传统 MHA**：每个房间的家具都要分别打包搬运（K 一车，V 一车）
- **MLA**：所有家具都能从一个"万能箱子"里变形出来，只搬一个箱子

但到了新家后：
- 你要从箱子里"变出"沙发（K）来摆客厅
- 你要从箱子里"变出"床（V）来摆卧室
- 变出沙发和变出床是**两次操作**（两次计算），但搬运只发生一次

### 5.3 用公式表达"一体"和"分开"

```
搬运阶段（一体）：
  Memory(HBM → SMEM) = 2·s_k·d_k   ← 一份 latent

使用阶段（分开）：
  算 Q×K^T: 把 latent 投影成 K，做矩阵乘 → 计算量 = 2·h_q·s_q·s_k·d_k
  算 P×V:   把 latent 投影成 V，做矩阵乘 → 计算量 = 2·h_q·s_q·s_k·d_v
```

**搬运只算一次（latent），计算要算两次（K 投影 + V 投影）**——这就是 MLA 的精妙之处：用多一点的计算换大量的显存和带宽节省。

---

## 六、完整的访存量公式重写

把上面所有澄清综合起来，MLA 解码的完整访存量应该是：

$$
\text{Memory} = \underbrace{2 \cdot h_q s_q d_k}_{\text{Q 读}} + \underbrace{2 \cdot s_k d_k}_{\text{KV Cache 读（K/V 共享 latent）}} + \underbrace{2 \cdot h_q s_q d_v}_{\text{O 写}}
$$

### 6.1 代入 DeepSeek V3 数字看量级

$h_q = 128, s_q = 1, d_k = d_v = 512, s_k = 4096$：

| 项 | 计算式 | 字节数 | 占比 |
|---|--------|-------|------|
| Q 读 | $2 \cdot 128 \cdot 1 \cdot 512$ | 131,072 | 3% |
| **KV Cache 读** | $2 \cdot 4096 \cdot 512$ | **4,194,304** | **93%** |
| O 写 | $2 \cdot 128 \cdot 1 \cdot 512$ | 131,072 | 3% |
| **总计** | | **4,456,448** | 100% |

**KV Cache 搬运占 93%**——这就是为什么博客公式可以近似为 $\text{Memory} \approx 2 s_k d_k$。

### 6.2 为什么博客只写 $h_q s_q d_v$ 不写 $h_q s_q d_k + h_q s_q d_v$？

仔细看博客原文：

$$
\text{Memory} = 2 \cdot (h_q s_q d_k + s_k d_k + h_q s_q d_v)
$$

三项分别是 Q、K（latent）、O。**博客把 V 省略了**——因为 V 共享 latent，已经被 $s_k d_k$ 这一项算过。

**用户看到的 $h_q s_q d_v$ 是 O 的写入量**，不是 V 的搬运量。这是问题的关键。

---

## 七、那 $d_k$ 和 $d_v$ 在 MLA 里相等吗？

### 7.1 在 MLA 里 $d_k = d_v$

MLA 中 K 和 V 从同一个 latent 投影出来，所以**它们在概念上有相同的维度**。DeepSeek V3 里 $d_k = d_v = 512$。

这就是为什么 MLA 算-访比公式能化简到 $2 h_q s_q$：

$$
\frac{\text{FLOPs}}{\text{Memory}} = \frac{2 h_q s_q s_k (d_k + d_v)}{2 s_k d_k} = h_q s_q \cdot \frac{d_k + d_v}{d_k} \stackrel{d_k = d_v}{=} 2 h_q s_q
$$

如果 $d_k \neq d_v$，化简不成立。

### 7.2 传统 MHA 里 $d_k$ 和 $d_v$ 不一定相等

- BERT：$d_k = d_v = 64$
- 有些模型设计 $d_v > d_k$（V 的维度更大，信息更丰富）
- 但传统 MHA 里 K 和 V 是分开存储的，所以 $d_k \neq d_v$ 只是改变存储量，不影响"是否共享"

---

## 八、常见误区与 FAQ

### Q1：KV Cache 不是 K 加 V 吗？为什么 MLA 里只有一份？

**传统 MHA 里确实是 K+V 两份**。MLA 把 K 和 V 都从一个 latent 投影出来，所以只缓存 latent——这就是 MLA 省 KV Cache 显存的核心。

接口里 `k_cache` 这个名字容易误导，它实际存的是 latent，但习惯上叫"k_cache"。

### Q2：那 V 不用从显存搬，是不是 V 的访问完全免费？

**不是免费，V 的"投影"是要算的**。

从 latent 投影成 V 是一次矩阵乘，消耗 FLOPs（已计入 FLOPs 公式）。"不用搬运"≠"不用计算"——只是把成本从带宽换成了算力。

这就是 MLA 的核心权衡：**用更多计算换更少访存**。在算力富余的计算密集型场景，这是赚的。

### Q3：为什么博客公式不显式写出 V 的搬运量？

因为 V 的搬运量是 0（已被 latent 那一项算过）。显式写 `+ 0` 反而容易让人困惑，所以省略。

但博客也没在公式注释里说清楚这一点——这是博客的一个小缺点，让读者（包括提问的用户）容易产生误解。

### Q4：Q 的搬运量 $h_q s_q d_k$ 和 O 的写入量 $h_q s_q d_v$ 为什么量级一样？

**因为解码时 $s_q = 1$ 且 $d_k = d_v$**：
- Q 形状：$h_q \times 1 \times d_k = 128 \times 1 \times 512$
- O 形状：$h_q \times 1 \times d_v = 128 \times 1 \times 512$

形状完全一样，搬运量自然一样。预填充时 $s_q$ 很大，但 Q 和 O 仍然量级相同（都是 $h_q s_q \times d$）。

### Q5：那 $h_q s_q d_v$ 到底是 V 还是 O？博客原文有说吗？

**博客原文没明说**。但从量级和位置看，最合理的解释是 **O 的写入量**：

- 如果是 V 的搬运量，前面应该有 $s_k$ 因子（V 形状是 $h_k \times s_k \times d_v$）
- $h_q s_q d_v$ 形状对应 $h_q \times s_q \times d_v$，正好是 O 的形状

**但有些读者会把它理解成 V 的搬运量**（V 投影到 Q 的 head 数后被"广播"了 $h_q$ 倍）。这种解释在工程上也成立——但严格说，MLA 的 V 是在 SM 内部投影的，不算"从显存搬运"。

### Q6：MLA 的 $h_k$ 为什么是 1？

**MLA 把所有 head 的 K 共享成一份**——这就是"Multi-head Latent"的"Latent"含义。所有 head 用同一个 latent，再各自投影出自己的 K 和 V。

所以 MLA 在显存层面是 **MQA**（Multi-Query Attention，所有 head 共享一份 K/V）的极端版本，但在计算层面仍然是 **MHA**（每个 head 独立算注意力）。

### Q7：为什么公式里 V 写成 $h_q s_q d_v$ 而不是 $h_k s_k d_v$？

如果按"V 的存储形状"写，应该是 $h_k s_k d_v$（MLA 中 $h_k = 1$）。但博客公式里是 $h_q s_q d_v$，这暗示它**不是 V 的存储搬运量**，而是 O 的写入量。

**这个细节就是用户问题的核心**——$h_q s_q d_v$ 的形状对应 O，不对应 V。把它误读成 V 的搬运量，就会产生"$K$ 搬运 $s_k d_k$、$V$ 搬运 $h_q s_q d_v$ 两者差 $h_q s_q$ 倍"的困惑。

---

## 九、总结：把这个误解彻底澄清

### 9.1 三个关键事实

1. **MLA 里 K 和 V 共享同一块显存（latent）** → 搬运只算一份，即 $2 s_k d_k$
2. **V 不需要单独搬运** → V 在 SM 内部从 latent 投影出来，是计算不是访存
3. **公式里的 $h_q s_q d_v$ 是 O 的写入量**，不是 V 的搬运量

### 9.2 用户问题的标准答案

> **问**：为什么 K 搬运 $2 s_k d_k$，V 搬运 $2 h_q s_q d_v$，差 $h_q s_q$ 倍？KV Cache 不是一体的吗？
>
> **答**：你的问题里有两个误解：
>
> 1. **K 和 V 是一体的**——共享同一份 latent，搬运只算一次 $2 s_k d_k$。
> 2. **$h_q s_q d_v$ 不是 V 的搬运量**，它是 O（输出）的写入量。V 的搬运量是 0（被 latent 那项吸收了）。
>
> 完整公式是 $\text{Memory} = 2 h_q s_q d_k + 2 s_k d_k + 2 h_q s_q d_v$，三项分别是 Q 读、KV Cache 读、O 写。**V 没有单独的搬运项**——这就是"KV Cache 一体"在公式里的体现。

### 9.3 一句话总结

> **MLA 里 K 和 V 共享 latent 存储，搬运只算一份；公式里 $h_q s_q d_v$ 是输出 O 的写入量，不是 V 的搬运量。V 在 SM 内部从 latent 投影出来，消耗算力但不消耗带宽——这就是 MLA"用计算换带宽"的核心设计。**

---

## 附：完整搬运量对照表

| 张量 | 形状 | 搬运方向 | 字节数（BF16） | 在 MLA 里 |
|------|------|---------|---------------|----------|
| Q | $h_q \times s_q \times d_k$ | HBM → SM | $2 h_q s_q d_k$ | 独立搬运 |
| K（latent） | $1 \times s_k \times d_k$ | HBM → SM | $2 s_k d_k$ | **K/V 共享这一份** |
| V | — | — | 0 | **从 latent 投影，不单独搬运** |
| O | $h_q \times s_q \times d_v$ | SM → HBM | $2 h_q s_q d_v$ | 输出写回 |
| **总访存** | | | $2 h_q s_q d_k + 2 s_k d_k + 2 h_q s_q d_v$ | |
| **近似**（$s_k \gg h_q s_q$） | | | $\approx 2 s_k d_k$ | |

---

## 附：进一步阅读

- [FlashMLA-MLA计算访存比推导详解.md](FlashMLA-MLA计算访存比推导详解.md) —— 算-访比完整推导（本文澄清了其中一处含糊）
- [FlashMLA-计算访存比详解.md](FlashMLA-计算访存比详解.md) —— 算-访比入门
- [FlashMLA-Q形状详解.md](FlashMLA-Q形状详解.md) —— Q 形状三维含义
- [FlashMLA-Kernel深度解读.md](FlashMLA-Kernel深度解读.md) —— FlashMLA 整体技术解读
- 原博客：`docs/20250422-new-kernel-deep-dive.md`
