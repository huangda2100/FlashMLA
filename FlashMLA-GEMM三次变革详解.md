# GPU 矩阵乘（GEMM）的三次变革：WMMA → WGMMA → UMMA

> 这篇文档回答一个问题：**为什么 GPU 上算矩阵乘的"指令"要换三代？每一代解决了上一代的什么痛点？**
>
> 面向零基础读者。会矩阵乘就行，不需要懂 GPU 架构。

---

## 一、问题背景：为什么 GEMM 这么重要？

### 1.1 大模型几乎全是矩阵乘

Transformer 模型里几乎所有的重计算都是矩阵乘（GEMM）：

- **Attention**：Q × K^T、P × V
- **FFN**：两层全连接，都是 GEMM
- **Linear/Projection**：QKV 投影、输出投影，都是 GEMM

训练一个大模型，**90%+ 的 FLOPs 来自 GEMM**。推理也一样——FlashMLA 内核的核心就是两个 GEMM（Q×K^T 和 P×V）。

**谁能让 GEMM 跑得快，谁就能让大模型跑得快。** 这就是 NVIDIA 三代架构（Volta → Hopper → Blackwell）持续改进 GEMM 指令的根本原因。

### 1.2 GEMM 是什么？

GEMM = GEneral Matrix Multiply，通用矩阵乘。计算：

$$
C_{M \times N} = A_{M \times K} \times B_{K \times N} + C_{M \times N}
$$

- 输入：矩阵 A（M×K）、矩阵 B（K×N）
- 输出：矩阵 C（M×N）
- 计算量：≈ 2MKN 次浮点运算
- 数据复用：A 的每个元素参与 N 次计算，B 的每个元素参与 M 次计算

**GEMM 的核心特点：数据高度复用**。同一个数据被反复用很多次，所以"算得越多、搬得越少"——这是 GPU 优化的金矿。

### 1.3 为什么需要专用指令？

CPU 也能算矩阵乘，为什么 GPU 要专门搞一套指令？

| 比较维度 | CPU（标量/向量） | GPU（早期 CUDA Core） | GPU（Tensor Core） |
|---------|-----------------|---------------------|-------------------|
| 一次算多少 | 1 个或 16 个乘加 | 数千个乘加 | 数千个**矩阵**乘加 |
| 单元 | ALU / AVX 向量单元 | CUDA Core | Tensor Core |
| 单次操作 | 标量乘加 | 标量乘加 | **小矩阵乘（如 16×16）** |
| 性能 | 慢 | 中 | 极快 |

**Tensor Core 的革命性**：一次指令直接算出一个**小矩阵乘**（比如 16×16×16 的 D = A × B + C），而不是一个一个元素算。这就是 GEMM 专用指令的本质。

---

## 二、三代 GEMM 指令一览

先给一张全景图，再逐个展开。

| 维度 | WMMA | WGMMA | UMMA |
|------|------|-------|------|
| **首次架构** | Volta（V100, 2017） | Hopper（H100/H800, 2022） | Blackwell（B100/200, 2024） |
| **指令助记符** | `mma` | `wgmma` | `tcgen05.mma` |
| **执行单元** | 1 个 warp（32 线程） | 1 个 warpgroup（4 个 warp = 128 线程） | 1 个 warpgroup + UMMA 引擎 |
| **操作数 A 位置** | 寄存器 | **共享内存** | **TMEM（专用张量内存）** |
| **操作数 B 位置** | 寄存器 | 共享内存 | 共享内存 |
| **结果 D 位置** | 寄存器 | 寄存器 | **TMEM（专用张量内存）** |
| **单次最大形状** | 16×16×16 | 64×256×16（甚至更大） | 128×256×16（甚至更大） |
| **异步执行** | 否 | **是** | **是** |
| **核心痛点解决** | 让 GPU 能算矩阵乘 | 解决寄存器瓶颈 + 异步 | 解决"寄存器装不下大矩阵" |

**一句话总结三代的演进逻辑**：

> - **WMMA（Volta）**：发明了 Tensor Core，让 GPU 能"一条指令算一个小矩阵乘"
> - **WGMMA（Hopper）**：操作数从寄存器搬到共享内存，单次能算更大的矩阵，并支持异步
> - **UMMA（Blackwell）**：引入专用张量内存 TMEM，操作数和结果都不挤占通用寄存器，单次算更大的矩阵

下面逐代展开。

---

## 三、第一代：WMMA（Volta，2017）

### 3.1 解决什么问题？

**问题**：在 Volta 之前，GPU 只能做"标量乘加"。算一个 16×16 的矩阵乘需要发射几百条指令，每条指令只算 1 个乘加——效率极低。

**Volta 的革命**：引入 **Tensor Core**，一条指令算出一个**小矩阵乘**。

```
一条 mma.sync 指令：
  D[16×16] = A[16×16] × B[16×16] + C[16×16]
```

16×16×16 = 4096 次乘加，一条指令完成。

### 3.2 WMMA 怎么工作？

**WMMA = Warp Matrix Multiply Accumulate**，warp 级别的矩阵乘加。

- **执行单位**：1 个 warp（32 个线程协作）
- **数据位置**：A、B、C、D 都在**寄存器**中
- **数据分片**：32 个线程共同"持有"一个 16×16 矩阵，每个线程拿其中的若干元素

**关键限制**：操作数必须在寄存器里。

### 3.3 一个生活比喻

想象 32 个工人（1 个 warp）协作组装一个 16×16 的零件墙：

- 每个工人手里拿着几块零件（寄存器里的 A、B 片段）
- 工头喊一声"组装！"（mma 指令）
- 工人们同时动手，把零件拼成新的 16×16 块（结果 D）
- 结果还是分摊在 32 个工人手里（D 在寄存器里）

**优点**：第一次让 GPU 能高效算矩阵乘。
**缺点**：零件都在工人手里（寄存器），但工人手就那么大——**寄存器是稀缺资源**。

### 3.4 WMMA 的瓶颈

1. **寄存器压力大**：每个 16×16 矩阵占 8 个寄存器（FP16），算更大的矩阵要更多寄存器。GPU 每个 SM 只有 65536 个 32 位寄存器，分给上千个线程后每人就那么几个。

2. **同步开销**：wmma.sync 是同步指令——发射后必须等结果出来才能继续。计算和访存无法重叠。

3. **单次形状太小**：16×16×16 对现代大模型来说粒度太细，要循环很多次才能算完一个 GEMM。

**这些瓶颈在 Ampere（2020）时代被进一步放大**——算力涨了 10 倍，但 wmma 指令的单次形状只稍微变大（如 16×8×16 → 16×16×16），寄存器压力没缓解。

NVIDIA 决定在 Hopper 上彻底重构。

---

## 四、第二代：WGMMA（Hopper，2022）

### 4.1 解决什么问题？

**WMMA 的核心痛点**：操作数在寄存器，单次能算的形状被寄存器容量卡死。

**Hopper 的思路**：让操作数从寄存器搬到**共享内存**——共享内存比寄存器大得多（SMEM 是 256KB，寄存器是 256KB 但分给所有线程），可以装更大的矩阵。

### 4.2 WGMMA 怎么工作？

**WGMMA = Warpgroup GEMM**，warpgroup 级别的矩阵乘。

- **执行单位**：1 个 **warpgroup**（4 个 warp = 128 个线程），比 WMMA 大 4 倍
- **操作数 A**：在**共享内存**（SMEM）
- **操作数 B**：在**共享内存**（SMEM）
- **结果 D**：在**寄存器**（RMEM）
- **执行模式**：**异步**——发射后立刻返回，线程可以继续干别的，过段时间再来取结果

**单次形状最大可达 64×256×16**——比 WMMA 的 16×16×16 大了两个数量级。

### 4.3 一个生活比喻

升级版工厂：

- 4 组工人共 128 人（1 个 warpgroup），规模是 WMMA 的 4 倍
- 零件不再攥在工人手里，而是放在**中央工作台**（共享内存）上
- 工头喊"开始组装！"（wgmma 指令），但**工头不用等**——他可以去安排下一批零件进工作台
- 过一段时间，工头来取成品（结果 D 在寄存器里）

**两个关键升级**：
1. **中央工作台（共享内存）比手大得多** → 可以装更大的矩阵
2. **工头不用盯着等（异步）** → 可以一边算一边搬下一批数据，**计算和访存重叠**

### 4.4 WGMMA 的实际代码长什么样

FlashMLA 的 SM90 代码里用到的 WGMMA 长这样（伪代码）：

```cuda
// 1. 把数据从显存搬到共享内存（用 TMA 异步拷贝）
TMA::copy(K_block_smem, K_block_hbm);
TMA::copy(V_block_smem, V_block_hbm);

// 2. 等数据到位
wait_tma();

// 3. 发射 WGMMA（异步！）
wgmma.async(D_registers, K_block_smem, V_block_smem);

// 4. 不用等结果，可以继续发起下一个 TMA 拷贝
TMA::copy(next_K_block_smem, next_K_block_hbm);

// 5. 等结果出来再做 softmax
wait_wgmma();
softmax(D_registers);
```

**关键**：第 3 步的 wgmma 和第 4 步的 TMA 可以**同时进行**——这是 Hopper 性能爆发的核心。

### 4.5 WGMMA 的瓶颈

1. **结果 D 还在寄存器**：单次 WGMMA 的 D 可以是 64×256，占 32768 个 32 位寄存器——一个 SM 的寄存器总量（65536）的一半就没了。**没法同时存两个 D 做 ping-pong**。这就是 FlashAttention-3 的 ping-pong 调度在 H100 上做不了的根本原因，FlashMLA 不得不发明 seesaw 调度。

2. **共享内存也有限**：256KB 的 SMEM 要装 K、V、Q 还要做 ping-pong 缓冲，紧张。

3. **A 必须在 SMEM**：每次算 GEMM 前必须把 A 搬到 SMEM，有搬运开销。

**Hopper 把 GEMM 推到了 80% 算力利用率，但寄存器墙还在。** Blackwell 决定再推一把。

---

## 五、第三代：UMMA（Blackwell，2024）

### 5.1 解决什么问题？

**WGMMA 的核心痛点**：结果 D 必须放在寄存器，而寄存器是稀缺资源——大矩阵装不下，做不了真正的 ping-pong。

**Blackwell 的思路**：给 Tensor Core 配一块**专用内存**，让操作数和结果都不挤占通用寄存器。

### 5.2 UMMA 和 TMEM 是什么？

**UMMA = U MMA**，第五代 Tensor Core 的矩阵乘指令（助记符 `tcgen05.mma`）。

- **执行单位**：1 个 warpgroup（128 线程）
- **操作数 A**：在 **TMEM**（Tensor Memory，新引入的专用内存）
- **操作数 B**：在**共享内存**（SMEM）
- **结果 D**：在 **TMEM**
- **执行模式**：异步

**TMEM 是什么？**

> TMEM = Tensor Memory，Blackwell 引入的专用高速内存，**只给 Tensor Core 用**。

- 容量比寄存器大得多（每个 SM 256KB+）
- 带宽比共享内存高
- 不占用通用寄存器
- 和通用寄存器之间有专门的数据搬运路径

### 5.3 一个生活比喻

终极版工厂：

- 4 组工人 128 人（1 个 warpgroup）
- 工厂里多了一个**专用零件仓**（TMEM），只给组装线用，不放别的东西
- A 零件预先放进专用仓（TMEM 里的 A）
- B 零件还在中央工作台（SMEM）
- 工头喊"组装！"（tcgen05.mma），**结果直接放进专用仓**（D 在 TMEM）
- 通用工具箱（通用寄存器）完全不被占用，可以专心做别的事（softmax、rescale 等）

**关键升级**：
1. **D 不再挤占通用寄存器** → 可以同时存好几个 D，**真正的 ping-pong 终于可以做了**
2. **TMEM 容量大** → 单次 GEMM 可以算更大的矩阵
3. **专用带宽** → TMEM 和 Tensor Core 之间有专用高速通道，搬运不再堵

### 5.4 UMMA 的实际代码长什么样

FlashMLA 的 SM100 代码里（`csrc/kerutils/include/kerutils/device/sm100/gemm.cuh`）有这样的内联汇编：

```cuda
asm volatile(
    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\n\t"
    :
    : "r"(d_desc),    // D 在 TMEM 的描述符
      "r"(a_desc),    // A 在 TMEM 的描述符
      "r"(b_desc),    // B 在 SMEM 的描述符
      "r"(idesc)      // 指令描述符（形状、精度等）
);
```

注意几个特征：
- 操作数和结果都是**描述符**（一个 64 位整数），不是寄存器数组
- `tcgen05.mma` 是新指令助记符，`tcgen05` 表示第五代 Tensor Core
- `.ws` 表示 warpgroup-specialized（warpgroup 专用）
- `.kind::f16` 指定数据类型

**这就是 UMMA 的"长相"**——和 WGMMA 看起来类似，但操作数位置完全不同。

### 5.5 UMMA 的优势

1. **寄存器完全释放**：D 在 TMEM，通用寄存器可以全用来做 softmax、rescale、loop 控制等。

2. **真正的 ping-pong**：可以同时存 2 个甚至多个 D，做计算-softmax 流水线。

3. **更大的单次形状**：128×256×16 甚至更大，进一步减少循环开销。

4. **更高的算力利用率**：配合 ping-pong 调度，可以接近 90%+ 的 Tensor Core 利用率。

### 5.6 UMMA 的代价

1. **编程复杂度爆炸**：要管理 TMEM 分配、描述符、专用 barrier，比 WGMMA 难写得多。

2. **数据搬运路径更复杂**：HBM → SMEM → TMEM → Tensor Core → TMEM → SMEM → HBM，每一段都要显式管理。

3. **新概念学习曲线陡峭**：TMem、UMMA descriptor、tcgen05 指令集——从 Hopper 迁移过来需要重新学。

---

## 六、三代对比：一张表读全

| 维度 | WMMA（Volta） | WGMMA（Hopper） | UMMA（Blackwell） |
|------|--------------|-----------------|-------------------|
| **年代** | 2017 | 2022 | 2024 |
| **指令** | `mma.sync` | `wgmma.async` | `tcgen05.mma` |
| **执行单位** | warp（32 线程） | warpgroup（128 线程） | warpgroup（128 线程） |
| **A 位置** | 寄存器 | 共享内存 | **TMEM** |
| **B 位置** | 寄存器 | 共享内存 | 共享内存 |
| **D 位置** | 寄存器 | 寄存器 | **TMEM** |
| **单次形状** | 16×16×16 | 64×256×16 | 128×256×16+ |
| **异步** | ❌ | ✅ | ✅ |
| **寄存器压力** | 高 | 高（D 占一半） | **低（D 在 TMEM）** |
| **Ping-pong 可行** | ❌ | ❌（D 装不下两个） | ✅ |
| **编程难度** | 中 | 高 | 极高 |
| **代表 GPU** | V100 | H100 / H800 | B100 / B200 |
| **代表应用** | BERT 时代 GEMM | LLM 训练/推理 | 万亿参数模型推理 |

---

## 七、演进的底层逻辑：每代都在"搬数据"

仔细看三代的演进，你会发现一个共同主题：**把数据从"小而快"的地方挪到"大而专用"的地方**。

```
WMMA:    A, B, C, D 全在寄存器
              ↓ 寄存器不够用
WGMMA:   A, B 搬到共享内存，D 还在寄存器
              ↓ 寄存器还是不够（D 太大）
UMMA:    A, D 搬到 TMEM（专用内存），B 留在共享内存
```

**每一代都是在"找一个更大的地方放操作数和结果"**。

为什么？因为 GEMM 的核心瓶颈是**数据搬运**，不是计算本身。Tensor Core 算得够快——只要你能把数据喂给它。所以三代演进都在解决"怎么把数据准备好，让 Tensor Core 不饿"。

---

## 八、这跟 FlashMLA 有什么关系？

FlashMLA 的代码库正好**同时支持 Hopper 和 Blackwell 两代**，可以对比看：

| 路径 | 架构 | 用的 GEMM 指令 |
|------|------|---------------|
| `csrc/sm90/` | Hopper（SM90） | WGMMA |
| `csrc/sm100/` | Blackwell（SM100） | UMMA（tcgen05） |

### 8.1 Hopper 版本（`csrc/sm90/`）的特点

- 用 `wgmma.async` 指令
- 输出矩阵 D 在寄存器
- 因为寄存器装不下两个 D，**没法做 FlashAttention-3 的 ping-pong**
- 不得不发明 **seesaw 调度**：把 D 竖着劈成两半（O_L 和 O_R），两个 warpgroup 交替操作

### 8.2 Blackwell 版本（`csrc/sm100/`）的特点

- 用 `tcgen05.mma` 指令
- 输出矩阵 D 在 TMEM，不占通用寄存器
- **理论上可以做真正的 ping-pong**（虽然代码实现可能仍用 seesaw 的变种）
- 寄存器更宽裕，可以做更复杂的 softmax 和 rescale

**所以你看，每一代 GEMM 指令的演进，直接决定了一个 attention kernel 能不能做某种调度方案。** FlashMLA 的 seesaw 调度本质上是"在 WGMMA 寄存器限制下的无奈之举"——到了 UMMA 时代，这个限制就缓解了。

---

## 九、常见误区与 FAQ

### Q1：WMMA、WGMMA、UMMA 是不是替换关系？旧代码还能跑吗？

**不是替换，是叠加**。每一代架构都保留了对旧指令的支持：
- Blackwell 既支持 UMMA，也支持 WGMMA 和 WMMA
- Hopper 既支持 WGMMA，也支持 WMMA

但**要用满新架构的算力，必须用新指令**。在 H100 上用 WMMA，只能跑到 H100 算力的 1/3 左右。

### Q2：为什么不一开始就搞 TMEM？

**因为不需要**。2017 年的矩阵乘规模小（BERT 才 1 亿参数），WMMA 的 16×16 就够用了。TMEM 是个昂贵的资源——专门给 Tensor Core 划一块高速内存，硅片面积和成本都不小。

只有当 GEMM 规模大到寄存器装不下结果时，TMEM 才划算。2024 年的万亿参数模型推理才到了这个临界点。

### Q3：UMMA 的 TMEM 和寄存器、共享内存是什么关系？

GPU 内存的层次结构（Blackwell 时代）：

```
                  ┌─────────────────┐
   最快但最小    │  寄存器 (RMEM)  │  每线程私有，~256 个
                  ├─────────────────┤
                  │  TMEM (新!)     │  SM 级，给 Tensor Core 专用
                  ├─────────────────┤
                  │  共享内存 (SMEM)│  SM 级，~256KB，程序员管理
                  ├─────────────────┤
                  │  L2 缓存        │  全 GPU 共享，~50MB
                  ├─────────────────┤
   最慢但最大    │  显存 (HBM)     │  全 GPU 共享，~80GB
                  └─────────────────┘
```

TMEM 介于寄存器和共享内存之间，**专门给 Tensor Core 用**。

### Q4：WGMMA 已经是异步了，UMMA 的异步有什么不一样？

WGMMA 的异步是"发射后可以做别的事，但结果必须在寄存器里取"——结果出来后还是会占用寄存器。

UMMA 的异步是"发射后可以做别的事，结果直接进 TMEM，连取都不用取"——下一步要用的时候直接从 TMEM 读。**整条流水线更顺畅**。

### Q5：写 UMMA 代码难吗？

**非常难**。比 WGMMA 难一个量级。主要难点：
- 要手写描述符（64 位数编码形状、布局、地址）
- 要管理 TMEM 分配和释放
- 要用专门的 barrier（mbarrier）做同步
- 调试工具少，错了不好排查

所以现在大部分项目还停留在 WGMMA，UMMA 主要被大厂和库作者（CUTLASS、FlashAttention 团队）使用。

### Q6：FlashMLA 用的哪一代？

**两代都用**。看 `csrc/sm90/` 是 Hopper 版本（WGMMA + seesaw 调度），`csrc/sm100/` 是 Blackwell 版本（UMMA）。运行时根据 GPU 自动选择。

### Q7：下一代（Rubin？）会有什么新指令？

NVIDIA 的 Rubin 架构（预计 2025-2026）可能会进一步扩展 TMEM、增加新的数据类型支持、提高单次 GEMM 形状。但具体指令集要等官方文档。**历史规律：每一代都在"搬数据到更专用的地方"**，下一代可能继续这个趋势。

---

## 十、一张图回顾三代演进

```
        WMMA (2017)              WGMMA (2022)              UMMA (2024)
        ─────────                ──────────                ──────────
        
        Volta/V100               Hopper/H100               Blackwell/B100
        
        mma.sync                 wgmma.async               tcgen05.mma
        
        warp (32 threads)        warpgroup (128)           warpgroup (128)
        
        A: reg                   A: smem                   A: TMEM ←─┐
        B: reg                   B: smem                   B: smem   │
        C: reg                   C: reg                    C: TMEM ←─┴─ 新增专用内存
        D: reg                   D: reg                    D: TMEM ←─┘
        
        shape: 16³               shape: 64×256×16          shape: 128×256×16+
        
        sync                     async                     async
        
        ❌ ping-pong             ❌ ping-pong              ✅ ping-pong
        (寄存器不够)             (D 太大占满寄存器)        (D 在 TMEM)
        
        ↓                        ↓                         ↓
        BERT 时代                LLM 训练/推理             万亿参数推理
        
        寄存器墙 ──────→ 共享内存墙 ──────→ 专用内存（TMEM）
```

---

## 十一、一句话总结

> **三代 GEMM 指令的演进，本质是"把操作数和结果从寄存器里搬出去"的历史**：
>
> - **WMMA**（Volta, 2017）：A、B、C、D 全在寄存器——发明了 Tensor Core，但寄存器是瓶颈
> - **WGMMA**（Hopper, 2022）：A、B 搬到共享内存，D 还在寄存器——单次形状放大 100 倍，但 D 太大装不下两个，做不了 ping-pong
> - **UMMA**（Blackwell, 2024）：A、D 搬到专用 TMEM——寄存器完全释放，真正的 ping-pong 终于可行
>
> **每一代都是在"找一个更大的地方放数据，让 Tensor Core 不饿"**。FlashMLA 的 seesaw 调度就是 WGMMA 时代寄存器限制下的"曲线救国"，到了 UMMA 时代这个限制自然缓解。

---

## 附：术语速查表

| 术语 | 全称 | 含义 |
|------|------|------|
| GEMM | General Matrix Multiply | 通用矩阵乘 |
| WMMA | Warp Matrix Multiply Accumulate | Volta 时代的 Tensor Core 指令 |
| WGMMA | Warpgroup GEMM | Hopper 时代的 Tensor Core 指令 |
| UMMA | (U) MMA | Blackwell 时代的 Tensor Core 指令（第五代） |
| Warp | - | 32 个线程的协作单元 |
| Warpgroup | - | 4 个 warp = 128 线程的协作单元（Hopper 引入） |
| RMEM | Register Memory | 寄存器内存 |
| SMEM | Shared Memory | 共享内存，SM 级 |
| TMEM | Tensor Memory | 张量内存，Blackwell 引入，Tensor Core 专用 |
| TMA | Tensor Memory Accelerator | 张量内存加速器，Hopper 引入的异步拷贝单元 |
| Tensor Core | - | GPU 上专做矩阵乘的计算单元 |
| SM | Streaming Multiprocessor | GPU 的计算块，相当于 CPU 的核心 |

---

## 附：进一步阅读

- [FlashMLA-Kernel深度解读.md](FlashMLA-Kernel深度解读.md) —— FlashMLA 整体技术解读（含 seesaw 调度）
- [FlashMLA-计算访存比详解.md](FlashMLA-计算访存比详解.md) —— 计算-访存比入门
- [FlashMLA-MLA计算访存比推导详解.md](FlashMLA-MLA计算访存比推导详解.md) —— MLA 算-访比公式推导
- CUTLASS 文档：https://github.com/NVIDIA/cutlass
- Hopper WGMMA 白皮书：NVIDIA H100 Tensor Core 白皮书
- Blackwell 架构白皮书：NVIDIA B100/B200 技术文档
