# TMA Pipelining 是什么？

> 这篇文档回答一个问题：**TMA pipelining 到底是什么？它解决了什么问题？为什么 FlashMLA 要把它列为关键技术？**
>
> 面向零基础读者。会"做饭"和"工厂流水线"的比喻就够，不需要懂 GPU。

---

## 一、问题背景：GPU 计算的"等料"困境

### 1.1 一个工厂的困境

想象你开了一家工厂，有一条产线：

- **产线速度**：每分钟可以加工 100 个零件
- **原料仓库**：在工厂外面，离产线很远
- **搬运工**：每分钟能搬 100 个零件到产线
- **搬运距离**：从仓库搬到产线要 10 分钟

现在你接到一个 1000 个零件的订单。怎么生产？

**笨办法（串行）**：

```
10:00  开始搬第 1 批 100 个零件
10:10  搬到产线，开始加工
10:11  加工完，开始搬第 2 批
10:21  第 2 批到，开始加工
10:22  加工完，开始搬第 3 批
...
```

每批要花 11 分钟（10 分钟搬运 + 1 分钟加工），1000 个零件要 110 分钟。**产线 99% 的时间在等料**——这就是"等料困境"。

**聪明办法（流水线）**：

```
10:00  开始搬第 1 批
10:01  搬第 1 批还在路上 → 同时开始搬第 2 批
10:02  搬第 1、2 批在路上 → 同时开始搬第 3 批
...
10:10  第 1 批到产线，开始加工；同时第 2-10 批都在路上
10:11  第 1 批加工完；第 2 批到产线，开始加工
...
```

10 分钟预热后，产线**满负荷运转**，每分钟出 100 个。1000 个零件只要 20 分钟。**搬运和加工同时进行，谁都不等谁**。

**这就是 pipelining（流水线）的核心思想**。

### 1.2 GPU 上的等料困境

GPU 也面临同样的问题：

- **产线 = Tensor Core**：每秒能做 865 万亿次浮点运算
- **原料仓库 = 显存（HBM）**：存着模型参数、KV Cache
- **搬运工 = 内存总线**：每秒能搬 3.35 万亿字节
- **搬运距离 = 内存延迟**：从发起到数据就绪要几百个时钟周期

**如果等数据到了才开始算，Tensor Core 大部分时间在闲着**——这就是为什么需要 pipelining。

### 1.3 TMA 是什么？TMA pipelining 又是什么？

**TMA = Tensor Memory Accelerator**（张量内存加速器）

Hopper 架构（H100/H800）引入的**专用硬件单元**，专门用来"搬张量"。它的特点是：

- **异步**：CPU/GPU 发起 TMA 拷贝后不用等，可以继续干别的
- **多维度**：一条指令直接搬一个 2D/3D/4D/5D 的小块，不用循环
- **带 barrier**：拷贝完成时可以发信号通知，不需要轮询

**TMA pipelining** = 用 TMA 的异步特性，把"搬数据"和"算数据"重叠起来——**搬下一批数据的同时，算上一批数据**。

这就是前面"聪明办法"的 GPU 版本。

---

## 二、先看没有 pipelining 的世界：cp.async 时代

### 2.1 Volta/Ampere 时代怎么搬数据？

TMA 是 Hopper 才有的。在那之前（Volta 2017、Ampere 2020），GPU 用 `cp.async` 指令搬数据：

```cuda
// Volta/Ampere 搬数据
cp.async.ca.shared.global [smem_addr], [global_addr], 16;
// 异步拷贝 16 字节从显存到共享内存
```

`cp.async` 也是异步的，但有几个问题：

| 问题 | 后果 |
|------|------|
| **粒度小**（一次 16 字节） | 搬一个大块要循环很多次 |
| **不感知形状** | 2D/3D 张量要手动算地址，代码复杂 |
| **同步开销** | 要用 `cp.async.wait_group` 等待，多少有点阻塞 |
| **占用线程** | 每个线程都要发 cp.async 指令，浪费算力 |

### 2.2 一个具体例子感受痛点

假设要搬一个 64×576 的 K 块（FP16，72KB）从显存到共享内存：

**用 cp.async**：

```cuda
// 64×576 = 36864 个元素，每个 2 字节
// cp.async 一次搬 16 字节 = 8 个元素
// 要循环 36864 / 8 = 4608 次
for (int i = 0; i < 4608; i++) {
    cp.async.ca.shared.global [smem + i*16], [global + i*16], 16;
}
cp.async.wait_group 0;  // 等全部搬完
```

**问题**：
- 4608 条指令，每条要 1 个线程发射，占用大量线程时间
- 中间不能开始计算，必须等全部搬完

**Hopper 之前的"流水线"**：用多缓冲区（ping-pong buffer）+ cp.async，能部分重叠搬运和计算，但粒度还是小、指令还是多。

---

## 三、TMA 怎么工作？

### 3.1 TMA 的核心特征

**TMA = Tensor Memory Accelerator**，Hopper 引入的硬件单元。它把"搬张量"这件事彻底异步化、形状化。

#### 特征 1：一条指令搬一个多维块

TMA 一次可以搬一个 **2D/3D/4D/5D 的小块**，不用循环。

```cuda
// 用 TMA 搬一个 64×576 的块（Hopper）
// 伪代码，实际用 CUTLASS 包装
cp.async.bulk.tensor.2d.shared::cta.global.tile
    [smem_addr], [tensor_map, {x, y}];
```

**一条指令搞定**——不用循环 4608 次。

#### 特征 2：完全异步

TMA 发起后**立即返回**，线程可以继续干别的（比如算上一个块）。完成时通过 **mbarrier**（memory barrier，内存屏障）发信号通知。

```cuda
// 发起 TMA 拷贝
cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes ...

// 线程立即继续，不用等
// ... 干别的事 ...

// 等数据就绪（只在这里等，前面不阻塞）
mbarrier.try_wait ...
```

#### 特征 3：带缓存提示

TMA 可以指定 L2 缓存策略，比如 `EVICT_FIRST`（用完优先驱逐）——这对 KV Cache 这种大数据很有用。

### 3.2 一个生活比喻

**cp.async 时代**：

- 搬运工每次只能搬 1 个箱子（16 字节）
- 搬 4608 个箱子要 4608 趟
- 每次都要工头发指令"搬这个"，工头很忙

**TMA 时代**：

- 搬运工一次能搬一整车（多维块）
- 工头说"把这堆箱子搬到产线" → 一句话搞定
- 搬运工到了自动按铃（mbarrier），工头不用盯着

### 3.3 TMA 的硬件基础

Hopper SM 里专门有 TMA 单元：

```
┌─────────────────────────────────────┐
│              SM (Hopper)            │
│                                     │
│   ┌──────────┐    ┌──────────┐      │
│   │  Tensor  │    │  CUDA    │      │
│   │  Cores   │    │  Cores   │      │
│   └────┬─────┘    └──────────┘      │
│        │                            │
│   ┌────┴─────────────┐              │
│   │  共享内存 (SMEM)  │             │
│   └────┬─────────────┘              │
│        │                            │
│   ┌────┴─────┐                      │
│   │   TMA    │  ← 专用搬运单元      │
│   └────┬─────┘                      │
└────────┼────────────────────────────┘
         │
    ┌────┴────┐
    │   HBM   │  ← 显存
    └─────────┘
```

**关键**：TMA 是独立硬件单元，搬数据时**不占用 Tensor Core**，也不占用 CUDA Core——搬运和计算真正可以并行。

---

## 四、TMA pipelining 的核心思想

### 4.1 把搬运和计算重叠起来

**TMA pipelining 的本质**：用 TMA 的异步特性，让"搬下一块数据"和"算上一块数据"**同时进行**。

```
时间轴 →

搬运:  [搬块1][搬块2][搬块3][搬块4]...
计算:         [算块1][算块2][算块3]...
                     ↑
              搬块2 和 算块1 同时进行
```

**关键**：只要"搬一块"的时间 ≤ "算一块"的时间，搬运就能完全隐藏在计算后面——Tensor Core 永不"等料"。

### 4.2 多缓冲区（multi-buffer）实现

实际实现需要**多个共享内存缓冲区**交替使用：

```
SMEM 布局：
┌──────────┬──────────┬──────────┬──────────┐
│ Buffer A │ Buffer B │ Buffer C │ Buffer D │
└──────────┴──────────┴──────────┴──────────┘

时间线：
  t=0: TMA 搬块1 → Buffer A
  t=1: TMA 搬块2 → Buffer B  | 等 Buffer A 就绪
  t=2: TMA 搬块3 → Buffer C  | 算块1（从 Buffer A） | 等 Buffer B 就绪
  t=3: TMA 搬块4 → Buffer D  | 算块2（从 Buffer B） | 等 Buffer C 就绪
  t=4: TMA 搬块5 → Buffer A  | 算块3（从 Buffer C） | 等 Buffer D 就绪
  ...
```

**至少需要 2 个 buffer（ping-pong）**，常用 3-4 个让流水线更稳定。

### 4.3 用生活比喻再讲一遍

**没有 pipelining（串行）**：

```
搬箱1 → 拆箱1 → 搬箱2 → 拆箱2 → 搬箱3 → 拆箱3
▓▓▓     ░░░     ▓▓▓     ░░░     ▓▓▓     ░░░
```

搬箱子（▓）和拆箱子（░）交替进行，总时间 = 搬箱时间 + 拆箱时间。

**有 pipelining（流水线）**：

```
搬箱1 ▓▓▓
搬箱2     ▓▓▓
搬箱3         ▓▓▓
搬箱4             ▓▓▓
拆箱1   ░░░
拆箱2       ░░░
拆箱3           ░░░
拆箱4               ░░░
```

**搬和拆同时进行**，总时间 ≈ max(搬箱时间, 拆箱时间)。

这就是 pipelining 的威力——把串行变并行。

---

## 五、FlashMLA 的 TMA pipelining 具体怎么做？

### 5.1 博客原文的描述

FlashMLA 博客对 TMA pipelining 的描述：

> **Fine-grained TMA copy - GEMM pipelining:** For a 64×576 K block, we launch 9 TMA copies (each moving a 64×64 block). GEMM operations begin as soon as each TMA copy completes (When the first TMA copy is done, we can start the first GEMM operation, and so on), improving memory latency tolerance.

翻译过来：

- 要搬一个 64×576 的 K 块
- **不一次搬完**，而是切成 **9 个 64×64 的小块**（576 ÷ 64 = 9）
- 每搬完一个小块，**立即开始算对应的部分矩阵乘**
- 不用等整个 64×576 全搬完

### 5.2 为什么要切小块？

**关键问题**：如果等整个 64×576 全搬完再开始算，等待时间长；如果搬完一小块就开始算，等待时间短。

**对比**：

| 方案 | 等待时间 | 重叠程度 |
|------|---------|---------|
| 整块搬完再算 | 长（搬 72KB 的时间） | 无重叠 |
| 切 9 块，搬完第 1 块就算 | 短（搬 8KB 的时间） | 高度重叠 |

**切小后的时间线**：

```
TMA:  [搬小块1][搬小块2][搬小块3]...[搬小块9]
GEMM:          [算小块1][算小块2]...[算小块8][算小块9]
                          ↑
                    搬小块3 和 算小块1 同时
```

**每个小块的搬运时间 ÷ 9，平均等待时间也大幅下降**。

### 5.3 实际代码长什么样

FlashMLA 的 SM90 代码里（`csrc/sm90/helpers.h` 等文件），TMA 用 CUTLASS 的 `cute::make_tma_copy` 包装：

```cuda
// 1. 创建 TMA 拷贝对象（包含 tensor map，描述多维形状）
auto tma_K = cute::make_tma_copy(
    SM90_TMA_LOAD{},          // TMA 加载指令
    gmem_tensor_K,            // 全局内存的 K 张量
    smem_layout_K,            // 共享内存布局
    shape_64x64,              // 单次搬运的形状
    sizeof(bf16)              // 元素大小
);

// 2. 发起 TMA 拷贝（异步！立即返回）
cute::copy(tma_K, gmem_tensor_K_block, smem_tensor_K_block);

// 3. 发起 mbarrier 信号，告诉 GPU "这次拷贝完成时通知我"
cute::tma_store_arrive();  // 或者 mbarrier::arrive

// 4. 不用等，继续发起下一个 TMA 拷贝
cute::copy(tma_K, next_gmem_K_block, next_smem_K_block);

// 5. 等第 1 块就绪，开始 GEMM
cute::wait_barrier(barrier_1);
wgmma.async(D_registers, smem_K_block_1, smem_V_block_1);

// 6. 等第 2 块就绪，继续 GEMM
cute::wait_barrier(barrier_2);
wgmma.async(D_registers, smem_K_block_2, smem_V_block_2);
```

**关键点**：
- 步骤 2 发起后立即返回，线程不阻塞
- 步骤 4 可以在步骤 5 之前发起——**搬下一块的同时算上一块**
- 步骤 5 的 `wait_barrier` 只在数据没就绪时才阻塞，数据就绪了立即返回

### 5.4 真实的汇编指令

更底层的 `cp.async.bulk.tensor` 指令长这样（`csrc/kerutils/README.md` 里有真实例子）：

```cuda
// TMA 多维聚合加载（2D，带 mbarrier，带 L2 缓存提示）
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
    :
    : "r"(smem_addr),    // 共享内存目的地址
      "r"(tensor_map),   // 张量描述符（编码了形状、stride 等）
      "r"(coord_x), "r"(coord_y), ...  // 多维坐标
      "r"(barrier_addr), // mbarrier 地址
      "r"(cache_hint)    // L2 缓存提示（如 EVICT_FIRST）
);
```

**一条指令**完成：
- 从多维张量搬一个块到共享内存
- 搬完自动通知 mbarrier
- 带 L2 缓存提示

**这就是 TMA 的威力**——cp.async 时代要循环几百次的事，TMA 一条指令搞定。

---

## 六、TMA pipelining 解决了什么问题？

### 6.1 隐藏内存延迟

**问题**：从发起到数据就绪有几百个时钟周期的延迟。如果同步等待，Tensor Core 大量空闲。

**TMA pipelining 的解法**：提前发起多个 TMA 拷贝，让数据"在路上"，等需要时已经就绪。

**生活比喻**：你点外卖，下完单不用盯着餐厅做——你可以做别的事，等外卖到了再吃。TMA 让 GPU 也能"下完单就去做别的"。

### 6.2 提高带宽利用率

**问题**：如果搬运之间有间隔（因为要等计算），带宽就用不满。

**TMA pipelining 的解法**：TMA 拷贝连续发起，没有间隔，带宽打满。

```
无 pipelining:  搬▓░░░搬▓░░░搬▓░░░    ← 带宽利用率 33%
有 pipelining:  搬▓▓▓▓▓▓▓▓▓▓▓▓         ← 带宽利用率 100%
                   算░░░░░░░░░░░
```

### 6.3 释放线程

**问题**：cp.async 时代，每个线程都要发搬运指令，占用大量线程时间。

**TMA pipelining 的解法**：TMA 是独立硬件单元，**1 个线程发起 1 条指令**就够，其他线程可以专心做计算。

**这就是为什么 Hopper 的"warpgroup"概念成立**——4 个 warp（128 线程）分工：
- 1 个 warp 专门发 TMA（"生产者"）
- 3 个 warp 专门算 GEMM/softmax（"消费者"）

### 6.4 配合 WGMMA 的异步特性

WGMMA（Hopper 的矩阵乘指令）也是异步的——发起后立即返回。所以可以：

```
TMA:  [搬块1][搬块2][搬块3]...
WGMMA:        [算块1][算块2][算块3]...
                   ↑
              搬和算完全并行
```

**TMA + WGMMA 的组合是 Hopper 性能爆发的核心**。两者都是异步，可以完美流水线。

---

## 七、TMA pipelining 的代价

### 7.1 共享内存压力

每个"在途"的 TMA 拷贝都要一个独立 buffer。如果用 4 级流水线，要 4 个 buffer——共享内存（256KB）会被吃掉一大块。

**FlashMLA 的应对**：用 seesaw 调度把输出矩阵也切分，减少寄存器/共享内存压力，给 TMA buffer 留空间。

### 7.2 同步复杂度

多个 TMA 拷贝在途时，要管理多个 mbarrier，确保"算块 N 时块 N 已就绪"——同步逻辑容易出 bug。

### 7.3 编程难度

TMA 要写 tensor map 描述符、管理 mbarrier、设计 buffer 周转——比 cp.async 复杂得多。CUTLASS 提供了 `cute::make_tma_copy` 等封装来降低难度，但学习曲线仍陡峭。

---

## 八、TMA pipelining 和普通 pipelining 有什么区别？

**"pipelining"是思想，"TMA pipelining"是具体实现**。

| 维度 | 普通 pipelining（cp.async 时代） | TMA pipelining（Hopper 时代） |
|------|--------------------------------|------------------------------|
| 搬运指令 | `cp.async`（16 字节粒度） | `cp.async.bulk.tensor`（多维块） |
| 指令数 | 多（要循环） | 少（一条搬一块） |
| 同步 | `cp.async.wait_group` | `mbarrier`（更精细） |
| 线程占用 | 多个线程搬运 | 1 个线程发起 |
| 形状感知 | 无（要手动算地址） | 有（tensor map 描述多维形状） |
| 缓存控制 | 弱 | 强（`EVICT_FIRST` 等） |

**TMA pipelining 是普通 pipelining 的升级版**——思想一样，但实现更高效、更精细。

---

## 九、跟 FlashMLA 的关系

### 9.1 FlashMLA 为什么必须用 TMA pipelining？

回顾前面的 roof-line 分析：**MLA 解码是计算密集型**——算力是瓶颈，带宽有富余。

但"带宽有富余"不等于"延迟可以忽略"。如果数据没及时到位，Tensor Core 仍然会空闲——这在计算密集型场景下是灾难性的（算力本就紧张，不能再浪费）。

**TMA pipelining 解决的就是"延迟"问题**：

- 带宽：够用（计算密集型）
- 算力：紧张（要满载）
- 延迟：必须隐藏（不然算力浪费）

**TMA pipelining 让 Tensor Core 永不等料**——这是把算力利用率推到 80% 的关键。

### 9.2 FlashMLA 的两个 TMA 优化

博客提到两个 TMA 相关的优化：

#### 优化 1：细粒度 TMA-GEMM pipelining

> 把 64×576 的 K 块切成 9 个 64×64 的小块，每搬完一块就开始算

**核心思想**：切小粒度，让搬运和计算高度重叠。

#### 优化 2：缓存提示 EVICT_FIRST

> 对 TMA 拷贝使用 `cute::TMA::CacheHintSm90::EVICT_FIRST`

**核心思想**：KV Cache 数据量巨大（几十 GB），L2 装不下；标记为"用完即驱逐"，避免污染 L2，让更重要的数据（Q、中间结果）留在 L2。

**两个优化配合**：细粒度切分 + 智能缓存管理 → 既隐藏延迟又避免缓存污染 → 算力利用率 80%+。

---

## 十、常见误区与 FAQ

### Q1：TMA 和 DMA 是一回事吗？

**不是**。DMA（Direct Memory Access）是 CPU 时代的概念，指外设和内存之间直接搬运数据不用 CPU 介入。

TMA 是 GPU Hopper 引入的专用单元，**功能类似 DMA 但更进一步**：
- 支持多维张量（DMA 只搬一维）
- 带 mbarrier 同步（DMA 通常用中断）
- 集成在 SM 内部（DMA 通常在芯片级总线）

可以理解为"TMA 是 DMA 的 GPU 版超集"。

### Q2：TMA pipelining 和 WGMMA 的异步有什么关系？

**两者配合才能完整流水线**：

- TMA 异步：搬数据时不用等
- WGMMA 异步：算矩阵时不用等

**单独一个不够**：
- 只有 TMA 异步，WGMMA 同步：搬运并行了，但计算要等
- 只有 WGMMA 异步，TMA 同步：计算并行了，但搬运要等
- **两者都异步**：搬运和计算完全并行

Hopper 的设计哲学：**让所有重操作都异步**，然后通过 mbarrier 协调。

### Q3：为什么是 64×64 的小块？不是 32×32 或 128×128？

这是工程权衡：

- **太小（32×32）**：TMA 指令数变多，开销增大
- **太大（128×128）**：等待时间长，重叠程度低
- **64×64**：Hopper 的 WGMMA 单次形状是 64×N×16，匹配得最好

**64 是 Hopper 架构的"自然粒度"**——和 WGMMA 的 M 维一致，可以直接喂给 Tensor Core。

### Q4：TMA 一定要用共享内存吗？能直接搬到寄存器吗？

**不能直接搬到寄存器**。TMA 的目的地只能是**共享内存**。

数据流是：`HBM →(TMA)→ SMEM →(手动加载)→ 寄存器 →(WGMMA)→ 寄存器`

**所以 TMA pipelining 实际上是 HBM 和 SMEM 之间的流水线**，SMEM 到寄存器还要单独管理。

### Q5：Blackwell（B100）的 TMA 升级了吗？

**有升级**。Blackwell 引入了 TMEM（Tensor Memory），数据流变成：

```
HBM →(TMA)→ SMEM →(TMA 变种)→ TMEM →(UMMA)→ TMEM
```

TMEM 是 Tensor Core 专用内存，比寄存器大。Blackwell 的 TMA 可以搬到 TMEM，进一步减少通用寄存器压力。

详见 [FlashMLA-GEMM三次变革详解.md](FlashMLA-GEMM三次变革详解.md)。

### Q6：TMA pipelining 能在 Volta/Ampere 上用吗？

**不能用 TMA**（TMA 是 Hopper 才有的硬件），但**pipelining 思想可以用**：

- Volta/Ampere 用 `cp.async` + 多缓冲区实现 pipelining
- 性能比 TMA 差，但思想一样

**TMA 是 Hopper 的"硬件加速 pipelining"**——把需要软件做的大量工作硬件化。

### Q7：TMA pipelining 的"级数"是什么意思？

**级数 = 同时在途的 TMA 拷贝数**。

- 2 级 = 2 个 buffer 交替（ping-pong）
- 3 级 = 3 个 buffer 轮转
- 4 级 = 4 个 buffer 轮转

**级数越多，隐藏延迟能力越强**，但共享内存压力越大。FlashMLA 根据场景用 2-4 级。

### Q8：TMA pipelining 和 CPU 的指令流水线是一回事吗？

**思想一样，实现不同**。

- **CPU 指令流水线**：把一条指令的执行分成取指/译码/执行/写回等阶段，多条指令重叠
- **TMA pipelining**：把"搬数据"和"算数据"两个阶段重叠

两者都是"把串行变并行"的思想，只是粒度不同——CPU 流水线在指令级，TMA pipelining 在"搬运-计算"级。

---

## 十一、一张图回顾 TMA pipelining

```
                没有 pipelining（串行）
                ─────────────────────
时间轴 →
TMA:  [──────搬块1──────][          ][──────搬块2──────][          ]
GEMM: [                ][────算块1────][                ][────算块2────]
                                                     
                              总时间 = 2×(搬+算)


                有 TMA pipelining（流水线）
                ────────────────────────
时间轴 →
TMA:  [──搬块1──][──搬块2──][──搬块3──][──搬块4──]
GEMM: [        ][──算块1──][──算块2──][──算块3──][──算块4──]
                              ↑
                    搬块N+1 和 算块N 同时

                              总时间 ≈ max(搬, 算) × N


                FlashMLA 的细粒度版本
                ─────────────────────
原本搬一个 64×576 块要等很久 → 切成 9 个 64×64 小块
                              
TMA:  [搬1][搬2][搬3]...[搬9]
GEMM:    [算1][算2]...[算8][算9]
              ↑
        搬小块2 和 算小块1 同时
```

---

## 十二、一句话总结

> **TMA pipelining = 用 Hopper 的 TMA 硬件单元，把"搬数据"和"算数据"重叠起来，让 Tensor Core 永不等料。**
>
> - **TMA**：Hopper 引入的专用张量搬运单元，异步、多维、带 mbarrier 同步
> - **pipelining**：搬下一块数据的同时算上一块数据，串行变并行
> - **FlashMLA 的细粒度版本**：把 64×576 切成 9 个 64×64，每搬完一块就开始算，最大化重叠
>
> **核心价值**：在计算密集型场景下让 Tensor Core 满载（80%+ 利用率），这是 FlashMLA 达到 660 TFlops 的关键。

---

## 附：术语速查表

| 术语 | 含义 |
|------|------|
| TMA | Tensor Memory Accelerator，Hopper 引入的张量搬运单元 |
| cp.async | Volta/Ampere 时代的异步拷贝指令，TMA 的前身 |
| mbarrier | memory barrier，GPU 上的同步原语，TMA 用它发完成信号 |
| SMEM | Shared Memory，共享内存，TMA 的目的地 |
| HBM | High Bandwidth Memory，显存，TMA 的源 |
| Tensor Core | GPU 上专做矩阵乘的计算单元 |
| WGMMA | Hopper 的异步矩阵乘指令，和 TMA 配合做流水线 |
| pipelining | 流水线，把串行操作变并行 |
| ping-pong buffer | 双缓冲，最简单的 pipelining 实现 |
| EVICT_FIRST | L2 缓存提示，标记数据"用完即驱逐" |
| warpgroup | 4 个 warp = 128 线程的协作单元，Hopper 引入 |

---

## 附：进一步阅读

- [FlashMLA-Kernel深度解读.md](FlashMLA-Kernel深度解读.md) —— FlashMLA 整体技术解读（含 TMA pipelining 部分）
- [FlashMLA-GEMM三次变革详解.md](FlashMLA-GEMM三次变革详解.md) —— WGMMA/UMMA 与 TMA 的配合
- [FlashMLA-计算访存比详解.md](FlashMLA-计算访存比详解.md) —— 为什么计算密集型仍需关心延迟
- 原博客：`docs/20250422-new-kernel-deep-dive.md`
- Hopper TMA 白皮书：NVIDIA H100 Tensor Core 白皮书
- CUTLASS CuTe 文档：https://github.com/NVIDIA/cutlass
