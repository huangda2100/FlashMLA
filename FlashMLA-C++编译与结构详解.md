# FlashMLA 的 C++ 代码结构详解

> 用最通俗的语言，说清楚 `csrc` 目录下的所有 C++ 文件是怎么组织的、怎么编译的、怎么跑起来的。

---

## 一、C++ 源文件的两种角色（.h vs .cpp/.cu）

想象你在盖一栋楼：

- **`.h` 头文件** = **设计图纸（蓝图）**。它告诉别人："这里有一堵墙，长这样、宽这样、门在这里。"图纸本身不是墙，但它规定墙应该长什么样。

- **`.cpp` / `.cu` 源文件** = **实际施工**。施工队拿着图纸，把真正的墙砌起来。墙能砌在多个地方（多个施工队有不同的施工方式）。

在 C++ 里，头文件（.h）只放**声明**（函数签名、类定义、宏、模板），而源文件（.cpp / .cu）放**实现**（函数体、模板实例化）。

### 具体例子

看 FlashMLA 中的一个实际例子：

**图纸层（头文件）：** `csrc/api/sparse_fwd.h`

```cpp
// 声明：不写具体怎么做，只写它的"签名"
class Fwd_Sm90_Impl : public FwdImplBase {
    void run_(const SparseAttnFwdParams &params, ...) override;
};
```

**施工层（源文件）：** `csrc/sm90/prefill/sparse/fwd.cu`

```cpp
// 实现：这里才真正写具体怎么做
void sm90::run_fwd_kernel(const SparseAttnFwdParams& params) {
    // ... 实际的 CUDA kernel launch 代码
}
```

### 为什么需要 .h + .cu 这种两文件结构？

核心原因：**C++ 是一个需要"先声明、后使用"的语言**。编译器读文件是从上往下读的，如果它看到一个函数被调用但还没见过它的声明，它就会报错。

假设没有头文件，只有 `.cu` 源文件：

```
// file: a.cu
void run_kernel() { helper_func(); }  // ❌ 报错！helper_func 还没声明
// 非要写在下面...
void helper_func() { ... }
```

你可以把 `helper_func` 写在上面，但如果有 10 个文件互相调用，谁都放在谁上面是不可能的。**头文件就是破解这个困局的方案**：把所有"签名"集中在头文件，所有源文件只需要 `#include` 它就能使用。

---

## 二、C++ 编译的完整流程（这是一切的基础）

C++ 的编译分为 **4 个阶段**，理解这个你就理解了一半的工程：

```
源代码(.h + .cu)   ──①预处理器──→  展开后的代码   ──②编译器──→  汇编(.s)
  ──③汇编器──→  目标文件(.o)   ──④链接器──→  可执行文件/动态库(.so)
```

### ① 预处理器（Preprocessor）

> 处理所有以 `#` 开头的东西

```cpp
#include "common.h"    // 把 common.h 的全文复制粘贴到这里
#define LOG_2_E 1.442  // 把所有 LOG_2_E 替换成 1.442
#pragma once           // 告诉预处理器"这个文件只被粘贴一次，不要重复粘贴"
```

你写的 `#include` 本质上就是**文本复制粘贴**。编译器的预处理器把被 include 的文件内容完全展开、替换宏，生成一个纯的 C++ 代码文件。

**关键推论：** 如果一个头文件被 10 个源文件 `#include`，它的内容就被复制粘贴了 10 次！这就是为什么头文件尽量只放声明不放实现——否则每个源文件都编译一遍同样的东西，浪费时间。

### ② 编译器（Compiler）

> 把 C++ 代码翻译成机器指令（汇编）

这是最耗时的步骤。每个 `.cu` 文件被**独立编译**成一个 `.o` 目标文件。

**核心原则：一个 `.cu` 文件 = 一个编译单元。** 每个编译单元之间完全独立、互不知晓。

这就是为什么在 `a.cu` 里写了某个函数，`b.cu` 即使不知道它的实现细节也能调用它——只要通过头文件知道它的签名就行。实现细节的"牵线搭桥"是链接器的工作。

### ③ 汇编器（Assembler）

> 把汇编代码转成二进制机器码，生成 `.o` 文件（目标文件）

`.o` 文件已经是二进制了，但还**不能运行**，因为里面有"未解决的引用"——比如 `a.cu` 调用了 `b.cu` 里的函数，但 `a.o` 里并不知道 `b.o` 里的函数地址是什么。

### ④ 链接器（Linker）

> 把所有 `.o` 文件拼在一起，填上所有未解决的地址

链接器做的是**拼图**工作：把 `a.o` 里调用的函数地址填上 `b.o` 里对应函数的实际地址，最终生成一个完整的 `.so` 动态库（So file = Shared Object，Linux 上的共享库）。

---

## 三、C++ 的 #include 机制详解

`#include` 看似简单，但很多初学者对它理解有误。

### `#include` 就是文本复制粘贴

当你在 `api.cpp` 里写：

```cpp
#include "sparse_fwd.h"
```

等价于编译器把 `sparse_fwd.h` 的全部内容拿出来，粘贴到 `#include` 这一行的位置。仅此而已。

### 引号 `""` vs 尖括号 `<>`

- `#include "sparse_fwd.h"` — **从当前源文件所在目录开始找**，找不到再去系统路径找。用于项目内部文件。
- `#include <torch/extension.h>` — **直接去系统路径找**。（尖括号优先系统路径，速度更快。）

### 为什么不能循环 include？

`a.h` 包含 `b.h`，`b.h` 又包含 `a.h` —— 这就变成了无限递归！所以 C++ 有 `#pragma once` 来防止这种情况：每一个头文件在同一个编译单元中只被展开一次。

---

## 四、FlashMLA 中特有的 .h / .cuh / .cu 文件

FlashMLA 是一个 **CUDA 项目**（NVIDIA GPU 编程），所以除了标准 C++ 的概念外，还有 CUDA 特有的约定：

### 文件后缀的约定

| 后缀 | 是什么 | 作用 | 编译方式 |
|------|--------|------|----------|
| `.h` | 纯 C++ 头文件 | 声明和模板定义，**不包含 CUDA 特有代码** | 不独立编译，通过 `#include` 嵌入 |
| `.cuh` | CUDA 头文件 | 包含 CUDA 特有代码（`__global__`、`__device__`、`cudaStream_t` 等） | 不独立编译，通过 `#include` 嵌入 |
| `.cpp` | C++ 源文件 | 纯 CPU 端代码 | NVCC → 宿主编译器(GCC/Clang) → `.o` |
| `.cu` | CUDA 源文件 | 可能同时包含 CPU 端和 GPU 端代码 | NVCC 双路编译（GPU 路径 + CPU 路径）→ `.o` |

### 一个关键的区分：.h vs .cuh

在 FlashMLA 中：

- `.h` 放的是**不含 CUDA 关键字**的声明和参数结构体
- `.cuh` 放的是**实际 CUDA kernel 定义**（包含 `__global__`、`__device__`、`__shared__` 等 GPU 专属关键字）

```
csrc/sm90/decode/dense/
├── splitkv_mla.h      ← 只是声明
├── splitkv_mla.cuh     ← 实际 CUDA kernel 实现 (__global__ void kernel...)
├── config.h            ← 只是配置常量
├── traits.h            ← 数据类型 traits
└── instantiations/
    ├── fp16.cu          ← 模板实例化（实际编译入口）
    └── bf16.cu          ← 模板实例化（实际编译入口）
```

---

## 五、FlashMLA 的具体编译流程

把 `setup.py` 中描述的过程翻译成大白话：

### 第 1 步：准备 CUTLASS

```
git submodule update --init csrc/cutlass
```

FlashMLA 依赖 NVIDIA 的 CUTLASS 库（通用 CUDA 矩阵乘法模板库），首先要把它拉下来。

### 第 2 步：确定 GPU 架构

```
NVCC 版本检测 → 决定编译 SM90 (H100) 还是 SM100 (B200) 代码
```

FlashMLA 支持两种 NVIDIA GPU 架构：
- **SM90** = Hopper 架构（H100 / H800 GPU）
- **SM100** = Blackwell 架构（B200 GPU）

不同的架构需要不同的编译标志和不同的代码路径。

### 第 3 步：逐个编译源文件

`setup.py` 中列出了要编译的 `sources` 列表，例如：

```python
sources=[
    "csrc/api/api.cpp",                                    # Python-CUDA 桥接层
    "csrc/smxx/decode/get_decoding_sched_meta/xxx.cu",     # 调度器
    "csrc/smxx/decode/combine/combine.cu",                 # 合并 kernel
    "csrc/sm90/decode/dense/instantiations/fp16.cu",       # 密集解码 FP16
    "csrc/sm90/decode/dense/instantiations/bf16.cu",       # 密集解码 BF16
    "csrc/sm90/decode/sparse_fp8/instantiations/xxx.cu",   # 稀疏解码
    "csrc/sm90/prefill/sparse/fwd.cu",                     # 稀疏预填充
    "csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",  # SM100 密集 MHA
    "csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",  # SM100 密集 MHA 反向
    # ... 共计约 20 个 .cu/.cpp 文件
]
```

每个 `.cu` 文件被 **NVCC（NVIDIA CUDA 编译器）** 独立编译成一个 `.o` 文件。

### 第 4 步：链接成 Python 扩展

所有 `.o` 文件被链接成一个 `.so` 动态库文件。这个 `.so` 文件就是 Python 可以直接 `import` 的 C 扩展模块。

Python 中 `import flash_mla.cuda` 时，实际上加载的就是这个编译好的 `flash_mla/cuda.so` 文件。

---

## 六、模板与实例化——FlashMLA 中最重要的 C++ 概念

模板（Template）是 C++ 中最强大的特性，也是 FlashMLA 中最核心的工程技巧。

### 什么是模板？

模板就是**C++ 中的"填空"**——你写一份代码，里面留一些空（类型、数值），让编译器在编译时填入具体的值。

```cpp
// 模板定义：一个通用的加法器
template<typename T>
T add(T a, T b) { return a + b; }

// 使用方式
add<int>(1, 2);        // 编译器生成：int add(int, int)
add<float>(1.0, 2.0);  // 编译器生成：float add(float, float)
```

模板的好处是：**不需要为每种类型手写一遍同样的代码**。

### FlashMLA 中模板的巨大威力

看 `csrc/api/common.h` 中的 `DISPATCH_HEAD_DIM` 宏：

```cpp
#define DISPATCH_HEAD_DIM(HEAD_DIM, CONSTEXPR_NAME, ...) \
[&] () { \
    if (HEAD_DIM == 576) { \
        static constexpr int CONSTEXPR_NAME = 576; \
        return __VA_ARGS__(); \
    } else if (HEAD_DIM == 512) { \
        static constexpr int CONSTEXPR_NAME = 512; \
        return __VA_ARGS__(); \
    } ...
```

当用户传进来 `head_dim=576` 时，它**在运行时判断、但在编译时选择不同的模板实例化**。因为在 CUDA 编程中，很多优化（比如循环展开、共享内存大小、寄存器分配）必须在**编译时就知道维度**。如果你用运行时变量，性能会大幅下降。

这就是为什么 FlashMLA 要把 `head_dim=576` 和 `head_dim=512` 分别编译成独立的 kernel——因为它们在 GPU 上运行的机器码完全不同。

### 为什么需要 .cu 文件里的 "instantiations"？

看 `csrc/sm90/prefill/sparse/` 的结构：

```
fwd.h               ← 模板声明
phase1.cuh          ← 模板实现（完整的 CUDA kernel）
instantiations/
    phase1_k512.cu       ← 实例化: head_dim=512 的情况
    phase1_k576.cu       ← 实例化: head_dim=576 的情况
    phase1_k512_topklen.cu
    phase1_k576_topklen.cu
```

C++ 模板有一个重要规则：**模板定义在 `.h` / `.cuh` 文件中，但如果不在 `.cu` 文件中显式实例化，编译器不会生成任何机器码。**

`phase1_k512.cu` 的内容非常简单：

```cpp
// 只有一两行，显式要求编译器为 head_dim=512 生成 kernel
#include "../phase1.cuh"
template void sm100::fwd::head128::run_fwd_phase1_kernel<512>(...);
```

它 `#include` 了实际的模板实现（`phase1.cuh`），然后告诉编译器："请为 head_dim=512 生成一份完整的 kernel。"

这样做的目的是：
1. **编译时间分摊**：每次 `.cu` 是独立的编译单元，可以并行编译
2. **避免重复编译**：修改 `phase1_k512.cu` 只需要重编这一个文件
3. **二进制大小控制**：只编译真正需要的组合，不浪费空间

---

## 七、用一个具体例子串起全部概念

假设 Python 调用 `flash_mla_sparse_fwd(q, kv, indices, ...)`，整个过程如下：

### 7.1 Python 端

```python
# flash_mla/flash_mla_interface.py
results = flash_mla_cuda.sparse_prefill_fwd(q, kv, indices, ...)
```

### 7.2 C++ API 层（桥接）

```cpp
// csrc/api/api.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
}
```

这里 PYBIND11 把 C++ 函数 `sparse_attn_prefill_interface` 注册成 Python 可调用的模块函数。

### 7.3 API 实现层（参数解析 + 调度）

```cpp
// csrc/api/sparse_fwd.h
static std::vector<at::Tensor> sparse_attn_prefill_interface(...) {
    // 1. 获取 GPU 架构信息
    Arch arch = Arch();           // ← 在 common.h 中定义
    bool is_sm90a = arch.is_sm90a();
    
    // 2. 参数检查和转换
    //    确保 tensor shape/dtype 正确、设备正确
    
    // 3. 打包参数结构体
    SparseAttnFwdParams params = {...};
    
    // 4. 根据架构和 head 数选择不同实现
    if (is_sm90a) {
        Fwd_Sm90_Impl fwd_impl;
        fwd_impl.run(params, required_features);
    } else {
        // SM100: 根据 topk 大小选择优化版本
        if (use_small_topk_impl) {
            small_topk_impl.run(...);
        } else {
            regular_impl.run(...);
        }
    }
}
```

### 7.4 CUDA Kernel 层（GPU 计算）

```cpp
// csrc/sm90/prefill/sparse/phase1.cuh
template<int HEAD_DIM_QK>
__global__ void sparse_phase1_kernel(...) {
    // 这里是在 GPU 上实际执行的代码
    // HEAD_DIM_QK 在编译时就是常量（576 或 512）
    // 编译器可以利用这个信息做极致优化
}
```

**完整数据流：**

```
Python 
  → C++ API (api.cpp) 
    → 参数结构体 (params.h) 
      → 调度器 (sparse_fwd.h) 
        → 实现类 (Fwd_Sm90_Impl) 
          → CUDA kernel (phase1.cuh) 
            → GPU 上执行
```

### 7.5 编译/运行分离

重要区分：

- **编译时**（你运行 `pip install` 时）：NVCC 把所有的 `.h` / `.cu` 编译成 `.so` 文件。模板参数在此刻确定，生成针对特定形状优化的 GPU 指令。
- **运行时**（你运行 Python 脚本时）：Python 加载 `.so` 到内存，调用具体函数。此时 C++ 代码已经在运行了。

---

## 八、FlashMLA 中的 C++ 设计模式一览

### 1. ImplBase 抽象基类模式（sparse_fwd.h）

```cpp
class ImplBase {     // ← 定义接口契约
    void run(params);
    virtual void run_(params) = 0;  // ← 子类必须实现
};

class Fwd_Sm90_Impl : public ImplBase { ... };   // SM90 实现
class Fwd_Sm100_Impl : public ImplBase { ... };  // SM100 实现
```

效果：运行时根据 GPU 型号自动选择实现，新增架构只需加一个新类。

### 2. Feature 检查模式

```cpp
class Fwd_Sm90_Impl {
    DECLARE_SUPPORTED_FEATURES(
        FwdFeatures::HEAD_64,
        FwdFeatures::HEAD_128,
        FwdFeatures::HEAD_DIM_512,
        FwdFeatures::HEAD_DIM_576,
        FwdFeatures::ATTN_SINK,
        ...
    )
};
```

每个实现类声明自己"我支持哪些功能组合"。运行时检查是否满足用户请求的所有功能，不满足时给出清晰报错。

### 3. DISPATCH 宏模式（编译时分支选择）

```cpp
DISPATCH_HEAD_DIM(params.d_qk, HEAD_DIM_QK, [&]() {
    // 这里 HEAD_DIM_QK 是编译期常量！
    // 编译器会为不同的值生成完全不同的机器码
    run_kernel<HEAD_DIM_QK>(params);
});
```

运行时判断、编译时分支——既有灵活性，又有极致性能。

### 4. PImpl + 显式模板实例化

模板实现在 `.cuh` 文件中，但不在头文件中实例化。开发者在 `instantiations/` 目录下的小 `.cu` 文件中显式实例化需要的组合。

---

## 九、常见疑问解答

### Q: 为什么 FlashMLA 源码大部分在 .h / .cuh 而不是 .cu？

因为这是**CUDA 模板项目**的典型风格。模板的定义通常放在头文件中，因为 C++ 编译器在实例化模板时需要看到完整的模板定义。如果把模板实现藏在 `.cu` 里，其他文件就无法使用了。

但这也意味着：**大量代码在头文件中 → 每个包含它的 .cu 文件都要编译一次 → 编译时间变长。** 所以 FlashMLA 要把模板实例化分散到 `instantiations/` 目录的多个 `.cu` 文件中，让它们可以并行编译、单独缓存。

### Q: `pragma once` 有什么用？

防止同一个头文件被一个 `.cu` 文件多次包含（间接 `#include` 导致的）。写下 `#pragma once` 后，预处理器保证这个文件在同一个编译单元中只被展开一次。

### Q: 头文件中的 inline 关键字是做什么的？

C++ 中，如果在头文件里定义了一个函数（不是声明），它默认会被标记为 `inline`。`inline` 的意思是："允许这个函数在多个编译单元中重复定义，编译器保证最终只保留一份。"没有 `inline`，如果在多个 `.cu` 中包含同一个头文件中的函数定义，链接器会报"重复定义"错误。

看 `csrc/api/common.h`：

```cpp
inline int int64_stride_to_int(int64_t orig_stride) {
    ...
}
```

这个函数定义在头文件中，被 `sparse_fwd.h`、`dense_decode.h` 等多个文件包含。如果没有 `inline`，链接时报重定义错误。

### Q: `__restrict__` 是什么？

CUDA/C++ 关键字，**告诉编译器不同的指针不会指向同一块内存**。这样编译器可以做更激进的优化（比如调整指令顺序），不用担心意外修改了同一块内存导致结果错误。

```cpp
float* __restrict__ lse;       // 保证没有其他指针指向 lse 指向的内存
cutlass::bfloat16_t* __restrict__ out;  // 同上
```

在 FlashMLA 中几乎每个函数参数都有 `__restrict__`——因为 GPU kernel 对指令级并行极度敏感，多一个优化机会就多一分性能。

---

## 十、总结：读完本节你应该能回答的问题

1. **`.h` 和 `.cu` 文件的本质区别是什么？** — 声明 vs 实现，头文件不独立编译，源文件独立编译。

2. **C++ 编译的四步流程是什么？** — 预处理(展开 include/define) → 编译(生成汇编) → 汇编(生成机器码 .o) → 链接(合并 .o → .so)。

3. **模板为什么在 CUDA 中特别重要？** — 因为 GPU 端无法忍受运行时分支的开销，模板让编译器在编译时就确定所有参数，生成极致优化的二进制代码。

4. **为什么 FlashMLA 的 `instantiations/` 目录有那么多 .cu 文件？** — 每个 .cu 是一个独立的编译单元，用于显式实例化一个特定参数组合的模板 kernel，支持并行编译并避免模板膨胀。

5. **`DISPATCH_HEAD_DIM` 这种宏的技巧是什么？** — 在"运行时判断"和"编译时选择"之间建立桥梁：运行时检查维度值，然后引导编译器为特定维度生成专门的 kernel。

6. **从 Python 调用到 GPU 执行，数据流是什么样的？** — Python → PYBIND11 → C++ 参数解析 → 架构调度 → CUDA kernel launch → GPU 执行。
