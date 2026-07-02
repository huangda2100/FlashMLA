# `csrc/kerutils/` —— CUDA kernel 的"工具箱"

## 这个目录解决什么问题？

写 CUDA kernel 不是写普通 C++ 代码。它有几个特殊痛点：

1. **GPU 指令不在标准 C++ 里**：要发起一次 TMA（Tensor Memory Accelerator）异步拷贝、一次 WGMMA 矩阵乘、一次 cluster barrier 同步——这些都需要写 inline PTX 汇编（`asm volatile(...)`），CUDA Runtime API 没有封装。
2. **不同 GPU 架构指令不一样**：SM80（A100）用 `cp.async.cg`，SM90（H100）用 TMA + WGMMA，SM100（B200）用 UTCMMA + 2-CTA TMA。每代架构都得自己写一套。
3. **PyTorch tensor 检查重复乏味**：每个 C++ 接口都要检查维度、dtype、device、stride、shape——同样的 `TORCH_CHECK` 写了几十遍。
4. **CUTLASS/CuTe 库有缺漏**：某些新指令（如 SM100 的 UTCMMA.ws）官方 CuTe 还没支持，或者支持了但行为不合适（如自带 `elect_one_sync` 难以复用）。

`csrc/kerutils/` 就是装这些**重复使用的工具**的箱子——把汇编封装成函数、把架构差异藏进宏、把 tensor 检查压成一行宏，让上层 kernel 代码专注于算法本身。

## 目录结构

```
kerutils/
└── include/
    └── kerutils/
        ├── kerutils.cuh              ← 总入口，include 它就拉入所有工具
        ├── common/common.h           ← 命名空间、打印宏、别名 ku = kerutils
        ├── host/host.h               ← CPU 侧工具（错误检查、tensor map 创建）
        ├── supplemental/
        │   └── torch_tensors.h       ← PyTorch tensor 检查宏（KU_CHECK_DEVICE 等）
        └── device/                   ← GPU 侧工具（运行在 kernel 里）
            ├── common.h              ← 通用类型 + 架构宏（KERUTILS_ENABLE_SM90 等）
            ├── device.cuh            ← device 总入口
            ├── sm80/                 ← A100 工具
            │   ├── intrinsics.cuh    ← cp.async.cg、createpolicy
            │   └── helpers.cuh       ← get_sm_id、shared memory 读写
            ├── sm90/                 ← H100 工具
            │   ├── intrinsics.cuh    ← st.async、TMA bulk reduce、cluster barrier
            │   └── helpers.cuh       ← WGMMA 封装、TMA copy、fragment 索引转换
            └── sm100/                ← B200 工具
                ├── intrinsics.cuh    ← TMA gather4、float2x2 加法等
                ├── helpers.cuh       ← UTCMMA SS/TS 封装、UMMA 布局
                ├── gemm.cuh          ← 扩展 CuTe：SM100 MMA.ws NOELECT 等自定义 trait
                └── tma_cta_group2_nosplit.cuh ← 扩展 CuTe：2-CTA TMA 不拆分版本
```

## 三层工具：host / device / supplemental

### 1. `host/` —— CPU 侧工具（不在 kernel 里运行）

**解决什么**：kernel 启动前后在 CPU 上做的杂事。

| 工具 | 作用 |
|------|------|
| `KU_CUDA_CHECK(call)` | 包装 CUDA 调用，失败时打印文件名行号 + 抛异常 |
| `KU_CUTLASS_CHECK(call)` | 同上，针对 CUTLASS API |
| `KU_ASSERT(cond)` | 无论是否 `-DNDEBUG` 都生效的断言 |
| `KU_CHECK_KERNEL_LAUNCH()` | kernel 启动后调一次，捕获异步错误 |
| `make_tensor_map(...)` | 创建 `CUtensorMap`（TMA 描述符），失败时打印所有参数 |
| `ceil_div(a, b)` | 整除向上取整，host/device 都能用 |

`make_tensor_map` 值得一说：TMA 是 H100 引入的硬件异步拷贝引擎，使用前要把 tensor 的形状/步长/分块打包成一个描述符（`CUtensorMap`），创建失败时 CUDA driver 只给个错误码。这里的封装在失败时会把 `size`、`strides`、`box_size` 等全部打印出来，否则你根本不知道哪个参数错了。

### 2. `device/` —— GPU 侧工具（在 kernel 里运行）

这是这个目录的**核心**。文件按 GPU 架构分（`sm80/`、`sm90/`、`sm100/`），每个架构下分 `intrinsics.cuh`（汇编封装）和 `helpers.cuh`（更高层的工具函数）。

#### 什么是 PTX 汇编？为什么需要它？

PTX（Parallel Thread Execution）是 NVIDIA 的虚拟指令集。GPU 的硬件能力（TMA、WGMMA、barrier）最终都要用 PTX 指令触发，但 C++ 没有这些指令。所以代码长这样：

```cpp
asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
    : 
    : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), 
      "r"(row_idxs.x), "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), 
      "r"(mbar_addr), "l"(cache_hint)
    : "memory"
);
```

这一长串汇编的意思是："用 TMA 的 gather4 指令，从全局内存按一组 row 索引批量取 4 个 tile 到共享内存，完成后通知 mbarrier。" 直接写汇编容易出错（寄存器约束、内存序、地址对齐），所以封装成函数 `tma_gather4(...)`，调用方只看到参数列表。

#### 三代架构的对比

| 架构 | GPU | 核心新指令 | 本目录封装 |
|------|-----|-----------|-----------|
| SM80 | A100 | `cp.async.cg`（异步拷贝到共享内存） | `cp_async_cacheglobal`、`create_fraction_based_cache_policy` |
| SM90 | H100 | TMA（张量异步拷贝）、WGMMA（warpgroup 矩阵乘）、cluster barrier | `wgmma`、`wgmma_ss`、`launch_tma_copy`、`st_async`、`get_peer_addr`、`tma_bulk_reduce_add` |
| SM100 | B200 | UTCMMA（tmem 矩阵乘）、2-CTA TMA、TMA gather4 | `utcmma_ss`、`utcmma_ts`、`tma_gather4`、`tma_gather4_cta_group_2`、自定义 MMA trait |

#### `common.h` 里的架构宏

```cpp
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define KERUTILS_ENABLE_SM80
#endif
// ... 同理 SM90 / SM90A / SM100 / SM100A
```

`__CUDA_ARCH__` 是 nvcc 编译时为每个 kernel 定义的宏，值代表架构（800=SM80, 900=SM90, 1000=SM100）。kernel 代码可以用 `#ifdef KERUTILS_ENABLE_SM90` 选择性启用 H100 专用代码路径。底层不支持 SM80 以下的 GPU（`static_assert(false, ...)`）。

#### `gemm.cuh` 和 `tma_cta_group2_nosplit.cuh`：扩展 CuTe

CuTe（CUTLASS 的 tensor 抽象层）提供了 SM100 MMA 的封装，但有两个问题：

1. **不支持 `.ws`（warp special）变体**：`SM100_MMA_F16BF16_WS_TS_NOELECT` 自己实现了一个，注释里说 "CuTe don't support UTCMMA with .ws, so we add it here"。
2. **2-CTA TMA 自带数据切分**：CuTe 的 `SM100_TMA_2SM_LOAD_1D` 会把数据在两个 CTA 之间切分，注释里吐槽 "which is really annoying to use"。所以这里写了 `SM100_TMA_2SM_LOAD_*_NOSPLIT`，两个 CTA 都拿到完整数据。

这两个文件甚至不在 `kerutils` 命名空间里——它们直接 `namespace cute` 往 CuTe 里加新类型，因为 CuTe 的 dispatch 机制是在 `cute` 命名空间里通过 trait 匹配的。

### 3. `supplemental/torch_tensors.h` —— PyTorch tensor 检查

**解决什么**：把重复的 `TORCH_CHECK` 压成一行。

```cpp
// 没有这个宏之前：
TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
TORCH_CHECK(q.dim() == 4, "q must have 4 dimensions");
TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
TORCH_CHECK(q.size(-1) == 1 || q.stride(-1) == 1, "q must have contiguous last dim");

// 有了之后：
KU_CHECK_DEVICE(q);
KU_CHECK_NDIM(q, 4);
KU_CHECK_DTYPE(q, torch::kBFloat16);
KU_CHECK_LAST_DIM_CONTIGUOUS(q);
```

**巧妙之处**：所有宏都支持 `at::Tensor` **和** `std::optional<at::Tensor>`。例如 `KU_CHECK_DEVICE(attn_sink)` 当 `attn_sink` 是 `nullopt` 时直接返回 `true`，不会误判。模板函数 `_check_optional_tensor` 用 `if constexpr` 区分两种类型。

另外两个工具函数：

- `get_tensor_ptr<T>(tensor)` —— 取 tensor 数据指针，没存储返回 `nullptr`
- `get_optional_tensor_ptr<T>(tensor_or_opt)` —— 同上但支持 optional，常用于把 optional tensor 转成可空指针传给 kernel

## 怎么使用

在 kernel 文件里 `#include <kerutils/kerutils.cuh>` 就能用所有工具。`common.h` 里定义了别名 `namespace ku = kerutils;`，所以可以这样写：

```cpp
#include <kerutils/kerutils.cuh>

__global__ void my_kernel(...) {
    // 用架构宏选代码路径
    #ifdef KERUTILS_ENABLE_SM90
        ku::wgmma<true>(tiled_mma, tCrA, tCrB, tCrC, /*zero_init=*/true);
    #endif
}
```

CPU 侧接口文件用 `<kerutils/supplemental/torch_tensors.h>`（间接包含 host 工具）：

```cpp
#include <kerutils/supplemental/torch_tensors.h>

KU_CHECK_DEVICE(q);
KU_CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
KU_CUDA_CHECK(cudaMalloc(...));
```

## 设计哲学

1. **按架构分文件**：新增 GPU 架构时，新建一个 `sm120/` 目录就行，不影响旧架构代码。
2. **汇编封函数**：一行 PTX 汇编难写难读，封装成有名字、有参数类型、有文档注释的 C++ 函数。
3. **架构宏做开关**：`KERUTILS_ENABLE_SM90` 让同一份代码在不同架构下编译出不同实现，避免 `#ifdef __CUDA_ARCH__` 散落各处。
4. **直接扩展 CuTe**：当库不够用时，往 `namespace cute` 里加 trait，让自定义指令能融入 CuTe 的 dispatch 体系。
5. **可选 tensor 友好**：所有检查宏同时支持 `Tensor` 和 `optional<Tensor>`，避免调用方写两套。
