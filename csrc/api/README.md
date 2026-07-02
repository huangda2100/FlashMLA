# `csrc/api/` —— Python 与 CUDA kernel 之间的"接待大厅"

## 这个目录解决什么问题？

想象一家工厂：

- **外面的人**（Python 代码）想下单：`flash_mla_cuda.sparse_decode_fwd(...)`
- **里面的车间**（`sm90/`、`sm100/` 里的 CUDA kernel）真正干活
- 但外面的人和车间之间不能直接对接——订单格式不对、车间不知道该开哪条产线、参数对不对都没人验

`csrc/api/` 就是这家工厂的**接待大厅**。它做的事情：

1. **接单**：通过 pybind11 把 C++ 函数注册成 Python 可调用的模块
2. **验货**：检查 tensor 的维度、数据类型、设备、stride 是否合法
3. **分诊**：判断 GPU 型号（H100 还是 B200）、head 数、head 维度，挑出合适的 kernel
4. **打包**：把零散的 tensor 包装成 kernel 需要的参数结构体
5. **回执**：执行完 kernel 后把输出 tensor 整理好返回给 Python

## 文件结构一览

| 文件 | 作用 | 类比 |
|------|------|------|
| `api.cpp` | 模块注册入口，把 5 个 C++ 函数暴露给 Python | 工厂前台挂的"服务项目"招牌 |
| `common.h` | 通用工具：GPU 架构检测、dispatch 宏、ImplBase 基类 | 接待员用的通用手册 |
| `dense_decode.h` | 稠密解码接口（`dense_decode_fwd`） | 标准订单处理流程 |
| `dense_fwd.h` | 稠密预填充接口（`dense_prefill_fwd`/`_bwd`） | 转交给 SM100 车间处理 |
| `sparse_decode.h` | 稀疏解码接口（`sparse_decode_fwd`） | 复杂订单处理流程 |
| `sparse_fwd.h` | 稀疏预填充接口（`sparse_prefill_fwd`） | 另一种复杂订单处理 |

## 调用链：从 Python 到 GPU

```
Python:  flash_mla_with_kvcache(...)
           ↓
Python:  flash_mla_cuda.sparse_decode_fwd(...)        ← PyTorch 扩展模块
           ↓
C++:     sparse_attn_decode_interface(...)            ← 本目录 sparse_decode.h
           ↓
C++:     某个具体实现类的 run_(...)                     ← 如 Decode_Sm90_Impl
           ↓
CUDA:    sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<...>(...)
           ↓
GPU:     真正开始计算
```

`api.cpp` 里 5 个暴露给 Python 的函数：

```cpp
m.def("sparse_decode_fwd",  &sparse_attn_decode_interface);   // 稀疏解码
m.def("dense_decode_fwd",   &dense_attn_decode_interface);    // 稠密解码
m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);  // 稀疏预填充
m.def("dense_prefill_fwd",  &FMHACutlassSM100FwdRun);         // 稠密预填充前向
m.def("dense_prefill_bwd",  &FMHACutlassSM100BwdRun);         // 稠密预填充反向
```

## 关键概念

### 1. `Arch` —— 当前 GPU 是什么型号？

`common.h` 里的 `Arch` 结构体在构造时读取当前 CUDA 设备属性，提供两个判断函数：

- `is_sm90a()` → H100 GPU（Hopper 架构）
- `is_sm100f()` → B200 GPU（Blackwell 架构）

不同架构走不同的 kernel 实现，这是分诊的第一步。

### 2. Dispatch 宏 —— 编译期选择模板参数

CUDA kernel 通常是模板函数，`head_dim=576` 和 `512` 是不同的特化版本。`common.h` 提供了一组宏：

```cpp
DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
    // 在这里 NUM_HEADS 是编译期常量（128 或 64）
    // 调用 kernel<NUM_HEADS>(...) 会触发对应的模板特化
});
```

为什么不用 if-else？因为模板参数必须是编译期常量，运行时的 if-else 没法选模板。这些宏本质上是把 if-else 包装成"在分支内调用 lambda"的形式。

### 3. `ImplBase` —— kernel 实现的统一合同

这是 `common.h` 里最重要的设计。每种 kernel 实现（如 `Decode_Sm90_Impl`）都继承自 `ImplBase<ParamsType, FeatureType>`，必须实现两个函数：

- `run_(params, required_features)` —— 真正调用 kernel
- `get_supported_features()` —— 声明自己支持哪些功能

调用时的流程：

```
用户需要 [HEAD_128, HEAD_DIM_576, ATTN_SINK]
           ↓
检查实现的支持列表：[HEAD_64, HEAD_128, HEAD_DIM_512, HEAD_DIM_576, ...]
           ↓
全部命中？→ 调用 run_()
缺失某项？→ 报错，列出哪些功能不被支持
```

这个设计的好处：**新增一个 kernel 实现时，只需声明它支持哪些功能，dispatcher 会自动选对**。不需要在调用方写一堆 if-else。

### 4. 功能枚举（Features）

每种接口定义自己的功能枚举：

- `DecodeFeatures`（在 `sparse_decode.h`）：`HEAD_64`、`HEAD_128`、`HEAD_DIM_576`、`V32_KVCACHE_FORMAT`、`ATTN_SINK`、`TOPK_LENGTH`、`EXTRA_KVCACHE` 等
- `FwdFeatures`（在 `sparse_fwd.h`）：`HEAD_64`、`HEAD_128`、`HEAD_DIM_576`、`ATTN_SINK`、`SINK_LSE`、`TOPK_LENGTH` 等

这些是"功能开关"——用户需要哪些功能、实现支持哪些功能，都用这些枚举表示。

## 一个具体例子：`dense_decode_fwd` 的流程

`dense_decode.h` 里的 `dense_attn_decode_interface` 是相对简洁的范例：

```
步骤 1: Arch()              → 判断 GPU 型号（必须是 SM90a）
步骤 2: TORCH_CHECK(...)    → 检查 dtype、device、stride、shape
步骤 3: torch::empty(...)   → 分配输出 tensor（out、lse）
步骤 4: 如果没传 tile_scheduler_metadata，就分配并调用
        smxx::decode::run_get_decoding_sched_meta_kernel 生成调度方案
步骤 5: 填充 DenseAttnDecodeParams 结构体
步骤 6: 根据 dtype 调用 sm90::run_flash_splitkv_mla_kernel<bf16/half>
步骤 7: 调用 smxx::decode::run_flash_mla_combine_kernel 合并 Split-KV 结果
步骤 8: reshape 输出 tensor，返回 {out, lse, tile_scheduler_metadata, num_splits}
```

注意第 4 步——**懒初始化**（lazy init）。调度元数据第一次算好后返回给 Python，下次调用时 Python 可以把同一个 tensor 传回来复用，避免重复计算。

## 为什么要单独分这一层？

如果把参数检查、kernel 选择、输出分配都塞进 kernel 文件里，代码会变得难以维护。分成三层后：

- **`api/`**：只关心"怎么和 Python 对接"——参数检查、tensor 分配、调度策略
- **`sm90/`、`sm100/`、`smxx/`**：只关心"怎么在特定 GPU 上算得快"——kernel 实现
- **`params.h`**：只定义数据结构，不含逻辑

新增一个 GPU 架构（比如未来的 SM120）时，只需要在 `sm120/` 加 kernel 实现，在 `api/` 加一个 `Impl` 类声明支持列表，其余代码不动。
