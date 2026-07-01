from typing import Optional, Tuple
import dataclasses

import torch

import flash_mla.cuda as flash_mla_cuda


# ============================================================================
# FlashMLASchedMeta —— 解码阶段的"调度管家"
#
# 这是什么问题？
#   大模型生成文字是逐字进行的（这叫"自回归"）。每生成一个字，都要拿
#   当前这个字的 Q（查询向量）去和之前所有字的 KV（缓存的历史信息）
#   做一次注意力计算。所以每生成一个字就要调用一次 flash_mla_with_kvcache。
#
#   问题在于：每次调用时，GPU 该让哪个计算单元（SM）处理哪一段数据？
#   这就好比一个工厂有几十条产线（SM），来了一批订单（多个请求），
#   每张订单要查看一堆档案（KV Cache）。"调度"就是决定"产线A处理订单1
#   的档案第1-100页，产线B处理订单1的档案第101-200页..."这样的分工方案。
#
#   好消息是：只要 batch size（订单数）、序列长度（档案页数）等参数不变，
#   这个分工方案其实每次都是一样的。所以第一次算好后存下来，后面直接复用。
#
# 这个类做了什么：
#   1. 缓存"分工方案"（tile_scheduler_metadata、num_splits）
#   2. 记录第一次调用时的配置参数（batch size、head 数等）
#   3. 后续调用时检查参数有没有变——变了就报错，避免用了过期的分工方案
# ============================================================================
@dataclasses.dataclass
class FlashMLASchedMeta:
    """
    A class that stores the tile scheduler metadata of FlashMLA
    """

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int

        causal: bool
        is_fp8_kvcache: bool
        topk: Optional[int]

        extra_page_block_size: Optional[int]
        extra_topk: Optional[int]

    have_initialized: bool = False

    config: Optional[Config] = None

    tile_scheduler_metadata: Optional[torch.Tensor] = None   # (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
    num_splits: Optional[torch.Tensor] = None                # (1), dtype torch.int32.


# ============================================================================
# get_mla_metadata —— 创建调度管家对象（空壳版）
#
# 以前 vs 现在：
#   旧版：用户必须自己先算好调度方案，传给这个函数，才能开始解码。
#   新版：用户什么都不用管，第一次调用 flash_mla_with_kvcache 时自动算。
#   我们保留这个函数只是为了让旧代码不报错——它不再需要任何参数。
#
# *args / **kwargs 是什么意思？
#   旧代码调用时会传很多参数进来。新版虽然不用这些参数了，但如果直接
#   删掉参数，旧代码就会报错（"传了多余的参数"）。所以用 *args, **kwargs
#   来"吃掉"所有参数——不管传什么进来，我们都不理，直接返回一个空壳对象。
#
# 返回值：
#   第一个 = 调度管家空壳（FlashMLASchedMeta()）
#   第二个 = None（历史遗留，没用）
# ============================================================================
def get_mla_metadata(
    *args,
    **kwargs
) -> Tuple[FlashMLASchedMeta, None]:
    """
    Returns an empty instance of FlashMLASchedMeta. The actual scheduling metadata will be generated during the first invocation of flash_mla_with_kvcache.

    Arguments:
        This function does not need any arguments, but we keep *args and **kwargs to be compatible with the old interface.

    Return:
        A tuple. Due to historical reasons, we return a tuple of (FlashMLASchedMeta, None) now. Only the first element is useful.
    """
    return FlashMLASchedMeta(), None


# ============================================================================
# flash_mla_with_kvcache —— 解码阶段的核心：做一次注意力计算
#
# 先理解"解码"是什么：
#   大模型生成文字 = 不断重复"看前面的字 → 猜下一个字"的循环。循环中每次
#   只生成一个字。这个函数就是"看前面的字"这一步的核心计算。
#
# 为什么叫"with_kvcache"（带 KV 缓存）？
#   每次猜下一个字时，都要看前面已经生成的字的 K 和 V。如果每次都重新算，
#   那就太慢了。所以把之前所有字的 K 和 V 存下来（叫 KV Cache），每次
#   只需要拿当前这个字的 Q 去和缓存里的 K/V 做计算。这就是"带 KVCache"的含义。
#
# 这个函数每次调用做了什么：
#   输入：当前 token 的 Q + 整个 KV Cache → 输出：注意力计算结果
#   （被模型的下一层拿去预测下一个 token）
#
# 支持的两种模式：
#   [密集模式] indices=None → 每个 query 关注 KV Cache 里的所有 token
#   [稀疏模式] indices≠None → 每个 query 只看指定的部分 token（节省计算）
#   稀疏模式是 DeepSeek 的优化——不是所有历史信息都重要，只看关键的几个就行。
#
# 函数内部的流程：
#   步骤 A：参数准备（取别名、校验类型、提取维度信息、设置默认值）
#   步骤 B：第一次调用 → 创建"调度方案"（谁处理哪部分数据）+ 校验参数
#           后续调用 → 检查参数是否变了（变了就报错）
#   步骤 C：根据密集/稀疏模式，调用对应的 C++ CUDA kernel
#   步骤 D：更新调度方案供下一次调用复用
# ============================================================================
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
                Different modes (including fp8/bf16, and sparsity) has different KV cache layouts. See comments below for details.
                The KV cache must be contiguously valid for sparse attention on sm100. Here "contiguously valid" means that every byte, from the very beginning of the KV cache, till the last byte in the KV cache, is valid memory address to visit (i.e. won't IMA). In other words, the KV cache could be a slice of a larger array, but cannot be a list of disjoint memory blocks.
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32. Can be None when sparse attention is used.
        cache_seqlens: (batch_size), torch.int32. Can be None when sparse attention is used.
        head_dim_v: Head_dim of v. Must be 512
        sched_meta: FlashMLASchedMeta, return by get_mla_metadata. You may reuse the same sched_meta across different invocations, but only when the tensor shapes and the values of cache_seqlens, topk_length, and extra_topk_length remain the same.
        num_splits_placeholder: must be "None" (to be compatible with the old interface).
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim_k).
        causal: bool. Whether to apply causal attention mask. Only valid for dense attention
        is_fp8_kvcache: bool.
        indices: (batch_size, seq_len_q, topk). KV indices when sparse attention is enabled.
                    Pay attention that indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * block_size + (the offset of token t among the page block),
                    where t is the k-th token of the j-th q-sequence in the i-th batch.
        attn_sink: Optional[torch.Tensor], (num_heads_q, ), torch.float32. If presented, the final output will be scaled by exp(lse) / (exp(lse) + exp(attn_sink)). Have no affect on the returned softmax_lse. +inf will cause the result to become 0.
        extra_k_cache and extra_indices_in_kvcache: If provided, will attend to these extra tokens in addition to those in k_cache and indices_in_kvcache. Their format requirements are the same as k_cache and indices_in_kvcache respectively.
        topk_length/extra_topk_length: (batch_size, ), torch.int32. If provided, only the leftmost topk_length indices will be processed. Useful when the actual topk for different queries are different so that we can save some computation, compared to masking.
    
    For DeepSeek V3, DeepSeek V3.1, and DeepSeek V3.2:
        head_dim should be 576 while head_dim_v should be 512.
        In FP8+sparse mode, each token's KV cache is 656 Bytes, structured as:
            - The shape of the tensor `k_cache` is (num_blocks, page_block_size, num_heads_k, head_dim), and num_heads_k must be 1.
            - First 512 bytes: The "quantized NoPE" part, containing 512 float8_e4m3 values.
            - Next 16 bytes: Scale factors, containing 4 float32 values. The first float32 is the scale for the first 128 float8_e4m3 values, the second for the next 128, and so on.
            - Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    # ====== 步骤 A：参数准备 ======

    # 给参数取短名，方便后面写代码。就像给"张三"起个外号叫"阿三"。
    sched_meta = tile_scheduler_metadata
    indices_in_kvcache = indices

    # 检查参数类型。如果类型不对，在这里就报错，比跑到 C++ 里崩掉好排查得多。
    assert isinstance(sched_meta, FlashMLASchedMeta), "tile_scheduler_metadata must be of type FlashMLASchedMeta"
    assert num_splits is None, "num_splits must be None"

    # 从 indices 的形状推断"每个 query 关注几个 token"（topk）。
    # 比如 indices.shape = (batch=4, seq=1, topk=64) → topk=64
    # 如果没用稀疏模式（indices=None），topk=None。
    topk = indices_in_kvcache.shape[-1] if indices_in_kvcache is not None else None

    # 如果有"额外 KV Cache"（有些模型结构需要），获取它的 page block 大小和 topk。
    extra_k_page_block_size = extra_k_cache.shape[1] if extra_k_cache is not None else None
    extra_topk = extra_indices_in_kvcache.shape[-1] if extra_indices_in_kvcache is not None else None

    # 设置 softmax 缩放系数，默认 = 1/sqrt(head_dim_k)。
    # 为什么要有这个缩放？Q*K^T 的内积会随着向量维度增大而变大，
    # 不缩放的话 softmax 会"两极分化"（一个接近1，其他接近0），梯度就没法传了。
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # ====== 步骤 B：调度方案的"懒初始化" ======

    if not sched_meta.have_initialized:
        # --- B1：第一次调用，需要创建调度方案 ---

        # 参数合法性检查：稀疏模式下不能用 causal mask（因果遮罩）。
        # 原因：稀疏模式已经指定了要看哪些 token，不需要 causal 再来限制一次。
        if indices_in_kvcache is not None:
            assert not causal, "causal must be False when indices_in_kvcache is not None (i.e. sparse attention is enabled)"

        # 把当前调用的所有配置参数存到 sched_meta.config 里。
        # 这样后续调用时可以对比一下：参数没变就用缓存的分工方案，变了就报错。
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],       # batch_size：这次同时处理几个序列？
            q.shape[1],       # seq_len_q：每个序列的 Q 长度（解码时通常是 1）
            q.shape[2],       # num_heads_q：Q 有几个"头"（MLA 是 128 或 64）
            k_cache.shape[1], # page_block_size：一个 KV 缓存块存几个 token
            k_cache.shape[2], # num_heads_k：K 有几个"头"（MLA 是 1，MQA）
            causal,           # 是否使用因果遮罩（只看历史，不看未来）
            is_fp8_kvcache,   # KV Cache 是不是 FP8 格式的？
            topk,             # 稀疏模式下，每个 query 关注几个 token？
            extra_k_page_block_size,  # 额外 KV Cache 的 block size
            extra_topk,               # 额外 KV Cache 的 topk
        )
    else:
        # --- B2：不是第一次调用了——检查参数有没有变 ---

        helper_msg = " Your input arguments are inconsistent with sched_meta. Please make sure the input arguments are consistent across different invocations of flash_mla_with_kvcache on the same sched_meta."
        assert sched_meta.config is not None
        # 下面逐一对比当前参数 vs 第一次调用时保存的参数。
        # batch_size 变了？ → 报错
        assert sched_meta.config.b == q.shape[0], "sched_meta.config.b must be equal to batch_size." + helper_msg
        # 序列长度变了？ → 报错
        assert sched_meta.config.s_q == q.shape[1], "sched_meta.config.s_q must be equal to seq_len_q." + helper_msg
        # head 数变了？ → 报错
        assert sched_meta.config.h_q == q.shape[2], "sched_meta.config.h_q must be equal to num_heads_q." + helper_msg
        # page block size 变了？ → 报错
        assert sched_meta.config.page_block_size == k_cache.shape[1], "sched_meta.config.page_block_size must be equal to page_block_size." + helper_msg
        # K 的 head 数变了？ → 报错
        assert sched_meta.config.h_k == k_cache.shape[2], "sched_meta.config.h_k must be equal to num_heads_k." + helper_msg
        # causal 标志变了？ → 报错
        assert sched_meta.config.causal == causal, "sched_meta.config.causal must be equal to causal." + helper_msg
        # FP8 标志变了？ → 报错
        assert sched_meta.config.is_fp8_kvcache == is_fp8_kvcache, "sched_meta.config.is_fp8_kvcache must be equal to is_fp8_kvcache." + helper_msg
        # topk 变了？ → 报错
        assert sched_meta.config.topk == topk, "sched_meta.config.topk must be equal to the last dim of indices_in_kvcache." + helper_msg
        # extra page block size 变了？ → 报错
        assert sched_meta.config.extra_page_block_size == extra_k_page_block_size, "sched_meta.config.extra_page_block_size must be equal to the page_block_size of extra_k_cache." + helper_msg
        # extra topk 变了？ → 报错
        assert sched_meta.config.extra_topk == extra_topk, "sched_meta.config.extra_topk must be equal to the last dim of extra_indices_in_kvcache." + helper_msg

    # ====== 步骤 C：调用 CUDA kernel 做实际计算 ======

    if topk is not None:
        # --- C1：稀疏注意力路径 ---
        # 用户给了 indices，指定了每个 query 只看哪些 token。
        # 相当于告诉模型："不用看全部历史，只看这 topk 个 KV 块就够了。"
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True when sparse attention is enabled"

        # 调用 C++ 的稀疏解码函数。C++ 层会：
        #   1. 解析 indices，找到每个 query 要看的 KV 块
        #   2. 启动 CUDA kernel，在 GPU 上并行计算注意力
        #   3. 返回注意力结果（out）和 log-sum-exp（lse，后面做 Softmax 用）
        #   4. 返回更新后的调度元数据（给下一次调用复用）
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale
        )
    else:
        # --- C2：密集注意力路径 ---
        # 没有 indices。每个 query 关注 KV Cache 里的所有 token。
        # 这是标准的注意力计算方式。

        # 密集模式下，稀疏模式的参数必须为 None
        assert indices_in_kvcache is None and attn_sink is None and extra_k_cache is None and extra_indices_in_kvcache is None and topk_length is None and extra_topk_length is None, "indices_in_kvcache, attn_sink, extra_k_cache, extra_indices_in_kvcache, topk_length and extra_topk_length must be None when dense attention is used."
        # 密集模式必须有 block_table（页表）和 cache_seqlens（每个序列的长度）
        assert block_table is not None and cache_seqlens is not None, "block_table and cache_seqlens must be provided when dense attention is used."

        # 调用 C++ 的密集解码函数。和稀疏不同的是需要传 block_table 和
        # cache_seqlens，因为在密集模式中 kernel 需要通过页表找到所有 KV 块。
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.dense_decode_fwd(
            q, k_cache, head_dim_v,
            cache_seqlens, block_table,
            softmax_scale, causal,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits
        )

    # ====== 步骤 D：更新调度方案 ======
    # C++ kernel 可能更新了调度元数据（比如 num_splits 变了）。
    # 把新值存回 sched_meta，下次调用直接用，不用重新算。
    sched_meta.tile_scheduler_metadata = new_tile_scheduler_metadata
    sched_meta.num_splits = new_num_splits

    # 返回注意力结果 + softmax 的 log-sum-exp（用于后续的数值计算）
    return (out, lse)


# ============================================================================
# flash_mla_sparse_fwd —— 预填充阶段的稀疏注意力
#
# 预填充（Prefill）是什么？
#   用户输入了一段文字（prompt），模型要一次性"看完"整段话，生成第一个回复字。
#   这个阶段叫预填充。和"解码"（逐字生成）不同，预填充是一次性处理所有 token。
#
# 为什么需要稀疏版本？
#   DeepSeek V3.2 的 DSA 机制认为：不需要算每个 token 和所有其他 token 的
#   注意力，只算和部分"重要 token"的注意力就够了——这叫"稀疏注意力"。
#   这大幅降低了预填充的计算量。
#
# 和 flash_mla_with_kvcache（解码）的区别：
#   [解码] Q 是当前 token → K/V 从缓存读取 → 逐个生成 → 每步调用一次
#   [预填充] Q/K/V 全量传入 → 一次性算完所有 token → 生成第一个 token
#
# 返回值的区别：
#   [解码] 返回 out + lse
#   [预填充] 返回 out + max_logits + lse（多了一个 max_logits 用于后续处理）
# ============================================================================
def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by exp(lse) / (exp(lse) + exp(attn_sink)).
            +-inf in attn_sink will be handled normally (i.e., -inf has no effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32. If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]], ignoring later k/v tokens (even if provided in indices).
            In extremely rare cases (topk_length provided, there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    return results


# ============================================================================
# _flash_attn_varlen_forward —— 标准多头注意力的前向（内部函数，B200 GPU 专用）
#
# 这和前面的函数有什么区别？
#   前面所有函数都是为了 DeepSeek 的 MLA（Multi-head Latent Attention）设计的。
#   这个函数做的是"标准多头注意力（MHA）"——就是 Transformer 论文里那个原始的
#   Q/K/V 有相同数量 head 的注意力。
#
#   这个 kernel 是基于 NVIDIA CUTLASS 库实现的，由 NVIDIA 贡献给 FlashMLA。
#   只在 B200 GPU（SM100 架构）上可用。
#
# varlen（变长序列）是什么意思？
#   假设 batch 里有 3 个句子，长度分别是 10、20、30 个 token。
#   普通做法：把 3 个句子都 padding（填充）到 30 个 token → 浪费算力
#   变长做法：把 3 个句子拼成一个长序列（总共 60 个 token），用
#   cu_seqlens 数组记录每个句子的起止位置 → 不浪费一个 token
#
# 工作的流程：
#   1. 根据 q/k/v 的形状推断维度信息
#   2. 分配输出 tensor（out）和 log-sum-exp tensor（lse）
#   3. 分配 workspace buffer（32MB 固定大小，CUDA kernel 的工作空间）
#   4. 调用 C++ 的 dense_prefill_fwd CUDA kernel
# ============================================================================
def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 从 tensor 形状中提取维度信息
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    # causal（因果遮罩）转成 C++ kernel 需要的编码格式：1 = 启用, 0 = 禁用
    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    # 如果没传 out/lse tensor，就自己分配
    if out is None:
        out = torch.empty(qo_total_len, num_qo_heads, head_dim_vo, device=q.device, dtype=q.dtype)
    if lse is None:
        # .T 确保 lse 在 seqlen 维度上是连续的（内存布局优化）
        lse = torch.empty(num_qo_heads, qo_total_len, device=q.device, dtype=torch.float32).T

    # 分配 32MB 的工作空间（CUDA kernel 需要的临时缓冲区）
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    # 调用 C++ 的密集前向 CUDA kernel
    flash_mla_cuda.dense_prefill_fwd(
        workspace_buffer,
        q,
        k,
        v,
        cu_seqlens_qo,
        cu_seqlens_kv,
        out,
        lse,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return out, lse


# ============================================================================
# _flash_attn_varlen_backward —— 标准多头注意力的反向传播（内部函数）
#
# 反向传播是干嘛的？
#   训练模型时，先做前向计算（算出预测结果），然后和正确答案对比得到 loss
#   （误差），再反向传播误差来调整模型参数。这个函数就是"反向传播"这一步。
#   推理时不需要这个函数。
#
# 为什么需要这么复杂的 workspace 分配？
#   反向计算需要更多的临时缓冲区（dQ_acc 累积 Q 的梯度、sum_OdO 做数值稳定等）。
#   这里精确计算需要多少字节，不多分配也不少分配——**精确分配 = 节省显存**。
#   显存是 GPU 上最宝贵的资源之一。
#
# 当前限制（TODO）：
#   还不支持 GQA（分组查询注意力），要求 Q 的 head 数 = K 的 head 数。
#   原因是反向传播的 CUTLASS 实现还没支持 GQA 的梯度计算。
# ============================================================================
def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qo_total_len, num_qo_heads, head_dim_qk = q.shape
    kv_total_len, num_kv_heads, head_dim_vo = v.shape

    # TODO: fix bwd GQA
    if num_qo_heads != num_kv_heads:
        raise ValueError(f"SM100 bwd doesn't support GQA now. num_qo_heads: {num_qo_heads}, num_kv_heads: {num_kv_heads}.")

    # causal mask 编码（1=启用, 0=禁用）
    mask_mode_code = 1 if causal else 0
    if softmax_scale is None:
        softmax_scale = head_dim_qk ** (-0.5)

    # 如果没传梯度 tensor，就自己分配
    # dq/dk/dv = loss 对 Q/K/V 的梯度
    if dq is None:
        dq = torch.empty(qo_total_len, num_qo_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dk is None:
        dk = torch.empty(kv_total_len, num_kv_heads, head_dim_qk, device=q.device, dtype=q.dtype)
    if dv is None:
        dv = torch.empty(kv_total_len, num_kv_heads, head_dim_vo, device=q.device, dtype=q.dtype)

    # 精确计算反向 kernel 需要多少 workspace 缓冲区
    # 8 字节对齐 max_seqlen_qo（up-align 到 8 的倍数）
    max_seqlen_qo_aligned = (max_seqlen_qo + 7) // 8 * 8
    # batch size = cu_seqlens 数组长度 - 1
    bs = cu_seqlens_qo.shape[0] - 1
    workspace_bytes = 0
    workspace_bytes += 4 * bs * max_seqlen_qo_aligned * num_qo_heads * head_dim_qk  # dQ_acc: 累积 Q 梯度的中间缓冲区
    workspace_bytes += 4 * max_seqlen_qo_aligned * bs * num_qo_heads * 2  # sum_OdO + scaled_lse: 数值稳定的临时变量
    if num_qo_heads != num_kv_heads:
        workspace_bytes += 2 * kv_total_len * num_qo_heads * (head_dim_qk + head_dim_vo)  # dKV_acc: GQA 模式下累积 K/V 梯度
    workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    # 调用 C++ 的密集反向 CUDA kernel
    flash_mla_cuda.dense_prefill_bwd(
        workspace_buffer,
        do,
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_qo,
        cu_seqlens_kv,
        dq,
        dk,
        dv,
        mask_mode_code,
        softmax_scale,
        max_seqlen_qo,
        max_seqlen_kv,
        is_varlen,
    )

    return dq, dk, dv


# ============================================================================
# FlashAttnVarlenFunc —— 把前向和反向打包成一个"自动求导操作"
#
# 为什么要打包成一个类？
#   PyTorch 训练模型时，会自动记录每一步操作（这叫"计算图"），然后反向传播
#   梯度。但我们的注意力计算是调用 C++ CUDA kernel，PyTorch 不知道里面干了什么。
#   通过 torch.autograd.Function 把前向和反向打包在一起，PyTorch 就能：
#     训练时 → 调用 forward() 做前向计算
#     自动求梯度时 → 调用 backward() 反向传播
#
# forward() 做了什么：
#   1. 调用 _flash_attn_varlen_forward（真正在 GPU 上算注意力）
#   2. 把 Q, K, V, out, lse 存起来（反向传播时需要这些算梯度）
#   3. 保存 max_seqlen、causal 等配置信息
#
# backward() 做了什么：
#   1. 从 ctx 取出 forward 时保存的 Q, K, V, out, lse
#   2. 调用 _flash_attn_varlen_backward 计算 dQ, dK, dV（梯度）
#   3. 对 cu_seqlens、max_seqlen 等"不需要梯度"的参数返回 None
# ============================================================================
class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        is_varlen: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, lse = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
            causal=causal, softmax_scale=softmax_scale,
            is_varlen=is_varlen,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv)
        ctx.max_seqlen_qo = max_seqlen_qo
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.is_varlen = is_varlen
        return out, lse

    def backward(
        ctx,
        do: torch.Tensor,
        dlse: torch.Tensor,
    ):
        del dlse  # LSE doesn't support backward currently
        q, k, v, out, lse, cu_seqlens_qo, cu_seqlens_kv = ctx.saved_tensors
        dq, dk, dv = _flash_attn_varlen_backward(
            do, q, k, v, out, lse,
            cu_seqlens_qo, cu_seqlens_kv, ctx.max_seqlen_qo, ctx.max_seqlen_kv,
            causal=ctx.causal, softmax_scale=ctx.softmax_scale,
            is_varlen=ctx.is_varlen,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


# ============================================================================
# flash_attn_varlen_func —— 标准多头注意力的公开 API
#
# 这就是给用户调用的接口。参数和 Dao-AILab 的 flash_attn 库兼容。
# 如果你原来用 flash_attn.flash_attn_varlen_func，现在可以直接换成
# flash_mla.flash_attn_varlen_func，不用改其他代码。
#
# 参数 dropout_p 和 deterministic 必须为默认值（0.0 和 False）。
# 保留它们只是为了接口兼容——如果用户传了 dropout=0.5，这里直接 assert 拦住。
# ============================================================================
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )


# ============================================================================
# flash_attn_varlen_qkvpacked_func —— 标准注意力的 API（QKV 打包版）
#
# 什么是 QKV 打包？
#   有些 Transformer 实现会把 Q、K、V 拼成一个大矩阵，一次矩阵乘算出全部。
#   输出形状是 [total_seq_len, num_heads, 3*head_dim]。
#
#   这个函数负责"拆包"：
#     qkv[:, :, :head_dim_qk]           → Q
#     qkv[:, :, head_dim_qk:2*head_dim] → K
#     qkv[:, :, 2*head_dim:]             → V
#
#   因为是 self-attention（Q 和 K/V 来自相同序列），所以 cu_seqlens 和
#   max_seqlen 都传两次（Q 和 KV 各一份）。
# ============================================================================
def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        qkv[:, :, :head_dim_qk], qkv[:, :, head_dim_qk:head_dim_qk * 2], qkv[:, :, head_dim_qk * 2:],
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        causal, softmax_scale, is_varlen,
    )


# ============================================================================
# flash_attn_varlen_kvpacked_func —— 标准注意力的 API（KV 打包版）
#
# 什么是 KV 打包？
#   有些情况下 K 和 V 会被拼在一起（叫 KV-packed 格式），但 Q 是独立的。
#   常见于 encoder-decoder 模型的 cross-attention 层。
#
#   这个函数负责"拆包"：
#     kv[:, :, :head_dim_qk]   → K
#     kv[:, :, head_dim_qk:]   → V
#   然后调用标准接口。
# ============================================================================
def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dropout_p == 0.0
    assert not deterministic
    return FlashAttnVarlenFunc.apply(
        q, kv[:, :, :head_dim_qk], kv[:, :, head_dim_qk:],
        cu_seqlens_qo, cu_seqlens_kv, max_seqlen_qo, max_seqlen_kv,
        causal, softmax_scale, is_varlen,
    )
