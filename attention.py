from typing import NamedTuple

import flash_attn
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

FA_BLOCK_SIZE = 256


class BatchInfo(NamedTuple):
    pos_ids: Tensor  # [total_seqlen]

    # for packing
    cu_seqlens: Tensor | None = None  # [num_seqs + 1]
    max_seqlen: int = 0

    # for generation
    kv_cache: Tensor | None = None  # [num_layers, 2, num_blocks, block_size, num_heads, head_dim]
    kv_seqlens: Tensor | None = None  # [num_seqs]
    block_table: Tensor | None = None  # [num_seqs, max_num_blocks_per_seq]

    @staticmethod
    def init_packing(seqs: list[list[int]], device: torch.types.Device = None) -> tuple[Tensor, "BatchInfo"]:
        input_ids = []
        pos_ids = []
        cu_seqlens = [0]
        max_seqlen = 0

        for seq in seqs:
            input_ids.extend(seq)
            seqlen = len(seq)
            pos_ids.extend(range(seqlen))
            cu_seqlens.append(cu_seqlens[-1] + seqlen)
            max_seqlen = max(max_seqlen, seqlen)

        input_ids = torch.tensor(input_ids, device=device)
        info = BatchInfo(
            pos_ids=torch.tensor(pos_ids, device=device),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),  # FA expects int32
            max_seqlen=max_seqlen,
        )
        return input_ids, info

    @staticmethod
    def init_prefill(seqs: list[list[int]], kv_cache: Tensor, device: torch.types.Device = None):
        input_ids, info = BatchInfo.init_packing(seqs, device=device)

        seqlens = [len(seq) for seq in seqs]
        num_blocks = [triton.cdiv(seqlen, FA_BLOCK_SIZE) for seqlen in seqlens]
        max_num_blocks = max(num_blocks)

        # TODO: think about how to allocate blocks
        block_table = []
        block_idx = 0
        for num_block in num_blocks:
            block_ids = list(range(block_idx, block_idx + num_block)) + [0] * (max_num_blocks - num_block)
            block_table.append(block_ids)
            block_idx += num_block

        info = BatchInfo(
            *info,
            kv_cache=kv_cache,
            kv_seqlens=torch.tensor(seqlens, device=device),
            block_table=torch.tensor(block_table, device=device),
        )
        return input_ids, info


def attention(q: Tensor, k: Tensor, v: Tensor, dropout: float, info: BatchInfo | None = None, layer_idx: int = 0):
    # normal case
    if info is None:
        ndim3 = q.ndim == 3  # F.sdpa() does not dispatch FA/CuDNN if ndim==3
        if ndim3:
            q, k, v = [_x.unsqueeze(0) for _x in [q, k, v]]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=dropout,
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)
        if ndim3:
            out = out.squeeze(0)
        return out

    # when BatchInfo is used, q/k/v are sequence-packed
    # i.e. [total_tokens, num_heads, head_dim]

    # fill KV cache for generation
    if info.kv_cache is not None:
        _fa_insert_kv_cache(
            k,
            v,
            info.kv_cache[layer_idx, 0],
            info.kv_cache[layer_idx, 1],
            info.cu_seqlens,
            info.block_table,
        )

    # sequence packing or generation prefill
    if info.cu_seqlens is not None:
        out = flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=info.cu_seqlens,
            cu_seqlens_k=info.cu_seqlens,
            max_seqlen_q=info.max_seqlen,
            max_seqlen_k=info.max_seqlen,
            dropout_p=dropout,
            causal=True,
        )

    # generation decode
    # each token is 1 sequence
    else:
        out = flash_attn.flash_attn_with_kvcache(
            q.unsqueeze(1),
            info.kv_cache[layer_idx, 0],
            info.kv_cache[layer_idx, 1],
            cache_seqlens=info.kv_seqlens,
            block_table=info.block_table,
        )

    return out


def _fa_insert_kv_cache(
    k: Tensor,  # [total_seqlen, num_heads, head_dim]
    v: Tensor,  # [total_seqlen, num_heads, head_dim]
    k_cache: Tensor,  # [num_blocks, block_size, num_heads, head_dim]
    v_cache: Tensor,  # [num_blocks, block_size, num_heads, head_dim]
    cu_seqlens: Tensor,  # [num_seqs+1]
    block_table: Tensor,  # [num_seqs, max_num_blocks_per_seq]
) -> None:
    num_seqs = block_table.shape[0]
    num_heads, head_dim = k.shape[1:]
    DIM = num_heads * head_dim
    BLOCK_SIZE = k_cache.shape[1]
    assert all(x.is_contiguous() for x in [k, v, k_cache, v_cache, cu_seqlens, block_table])

    _fa_insert_kv_cache_impl[(num_seqs,)](
        k,
        v,
        k_cache,
        v_cache,
        cu_seqlens,
        block_table,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        block_table.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        DIM=DIM,
    )


@triton.jit
def _fa_insert_kv_cache_impl(
    k_ptr,  # [total_seqlen, num_heads * head_dim]
    v_ptr,  # [total_seqlen, num_heads * head_dim]
    k_cache_ptr,  # [num_blocks, block_size, num_heads * head_dim]
    v_cache_ptr,  # [num_blocks, block_size, num_heads * head_dim]
    cu_seqlens_ptr,  # [num_seqs+1]
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    k_stride,
    v_stride,
    k_cache_stride,
    v_cache_stride,
    block_table_stride,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,  # num_heads * head_dim
):
    # 1 program for 1 sequence
    seq_id = tl.program_id(0)  # [0, num_seqs-1]
    cu_seqlens_ptr += seq_id
    block_table_ptr += seq_id * block_table_stride

    offset = tl.load(cu_seqlens_ptr)
    seqlen = tl.load(cu_seqlens_ptr + 1) - offset

    # select KV from sequence-packed KV
    k_ptrs = k_ptr + (offset + tl.arange(0, BLOCK_SIZE)[:, None]) * k_stride + tl.arange(0, DIM)
    v_ptrs = v_ptr + (offset + tl.arange(0, BLOCK_SIZE)[:, None]) * v_stride + tl.arange(0, DIM)

    k_cache_ptrs = k_cache_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * DIM + tl.arange(0, DIM)
    v_cache_ptrs = v_cache_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * DIM + tl.arange(0, DIM)

    num_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    for logical_block_id in range(0, num_blocks):
        mask = (logical_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]) < seqlen
        k_block = tl.load(k_ptrs, mask)  # [BLOCK_SIZE, DIM]
        v_block = tl.load(v_ptrs, mask)

        physical_block_id = tl.load(block_table_ptr + logical_block_id)
        tl.store(k_cache_ptrs + physical_block_id * k_cache_stride, k_block, mask)
        tl.store(v_cache_ptrs + physical_block_id * v_cache_stride, v_block, mask)

        k_ptrs += BLOCK_SIZE * k_stride
        v_ptrs += BLOCK_SIZE * v_stride
