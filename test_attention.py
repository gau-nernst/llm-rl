import random

import torch
from torch import Tensor

from attention import _fa_insert_kv_cache


def cdiv(a: int, b: int):
    return (a + b - 1) // b


def test_fa_insert_kv_cache():
    def ref_impl(
        k: Tensor,  # [total_seqlen, num_heads, head_dim]
        v: Tensor,  # [total_seqlen, num_heads, head_dim]
        k_cache: Tensor,  # [num_blocks, block_size, num_heads, head_dim]
        v_cache: Tensor,  # [num_blocks, block_size, num_heads, head_dim]
        cu_seqlens: Tensor,  # [num_seqs+1]
        block_table: Tensor,  # [num_seqs, max_num_blocks_per_seq]
    ):
        num_seqs = block_table.shape[0]
        block_size = k_cache.shape[1]

        for seq_id in range(num_seqs):
            offset = cu_seqlens[seq_id].item()
            seqlen = cu_seqlens[seq_id + 1].item() - offset
            num_blocks = cdiv(seqlen, block_size)

            for logical_block_id in range(num_blocks):
                actual_block_size = min(block_size, seqlen - logical_block_id * block_size)
                k_block = k[offset : offset + actual_block_size]
                v_block = v[offset : offset + actual_block_size]

                # NOTE: writing like this is bad, since k_cache[...] can be a copy
                physical_block_id = block_table[seq_id, logical_block_id].item()
                k_cache[physical_block_id, :actual_block_size].copy_(k_block)
                v_cache[physical_block_id, :actual_block_size].copy_(v_block)

                offset += block_size

    # generate input data
    num_seqs = 5
    num_heads = 4
    head_dim = 16
    max_seqlen = 200
    block_size = 16
    max_blocks = cdiv(max_seqlen, block_size)
    num_blocks = max_blocks * num_seqs
    device = "cuda"
    dtype = torch.bfloat16

    k_list = []
    v_list = []
    cu_seqlens = [0]
    block_table = []
    available_block_ids = list(range(num_blocks))
    random.shuffle(available_block_ids)
    num_used_blocks = 0

    for _ in range(num_seqs):
        seqlen = torch.randint(10, max_seqlen, size=(1,)).item()
        k_list.append(torch.randn(seqlen, num_heads, head_dim, dtype=dtype, device=device))
        v_list.append(torch.randn(seqlen, num_heads, head_dim, dtype=dtype, device=device))
        cu_seqlens.append(cu_seqlens[-1] + seqlen)

        this_num_blocks = cdiv(seqlen, block_size)
        this_block_ids = available_block_ids[num_used_blocks : num_used_blocks + this_num_blocks]
        block_table.append(this_block_ids + [0] * (max_blocks - this_num_blocks))
        num_used_blocks += this_num_blocks

    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    cu_seqlens = torch.tensor(cu_seqlens, device=device)
    block_table = torch.tensor(block_table, device=device)

    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)

    k_cache_cloned = k_cache.clone()
    v_cache_cloned = v_cache.clone()

    print(cu_seqlens)
    print(block_table)

    _fa_insert_kv_cache(k, v, k_cache, v_cache, cu_seqlens, block_table)
    ref_impl(k, v, k_cache_cloned, v_cache_cloned, cu_seqlens, block_table)
    assert (k_cache == k_cache_cloned).all()  # should be bit-identical
    assert (v_cache == v_cache_cloned).all()  # should be bit-identical
