# https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/qwen3/modeling_qwen3.py

from typing import NamedTuple

import flash_attn
import safetensors.torch
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from torch import Tensor, nn
from transformers import Qwen3Config


class BatchInfo(NamedTuple):
    pos_ids: Tensor

    # for packing
    cu_seqlens: Tensor | None = None
    max_seqlen: int = 0

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
        pos_ids = torch.tensor(pos_ids, device=device)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)  # FA expects int32
        info = BatchInfo(pos_ids=pos_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        return input_ids, info


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# NOTE: if x and pos_embeds are BF16, the computation is done in BF16
def apply_rope(x: Tensor, pos_embeds: Tensor) -> Tensor:
    # x: [*, L, num_heads, dim]
    # pos_embeds: [*, L, dim]
    # pos_embeds may have fewer leading dimensions than x's
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = pos_embeds.unsqueeze(-2).chunk(2, dim=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2], dim=-1)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.attention_dropout = cfg.attention_dropout
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, pos_embeds: Tensor, info: BatchInfo | None = None) -> Tensor:
        hidden_shape = (*x.shape[:-1], -1, self.head_dim)
        q = apply_rope(self.q_norm(self.q_proj(x).view(hidden_shape)), pos_embeds)
        k = apply_rope(self.k_norm(self.k_proj(x).view(hidden_shape)), pos_embeds)
        v = self.v_proj(x).view(hidden_shape)

        dropout = self.attention_dropout if self.training else 0.0

        if info is None:
            if x.ndim == 2:  # F.sdpa() does not dispatch FA/CuDNN if ndim==3
                q, k, v = [_x.unsqueeze(0) for _x in [q, k, v]]
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout,
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
            if x.ndim == 2:
                out = out.squeeze(0)

        else:
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

        out = self.o_proj(out.flatten(-2))
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg, layer_idx)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: Tensor, pos_embeds: Tensor, info: BatchInfo | None = None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos_embeds, info)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def compute_rope(pos_ids: Tensor, rope_theta: float, dim: int) -> Tensor:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=pos_ids.device) / dim))
    freqs = pos_ids.unsqueeze(-1) * inv_freq
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [*pos_ids.shape, dim]


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg, layer_idx) for layer_idx in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, input_ids: Tensor, info: BatchInfo | None = None) -> Tensor:
        pos_ids = info.pos_ids if info is not None else torch.arange(input_ids.shape[-1], device=input_ids.device)
        pos_embeds = compute_rope(pos_ids, self.cfg.rope_theta, self.cfg.head_dim)
        pos_embeds = pos_embeds.to(self.embed_tokens.weight.dtype)

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_embeds, info)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, info: BatchInfo | None = None) -> Tensor:
        hidden_states = self.model(input_ids, info)
        logits = self.lm_head(hidden_states)
        return logits

    @staticmethod
    def from_pretrained(model_id: str) -> "Qwen3ForCausalLM":
        cfg = Qwen3Config.from_pretrained(model_id)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(cfg)

        state_dict = load_hf_state_dict(model_id)
        model.load_state_dict(state_dict, assign=True)
        return model


def load_hf_state_dict(model_id: str) -> Qwen3ForCausalLM:
    state_dict = dict()

    for filename in list_repo_files(model_id):
        if not filename.endswith(".safetensors"):
            continue

        local_path = hf_hub_download(model_id, filename)
        state_dict.update(safetensors.torch.load_file(local_path))

    return state_dict
