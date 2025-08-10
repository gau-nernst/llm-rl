# https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/qwen3/modeling_qwen3.py

import safetensors.torch
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files
from torch import Tensor, nn
from transformers import Qwen3Config


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

    def forward(self, x: Tensor, pos_embeds: Tensor) -> Tensor:
        hidden_shape = (*x.shape[:-1], -1, self.head_dim)
        q = apply_rope(self.q_norm(self.q_proj(x).view(hidden_shape)), pos_embeds).transpose(1, 2)
        k = apply_rope(self.k_norm(self.k_proj(x).view(hidden_shape)), pos_embeds).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        dropout = self.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True, enable_gqa=True)

        out = out.transpose(1, 2).flatten(-2)
        out = self.o_proj(out)
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg, layer_idx)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: Tensor, pos_embeds: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos_embeds)
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

    def forward(self, input_ids: Tensor) -> Tensor:
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        pos_embeds = compute_rope(pos_ids, self.cfg.rope_theta, self.cfg.head_dim)
        pos_embeds = pos_embeds.to(self.embed_tokens.weight.dtype)

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_embeds)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.model(input_ids)
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
