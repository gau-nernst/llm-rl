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


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# NOTE: if x and pos_embeds are BF16, the computation is done in BF16
def apply_rotary_pos_emb(x: Tensor, pos_embeds: tuple[Tensor, Tensor], unsqueeze_dim: int = 1):
    cos = pos_embeds[0].unsqueeze(unsqueeze_dim)
    sin = pos_embeds[1].unsqueeze(unsqueeze_dim)
    out = (x * cos) + (rotate_half(x) * sin)
    return out.to(x.dtype)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.attention_dropout = cfg.attention_dropout
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.num_attention_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, pos_embeds: tuple[Tensor, Tensor]) -> Tensor:
        hidden_shape = (*x.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        q = apply_rotary_pos_emb(q, pos_embeds)
        k = apply_rotary_pos_emb(k, pos_embeds)
        dropout = self.attention_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True, enable_gqa=True)

        out = out.transpose(1, 2).flatten(-2)
        out = self.o_proj(out)
        return out


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: Tensor, pos_embeds: tuple[Tensor, Tensor]) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos_embeds)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def compute_inv_freq(rope_theta: float, dim: int) -> Tensor:
    return 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        inv_freq = compute_inv_freq(cfg.rope_theta, cfg.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def init_buffers(self) -> None:
        if self.inv_freq.is_meta:
            self.to_empty(device=torch.get_default_device())
        self.inv_freq.data = compute_inv_freq(self.cfg.rope_theta, self.cfg.head_dim)

    @torch.no_grad()
    def rotary_emb(self, pos_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        freqs = pos_ids[:, :, None].float() * self.inv_freq[None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pos_embeds = self.rotary_emb(pos_ids, hidden_states.dtype)

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
        model.model.init_buffers()

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
