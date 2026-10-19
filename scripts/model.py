"""Llama-style decoder-only transformer in plain PyTorch.

Module and parameter names mirror Hugging Face's LlamaForCausalLM (embed_tokens, layers.N.self_attn.q_proj, ...),
and RoPE uses the same rotate-half layout, so the export is a key rename and the stock GGUF converter applies.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12  # < n_heads gives grouped-query attention
    ffn_dim: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self):
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # normalize in fp32 like the reference implementation, then scale in the input dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


def rope_tables(cfg):
    inv_freq = 1.0 / cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
    freqs = torch.outer(torch.arange(cfg.max_seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cos[:t].to(q.dtype), sin[:t].to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_kv_heads != self.n_heads)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0 and cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        # GPT-2 style: N(0, 0.02), with the residual output projections scaled down by depth
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * self.cfg.n_layers) if name.endswith(("o_proj.weight", "down_proj.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        # the output head is tied to the input embedding
        logits = F.linear(self.norm(x), self.embed_tokens.weight)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, eos_id=2):
        """Plain sampling without a KV cache; good enough for progress samples during training."""
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.max_seq_len:])[:, -1].float()
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    logits[logits < logits.topk(top_k, dim=-1).values[:, -1:]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return idx


def flops_per_token(cfg, seq_len):
    """Training FLOPs per token as in the PaLM paper: 6 per weight plus 12 * layers * dim * seq_len for attention."""
    attn = 2 * cfg.dim * cfg.n_heads * cfg.head_dim + 2 * cfg.dim * cfg.n_kv_heads * cfg.head_dim
    weights = cfg.n_layers * (attn + 3 * cfg.dim * cfg.ffn_dim) + cfg.vocab_size * cfg.dim
    return 6 * weights + 12 * cfg.n_layers * cfg.dim * seq_len
