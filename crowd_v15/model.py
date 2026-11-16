"""A small GPT whose hidden linears can be perturbed in the forward pass.

Pre-norm blocks, fused-QKV attention, SwiGLU MLP, tied embeddings. The only
unusual part is that every hidden projection goes through :func:`_apply_lin`,
which can add ``sign * sigma * noise`` to the layer output and can record the
layer input for the DUST outer product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 4096
    n_layers: int = 20
    d_model: int = 2048
    n_heads: int = 16
    context_length: int = 2048
    mlp_ratio: float = 4.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads

    @property
    def hidden(self) -> int:
        return int(self.mlp_ratio * self.d_model)

    @property
    def n_params(self) -> int:
        """Analytic parameter count, for printing before the model is built."""
        per_layer = 4 * self.d_model ** 2 + 3 * self.d_model * self.hidden
        emb = self.vocab_size * self.d_model
        return self.n_layers * per_layer + emb * (1 if self.tie_embeddings else 2)

    @classmethod
    def preset(cls, name: str) -> "ModelConfig":
        if name == "tiny":      # CPU smoke tests
            return cls(vocab_size=256, n_layers=4, d_model=128, n_heads=4, context_length=64)
        if name == "10m":
            return cls(vocab_size=4096, n_layers=6, d_model=256, n_heads=8, context_length=512)
        if name == "100m":
            return cls(vocab_size=4096, n_layers=12, d_model=768, n_heads=12, context_length=1024)
        if name == "1b":        # the maintainer's 1B sizing
            return cls(vocab_size=4096, n_layers=20, d_model=2048, n_heads=16)
        if name == "2b":
            return cls(vocab_size=4096, n_layers=32, d_model=2304, n_heads=18)
        raise ValueError(f"unknown preset {name!r}")


def canonical_params(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """``name -> tensor`` with tied weights listed once, under their first name.

    With ``tie_embeddings`` the embedding and the head are the same storage, so a
    plain ``named_parameters()`` dict silently drops one of the two names and a
    wire format built from it would disagree with the one built from the other.
    First registration order wins (``tok_emb.weight`` before ``head.weight``),
    which is also what :func:`crowd_v15.dust.dust_gradients` uses as its
    canonical key, so the client's upload layout matches the served weights.
    """
    seen: set[int] = set()
    canonical: dict[str, torch.Tensor] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            continue
        seen.add(id(param))
        canonical[name] = param
    return canonical


@dataclass
class Perturb:
    """Per-forward perturbation state.

    ``noise[name]`` must already have the shape of that layer's output so the
    same tensors can be reused by the ``+a`` and ``-a`` passes. ``record``, when
    a dict, collects the *clean* layer inputs for the outer product.
    """

    enabled: bool = False
    sigma: float = 0.0
    sign: int = 0
    noise: dict[str, torch.Tensor] = field(default_factory=dict)
    record: dict[str, torch.Tensor] | None = None
    only: str | None = None


def _apply_lin(x: torch.Tensor, weight: torch.Tensor, name: str,
               st: Perturb | None) -> torch.Tensor:
    if st is not None and st.record is not None:
        st.record[name] = x
    y = x @ weight.t()
    if st is None or not st.enabled:
        return y
    if st.only is not None and st.only != name:
        return y
    noise = st.noise.get(name)
    if noise is None:
        return y
    return y + st.sign * st.sigma * noise


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, st: Perturb | None,
                prefix: str) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = _apply_lin(x, self.qkv.weight, f"{prefix}.qkv", st)
        q, k, v = qkv.split(self.d, dim=-1)
        shape = (b, t, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask, float("-inf"))
        y = torch.softmax(att, dim=-1) @ v
        y = y.transpose(1, 2).reshape(b, t, self.d)
        return _apply_lin(y, self.o.weight, f"{prefix}.o", st)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.hidden, bias=False)
        self.gate = nn.Linear(cfg.d_model, cfg.hidden, bias=False)
        self.down = nn.Linear(cfg.hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, st: Perturb | None, prefix: str) -> torch.Tensor:
        up = _apply_lin(x, self.up.weight, f"{prefix}.up", st)
        gate = _apply_lin(x, self.gate.weight, f"{prefix}.gate", st)
        return _apply_lin(F.silu(gate) * up, self.down.weight, f"{prefix}.down", st)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, st: Perturb | None,
                prefix: str) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, st, f"{prefix}.attn")
        return x + self.mlp(self.norm2(x), st, f"{prefix}.mlp")


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def noise_shapes(self) -> dict[str, int]:
        """Perturbed layer name -> output width. Order is deterministic."""
        # The embedding is a layer too: its input is one-hot, so its outer
        # product is a scatter-add into the touched rows. With tied embeddings
        # that term is required for the shared tensor's gradient to be complete.
        shapes: dict[str, int] = {"tok_emb": self.cfg.d_model}
        for i in range(self.cfg.n_layers):
            p = f"blocks.{i}"
            shapes[f"{p}.attn.qkv"] = 3 * self.cfg.d_model
            shapes[f"{p}.attn.o"] = self.cfg.d_model
            shapes[f"{p}.mlp.up"] = self.cfg.hidden
            shapes[f"{p}.mlp.gate"] = self.cfg.hidden
            shapes[f"{p}.mlp.down"] = self.cfg.d_model
        return shapes

    def forward(self, idx: torch.Tensor, st: Perturb | None = None) -> torch.Tensor:
        """Return logits ``(B, T, vocab)``. ``st`` records inputs when it has a dict."""
        b, t = idx.shape
        if t > self.cfg.context_length:
            raise ValueError(f"sequence {t} exceeds context {self.cfg.context_length}")
        mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1
        )[None, None]
        x = self.tok_emb(idx)
        if st is not None and st.record is not None:
            st.record["tok_emb"] = idx
        if st is not None and st.enabled and (st.only is None or st.only == "tok_emb"):
            x = x + st.sign * st.sigma * st.noise["tok_emb"]
        for i, block in enumerate(self.blocks):
            x = block(x, mask, st, f"blocks.{i}")
        h = self.norm_f(x)
        if st is not None and st.record is not None:
            st.record["__head__"] = h
        return h @ self.head.weight.t()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
