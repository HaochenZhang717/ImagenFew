"""
DiT-style generator for Drifting Models.
Adapted for 32x32 images (MNIST, CIFAR-10) with adaLN-Zero conditioning.

Key differences from standard DiT:
- No timestep input (one-step generator, not diffusion)
- Conditioning = class_embed + alpha_embed + style_embed
- Uses register tokens, RoPE, SwiGLU, RMSNorm, QK-Norm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from abc import ABC, abstractmethod
import torch
import numpy as np
import os
import sys
import torch
import functools
import pandas as pd



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k."""
    qk_embed = (qk * cos) + (rotate_half(qk) * sin)
    return qk_embed

class SwiGLU(nn.Module):
    """SwiGLU activation function with gated linear unit."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with QK-Norm and optional RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # QK-Norm for training stability
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class CrossAttention(nn.Module):
    """Cross-attention: x attends to condition tokens c"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # q from x, k/v from c
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,   # (B, N, C)  main tokens
        c: torch.Tensor,   # (B, Nc, C) condition tokens
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, N, C = x.shape
        Nc = c.shape[1]

        # projections
        q = self.q_proj(x)
        k = self.k_proj(c)
        v = self.v_proj(c)

        # reshape
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)

        # QK norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, Nc)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)   # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)

        return out


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlockCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        # norms
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.cond_norm = RMSNorm(dim)

        # attention
        self.self_attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        # adaLN-Zero
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 9 * dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        condition_tokens: int = 4
    ) -> torch.Tensor:

        # split tokens
        cond = condition_tokens     # (B, Nc, C)
        main = x   # (B, N-Nc, C)

        # modulation
        modulation = self.adaLN_modulation(c).chunk(9, dim=1)
        (shift_msa, scale_msa, gate_msa,
         shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = modulation

        # === Self-Attention (only main tokens) ===
        main = main + gate_msa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(main), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
        )

        # === Cross-Attention (main attends condition) ===
        main = main + gate_ca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(main), shift_ca, scale_ca),
            self.cond_norm(cond),
            rope_cos,
            rope_sin
        )

        # === MLP ===
        main = main + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm3(main), shift_mlp, scale_mlp)
        )

        return main

class DiTBlockAdaLN(nn.Module):
    """
    DiT Block with adaLN-Zero conditioning.

    Regresses 6 modulation parameters from conditioning:
    - shift_msa, scale_msa, gate_msa (for attention)
    - shift_mlp, scale_mlp, gate_mlp (for MLP)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        # Pre-norm with RMSNorm (no learned affine - will be modulated)
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        # Use SwiGLU as per paper
        self.mlp = SwiGLU(dim, mlp_hidden, dim)


        # adaLN-Zero modulation: 6 parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )



    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        num_condition_tokens: int = 4
    ) -> torch.Tensor:
        # Get modulation parameters
        modulation = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        # Self-attention with adaLN + gating
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
            num_condition_tokens
        )

        # MLP with adaLN + gating
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation and 1D patch projection."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * out_channels)

        # adaLN modulation: 2 parameters (shift, scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """Convert 1D sequence patches to embeddings using Conv1d."""

    def __init__(
        self,
        seq_len: int = 128,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.padded_seq_len = math.ceil(seq_len / patch_size) * patch_size
        self.num_patches = self.padded_seq_len // patch_size

        self.proj = nn.Conv1d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, N, D)
        if x.shape[-1] < self.padded_seq_len:
            x = F.pad(x, (0, self.padded_seq_len - x.shape[-1]))
        x = self.proj(x)  # (B, D, N)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class LabelEmbedder(nn.Module):
    """Embed class labels with null class for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        # +1 for null/unconditional class
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Randomly drop labels to null class for CFG training."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dropout_prob > 0 and train:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class AlphaEmbedder(nn.Module):
    """Embed CFG alpha scale using Fourier features."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def fourier_features(alpha: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        """Create sinusoidal embeddings for alpha."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=alpha.device) / half
        )
        args = alpha[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        fourier = self.fourier_features(alpha, self.frequency_embedding_size)
        return self.mlp(fourier)


class StyleEmbedder(nn.Module):
    """
    Style embeddings from paper Sec A.2.
    32 random style tokens index into a codebook of 64 learnable embeddings.
    """

    def __init__(self, hidden_size: int, num_tokens: int = 32, codebook_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random style embeddings."""
        # Random indices for each sample in the batch
        indices = torch.randint(
            0, self.codebook_size, (batch_size, self.num_tokens), device=device
        )
        embeddings = self.codebook(indices)  # (B, num_tokens, D)
        # Sum over tokens
        style = embeddings.sum(dim=1)  # (B, D)
        return style


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class DriftDiT(nn.Module):
    """
    DiT-style denoiser for 1D latent sequences.

    Input: (B, C, L), where C is latent_dim and L is latent_seq_len.
    Condition: token embeddings (B, N_cond, condition_dim), or None.
    Output: predicted noise with the same shape as input.
    """

    def __init__(
        self,
        num_steps: int=50,
        seq_len: int = 128,
        patch_size: int = 4,
        in_channels: int = 3,
        condition_dim: int = 128,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=num_steps,
            embedding_dim=hidden_size
        )
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.padded_seq_len = math.ceil(seq_len / patch_size) * patch_size
        self.num_patches = self.padded_seq_len // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbed(
            seq_len=seq_len,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # RoPE for positional encoding
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + 64,
        )

        # Conditioning embeddings
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim, hidden_size),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlockCrossAttention(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qk_norm=True,
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with specific strategy for adaLN-Zero."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        # Zero-init adaLN modulation layers (critical for training stability)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-init final layer adaLN modulation only
        # NOTE: For drifting models (one-step generator), we keep the final linear
        # layer with small random weights so the model outputs non-zero images initially
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        # Use small initialization for final linear (not zero!)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Convert patch tokens back to sequence.
        x: (B, N, patch_size * C)
        Returns: (B, C, L)
        """
        c = self.out_channels
        p = self.patch_size
        n = x.shape[1]

        x = x.reshape(-1, n, p, c)
        x = torch.einsum("bnpc->bcnp", x)
        x = x.reshape(-1, c, n * p)
        return x[..., :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        condition_tokens: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        x_seq = x
        B = x_seq.shape[0]
        seq_len = x_seq.shape[-1]

        # Patch embedding
        x_seq = self.patch_embed(x_seq)  # (B, N, D)

        if condition_tokens is None:
            condition_tokens = x_seq.new_zeros(B, 1, self.hidden_size)
        else:
            condition_tokens = self.condition_projection(condition_tokens)

        # Get RoPE embeddings
        num_tokens = x_seq.shape[1]
        rope_cos, rope_sin = self.rope(x_seq, num_tokens)

        # Time Conditioning
        c = self.diffusion_embedding(diffusion_step) + condition_tokens.mean(dim=1)

        # Transformer blocks
        for block in self.blocks:
            x_seq = block(x_seq, c, rope_cos, rope_sin, condition_tokens)

        # Final layer and unpatchify
        x_seq = self.final_layer(x_seq, c)
        x_seq = self.unpatchify(x_seq, seq_len)
        return x_seq, {}


def DiT_Tiny(configs, in_channels):
    """DriftDiT-Tiny: depth=6, hidden_dim=256, heads=4 -> ~5M params"""
    return DriftDiT(
        num_steps=configs["num_steps"],
        seq_len=configs['seq_len'],
        patch_size=configs['patch_size'],
        in_channels=in_channels,
        condition_dim=configs.get("condition_dim", 128),
        hidden_size=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        device=configs.get("device", None),
    )

def DiT_Small(configs, in_channels):
    """DriftDiT-Small: depth=8, hidden_dim=384, heads=6 -> ~15M params"""
    return DriftDiT(
        num_steps=configs["num_steps"],
        seq_len=configs['seq_len'],
        patch_size=configs['patch_size'],
        in_channels=in_channels,
        condition_dim=configs.get("condition_dim", 128),
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        device=configs.get("device", None),
    )

# Model registry
DriftDiT_models = {
    "DriftDiT-Tiny": DiT_Tiny,
    "DriftDiT-Small": DiT_Small,
}
