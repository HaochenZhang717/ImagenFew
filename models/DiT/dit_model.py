"""
DiT-style backbone with adaLN-Zero conditioning.

Used here as the score-model backbone inside the existing EDM diffusion
framework. Inputs are noisy images, a noise embedding scalar, and an optional
continuous conditioning vector.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
import re


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
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross attention from data tokens (queries) to context tokens (keys/values)."""

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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, dim = x.shape
        _, kv_len, _ = context.shape

        q = self.q(x).reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if context_mask is not None:
            if context_mask.ndim != 2:
                raise ValueError(f"Expected context_mask with shape (B, N_ctx), got {tuple(context_mask.shape)}")
            context_mask = context_mask.to(torch.bool)
            # Guarantee at least one valid token per sample to avoid NaNs.
            has_any = context_mask.any(dim=-1)
            if not bool(has_any.all()):
                context_mask = context_mask.clone()
                context_mask[~has_any, 0] = True
            mask = context_mask[:, None, None, :]
            attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, q_len, dim)
        return self.proj(out)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
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
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        # Use SwiGLU as per paper
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        # adaLN-Zero modulation: 6 parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 7 * dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get modulation parameters
        modulation = self.adaLN_modulation(c).chunk(7, dim=1)
        shift_msa, scale_msa, gate_msa, gate_xattn, shift_mlp, scale_mlp, gate_mlp = modulation

        # Self-attention with adaLN + gating
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
        )

        # Cross-attention to context tokens (if provided).
        if context is not None:
            x = x + gate_xattn.unsqueeze(1) * self.cross_attn(
                self.norm_cross(x), context, context_mask=context_mask
            )

        # MLP with adaLN + gating
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation and linear projection."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

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
    """Convert image patches to embeddings using Conv2d."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class AlphaEmbedder(nn.Module):
    """Embed scalar noise labels using Fourier features."""

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


class FrozenTextEncoder(nn.Module):
    """Frozen HF encoder that returns token-level hidden states."""

    def __init__(self, model_name: str, max_length: int = 128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = int(max_length)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Union[str, List[str]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device=device, dtype=torch.long)
        attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.long)
        model_output = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


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

def split_segments(text: str):
    # 按 [Segment k]: 分割
    segments = re.split(r"\[Segment \d+\]:", text)
    segments = [seg.strip() for seg in segments if seg.strip()]
    return segments


class DriftDiT(nn.Module):
    """
    DiT-style backbone for EDM diffusion with optional token-level conditioning.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        condition_dim: int = 256,
        label_dropout: float = 0.1,
        num_register_tokens: int = 8,
        use_style_embed: bool = True,
        use_text_encoder: bool = True,
        text_encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_max_length: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # Register (in-context) tokens
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_register_tokens, hidden_size) * 0.02
        )

        # RoPE for positional encoding
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + num_register_tokens + 64,
        )

        # Conditioning: scalar noise embedding (always) + optional context tokens via cross attention.
        self.condition_dropout = label_dropout
        self.use_text_encoder = use_text_encoder
        if self.use_text_encoder:
            self.text_encoder = FrozenTextEncoder(text_encoder_model_name, max_length=text_max_length)
            condition_input_dim = self.text_encoder.hidden_size
        else:
            self.text_encoder = None
            condition_input_dim = condition_dim
        self.context_proj = nn.Sequential(
            nn.LayerNorm(condition_input_dim),
            nn.Linear(condition_input_dim, hidden_size),
        )
        self.alpha_embed = AlphaEmbedder(hidden_size)
        self.use_style_embed = use_style_embed
        if use_style_embed:
            self.style_embed = StyleEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
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

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to image.
        x: (B, N, patch_size^2 * C)
        Returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        text_condition:  List[str],
        alpha: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy input image, shape (B, C, H, W)
            text_condition: list of string
            alpha: Noise embedding scalar, shape (B,)
            force_drop_ids: Force condition dropout for specific samples

        Returns:
            Generated images, shape (B, C, H, W)
        """
        B = x.shape[0]
        device = x.device

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add register tokens
        register = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([register, x], dim=1)  # (B, num_reg + N, D)

        # Get RoPE embeddings
        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Conditioning
        c = self.alpha_embed(alpha)

        text_list = [text_condition] if isinstance(text_condition, str) else text_condition

        if len(text_list) != B:
            raise ValueError(f"Mismatch: {len(text_list)} texts vs batch {B}")

        # =========================
        # ⭐ STEP 1: split segments
        # =========================
        batch_segments = [split_segments(t) for t in text_list]
        num_segments = [len(segs) for segs in batch_segments]
        max_segments = max(num_segments)

        # flatten 成 (B * num_segments)
        flat_segments = [seg for segs in batch_segments for seg in segs]

        # =========================
        # ⭐ STEP 2: encode
        # =========================
        encoded_tokens = self.text_encoder.encode_texts(
            flat_segments, device=x.device
        )
        breakpoint()
        # (B*num_segments, D)

        # =========================
        # ⭐ STEP 3: reshape
        # =========================
        L = encoded_tokens.shape[1]
        D = encoded_tokens.shape[2]

        encoded_tokens = encoded_tokens.view(B, max_segments, L, D)
        attention_mask = attention_mask.view(B, max_segments, L)

        # 合并 segment 和 token 维度
        context_tokens = encoded_tokens.reshape(B, max_segments * L, D)
        context_mask = attention_mask.reshape(B, max_segments * L).to(torch.bool)

        # =========================
        # ⭐ STEP 4: projection
        # =========================
        context_tokens = self.context_proj(context_tokens.to(torch.float32))

        if self.use_style_embed:
            c = c + self.style_embed(B, device)

        # Transformer blocks
        for block in self.blocks:
            x = block(
                x,
                c,
                context=context_tokens,
                context_mask=context_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        # Remove register tokens
        x = x[:, self.num_register_tokens:, :]

        # Final layer and unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        text_condition: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[str], str],
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        Runs two forward passes: conditional and unconditional.

        Args:
            x: Noisy input image, shape (B, C, H, W)
            text_condition: Conditioning token vectors, shape (B, N_ctx, D_ctx)
            alpha: Guidance scalar

        Returns:
            Generated images, shape (B, C, H, W)
        """
        B = x.shape[0]
        device = x.device

        # Create alpha tensor
        alpha_tensor = torch.full((B,), alpha, device=device, dtype=x.dtype)

        # Duplicate inputs for conditional and unconditional
        x_combined = torch.cat([x, x], dim=0)
        if isinstance(text_condition, tuple):
            if len(text_condition) != 2:
                raise ValueError("Expected text_condition tuple as (input_ids, attention_mask).")
            text_condition_combined = (
                torch.cat([text_condition[0], text_condition[0]], dim=0),
                torch.cat([text_condition[1], text_condition[1]], dim=0),
            )
        elif isinstance(text_condition, str):
            text_condition_combined = [text_condition] * (2 * B)
        elif isinstance(text_condition, list) and (len(text_condition) == 0 or isinstance(text_condition[0], str)):
            if len(text_condition) == 1 and B > 1:
                text_condition = text_condition * B
            if len(text_condition) != B:
                raise ValueError(
                    f"String text_condition batch mismatch for CFG: got {len(text_condition)} texts for batch size {B}"
                )
            text_condition_combined = text_condition + text_condition
        else:
            text_condition_combined = torch.cat([text_condition, text_condition], dim=0)
        alpha_combined = torch.cat([alpha_tensor, alpha_tensor], dim=0)

        # Force unconditional for second half
        force_drop = torch.cat([
            torch.zeros(B, device=device),
            torch.ones(B, device=device),
        ]).bool()

        # Forward pass
        out = self.forward(x_combined, text_condition_combined, alpha_combined, force_drop)

        # Split and apply CFG
        cond, uncond = out.chunk(2, dim=0)
        return uncond + alpha * (cond - uncond)


def DriftDiT_Tiny(img_size=32, in_channels=3, condition_dim=256, **kwargs):
    """DriftDiT-Tiny: depth=6, hidden_dim=256, heads=4 -> ~5M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        condition_dim=condition_dim,
        **kwargs,
    )


def DriftDiT_Small(img_size=32, in_channels=3, condition_dim=256, **kwargs):
    """DriftDiT-Small: depth=8, hidden_dim=384, heads=6 -> ~15M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        condition_dim=condition_dim,
        **kwargs,
    )


# Model registry
DriftDiT_models = {
    "DriftDiT-Tiny": DriftDiT_Tiny,
    "DriftDiT-Small": DriftDiT_Small,
}
