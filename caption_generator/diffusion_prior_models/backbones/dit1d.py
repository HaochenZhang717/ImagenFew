import numpy as np
import torch
import torch.nn as nn

from .model_utils import GaussianFourierEmbedding, NormAttention, RMSNorm, SwiGLUFFN, modulate


class DiT1DBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_rmsnorm=True,
    ):
        super().__init__()
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiT1DFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, use_rmsnorm=True):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT1D(nn.Module):
    def __init__(
        self,
        seq_len,
        token_dim,
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_rmsnorm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = token_dim
        self.out_channels = token_dim
        self.hidden_size = hidden_size

        self.x_embedder = nn.Linear(token_dim, hidden_size)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiT1DBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DiT1DFinalLayer(hidden_size, token_dim, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos = np.arange(self.seq_len, dtype=np.float32)
        half_dim = self.hidden_size // 2
        omega = np.arange(half_dim, dtype=np.float64)
        omega /= max(half_dim, 1)
        omega = 1.0 / (10000 ** omega)
        out = np.einsum("m,d->md", pos, omega)
        pos_embed = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        if pos_embed.shape[1] < self.hidden_size:
            pos_embed = np.pad(pos_embed, ((0, 0), (0, self.hidden_size - pos_embed.shape[1])))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed[:, : self.hidden_size]).float().unsqueeze(0))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y=None):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, L, D), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")

        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, c)
        return self.final_layer(x, c)

    def forward_with_cfg(self, x, t, y=None, cfg_scale=None, **kwargs):
        return self.forward(x, t, y=y)

    def forward_with_autoguidance(self, x, t, y=None, additional_model_forward=None, **kwargs):
        return self.forward(x, t, y=y)
