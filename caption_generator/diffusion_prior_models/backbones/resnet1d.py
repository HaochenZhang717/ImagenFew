import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import GaussianFourierEmbedding, RMSNorm, modulate


class ResBlock1D(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, use_rmsnorm=True, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size),
        )
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=1)

        h = modulate(self.norm1(x), shift1, scale1)
        h = F.silu(h)
        h = h.transpose(1, 2)
        h = self.conv1(h)
        h = h.transpose(1, 2)

        h = modulate(self.norm2(h), shift2, scale2)
        h = F.silu(h)
        h = self.dropout(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        return x + h


class ResNet1D(nn.Module):
    def __init__(
        self,
        seq_len,
        token_dim,
        hidden_size=256,
        depth=8,
        kernel_size=3,
        use_rmsnorm=True,
        dropout=0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = token_dim
        self.out_channels = token_dim
        self.hidden_size = hidden_size

        self.x_embedder = nn.Linear(token_dim, hidden_size)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.blocks = nn.ModuleList(
            [
                ResBlock1D(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    use_rmsnorm=use_rmsnorm,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        if use_rmsnorm:
            self.final_norm = RMSNorm(hidden_size)
        else:
            self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(hidden_size, token_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.final_proj.bias, 0)

    def forward(self, x, t, y=None):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, L, D), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")

        h = self.x_embedder(x)
        c = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, c)
        h = self.final_norm(h)
        return self.final_proj(h)

    def forward_with_cfg(self, x, t, y=None, cfg_scale=None, **kwargs):
        return self.forward(x, t, y=y)

    def forward_with_autoguidance(self, x, t, y=None, additional_model_forward=None, **kwargs):
        return self.forward(x, t, y=y)
