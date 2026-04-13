import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MultiScaleVAE.modules import DynamicConv1d


class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        mlp_ratio = 4.0
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = NormAttention(hidden_size, num_heads=num_heads)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim), hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicConvStem(nn.Module):
    """
    Input stem with a dynamic first convolution so the encoder can accept
    varying channel counts during pretraining.
    """

    def __init__(self, dynamic_size, hidden_size):
        super().__init__()
        self.conv1 = DynamicConv1d(dynamic_size, kernel=2, out_channels=hidden_size // 2, down=True)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


class DynamicConvHead(nn.Module):
    """
    Output projection with dynamic output channels so the decoder can reconstruct
    datasets with different numbers of variables.
    """

    def __init__(self, dynamic_size):
        super().__init__()
        self.proj = DynamicConv1d(dynamic_size, kernel=5)

    def forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        return self.proj(x, out_features=out_channels)


class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        dynamic_size=None,
    ):
        super().__init__()
        if dynamic_size is None:
            dynamic_size = input_dim

        self.conv = DynamicConvStem(dynamic_size=dynamic_size, hidden_size=hidden_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        x: (B, C, T)
        returns:
            mu:     (B, T/4, latent_dim)
            logvar: (B, T/4, latent_dim)
        """
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x)

        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)
        return mu, logvar


class TimeSeriesDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
        dynamic_size=None,
    ):
        super().__init__()
        if dynamic_size is None:
            dynamic_size = output_dim

        self.input_proj = nn.Conv1d(latent_dim, hidden_size, kernel_size=1)
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2)
        self.output_proj = DynamicConvHead(dynamic_size=dynamic_size)

    def forward(self, z, out_channels: int):
        """
        z: (B, latent_dim, T/4)
        return: (B, C, T)
        """
        x = self.input_proj(z)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.output_proj(x, out_channels=out_channels)
        return x


class SimpleVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        beta=0.001,
        dynamic_size=None,
    ):
        super().__init__()
        if dynamic_size is None:
            dynamic_size = max(input_dim, output_dim)

        self.beta = beta
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dynamic_size=dynamic_size,
        )
        self.decoder = TimeSeriesDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            dynamic_size=dynamic_size,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: (B, C, T)
        returns:
            mu, logvar, z with shape (B, T/4, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z, out_channels=None):
        if out_channels is None:
            out_channels = self.output_dim
        return self.decoder(z.permute(0, 2, 1), out_channels=out_channels)

    def get_embedding(self, x, use_mu=True):
        mu, logvar = self.encoder(x)
        if use_mu:
            return mu
        return self.reparameterize(mu, logvar)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z, out_channels=x.shape[1])
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


if __name__ == "__main__":
    model = SimpleVAE(
        input_dim=8,
        output_dim=8,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        latent_dim=128,
        beta=0.001,
        dynamic_size=32,
    )

    x = torch.randn(8, 5, 128)
    out = model(x)
    loss_dict = model.loss_function(x, out["recon"], out["mu"], out["logvar"])
    loss = loss_dict["loss"]
    loss.backward()
    print("recon:", out["recon"].shape)
    print("mu:", out["mu"].shape)
    print("logvar:", out["logvar"].shape)
    print("z:", out["z"].shape)
    print("backward ok")
