import torch
import torch.nn as nn
import torch.nn.functional as F


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


class EncoderConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        kernel_size = 2 if downsample else 3
        padding = 0 if downsample else 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ConvStem(nn.Module):
    def __init__(self, input_dim, encoder_channels, downsample_stages: int):
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must contain at least 2 stage widths")
        if downsample_stages < 1 or downsample_stages > len(encoder_channels):
            raise ValueError("downsample_stages must be in [1, len(encoder_channels)]")
        self.input_proj = EncoderConvStage(
            input_dim,
            encoder_channels[0],
            downsample=downsample_stages >= 1,
        )
        self.stages = nn.ModuleList()

        in_channels = encoder_channels[0]
        for idx, out_channels in enumerate(encoder_channels[1:], start=1):
            self.stages.append(
                EncoderConvStage(
                    in_channels,
                    out_channels,
                    downsample=idx < downsample_stages,
                )
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for stage in self.stages:
            x = stage(x)
        return x


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        dynamic_size=None,
        encoder_channels=None,
        encoder_downsample_stages=2,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [
                max(hidden_size // 2, 1),
                hidden_size,
                hidden_size,
                hidden_size,
            ]
        encoder_channels = list(encoder_channels)
        self.downsample_factor = 2 ** encoder_downsample_stages

        self.conv = ConvStem(
            input_dim=input_dim,
            encoder_channels=encoder_channels,
            downsample_stages=encoder_downsample_stages,
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(encoder_channels[-1], num_heads) for _ in range(num_layers)]
        )
        self.to_mu = nn.Linear(encoder_channels[-1], latent_dim)
        self.to_logvar = nn.Linear(encoder_channels[-1], latent_dim)

    def forward(self, x):
        """
        x: (B, C, T)
        returns:
            mu:     (B, T/downsample_factor, latent_dim)
            logvar: (B, T/downsample_factor, latent_dim)
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
        seq_len=24,
        dynamic_size=None,
        decoder_channels=None,
        decoder_res_blocks=1,
        decoder_dropout=0.0,
        upsample_stages=2,
        latent_downsample_factor=4,
    ):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [hidden_size, hidden_size, max(hidden_size // 2, 1), max(hidden_size // 2, 1)]
        decoder_channels = list(decoder_channels)
        if len(decoder_channels) < 4:
            raise ValueError(
                "decoder_channels must contain at least 4 stage widths"
            )
        if latent_downsample_factor < 1 or seq_len % latent_downsample_factor != 0:
            raise ValueError("seq_len must be divisible by latent_downsample_factor")
        if upsample_stages < 1 or upsample_stages > len(decoder_channels):
            raise ValueError("upsample_stages must be in [1, len(decoder_channels)]")

        self.seq_len = seq_len
        self.latent_seq_len = seq_len // latent_downsample_factor
        self.input_proj = nn.Conv1d(latent_dim, decoder_channels[0], kernel_size=1)
        self.stages = nn.ModuleList()

        in_channels = decoder_channels[0]
        for idx, out_channels in enumerate(decoder_channels):
            stage = nn.ModuleDict({
                "resblocks": nn.Sequential(
                    *[
                        ResidualConvBlock(
                            in_channels,
                            kernel_size=5,
                            dropout=decoder_dropout,
                        )
                        for _ in range(decoder_res_blocks)
                    ]
                ),
                "channel_proj": nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity(),
            })
            stage["upsample"] = (
                nn.Upsample(scale_factor=2, mode="nearest")
                if idx < upsample_stages
                else nn.Identity()
            )
            self.stages.append(stage)
            in_channels = out_channels

        self.output_proj = ConvHead(in_channels=in_channels, out_channels=output_dim)

    def forward(self, z, out_channels: int):
        """
        z: (B, latent_dim, T/latent_downsample_factor)
        return: (B, C, T)
        """
        if out_channels != self.output_proj.proj.out_channels:
            raise ValueError(
                f"Decoder was built for out_channels={self.output_proj.proj.out_channels}, "
                f"but got out_channels={out_channels}."
            )
        x = self.input_proj(z)
        for stage in self.stages:
            x = stage["resblocks"](x)
            x = stage["channel_proj"](x)
            x = stage["upsample"](x)
        x = self.output_proj(x)
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
        encoder_channels=None,
        encoder_downsample_stages=2,
        decoder_channels=None,
        decoder_res_blocks=1,
        decoder_dropout=0.0,
        decoder_upsample_stages=2,
        seq_len=24,
    ):
        super().__init__()
        self.beta = beta
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.latent_downsample_factor = 2 ** encoder_downsample_stages
        if seq_len % self.latent_downsample_factor != 0:
            raise ValueError("seq_len must be divisible by 2 ** encoder_downsample_stages")
        self.latent_seq_len = seq_len // self.latent_downsample_factor

        self.encoder = TimeSeriesEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dynamic_size=dynamic_size,
            encoder_channels=encoder_channels,
            encoder_downsample_stages=encoder_downsample_stages,
        )
        self.decoder = TimeSeriesDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
            dynamic_size=dynamic_size,
            decoder_channels=decoder_channels,
            decoder_res_blocks=decoder_res_blocks,
            decoder_dropout=decoder_dropout,
            upsample_stages=decoder_upsample_stages,
            latent_downsample_factor=self.latent_downsample_factor,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: (B, C, T)
        returns:
            mu, logvar, z with shape (B, T/latent_downsample_factor, latent_dim)
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
        encoder_channels=[64, 96, 128, 128],
        encoder_downsample_stages=2,
        decoder_channels=[128, 128, 96, 64],
        decoder_res_blocks=2,
        decoder_dropout=0.0,
        decoder_upsample_stages=2,
        seq_len=128,
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
