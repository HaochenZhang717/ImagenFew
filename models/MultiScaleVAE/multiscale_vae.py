import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Decoder, Encoder, GuidedDecoder, GuidedEncoder
# from modules import Decoder, Encoder, GuidedDecoder, GuidedEncoder
import torch
import math



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C]
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(
            1, math.floor((self.kernel_size - 1) // 2), 1
        )
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


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
            attn_drop: float = 0.1,
            proj_drop: float = 0.1,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)  # rope may change the q,k's dtype
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
            is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
    ):
        super().__init__()
        mlp_ratio = 4.0
        # Initialize normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Initialize attention layer
        self.attn = NormAttention(hidden_size, num_heads=num_heads)

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim), hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = moving_avg(5, stride=1)
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        coarse_seasonal_down = self.downsample(seasonal.transpose(1, 2))
        coarse_seasonal = self.upsample(coarse_seasonal_down).transpose(1, 2)

        return trend, coarse_seasonal, seasonal



class SymmetricFusion(nn.Module):
    def __init__(self, z_channels):
        super().__init__()
        self.fc_fusion = nn.Linear(z_channels * 3, z_channels)

    def forward(self, z1, z2, z3):
        # z1,z2,z3: [B, C, L]
        fused = torch.cat([z1, z2, z3], dim=1)          # [B, 3C, L]
        fused = fused.permute(0, 2, 1)                  # [B, L, 3C]
        fused = self.fc_fusion(fused)                   # [B, L, C]
        fused = fused.permute(0, 2, 1)                  # [B, C, L]
        return fused


class SymmetricDecomp(nn.Module):
    def __init__(self, z_channels):
        super().__init__()
        self.fc_decomp = nn.Linear(z_channels, z_channels * 3)

    def forward(self, f):
        # f: [B, C, L]
        decomp = self.fc_decomp(f.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 3C, L]
        return torch.chunk(decomp, 3, dim=1)


class DiagGaussianHead(nn.Module):
    """
    Map feature f -> (mu, logvar), then sample with reparameterization trick.
    """
    def __init__(self, z_channels, latent_channels=None):
        super().__init__()
        if latent_channels is None:
            latent_channels = z_channels
        self.to_mu = nn.Conv1d(z_channels, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv1d(z_channels, latent_channels, kernel_size=1)

    def forward(self, x):
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        if torch.is_grad_enabled():
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


def kl_divergence_standard_normal(mu, logvar, reduce=True):
    """
    KL(q(z|x) || p(z)), where p(z)=N(0,I)
    mu, logvar: [B, C, L]
    """
    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar)
    if reduce:
        return kl.mean()
    return kl


class DualVAE(nn.Module):
    def __init__(
        self,
        z_channels=32,
        latent_channels=None,
        ch=128,
        dropout=0.0,
        test_mode=False,
        ch_mult=(1, 1, 2),
        dynamic_size=128,
        num_res_blocks=2,
        seq_len=24,
        one_token_pool=False,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.z_channels = z_channels
        self.latent_channels = latent_channels if latent_channels is not None else z_channels
        self.seq_len = seq_len
        self.one_token_pool = one_token_pool
        self.downsample_ratio = 2 ** (len(ch_mult) - 1)
        self.default_latent_len = max(1, seq_len // self.downsample_ratio)

        self.decomp_ts = series_decomp_multi()

        coarse_config = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            using_sa=False,
            using_mid_sa=False,
            dynamic_size=dynamic_size,
        )
        fine_config = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            using_sa=False,
            using_mid_sa=False,
            dynamic_size=dynamic_size,
        )

        self.encoder_low_freq = Encoder(**coarse_config)
        self.decoder_low_freq = Decoder(**coarse_config)

        if self.one_token_pool:
            self.encoder_mid_freq = Encoder(**fine_config)
            self.decoder_mid_freq = Decoder(**fine_config)
        else:
            self.encoder_mid_freq = GuidedEncoder(**fine_config)
            self.decoder_mid_freq = GuidedDecoder(**fine_config)

        self.encoder_high_freq = GuidedEncoder(**fine_config)
        self.decoder_high_freq = GuidedDecoder(**fine_config)

        # VAE head: fused feature -> mu/logvar
        self.posterior_low_freq = DiagGaussianHead(z_channels=z_channels, latent_channels=self.latent_channels)
        self.posterior_mid_freq = DiagGaussianHead(z_channels=z_channels, latent_channels=self.latent_channels)
        self.posterior_high_freq = DiagGaussianHead(z_channels=z_channels, latent_channels=self.latent_channels)

        # optional post latent conv
        self.post_latent_conv_low_freq = nn.Conv1d(self.latent_channels, z_channels, kernel_size=3, padding=1)
        self.post_latent_conv_mid_freq = nn.Conv1d(self.latent_channels, z_channels, kernel_size=3, padding=1)
        self.post_latent_conv_high_freq = nn.Conv1d(self.latent_channels, z_channels, kernel_size=3, padding=1)


        if test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    def _pool_to_one_token(self, x):
        if not self.one_token_pool:
            return x
        return x.mean(dim=-1, keepdim=True)

    def _expand_one_token_latent(self, z, target_len=None):
        if not self.one_token_pool:
            return z
        if target_len is None:
            target_len = self.default_latent_len
        if z.shape[-1] == target_len:
            return z
        if z.shape[-1] != 1:
            raise ValueError(
                f"Expected one-token latent with length 1 in one_token_pool mode, got {z.shape[-1]}"
            )
        return z.expand(-1, -1, target_len)

    def _concat_one_token_latents(self, z_low_freq, z_mid_freq, z_high_freq):
        if not self.one_token_pool:
            return z_low_freq, z_mid_freq, z_high_freq
        return torch.cat([z_low_freq, z_mid_freq, z_high_freq], dim=1)

    def _split_one_token_latents(self, z):
        if not self.one_token_pool:
            raise ValueError("Channel-wise latent splitting is only used in one_token_pool mode.")
        expected_channels = self.latent_channels * 3
        if z.ndim != 3:
            raise ValueError(f"Expected latent tensor with shape [B, C, L], got {tuple(z.shape)}")
        if z.shape[1] != expected_channels:
            raise ValueError(
                f"Expected concatenated latent channel size {expected_channels}, got {z.shape[1]}"
            )
        return torch.chunk(z, 3, dim=1)

    def encode_parts(self, inp):
        """
        inp: [B, L, C]
        """
        low_freq, mid_freq, high_freq = self.decomp_ts(inp)
        ze_low_freq = self.encoder_low_freq(low_freq)
        ze_mid_freq = self.encoder_mid_freq(mid_freq)
        ze_high_freq = self.encoder_high_freq(high_freq)
        return low_freq, mid_freq, high_freq, ze_low_freq, ze_mid_freq, ze_high_freq

    def encode_to_posterior(self, inp):
        low_freq, mid_freq, high_freq, ze_low_freq, ze_mid_freq, ze_high_freq = self.encode_parts(inp)
        ze_low_freq_pooled = self._pool_to_one_token(ze_low_freq)
        ze_mid_freq_pooled = self._pool_to_one_token(ze_mid_freq)
        ze_high_freq_pooled = self._pool_to_one_token(ze_high_freq)
        mu_low_freq, logvar_low_freq = self.posterior_low_freq(ze_low_freq_pooled)
        mu_mid_freq, logvar_mid_freq = self.posterior_mid_freq(ze_mid_freq_pooled)
        mu_high_freq, logvar_high_freq = self.posterior_high_freq(ze_high_freq_pooled)

        return low_freq, mid_freq, high_freq, ze_low_freq, ze_mid_freq, ze_high_freq, mu_low_freq, logvar_low_freq, mu_mid_freq, logvar_mid_freq, mu_high_freq, logvar_high_freq


    def decode_from_latent(self, z_low_freq, z_mid_freq, z_high_freq, data_channels, latent_len=None):
        """
        z_trend: [B, latent_channels, L']
        z_seasonal: [B, latent_channels, L']
        z_coarse_seasonal: [B, latent_channels, L']
        """
        z_low_freq = self._expand_one_token_latent(z_low_freq, target_len=latent_len)
        z_mid_freq = self._expand_one_token_latent(z_mid_freq, target_len=latent_len)
        z_high_freq = self._expand_one_token_latent(z_high_freq, target_len=latent_len)

        recon_low_freq = self.decoder_low_freq(self.post_latent_conv_low_freq(z_low_freq), data_channels)

        if self.one_token_pool:
            recon_mid_freq = self.decoder_mid_freq(
                self.post_latent_conv_mid_freq(z_mid_freq),
                data_channels,
            )
        else:
            recon_mid_freq = self.decoder_mid_freq(
                self.post_latent_conv_mid_freq(z_mid_freq),
                recon_low_freq,
                out_features=data_channels,
            )

        recon_high_freq = self.decoder_high_freq(
            self.post_latent_conv_high_freq(z_high_freq),
            recon_mid_freq,
            out_features=data_channels,
        )

        total_recon = recon_low_freq + recon_high_freq
        return z_low_freq, z_mid_freq, z_high_freq, recon_low_freq, recon_mid_freq, recon_high_freq, total_recon

    def forward(self, inp):
        (low_freq, mid_freq, high_freq,
         ze_low_freq, ze_mid_freq, ze_high_freq,
         mu_low_freq, logvar_low_freq,
         mu_mid_freq, logvar_mid_freq,
         mu_high_freq, logvar_high_freq
         ) = self.encode_to_posterior(inp)


        z_low_freq = DiagGaussianHead.reparameterize(mu_low_freq, logvar_low_freq)
        z_mid_freq = DiagGaussianHead.reparameterize(mu_mid_freq, logvar_mid_freq)
        z_high_freq = DiagGaussianHead.reparameterize(mu_high_freq, logvar_high_freq)

        (
            z_low_freq, z_mid_freq, z_high_freq,
            recon_low_freq, recon_mid_freq, recon_high_freq, total_recon
        ) = self.decode_from_latent(
            z_low_freq,
            z_mid_freq,
            z_high_freq,
            inp.shape[-1],
            latent_len=ze_low_freq.shape[-1],
        )

        kl_loss_low_freq = kl_divergence_standard_normal(mu_low_freq, logvar_low_freq, reduce=True)
        kl_loss_mid_freq = kl_divergence_standard_normal(mu_mid_freq, logvar_mid_freq, reduce=True)
        kl_loss_high_freq = kl_divergence_standard_normal(mu_high_freq, logvar_high_freq, reduce=True)

        recon_loss_low_freq = nn.MSELoss()(recon_low_freq, low_freq.detach())
        # print(f"recon_low_freq: {recon_low_freq.shape}")

        recon_loss_mid_freq = nn.MSELoss()(recon_mid_freq, mid_freq.detach())
        recon_loss_high_freq = nn.MSELoss()(recon_high_freq, high_freq.detach())
        recon_loss_overall = nn.MSELoss()(total_recon, inp.detach())

        return {
            "low_freq": low_freq,
            "mid_freq": mid_freq,
            "high_freq": high_freq,

            "recon_low_freq": recon_low_freq,
            "recon_mid_freq": recon_mid_freq,
            "recon_high_freq": recon_high_freq,
            "total_recon": total_recon,

            "mu_low_freq": mu_low_freq,
            "mu_mid_freq": mu_mid_freq,
            "mu_high_freq": mu_high_freq,

            "logvar_low_freq": logvar_low_freq,
            "logvar_mid_freq": logvar_mid_freq,
            "logvar_high_freq": logvar_high_freq,

            "z_low_freq": z_low_freq,
            "z_mid_freq": z_mid_freq,
            "z_high_freq": z_high_freq,

            'kl_loss_low_freq': kl_loss_low_freq,
            'kl_loss_mid_freq': kl_loss_mid_freq,
            'kl_loss_high_freq': kl_loss_high_freq,

            "recon_loss_low_freq": recon_loss_low_freq,
            "recon_loss_mid_freq": recon_loss_mid_freq,
            "recon_loss_high_freq": recon_loss_high_freq,
            "recon_loss_overall": recon_loss_overall,
        }

    def ts_to_z(self, inp, sample=True, return_dict=False):
        (
            _, _, _,
            _, _, _,
            mu_low_freq, logvar_low_freq,
            mu_mid_freq, logvar_mid_freq,
            mu_high_freq, logvar_high_freq,
        ) = self.encode_to_posterior(inp)

        if sample:
            z_low_freq = DiagGaussianHead.reparameterize(mu_low_freq, logvar_low_freq)
            z_mid_freq = DiagGaussianHead.reparameterize(mu_mid_freq, logvar_mid_freq)
            z_high_freq = DiagGaussianHead.reparameterize(mu_high_freq, logvar_high_freq)
        else:
            z_low_freq = mu_low_freq
            z_mid_freq = mu_mid_freq
            z_high_freq = mu_high_freq

        if return_dict:
            if self.one_token_pool:
                z_concat = self._concat_one_token_latents(z_low_freq, z_mid_freq, z_high_freq)
                mu_concat = self._concat_one_token_latents(mu_low_freq, mu_mid_freq, mu_high_freq)
                logvar_concat = self._concat_one_token_latents(logvar_low_freq, logvar_mid_freq, logvar_high_freq)
            else:
                z_concat = None
                mu_concat = None
                logvar_concat = None
            return {
                "z_low_freq": z_low_freq,
                "z_mid_freq": z_mid_freq,
                "z_high_freq": z_high_freq,
                "mu_low_freq": mu_low_freq,
                "mu_mid_freq": mu_mid_freq,
                "mu_high_freq": mu_high_freq,
                "logvar_low_freq": logvar_low_freq,
                "logvar_mid_freq": logvar_mid_freq,
                "logvar_high_freq": logvar_high_freq,
                "z": z_concat,
                "mu": mu_concat,
                "logvar": logvar_concat,
            }

        if self.one_token_pool:
            return self._concat_one_token_latents(z_low_freq, z_mid_freq, z_high_freq)

        return z_low_freq, z_mid_freq, z_high_freq

    def z_to_ts(self, z, data_channels=7, latent_len=None):
        """
        z can be:
          - a tuple/list: (z_low_freq, z_mid_freq, z_high_freq)
          - a dict with keys: z_low_freq, z_mid_freq, z_high_freq
        """
        if isinstance(z, dict):
            if self.one_token_pool and z.get("z") is not None:
                z_low_freq, z_mid_freq, z_high_freq = self._split_one_token_latents(z["z"])
            else:
                z_low_freq = z["z_low_freq"]
                z_mid_freq = z["z_mid_freq"]
                z_high_freq = z["z_high_freq"]
        elif torch.is_tensor(z):
            if self.one_token_pool:
                z_low_freq, z_mid_freq, z_high_freq = self._split_one_token_latents(z)
            else:
                raise ValueError("Tensor latent input is only supported in one_token_pool mode.")
        else:
            z_low_freq, z_mid_freq, z_high_freq = z

        _, _, _, recon_low_freq, recon_mid_freq, recon_high_freq, total_recon = \
            self.decode_from_latent(
                z_low_freq,
                z_mid_freq,
                z_high_freq,
                data_channels=data_channels,
                latent_len=latent_len,
            )

        return total_recon

    def z_to_ts_decomp(self, z, data_channels=7, latent_len=None):
        """
        z can be:
          - a tuple/list: (z_low_freq, z_mid_freq, z_high_freq)
          - a dict with keys: z_low_freq, z_mid_freq, z_high_freq
        """
        if isinstance(z, dict):
            if self.one_token_pool and z.get("z") is not None:
                z_low_freq, z_mid_freq, z_high_freq = self._split_one_token_latents(z["z"])
            else:
                z_low_freq = z["z_low_freq"]
                z_mid_freq = z["z_mid_freq"]
                z_high_freq = z["z_high_freq"]
        elif torch.is_tensor(z):
            if self.one_token_pool:
                z_low_freq, z_mid_freq, z_high_freq = self._split_one_token_latents(z)
            else:
                raise ValueError("Tensor latent input is only supported in one_token_pool mode.")
        else:
            z_low_freq, z_mid_freq, z_high_freq = z

        _, _, _, recon_low_freq, recon_mid_freq, recon_high_freq, _ = \
            self.decode_from_latent(
                z_low_freq,
                z_mid_freq,
                z_high_freq,
                data_channels=data_channels,
                latent_len=latent_len,
            )

        return recon_low_freq, recon_mid_freq, recon_high_freq


def test_dual_vae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, L, C = 4, 128, 6
    x = torch.randn(B, L, C).to(device)

    model = DualVAE(
        in_channels=C,
        z_channels=32,
        ch=128,
        ch_mult=(1, 1, 2),
        test_mode=False,
    ).to(device)

    print("=" * 50)
    print("🔍 Running forward pass...")
    out = model(x)

    # ========================
    # 1. Shape checks
    # ========================
    print("\n[Shape Check]")
    print("input:", x.shape)
    print("trend:", out["trend"].shape)
    print("seasonal:", out["seasonal"].shape)
    print("coarse_seasonal:", out["coarse_seasonal"].shape)
    print("recon:", out["total_recon"].shape)

    print("z_t:", out["z_t"].shape)
    print("z_s:", out["z_s"].shape)
    print("z_c:", out["z_c"].shape)

    # ========================
    # 2. Reconstruction check
    # ========================
    print("\n[Reconstruction Check]")
    recon_loss = F.mse_loss(out["total_recon"], x)
    print("recon_loss:", recon_loss.item())

    # ========================
    # 3. KL check
    # ========================
    print("\n[KL Check]")
    print("kl_loss:", out["kl_loss"].item())

    # ========================
    # 4. Encode → Decode consistency
    # ========================
    print("\n[Encode → Decode Test]")
    z = model.ts_to_z(x, sample=False)
    x_recon = model.z_to_ts(z)

    recon2_loss = F.mse_loss(x_recon, x)
    print("recon_from_mu_loss:", recon2_loss.item())

    # ========================
    # 5. Decomposition decode
    # ========================
    print("\n[Decomposition Decode]")
    trend_rec, coarse_rec, seasonal_rec = model.z_to_ts_decomp(z)
    print("trend_rec:", trend_rec.shape)
    print("coarse_rec:", coarse_rec.shape)
    print("seasonal_rec:", seasonal_rec.shape)

    # ========================
    # 6. Backward check
    # ========================
    print("\n[Backward Check]")
    loss = recon_loss + out["aux_loss"]
    loss.backward()
    print("backward: ✅ success")

    # ========================
    # 7. Gradient sanity
    # ========================
    print("\n[Gradient Check]")
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item()

    print("grad_norm:", total_norm)

    print("\n✅ All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_dual_vae()


    ##scp -r haochenz@unites4.cs.unc.edu:/playpen-shared/haochenz/ckpts_multiscale/ETTh1/plots ../
