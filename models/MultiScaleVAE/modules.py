import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':  return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':   return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')



class DynamicConv1d(torch.nn.Module):
    def __init__(self,
        dynamic_size, kernel, out_channels=None, bias=True, up=False, down=False,
        resample_filter=[1, 1], fused_resample=False,
        init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()

        if isinstance(dynamic_size, int):
            dynamic_size = [dynamic_size, dynamic_size]
        assert len(dynamic_size) == 2

        self.dynamic_size = dynamic_size
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample

        init_kwargs = dict(
            mode=init_mode,
            fan_in=self.dynamic_size[0] * kernel,
            fan_out=self.dynamic_size[1] * kernel
        )

        # [k, dyn_out, dyn_in]
        self.dynamic_weights = torch.nn.Parameter(
            weight_init([kernel, self.dynamic_size[1], self.dynamic_size[0]], **init_kwargs) * init_weight
        ) if kernel else None

        self.dynamic_bias = torch.nn.Parameter(
            weight_init([self.dynamic_size[1]], **init_kwargs) * init_bias
        ) if kernel and bias else None

        # 1D resample filter
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f[None, None, :] / f.sum()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, out_features=None):
        # x: [B, C, L]

        if out_features is None and self.out_channels is None:
            raise Exception("out_features & self.out_channels cannot both be None")

        out_features = self.out_channels if self.out_channels is not None else out_features

        # ---- weight interpolation ----
        if self.dynamic_weights is not None:
            # [k, dyn_out, dyn_in] → treat as [N=k, C=1, L=dyn_out*dyn_in]
            w = self.dynamic_weights.view(
                self.dynamic_weights.shape[0], 1, -1
            )

            w = torch.nn.functional.interpolate(
                w,
                size=out_features * x.size(1),
                mode='linear',
                align_corners=False
            )

            w = w.view(
                self.dynamic_weights.shape[0],
                out_features,
                x.size(1)
            )

            w = w.permute(1, 2, 0)  # [out, in, k]
        else:
            w = None

        # ---- bias interpolation ----
        if self.dynamic_bias is not None:
            b = torch.nn.functional.interpolate(
                self.dynamic_bias.view(1, 1, -1),
                size=out_features,
                mode='linear',
                align_corners=False
            ).view(-1)
        else:
            b = None

        w = w.to(x.dtype) if w is not None else None
        b = b.to(x.dtype) if b is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None

        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        # ---- fused resample ----
        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose1d(
                x,
                f.mul(2).tile([x.shape[1], 1, 1]),
                groups=x.shape[1],
                stride=2,
                padding=max(f_pad - w_pad, 0)
            )
            x = torch.nn.functional.conv1d(x, w, padding=max(w_pad - f_pad, 0))

        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv1d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv1d(
                x,
                f.tile([out_features, 1, 1]),
                groups=out_features,
                stride=2
            )

        else:
            if self.up:
                x = torch.nn.functional.conv_transpose1d(
                    x,
                    f.mul(2).tile([x.shape[1], 1, 1]),
                    groups=x.shape[1],
                    stride=2,
                    padding=f_pad
                )

            if self.down:
                x = torch.nn.functional.conv1d(
                    x,
                    f.tile([x.shape[1], 1, 1]),
                    groups=x.shape[1],
                    stride=2,
                    padding=f_pad
                )

            if w is not None:
                x = torch.nn.functional.conv1d(x, w, padding=w_pad)

        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))

        return x



class DynamicLinear(nn.Module):
    """
    A dynamic linear layer that can interpolate the weight size to support any given input and output feature dimension.
    """

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super(DynamicLinear, self).__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required !!!"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features=None):
        """
        Forward pass for the dynamic linear layer.
        """
        fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in:]
        this_bias = self.bias
        in_features = x.shape[-1]

        if out_features is None:
            out_features = self.out_features

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features-self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
                    out_features, self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)




# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder']

"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""


# Helper activation function: Swish
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=16):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))
        # return self.conv(F.interpolate(x, scale_factor=2, mode='linear'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None,
                 dropout):  # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        # Main processing path
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Shortcut connection if channel sizes differ
        self.nin_shortcut = nn.Conv1d(in_channels, self.out_channels, kernel_size=1) \
            if in_channels != self.out_channels else nn.Identity()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)  # Swish activation
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.nin_shortcut(x) + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = nn.Conv1d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Compute Q, K, V projections
        qkv = self.qkv(self.norm(x))  # [B, 3C, L]
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, C, L]

        # Reshape for matrix multiplication
        q = q.permute(0, 2, 1)  # [B, L, C]
        k = k  # [B, C, L]
        v = v  # [B, C, L]

        # Compute attention scores
        attn_weights = torch.bmm(q, k) * self.w_ratio  # [B, L, L]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        h = torch.bmm(v, attn_weights)  # [B, C, L]

        # Final projection and residual connection
        return x + self.proj_out(h)

def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
            self, *,
            ch=128,  # Base channel count
            ch_mult=(1, 2, 4, 8),  # Channel multipliers per resolution stage
            num_res_blocks=2,  # Number of residual blocks per stage
            dropout=0.0,  # Dropout probability
            in_channels=6,  # Input channels (e.g., for 3-channel features)
            z_channels,  # Latent space channels
            double_z=False,  # Whether to output double channels (μ and σ for VAE)
            using_sa=True,  # Use self-attention in last stage
            using_mid_sa=True,  # Use self-attention in middle block
            dynamic_size=None,  # If set, use DynamicConv1d for conv_in and conv_out
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1) # Total reduction factor
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # Initial convolution
        self.conv_in = DynamicConv1d(dynamic_size, 3, out_channels=self.ch)
        # if dynamic_size is not None:
        #     self.conv_in = DynamicConv1d(dynamic_size, 3, out_channels=self.ch)
        # else:
        #     self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, padding=1)
        # Downsampling stages
        in_ch_mult = (1,) + tuple(ch_mult)
        # print("ch_mult",ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, 3, stride=1, padding=1)

    def forward(self, x):
        # downsampling
        x = x.transpose(1, 2) # B,L,C -> B,C,L
        h = self.conv_in(x) # 128*32*24
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # print("h",h.shape)
        # Output shape: [B, z_channels*2, L//downsample_ratio]
        return h


class Decoder(nn.Module):
    def __init__(
        self, *,
        ch=128,                  # Base channel count
        ch_mult=(1, 2, 4, 8),    # Channel multipliers (reverse of encoder)
        num_res_blocks=2,        # Number of residual blocks per stage
        dropout=0.0,
        z_channels,              # Latent space channels
        using_sa=True,           # Use self-attention in first stage
        using_mid_sa=True,        # Use self-attention in middle block
        dynamic_size=None,  # If set, use DynamicConv1d for conv_in and conv_out
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # self.in_channels = in_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv1d(block_in, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = DynamicConv1d(dynamic_size, 3,)


    def forward(self, z, out_features):
        # Project latent to initial channels
        h = self.conv_in(z)

        # Middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h)), out_features)
        h = h.transpose(1, 2)  # B,C,L -> B,L,C
        return h
    

class CrossAttnBlock(nn.Module):
    def __init__(self, query_channels, guide_channels=None, heads=4, dynamic_guide_size=None):
        super().__init__()
        self.dynamic_guide = dynamic_guide_size is not None
        if self.dynamic_guide:
            self.guide_proj = DynamicConv1d(dynamic_guide_size, kernel=1, out_channels=query_channels)
        elif guide_channels != query_channels:
            self.guide_proj = nn.Conv1d(guide_channels, query_channels, kernel_size=1)
        else:
            self.guide_proj = nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=query_channels, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(query_channels)

    def forward(self, x, guide):
        g = guide.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        if self.dynamic_guide:
            guide_aligned = self.guide_proj(g, out_features=x.size(1))
        else:
            guide_aligned = self.guide_proj(g)
        #  (B, L, C)
        q  = x.permute(0, 2, 1)          # (B, L_q, C_q)
        kv = guide_aligned.permute(0, 2, 1)  # (B, L_g, C_q)
        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        out = self.norm(attn_out + q)
        return out.permute(0, 2, 1)      # (B, C_q, L_q)


class DummyCross(nn.Module):
    def forward(self, x, guide):
        return x  #

class SpectralFeatureExtractor(nn.Module):
    def __init__(self,
                 dynamic_size,
                 feat_dim,
                 n_fft=24,
                 low_freq: int = 1,
                 factor: float = 1.0):
        super().__init__()
        self.n_fft = n_fft
        self.low_freq = low_freq
        self.factor = factor
        max_f = n_fft // 2 + 1 - low_freq
        max_topk = int(factor * math.log(max_f))
        self.mlp = nn.Sequential(
            # nn.Linear(in_channels * max_topk * 2, feat_dim),
            DynamicLinear(dynamic_size * max_topk * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, x):
        B, C, T = x.shape
        spec = torch.fft.rfft(x, n=self.n_fft, dim=2)  #  FFT
        spec = spec[:, :, self.low_freq:]
        F_bins = spec.size(2)
        topk = max(1, int(self.factor * math.log(F_bins)))
        mag = spec.abs().reshape(B * C, F_bins)
        vals, idx = torch.topk(mag, topk, dim=1, largest=True, sorted=True)
        batch_chan = torch.arange(B * C, device=x.device).unsqueeze(1)
        sel_spec = spec.reshape(B * C, F_bins)[batch_chan, idx]
        amp   = sel_spec.abs()
        phase = sel_spec.angle()
        feats = torch.cat([amp, phase], dim=1)
        feats = feats.view(B, C * topk * 2)
        out = self.mlp(feats)
        return out.unsqueeze(1)


class GuidedEncoder(nn.Module):
    def __init__(
            self, *,
            ch=128,  # Base channel count
            ch_mult=(1, 2, 4, 8),  # Channel multipliers per resolution stage
            num_res_blocks=2,  # Number of residual blocks per stage
            dropout=0.0,  # Dropout probability
            in_channels=6,  # Input channels (e.g., for 3-channel features)
            z_channels,  # Latent space channels
            using_sa=True,  # Use self-attention in last stage
            using_mid_sa=True,  # Use self-attention in middle block
            num_heads=4,
            cross_attn_level=[1],
            dynamic_size=None
            # cross_attn_level=[]
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1) # Total reduction factor
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.cross_attn_level = cross_attn_level  # 

        # Initial convolution
        # self.conv_in = torch.nn.Conv1d(in_channels, self.ch, kernel_size=3, padding=1)
        self.conv_in = DynamicConv1d(dynamic_size, kernel=3, out_channels=self.ch)
        self.spec_extractor=SpectralFeatureExtractor(dynamic_size, n_fft=12, feat_dim=z_channels)
        # Downsampling stages
        in_ch_mult = (1,) + tuple(ch_mult)
        # print("ch_mult",ch_mult)
        self.down = nn.ModuleList()
        self.cross = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            crosses = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))

                # 
                if i_level in self.cross_attn_level:
                    crosses.append(CrossAttnBlock(
                        query_channels=block_out,
                        guide_channels=z_channels,
                        heads=num_heads
                    ))
                else:
                    crosses.append(DummyCross())  # 


            down = nn.Module()
            down.block = block
            down.attn = attn
            down.cross   = crosses

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
            self.cross.insert(0, crosses)
            

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, z_channels, 3, stride=1, padding=1)


    def forward(self, x):
        # downsampling
        x = x.transpose(1, 2) # B,L,C -> B,C,L
        h = self.conv_in(x) # 128*32*24
        # 
        if self.cross_attn_level is not None:
            spec_feats = self.spec_extractor(x)  # [B, C, L] -> [B, feat_dim]
        for i_level in range(self.num_resolutions):
            # print("h.shape", h.shape)
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h = self.down[i_level].cross[i_block](h, spec_feats)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # Middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Final output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # print("h",h.shape)
        # Output shape: [B, z_channels*2, L//downsample_ratio]
        return h


class GuidedDecoder(nn.Module):
    def __init__(
        self, *,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        z_channels,
        using_sa=True,
        using_mid_sa=True,
        num_heads=4,
        cross_attn_level=[0],  #
        dynamic_size=None,  # If set, use DynamicConv1d for conv_in and conv_out
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.cross_attn_level = cross_attn_level  # 

        # initial conv projecting z to feature map
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv1d(z_channels, block_in, kernel_size=3, padding=1)

        # middle layers
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # prepare upsampling structures
        self.up = nn.ModuleList()
        self.cross = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            blocks = nn.ModuleList()
            atts   = nn.ModuleList()
            crosses = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                # optional self-attention only at highest resolution
                if i_level == self.num_resolutions - 1 and using_sa:
                    atts.append(make_attn(block_out, True))
                else:
                    atts.append(nn.Identity())
                # 
                if i_level in self.cross_attn_level:
                    crosses.append(CrossAttnBlock(
                        query_channels=block_out,
                        dynamic_guide_size=dynamic_size,
                        heads=num_heads
                    ))
                else:
                    crosses.append(DummyCross())  #
                block_in = block_out

            up_module = nn.Module()
            up_module.block   = blocks
            up_module.attn    = atts
            up_module.cross   = crosses
            if i_level != 0:
                up_module.upsample = Upsample2x(block_in)
            self.up.insert(0, up_module)
            # self.cross.insert(0, crosses)
        # output layers
        self.norm_out = Normalize(block_in)
        # self.conv_out = nn.Conv1d(block_in, in_channels, kernel_size=3, padding=1)
        self.conv_out = DynamicConv1d(dynamic_size, kernel=3)

    def forward(self, z, recon_coarse, out_features):
        # z: (B, C, L_high), recon_coarse: (B, C, L_low)
        h = self.conv_in(z)

        # middle processing
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling with cross-attention at every block
        for i_level in reversed(range(self.num_resolutions)):
            level = self.up[i_level]
            for j, block in enumerate(level.block):
                h = block(h)
                h = level.attn[j](h)
                h = level.cross[j](h, recon_coarse)
            if hasattr(level, 'upsample'):
                h = level.upsample(h)

        # finalize and return (B, L, C)
        h = self.conv_out(F.silu(self.norm_out(h)), out_features)
        return h.transpose(1, 2)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.linear2(F.gelu(self.linear1(x)))
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x



class DynamicConv2d(torch.nn.Module):
    def __init__(self,
        dynamic_size, kernel, out_channels=None, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        if isinstance(dynamic_size, int):
            dynamic_size = [dynamic_size, dynamic_size]
        assert len(dynamic_size) == 2
        self.dynamic_size = dynamic_size
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=self.dynamic_size[0]*kernel*kernel, fan_out=self.dynamic_size[1]*kernel*kernel)
        self.dynamic_weights = torch.nn.Parameter(weight_init([kernel, kernel] + self.dynamic_size, **init_kwargs) * init_weight) if kernel else None
        self.dynamic_bias = torch.nn.Parameter(weight_init([self.dynamic_size[1]], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, out_features=None):
        if out_features is None and self.out_channels is None:
            raise Exception("out_features & self.out_channels cannot both be None")
        out_features = self.out_channels if self.out_channels is not None else out_features
        weight = torch.nn.functional.interpolate(
                self.dynamic_weights,
                size=(out_features, x.size(1)),
                mode='bicubic',
                align_corners=False).permute(2,3,0,1) if self.dynamic_weights is not None else None
        bias = torch.nn.functional.interpolate(
                self.dynamic_bias.unflatten(0, [1,1,1,-1]),
                size=(1, out_features),
                mode='bicubic',
                align_corners=False).flatten() if self.dynamic_bias is not None else None
        w = weight.to(x.dtype) if weight is not None else None
        b = bias.to(x.dtype) if bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x
