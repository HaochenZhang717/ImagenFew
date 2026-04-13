import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .side_encoder import SideEncoder_Var


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_cross_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerDecoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerDecoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
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


class TsPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len * channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl * C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SidePatchEmbedding(nn.Module):
    def __init__(self, L_patch_len, channels, d_model):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Linear(L_patch_len * channels, d_model)

    def forward(self, x_in):
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl * C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchDecoder(nn.Module):
    def __init__(self, L_patch_len, d_model, channels):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.linear = nn.Linear(d_model, L_patch_len * channels)

    def forward(self, x):
        B, D, n_var, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        x = x.reshape(B, n_var, Nl, self.L_patch_len, self.channels).permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.channels, n_var, Nl * self.L_patch_len)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, condition_type="add"):
        super().__init__()
        self.condition_type = condition_type
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        if condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(heads=nheads, layers=1, channels=channels)
        elif condition_type == "adaLN":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True)
            )

    def forward_time(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward_cross_attention(self, y, cond):
        B, channel, K, L = y.shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        cond = cond.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        y = self.condition_cross_attention(tgt=y, memory=cond).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3)
        return y

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None):
        condition_type = self.condition_type

        if condition_type == "add":
            x = x + attr_emb
        elif condition_type == "cross_attention":
            x = self.forward_cross_attention(x, attr_emb)
        elif condition_type == "adaLN":
            gama, beta, alpha = self.adaLN_modulation(attr_emb.permute(0, 2, 3, 1)).chunk(3, dim=-1)
            gama = gama.permute(0, 3, 1, 2)
            beta = beta.permute(0, 3, 1, 2)
            alpha = alpha.permute(0, 3, 1, 2)

        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb

        if condition_type == "adaLN":
            y = y * (1 + gama) + beta

        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, None)

        if condition_type == "adaLN":
            y = y.reshape(B, channel, K, L)
            y = alpha * y
            y = y.reshape(B, channel, K * L)

        y = y.reshape(B, channel, K * L)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class VerbalTS(nn.Module):
    """
    Multi-scale patch-based diffusion backbone conditioned on side info and attr embedding.
    """
    def __init__(self, config, inputdim=1):
        super().__init__()
        self.config = config
        self.n_var = config["n_var"]
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        config["side"]["device"] = config["device"]
        self.side_encoder = SideEncoder_Var(configs=config["side"])
        side_dim = self.side_encoder.total_emb_dim

        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList()
        self.side_downsample = nn.ModuleList()
        self.patch_decoder = nn.ModuleList()

        for i in range(self.multipatch_num):
            patch_len = config["base_patch"] * config["L_patch_len"] ** i
            self.ts_downsample.append(
                TsPatchEmbedding(L_patch_len=patch_len, channels=inputdim, d_model=self.channels)
            )
            self.patch_decoder.append(
                PatchDecoder(L_patch_len=patch_len, d_model=self.channels, channels=1)
            )
            self.side_downsample.append(
                SidePatchEmbedding(L_patch_len=patch_len, channels=side_dim, d_model=side_dim)
            )

        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=side_dim,
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
                condition_type=config["condition_type"],
            )
            for _ in range(config["layers"])
        ])

    def forward(self, x_raw, tp, attr_emb_raw, diffusion_step):
        """
        Args:
            x_raw: (B, 1, n_var, L) noisy time series
            tp:    (B, L) time positions
            attr_emb_raw: (B, n_var, 1, channels) conditioning embedding, or None
            diffusion_step: (B,) integer timestep indices
        Returns:
            output: (B, n_var, L) predicted noise
            loss_dict: {}
        """
        B_raw, inputdim, n_var, L = x_raw.shape
        side_emb_raw = self.side_encoder(x_raw, tp)           # (B, side_dim, n_var, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)  # (B, diff_emb_dim)

        x_list, side_list, scale_lengths = [], [], []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
            scale_lengths.append(x.shape[-1])

        x_in = torch.cat(x_list, dim=-1)       # (B, channels, n_var, total_Nl)
        side_in = torch.cat(side_list, dim=-1)

        # Build attr_emb to match x_in shape
        # attr_emb_raw: (B, n_var, n_scale, channels) from TextProjectorMVarMScaleMStep,
        #               or (B, n_var, 1, channels) for simple projectors, or None.
        breakpoint()
        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(x_in)
        else:
            n_scale = attr_emb_raw.shape[2]
            if n_scale == len(scale_lengths):
                # Per-scale expansion: assign each scale slice to its patch length
                mscale_list = []
                for i in range(n_scale):
                    tmp = attr_emb_raw[:, :, i:i + 1, :].expand(-1, -1, scale_lengths[i], -1)
                    mscale_list.append(tmp)
                attr_emb = torch.cat(mscale_list, dim=2)   # (B, n_var, total_Nl, channels)
            else:
                # Collapse the scale dim (mean) and broadcast uniformly
                collapsed = attr_emb_raw.mean(dim=2, keepdim=True)            # (B, n_var, 1, channels)
                attr_emb = collapsed.expand(-1, -1, x_in.shape[-1], -1)       # (B, n_var, total_Nl, channels)
            attr_emb = attr_emb.permute(0, 3, 1, 2)                           # (B, channels, n_var, total_Nl)


        B, _, Nk, Nl = x_in.shape
        _x_in = x_in
        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(x_in + _x_in, side_in, attr_emb, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        start_id = 0
        all_out = []
        for i, x_scale in enumerate(x_list):
            x_out = x[:, :, :, start_id:start_id + x_scale.shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_scale.shape[-1]

        all_out = torch.cat(all_out, dim=1)                              # (B, multipatch_num, n_var, L)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1))    # (B, n_var, L, 1)
        all_out = all_out.reshape(B_raw, n_var, L)
        return all_out, {}
