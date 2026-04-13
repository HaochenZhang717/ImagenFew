import torch
import torch.nn as nn
from contextlib import contextmanager

from .networks import VerbalTS
from .ddpm import DDPMSampler
from .ddim import DDIMSampler
from ..ImagenTime.ema import LitEma

from models.MultiScaleVAE.multiscale_vae import DualVAE
import math
import torch.nn.functional as F


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
        # fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights
        this_bias = self.bias
        in_features = x.shape[-1]

        if out_features is None:
            out_features = self.out_features

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features-self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            # if self.fixed_in != 0:
            #     fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
            #         out_features, self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).squeeze(0)

        return F.linear(x, dynamic_weights, this_bias)
        # return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)



class TextProjectorMVarMScaleMStep(nn.Module):
    def __init__(self, n_scale, n_steps, n_stages, dim_in=128, dim_out=128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        # self.n_var = n_var
        self.seg_size = n_steps // n_stages + 1
        self.n_scale = n_scale
        self.n_stages = n_stages
        # self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        # self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))

        self.var_embed_linear = DynamicLinear(dim_in, 32*dim_in)
        self.scale_embed_linear = nn.Linear(dim_in, n_scale*dim_in)


        self.step_emb_linear = nn.Linear(dim_in, n_stages*dim_in)
        # self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))

        var_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.var_cross_attn = nn.TransformerDecoder(var_cross_attn_layer, num_layers=2)
        scale_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.scale_cross_attn = nn.TransformerDecoder(scale_cross_attn_layer, num_layers=2)
        step_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.step_cross_attn = nn.TransformerDecoder(step_cross_attn_layer, num_layers=2)
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, n_var, attr, diffusion_step, attention_mask=None):
        # attr.shape == (batch_size, seq_len, dim)
        B = attr.shape[0]

        memory_key_padding_mask = None
        if attention_mask is not None:
            memory_key_padding_mask = (attention_mask == 0)  # True means masked
            attr = attr * attention_mask.unsqueeze(-1)

        attr_global = attr.mean(dim=1)

        var_emb = self.var_embed_linear(attr_global, out_features=n_var*self.dim_in).reshape(-1, n_var, self.dim_in)
        # var_emb = self.var_emb.expand([B,-1,-1]) # (B, n_var, dim_in)
        mvar_attr = self.var_cross_attn(
            tgt=var_emb, memory=attr,
            memory_key_padding_mask=memory_key_padding_mask
        ) # (B, n_var, dim_in)
        mvar_attr = mvar_attr[:,:,None,:] # (B, n_var, 1, dim_in)

        # scale_emb = self.scale_emb.expand([B,-1,-1])
        scale_emb = self.scale_embed_linear(attr_global).reshape(-1, self.n_scale, self.dim_in)
        mscale_attr = self.scale_cross_attn(
            tgt=scale_emb, memory=attr,
            memory_key_padding_mask=memory_key_padding_mask
        )
        mscale_attr = mscale_attr[:,None,:,:].expand([-1,n_var,-1,-1]) # (B, 1, n_scale, dim_in)

        # step_emb = self.step_emb.expand([B,-1,-1])
        # self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))
        step_emb = self.step_emb_linear(attr_global).reshape(-1, self.n_stages, self.dim_in)
        mstep_attr = self.step_cross_attn(
            tgt=step_emb, memory=attr,
            memory_key_padding_mask=memory_key_padding_mask
        )
        indices = diffusion_step // self.seg_size
        indices = indices[:,None,None]
        mstep_attr = torch.gather(mstep_attr, dim=1, index=indices.expand([-1, -1, mstep_attr.shape[-1]]))
        mstep_attr = mstep_attr[:,None,:,:].expand([-1, n_var, -1, -1])
        mix_attr = mvar_attr + mscale_attr + mstep_attr
        out = self.proj_out(mix_attr)
        return out



class SelfConditionalGenerator(nn.Module):
    """
    Class-conditional diffusion model for time series generation.

    The model conditions on an integer class label (dataset index in the
    multi-dataset setting) via a learned embedding that is injected into every
    residual block of the VerbalTS backbone.
    """
    def __init__(self, configs):
        super().__init__()
        self.device = configs["device"]
        self.ema_decay = configs.get("ema_decay", 0.9999)
        self.ema_warmup = configs.get("ema_warmup", 0)
        self.use_ema = bool(configs.get("ema", False))
        diff_cfg = configs["diffusion"]

        self.num_steps = diff_cfg["num_steps"]
        self.n_var = diff_cfg["n_var"]
        self.seq_len = configs["seq_len"]

        # Diffusion backbone
        diff_cfg["device"] = self.device
        self.diff_model = VerbalTS(diff_cfg, inputdim=1).to(self.device)

        self.multi_scale_vae = DualVAE(**configs["multi_scale_vae"])

        if configs["pretrained_vae_weights"]:
            pretrained_vae_weights = torch.load(configs["pretrained_vae_weights"], map_location="cpu")
            self.multi_scale_vae.load_state_dict(pretrained_vae_weights['model'])
            print("loaded pretrained VAE weights from {}".format(configs["pretrained_vae_weights"]))

        for param in self.multi_scale_vae.parameters():
            param.requires_grad = False
        self.multi_scale_vae.eval()

        self.cond_projector = TextProjectorMVarMScaleMStep(
            n_scale=diff_cfg["n_scales"],
            n_stages=diff_cfg["n_stages"],
            n_steps=diff_cfg["num_steps"],
            dim_in=configs["cond_dim_in"],
            dim_out=configs["cond_dim_out"]
        )
        # Noise schedulers
        self.ddpm = DDPMSampler(
            self.num_steps,
            diff_cfg["beta_start"],
            diff_cfg["beta_end"],
            diff_cfg["schedule"],
            self.device,
        )
        self.ddim = DDIMSampler(
            self.num_steps,
            diff_cfg["beta_start"],
            diff_cfg["beta_end"],
            diff_cfg["schedule"],
            self.device,
        )

        if self.use_ema:
            self.model_ema = LitEma(
                self.diff_model,
                decay=self.ema_decay,
                use_num_upates=True,
                warmup=self.ema_warmup,
            )

    def reset_ema(self):
        if self.use_ema:
            self.model_ema = LitEma(
                self.diff_model,
                decay=self.ema_decay,
                use_num_upates=True,
                warmup=self.ema_warmup,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_tp(self, B, L):
        """Create time-position array (B, L) on the model device."""
        return torch.arange(L, device=self.device).unsqueeze(0).expand(B, -1)

    def _noise_estimation_loss(self, x, tp, attr_emb, t):
        """
        Compute the noise-estimation (MSE) loss at timestep t.
        Args:
            x:        (B, n_var, L) clean time series
            tp:       (B, L) time positions
            attr_emb: (B, n_var, 1, channels) or None
            t:        (B,) integer timestep indices
        Returns:
            loss_dict with keys 'noise_loss' and 'all'
        """
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)
        noisy_x_in = noisy_x.unsqueeze(1)                 # (B, 1, n_var, L)
        pred_noise, extra = self.diff_model(noisy_x_in, tp, attr_emb, t)

        residual = noise - pred_noise
        loss_dict = {**extra, "noise_loss": (residual ** 2).mean()}
        loss_dict["all"] = sum(loss_dict.values())
        return loss_dict

    def _predict_noise(self, xt, tp, attr_emb, t):
        noisy_x_in = xt.unsqueeze(1)
        pred_noise, extra = self.diff_model(noisy_x_in, tp, attr_emb, t)
        return pred_noise, extra

    @torch.no_grad()
    def encode_posterior_context(self, x):
        """
        Encode clean time-series into the frozen VAE posterior context used by the conditioner.
        Args:
            x: (B, seq_len, n_var)
        Returns:
            context: (B, latent_seq_len, cond_dim_in)
        """
        x_latent = x.permute(0, 2, 1)
        context_trend, context_coarse_seasonal, context_seasonal = self.multi_scale_vae.ts_to_z(
            x_latent.permute(0, 2, 1), sample=True
        ) # here I use sample

        context = torch.cat([context_trend, context_coarse_seasonal, context_seasonal], dim=-1)
        context = context.permute(0, 2, 1)
        return context

    def forward_with_context(self, x, context, is_train=True):
        B, _, n_var = x.shape
        x = x.permute(0, 2, 1)  # (B, C, L)
        tp = self._make_tp(B, x.shape[-1])

        if is_train:
            t = torch.randint(0, self.num_steps, (B,), device=self.device)
            attr_emb = self.cond_projector(n_var, context, t)
            return self._noise_estimation_loss(x, tp, attr_emb, t)

        loss_dict = {}
        for step in range(self.num_steps):
            t = (torch.ones(B, device=self.device) * step).long()
            attr_emb = self.cond_projector(n_var, context, t)
            tmp = self._noise_estimation_loss(x, tp, attr_emb, t)
            for k, v in tmp.items():
                loss_dict[k] = loss_dict.get(k, 0) + v
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.num_steps
        return loss_dict

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, x, is_train=True):
        context = self.encode_posterior_context(x)
        return self.forward_with_context(x, context, is_train=is_train)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.diff_model.parameters())
            self.model_ema.copy_to(self.diff_model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.diff_model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.diff_model)

    @torch.no_grad()
    def generate_from_context(self, context, seq_len, n_var, sampler="ddim"):
        B = context.shape[0]
        tp = self._make_tp(B, seq_len)
        x = torch.randn(B, n_var, seq_len, device=self.device)
        for t_idx in range(self.num_steps - 1, -1, -1):
            t = (torch.ones(B, device=self.device) * t_idx).long()
            attr_emb = self.cond_projector(n_var, context, t)
            pred_noise, _ = self._predict_noise(x, tp, attr_emb, t)
            noise = torch.randn_like(x)
            x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)

        # (B, n_var, L) -> (B, L, n_var)
        return x.permute(0, 2, 1).cpu()

    @torch.no_grad()
    def generate(self, n_samples, test_batch, seq_len, n_var, sampler="ddim"):
        context_trend, context_coarse_seasonal, context_seasonal = self.multi_scale_vae.ts_to_z(test_batch, sample=False)
        context = torch.cat([context_trend, context_coarse_seasonal, context_seasonal], dim=-1) # (B, C, L)
        context = context.permute(0, 2, 1)
        return self.generate_from_context(context, seq_len, n_var, sampler=sampler)
