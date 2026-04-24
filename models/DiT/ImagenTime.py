from contextlib import contextmanager

import torch
import torch.nn as nn

from .ema import LitEma
from .img_transformations import DelayEmbedder, STFTEmbedder
from .networks import EDMPrecond


class ImagenTime(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self.sigma_min = 0.002
        self.sigma_max = 80
        self.rho = 7
        self.T = args.diffusion_steps

        self.device = device
        self.cond_dim = int(getattr(args, "condition_dim", getattr(args, "context_dim", args.n_classes)))
        self.label_dropout = float(getattr(args, "label_dropout", 0.0))
        self.guidance_scale = float(getattr(args, "guidance_scale", 1.0))

        self.net = EDMPrecond(
            args.img_resolution,
            args.input_channels,
            text_condition_dim=self.cond_dim,
            patch_size=int(getattr(args, "dit_patch_size", 4)),
            hidden_size=int(getattr(args, "dit_hidden_size", getattr(args, "unet_channels", 256))),
            depth=int(getattr(args, "dit_depth", 6)),
            num_heads=int(getattr(args, "dit_num_heads", 4)),
            mlp_ratio=float(getattr(args, "dit_mlp_ratio", 4.0)),
            label_dropout=self.label_dropout,
            num_register_tokens=int(getattr(args, "dit_num_register_tokens", 8)),
            use_style_embed=bool(getattr(args, "dit_use_style_embed", False)),
            use_text_encoder=bool(getattr(args, "dit_use_text_encoder", True)),
            text_encoder_model_name=str(
                getattr(args, "text_encoder_model_name", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            text_max_length=int(getattr(args, "text_max_length", 128)),
            use_fp16=bool(getattr(args, "dit_use_fp16", False)),
        )

        if not args.use_stft:
            self.delay = args.delay
            self.embedding = args.embedding
            self.seq_len = args.seq_len
            self.ts_img = DelayEmbedder(self.device, args.seq_len, args.delay, args.embedding)
        else:
            self.ts_img = STFTEmbedder(self.device, args.seq_len, args.n_fft, args.hop_length)

        if args.ema:
            self.use_ema = True
            self.model_ema = LitEma(self.net, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)
        else:
            self.use_ema = False

    def ts_to_img(self, signal, pad_val=None):
        return self.ts_img.ts_to_img(signal, True, pad_val) if pad_val else self.ts_img.ts_to_img(signal)

    def img_to_ts(self, img):
        return self.ts_img.img_to_ts(img)

    def forward(self, x, text_condition=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        D_yn = self.net(x + n, sigma, text_condition)
        return D_yn, weight

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())

    def on_train_batch_end(self, *args):
        if self.use_ema:
            self.model_ema(self.net)
