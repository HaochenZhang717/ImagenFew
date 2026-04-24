import torch

from .dit_model import DriftDiT


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        text_condition_dim=0,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.text_condition_dim = text_condition_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = DriftDiT(
            img_size=img_resolution,
            in_channels=img_channels,
            condition_dim=text_condition_dim,
            **model_kwargs,
        )

    def forward(self, x, sigma, text_condition=None, force_fp32=False, **model_kwargs):
        model_kwargs.pop("augment_labels", None)
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if text_condition is not None and not isinstance(text_condition, (tuple, list, str)):
            text_condition = text_condition.to(torch.float32)
            if text_condition.ndim == 2:
                if text_condition.shape[-1] != self.text_condition_dim:
                    raise ValueError(
                        f"Condition dim mismatch: expected {self.text_condition_dim}, got {text_condition.shape[-1]}"
                    )
            elif text_condition.ndim == 3:
                if text_condition.shape[-1] != self.text_condition_dim:
                    raise ValueError(
                        f"Condition token dim mismatch: expected {self.text_condition_dim}, got {text_condition.shape[-1]}"
                    )
            else:
                raise ValueError(
                    f"Expected condition tensor shape (B, D) or (B, N_ctx, D), got {tuple(text_condition.shape)}"
                )
        elif isinstance(text_condition, tuple):
            if len(text_condition) != 2:
                raise ValueError("Expected text_condition tuple as (input_ids, attention_mask).")
            input_ids, attention_mask = text_condition
            if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
                raise TypeError("text_condition tensors must be torch.Tensor.")
            if input_ids.ndim != 2 or attention_mask.ndim != 2:
                raise ValueError(
                    f"Expected input_ids/attention_mask with shape (B, N_ctx), got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
                )
            if input_ids.shape != attention_mask.shape:
                raise ValueError(
                    f"Token label shape mismatch: input_ids {tuple(input_ids.shape)} vs attention_mask {tuple(attention_mask.shape)}"
                )
        elif isinstance(text_condition, list):
            if len(text_condition) > 0 and not isinstance(text_condition[0], str):
                raise TypeError("List text_condition must be a list of strings.")

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model((c_in * x).to(dtype), text_condition=text_condition, alpha=c_noise.flatten(), **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
