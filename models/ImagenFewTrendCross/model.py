import torch

from ..ImagenFewCrossAttention.ImagenFew import ImagenFew
from ..MultiScaleVAE.modules import DynamicLinear


class TrendConditionedImagenFew(ImagenFew):
    def __init__(self, args, device):
        super().__init__(args, device)
        if getattr(args, "context_dim", None) is None:
            raise ValueError("context_dim must be provided for TrendConditionedImagenFew.")

        dynamic_size = getattr(args, "dynamic_size", [128, 128])
        if isinstance(dynamic_size, int):
            max_input_dim = dynamic_size
        else:
            max_input_dim = int(dynamic_size[0])

        self.trend_context_projection = DynamicLinear(
            in_features=max_input_dim,
            out_features=int(args.context_dim),
            fixed_in=0,
        )

    def prepare_trend_context(self, trend_context):
        trend_context = trend_context.to(device=self.device, dtype=torch.float32)
        return self.trend_context_projection(trend_context, out_features=self.context_dim)

    def forward(self, x, mask, labels=None, trend_context=None, context=None, augment_pipe=None):
        if context is None and trend_context is not None:
            context = self.prepare_trend_context(trend_context)
        return super().forward(x, mask, labels=labels, context=context, augment_pipe=augment_pipe)
