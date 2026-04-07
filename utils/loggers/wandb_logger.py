from .base_logger import BaseLogger
from typing import Dict, Any, List
import numpy as np


class WandbLogger(BaseLogger):

    def __init__(self, project=None, name=None, tags=None, config=None, *args, **kwargs):
        super(WandbLogger, self).__init__(*args, **kwargs)
        import wandb
        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=name,
            tags=tags,
            config=config,
        )

    def stop(self):
        self.run.finish()

    def log(self, name: str, data: Any, step=None):
        self.wandb.log({name: data}, step=step)

    def _log_fig(self, name: str, fig: Any):
        if isinstance(fig, np.ndarray):
            self.wandb.log({name: self.wandb.Image(fig)})
        else:
            self.wandb.log({name: fig})

    def log_hparams(self, params: Dict[str, Any]):
        self.wandb.config.update(params)

    def log_params(self, params: Dict[str, Any]):
        self.wandb.config.update(params)

    def add_tags(self, tags: List[str]):
        self.run.tags = self.run.tags + tuple(tags)

    def log_name_params(self, name: str, params: Any):
        # Write to run.summary instead of wandb.log so we don't advance
        # wandb's internal step counter (which would break explicit step= logging).
        self.run.summary[name] = params