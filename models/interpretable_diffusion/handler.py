import os
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from omegaconf import OmegaConf
from models.generative_handler import generativeHandler
from models.interpretable_diffusion.ema import LitEma
from utils.io_utils import instantiate_from_config


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        self.config = OmegaConf.to_object(OmegaConf.load(args.config))
        super().__init__(args, rank)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['solver'].get('base_lr', 1e-4),
            betas=(0.9, 0.96),
        )
        self.ema = LitEma(self.model, decay=0.9999, use_num_upates=True, warmup=args.ema_warmup)

        self.config['solver']['scheduler']['params']['optimizer'] = self.optimizer
        self.scheduler = instantiate_from_config({
            **self.config['solver']['scheduler']
        })

        self.step = 0

    def build_model(self):
        return instantiate_from_config(self.config['model'])

    def train_iter(self, train_dataloader, logger):
        epoch = getattr(self, "epoch", None)
        train_loss = 0
        num_batches = 0
        for _, data in enumerate(train_dataloader, 1):
            x = data[0].to(self.device).float()

            loss = self.model(x, target=x)
            loss.backward()

            train_loss += loss.item()
            num_batches += 1

            clip_grad_norm_(self._model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step(loss.item())
            self.optimizer.zero_grad()
            self.step += 1
            self.ema(self.model)

        logger.log(f'train/loss', train_loss / num_batches, step=epoch)
        logger.log(f'train/epoch', epoch, step=epoch)

    def sample(self, n_samples, class_label=None, class_metadata=None, test_data=None):
        generated = []
        sample_batch_size = min(getattr(self.args, "batch_size", n_samples), n_samples)

        with torch.no_grad():
            with self.ema_scope():
                for start in range(0, n_samples, sample_batch_size):
                    current_batch = min(sample_batch_size, n_samples - start)
                    generated.append(self.model.generate_mts(batch_size=current_batch))

        return torch.cat(generated, dim=0)

    def ema_scope(self):
        class EMAScope:
            def __init__(self, model, ema):
                self.model = model
                self.ema = ema

            def __enter__(self):
                self.ema.store(self.model.parameters())
                self.ema.copy_to(self.model)

            def __exit__(self, exc_type, exc_value, traceback):
                self.ema.restore(self.model.parameters())

        return EMAScope(self.model, self.ema)

    def save_model(self, ckpt_dir):
        torch.save(self._model.state_dict(), ckpt_dir)
