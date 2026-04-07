from .multiscale_vae import DualVAE
from ..generative_handler import generativeHandler

import os
import torch
import logging


class Handler(generativeHandler):

    def __init__(self, args, rank=None):
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def build_model(self):
        # in_channels = getattr(self.args, 'input_channels', 6)
        # z_channels = getattr(self.args, 'z_channels', 32)
        # ch = getattr(self.args, 'unet_channels', 128)
        # ch_mult = tuple(getattr(self.args, 'ch_mult', [1, 1, 2]))
        # dynamic_size = getattr(self.args, 'dynamic_size', 128)
        # dropout = getattr(self.args, 'dropout', 0.0)

        z_channels = getattr(self.args, 'z_channels')
        ch = getattr(self.args, 'unet_channels')
        ch_mult = tuple(getattr(self.args, 'ch_mult'))
        dynamic_size = getattr(self.args, 'dynamic_size')
        dropout = getattr(self.args, 'dropout')


        self.model = DualVAE(
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            dynamic_size=dynamic_size,
            dropout=dropout,
        )
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
        return self.model

    def train_iter(self, train_dataloader, logger):
        w_kl = getattr(self.args, 'w_kl', 0.007)
        epoch_losses = {
            'recon_loss': 0., 'recon_loss_low_freq': 0., 'recon_loss_mid_freq': 0., 'recon_loss_high_freq': 0.,
            'kl_loss': 0., 'kl_loss_low_freq': 0., 'kl_loss_mid_freq': 0., 'kl_loss_high_freq': 0.,
            'loss': 0.,
        }
        num_batches = 0

        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()

            x_ts = data[0].to(self.args.device).to(torch.float32)
            out = self.model(x_ts)

            recon_loss = out['recon_loss_overall'] + out['recon_loss_low_freq'] + out['recon_loss_mid_freq'] + out['recon_loss_high_freq']
            kl_loss = (
                out['kl_loss_low_freq']
                + out['kl_loss_mid_freq']
                + out['kl_loss_high_freq']
            )
            loss = recon_loss + w_kl * kl_loss

            epoch_losses['recon_loss'] += recon_loss.detach().item()
            epoch_losses['recon_loss_low_freq'] += out['recon_loss_low_freq'].detach().item()
            epoch_losses['recon_loss_mid_freq'] += out['recon_loss_mid_freq'].detach().item()
            epoch_losses['recon_loss_high_freq'] += out['recon_loss_high_freq'].detach().item()
            epoch_losses['kl_loss'] += kl_loss.detach().item()
            epoch_losses['kl_loss_low_freq'] += out['kl_loss_low_freq'].detach().item()
            epoch_losses['kl_loss_mid_freq'] += out['kl_loss_mid_freq'].detach().item()
            epoch_losses['kl_loss_high_freq'] += out['kl_loss_high_freq'].detach().item()
            epoch_losses['loss'] += loss.detach().item()
            num_batches += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        for key in epoch_losses:
            logger.log(f'train/{key}', epoch_losses[key] / num_batches)

    def sample(self, n_samples, class_label, class_metadata):
        generated_set = []
        z_channels = self._model.latent_channels
        # Compute latent sequence length from encoder downsample ratio
        downsample_ratio = 2 ** (len(getattr(self.args, 'ch_mult', [1, 1, 2])) - 1)
        latent_len = self.args.seq_len // downsample_ratio

        for sample_size in [
            min(self.args.batch_size, n_samples - i)
            for i in range(0, n_samples, self.args.batch_size)
        ]:
            z_low = torch.randn(sample_size, z_channels, latent_len, device=self.args.device)
            z_mid = torch.randn(sample_size, z_channels, latent_len, device=self.args.device)
            z_high = torch.randn(sample_size, z_channels, latent_len, device=self.args.device)
            x_ts = self._model.z_to_ts((z_low, z_mid, z_high))
            x_ts = x_ts[:, :, :class_metadata['channels']]
            generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)

    def save_model(self, ckpt_dir):
        state = {'model': self._model.state_dict()}
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. "
                            f"Returned the same state as input")
        else:
            loaded_state = torch.load(ckpt_dir, map_location=device)
            if 'model' in loaded_state:
                self.model.load_state_dict(loaded_state['model'], strict=False)
            else:
                self.model.load_state_dict(loaded_state, strict=False)
            logging.info('Successfully loaded previous state')