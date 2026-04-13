import logging
import os

import torch

from .simple_vae import SimpleVAE
from ..generative_handler import generativeHandler


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def build_model(self):
        dynamic_size = getattr(self.args, "dynamic_size", 32)
        if isinstance(dynamic_size, (list, tuple)):
            model_dynamic_size = dynamic_size
            placeholder_dim = dynamic_size[0]
        else:
            model_dynamic_size = dynamic_size
            placeholder_dim = dynamic_size

        self.model = SimpleVAE(
            input_dim=placeholder_dim,
            output_dim=placeholder_dim,
            hidden_size=getattr(self.args, "hidden_size", 128),
            num_layers=getattr(self.args, "num_layers", 4),
            num_heads=getattr(self.args, "num_heads", 4),
            latent_dim=getattr(self.args, "latent_dim", 64),
            beta=getattr(self.args, "beta", 0.001),
            dynamic_size=model_dynamic_size,
        )
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
        return self.model

    def train_iter(self, train_dataloader, logger):
        epoch_losses = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
        }
        num_batches = 0

        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()

            x_ts = data[0].to(self.args.device).to(torch.float32)
            out = self.model(x_ts.transpose(1, 2))
            loss_dict = self.model.loss_function(
                x_ts.transpose(1, 2),
                out["recon"],
                out["mu"],
                out["logvar"],
            )

            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].detach().item()
            num_batches += 1

        for key in epoch_losses:
            logger.log(f"train/{key}", epoch_losses[key] / num_batches)

    def sample(self, n_samples, class_label, class_metadata):
        generated = []
        latent_dim = getattr(self.args, "latent_dim", 64)

        for sample_size in [
            min(self.args.batch_size, n_samples - i)
            for i in range(0, n_samples, self.args.batch_size)
        ]:
            z = torch.randn(sample_size, latent_dim, device=self.args.device)
            x_ts = self._model.decode(z, out_channels=class_metadata["channels"])
            generated.append(x_ts.transpose(1, 2))

        return torch.cat(generated, dim=0)

    def save_model(self, ckpt_dir):
        state = {"model": self._model.state_dict()}
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(
                f"No checkpoint found at {ckpt_dir}. Returned the same state as input"
            )
        else:
            loaded_state = torch.load(ckpt_dir, map_location=device)
            if "model" in loaded_state:
                self.model.load_state_dict(loaded_state["model"], strict=False)
            else:
                self.model.load_state_dict(loaded_state, strict=False)
            logging.info("Successfully loaded previous state")
