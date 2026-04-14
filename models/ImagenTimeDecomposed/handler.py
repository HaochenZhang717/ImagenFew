import logging
import os

import torch
import torch.nn as nn

from ..decomposition import series_decomp
from ..generative_handler import generativeHandler
from ..ImagenTime.ImagenTime import ImagenTime
from ..ImagenTime.sampler import DiffusionProcess


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        self.decomp = series_decomp(getattr(args, "decomp_kernel_size", 25))
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.resume_epoch = 0
        self.best_score = float("inf")

    def build_model(self):
        self.model = ImagenTime(self.args, self.args.device)
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
        return self.model

    def _decompose(self, x_ts):
        seasonal, trend = self.decomp(x_ts)
        return seasonal.to(x_ts.dtype), trend.to(x_ts.dtype)

    def train_iter(self, train_dataloader, logger):
        epoch = getattr(self, "epoch", None)
        train_loss = 0.0
        num_batches = 0
        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()
            x_ts = data[0].to(self.args.device, dtype=torch.float32)
            _, trend = self._decompose(x_ts)

            x_img = self.model.ts_to_img(trend)
            class_indices = data[1].to(self.args.device)
            labels = nn.functional.one_hot(class_indices, num_classes=self.args.n_classes)
            output, weight = self.model(x_img, labels=labels)
            time_loss = (output - x_img).square()
            loss = (weight * time_loss).mean()

            train_loss += loss.item()
            num_batches += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.model.on_train_batch_end()

        logger.log("train/karras loss", train_loss / max(num_batches, 1), step=epoch)

    def sample(self, n_samples, class_label, class_metadata, test_data=None):
        generated_set = []
        with self._model.ema_scope():
            process = DiffusionProcess(
                self.args,
                self._model.net,
                (self.args.input_channels, self.args.img_resolution, self.args.img_resolution),
            )
            for sample_size in [
                min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)
            ]:
                class_labels = torch.full((sample_size,), class_label, device=self.args.device)
                oh_class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.args.n_classes)
                x_img_sampled = process.sampling(sample_size, class_labels=oh_class_labels)
                x_ts = self._model.img_to_ts(x_img_sampled)[:, :, : class_metadata["channels"]]
                generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)

    def sample_variants(self, n_samples, class_label, class_metadata, test_data=None):
        return {"prior": self.sample(n_samples, class_label, class_metadata, test_data)}

    def save_model(self, ckpt_dir):
        state = {
            "model": self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": int(getattr(self, "epoch", 0)),
            "global_step": self.global_step,
            "best_score": getattr(self, "best_score", float("inf")),
        }
        if self.args.ema is not None:
            state["ema_model"] = self._model.model_ema.state_dict()
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
            return

        loaded_state = torch.load(ckpt_dir, map_location=device)
        model_state = loaded_state["model"] if isinstance(loaded_state, dict) and "model" in loaded_state else loaded_state
        self.model.load_state_dict(model_state, strict=False)
        if isinstance(loaded_state, dict) and "ema_model" in loaded_state and self.args.ema is not None:
            self.model.model_ema.load_state_dict(loaded_state["ema_model"], strict=False)
        self.resume_epoch = int(loaded_state.get("epoch", 0)) if isinstance(loaded_state, dict) else 0
        self.global_step = int(loaded_state.get("global_step", 0)) if isinstance(loaded_state, dict) else 0
        self.best_score = float(loaded_state.get("best_score", float("inf"))) if isinstance(loaded_state, dict) else float("inf")
        logging.info("Successfully loaded previous state")
