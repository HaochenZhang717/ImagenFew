import logging
import os
import random

import numpy as np
import torch

from ..decomposition import series_decomp
from ..generative_handler import generativeHandler
from ..ImagenFewCrossAttention.sampler import DiffusionProcess
from ..ImagenTime.ImagenTime import ImagenTime
from .model import TrendConditionedImagenFew


class Handler(generativeHandler):
    def __init__(self, args, rank=None):
        if getattr(args, "context_dim", None) is None:
            args.context_dim = int(getattr(args, "trend_context_dim", 128))
        if getattr(args, "trend_model_ckpt", None) is None:
            env_trend_ckpt = os.environ.get("TREND_MODEL_CKPT")
            if env_trend_ckpt:
                args.trend_model_ckpt = env_trend_ckpt
        self.decomp = series_decomp(getattr(args, "decomp_kernel_size", 25))
        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.resume_epoch = 0
        self.best_score = float("inf")
        self._trend_model = None

        if not self.args.finetune:
            self._load_optimizer_state(self.args.model_ckpt, self.args.device)

    def build_model(self):
        self.model = TrendConditionedImagenFew(self.args, self.args.device)
        if self.args.model_ckpt is not None:
            self._load_model(self.args.model_ckpt, self.args.device)
            if self.args.finetune and self.args.ema:
                self.model.setup_finetune(self.args)
        return self.model

    def _decompose(self, x_ts):
        seasonal, trend = self.decomp(x_ts)
        return seasonal.to(x_ts.dtype), trend.to(x_ts.dtype)

    def train_iter(self, train_dataloader, logger):
        avg_loss = 0.0
        epoch = getattr(self, "epoch", None)

        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()
            x_ts = data[0].to(self.args.device, dtype=torch.float32)
            seasonal, trend = self._decompose(x_ts)
            x_img = self.model.ts_to_img(seasonal)
            x_ts_mask = torch.zeros_like(seasonal)
            x_img_mask = self.model.ts_to_img(x_ts_mask, pad_val=1)

            output, time_weight = self.model(x_img, x_img_mask, trend_context=trend)
            time_loss = (output - x_img).square()
            loss = ((time_weight * time_loss) * (1 - x_img_mask)).mean()
            avg_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self._model.on_train_batch_end()

        avg_loss /= max(len(train_dataloader), 1)
        logger.log("train/karras loss", avg_loss, step=epoch)

    def sample(self, n_samples, class_label, class_metadata, test_data):
        sample_source = self._get_sample_source()
        if sample_source == "prior":
            return self._sample_with_prior(n_samples, class_label, class_metadata)
        if sample_source == "posterior":
            return self._sample_with_posterior(n_samples, class_metadata, test_data)
        if sample_source == "both":
            return self._sample_with_prior(n_samples, class_label, class_metadata)
        raise ValueError(f"Unsupported sample_source: {sample_source}")

    def sample_variants(self, n_samples, class_label, class_metadata, test_data):
        sample_source = self._get_sample_source()
        if sample_source == "both":
            return {
                "prior": self._sample_with_prior(n_samples, class_label, class_metadata),
                "posterior": self._sample_with_posterior(n_samples, class_metadata, test_data),
            }
        return {
            sample_source: self.sample(n_samples, class_label, class_metadata, test_data),
        }

    def _sample_with_posterior(self, n_samples, class_metadata, test_data):
        if test_data is None:
            raise ValueError("test_data must be provided for posterior sampling.")
        generated_set = []
        seasonal_all, trend_all = self._decompose(test_data.to(self.args.device, dtype=torch.float32))
        with self._model.ema_scope():
            process = DiffusionProcess(
                self.args,
                self._model.net,
                (class_metadata["channels"], self.args.img_resolution, self.args.img_resolution),
            )
            for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
                indices = torch.randperm(trend_all.shape[0], device=trend_all.device)[:sample_size]
                trend_batch = trend_all[indices]
                seasonal_batch = self._sample_seasonal_from_trend_batch(process, trend_batch, class_metadata["channels"])
                generated_set.append(trend_batch + seasonal_batch)
        return torch.concat(generated_set, dim=0)

    def _sample_with_prior(self, n_samples, class_label, class_metadata):
        generated_set = []
        with self._model.ema_scope():
            process = DiffusionProcess(
                self.args,
                self._model.net,
                (class_metadata["channels"], self.args.img_resolution, self.args.img_resolution),
            )
            for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
                trend_batch = self._sample_trend_prior(sample_size, class_label, class_metadata)
                seasonal_batch = self._sample_seasonal_from_trend_batch(process, trend_batch, class_metadata["channels"])
                generated_set.append(trend_batch + seasonal_batch)
        return torch.concat(generated_set, dim=0)

    def _sample_seasonal_from_trend_batch(self, process, trend_batch, channels):
        sample_size = trend_batch.shape[0]
        x_img = torch.zeros(sample_size, channels, self.args.img_resolution, self.args.img_resolution, device=self.args.device)
        x_ts_mask = self._model.ts_to_img(
            torch.zeros(sample_size, self.args.seq_len, channels, device=self.args.device),
            pad_val=1,
        )
        context = self._model.prepare_trend_context(trend_batch)
        x_img_sampled = process.interpolate(x_img, x_ts_mask, context=context)
        return self._model.img_to_ts(x_img_sampled)[:, :, :channels]

    def _init_trend_model(self):
        if self._trend_model is not None:
            return
        trend_model_ckpt = getattr(self.args, "trend_model_ckpt", None)
        if not trend_model_ckpt:
            raise ValueError("trend_model_ckpt must be provided for prior seasonal sampling.")
        if not os.path.exists(trend_model_ckpt):
            raise FileNotFoundError(f"Trend checkpoint not found: {trend_model_ckpt}")

        self._trend_model = ImagenTime(self.args, self.args.device).to(self.args.device)
        state = torch.load(trend_model_ckpt, map_location=self.args.device)
        use_ema = bool(getattr(self.args, "trend_use_ema", True))
        if use_ema and "ema_model" in state:
            self._trend_model.model_ema.load_state_dict(state["ema_model"], strict=False)
            self._trend_model.use_ema = True
        self._trend_model.load_state_dict(state["model"], strict=False)
        self._trend_model.eval()

    def _sample_trend_prior(self, n_samples, class_label, class_metadata):
        self._init_trend_model()
        generated_set = []
        trend_ctx = self._trend_model.ema_scope() if getattr(self.args, "trend_use_ema", True) else torch.no_grad()
        with trend_ctx, torch.no_grad():
            process = DiffusionProcess(
                self.args,
                self._trend_model.net,
                (self.args.input_channels, self.args.img_resolution, self.args.img_resolution),
            )
            class_labels = torch.full((n_samples,), class_label, device=self.args.device)
            oh_class_labels = torch.nn.functional.one_hot(class_labels, num_classes=self.args.n_classes)
            start = 0
            while start < n_samples:
                end = min(start + self.args.batch_size, n_samples)
                x_img_sampled = process.sampling(end - start, class_labels=oh_class_labels[start:end])
                x_ts = self._trend_model.img_to_ts(x_img_sampled)[:, :, : class_metadata["channels"]]
                generated_set.append(x_ts)
                start = end
        return torch.concat(generated_set, dim=0)

    def save_model(self, ckpt_dir):
        epoch = int(getattr(self, "epoch", 0))
        state = {
            "model": self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "best_score": getattr(self, "best_score", float("inf")),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        if self.args.ema is not None:
            state["ema_model"] = self._model.model_ema.state_dict()
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(state, ckpt_dir)

    def _load_model(self, ckpt_dir, device):
        if not os.path.exists(ckpt_dir):
            os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
            logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
            return

        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        self.model.load_state_dict(loaded_state["model"], strict=False)
        if "ema_model" in loaded_state and self.args.ema is not None:
            self.model.model_ema.load_state_dict(loaded_state["ema_model"], strict=False)

        if not self.args.finetune:
            self.resume_epoch = int(loaded_state.get("epoch", 0))
            self.global_step = int(loaded_state.get("global_step", 0))
            self.best_score = float(loaded_state.get("best_score", float("inf")))

        if self.args.finetune and self.args.ema:
            self.model.setup_finetune(self.args)
        logging.info("Successfully loaded previous state")

    def _load_optimizer_state(self, ckpt_dir, device):
        if ckpt_dir is None or not os.path.exists(ckpt_dir):
            return
        loaded_state = torch.load(ckpt_dir, map_location=device)
        if not isinstance(loaded_state, dict):
            return
        if "optimizer" in loaded_state:
            self.optimizer.load_state_dict(loaded_state["optimizer"])
        if "torch_rng_state" in loaded_state:
            torch.set_rng_state(loaded_state["torch_rng_state"])
        if "cuda_rng_state_all" in loaded_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(loaded_state["cuda_rng_state_all"])
        if "numpy_rng_state" in loaded_state:
            np.random.set_state(loaded_state["numpy_rng_state"])
        if "python_rng_state" in loaded_state:
            random.setstate(loaded_state["python_rng_state"])

    def _get_sample_source(self):
        return str(getattr(self.args, "sample_source", "posterior")).lower()
