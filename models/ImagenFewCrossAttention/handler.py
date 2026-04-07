from .ImagenFew import ImagenFew
from .sampler import DiffusionProcess
from ..generative_handler import generativeHandler
from ..MultiScaleVAE.multiscale_vae import DualVAE

import os
import random
import torch
import logging
import numpy as np

   
class Handler(generativeHandler):

    def __init__(self, args, rank=None):
        if getattr(args, "context_dim", None) is None:
            z_channels = getattr(args, "z_channels", None)
            if z_channels is None:
                z_channels = getattr(args, "latent_channels", None)
            if z_channels is None:
                z_channels = getattr(args, "multi_scale_vae", {}).get("latent_channels")
            if z_channels is None:
                z_channels = getattr(args, "multi_scale_vae", {}).get("z_channels")
            if z_channels is None:
                raise ValueError("Unable to infer context_dim. Please set args.context_dim or multi_scale_vae.z_channels.")
            args.context_dim = int(z_channels)

        super().__init__(args, rank)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.pretrained_vae = DualVAE(**args.multi_scale_vae).to(self.args.device)
        self.resume_epoch = 0
        self.best_score = float("inf")

        pretrained_vae_weights = torch.load(args.pretrained_vae_weights, map_location="cpu")
        self.pretrained_vae.load_state_dict(pretrained_vae_weights['model'])
        print("loaded pretrained VAE weights from {}".format(args.pretrained_vae_weights))
        for param in self.pretrained_vae.parameters():
            param.requires_grad = False
        self.pretrained_vae.eval()

        if not self.args.finetune:
            self._load_optimizer_state(self.args.model_ckpt, self.args.device)


    def build_model(self):
        self.model = ImagenFew(self.args, self.args.device)
        if self.args.finetune:
            self._load_model(self.args.model_ckpt, self.args.device)
            if self.args.ema:
                self.model.setup_finetune(self.args)
        else:
            if self.args.model_ckpt is not None:
                self._load_model(self.args.model_ckpt, self.args.device)
        return self.model
    
    def train_iter(self, train_dataloader, logger):
        avg_loss = 0.0

        for _, data in enumerate(train_dataloader, 1):
            self.optimizer.zero_grad()

            # Time series & mask
            # x_ts = data[0].to(self.args.device)
            x_ts = data[0].to(self.args.device, dtype=torch.float32)
            context = self._encode_context(x_ts)
            x_ts_mask = torch.zeros_like(x_ts)

            # Convert time series & mask to image
            x_img = self.model.ts_to_img(x_ts)
            x_img_mask = self.model.ts_to_img(x_ts_mask, pad_val=1)

            output, time_weight = self.model(x_img, x_img_mask, context=context)
            time_loss   = (output - x_img).square()
            loss = ((time_weight * time_loss) * (1 - x_img_mask)).mean()
            # logger.log(f'train/karras loss', loss.detach())
            avg_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
            self._model.on_train_batch_end()

        avg_loss /= len(train_dataloader)
        logger.log(f'train/karras loss', avg_loss)
        # logger.log(f'train/train epoch', loss.detach())

    def sample(self, n_samples, class_label, class_metadata, test_data):
        generated_set = []
        with self._model.ema_scope():
            self.process = DiffusionProcess(self.args, self._model.net, (class_metadata['channels'], self.args.img_resolution, self.args.img_resolution))
            candidate_data = test_data if test_data is not None else class_label
            context_bank = self._prepare_sample_context(candidate_data, n_samples)
            for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
                if context_bank is None:
                    batch_context = None
                else:
                    indices = torch.randperm(context_bank.shape[0], device=context_bank.device)[:sample_size]
                    batch_context = context_bank[indices]
                x_img = torch.zeros(sample_size, class_metadata['channels'], self.args.img_resolution, self.args.img_resolution).to(self.args.device)
                x_ts_mask = self._model.ts_to_img(torch.zeros(sample_size, self.args.seq_len, class_metadata['channels']).to(self.args.device), pad_val=1)
                x_img_sampled = self.process.interpolate(x_img, x_ts_mask, context=batch_context)
                x_ts = self._model.img_to_ts(x_img_sampled)[:,:,:class_metadata['channels']]
                generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)
    
    def save_model(self, ckpt_dir):
        ckpt_path = self._resolve_ckpt_path(ckpt_dir, must_exist=False)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        epoch = int(getattr(self, 'epoch', 0))
        state = {
            'model': self._model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_score': getattr(self, 'best_score', float("inf")),
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        if self.args.ema is not None:
            state['ema_model'] = self._model.model_ema.state_dict()
        if torch.cuda.is_available():
            state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        epoch_ckpt_path = self._epoch_ckpt_path(ckpt_path, epoch)
        self._atomic_save(state, epoch_ckpt_path)
        self._atomic_save(state, ckpt_path)
        logging.info(f"Saved checkpoint to {epoch_ckpt_path} and updated latest checkpoint {ckpt_path}")

    def _atomic_save(self, state, ckpt_path):
        tmp_path = f"{ckpt_path}.tmp"
        with open(tmp_path, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ckpt_path)

    def _load_model(self, ckpt_dir, device):
        ckpt_path = self._resolve_ckpt_path(ckpt_dir, must_exist=True)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            logging.warning(f"No checkpoint found at {ckpt_dir}. "
                            f"Returned the same state as input")
        else:
            try:
                loaded_state = torch.load(ckpt_path, map_location=device)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint from {ckpt_path}. The file may be corrupted "
                    f"or may not be a valid PyTorch checkpoint."
                ) from exc
            model_state = loaded_state['model'] if isinstance(loaded_state, dict) and 'model' in loaded_state else loaded_state
            self.model.load_state_dict(model_state, strict=False)
            if 'ema_model' in loaded_state and self.args.ema is not None:
                self.model.model_ema.load_state_dict(loaded_state['ema_model'], strict=False)
            self.resume_epoch = int(loaded_state.get('epoch', 0))
            self.global_step = int(loaded_state.get('global_step', 0))
            self.best_score = float(loaded_state.get('best_score', float("inf")))
            if self.args.finetune and self.args.ema:
                self.model.setup_finetune(self.args)
            logging.info(f'Successfully loaded previous state')

    def _load_optimizer_state(self, ckpt_dir, device):
        ckpt_path = self._resolve_ckpt_path(ckpt_dir, must_exist=True)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            return

        try:
            loaded_state = torch.load(ckpt_path, map_location=device)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load optimizer state from {ckpt_path}. The checkpoint may be corrupted."
            ) from exc
        if not isinstance(loaded_state, dict):
            return

        if 'optimizer' in loaded_state:
            self.optimizer.load_state_dict(loaded_state['optimizer'])
            logging.info('Successfully restored optimizer state')
        if 'torch_rng_state' in loaded_state:
            torch.set_rng_state(loaded_state['torch_rng_state'])
        if 'cuda_rng_state_all' in loaded_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(loaded_state['cuda_rng_state_all'])
        if 'numpy_rng_state' in loaded_state:
            np.random.set_state(loaded_state['numpy_rng_state'])
        if 'python_rng_state' in loaded_state:
            random.setstate(loaded_state['python_rng_state'])

    def _encode_context(self, x_ts):
        with torch.no_grad():
            z_low_freq, z_mid_freq, z_high_freq = self.pretrained_vae.ts_to_z(x_ts, sample=False)
        context = torch.cat([z_low_freq, z_mid_freq, z_high_freq], dim=-1).permute(0, 2, 1).contiguous()
        return context.to(self.args.device, dtype=torch.float32)

    def _prepare_sample_context(self, context_source, n_samples):
        if context_source is None:
            return None

        if not torch.is_tensor(context_source):
            raise TypeError(
                "Expected context source to be a torch.Tensor containing either raw time series (B, L, C_ts) "
                "or precomputed context tokens (B, L, C) / (L, C)."
            )

        context_source = context_source.to(self.args.device)
        if context_source.ndim == 3 and self._looks_like_raw_timeseries(context_source):
            context = self._encode_context(context_source)
        else:
            context = context_source.to(self.args.device, dtype=torch.float32)

        if context.ndim == 2:
            context = context.unsqueeze(0).expand(n_samples, -1, -1)
        elif context.ndim == 3:
            if context.shape[0] == 1 and n_samples > 1:
                context = context.expand(n_samples, -1, -1)
        else:
            raise ValueError(
                f"Expected sampling context with shape (B, L, C) or (L, C), got {tuple(context.shape)}."
            )

        return context

    def _looks_like_raw_timeseries(self, tensor):
        context_dim = getattr(self.args, "context_dim", None)
        ts_channels = getattr(self.args, "input_channels", None)
        if context_dim is not None and tensor.shape[-1] == context_dim:
            return False
        if ts_channels is not None and tensor.shape[-1] == ts_channels:
            return True
        return True

    def _resolve_ckpt_path(self, ckpt_path, must_exist=True):
        if ckpt_path is None:
            return None

        candidates = [ckpt_path]
        if not ckpt_path.endswith((".pt", ".pth", ".ckpt")):
            candidates = [f"{ckpt_path}.pt", ckpt_path]

        if must_exist:
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return None

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _epoch_ckpt_path(self, ckpt_path, epoch):
        base, ext = os.path.splitext(ckpt_path)
        ext = ext or ".pt"
        return f"{base}_epoch_{epoch:04d}{ext}"
