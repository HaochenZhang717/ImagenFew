from .ImagenFew import ImagenFew
from .sampler import DiffusionProcess
from ..generative_handler import generativeHandler
from ..MultiScaleVAE.multiscale_vae import DualVAE

import os
import random
import torch
import logging
import numpy as np
from torchdiffeq import odeint

from diffusion_prior.models import DiT1D
from diffusion_prior.models.transport import Sampler, create_transport

   
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
        self._prior_model = None
        self._prior_sample_fn = None
        self._prior_seq_len = None
        self._prior_token_dim = None
        self._prior_context_cache = None
        self._prior_context_cursor = 0
        self._prior_transport = None
        self._prior_sampler = None

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
        epoch = getattr(self, "epoch", None)

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
        logger.log(f'train/karras loss', avg_loss, step=epoch)
        # logger.log(f'train/train epoch', loss.detach())

    def sample(self, n_samples, class_label, class_metadata, test_data):
        sample_source = self._get_sample_source()
        if sample_source == "prior":
            return self._sample_with_prior(n_samples, class_metadata)
        if sample_source == "posterior":
            return self._sample_with_posterior(n_samples, class_metadata, test_data)
        if sample_source == "both":
            return self._sample_with_prior(n_samples, class_metadata)
        raise ValueError(f"Unsupported sample_source: {sample_source}")

    def sample_variants(self, n_samples, class_label, class_metadata, test_data):
        sample_source = self._get_sample_source()
        if sample_source == "both":
            return {
                "prior": self._sample_with_prior(n_samples, class_metadata),
                "posterior": self._sample_with_posterior(n_samples, class_metadata, test_data),
            }
        return {
            sample_source: self.sample(n_samples, class_label, class_metadata, test_data),
        }

    def _sample_with_posterior(self, n_samples, class_metadata, test_data):
        generated_set = []
        with self._model.ema_scope():
            self.process = DiffusionProcess(self.args, self._model.net, (class_metadata['channels'], self.args.img_resolution, self.args.img_resolution))
            if test_data is None:
                raise ValueError("test_data must be provided for posterior sampling.")
            context_bank = self._prepare_sample_context(test_data)
            for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
                indices = torch.randperm(context_bank.shape[0], device=context_bank.device)[:sample_size]
                batch_context = context_bank[indices]

                x_img = torch.zeros(sample_size, class_metadata['channels'], self.args.img_resolution, self.args.img_resolution).to(self.args.device)
                x_ts_mask = self._model.ts_to_img(torch.zeros(sample_size, self.args.seq_len, class_metadata['channels']).to(self.args.device), pad_val=1)
                x_img_sampled = self.process.interpolate(x_img, x_ts_mask, context=batch_context)
                x_ts = self._model.img_to_ts(x_img_sampled)[:,:,:class_metadata['channels']]
                generated_set.append(x_ts)
        return torch.concat(generated_set, dim=0)

    def _sample_with_prior(self, n_samples, class_metadata):
        generated_set = []
        with self._model.ema_scope():
            self.process = DiffusionProcess(
                self.args,
                self._model.net,
                (class_metadata['channels'], self.args.img_resolution, self.args.img_resolution),
            )
            self._ensure_prior_context_cache(n_samples)
            for sample_size in [min(self.args.batch_size, n_samples - i) for i in range(0, n_samples, self.args.batch_size)]:
                batch_context = self._sample_prior_context(sample_size)
                x_img = torch.zeros(sample_size, class_metadata['channels'], self.args.img_resolution, self.args.img_resolution).to(self.args.device)
                x_ts_mask = self._model.ts_to_img(torch.zeros(sample_size, self.args.seq_len, class_metadata['channels']).to(self.args.device), pad_val=1)
                x_img_sampled = self.process.interpolate(x_img, x_ts_mask, context=batch_context)
                x_ts = self._model.img_to_ts(x_img_sampled)[:, :, :class_metadata['channels']]
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
                loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint from {ckpt_path}. The file may be corrupted "
                    f"or may not be a valid PyTorch checkpoint."
                ) from exc
            model_state = loaded_state['model'] if isinstance(loaded_state, dict) and 'model' in loaded_state else loaded_state
            self.model.load_state_dict(model_state, strict=False)
            if 'ema_model' in loaded_state and self.args.ema is not None:
                self.model.model_ema.load_state_dict(loaded_state['ema_model'], strict=False)

            if not self.args.finetune:
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

    def _prepare_sample_context(self, context_source):
        if context_source is None:
            return None

        if not torch.is_tensor(context_source):
            raise TypeError(
                "Expected context source to be a torch.Tensor containing either raw time series (B, L, C_ts) "
                "or precomputed context tokens (B, L, C) / (L, C)."
            )

        context_source = context_source.to(self.args.device, dtype=torch.float32)
        if context_source.ndim == 2 and context_source.shape[-1] == self.args.context_dim:
            context = context_source.unsqueeze(0)
        elif context_source.ndim == 3 and context_source.shape[-1] == self.args.context_dim:
            context = context_source
        else:
            context = self._encode_context(context_source)
        # breakpoint()
        # if context_source.ndim == 3 and self._looks_like_raw_timeseries(context_source):
        #     context = self._encode_context(context_source)
        # else:
        #     context = context_source.to(self.args.device, dtype=torch.float32)
        #
        # if context.ndim == 2:
        #     context = context.unsqueeze(0).expand(n_samples, -1, -1)
        # elif context.ndim == 3:
        #     if context.shape[0] == 1 and n_samples > 1:
        #         context = context.expand(n_samples, -1, -1)
        # else:
        #     raise ValueError(
        #         f"Expected sampling context with shape (B, L, C) or (L, C), got {tuple(context.shape)}."
        #     )

        return context

    def _init_diffusion_prior(self):
        if self._prior_model is not None:
            return

        prior_ckpt = getattr(self.args, "prior_ckpt", None)
        if not prior_ckpt:
            raise ValueError("prior_ckpt must be provided to sample latent contexts from diffusion prior.")
        if not os.path.exists(prior_ckpt):
            raise FileNotFoundError(f"Diffusion prior checkpoint not found: {prior_ckpt}")

        state = torch.load(prior_ckpt, map_location=self.device, weights_only=False)
        model_args = state["model_args"]
        self._prior_seq_len = state["seq_len"]
        self._prior_token_dim = state["token_dim"]
        self._prior_model = DiT1D(
            seq_len=self._prior_seq_len,
            token_dim=self._prior_token_dim,
            hidden_size=model_args["hidden_size"],
            depth=model_args["depth"],
            num_heads=model_args["num_heads"],
            mlp_ratio=model_args["mlp_ratio"],
            use_qknorm=model_args["use_qknorm"],
            use_rmsnorm=model_args["use_rmsnorm"],
        ).to(self.device)

        prior_state = state["ema_model"] if getattr(self.args, "prior_use_ema", True) and "ema_model" in state else state["model"]
        self._prior_model.load_state_dict(prior_state, strict=True)
        self._prior_model.eval()

        if self._prior_token_dim != int(self.args.context_dim):
            raise ValueError(
                f"Diffusion prior token_dim ({self._prior_token_dim}) does not match ImagenFew context_dim ({self.args.context_dim})."
            )

        transport = create_transport(**state["transport_args"])
        sampler = Sampler(transport)
        self._prior_sampler = sampler
        self._prior_transport = transport
        sampler_cfg = self._get_prior_sampler_config()
        if sampler_cfg["mode"].upper() != "ODE":
            raise ValueError(f"Unsupported prior sampler mode: {sampler_cfg['mode']}. Only ODE is supported.")
        self._prior_sample_fn = sampler.sample_ode(
            sampling_method=sampler_cfg["sampling_method"],
            num_steps=sampler_cfg["num_steps"],
            atol=sampler_cfg["atol"],
            rtol=sampler_cfg["rtol"],
            reverse=sampler_cfg["reverse"],
        )
        logging.info(
            f"Loaded diffusion prior from {prior_ckpt} "
            f"with latent shape (L={self._prior_seq_len}, D={self._prior_token_dim})"
        )

    def _sample_prior_context(self, batch_size):
        if self._prior_context_cache is None:
            self._ensure_prior_context_cache(batch_size)

        start = self._prior_context_cursor
        end = start + batch_size
        if end > self._prior_context_cache.shape[0]:
            self._prior_context_cursor = 0
            start = 0
            end = batch_size

        self._prior_context_cursor = end
        return self._prior_context_cache[start:end].to(self.device, non_blocking=True)

    def _ensure_prior_context_cache(self, num_samples):
        requested_cache_size = getattr(self.args, "prior_cache_size", None)
        cache_size = num_samples if requested_cache_size is None else int(requested_cache_size)
        cache_size = max(num_samples, cache_size)
        if self._prior_context_cache is not None and self._prior_context_cache.shape[0] >= cache_size:
            self._prior_context_cursor = 0
            return

        self._prior_context_cache = self._draw_prior_contexts(cache_size)
        self._prior_context_cursor = 0
        logging.info(f"Cached {cache_size} diffusion-prior latent contexts on CPU.")

    def _draw_prior_contexts(self, num_samples):
        self._init_diffusion_prior()
        sample_batch_size = max(1, int(getattr(self.args, "prior_cache_batch_size", getattr(self.args, "batch_size", 128))))
        chunks = []
        remaining = num_samples
        while remaining > 0:
            bs = min(sample_batch_size, remaining)
            init = torch.randn(
                bs,
                self._prior_seq_len,
                self._prior_token_dim,
                device=self.device,
            )
            with torch.no_grad():
                xs = self._prior_sample_fn(init, self._prior_model)
            chunks.append(xs[-1].detach().cpu())
            remaining -= bs
        return torch.cat(chunks, dim=0)

    def _get_prior_sampler_config(self):
        sampler_cfg = getattr(self.args, "sampler", None)
        if isinstance(sampler_cfg, dict):
            params = dict(sampler_cfg.get("params", {}))
            return {
                "mode": sampler_cfg.get("mode", "ODE"),
                "sampling_method": params.get("sampling_method", "dopri5"),
                "num_steps": int(params.get("num_steps", 50)),
                "atol": float(params.get("atol", 1e-6)),
                "rtol": float(params.get("rtol", 1e-3)),
                "reverse": bool(params.get("reverse", False)),
            }

        return {
            "mode": "ODE",
            "sampling_method": getattr(self.args, "prior_method", "dopri5"),
            "num_steps": int(getattr(self.args, "prior_num_steps", 50)),
            "atol": float(getattr(self.args, "prior_atol", 1e-6)),
            "rtol": float(getattr(self.args, "prior_rtol", 1e-3)),
            "reverse": bool(getattr(self.args, "prior_reverse", False)),
        }

    def _get_sample_source(self):
        sample_source = getattr(self.args, "sample_source", None)
        if sample_source is None:
            return "prior" if getattr(self.args, "prior_ckpt", None) else "posterior"

        sample_source = str(sample_source).lower()
        if sample_source == "prior" and not getattr(self.args, "prior_ckpt", None):
            raise ValueError("sample_source='prior' requires prior_ckpt to be set.")
        if sample_source not in {"prior", "posterior", "both"}:
            raise ValueError("sample_source must be one of {'prior', 'posterior', 'both'}.")
        return sample_source

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
