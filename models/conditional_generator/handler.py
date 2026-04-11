import os
import logging
import math
import tempfile
import torch
import torch.nn as nn
import torch.distributed as tdist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from ..generative_handler import generativeHandler
from .model import SelfConditionalGenerator
from diffusion_prior.models import DiT1D
from diffusion_prior.models.transport import Sampler, create_transport


class Handler(generativeHandler):
    """
    Handler for the class-conditional diffusion model (ConditionalGenerator).

    Conditioning signal: integer class label (dataset index) → nn.Embedding →
    context of shape (B, 1, cond_dim_in), then processed by
    TextProjectorMVarMScaleMStep inside ConditionalGenerator.

    Required config keys:
        handler:   models.conditional_generator.handler
        seq_len:   <int>
        n_var:     <int>   # number of variables in the target dataset
        cond_dim_in:  128  # embedding dim fed into the projector
        cond_dim_out: 64   # projector output dim (should == diffusion.channels)
        diffusion:
            n_var:                <int>
            channels:             64
            num_steps:            50
            beta_start:           0.0001
            beta_end:             0.5
            schedule:             quad
            diffusion_embedding_dim: 128
            nheads:               8
            layers:               4
            multipatch_num:       3
            base_patch:           1
            L_patch_len:          2
            condition_type:       add   # add | cross_attention | adaLN
            n_stages:             5     # timestep stages in TextProjectorMVarMScaleMStep
            side:
                num_var:  <int>
                var_emb:  16
                time_emb: 16
        pretrain_path: ""
        sampler:       ddim   # ddim | ddpm
    """

    def __init__(self, args, rank=None):
        args.find_unused_parameters = True
        super().__init__(args, rank)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = None
        self.resume_epoch = 0
        self.best_score = float("inf")
        self._pending_resume_state = None
        self._prior_model = None
        self._prior_sample_fn = None
        self._prior_seq_len = None
        self._prior_token_dim = None
        self._prior_context_cache = None
        self._prior_context_cursor = 0

        resume_ckpt = getattr(self.args, "resume_ckpt", None)
        if getattr(self.args, "pretrain", False) and resume_ckpt is None:
            # Backward-compatible fallback for older pretrain configs that used model_ckpt.
            resume_ckpt = getattr(self.args, "model_ckpt", None)

        if resume_ckpt:
            self._load_checkpoint(resume_ckpt)

    # ------------------------------------------------------------------
    # generativeHandler interface
    # ------------------------------------------------------------------

    def build_model(self):
        configs = self._build_configs()
        model = SelfConditionalGenerator(configs)
        if getattr(self.args, "pretrain_path", None):
            ckpt_state_dict = torch.load(self.args.pretrain_path, map_location="cpu")
            pretrained_state = ckpt_state_dict.get("model", ckpt_state_dict)

            diff_model_state = {
                key[len("diff_model."):]: value
                for key, value in pretrained_state.items()
                if key.startswith("diff_model.")
            }

            multi_scale_vae_state = {
                key[len("multi_scale_vae."):]: value
                for key, value in pretrained_state.items()
                if key.startswith("multi_scale_vae.")
            }

            cond_projector_state = {
                key[len("cond_projector."):]: value
                for key, value in pretrained_state.items()
                if key.startswith("cond_projector.")
            }

            if diff_model_state:
                model.diff_model.load_state_dict(diff_model_state, strict=True)
            else:
                raise ValueError("model.diff_model checkpoint not found.")

            if multi_scale_vae_state:
                model.multi_scale_vae.load_state_dict(multi_scale_vae_state, strict=True)
            else:
                raise ValueError("model.multi_scale_vae checkpoint not found.")

            if cond_projector_state:
                model.cond_projector.load_state_dict(cond_projector_state, strict=True)
            else:
                raise ValueError("model.cond_projector checkpoint not found.")

            # Newer alternative kept for future comparison:
            # if model.use_ema:
            #     if ema_diff_model_state:
            #         model.model_ema.load_state_dict(ema_diff_model_state, strict=True)

            if model.use_ema:
                # Match the previous behavior: initialize EMA from the finetune-start weights
                # instead of inheriting a stale EMA state from the pretrain checkpoint.
                model.reset_ema()
            print(f"Loaded pretrained conditional diffusion model from {self.args.pretrain_path}")

        return model

    def train_iter(self, train_dataloader, logger):
        if self.scheduler is None:
            self._init_scheduler(train_dataloader)

        train_loss = 0.0
        num_batches = 0.0
        epoch = getattr(self, "epoch", None)
        accumulation_steps = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))

        is_main = not (tdist.is_available() and tdist.is_initialized() and tdist.get_rank() != 0)
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False, disable=not is_main)
        self.optimizer.zero_grad()
        for batch_idx, (data, class_indices) in enumerate(pbar, 1):
            # data: (B, seq_len, n_var)
            x = data.to(self.device).float()
            loss_dict = self.model(x, is_train=True)
            loss = loss_dict["all"]
            loss_for_backward = loss / accumulation_steps

            loss_for_backward.backward()

            train_loss += loss.item()
            num_batches += 1
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{train_loss / num_batches:.4f}",
                lr=f"{current_lr:.2e}",
            )

            should_step = (batch_idx % accumulation_steps == 0) or (batch_idx == len(train_dataloader))
            if should_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self._model.on_train_batch_end()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

        logger.log("train/loss", train_loss / num_batches, step=epoch)
        logger.log("train/epoch", epoch, step=epoch)
        logger.log("train/lr", self.scheduler.get_last_lr()[0], step=epoch)

    def sample(self, n_samples, class_label, class_metadata, test_data):
        """
        Generate n_samples time series conditioned on class_label (int).
        Returns: (n_samples, seq_len, n_var) on CPU.
        """
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

    def _sample_with_prior(self, n_samples, class_metadata):
        n_var = class_metadata["channels"]
        seq_len = self.args.seq_len
        batch_size = getattr(self.args, "batch_size", 128)
        self._model.eval()

        generated = []
        remaining = n_samples
        with self._model.ema_scope(), torch.no_grad():
            if getattr(self.args, "prior_ckpt", None):
                self._ensure_prior_context_cache(n_samples)

            while remaining > 0:
                bs = min(batch_size, remaining)
                context = self._sample_prior_context(bs)
                samples = self._model.generate_from_context(context, seq_len, n_var)
                generated.append(samples)
                remaining -= bs

        return torch.cat(generated, dim=0)

    def _sample_with_posterior(self, n_samples, class_metadata, test_data):
        n_var = class_metadata["channels"]
        seq_len = self.args.seq_len
        batch_size = getattr(self.args, "batch_size", 128)
        self._model.eval()

        generated = []
        remaining = n_samples
        if test_data is None:
            raise ValueError("test_data must be provided for posterior sampling.")

        with self._model.ema_scope(), torch.no_grad():
            while remaining > 0:
                bs = min(batch_size, remaining)
                indices = torch.randperm(test_data.shape[0], device=test_data.device)[:bs]
                test_batch = test_data[indices].to(device=self.device, dtype=torch.float32)
                samples = self._model.generate(bs, test_batch, seq_len, n_var)
                generated.append(samples)
                remaining -= bs

        return torch.cat(generated, dim=0)

    def save_model(self, ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir) if os.path.dirname(ckpt_dir) else ".", exist_ok=True)
        state = {
            "model": self._model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": int(getattr(self, "epoch", 0)),
            "global_step": int(getattr(self, "global_step", 0)),
            "best_score": float(getattr(self, "best_score", float("inf"))),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self._model.use_ema:
            state["ema_model"] = self._model.model_ema.state_dict()
        self._atomic_torch_save(state, ckpt_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_configs(self):
        args = self.args
        diff_cfg = dict(getattr(args, "diffusion", {}))

        diff_cfg.setdefault("num_steps",              getattr(args, "num_steps", 50))
        diff_cfg.setdefault("beta_start",             getattr(args, "beta_start", 1e-4))
        diff_cfg.setdefault("beta_end",               getattr(args, "beta_end", 0.5))
        diff_cfg.setdefault("schedule",               getattr(args, "schedule", "quad"))
        diff_cfg.setdefault("channels",               getattr(args, "channels", 64))
        diff_cfg.setdefault("diffusion_embedding_dim",getattr(args, "diffusion_embedding_dim", 128))
        diff_cfg.setdefault("nheads",                 getattr(args, "nheads", 8))
        diff_cfg.setdefault("layers",                 getattr(args, "layers", 4))
        diff_cfg.setdefault("multipatch_num",         getattr(args, "multipatch_num", 3))
        diff_cfg.setdefault("base_patch",             getattr(args, "base_patch", 1))
        diff_cfg.setdefault("L_patch_len",            getattr(args, "L_patch_len", 2))
        diff_cfg.setdefault("condition_type",         getattr(args, "condition_type", "add"))
        diff_cfg.setdefault("n_stages",               getattr(args, "n_stages", 5))
        diff_cfg.setdefault("n_var",                  getattr(args, "n_var", None))

        side_cfg = dict(diff_cfg.get("side", {}))
        side_cfg.setdefault("num_var",  diff_cfg["n_var"])
        side_cfg.setdefault("var_emb",  getattr(args, "var_emb", 16))
        side_cfg.setdefault("time_emb", getattr(args, "time_emb", 16))
        side_cfg["device"] = self.device
        diff_cfg["side"] = side_cfg
        diff_cfg["device"] = self.device

        cond_dim_in  = getattr(args, "cond_dim_in",  128)
        cond_dim_out = getattr(args, "cond_dim_out", diff_cfg["channels"])

        configs = {
            "device":                 self.device,
            "seq_len":                args.seq_len,
            "n_classes":              getattr(args, "n_classes", 1),
            "cond_dim_in":            cond_dim_in,
            "cond_dim_out":           cond_dim_out,
            "pretrain_path":          getattr(args, "pretrain_path", ""),
            "diffusion":              diff_cfg,
            "multi_scale_vae":        dict(getattr(args, "multi_scale_vae", {})),
            "pretrained_vae_weights": getattr(args, "pretrained_vae_weights", ""),
            "ema":                    getattr(args, "ema", False),
            "ema_warmup":             getattr(args, "ema_warmup", 0),
            "ema_decay":              getattr(args, "ema_decay", 0.9999),
        }
        return configs

    def _load_model(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            logging.warning(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self._model.load_state_dict(state.get("model", state), strict=False)
        if self._model.use_ema and "ema_model" in state:
            self._model.model_ema.load_state_dict(state["ema_model"], strict=True)
        logging.info(f"Loaded checkpoint from {ckpt_path}")

    def _load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            logging.warning(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
            return

        state = torch.load(ckpt_path, map_location=self.device)
        model_state = state.get("model", state)
        self._model.load_state_dict(model_state, strict=False)

        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self._model.use_ema and "ema_model" in state:
            self._model.model_ema.load_state_dict(state["ema_model"], strict=True)

        self.resume_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_score = float(state.get("best_score", float("inf")))
        self._pending_resume_state = state
        logging.info(
            f"Resumed checkpoint from {ckpt_path} "
            f"(epoch={self.resume_epoch}, global_step={self.global_step}, best_score={self.best_score})"
        )

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

        expected_cond_dim = getattr(self.args, "cond_dim_in", None)
        if expected_cond_dim is not None and self._prior_token_dim != expected_cond_dim:
            raise ValueError(
                f"Diffusion prior token_dim ({self._prior_token_dim}) does not match cond_dim_in ({expected_cond_dim})."
            )

        transport = create_transport(**state["transport_args"])
        sampler = Sampler(transport)
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
                "sampling_method": params.get("sampling_method", "euler"),
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

    def _init_scheduler(self, train_dataloader):
        steps_per_epoch = max(1, len(train_dataloader))
        accumulation_steps = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1)))
        optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accumulation_steps)
        total_epochs = max(1, getattr(self.args, "epochs", 1) - 1)
        num_training_steps = optimizer_steps_per_epoch * total_epochs

        warmup_steps = getattr(self.args, "warmup_steps", None)
        if warmup_steps is None:
            warmup_steps = getattr(self.args, "num_warmup_steps", None)
        if warmup_steps is None:
            warmup_ratio = getattr(self.args, "warmup_ratio", 0.0)
            warmup_steps = int(num_training_steps * warmup_ratio)

        min_lr = float(getattr(self.args, "min_lr", 0.0))
        warmup_steps = min(max(0, int(warmup_steps)), num_training_steps)
        self.scheduler = self._get_cosine_schedule_with_warmup_min_lr(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )
        if self._pending_resume_state is not None and "scheduler" in self._pending_resume_state:
            self.scheduler.load_state_dict(self._pending_resume_state["scheduler"])
            self._pending_resume_state = None
        logging.info(
            "Initialized cosine LR scheduler with linear warmup: "
            f"steps_per_epoch={steps_per_epoch}, optimizer_steps_per_epoch={optimizer_steps_per_epoch}, "
            f"gradient_accumulation_steps={accumulation_steps}, total_epochs={total_epochs}, "
            f"num_training_steps={num_training_steps}, num_warmup_steps={warmup_steps}, "
            f"min_lr={min_lr}"
        )

    def _atomic_torch_save(self, state, ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else "."
        fd, tmp_path = tempfile.mkstemp(dir=ckpt_dir, prefix=".tmp_conditional_generator_", suffix=".pt")
        os.close(fd)
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, ckpt_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_cosine_schedule_with_warmup_min_lr(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        min_lr=0.0,
        num_cycles=0.5,
        last_epoch=-1,
    ):
        base_lr = float(getattr(self.args, "learning_rate", 0.0))
        min_lr = max(0.0, float(min_lr))
        min_lr_ratio = min_lr / base_lr if base_lr > 0.0 else 0.0
        min_lr_ratio = min(max(0.0, min_lr_ratio), 1.0)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_factor = max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)),
            )
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        return LambdaLR(optimizer, lr_lambda, last_epoch)
