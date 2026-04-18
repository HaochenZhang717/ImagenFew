import argparse
import json
import math
import os
import random
import traceback
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

from diffusion_prior_models import DiT1D, ResNet1D, create_transport
from diffusion_prior_models.transport.transport import Sampler
from simple_vae import SimpleVAE
from stage1_dataset import compute_train_normalization_stats, load_split_arrays
from stage1_model import Stage1LatentCaptionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 diffusion prior over Stage 1 latents.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_plain_dict(cfg) -> Dict:
    return OmegaConf.to_container(cfg, resolve=True)


def setup_distributed(cfg: Dict):
    training_cfg = cfg.get("training", {})
    ddp_requested = bool(training_cfg.get("ddp", False))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = ddp_requested or world_size > 1

    if not use_ddp:
        return {
            "enabled": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main": True,
        }

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed(ddp_state: Dict) -> None:
    if ddp_state["enabled"] and dist.is_initialized():
        distributed_barrier(ddp_state)
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def reduce_scalar(value: float, device: torch.device, ddp_state: Dict) -> float:
    if not ddp_state["enabled"]:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= ddp_state["world_size"]
    return float(tensor.item())


def distributed_barrier(ddp_state: Dict) -> None:
    if not ddp_state["enabled"]:
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[ddp_state["local_rank"]])
    else:
        dist.barrier()


def rank0_print(ddp_state: Dict, message: str) -> None:
    if ddp_state["is_main"]:
        print(message, flush=True)


def resolve_local_batch_size(global_batch_size: int, ddp_state: Dict, field_name: str) -> int:
    world_size = ddp_state["world_size"]
    if global_batch_size < 1:
        raise ValueError(f"{field_name} must be >= 1, got {global_batch_size}")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"{field_name}={global_batch_size} must be divisible by world_size={world_size} "
            "when using DDP so the YAML value remains a true global batch size."
        )
    return global_batch_size // world_size


def maybe_cast_dtype(dtype_name: Optional[str]):
    if dtype_name is None or dtype_name == "auto":
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def initialize_wandb(cfg: Dict, ddp_state: Dict):
    if not ddp_state["is_main"]:
        return None
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in config, but the wandb package is not installed.")

    return wandb.init(
        project=wandb_cfg.get("project", "caption_generator_stage2"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        dir=cfg["output_dir"],
        config=cfg,
        mode=wandb_cfg.get("mode", "online"),
        resume=wandb_cfg.get("resume", "allow"),
    )


def log_to_wandb(run, metrics: Dict) -> None:
    if run is None:
        return
    wandb.log(metrics, step=metrics.get("global_step"))


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buf in model_buffers.items():
        if name in ema_buffers:
            ema_buffers[name].copy_(buf)


def build_simple_vae(cfg: Dict) -> SimpleVAE:
    vae_cfg = cfg["vae"]
    return SimpleVAE(
        input_dim=vae_cfg["input_dim"],
        output_dim=vae_cfg["output_dim"],
        hidden_size=vae_cfg.get("hidden_size", 128),
        num_layers=vae_cfg.get("num_layers", 4),
        num_heads=vae_cfg.get("num_heads", 4),
        latent_dim=vae_cfg.get("latent_dim", 64),
        beta=vae_cfg.get("beta", 0.001),
        dynamic_size=vae_cfg.get("dynamic_size"),
        encoder_channels=vae_cfg.get("encoder_channels"),
        encoder_downsample_stages=vae_cfg.get("encoder_downsample_stages", 2),
        decoder_channels=vae_cfg.get("decoder_channels"),
        decoder_res_blocks=vae_cfg.get("decoder_res_blocks", 1),
        decoder_dropout=vae_cfg.get("decoder_dropout", 0.0),
        decoder_upsample_stages=vae_cfg.get("decoder_upsample_stages", 2),
        seq_len=vae_cfg["seq_len"],
    )


def load_stage1_config_and_checkpoint(cfg: Dict):
    stage1_cfg = to_plain_dict(OmegaConf.load(cfg["stage1"]["config_path"]))
    stage1_ckpt = torch.load(cfg["stage1"]["checkpoint_path"], map_location="cpu")
    return stage1_cfg, stage1_ckpt


def resolve_latent_cache_path(cfg: Dict) -> str:
    cache_path = cfg["data"].get("latents_cache_path")
    if cache_path:
        return cache_path
    return os.path.join(cfg["output_dir"], "stage1_latents_cache.pt")


@torch.no_grad()
def encode_split_latents(
    vae_model: SimpleVAE,
    dataset_root: str,
    split: str,
    stats: Dict[str, float],
    latent_source: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    ts, _, _ = load_split_arrays(dataset_root, split)
    ts = ((ts - stats["mean"]) / stats["std"]).astype(np.float32)

    storage = []
    for start in tqdm(range(0, len(ts), batch_size), desc=f"encode {split}", leave=False):
        batch = torch.from_numpy(ts[start : start + batch_size]).permute(0, 2, 1).contiguous().to(device)
        mu, logvar = vae_model.encoder(batch)
        if latent_source == "mu":
            latent = mu
        elif latent_source == "sample":
            latent = vae_model.reparameterize(mu, logvar)
        else:
            raise ValueError(f"Unsupported latent_source: {latent_source}")
        storage.append(latent.cpu())
    return torch.cat(storage, dim=0)


def prepare_cached_latents(
    cfg: Dict,
    stage1_cfg: Dict,
    stage1_ckpt: Dict,
    ts_stats: Dict[str, float],
    ddp_state: Dict,
    device: torch.device,
):
    dataset_root = cfg["data"].get("dataset_root", stage1_cfg["data"]["dataset_root"])
    eval_split = cfg["data"].get("eval_split", "valid")
    latent_source = cfg["data"].get("latent_source", "mu")
    encode_batch_size = cfg["data"].get("encode_batch_size", cfg["training"]["batch_size"])
    cache_path = resolve_latent_cache_path(cfg)
    rebuild_cache = bool(cfg["data"].get("rebuild_latents_cache", False))

    if ddp_state["is_main"] and (rebuild_cache or not os.path.exists(cache_path)):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        vae_model = build_simple_vae(stage1_cfg).to(device)
        vae_state = stage1_ckpt["model"]
        vae_state = {k.replace("vae.", "", 1): v for k, v in vae_state.items() if k.startswith("vae.")}
        missing, unexpected = vae_model.load_state_dict(vae_state, strict=False)
        if missing:
            print(f"[warn] missing {len(missing)} VAE keys when loading Stage 1 checkpoint")
        if unexpected:
            print(f"[warn] unexpected {len(unexpected)} VAE keys when loading Stage 1 checkpoint")
        vae_model.eval().requires_grad_(False)

        train_latents = encode_split_latents(
            vae_model=vae_model,
            dataset_root=dataset_root,
            split="train",
            stats=ts_stats,
            latent_source=latent_source,
            batch_size=encode_batch_size,
            device=device,
        )
        valid_latents = encode_split_latents(
            vae_model=vae_model,
            dataset_root=dataset_root,
            split=eval_split,
            stats=ts_stats,
            latent_source=latent_source,
            batch_size=encode_batch_size,
            device=device,
        )
        torch.save(
            {
                "train_latents": train_latents,
                "valid_latents": valid_latents,
                "meta": {
                    "dataset_root": dataset_root,
                    "eval_split": eval_split,
                    "latent_source": latent_source,
                    "stage1_checkpoint_path": cfg["stage1"]["checkpoint_path"],
                },
            },
            cache_path,
        )
        del vae_model

    distributed_barrier(ddp_state)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Latent cache was expected but not found: {cache_path}")

    latent_cache = torch.load(cache_path, map_location="cpu")
    train_latents = latent_cache["train_latents"].float()
    valid_latents = latent_cache["valid_latents"].float()

    if ddp_state["is_main"]:
        print(
            json.dumps(
                {
                    "latent_cache_path": cache_path,
                    "train_latents_shape": list(train_latents.shape),
                    "valid_latents_shape": list(valid_latents.shape),
                },
                indent=2,
            )
        )

    distributed_barrier(ddp_state)

    return train_latents, valid_latents


def build_backbone(cfg: Dict):
    model_cfg = cfg["diffusion_model"]
    name = model_cfg["name"].lower()
    kwargs = {
        "seq_len": model_cfg["seq_len"],
        "token_dim": model_cfg["token_dim"],
    }
    kwargs.update(model_cfg.get("kwargs", {}))

    if name == "dit1d":
        return DiT1D(**kwargs)
    if name == "resnet1d":
        return ResNet1D(**kwargs)
    raise ValueError(f"Unsupported diffusion backbone: {model_cfg['name']}")


def build_transport_and_sampler(cfg: Dict):
    transport_cfg = cfg["transport"]
    transport = create_transport(
        path_type=transport_cfg.get("path_type", "Linear"),
        prediction=transport_cfg.get("prediction", "velocity"),
        loss_weight=transport_cfg.get("loss_weight"),
        train_eps=transport_cfg.get("train_eps"),
        sample_eps=transport_cfg.get("sample_eps"),
        time_dist_type=transport_cfg.get("time_dist_type", "uniform"),
        time_dist_shift=transport_cfg.get("time_dist_shift", 1.0),
    )
    sampler = Sampler(transport)
    sample_cfg = cfg["sampling"]
    sample_fn = sampler.sample_ode(
        sampling_method=sample_cfg.get("ode_method", "dopri5"),
        num_steps=sample_cfg.get("ode_num_steps", 50),
        atol=sample_cfg.get("ode_atol", 1e-6),
        rtol=sample_cfg.get("ode_rtol", 1e-3),
        reverse=False,
    )
    return transport, sample_fn


def save_checkpoint(
    output_path: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    cfg: Dict,
) -> None:
    base_model = unwrap_model(model)
    ckpt = {
        "model": base_model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg,
    }
    torch.save(ckpt, output_path)


@torch.no_grad()
def evaluate(model, dataloader, transport, device: torch.device, ddp_state: Dict) -> Dict[str, float]:
    model.eval()
    base_model = unwrap_model(model)
    losses = []
    for (latent_batch,) in tqdm(dataloader, desc="eval", leave=False, disable=not ddp_state["is_main"]):
        latent_batch = latent_batch.to(device)
        terms = transport.training_losses(base_model, latent_batch)
        losses.append(float(terms["loss"].mean().detach().cpu()))
    if not losses:
        raise RuntimeError("Validation dataloader is empty.")
    local_loss = sum(losses) / len(losses)
    return {"loss": reduce_scalar(local_loss, device=device, ddp_state=ddp_state)}


def build_stage1_decoder(
    stage1_cfg: Dict,
    stage1_ckpt: Dict,
    device: torch.device,
) -> tuple[Stage1LatentCaptionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        stage1_cfg["model"]["llm_name"],
        trust_remote_code=stage1_cfg["model"].get("trust_remote_code", False),
        use_fast=stage1_cfg["model"].get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    decoder = Stage1LatentCaptionModel(stage1_cfg)
    decoder.load_state_dict(stage1_ckpt["model"], strict=False)
    decoder.requires_grad_(False)
    decoder.to(device).eval()
    return decoder, tokenizer


@torch.no_grad()
def decode_sampled_latents(
    decoder: Stage1LatentCaptionModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    sampled_latents: torch.Tensor,
    generation_cfg: Dict,
    device: torch.device,
):
    prompts = [prompt_text] * sampled_latents.size(0)
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    caption_latent = sampled_latents.to(device)
    soft_prompt_embeds = decoder.soft_prompt(caption_latent)
    token_embeds = decoder.llm.get_input_embeddings()(tokenized["input_ids"])
    inputs_embeds = torch.cat([soft_prompt_embeds, token_embeds], dim=1)
    prefix_mask = torch.ones(
        tokenized["attention_mask"].size(0),
        decoder.soft_prompt_tokens,
        device=device,
        dtype=tokenized["attention_mask"].dtype,
    )
    full_attention_mask = torch.cat([prefix_mask, tokenized["attention_mask"]], dim=1)

    outputs = decoder.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention_mask,
        max_new_tokens=generation_cfg.get("max_new_tokens", 192),
        do_sample=generation_cfg.get("temperature", 0.0) > 0,
        temperature=generation_cfg.get("temperature", 0.8),
        top_p=generation_cfg.get("top_p", 0.95),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


@torch.no_grad()
def sample_and_decode(
    model_for_sampling: torch.nn.Module,
    decoder: Stage1LatentCaptionModel,
    tokenizer: AutoTokenizer,
    sample_fn,
    cfg: Dict,
    device: torch.device,
):
    sampling_cfg = cfg["sampling"]
    latent_shape = (
        sampling_cfg["num_decode_samples"],
        cfg["diffusion_model"]["seq_len"],
        cfg["diffusion_model"]["token_dim"],
    )
    init = torch.randn(latent_shape, device=device)
    sample_traj = sample_fn(init, model_for_sampling)
    sampled_latents = sample_traj[-1]
    decoded = decode_sampled_latents(
        decoder=decoder,
        tokenizer=tokenizer,
        prompt_text=cfg["stage1_prompt"],
        sampled_latents=sampled_latents,
        generation_cfg=sampling_cfg,
        device=device,
    )
    return sampled_latents.detach().cpu(), decoded


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    cfg = to_plain_dict(cfg)
    ddp_state = setup_distributed(cfg)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if ddp_state["is_main"]:
        with open(os.path.join(cfg["output_dir"], "resolved_config.json"), "w", encoding="utf-8") as fp:
            json.dump(cfg, fp, indent=2)

    wandb_run = initialize_wandb(cfg, ddp_state)
    set_seed(int(cfg.get("seed", 42)) + ddp_state["rank"])

    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{ddp_state['local_rank']}" if ddp_state["enabled"] else "cuda")
        else:
            device = torch.device("cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    elif requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{ddp_state['local_rank']}" if ddp_state["enabled"] else "cuda")
    else:
        device = torch.device(requested_device)

    stage1_cfg, stage1_ckpt = load_stage1_config_and_checkpoint(cfg)
    dataset_root = cfg["data"].get("dataset_root", stage1_cfg["data"]["dataset_root"])
    cfg["stage1_prompt"] = stage1_cfg["data"]["prompt_template"]

    train_ts, _, _ = load_split_arrays(dataset_root, "train")
    ts_stats = compute_train_normalization_stats(train_ts)
    if ddp_state["is_main"]:
        with open(os.path.join(cfg["output_dir"], "ts_normalization_stats.json"), "w", encoding="utf-8") as fp:
            json.dump(ts_stats, fp, indent=2)

    train_latents, valid_latents = prepare_cached_latents(
        cfg=cfg,
        stage1_cfg=stage1_cfg,
        stage1_ckpt=stage1_ckpt,
        ts_stats=ts_stats,
        ddp_state=ddp_state,
        device=device,
    )

    cfg["diffusion_model"]["seq_len"] = int(train_latents.shape[1])
    cfg["diffusion_model"]["token_dim"] = int(train_latents.shape[2])

    train_batch_size = resolve_local_batch_size(
        cfg["training"]["batch_size"],
        ddp_state,
        "batch_size",
    )
    eval_batch_size = resolve_local_batch_size(
        cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]),
        ddp_state,
        "eval_batch_size",
    )
    train_dataset = TensorDataset(train_latents)
    valid_dataset = TensorDataset(valid_latents)

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=ddp_state["world_size"],
            rank=ddp_state["rank"],
            shuffle=True,
        )
        if ddp_state["enabled"]
        else None
    )
    valid_sampler = (
        DistributedSampler(
            valid_dataset,
            num_replicas=ddp_state["world_size"],
            rank=ddp_state["rank"],
            shuffle=False,
        )
        if ddp_state["enabled"]
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_backbone(cfg).to(device)
    if ddp_state["enabled"]:
        ddp_kwargs = {"device_ids": [ddp_state["local_rank"]]} if device.type == "cuda" else {}
        model = DDP(model, **ddp_kwargs)
    ema_model = deepcopy(unwrap_model(model)).to(device)
    ema_model.eval().requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )
    total_steps = max(len(train_loader) * cfg["training"]["num_epochs"], 1)
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.03))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    transport, sample_fn = build_transport_and_sampler(cfg)
    history_path = os.path.join(cfg["output_dir"], "metrics.jsonl")
    decoded_path = os.path.join(cfg["output_dir"], "decoded_samples.jsonl")
    global_step = 0
    best_val = float("inf")

    ema_decay = float(cfg["training"].get("ema_decay", 0.9995))
    ema_warmup_steps = int(cfg["training"].get("ema_warmup_steps", 0))
    sample_every_epochs = int(cfg["sampling"].get("sample_every_epochs", 1))
    decoder = None
    tokenizer = None
    if ddp_state["is_main"] and sample_every_epochs > 0:
        rank0_print(ddp_state, "[stage2] loading stage1 decoder for sampling")
        decoder, tokenizer = build_stage1_decoder(stage1_cfg, stage1_ckpt, device=device)
        rank0_print(ddp_state, "[stage2] finished loading stage1 decoder for sampling")
    rank0_print(
        ddp_state,
        json.dumps(
            {
                "output_dir": cfg["output_dir"],
                "num_epochs": int(cfg["training"]["num_epochs"]),
                "batch_size_global": int(cfg["training"]["batch_size"]),
                "eval_batch_size_global": int(
                    cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
                ),
                "sample_every_epochs": sample_every_epochs,
                "num_decode_samples": int(cfg["sampling"].get("num_decode_samples", 0)),
                "training_ddp": bool(cfg["training"].get("ddp", False)),
                "world_size": ddp_state["world_size"],
            },
            indent=2,
        ),
    )

    try:
        for epoch in range(1, cfg["training"]["num_epochs"] + 1):
            rank0_print(ddp_state, f"[stage2] starting epoch {epoch}/{cfg['training']['num_epochs']}")
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0

            progress = tqdm(
                train_loader,
                desc=f"stage2 epoch {epoch}",
                leave=False,
                disable=not ddp_state["is_main"],
            )
            for step, (latent_batch,) in enumerate(progress, start=1):
                latent_batch = latent_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                terms = transport.training_losses(model, latent_batch)
                loss = terms["loss"].mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()

                base_model = unwrap_model(model)
                if global_step < ema_warmup_steps:
                    ema_model.load_state_dict(base_model.state_dict())
                else:
                    update_ema(ema_model, base_model, decay=ema_decay)

                running_loss += float(loss.detach().cpu())
                global_step += 1
                progress.set_postfix(loss=f"{running_loss / step:.4f}")

            train_loss = reduce_scalar(running_loss / len(train_loader), device=device, ddp_state=ddp_state)
            valid_metrics = evaluate(model, valid_loader, transport, device, ddp_state)
            ema_valid_metrics = evaluate(ema_model, valid_loader, transport, device, ddp_state)

            metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": scheduler.get_last_lr()[0],
                "train/loss": train_loss,
                "valid/loss": valid_metrics["loss"],
                "valid_ema/loss": ema_valid_metrics["loss"],
            }

            if ddp_state["is_main"]:
                with open(history_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(metrics) + "\n")
                log_to_wandb(wandb_run, metrics)

                save_checkpoint(
                    output_path=os.path.join(cfg["output_dir"], "stage2_latest.pt"),
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                )
                if ema_valid_metrics["loss"] < best_val:
                    best_val = ema_valid_metrics["loss"]
                    save_checkpoint(
                        output_path=os.path.join(cfg["output_dir"], "stage2_best.pt"),
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        cfg=cfg,
                    )

            if ddp_state["is_main"] and sample_every_epochs > 0 and epoch % sample_every_epochs == 0:
                rank0_print(ddp_state, f"[stage2] sampling at epoch {epoch}")
                try:
                    _, decoded = sample_and_decode(
                        model_for_sampling=ema_model,
                        decoder=decoder,
                        tokenizer=tokenizer,
                        sample_fn=sample_fn,
                        cfg=cfg,
                        device=device,
                    )
                except Exception as exc:
                    rank0_print(ddp_state, f"[stage2] sampling failed at epoch {epoch}: {exc!r}")
                    rank0_print(ddp_state, traceback.format_exc())
                    raise

                sample_rows = []
                for sample_id, text in enumerate(decoded):
                    row = {
                        "epoch": epoch,
                        "sample_id": sample_id,
                        "text": text,
                    }
                    sample_rows.append(row)
                    with open(decoded_path, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")

                if wandb_run is not None:
                    table = wandb.Table(columns=["epoch", "sample_id", "text"])
                    for row in sample_rows:
                        table.add_data(row["epoch"], row["sample_id"], row["text"])
                    wandb.log({"samples/decoded_text": table}, step=global_step)

                print(json.dumps({"epoch": epoch, "decoded_samples": sample_rows}, ensure_ascii=False, indent=2))
                rank0_print(ddp_state, f"[stage2] finished sampling at epoch {epoch}")

            if ddp_state["is_main"]:
                print(json.dumps(metrics, indent=2))
            distributed_barrier(ddp_state)
    finally:
        distributed_barrier(ddp_state)
        rank0_print(ddp_state, "[stage2] all ranks reached training shutdown barrier")
        if wandb_run is not None:
            rank0_print(ddp_state, "[stage2] finishing wandb run")
            wandb_run.finish()
            rank0_print(ddp_state, "[stage2] finished wandb run")
        cleanup_distributed(ddp_state)


if __name__ == "__main__":
    main()
