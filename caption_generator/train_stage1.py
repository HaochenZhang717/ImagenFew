import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

from stage1_dataset import (
    CaptionCollator,
    TimeSeriesCaptionDataset,
    compute_train_normalization_stats,
    load_split_arrays,
    save_normalization_stats,
)
from stage1_model import Stage1LatentCaptionModel, Stage1LatentCaptionModelVE


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 latent-conditioned caption generator.")
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

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


def cleanup_distributed(ddp_state: Dict) -> None:
    if ddp_state["enabled"] and dist.is_initialized():
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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def initialize_wandb(cfg: Dict, ddp_state: Dict):
    if not ddp_state["is_main"]:
        return None
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in config, but the wandb package is not installed.")

    run = wandb.init(
        project=wandb_cfg.get("project", "caption_generator"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        dir=cfg["output_dir"],
        config=cfg,
        mode=wandb_cfg.get("mode", "online"),
        resume=wandb_cfg.get("resume", "allow"),
    )
    return run


def log_to_wandb(run, metrics: Dict) -> None:
    if run is None:
        return
    wandb.log(metrics, step=metrics.get("global_step"))


def get_phase_value(train_cfg: Dict, phase_cfg: Dict, key: str, default=None):
    return phase_cfg.get(key, train_cfg.get(key, default))


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


def build_optimizer(model: Stage1LatentCaptionModel, train_cfg: Dict, phase_cfg: Dict):
    base_model = unwrap_model(model)
    vae_params = [p for p in base_model.vae.parameters() if p.requires_grad]
    prompt_params = [p for p in base_model.soft_prompt.parameters() if p.requires_grad]
    llm_params = [p for _, p in base_model.llm.named_parameters() if p.requires_grad]

    default_lr = get_phase_value(train_cfg, phase_cfg, "learning_rate", 2.0e-4)
    param_groups = []
    if vae_params:
        param_groups.append(
            {"params": vae_params, "lr": get_phase_value(train_cfg, phase_cfg, "vae_lr", default_lr)}
        )
    if prompt_params:
        param_groups.append(
            {"params": prompt_params, "lr": get_phase_value(train_cfg, phase_cfg, "projector_lr", default_lr)}
        )
    if llm_params:
        param_groups.append(
            {"params": llm_params, "lr": get_phase_value(train_cfg, phase_cfg, "llm_lr", default_lr)}
        )
    return torch.optim.AdamW(
        param_groups,
        weight_decay=get_phase_value(train_cfg, phase_cfg, "weight_decay", 0.0),
    )


def set_joint_phase_trainability(model: Stage1LatentCaptionModel, cfg: Dict, phase_cfg: Dict) -> None:
    base_model = unwrap_model(model)
    joint_cfg = dict(cfg["training"].get("joint_phase", {}))
    joint_cfg.update(phase_cfg.get("joint_phase", {}))
    train_vae = bool(joint_cfg.get("train_vae", True))
    train_soft_prompt = bool(joint_cfg.get("train_soft_prompt", True))
    train_llm = bool(joint_cfg.get("train_llm", True))

    base_model.configure_trainable_modules(
        train_vae=train_vae,
        train_soft_prompt=train_soft_prompt,
        train_llm=train_llm,
    )


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp: bool, amp_dtype, phase_cfg: Dict, ddp_state: Dict):
    model.eval()
    base_model = unwrap_model(model)
    totals = {
        "loss": 0.0,
        "caption_loss": 0.0,
        "vae_loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
    }
    num_batches = 0

    for batch in tqdm(dataloader, desc="eval", leave=False, disable=not ddp_state["is_main"]):
        batch = move_batch_to_device(batch, device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
        )
        with autocast_ctx:
            if phase_cfg["name"] == "vae_pretrain":
                outputs = base_model.vae_pretrain_step(ts=batch["ts"])
            elif phase_cfg["name"] == "joint_caption":
                outputs = base_model.joint_caption_step(
                    ts=batch["ts"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    caption_loss_weight=phase_cfg.get("caption_loss_weight", 1.0),
                    kl_loss_weight=phase_cfg.get("kl_loss_weight", 0.0),
                )
            else:
                raise ValueError(f"Unknown phase name: {phase_cfg['name']}")
        for key in totals:
            totals[key] += float(outputs[key].detach().cpu())
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Evaluation dataloader is empty.")

    local_metrics = {key: value / num_batches for key, value in totals.items()}
    return {
        key: reduce_scalar(value, device=device, ddp_state=ddp_state)
        for key, value in local_metrics.items()
    }


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
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    train_ts, _, _ = load_split_arrays(data_cfg["dataset_root"], "train")
    stats = compute_train_normalization_stats(train_ts)
    if ddp_state["is_main"]:
        save_normalization_stats(stats, os.path.join(cfg["output_dir"], "normalization_stats.json"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["llm_name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=model_cfg.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_variant = str(model_cfg.get("stage1_model_class", "base")).strip().lower()
    ve_cfg = dict(cfg.get("vocab_expansion", {}))
    if not ve_cfg:
        ve_cfg = dict(model_cfg.get("vocab_expansion", {}))
    dataset_strategy = str(ve_cfg.get("dataset_strategy", "")).strip().lower()
    use_ve_model = (
        model_variant in {"ve", "vocab_expansion", "stage1latentcaptionmodelve"}
        or dataset_strategy == "ettm1"
    )
    if use_ve_model:
        model = Stage1LatentCaptionModelVE(cfg, tokenizer=tokenizer)
    else:
        model = Stage1LatentCaptionModel(cfg)

    caption_transform = None
    if isinstance(model, Stage1LatentCaptionModelVE):
        caption_transform = model.encode_caption_with_special_tokens
        if ddp_state["is_main"]:
            save_path = model.save_vocab_expansion_artifacts(cfg["output_dir"])
            if save_path is not None:
                print(f"Saved vocab expansion summary to: {save_path}")
            tokenizer_dir = os.path.join(cfg["output_dir"], "tokenizer_ve")
            tokenizer.save_pretrained(tokenizer_dir)
            print(f"Saved VE tokenizer to: {tokenizer_dir}")

    collator = CaptionCollator(
        tokenizer=tokenizer,
        max_prompt_length=data_cfg.get("max_prompt_length", 128),
        max_caption_length=data_cfg.get("max_caption_length", 256),
        caption_transform=caption_transform,
    )

    train_dataset = TimeSeriesCaptionDataset(
        dataset_root=data_cfg["dataset_root"],
        split="train",
        prompt_template=data_cfg["prompt_template"],
        normalization_stats=stats,
    )
    valid_dataset = TimeSeriesCaptionDataset(
        dataset_root=data_cfg["dataset_root"],
        split=data_cfg.get("eval_split", "valid"),
        prompt_template=data_cfg["prompt_template"],
        normalization_stats=stats,
    )

    if cfg["vae"].get("init_ckpt"):
        model.load_vae_weights(cfg["vae"]["init_ckpt"], map_location="cpu")
    model.to(device)
    if ddp_state["enabled"]:
        ddp_kwargs = {"device_ids": [ddp_state["local_rank"]]} if device.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=True, **ddp_kwargs)

    use_amp = device.type == "cuda" and model_cfg.get("torch_dtype") in {"float16", "bfloat16"}
    amp_dtype = getattr(torch, model_cfg["torch_dtype"]) if use_amp else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and model_cfg.get("torch_dtype") == "float16")

    history_path = os.path.join(cfg["output_dir"], "metrics.jsonl")
    global_step = 0
    phases = train_cfg.get(
        "phases",
        [
            {"name": "vae_pretrain", "num_epochs": train_cfg.get("vae_pretrain_epochs", 0)},
            {
                "name": "joint_caption",
                "num_epochs": train_cfg.get("joint_caption_epochs", 0),
                "caption_loss_weight": train_cfg.get("caption_loss_weight", 1.0),
                "kl_loss_weight": train_cfg.get("joint_kl_loss_weight", 0.0),
            },
        ],
    )

    try:
        for phase_cfg in phases:
            if phase_cfg.get("num_epochs", 0) <= 0:
                continue

            if phase_cfg["name"] == "vae_pretrain":
                unwrap_model(model).configure_trainable_modules(
                    train_vae=bool(phase_cfg.get("train_vae", True)),
                    train_soft_prompt=bool(phase_cfg.get("train_soft_prompt", False)),
                    train_llm=bool(phase_cfg.get("train_llm", False)),
                )
            elif phase_cfg["name"] == "joint_caption":
                set_joint_phase_trainability(model, cfg, phase_cfg)

            phase_global_batch_size = get_phase_value(train_cfg, phase_cfg, "batch_size", 1)
            phase_global_eval_batch_size = get_phase_value(
                train_cfg,
                phase_cfg,
                "eval_batch_size",
                phase_global_batch_size,
            )
            phase_batch_size = resolve_local_batch_size(
                phase_global_batch_size,
                ddp_state,
                "batch_size",
            )
            phase_eval_batch_size = resolve_local_batch_size(
                phase_global_eval_batch_size,
                ddp_state,
                "eval_batch_size",
            )
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
                batch_size=phase_batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=data_cfg.get("num_workers", 0),
                collate_fn=collator,
                pin_memory=torch.cuda.is_available(),
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=phase_eval_batch_size,
                shuffle=False,
                sampler=valid_sampler,
                num_workers=data_cfg.get("num_workers", 0),
                collate_fn=collator,
                pin_memory=torch.cuda.is_available(),
            )

            optimizer = build_optimizer(model, train_cfg, phase_cfg)
            grad_accum_steps = get_phase_value(train_cfg, phase_cfg, "gradient_accumulation_steps", 1)
            steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
            total_steps = steps_per_epoch * phase_cfg["num_epochs"]
            warmup_steps = int(total_steps * get_phase_value(train_cfg, phase_cfg, "warmup_ratio", 0.03))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max(total_steps, 1),
            )

            # scheduler = get_constant_schedule_with_warmup(
            #     optimizer,
            #     num_warmup_steps=warmup_steps,
            # )

            best_val = float("inf")

            for epoch in range(1, phase_cfg["num_epochs"] + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                model.train()
                base_model = unwrap_model(model)
                running = {
                    "loss": 0.0,
                    "caption_loss": 0.0,
                    "vae_loss": 0.0,
                    "recon_loss": 0.0,
                    "kl_loss": 0.0,
                }
                optimizer.zero_grad(set_to_none=True)

                progress = tqdm(
                    train_loader,
                    desc=f"{phase_cfg['name']} epoch {epoch}",
                    leave=False,
                    disable=not ddp_state["is_main"],
                )
                for step, batch in enumerate(progress, start=1):
                    batch = move_batch_to_device(batch, device)
                    autocast_ctx = (
                        torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
                    )
                    with autocast_ctx:
                        if phase_cfg["name"] == "vae_pretrain":
                            outputs = base_model.vae_pretrain_step(ts=batch["ts"])
                        elif phase_cfg["name"] == "joint_caption":
                            outputs = base_model.joint_caption_step(
                                ts=batch["ts"],
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                                caption_loss_weight=phase_cfg.get("caption_loss_weight", 1.0),
                                kl_loss_weight=phase_cfg.get("kl_loss_weight", 0.0),
                            )
                        else:
                            raise ValueError(f"Unknown phase name: {phase_cfg['name']}")
                        loss = outputs["loss"] / grad_accum_steps

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    for key in running:
                        running[key] += float(outputs[key].detach().cpu())

                    if step % grad_accum_steps == 0 or step == len(train_loader):
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            get_phase_value(train_cfg, phase_cfg, "max_grad_norm", 1.0),
                        )
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                    avg_loss = running["loss"] / step
                    progress.set_postfix(loss=f"{avg_loss:.4f}")

                train_metrics = {
                    f"train/{key}": reduce_scalar(
                        value / len(train_loader),
                        device=device,
                        ddp_state=ddp_state,
                    )
                    for key, value in running.items()
                }
                val_metrics = evaluate(model, valid_loader, device, use_amp, amp_dtype, phase_cfg, ddp_state)
                val_metrics = {f"valid/{key}": value for key, value in val_metrics.items()}

                merged_metrics = {
                    "phase": phase_cfg["name"],
                    "phase_epoch": epoch,
                    "global_step": global_step,
                    "global_batch_size": phase_global_batch_size,
                    "global_eval_batch_size": phase_global_eval_batch_size,
                    "batch_size": phase_batch_size,
                    "eval_batch_size": phase_eval_batch_size,
                    "gradient_accumulation_steps": grad_accum_steps,
                    "lr": scheduler.get_last_lr()[0],
                    "rank": ddp_state["rank"],
                    "world_size": ddp_state["world_size"],
                    **train_metrics,
                    **val_metrics,
                }
                if ddp_state["is_main"]:
                    with open(history_path, "a", encoding="utf-8") as fp:
                        fp.write(json.dumps(merged_metrics) + "\n")
                    log_to_wandb(wandb_run, merged_metrics)

                    ckpt = {
                        "model": unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "phase": phase_cfg["name"],
                        "phase_epoch": epoch,
                        "global_step": global_step,
                        "config": cfg,
                    }
                    torch.save(ckpt, os.path.join(cfg["output_dir"], f"{phase_cfg['name']}_latest.pt"))

                    current_val = val_metrics["valid/loss"]
                    if current_val < best_val:
                        best_val = current_val
                        torch.save(ckpt, os.path.join(cfg["output_dir"], f"{phase_cfg['name']}_best.pt"))

                    print(json.dumps(merged_metrics, indent=2))
                if ddp_state["enabled"]:
                    dist.barrier()
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        cleanup_distributed(ddp_state)


if __name__ == "__main__":
    main()
