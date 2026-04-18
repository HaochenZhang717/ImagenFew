import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
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
from stage1_model import Stage1LatentCaptionModel


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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    output = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output


def initialize_wandb(cfg: Dict):
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


def build_optimizer(model: Stage1LatentCaptionModel, cfg: Dict):
    train_cfg = cfg["training"]
    vae_params = list(model.vae.parameters())
    prompt_params = list(model.soft_prompt.parameters())
    llm_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]

    param_groups = [
        {"params": vae_params, "lr": train_cfg.get("vae_lr", train_cfg["learning_rate"])},
        {"params": prompt_params, "lr": train_cfg.get("projector_lr", train_cfg["learning_rate"])},
    ]
    if llm_params:
        param_groups.append(
            {"params": llm_params, "lr": train_cfg.get("llm_lr", train_cfg["learning_rate"])}
        )
    return torch.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 0.0))


def set_joint_phase_trainability(model: Stage1LatentCaptionModel, cfg: Dict) -> None:
    joint_cfg = cfg["training"].get("joint_phase", {})
    train_vae = bool(joint_cfg.get("train_vae", True))
    train_soft_prompt = bool(joint_cfg.get("train_soft_prompt", True))
    train_llm = bool(joint_cfg.get("train_llm", True))

    model.configure_trainable_modules(
        train_vae=train_vae,
        train_soft_prompt=train_soft_prompt,
        train_llm=train_llm,
    )


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp: bool, amp_dtype, phase_cfg: Dict):
    model.eval()
    totals = {
        "loss": 0.0,
        "caption_loss": 0.0,
        "vae_loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
    }
    num_batches = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
        )
        with autocast_ctx:
            if phase_cfg["name"] == "vae_pretrain":
                outputs = model.vae_pretrain_step(ts=batch["ts"])
            elif phase_cfg["name"] == "joint_caption":
                outputs = model.joint_caption_step(
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
    return {key: value / num_batches for key, value in totals.items()}


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    cfg = to_plain_dict(cfg)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "resolved_config.json"), "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2)
    wandb_run = initialize_wandb(cfg)

    set_seed(int(cfg.get("seed", 42)))

    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    train_ts, _, _ = load_split_arrays(data_cfg["dataset_root"], "train")
    stats = compute_train_normalization_stats(train_ts)
    save_normalization_stats(stats, os.path.join(cfg["output_dir"], "normalization_stats.json"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["llm_name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=model_cfg.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    collator = CaptionCollator(
        tokenizer=tokenizer,
        max_prompt_length=data_cfg.get("max_prompt_length", 128),
        max_caption_length=data_cfg.get("max_caption_length", 256),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_cfg.get("eval_batch_size", train_cfg["batch_size"]),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    model = Stage1LatentCaptionModel(cfg)
    if cfg["vae"].get("init_ckpt"):
        model.load_vae_weights(cfg["vae"]["init_ckpt"], map_location="cpu")
    model.to(device)

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
                model.configure_trainable_modules(
                    train_vae=True,
                    train_soft_prompt=False,
                    train_llm=False,
                )
            elif phase_cfg["name"] == "joint_caption":
                set_joint_phase_trainability(model, cfg)

            optimizer = build_optimizer(model, cfg)
            grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
            steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
            total_steps = steps_per_epoch * phase_cfg["num_epochs"]
            warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.03))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max(total_steps, 1),
            )
            best_val = float("inf")

            for epoch in range(1, phase_cfg["num_epochs"] + 1):
                model.train()
                running = {
                    "loss": 0.0,
                    "caption_loss": 0.0,
                    "vae_loss": 0.0,
                    "recon_loss": 0.0,
                    "kl_loss": 0.0,
                }
                optimizer.zero_grad(set_to_none=True)

                progress = tqdm(train_loader, desc=f"{phase_cfg['name']} epoch {epoch}", leave=False)
                for step, batch in enumerate(progress, start=1):
                    batch = move_batch_to_device(batch, device)
                    autocast_ctx = (
                        torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
                    )
                    with autocast_ctx:
                        if phase_cfg["name"] == "vae_pretrain":
                            outputs = model.vae_pretrain_step(ts=batch["ts"])
                        elif phase_cfg["name"] == "joint_caption":
                            outputs = model.joint_caption_step(
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
                            train_cfg.get("max_grad_norm", 1.0),
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

                train_metrics = {f"train/{key}": value / len(train_loader) for key, value in running.items()}
                val_metrics = evaluate(model, valid_loader, device, use_amp, amp_dtype, phase_cfg)
                val_metrics = {f"valid/{key}": value for key, value in val_metrics.items()}

                merged_metrics = {
                    "phase": phase_cfg["name"],
                    "phase_epoch": epoch,
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    **train_metrics,
                    **val_metrics,
                }
                with open(history_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(merged_metrics) + "\n")
                log_to_wandb(wandb_run, merged_metrics)

                ckpt = {
                    "model": model.state_dict(),
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
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
