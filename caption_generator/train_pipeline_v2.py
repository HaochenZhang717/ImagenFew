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
from transformers import get_cosine_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

try:
    from caption_generator.pipeline_v2_model import PipelineV2CaptionModel
    from caption_generator.pipeline_v2_utils import (
        DeterministicTimeSeriesRenderer,
        PipelineV2Collator,
        PreRenderedTimeSeriesImageCaptionDataset,
        TimeSeriesImageCaptionDataset,
        compute_global_stats,
        load_split_arrays,
        save_json,
    )
except ImportError:
    from pipeline_v2_model import PipelineV2CaptionModel
    from pipeline_v2_utils import (
        DeterministicTimeSeriesRenderer,
        PipelineV2Collator,
        PreRenderedTimeSeriesImageCaptionDataset,
        TimeSeriesImageCaptionDataset,
        compute_global_stats,
        load_split_arrays,
        save_json,
    )



def parse_args():
    parser = argparse.ArgumentParser(description="Train pipeline_v2 frozen-VLM caption generator.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: Dict) -> torch.device:
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def initialize_wandb(cfg: Dict):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    if wandb is None:
        raise ImportError("wandb is enabled in config, but the wandb package is not installed.")
    return wandb.init(
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


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp: bool, amp_dtype) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
        )
        with autocast_ctx:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                labels=batch["labels"],
            )
        total_loss += float(outputs.loss.item())
        num_batches += 1
    model.train()
    return {"loss": total_loss / max(num_batches, 1)}


def build_datasets(cfg: Dict, model: PipelineV2CaptionModel):
    train_ts, _, _ = load_split_arrays(cfg["data"]["dataset_root"], "train")
    stats = compute_global_stats(train_ts)
    image_root = cfg["data"].get("prerendered_image_root")
    use_prerendered = bool(cfg["data"].get("use_prerendered_images", True)) and bool(image_root)

    if use_prerendered:
        train_dataset = PreRenderedTimeSeriesImageCaptionDataset(
            dataset_root=cfg["data"]["dataset_root"],
            image_root=image_root,
            split="train",
            prompt_text=cfg["data"]["instruction_prompt"],
        )
        eval_dataset = PreRenderedTimeSeriesImageCaptionDataset(
            dataset_root=cfg["data"]["dataset_root"],
            image_root=image_root,
            split=cfg["data"].get("eval_split", "valid"),
            prompt_text=cfg["data"]["instruction_prompt"],
        )
    else:
        renderer = DeterministicTimeSeriesRenderer(**cfg["renderer"])
        dataset_kwargs = {
            "dataset_root": cfg["data"]["dataset_root"],
            "prompt_text": cfg["data"]["instruction_prompt"],
            "renderer": renderer,
            "normalization_stats": stats,
            "normalize_before_render": cfg["data"].get("normalize_before_render", False),
        }
        train_dataset = TimeSeriesImageCaptionDataset(split="train", **dataset_kwargs)
        eval_dataset = TimeSeriesImageCaptionDataset(split=cfg["data"].get("eval_split", "valid"), **dataset_kwargs)
    collator = PipelineV2Collator(
        processor=model.processor,
        instruction_prompt=cfg["data"]["instruction_prompt"],
    )
    return train_dataset, eval_dataset, collator, stats


def main():
    args = parse_args()
    cfg = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.override)),
        resolve=True,
    )
    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg)
    run = initialize_wandb(cfg)
    model = PipelineV2CaptionModel(cfg).to(device)

    train_dataset, eval_dataset, collator, stats = build_datasets(cfg, model)
    save_json(stats, os.path.join(cfg["output_dir"], "train_stats.json"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg["training"].get("learning_rate", 2e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    steps_per_epoch = max(math.ceil(len(train_loader) / max(int(cfg["training"].get("grad_accum_steps", 1)), 1)), 1)
    total_steps = steps_per_epoch * int(cfg["training"]["num_epochs"])
    warmup_steps = int(total_steps * float(cfg["training"].get("warmup_ratio", 0.03)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    amp_dtype = getattr(torch, cfg["training"].get("amp_dtype", "bfloat16"))
    use_amp = bool(cfg["training"].get("use_amp", device.type == "cuda"))
    grad_accum_steps = max(int(cfg["training"].get("grad_accum_steps", 1)), 1)
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))

    best_eval_loss = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(cfg["training"]["num_epochs"])):
        model.train()
        progress = tqdm(train_loader, desc=f"train epoch {epoch + 1}")
        running_loss = 0.0

        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
            )
            with autocast_ctx:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            running_loss += float(outputs.loss.item())

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                mean_running_loss = running_loss / min(step + 1, grad_accum_steps)
                progress.set_postfix(loss=f"{mean_running_loss:.4f}", step=global_step)
                if run is not None:
                    wandb.log(
                        {
                            "train/loss": float(outputs.loss.item()),
                            "train/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )
                running_loss = 0.0

        eval_metrics = evaluate(model, eval_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
        print(json.dumps({"epoch": epoch + 1, "eval": eval_metrics}))
        if run is not None:
            wandb.log(
                {
                    "eval/loss": eval_metrics["loss"],
                    "epoch": epoch + 1,
                    "global_step": global_step,
                },
                step=global_step,
            )

        latest_path = os.path.join(cfg["output_dir"], "pipeline_v2_latest.pt")
        checkpoint = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "global_step": global_step,
            "eval_metrics": eval_metrics,
        }
        torch.save(checkpoint, latest_path)
        if eval_metrics["loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["loss"]
            torch.save(checkpoint, os.path.join(cfg["output_dir"], "pipeline_v2_best.pt"))

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
