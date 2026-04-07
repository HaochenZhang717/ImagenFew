"""
Export pretrained MultiScaleVAE latents for all datasets used during VAE pretraining.

Example:
    python scripts/export_vae_latents.py \
        --config configs/pretrain/vae_pretrain.yaml \
        --ckpt ./logs/vae_pretrain/<run_id>/MultiScaleVAE.pt \
        --output ./logs/vae_latents/pretrain_latents.pt
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from data_provider.data_provider import get_train, get_test
from models.MultiScaleVAE.multiscale_vae import DualVAE


def load_config(config_path, overrides):
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def build_model(cfg, ckpt_path, device):
    model = DualVAE(
        z_channels=cfg["z_channels"],
        ch=cfg["unet_channels"],
        ch_mult=tuple(cfg["ch_mult"]),
        dynamic_size=cfg["dynamic_size"],
        dropout=cfg.get("dropout", 0.0),
        test_mode=True,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    weights = state.get("model", state)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"[warn] missing {len(missing)} keys while loading checkpoint")
    if unexpected:
        print(f"[warn] unexpected {len(unexpected)} keys while loading checkpoint")
    model.eval()
    return model


def _extract_timeseries(batch, seq_len):
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    return batch[:, :seq_len].to(torch.float32)


def export_dataset_latents(model, dataset_cfg, cfg, device, batch_size, sample, split):
    dataset_cfg = dict(dataset_cfg)
    dataset_cfg["seq_len"] = cfg["seq_len"]
    dataset_cfg["datasets_dir"] = cfg["datasets_dir"]

    if split == "train":
        dataset = get_train(dataset_cfg)
    elif split == "test":
        dataset = get_test(dataset_cfg)
    else:
        raise ValueError(f"Unsupported split: {split}")
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    storage = {
        "z_low_freq": [],
        "z_mid_freq": [],
        "z_high_freq": [],
        "mu_low_freq": [],
        "mu_mid_freq": [],
        "mu_high_freq": [],
        "logvar_low_freq": [],
        "logvar_mid_freq": [],
        "logvar_high_freq": [],
    }
    num_samples = 0
    seq_len = None
    channels = None

    for batch in loader:
        x = _extract_timeseries(batch, cfg["seq_len"]).to(device)
        num_samples += x.shape[0]
        seq_len = x.shape[1]
        channels = x.shape[2]

        with torch.no_grad():
            out = model.ts_to_z(x, sample=sample, return_dict=True)

        for key in storage:
            storage[key].append(out[key].cpu())

    dataset_latents = {key: torch.cat(value, dim=0) for key, value in storage.items()}
    dataset_latents["num_samples"] = num_samples
    dataset_latents["input_seq_len"] = seq_len
    dataset_latents["input_channels"] = channels
    dataset_latents["latent_seq_len"] = dataset_latents["z_low_freq"].shape[-1]
    dataset_latents["latent_channels"] = dataset_latents["z_low_freq"].shape[1]
    dataset_latents["split"] = split
    return dataset_latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to VAE pretrain config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to pretrained VAE checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Where to save exported latents")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--sample", action="store_true", help="Sample z instead of using posterior mean")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"], help="Dataset splits to export")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    model = build_model(cfg, args.ckpt, args.device)

    dataset_names = set(cfg["train_on_datasets"])
    datasets = [dataset for dataset in cfg["datasets"] if dataset["name"] in dataset_names]

    exported = {
        "config_path": args.config,
        "checkpoint_path": args.ckpt,
        "sample": args.sample,
        "splits": args.splits,
        "dataset_names": [dataset["name"] for dataset in datasets],
        "datasets": {},
    }

    for dataset_cfg in datasets:
        dataset_name = dataset_cfg["name"]
        exported["datasets"][dataset_name] = {}
        for split in args.splits:
            print(f"Exporting latents for {dataset_name} ({split})...")
            exported["datasets"][dataset_name][split] = export_dataset_latents(
                model=model,
                dataset_cfg=dataset_cfg,
                cfg=cfg,
                device=args.device,
                batch_size=args.batch_size,
                sample=args.sample,
                split=split,
            )
            summary = exported["datasets"][dataset_name][split]
            print(
                f"  done: {summary['num_samples']} samples, "
                f"latent shape {tuple(summary['z_low_freq'].shape)} per scale"
            )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(exported, args.output)
    print(f"Saved latents to {args.output}")


if __name__ == "__main__":
    main()
