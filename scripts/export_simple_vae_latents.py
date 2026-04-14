"""
Export pretrained SimpleVAE posterior means as a single latent tensor.

For each sample, the script takes posterior means with shape:
    [B, L, D]
where:
    - B is batch size
    - L is the latent sequence length after encoder downsampling
    - D is latent_dim

Finally, all requested datasets are concatenated along the sample dimension and
saved as one tensor with shape:
    [N, L, D]

Example:
    python scripts/export_simple_vae_latents.py \
        --config configs/pretrain/simple_vae_pretrain.yaml \
        --ckpt ./logs/simple_vae_pretrain/<run_id>/simple_vae.pt \
        --output ./logs/vae_latents/simple_pretrain_mu.pt
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from data_provider.combined_datasets import dataset_list
from data_provider.data_provider import data_provider
from models.MultiScaleVAE.simple_vae import SimpleVAE


def load_config(config_path, overrides):
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def build_model(cfg, ckpt_path, device):
    dynamic_size = cfg.get("dynamic_size", 32)
    if isinstance(dynamic_size, (list, tuple)):
        placeholder_dim = dynamic_size[0]
    else:
        placeholder_dim = dynamic_size

    model = SimpleVAE(
        input_dim=placeholder_dim,
        output_dim=placeholder_dim,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        latent_dim=cfg["latent_dim"],
        beta=cfg.get("beta", 0.001),
        dynamic_size=dynamic_size,
        encoder_channels=cfg.get("encoder_channels", None),
        encoder_downsample_stages=cfg.get("encoder_downsample_stages", 2),
        decoder_channels=cfg.get("decoder_channels", None),
        decoder_res_blocks=cfg.get("decoder_res_blocks", 1),
        decoder_dropout=cfg.get("decoder_dropout", 0.0),
        decoder_upsample_stages=cfg.get("decoder_upsample_stages", 2),
        seq_len=cfg["seq_len"],
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


def export_dataset_latents(model, dataset_name, tensor_dataset, device, batch_size):
    storage = []
    num_samples = tensor_dataset.shape[0]

    for start in range(0, num_samples, batch_size):
        x = tensor_dataset[start:start + batch_size].to(device=device, dtype=torch.float32)
        x = x.transpose(1, 2).contiguous()

        with torch.no_grad():
            mu, _ = model.encoder(x)

        storage.append(mu.cpu())

    if not storage:
        raise RuntimeError(f"No samples found for dataset {dataset_name}")

    dataset_latents = torch.cat(storage, dim=0)
    return dataset_latents, num_samples


def build_data_args(cfg, batch_size):
    args_dict = dict(cfg)
    args_dict.setdefault("batch_size", batch_size)
    args_dict.setdefault("num_workers", 0)
    args_dict.setdefault("input_channels", None)
    args_dict.setdefault("subset_p", None)
    args_dict.setdefault("subset_n", None)
    args_dict["device"] = "cpu"
    args_dict["ddp"] = False
    args_dict["finetune"] = not args_dict.get("pretrain", False)
    args_dict["train_on_datasets"] = [
        dataset for dataset in dataset_list if dataset in args_dict["train_on_datasets"]
    ]
    return SimpleNamespace(**args_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to SimpleVAE pretrain config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to pretrained SimpleVAE checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Where to save exported latents")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    model = build_model(cfg, args.ckpt, args.device)
    data_args = build_data_args(cfg, args.batch_size)
    _, _, trainsets, _ = data_provider(data_args)

    exported = []
    total_samples = 0

    for dataset_name in data_args.train_on_datasets:
        tensor_dataset = trainsets[dataset_name]
        print(f"Exporting latents for {dataset_name}...")
        dataset_latents, num_samples = export_dataset_latents(
            model=model,
            dataset_name=dataset_name,
            tensor_dataset=tensor_dataset,
            device=args.device,
            batch_size=args.batch_size,
        )
        exported.append(dataset_latents)
        total_samples += num_samples
        print(
            f"  done: {num_samples} samples, "
            f"latent shape {tuple(dataset_latents.shape)}"
        )

    exported = torch.cat(exported, dim=0)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(exported, args.output)
    print(f"Saved latents to {args.output} with shape {tuple(exported.shape)} from {total_samples} samples")


if __name__ == "__main__":
    main()
