"""
Export pretrained SimpleVAE posterior means for all finetune datasets listed in
a config directory.

Each dataset is exported separately as a tensor of shape (N, L, D), where:
- N is the number of samples in that dataset
- L is the latent sequence length after encoder downsampling
- D is latent_dim
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


def load_config(config_path):
    return OmegaConf.to_object(OmegaConf.load(config_path))


def build_simple_vae_model(vae_cfg, ckpt_path, device):
    dynamic_size = vae_cfg.get("dynamic_size", 32)
    if isinstance(dynamic_size, (list, tuple)):
        placeholder_dim = dynamic_size[0]
    else:
        placeholder_dim = dynamic_size

    model = SimpleVAE(
        input_dim=placeholder_dim,
        output_dim=placeholder_dim,
        hidden_size=vae_cfg["hidden_size"],
        num_layers=vae_cfg["num_layers"],
        num_heads=vae_cfg["num_heads"],
        latent_dim=vae_cfg["latent_dim"],
        beta=vae_cfg.get("beta", 0.001),
        dynamic_size=dynamic_size,
        encoder_channels=vae_cfg.get("encoder_channels", None),
        encoder_downsample_stages=vae_cfg.get("encoder_downsample_stages", 2),
        decoder_channels=vae_cfg.get("decoder_channels", None),
        decoder_res_blocks=vae_cfg.get("decoder_res_blocks", 1),
        decoder_dropout=vae_cfg.get("decoder_dropout", 0.0),
        decoder_upsample_stages=vae_cfg.get("decoder_upsample_stages", 2),
        seq_len=vae_cfg["seq_len"],
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    weights = state.get("model", state)
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


def export_dataset_latents(model, tensor_dataset, device, batch_size):
    chunks = []
    num_samples = tensor_dataset.shape[0]
    for start in range(0, num_samples, batch_size):
        x = tensor_dataset[start:start + batch_size].to(device=device, dtype=torch.float32)
        x = x.transpose(1, 2).contiguous()
        with torch.no_grad():
            mu, _ = model.encoder(x)
        chunks.append(mu.cpu())
    return torch.cat(chunks, dim=0)


def build_data_args(cfg, batch_size):
    args_dict = dict(cfg)
    args_dict.setdefault("batch_size", batch_size)
    args_dict.setdefault("num_workers", 0)
    args_dict.setdefault("input_channels", None)
    args_dict.setdefault("subset_p", None)
    args_dict.setdefault("subset_n", None)
    args_dict["device"] = "cpu"
    args_dict["ddp"] = False
    args_dict["pretrain"] = False
    args_dict["finetune"] = True
    args_dict["train_on_datasets"] = [
        dataset for dataset in dataset_list if dataset in args_dict["train_on_datasets"]
    ]
    return SimpleNamespace(**args_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-config", type=str, required=True, help="Path to SimpleVAE config")
    parser.add_argument("--vae-ckpt", type=str, required=True, help="Path to pretrained SimpleVAE checkpoint")
    parser.add_argument("--dataset-config-dir", type=str, required=True, help="Directory containing finetune configs")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save one latent .pt per dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    vae_cfg = load_config(args.vae_config)
    model = build_simple_vae_model(vae_cfg, args.vae_ckpt, args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    config_paths = sorted(
        [
            os.path.join(args.dataset_config_dir, name)
            for name in os.listdir(args.dataset_config_dir)
            if name.endswith(".yaml")
        ]
    )

    saved = {}
    for config_path in config_paths:
        cfg = load_config(config_path)
        data_args = build_data_args(cfg, args.batch_size)
        _, _, trainsets, _ = data_provider(data_args)

        for dataset_name in data_args.train_on_datasets:
            if dataset_name in saved:
                continue
            tensor_dataset = trainsets[dataset_name]
            print(f"Exporting latents for {dataset_name} from {os.path.basename(config_path)}...")
            latents = export_dataset_latents(
                model=model,
                tensor_dataset=tensor_dataset,
                device=args.device,
                batch_size=args.batch_size,
            )
            output_path = os.path.join(args.output_dir, f"{dataset_name}_mu.pt")
            torch.save(latents, output_path)
            saved[dataset_name] = {
                "path": output_path,
                "shape": tuple(latents.shape),
            }
            print(f"  saved {output_path} with shape {tuple(latents.shape)}")

    index_path = os.path.join(args.output_dir, "index.pt")
    torch.save(saved, index_path)
    print(f"Saved export index to {index_path}")


if __name__ == "__main__":
    main()
