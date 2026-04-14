"""
Quick visualization of SimpleVAE reconstruction.

Loads model architecture + dataset list from a YAML config (e.g.
configs/pretrain/simple_vae_pretrain.yaml), loads a checkpoint, then
reconstructs a batch from one of the configured datasets and plots the result.

Usage:
    python visualize_simple_vae.py \
        --config configs/pretrain/simple_vae_pretrain.yaml \
        --ckpt <path> \
        --dataset ETTh1 \
        --sample_idx 0
"""

import argparse
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_provider import dataset_to_tensor, get_train, random_permute
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

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        weights = state.get("model", state)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"  [warn] missing {len(missing)} keys (first 3: {missing[:3]})")
        if unexpected:
            print(f"  [warn] unexpected {len(unexpected)} keys (first 3: {unexpected[:3]})")
    else:
        print("[warn] No checkpoint loaded; showing an untrained model.")

    model.eval()
    return model


class _ArgsShim:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def load_dataset_samples(cfg, dataset_name, n_samples, device):
    ds_cfg = next((d for d in cfg["datasets"] if d["name"] == dataset_name), None)
    if ds_cfg is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config. Available: "
            f"{[d['name'] for d in cfg['datasets']]}"
        )

    ds_cfg = dict(ds_cfg)
    ds_cfg["seq_len"] = cfg["seq_len"]
    ds_cfg["datasets_dir"] = cfg["datasets_dir"]

    trainset = get_train(ds_cfg)
    testset = get_train(ds_cfg)
    trainset, _ = random_permute(trainset, testset)

    args_shim = _ArgsShim(
        {
            "batch_size": min(cfg.get("batch_size", 64), max(n_samples, 1)),
            "num_workers": 0,
            "seq_len": cfg["seq_len"],
        }
    )
    tensor = dataset_to_tensor(trainset, args_shim)
    tensor = tensor[:n_samples].to(device).float()
    print(f"Loaded {tensor.shape[0]} samples from '{dataset_name}' (shape={tuple(tensor.shape)})")
    return tensor


def plot_reconstruction(orig, recon, latent_mu, sample_idx, var_indices, save_path):
    n_vars = len(var_indices)
    fig = plt.figure(figsize=(4 * n_vars, 9))
    gs = gridspec.GridSpec(3, n_vars, hspace=0.4, wspace=0.3)

    def to_np(t, var):
        return t[sample_idx, :, var].detach().cpu().numpy()

    for col, var in enumerate(var_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(to_np(orig, var), label="original", lw=1.5)
        ax.plot(to_np(recon, var), label="recon", lw=1.5, ls="--")
        ax.set_title(f"Var {var} — input vs recon", fontsize=10)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, col])
        residual = to_np(orig, var) - to_np(recon, var)
        ax.plot(residual, color="red", lw=1.2)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"Var {var} — residual", fontsize=10)

        ax = fig.add_subplot(gs[2, col])
        mu = latent_mu[sample_idx].detach().cpu().numpy()  # (L, D)
        ax.imshow(mu.T, aspect="auto", origin="lower")
        ax.set_title(f"Sample {sample_idx} latent mu", fontsize=10)
        ax.set_xlabel("latent step")
        ax.set_ylabel("latent dim")

    fig.suptitle(f"SimpleVAE reconstruction — sample {sample_idx}", fontsize=12, y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_mse_summary(orig, recon, save_path):
    n_var = orig.shape[-1]
    mse = ((orig - recon) ** 2).mean(dim=(0, 1)).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(6, n_var * 0.6), 4))
    ax.bar(range(n_var), mse)
    ax.set_xlabel("Variable index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-variable reconstruction MSE")
    ax.set_xticks(range(n_var))
    overall = float(mse.mean())
    ax.axhline(overall, color="red", ls="--", label=f"mean = {overall:.4f}")
    ax.legend()

    if save_path:
        p = save_path.replace(".png", "_mse.png")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved figure: {p}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained SimpleVAE checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset in config to visualize")
    parser.add_argument("--n_samples", type=int, default=8, help="How many samples to load")
    parser.add_argument("--sample_idx", type=int, default=0, help="Which sample in the batch to plot")
    parser.add_argument("--vars", type=int, nargs="+", default=None, help="Variable indices to plot")
    parser.add_argument("--save", type=str, default=None, help="Save path for figure; shows if omitted")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    model = build_model(cfg, args.ckpt, args.device)

    x = load_dataset_samples(cfg, args.dataset, args.n_samples, args.device)
    n_var = x.shape[-1]
    var_indices = args.vars if args.vars is not None else list(range(min(4, n_var)))

    x_model = x.transpose(1, 2).contiguous()
    with torch.no_grad():
        out = model(x_model)
        recon = out["recon"].transpose(1, 2).contiguous()

    print(f"\nOverall reconstruction MSE: {torch.nn.functional.mse_loss(recon, x).item():.6f}")
    print(f"Latent mu shape: {tuple(out['mu'].shape)}")
    print(f"Latent z shape: {tuple(out['z'].shape)}")

    plot_reconstruction(x, recon, out["mu"], args.sample_idx, var_indices, args.save)
    plot_mse_summary(x, recon, args.save)


if __name__ == "__main__":
    main()
