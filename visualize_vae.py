"""
Quick visualization of MultiScaleVAE reconstruction.

Loads model architecture + dataset list from a YAML config (e.g.
configs/pretrain/vae_pretrain.yaml), loads a checkpoint, then reconstructs
a batch from one of the configured datasets and plots the result.

Usage:
    python visualize_vae.py --config configs/pretrain/vae_pretrain.yaml \
                            --ckpt <path> --dataset ETTh1 --sample_idx 0
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.MultiScaleVAE.multiscale_vae import DualVAE
from data_provider.data_provider import get_train, dataset_to_tensor, random_permute


# ---------------------------------------------------------------------------
# Config / model / data
# ---------------------------------------------------------------------------

def load_config(config_path, overrides):
    """Load YAML via OmegaConf and apply CLI overrides."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    # Turn it into a namespace-ish object for attribute access.
    return OmegaConf.to_object(cfg)


def build_model(cfg, ckpt_path, device):
    model = DualVAE(
        z_channels   = cfg["z_channels"],
        ch           = cfg["unet_channels"],
        ch_mult      = tuple(cfg["ch_mult"]),
        dynamic_size = cfg["dynamic_size"],
        dropout      = cfg.get("dropout", 0.0),
        test_mode    = True,
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
        print("[warn] No checkpoint loaded — showing an untrained model.")

    model.eval()
    return model


class _ArgsShim:
    """Tiny object the data_provider helpers expect (it reads attributes)."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def load_dataset_samples(cfg, dataset_name, n_samples, device):
    """Use the project's data_provider helpers to get a single dataset tensor."""
    # Find the dataset entry in the config
    ds_cfg = next((d for d in cfg["datasets"] if d["name"] == dataset_name), None)
    if ds_cfg is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in config. Available: "
            f"{[d['name'] for d in cfg['datasets']]}"
        )

    ds_cfg = dict(ds_cfg)
    ds_cfg["seq_len"]      = cfg["seq_len"]
    ds_cfg["datasets_dir"] = cfg["datasets_dir"]

    trainset = get_train(ds_cfg)
    testset  = get_train(ds_cfg)
    trainset, _ = random_permute(trainset, testset)

    args_shim = _ArgsShim({
        "batch_size":  min(cfg.get("batch_size", 64), max(n_samples, 1)),
        "num_workers": 0,
        "seq_len":     cfg["seq_len"],
    })
    tensor = dataset_to_tensor(trainset, args_shim)
    tensor = tensor[:n_samples].to(device).float()
    print(f"Loaded {tensor.shape[0]} samples from '{dataset_name}'  (shape={tuple(tensor.shape)})")
    return tensor


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstruction(orig, out, sample_idx, var_indices, save_path):
    n_vars = len(var_indices)
    n_rows = 5
    fig = plt.figure(figsize=(4 * n_vars, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_vars, hspace=0.45, wspace=0.3)

    rows = [
        ("input vs total recon", "input",     "total_recon"),
        ("low-freq",              "low_freq",  "recon_low_freq"),
        ("mid-freq",              "mid_freq",  "recon_mid_freq"),
        ("high-freq",             "high_freq", "recon_high_freq"),
    ]

    def to_np(t, var):
        return t[sample_idx, :, var].cpu().numpy()

    for col, var in enumerate(var_indices):
        # row 0: input vs total recon
        ax = fig.add_subplot(gs[0, col])
        ax.plot(to_np(orig, var),                 label="original", lw=1.5)
        ax.plot(to_np(out["total_recon"], var),   label="recon",    lw=1.5, ls="--")
        ax.set_title(f"Var {var} — input vs recon", fontsize=9)
        ax.legend(fontsize=7)

        # rows 1-3: decomposed components
        for row, (title, tgt_key, rec_key) in enumerate(rows[1:], start=1):
            ax = fig.add_subplot(gs[row, col])
            ax.plot(to_np(out[tgt_key], var), label="target", lw=1.5)
            ax.plot(to_np(out[rec_key], var), label="recon",  lw=1.5, ls="--")
            ax.set_title(f"Var {var} — {title}", fontsize=9)
            ax.legend(fontsize=7)

        # row 4: residual
        ax = fig.add_subplot(gs[4, col])
        residual = to_np(orig, var) - to_np(out["total_recon"], var)
        ax.plot(residual, color="red", lw=1.2)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(f"Var {var} — residual", fontsize=9)

    fig.suptitle(f"MultiScale VAE — sample {sample_idx}", fontsize=12, y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_mse_summary(orig, out, save_path):
    n_var = orig.shape[-1]
    mse = ((orig - out["total_recon"]) ** 2).mean(dim=(0, 1)).cpu().numpy()

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to YAML config (e.g. configs/pretrain/vae_pretrain.yaml)")
    parser.add_argument("--ckpt",       type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--dataset",    type=str, required=True,
                        help="Which dataset in the config to visualize (e.g. ETTh1)")
    parser.add_argument("--n_samples",  type=int, default=8,
                        help="How many samples to load")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Which of the loaded samples to plot in detail")
    parser.add_argument("--vars",       type=int, nargs="+", default=None,
                        help="Variable indices to plot (default: first 4)")
    parser.add_argument("--save",       type=str, default=None,
                        help="Save path for the figure. Shows interactively if omitted.")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--override",   type=str, nargs="*", default=[],
                        help='Dot-list overrides for the config, e.g. z_channels=32')
    args = parser.parse_args()

    # --- Config & model ---
    cfg    = load_config(args.config, args.override)
    model  = build_model(cfg, args.ckpt, args.device)

    # --- Data ---
    x = load_dataset_samples(cfg, args.dataset, args.n_samples, args.device)
    n_var = x.shape[-1]
    var_indices = args.vars if args.vars is not None else list(range(min(4, n_var)))

    # --- Forward ---
    with torch.no_grad():
        out = model(x)

    print(f"\nOverall reconstruction MSE : {torch.nn.functional.mse_loss(out['total_recon'], x).item():.6f}")
    print(f"  low_freq  recon MSE      : {out['recon_loss_low_freq'].item():.6f}")
    print(f"  mid_freq  recon MSE      : {out['recon_loss_mid_freq'].item():.6f}")
    print(f"  high_freq recon MSE      : {out['recon_loss_high_freq'].item():.6f}")

    # --- Plots ---
    plot_reconstruction(x, out, args.sample_idx, var_indices, args.save)
    plot_mse_summary(x, out, args.save)


if __name__ == "__main__":
    main()