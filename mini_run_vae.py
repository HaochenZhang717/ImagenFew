import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_provider.datasets.verbal_ts import Dataset_VerbalTS


# =========================
# Residual Block
# =========================
class ResBlock1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


# =========================
# Encoder (downsample ×16)
# =========================
class TimeSeriesEncoderCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        latent_dim=64,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # T → T/2
            nn.Conv1d(input_dim, hidden_size // 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),

            # T/2 → T/4
            nn.Conv1d(hidden_size // 4, hidden_size // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),

            # T/4 → T/8
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # T/8 → T/16
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # feature refine
            ResBlock1D(hidden_size),
            ResBlock1D(hidden_size),
        )

        self.to_mu = nn.Conv1d(hidden_size, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(hidden_size, latent_dim, kernel_size=1)

    def forward(self, x):
        x = self.net(x)  # (B, hidden, T/16)

        mu = self.to_mu(x)
        logvar = self.to_logvar(x)

        mu = mu.permute(0, 2, 1)        # (B, T/16, latent)
        logvar = logvar.permute(0, 2, 1)

        return mu, logvar


# =========================
# Decoder (upsample ×16)
# =========================
class TimeSeriesDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(latent_dim, hidden_size, kernel_size=1)

        self.net = nn.Sequential(
            ResBlock1D(hidden_size),

            # T/16 → T/8
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # T/8 → T/4
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),

            # T/4 → T/2
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden_size // 2, hidden_size // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),

            # T/2 → T
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(hidden_size // 2, hidden_size // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),

            nn.Conv1d(hidden_size // 4, output_dim, kernel_size=5, padding=2),
        )

    def forward(self, z):
        x = self.input_proj(z)
        x = self.net(x)
        return x


# =========================
# VAE
# =========================
class TimeSeriesVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=128,
        latent_dim=64,
        beta=0.001,
    ):
        super().__init__()

        self.beta = beta

        self.encoder = TimeSeriesEncoderCNN(
            input_dim=input_dim,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
        )

        self.decoder = TimeSeriesDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z.permute(0, 2, 1))

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x)

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


def _as_vae_input(batch, device):
    if isinstance(batch, dict):
        x = batch.get("ts", batch.get("data"))
    elif isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch

    if x is None:
        raise ValueError("Could not find a time-series tensor in the batch.")

    x = x.to(device=device, dtype=torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(f"Expected batch with shape [B, T, C] or [B, C, T], got {tuple(x.shape)}")

    # VerbalTSDatasets return [B, T, C]; Conv1d expects [B, C, T].
    return x.transpose(1, 2).contiguous()


def _match_recon_length(recon, x):
    if recon.shape[-1] == x.shape[-1]:
        return recon
    if recon.shape[-1] > x.shape[-1]:
        return recon[..., : x.shape[-1]]
    return F.pad(recon, (0, x.shape[-1] - recon.shape[-1]))


def _run_vae_epoch(model, dataloader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}
    num_batches = 0

    for batch in dataloader:
        x = _as_vae_input(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        out = model(x)
        recon = _match_recon_length(out["recon"], x)
        loss_dict = model.loss_function(x, recon, out["mu"], out["logvar"])

        if is_train:
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for key in totals:
            totals[key] += loss_dict[key].detach().item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Dataloader is empty; cannot train VAE.")

    return {key: value / num_batches for key, value in totals.items()}


@torch.no_grad()
def export_vae_latents(
    model,
    datasets_dir,
    rel_path,
    save_dir,
    batch_size,
    num_workers,
    device,
    scale=True,
):
    """Encode train/valid/test splits with the VAE encoder and save posterior means."""

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = {}
    for split in ("train", "valid", "test"):
        dataset = Dataset_VerbalTS(
            root_path=datasets_dir,
            rel_path=rel_path,
            flag=split,
            scale=scale,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        latent_chunks = []
        for batch in dataloader:
            x = _as_vae_input(batch, device)
            mu, _ = model.encoder(x)
            latent_chunks.append(mu.detach().cpu().numpy())

        latents = np.concatenate(latent_chunks, axis=0)
        save_path = os.path.join(save_dir, f"{split}_mini_vae_latent.npy")
        np.save(save_path, latents.astype(np.float32))
        saved_paths[split] = save_path
        print(f"Saved {split} latents: {save_path} {latents.shape}")

    return saved_paths


@torch.no_grad()
def visualize_test_reconstructions(
    model,
    datasets_dir,
    rel_path,
    save_dir,
    num_samples,
    device,
    scale=True,
):
    """Save original-vs-reconstruction plots for a few test samples."""

    os.makedirs(save_dir, exist_ok=True)
    mpl_cache_dir = os.path.join(save_dir, ".matplotlib")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", mpl_cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()

    dataset = Dataset_VerbalTS(
        root_path=datasets_dir,
        rel_path=rel_path,
        flag="test",
        scale=scale,
    )
    num_samples = min(num_samples, len(dataset))
    if num_samples <= 0:
        return []

    x = torch.stack([dataset[idx] for idx in range(num_samples)], dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    if x.ndim == 2:
        x = x.unsqueeze(-1)

    x_model = x.transpose(1, 2).contiguous()
    mu, _ = model.encoder(x_model)
    recon = model.decoder(mu.permute(0, 2, 1))
    recon = _match_recon_length(recon, x_model).transpose(1, 2).contiguous()

    x_plot = x.detach().cpu()
    recon_plot = recon.detach().cpu()
    if scale:
        x_plot = dataset.inverse_transform(x_plot)
        recon_plot = dataset.inverse_transform(recon_plot)

    x_np = x_plot.numpy()
    recon_np = recon_plot.numpy()

    saved_paths = []
    for sample_idx in range(num_samples):
        num_channels = x_np.shape[-1]
        rows = min(num_channels, 4)
        fig, axes = plt.subplots(
            rows,
            1,
            figsize=(10, 2.6 * rows),
            squeeze=False,
            sharex=True,
        )

        for row in range(rows):
            ax = axes[row][0]
            ax.plot(x_np[sample_idx, :, row], label="original", linewidth=1.8)
            ax.plot(recon_np[sample_idx, :, row], label="reconstruction", linewidth=1.5, linestyle="--")
            ax.set_ylabel(f"ch {row}")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.legend(loc="best")

        axes[-1][0].set_xlabel("time")
        fig.suptitle(f"Test reconstruction sample {sample_idx}")
        fig.tight_layout()

        save_path = os.path.join(save_dir, f"test_reconstruction_{sample_idx:03d}.png")
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
        saved_paths.append(save_path)
        print(f"Saved reconstruction plot: {save_path}")

    return saved_paths


def train_vae(
    datasets_dir="./data/",
    rel_path="VerbalTSDatasets/istanbul_traffic",
    save_dir="./logs/mini_vae",
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=0.0,
    hidden_size=128,
    latent_dim=64,
    beta=0.001,
    num_workers=0,
    device=None,
    scale=True,
    wandb_project="MiniTimeSeriesVAE",
    wandb_run_name=None,
    use_wandb=True,
    export_latents=True,
    latent_save_dir=None,
    visualize_reconstructions=True,
    num_recon_plots=8,
    recon_save_dir=None,
):
    """Train TimeSeriesVAE on a VerbalTSDatasets split and save the best validation checkpoint."""

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = Dataset_VerbalTS(
        root_path=datasets_dir,
        rel_path=rel_path,
        flag="train",
        scale=scale,
    )
    val_dataset = Dataset_VerbalTS(
        root_path=datasets_dir,
        rel_path=rel_path,
        flag="valid",
        scale=scale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    sample = train_dataset[0]
    if sample.ndim == 1:
        input_dim = 1
    else:
        input_dim = sample.shape[-1]
    seq_len = sample.shape[0]

    scaler_state = None
    if scale:
        scaler_state = {
            "mean": train_dataset.scaler.mean_.tolist(),
            "scale": train_dataset.scaler.scale_.tolist(),
            "var": train_dataset.scaler.var_.tolist(),
        }

    model = TimeSeriesVAE(
        input_dim=input_dim,
        output_dim=input_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        beta=beta,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    config = {
        "datasets_dir": datasets_dir,
        "rel_path": rel_path,
        "save_dir": save_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "beta": beta,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "scale": scale,
        "device": str(device),
        "export_latents": export_latents,
        "latent_save_dir": latent_save_dir,
        "visualize_reconstructions": visualize_reconstructions,
        "num_recon_plots": num_recon_plots,
        "recon_save_dir": recon_save_dir,
    }

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
        )

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(save_dir, "best.pt")
    last_ckpt_path = os.path.join(save_dir, "last.pt")

    for epoch in range(1, epochs + 1):
        train_metrics = _run_vae_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_metrics = _run_vae_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                optimizer=None,
            )

        log_payload = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/recon_loss": train_metrics["recon_loss"],
            "train/kl_loss": train_metrics["kl_loss"],
            "val/loss": val_metrics["loss"],
            "val/recon_loss": val_metrics["recon_loss"],
            "val/kl_loss": val_metrics["kl_loss"],
        }

        if wandb_run is not None:
            wandb.log(log_payload, step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
            "scaler": scaler_state,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, last_ckpt_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_ckpt_path)
            if wandb_run is not None:
                wandb_run.summary["best_val_loss"] = best_val_loss
                wandb_run.summary["best_epoch"] = epoch
                wandb.save(best_ckpt_path)

        print(
            f"Epoch {epoch:04d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"best_val_loss={best_val_loss:.6f}"
        )

    latent_paths = {}
    recon_plot_paths = []
    if export_latents or visualize_reconstructions:
        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")

        best_checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])

    if export_latents:
        latent_paths = export_vae_latents(
            model=model,
            datasets_dir=datasets_dir,
            rel_path=rel_path,
            save_dir=latent_save_dir or os.path.join(datasets_dir, rel_path),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            scale=scale,
        )

        if wandb_run is not None:
            for split, path in latent_paths.items():
                wandb.save(path)

    if visualize_reconstructions:
        recon_plot_paths = visualize_test_reconstructions(
            model=model,
            datasets_dir=datasets_dir,
            rel_path=rel_path,
            save_dir=recon_save_dir or os.path.join(save_dir, "reconstructions"),
            num_samples=num_recon_plots,
            device=device,
            scale=scale,
        )

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "test/reconstructions": [
                        wandb.Image(path) for path in recon_plot_paths
                    ]
                }
            )
            for path in recon_plot_paths:
                wandb.save(path)

    if wandb_run is not None:
        wandb.finish()

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "latent_paths": latent_paths,
        "recon_plot_paths": recon_plot_paths,
    }


def get_args():
    parser = argparse.ArgumentParser(description="Train TimeSeriesVAE on VerbalTSDatasets.")

    parser.add_argument("--datasets_dir", type=str, default="./data/")
    parser.add_argument("--rel_path", type=str, default="VerbalTSDatasets/istanbul_traffic")
    parser.add_argument("--save_dir", type=str, default="./logs/mini_vae")

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.001)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_scale", action="store_true", help="Disable train-set StandardScaler normalization.")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="MiniTimeSeriesVAE")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--no_export_latents", action="store_true", help="Skip exporting train/valid/test VAE latents.")
    parser.add_argument(
        "--latent_save_dir",
        type=str,
        default=None,
        help="Directory for *_mini_vae_latent.npy files. Defaults to the dataset directory.",
    )
    parser.add_argument("--no_visualize_reconstructions", action="store_true", help="Skip test reconstruction plots.")
    parser.add_argument("--num_recon_plots", type=int, default=8, help="Number of test reconstruction plots to save.")
    parser.add_argument(
        "--recon_save_dir",
        type=str,
        default=None,
        help="Directory for reconstruction plots. Defaults to save_dir/reconstructions.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    train_vae(
        datasets_dir=args.datasets_dir,
        rel_path=args.rel_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
        num_workers=args.num_workers,
        device=args.device,
        scale=not args.no_scale,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=args.wandb,
        export_latents=not args.no_export_latents,
        latent_save_dir=args.latent_save_dir,
        visualize_reconstructions=not args.no_visualize_reconstructions,
        num_recon_plots=args.num_recon_plots,
        recon_save_dir=args.recon_save_dir,
    )


if __name__ == "__main__":
    main()
