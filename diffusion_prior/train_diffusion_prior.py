import argparse
import os
import sys
import random
from contextlib import contextmanager

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset, random_split


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from diffusion_prior.models import DiT1D
from diffusion_prior.models.transport import create_transport


def atomic_torch_save(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}

    @contextmanager
    def average_parameters(self, model):
        self.backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def build_model(args, seq_len, token_dim, device):
    model = DiT1D(
        seq_len=seq_len,
        token_dim=token_dim,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        use_qknorm=args.use_qknorm,
        use_rmsnorm=not args.no_rmsnorm,
    ).to(device)
    return model


def load_latents(path):
    latents = torch.load(path, map_location="cpu")
    if not torch.is_tensor(latents):
        raise TypeError(f"Expected latent file to contain a tensor, got {type(latents)}")
    if latents.ndim != 3:
        raise ValueError(f"Expected latent tensor of shape (N, L, D), got {tuple(latents.shape)}")
    return latents.to(torch.float32)


def make_loaders(latents, batch_size, num_workers, val_ratio, seed):
    dataset = TensorDataset(latents)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    return train_loader, val_loader


def run_epoch(model, loader, transport, optimizer, device, ema=None, train=True):
    model.train(train)
    losses = []
    for (x,) in loader:
        x = x.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        terms = transport.training_losses(model, x)
        loss = terms["loss"].mean()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
        losses.append(loss.detach().item())
    return float(np.mean(losses)) if losses else float("nan")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--latents", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--mlp-ratio", type=float)
    parser.add_argument("--use-qknorm", action="store_true")
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--path-type", type=str, choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, choices=["velocity", "noise", "score"])
    parser.add_argument("--loss-weight", type=str, default=None)
    parser.add_argument("--time-dist-type", type=str)
    parser.add_argument("--time-dist-shift", type=float)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-name", type=str)
    parser.add_argument("--wandb-tags", nargs="+", type=str)
    parser.add_argument("--ema-decay", type=float)
    args = parser.parse_args()

    if args.config is not None:
        config = OmegaConf.to_object(OmegaConf.load(args.config))
        parser.set_defaults(**config)
        args = parser.parse_args()

    defaults = {
        "epochs": 200,
        "batch_size": 256,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "val_ratio": 0.02,
        "save_every": 10,
        "hidden_size": 512,
        "depth": 8,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "path_type": "Linear",
        "prediction": "velocity",
        "time_dist_type": "uniform",
        "time_dist_shift": 1.0,
        "wandb_project": "diffusion-prior",
        "wandb_name": None,
        "wandb_tags": [],
        "ema_decay": 0.9999,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.latents is None or args.output_dir is None:
        raise ValueError("Both --latents and --output-dir must be provided either via CLI or config.")

    return args


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=list(args.wandb_tags),
            config=vars(args),
        )

    latents = load_latents(args.latents)
    seq_len, token_dim = latents.shape[1], latents.shape[2]
    train_loader, val_loader = make_loaders(
        latents=latents,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = build_model(args, seq_len, token_dim, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay)
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        time_dist_type=args.time_dist_type,
        time_dist_shift=args.time_dist_shift,
    )

    start_epoch = 1
    best_val = float("inf")
    global_step = 0

    if args.resume is not None:
        state = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema_model" in state:
            ema.load_state_dict(state["ema_model"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        global_step = int(state.get("global_step", 0))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset) if val_loader is not None else 0
    print(f"Dataset split: train={train_size}, valid={val_size}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, transport, optimizer, args.device, ema=ema, train=True)
        global_step += len(train_loader)

        val_loss = float("nan")
        if val_loader is not None:
            with ema.average_parameters(model):
                with torch.no_grad():
                    val_loss = run_epoch(model, val_loader, transport, optimizer, args.device, train=False)

        print(f"epoch {epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "valid/loss": val_loss,
                    "train/size": train_size,
                    "valid/size": val_size,
                },
                step=epoch,
            )

        improved = val_loader is not None and val_loss < best_val
        if improved:
            best_val = val_loss

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema_model": ema.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "seq_len": seq_len,
            "token_dim": token_dim,
            "model_args": {
                "hidden_size": args.hidden_size,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
                "use_qknorm": args.use_qknorm,
                "use_rmsnorm": not args.no_rmsnorm,
            },
            "transport_args": {
                "path_type": args.path_type,
                "prediction": args.prediction,
                "loss_weight": args.loss_weight,
                "time_dist_type": args.time_dist_type,
                "time_dist_shift": args.time_dist_shift,
            },
        }

        latest_path = os.path.join(args.output_dir, "diffusion_prior_latest.pt")
        epoch_path = os.path.join(args.output_dir, f"diffusion_prior_epoch_{epoch:04d}.pt")
        if epoch % args.save_every == 0:
            atomic_torch_save(state, epoch_path)
        atomic_torch_save(state, latest_path)

        if improved:
            atomic_torch_save(state, os.path.join(args.output_dir, "diffusion_prior_best.pt"))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
