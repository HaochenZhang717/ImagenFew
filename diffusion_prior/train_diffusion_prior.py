import argparse
import os
import sys
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.distributed import DistributedSampler


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models import create_transport, DiT1D, ResNet1D


def atomic_torch_save(state, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def setup_distributed(args):
    args.ddp = is_distributed()
    if not args.ddp:
        device = torch.device(args.device)
        return device, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device, rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in unwrap_model(model).state_dict().items()
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        for name, param in unwrap_model(model).state_dict().items():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}

    @contextmanager
    def average_parameters(self, model):
        model = unwrap_model(model)
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
    backbone = getattr(args, "backbone", "dit1d").lower()
    if backbone == "dit1d":
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
    elif backbone in {"resnet1d", "cnn1d", "residual1d"}:
        model = ResNet1D(
            seq_len=seq_len,
            token_dim=token_dim,
            hidden_size=args.hidden_size,
            depth=args.depth,
            kernel_size=args.kernel_size,
            use_rmsnorm=not args.no_rmsnorm,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported backbone {backbone}")
    return model


def load_latents(path):
    latents = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(latents):
        raise TypeError(f"Expected latent file to contain a tensor, got {type(latents)}")
    if latents.ndim != 3:
        raise ValueError(f"Expected latent tensor of shape (N, L, D), got {tuple(latents.shape)}")
    return latents.to(torch.float32)


def make_loaders(latents, batch_size, num_workers, val_ratio, seed, use_train_as_val=False):
    dataset = TensorDataset(latents)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    if use_train_as_val:
        train_dataset, val_dataset = dataset, dataset
    elif val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = None
    val_sampler = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            drop_last=False,
        )
    return train_loader, val_loader, train_sampler, val_sampler


def run_epoch(model, loader, transport, optimizer, device, ema=None, train=True):
    model.train(train)
    loss_sum = torch.zeros(1, device=device)
    sample_count = torch.zeros(1, device=device)
    for (x,) in loader:
        x = x.to(device)
        batch_size = x.shape[0]
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
        loss_sum += loss.detach() * batch_size
        sample_count += batch_size
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    if sample_count.item() == 0:
        return float("nan")
    return (loss_sum / sample_count).item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--latents", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--finetune-ckpt", type=str, default=None)
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
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--dropout", type=float)
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
    parser.add_argument("--ema-eval-every", type=int)
    parser.add_argument("--use-train-as-val", action="store_true")
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
        "backbone": "dit1d",
        "kernel_size": 3,
        "dropout": 0.0,
        "path_type": "Linear",
        "prediction": "velocity",
        "time_dist_type": "uniform",
        "time_dist_shift": 1.0,
        "wandb_project": "diffusion-prior",
        "wandb_name": None,
        "wandb_tags": [],
        "ema_decay": 0.9999,
        "ema_eval_every": None,
        "use_train_as_val": False,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.resume is None and args.resume_ckpt is not None:
        args.resume = args.resume_ckpt

    if args.latents is None or args.output_dir is None:
        raise ValueError("Both --latents and --output-dir must be provided either via CLI or config.")

    if args.ema_eval_every is None:
        args.ema_eval_every = args.save_every

    return args


def main():
    args = parse_args()
    device, rank, world_size = setup_distributed(args)
    args.device = str(device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb_run = None
    if args.wandb and is_main_process():
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=list(args.wandb_tags),
            config=vars(args),
        )

    latents = load_latents(args.latents)
    seq_len, token_dim = latents.shape[1], latents.shape[2]
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {args.batch_size} must be divisible by world_size {world_size}."
        )
    per_rank_batch_size = args.batch_size // world_size
    train_loader, val_loader, train_sampler, val_sampler = make_loaders(
        latents=latents,
        batch_size=per_rank_batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_train_as_val=bool(getattr(args, "use_train_as_val", False)),
    )

    model = build_model(args, seq_len, token_dim, device)
    if args.ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
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

    if args.resume is None and args.finetune_ckpt is None:
        auto_resume = os.path.join(args.output_dir, "diffusion_prior_latest.pt")
        if os.path.exists(auto_resume):
            args.resume = auto_resume
            if is_main_process():
                print(f"Auto-resuming from {args.resume}")

    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "ema_model" in state:
            ema.load_state_dict(state["ema_model"])
        if "rng_state" in state:
            set_rng_state(state["rng_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_val", float("inf")))
        global_step = int(state.get("global_step", 0))
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")
    elif args.finetune_ckpt is not None:
        state = torch.load(args.finetune_ckpt, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(state["model"])
        # For finetuning we want EMA to start from the same weights as the loaded model
        # instead of inheriting a potentially stale EMA from the source checkpoint.
        ema.load_state_dict(unwrap_model(model).state_dict())
        if is_main_process():
            print(f"Initialized finetuning from {args.finetune_ckpt}")

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset) if val_loader is not None else 0
    if is_main_process():
        print(
            f"Dataset split: train={train_size}, valid={val_size}, "
            f"world_size={world_size}, global_batch_size={args.batch_size}, "
            f"per_rank_batch_size={per_rank_batch_size}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_loss = run_epoch(model, train_loader, transport, optimizer, device, ema=ema, train=True)
        global_step += len(train_loader)

        val_loss = float("nan")
        if val_loader is not None:
            with ema.average_parameters(model):
                with torch.no_grad():
                    val_loss = run_epoch(model, val_loader, transport, optimizer, device, train=False)

        ema_train_loss = float("nan")
        should_eval_ema_train = (
            val_loader is None
            and args.ema_eval_every is not None
            and int(args.ema_eval_every) > 0
            and (epoch % int(args.ema_eval_every) == 0)
        )
        if should_eval_ema_train:
            with ema.average_parameters(model):
                with torch.no_grad():
                    ema_train_loss = run_epoch(model, train_loader, transport, optimizer, device, train=False)

        if is_main_process():
            msg = f"epoch {epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            if should_eval_ema_train:
                msg += f" ema_train_loss={ema_train_loss:.6f}"
            print(msg)
        if wandb_run is not None and is_main_process():
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "train/size": train_size,
                "valid/size": val_size,
            }
            if should_eval_ema_train:
                log_dict["ema_train/loss"] = ema_train_loss
            wandb_run.log(log_dict, step=epoch)

        improved = val_loader is not None and val_loss < best_val
        if improved:
            best_val = val_loss

        state = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema_model": ema.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "rng_state": get_rng_state(),
            "seq_len": seq_len,
            "token_dim": token_dim,
            "model_args": {
                "backbone": args.backbone,
                "hidden_size": args.hidden_size,
                "depth": args.depth,
                "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
                "kernel_size": args.kernel_size,
                "dropout": args.dropout,
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
        if is_main_process():
            if epoch % args.save_every == 0:
                atomic_torch_save(state, epoch_path)
            atomic_torch_save(state, latest_path)

            if improved:
                atomic_torch_save(state, os.path.join(args.output_dir, "diffusion_prior_best.pt"))

    if wandb_run is not None and is_main_process():
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
