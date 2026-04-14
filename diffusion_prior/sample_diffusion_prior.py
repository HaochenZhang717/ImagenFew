import argparse
import os
import sys

import torch
from omegaconf import OmegaConf


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from diffusion_prior.models import DiT1D, ResNet1D
from diffusion_prior.models.transport import Sampler
from diffusion_prior.models.transport import create_transport


def build_model_from_ckpt(state, device):
    model_args = state["model_args"]
    backbone = model_args.get("backbone", "dit1d").lower()
    if backbone == "dit1d":
        model = DiT1D(
            seq_len=state["seq_len"],
            token_dim=state["token_dim"],
            hidden_size=model_args["hidden_size"],
            depth=model_args["depth"],
            num_heads=model_args["num_heads"],
            mlp_ratio=model_args["mlp_ratio"],
            use_qknorm=model_args["use_qknorm"],
            use_rmsnorm=model_args["use_rmsnorm"],
        ).to(device)
    elif backbone in {"resnet1d", "cnn1d", "residual1d"}:
        model = ResNet1D(
            seq_len=state["seq_len"],
            token_dim=state["token_dim"],
            hidden_size=model_args["hidden_size"],
            depth=model_args["depth"],
            kernel_size=model_args.get("kernel_size", 3),
            use_rmsnorm=model_args["use_rmsnorm"],
            dropout=model_args.get("dropout", 0.0),
        ).to(device)
    else:
        raise ValueError(f"Unsupported backbone {backbone}")
    model.load_state_dict(state["model"])
    model.eval()
    return model


def maybe_unnormalize_latents(samples, state):
    latent_norm = state.get("latent_norm", None)
    if not latent_norm or not latent_norm.get("enabled", False):
        return samples

    mean = latent_norm["mean"].to(samples.device)
    std = latent_norm["std"].to(samples.device)
    return samples * std.unsqueeze(0) + mean.unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", type=str)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.config is not None:
        config = OmegaConf.to_object(OmegaConf.load(args.config))
        parser.set_defaults(**config)
        args = parser.parse_args()

    defaults = {
        "method": "dopri5",
        "num_steps": 50,
        "atol": 1e-6,
        "rtol": 1e-3,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.ckpt is None or args.output is None or args.num_samples is None:
        raise ValueError("Need --ckpt, --output, and --num-samples either via CLI or config.")

    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    state = torch.load(args.ckpt, map_location=args.device)
    model = build_model_from_ckpt(state, args.device)

    transport = create_transport(**state["transport_args"])
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=args.method,
        num_steps=args.num_steps,
        atol=args.atol,
        rtol=args.rtol,
    )

    init = torch.randn(
        args.num_samples,
        state["seq_len"],
        state["token_dim"],
        device=args.device,
    )
    with torch.no_grad():
        xs = sample_fn(init, model)
    samples = maybe_unnormalize_latents(xs[-1], state).detach().cpu()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(samples, args.output)
    print(f"Saved samples to {args.output} with shape {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
