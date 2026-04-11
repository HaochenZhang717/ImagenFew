# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A minimal training script for SiT using PyTorch DDP.
"""
from collections import defaultdict, OrderedDict
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.nn.parallel import DistributedDataParallel as DDP
from time import time
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy

##### model imports
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *

from qwen_vl_utils import process_vision_info
from typing import Dict
from stage1_model import Qwen3VisionEncoder
from tqdm import tqdm


def save_checkpoint(
        path: str,
        step: int,
        epoch: int,
        model: DDP,
        ema_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
        path: str,
        model: DDP,
        ema_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on VLM vision encoder latents.")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--num-ch", type=int, required=True)
    parser.add_argument("--save-path", type=str, required=True)

    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode and model.forward).")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    return args


def main():
    """Trains a new SiT model using config-driven hyperparameters."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    rank, world_size, device = setup_distributed()
    full_cfg = OmegaConf.load(args.config)
    (
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config
    ) = parse_configs(full_cfg)

    if model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    # num_classes = int(misc.get("num_classes", 1000))
    # null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None)  # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 2500))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4))  # ckpt interval is epoch based
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))

    # here we do not do online evaluation
    do_eval = False

    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    autocast_enabled = use_fp16 or use_bf16
    # autocast_kwargs = dict(dtype=autocast_dtype, enabled=autocast_enabled)
    # scaler = GradScaler(enabled=use_fp16)

    transport_params = dict(transport_cfg.get("params", {}))
    # path_type = transport_params.get("path_type", "Linear")
    # prediction = transport_params.get("prediction", "velocity")
    # loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)


    vision_encoder = Qwen3VisionEncoder(
        model_name=misc['vlm_name'],
    ).eval().to(device)


    loader, sampler = prepare_dataloader_one_per_channel(
        image_path=args.image_path,
        vlm_name=misc['vlm_name'],
        num_ch=args.num_ch,
        batch_size=micro_batch_size,
        workers=num_workers, rank=rank,
        world_size=world_size,
        shuffle=False
    )

    loader_batches = len(loader)
    if loader_batches <= 0:
        raise ValueError("No batches were produced by the dataloader.")


    embedding_dict = {}

    for step, batch in enumerate(loader):

        if rank == 0:
            print(f"[Rank0] Step {step}/{len(loader)}")

        batch = {
            k: v.to(device) if k in ["pixel_values", "image_grid_thw"] else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            z = vision_encoder.encode_images(
                batch["pixel_values"],
                batch["image_grid_thw"]
            )  # shape: (B_ts * num_ch, D, H, W)

        z = z.cpu()

        image_names_by_sample = batch["image_names"]
        flat_image_names = [
            image_name
            for sample_image_names in image_names_by_sample
            for image_name in sample_image_names
        ]

        if len(flat_image_names) != z.shape[0]:
            raise ValueError(
                f"Mismatch between flattened image names ({len(flat_image_names)}) "
                f"and encoded embeddings ({z.shape[0]})."
            )

        for i, name in enumerate(flat_image_names):
            embedding_dict[name] = z[i]

    save_path = args.save_path.replace(".pt", f"_rank{rank}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(embedding_dict, save_path)

    print(f"[RANK {rank}] Saved {len(embedding_dict)} embeddings")
    torch.distributed.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
