# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A minimal training script for SiT using PyTorch DDP.
"""
from collections import defaultdict, OrderedDict
import torch
from contextlib import nullcontext

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

# from qwen_vl_utils import process_vision_info
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
    parser.add_argument("--precomputed-dir", type=str, default=None,
                        help="Optional directory containing precomputed vision embedding shards like train_rank0.pt.")
    # parser.add_argument("--jsonl-path", type=str, required=True)
    # parser.add_argument("--num-seg", type=int, required=True)
    parser.add_argument("--num-ch", type=int, required=True)

    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode and model.forward).")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    return args


def batch_to_latents(batch, device, vision_encoder=None):
    if "z_per_channel" in batch:
        z = batch["z_per_channel"].to(device=device, dtype=torch.float32, non_blocking=True)
    else:
        if vision_encoder is None:
            raise ValueError("vision_encoder must be provided when using raw images.")
        batch = {
            k: v.to(device) if k in ["pixel_values", "image_grid_thw"] else v
            for k, v in batch.items()
        }
        with torch.no_grad():
            z = vision_encoder.encode_images(batch["pixel_values"], batch["image_grid_thw"])
            z = z.to(torch.float32)

        B_ts = batch["batch_size_ts"]
        C = batch["num_ch"]
        D, H, W = z.shape[1:]
        z = z.view(B_ts, C, D, H, W)

    B_ts = batch["batch_size_ts"]
    C = batch["num_ch"]
    D, H, W = z.shape[2], z.shape[3], z.shape[4]
    z = z.permute(0, 2, 1, 3, 4)
    z = z.reshape(B_ts, D, C * H, W)
    return z


def main():
    """Trains a new SiT model using config-driven hyperparameters."""
    args = parse_args()

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("device_count =", torch.cuda.device_count())
    print("cuda_available =", torch.cuda.is_available())
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

    transport_params = dict(transport_cfg.get("params", {}))
    transport_params.pop("time_dist_shift", None)

    # def guidance_value(key: str, default: float) -> float:
    #     if key in guidance_cfg:
    #         return guidance_cfg[key]
    #     dashed_key = key.replace("_", "-")
    #     return guidance_cfg.get(dashed_key, default)

    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)

    print("creating model")
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    if args.compile:
        try:
            model.forward = torch.compile(model.forward)
        except:
            print('MODEL FORWARD compile meets error, falling back to no compile')

    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True)  # train stage2 model
    print("DDP-ing model")
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    model = ddp_model.module
    ddp_model.train()
    print("finish DDP-ing model")
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count / 1e6:.2f}M")

    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)

    print("preparing data")
    if args.precomputed_dir:
        loader, sampler = prepare_precomputed_dataloader_one_per_channel(
            precomputed_dir=args.precomputed_dir,
            num_ch=args.num_ch,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=True,
        )
        vision_encoder = None
        print(f"finish preparing precomputed data from {args.precomputed_dir}")
    else:
        loader, sampler = prepare_dataloader_one_per_channel(
            image_path=args.image_path,
            vlm_name=misc['vlm_name'],
            num_ch=args.num_ch,
            batch_size=micro_batch_size,
            workers=num_workers, rank=rank,
            world_size=world_size
        )
        print("finish preparing data")
        print("preparing vision encoder")
        vision_encoder = Qwen3VisionEncoder(
            model_name=misc['vlm_name'],
        ).eval().to(device)
        print("finish preparing vision encoder")



    loader_batches = len(loader)
    # if loader_batches % grad_accum_steps != 0:
    #     raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")

    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    #### Transport init
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    running_loss = 0.0

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0


    if rank == 0:
        save_worktree(experiment_dir, full_cfg)
        logger.info(f"Saved training worktree and config to {experiment_dir}.")

    ### Logging experiment details
    if rank == 0:
        # num_params = sum(p.numel() for p in rae.parameters())
        # logger.info(f"Stage-1 RAE parameters: {num_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Stage-2 Model parameters: {num_params / 1e6:.2f}M")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    print(f"[R{rank}] before barrier", flush=True)
    dist.barrier()
    print(f"[R{rank}] after barrier", flush=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        # optimizer.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0 and epoch > 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            # save_checkpoint(
            #     ckpt_path,
            #     global_step,
            #     epoch,
            #     ddp_model,
            #     ema_model,
            #     optimizer,
            #     scheduler,
            # )
        for step, batch in tqdm(enumerate(loader)):
            z = batch_to_latents(batch, device, vision_encoder)
            optimizer.zero_grad(set_to_none=True)
            with autocast(**autocast_kwargs):
                loss = transport.training_losses(ddp_model, z)["loss"].mean()
            loss = loss.float()
            if scaler:
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                (loss / grad_accum_steps).backward()
            if clip_grad:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            if global_step % grad_accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                update_ema(ema_model, ddp_model.module, decay=ema_decay)
            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval  # flow loss often has large variance so we record avg loss
                steps = torch.tensor(log_interval, device=device)
                stats = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb and rank == 0:
                    wandb_utils.log(
                        stats,
                        step=global_step,
                    )
                running_loss = 0.0

            global_step += 1
            num_batches += 1

        optimizer.zero_grad(set_to_none=True)

        for step, batch in tqdm(enumerate(loader), total=len(loader)):

            z = batch_to_latents(batch, device, vision_encoder)

            do_step = (step + 1) % grad_accum_steps == 0
            sync_context = nullcontext() if do_step else ddp_model.no_sync()

            with sync_context:

                with autocast(**autocast_kwargs):
                    loss = transport.training_losses(ddp_model, z)["loss"].mean()

                loss = loss / grad_accum_steps

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if do_step:

                if clip_grad:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler:
                    scheduler.step()

                update_ema(ema_model, ddp_model.module, decay=ema_decay)

                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum_steps
            epoch_metrics["loss"] += loss.detach() * grad_accum_steps

            global_step += 1
            num_batches += 1

        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_stats = {
                "epoch/loss": avg_loss, "epoch/lr": current_lr
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb and rank == 0:
                wandb_utils.log(
                    epoch_stats,
                    step=global_step
                )
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
