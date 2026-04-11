# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Multi-GPU sampling for a pre-trained SiT-style Stage2 diffusion prior + Stage1 VLM decoder.

Run with:
  torchrun --nproc_per_node=NUM_GPUS sample_multi_gpu.py \
    --config path/to/config.yaml \
    --ckpt path/to/ep-last.pt \
    --num_samples 1024 \
    --seed 0 \
    --output_jsonl /playpen/haochenz/diffusion_prior_results/DiTDH-S-samples.jsonl \
    --output_array /playpen/haochenz/diffusion_prior_results/DiTDH-S-samples.npy
"""

import math
import os
import sys
import json
import argparse
from time import time

import numpy as np
import torch
import torch.distributed as dist

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage2.transport import create_transport, Sampler
from stage2.models import Stage2ModelProtocol
from stage1_model import Stage1_Qwen3VLForConditionalGeneration


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _init_distributed():
    """
    Initializes torch.distributed if launched via torchrun.
    Returns (rank, world_size, local_rank, device).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        distributed = True
    else:
        # Fallback to single process
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False

    return distributed, rank, world_size, local_rank, device


def _mkdir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _ranked_path(base_path: str, rank: int, world_size: int, ext: str):
    """
    Convert:
      /a/b/out.jsonl -> /a/b/out.rank00000-of00004.jsonl
      /a/b/out.npy   -> /a/b/out.rank00000-of00004.npy
    """
    if not base_path.endswith(ext):
        base_path = base_path + ext
    stem = base_path[:-len(ext)]
    return f"{stem}.rank{rank:05d}-of{world_size:05d}{ext}"


def main(args):
    distributed, rank, world_size, local_rank, device = _init_distributed()

    # Seed: make per-rank different but reproducible
    torch.manual_seed(args.seed + rank)
    torch.set_grad_enabled(False)

    # Parse configs
    model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)
    model_config.update({"ckpt": args.ckpt})

    # Load Stage1 VLM decoder (each rank loads its own copy; simple + robust for inference)
    vlm = (
        Stage1_Qwen3VLForConditionalGeneration.from_pretrained(
            misc["vlm_name"],
            torch_dtype=torch.bfloat16,  # note: transformers warns torch_dtype deprecated in some contexts, but ok here
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            device_map=None,
        )
        .eval()
        .to(device)
    )

    # Load Stage2 diffusion prior
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()

    # Build sampler
    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if rank == 0:
        print(f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")

    transport = create_transport(**transport_config["params"], time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config["mode"], sampler_config["params"]

    if mode == "ODE":
        sample_fn = sampler.sample_ode(**sampler_params)
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")

    latent_size = misc.get("latent_size", (768, 16, 16))

    # ----------- Multi-GPU work split -----------
    # We generate exactly args.num_samples total, split across ranks (nearly evenly).
    total = int(args.num_samples)
    # A simple contiguous split:
    # rank r handles indices [start, end)
    base = total // world_size
    rem = total % world_size
    my_count = base + (1 if rank < rem else 0)
    my_start = rank * base + min(rank, rem)
    my_end = my_start + my_count

    if rank == 0:
        print(f"Total samples={total}, world_size={world_size}.")
    print(f"[rank {rank}/{world_size}] will generate sample_id in [{my_start}, {my_end}) (count={my_count}).")

    # Batch size per rank
    n = int(args.batch_size)
    num_batches = math.ceil(my_count / n) if my_count > 0 else 0

    # Per-rank output files (avoid multi-process write conflicts)
    out_jsonl_rank = _ranked_path(args.output_jsonl, rank, world_size, ".jsonl")
    out_npy_rank = _ranked_path(args.output_array, rank, world_size, ".npy")
    _mkdir_for_file(out_jsonl_rank)
    _mkdir_for_file(out_npy_rank)

    samples_list = []
    with open(out_jsonl_rank, "w") as f:
        # Generate in batches
        for batch_id in range(num_batches):
            # How many this batch actually needs (last batch may be smaller)
            remaining = my_count - batch_id * n
            bs = n if remaining >= n else remaining
            if bs <= 0:
                break

            z = torch.randn(bs, *latent_size, device=device)

            # No guidance in your original script
            model_kwargs = dict()
            model_fwd = model.forward

            start_time = time()
            samples: torch.Tensor = sample_fn(z, model_fwd, **model_kwargs)[-1]

            # Your reshape pipeline (kept identical, but respects bs instead of fixed n)
            samples = (
                samples.contiguous()
                .view(bs, 1152, -1)
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, 1152)
            )

            decoded_samples = vlm.decode(samples, image_size=450)
            print(f"[rank {rank}] batch {batch_id+1}/{num_batches} took {time() - start_time:.2f}s.")

            for i, cap in enumerate(decoded_samples):
                global_sample_id = my_start + batch_id * n + i
                record = {
                    "rank": rank,
                    "batch_id": batch_id,
                    "sample_id": int(global_sample_id),  # global id across all ranks
                    "caption": cap,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                samples_list.append(cap)

            f.flush()

    # Save per-rank numpy array
    np.save(out_npy_rank, np.array(samples_list, dtype=object))

    # ----------- Optional merge on rank0 -----------
    # We DO NOT delete per-rank files; we just create merged files for convenience.
    if distributed:
        dist.barrier()

    if rank == 0:
        merged_jsonl = args.output_jsonl if args.output_jsonl.endswith(".jsonl") else (args.output_jsonl + ".jsonl")
        merged_npy = args.output_array if args.output_array.endswith(".npy") else (args.output_array + ".npy")
        _mkdir_for_file(merged_jsonl)
        _mkdir_for_file(merged_npy)

        # Merge JSONL by concatenating then sorting by sample_id (stable, but may cost memory if huge)
        # For large outputs, you can skip sorting and just cat.
        records = []
        for r in range(world_size):
            part = _ranked_path(args.output_jsonl, r, world_size, ".jsonl")
            if not os.path.exists(part):
                continue
            with open(part, "r") as pf:
                for line in pf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        records.sort(key=lambda x: x.get("sample_id", 0))
        with open(merged_jsonl, "w") as mf:
            for rec in records:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Merge npy by rank order (already covers all captions)
        all_caps = []
        for r in range(world_size):
            part = _ranked_path(args.output_array, r, world_size, ".npy")
            if not os.path.exists(part):
                continue
            arr = np.load(part, allow_pickle=True)
            all_caps.extend(arr.tolist())

        # If you want strict global order, rebuild from JSONL order:
        # (Here we just save concatenated list; merged_jsonl is the authoritative ordered file.)
        np.save(merged_npy, np.array(all_caps, dtype=object))

        print(f"[rank0] Wrote merged JSONL: {merged_jsonl}")
        print(f"[rank0] Wrote merged NPY:  {merged_npy}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-rank batch size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_array", type=str, required=True)

    # keep parse_known_args like your original script
    args = parser.parse_known_args()[0]
    main(args)