# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Sample new images from a pre-trained SiT.
"""
import torch.nn as nn
import math
from time import time
import argparse
from utils.model_utils import instantiate_from_config
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from torchvision.utils import save_image
import torch
import sys
import os
from stage2.models import Stage2ModelProtocol
from stage1_model import Stage1_Qwen3VLForConditionalGeneration
import json
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config, transport_config, sampler_config, guidance_config, misc, _ = parse_configs(args.config)
    model_config.update({"ckpt": args.ckpt})
    vlm = Stage1_Qwen3VLForConditionalGeneration.from_pretrained(
        misc['vlm_name'],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        device_map=None
    ).eval().to(device)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()  # important!

    shift_dim = misc.get("time_dist_shift_dim", 768 * 16 * 16)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"Using time_dist_shift={time_dist_shift:.4f} = sqrt({shift_dim}/{shift_base}).")
    transport = create_transport(
        **transport_config['params'],
        time_dist_shift=time_dist_shift
    )
    sampler = Sampler(transport)
    mode, sampler_params = sampler_config['mode'], sampler_config['params']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            **sampler_params
        )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            **sampler_params,
            # sampling_method=args.sampling_method,
            # diffusion_form=args.diffusion_form,
            # diffusion_norm=args.diffusion_norm,
            # last_step=args.last_step,
            # last_step_size=args.last_step_size,
            # num_steps=args.num_sampling_steps,
        )
    else:
        raise NotImplementedError(f"Invalid sampling mode {mode}.")
    
    latent_size = misc.get("latent_size", (768, 16, 16))

    # Create sampling noise:
    # num_samples = 3072
    num_samples = args.num_samples
    n = 64
    num_batches = math.ceil(num_samples / n)

    samples_list = []
    with open(args.output_jsonl, "w") as f:
        for batch_id in range(num_batches):
            z = torch.randn(n, *latent_size, device=device)
            # set guidance setup
            model_kwargs = dict()
            model_fwd = model.forward
            # Sample images:
            start_time = time()
            samples:torch.Tensor = sample_fn(z, model_fwd, **model_kwargs)[-1]
            # breakpoint()
            samples = samples.contiguous().view(n, 1152, -1).permute(0, 2, 1).contiguous().view(-1, 1152)
            decoded_samples = vlm.decode(samples,image_size=450)
            print(f"Sampling took {time() - start_time:.2f} seconds.")
            for i, cap in enumerate(decoded_samples):
                record = {
                    "batch_id": batch_id,
                    "sample_id": batch_id * n + i,
                    "caption": cap
                }
                f.write(json.dumps(record) + "\n")
                samples_list.append(cap)
            f.flush()

    samples_array = np.array(samples_list)
    np.save(args.output_array, samples_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--output_array", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_known_args()[0]
    main(args)
