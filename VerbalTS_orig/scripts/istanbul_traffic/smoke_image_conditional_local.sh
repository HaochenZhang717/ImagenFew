#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"

DATA_FOLDER="${DATA_FOLDER:-../data/VerbalTSDatasets/istanbul_traffic}"
LONGCLIP_PATH="${LONGCLIP_PATH:-/playpen-shared/haochenz/long_clip}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMG_SIZE="${IMG_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "DATA_FOLDER=$DATA_FOLDER"
echo "LONGCLIP_PATH=$LONGCLIP_PATH"
echo "DEVICE=$DEVICE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "IMG_SIZE=$IMG_SIZE"

python - <<'PY'
import copy
import os
import yaml
import numpy as np
import torch

from data import GenerationDataset
from models.image_conditional_generator import ImageConditionalGenerator

data_folder = os.environ["DATA_FOLDER"]
longclip_path = os.environ["LONGCLIP_PATH"]
device = os.environ["DEVICE"]
batch_size = int(os.environ["BATCH_SIZE"])
img_size = int(os.environ["IMG_SIZE"])
num_workers = int(os.environ["NUM_WORKERS"])

if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"Requested {device}, but torch.cuda.is_available() is False")
if not os.path.isdir(data_folder):
    raise FileNotFoundError(f"DATA_FOLDER does not exist: {data_folder}")
if not os.path.isdir(longclip_path):
    raise FileNotFoundError(f"LONGCLIP_PATH does not exist: {longclip_path}")

with open("configs/istanbul_traffic/train.yaml", "r") as f:
    train_configs = yaml.safe_load(f)
with open("configs/istanbul_traffic/diff/model_text2ts_dep.yaml", "r") as f:
    diff_configs = yaml.safe_load(f)
with open("configs/istanbul_traffic/cond/text_msmdiffmv.yaml", "r") as f:
    cond_configs = yaml.safe_load(f)

train_configs = copy.deepcopy(train_configs)
diff_configs = copy.deepcopy(diff_configs)
cond_configs = copy.deepcopy(cond_configs)

train_configs["data"]["folder"] = data_folder
diff_configs["device"] = device
diff_configs["generator_pretrain_path"] = ""
diff_configs["diffusion"]["img_size"] = img_size
cond_configs["cond_modal"] = "text"
cond_configs["text"]["pretrain_model_path"] = longclip_path
cond_configs["text"]["tokenizer_path"] = longclip_path

train_ts_path = os.path.join(data_folder, "train_ts.npy")
train_ts = np.load(train_ts_path)
if train_ts.ndim == 2:
    seq_len, n_var = train_ts.shape[1], 1
else:
    seq_len, n_var = train_ts.shape[1], train_ts.shape[2]
diff_configs["diffusion"]["seq_len"] = int(seq_len)
diff_configs["diffusion"]["n_var"] = int(n_var)
diff_configs["diffusion"]["side"]["num_var"] = int(n_var)

print(f"Loaded train_ts: shape={train_ts.shape}, seq_len={seq_len}, n_var={n_var}")

dataset = GenerationDataset(train_configs["data"])
loader = dataset.get_loader(
    "train",
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)
batch = next(iter(loader))

cap = batch["cap"][0]
print("First caption:")
print(cap)

model = ImageConditionalGenerator(diff_configs, cond_configs).to(device)
model.train()

loss = model(batch, is_train=True)
print("Loss keys:", sorted(loss.keys()))
for key, value in loss.items():
    print(f"{key}: {float(value.detach().cpu()):.6f}")

loss["all"].backward()

has_grad = any(
    param.grad is not None
    for name, param in model.named_parameters()
    if param.requires_grad
)
if not has_grad:
    raise RuntimeError("Backward finished, but no trainable parameter received gradients")

print("Smoke test passed: forward and backward completed.")
PY
