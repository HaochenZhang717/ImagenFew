from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import glob
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
from qwen_vl_utils import process_vision_info
import os
import json
from transformers import AutoProcessor, Qwen3VLProcessor
from collections import defaultdict




def parse_configs(config: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    if isinstance(config, str):
        config = OmegaConf.load(config)
    # rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    # eval_config = config.get("eval", None)
    # return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, eval_config
    return stage2_config, transport_config, sampler_config, guidance_config, misc, training_config


def none_or_str(value):
    if value == 'None':
        return None
    return value


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def collate_fn(batch):
    out = {}
    keys = batch[0].keys()

    for k in keys:
        if k in ["input_ids", "attention_mask", "labels"]:
            pad_val = 0
            if k == "labels":
                pad_val = -100

            out[k] = torch.nn.utils.rnn.pad_sequence(
                [x[k] for x in batch],
                batch_first=True,
                padding_value=pad_val
            )
        elif k in ["raw_image_path"]:
            out[k] = [x[k] for x in batch]
        else:
            # pixel_values / image_grid_thw 等通常可以直接 stack
            out[k] = torch.stack([x[k] for x in batch], dim=0)
    loss_mask = (out["labels"] != -100).long()
    out["loss_mask"] = loss_mask

    return out



def build_timeseries_collate_fn(processor, num_seg=4, num_ch=2):

    def collate_fn(batch):
        flat_messages = []
        image_names_by_sample = []
        image_ids = []

        for sample in batch:
            image_ids.append(sample["image_id"])
            image_names_by_sample.append(sample["image_names"])

            # 每个 sample 的每张图都独立作为一个 message
            # 这里只是借 processor 生成 pixel_values / grid_thw
            for path in sample["image_paths"]:
                flat_messages.append(
                    [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": path,
                                "min_pixels": 0,
                                "max_pixels": 64 * 14 * 14,
                            }
                        ],
                    }]
                )

        image_inputs, video_inputs = process_vision_info(flat_messages)

        proc = processor(
            text=[""] * len(flat_messages),   # 这里文本无意义，只是占位
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": proc["pixel_values"],
            "image_grid_thw": proc["image_grid_thw"],
            "image_ids": image_ids,
            "image_names": image_names_by_sample,
            "batch_size_ts": len(batch),
            "num_seg": num_seg,
            "num_ch": num_ch,
        }

    return collate_fn



def build_timeseries_collate_fn_one_per_channel(processor, num_ch=2):

    def collate_fn(batch):

        flat_messages = []
        image_names_by_sample = []
        image_ids = []

        for sample in batch:

            image_ids.append(sample["image_id"])
            image_names_by_sample.append(sample["image_names"])

            for path in sample["image_paths"]:

                flat_messages.append(
                    [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": path,
                                "min_pixels": 0,
                                "max_pixels": 64 * 14 * 14,
                            }
                        ],
                    }]
                )

        image_inputs, video_inputs = process_vision_info(flat_messages)

        proc = processor(
            text=[""] * len(flat_messages),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": proc["pixel_values"],
            "image_grid_thw": proc["image_grid_thw"],
            "image_ids": image_ids,
            "image_names": image_names_by_sample,
            "batch_size_ts": len(batch),
            "num_ch": num_ch,
        }

    return collate_fn


class TimeSeriesImageDataset(Dataset):
    def __init__(self, image_path, jsonl_path, num_seg=4, num_ch=2):
        self.image_path = image_path
        self.num_seg = num_seg
        self.num_ch = num_ch

        data_by_image = defaultdict(dict)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                image_name = obj["image"]

                name = image_name.replace(".png", "")
                parts = name.split("_")

                image_id = int(parts[0][5:])
                seg_id = int(parts[1][3:])
                ch_id = int(parts[2][2:])

                data_by_image[image_id][(seg_id, ch_id)] = image_name

        self.data = []

        for image_id, items in data_by_image.items():
            # 过滤不完整样本
            if len(items) != num_seg * num_ch:
                continue

            image_names = []
            for seg in range(num_seg):
                for ch in range(num_ch):
                    image_name = items[(seg, ch)]
                    image_names.append(image_name)

            self.data.append({
                "image_id": image_id,
                "image_names": image_names,
            })

        print(f"[INFO] Loaded {len(self.data)} complete time-series samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_paths = [
            os.path.join(self.image_path, name)
            for name in item["image_names"]
        ]

        return {
            "image_id": item["image_id"],
            "image_names": item["image_names"],
            "image_paths": image_paths,
        }


class TimeSeriesImageDatasetOnePerChannel(Dataset):

    def __init__(self, image_path, num_ch=2):

        self.image_path = image_path
        self.num_ch = num_ch

        data_by_image = defaultdict(dict)

        image_files = [
            f for f in os.listdir(image_path)
            if f.endswith(".png")
        ]

        for image_name in image_files:

            # image12_ch1.png
            name = image_name.replace(".png", "")
            parts = name.split("_")

            image_id = int(parts[0][5:])
            ch_id = int(parts[1][2:])

            data_by_image[image_id][ch_id] = image_name

        self.data = []

        for image_id, items in data_by_image.items():

            # 保证 channel 完整
            if len(items) != num_ch:
                continue

            image_names = [
                items[ch] for ch in range(num_ch)
            ]

            self.data.append({
                "image_id": image_id,
                "image_names": image_names,
            })

        print(f"[INFO] Loaded {len(self.data)} complete time-series samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        image_paths = [
            os.path.join(self.image_path, name)
            for name in item["image_names"]
        ]

        return {
            "image_id": item["image_id"],
            "image_names": item["image_names"],
            "image_paths": image_paths,
        }


class LatentDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.paths = [
            "/playpen/haochenz/image_synthetic_u_precomputed_vision_latent/train_rank0.pt",
            "/playpen/haochenz/image_synthetic_u_precomputed_vision_latent/train_rank1.pt",
            "/playpen/haochenz/image_synthetic_u_precomputed_vision_latent/train_rank2.pt",
            "/playpen/haochenz/image_synthetic_u_precomputed_vision_latent/train_rank3.pt",
        ]

        self.data = []
        for path in self.paths:
            self.data.append(torch.load(path, map_location="cpu"))

        self.data = torch.cat(self.data)
        # print(self.data.shape)
        # breakpoint()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PrecomputedVisionEmbeddingDatasetOnePerChannel(Dataset):
    def __init__(self, precomputed_dir: str, num_ch: int):
        super().__init__()
        self.precomputed_dir = precomputed_dir
        self.num_ch = num_ch

        shard_paths = sorted(glob.glob(os.path.join(precomputed_dir, "*_rank*.pt")))
        if not shard_paths:
            raise ValueError(f"No precomputed embedding shards found in {precomputed_dir}.")

        self.shard_paths = shard_paths
        self.embedding_dict = {}
        duplicate_keys = []
        for shard_path in self.shard_paths:
            shard = torch.load(shard_path, map_location="cpu")
            if not isinstance(shard, dict):
                raise TypeError(f"Expected dict shard at {shard_path}, got {type(shard)}")
            shared_keys = set(self.embedding_dict).intersection(shard)
            if shared_keys:
                for key in sorted(shared_keys):
                    existing = self.embedding_dict[key]
                    incoming = shard[key]
                    if existing.shape != incoming.shape:
                        raise ValueError(
                            f"Duplicate key {key} has mismatched shapes across shards: "
                            f"{tuple(existing.shape)} vs {tuple(incoming.shape)}"
                        )
                    if not torch.equal(existing, incoming):
                        raise ValueError(
                            f"Duplicate key {key} has non-identical embeddings across shards."
                        )
                    duplicate_keys.append(key)
            for key, value in shard.items():
                self.embedding_dict.setdefault(key, value)

        if duplicate_keys:
            unique_dup_count = len(set(duplicate_keys))
            print(
                f"[INFO] Found {unique_dup_count} duplicated image keys across precomputed shards. "
                f"These are expected padding duplicates from DistributedSampler and were deduplicated."
            )

        data_by_image = defaultdict(dict)
        pattern = re.compile(r"image(\d+)_ch(\d+)\.png$")

        for image_name, embedding in self.embedding_dict.items():
            match = pattern.match(image_name)
            if match is None:
                continue
            image_id = int(match.group(1))
            ch_id = int(match.group(2))
            data_by_image[image_id][ch_id] = embedding

        self.data = []
        for image_id, items in data_by_image.items():
            if len(items) != num_ch:
                continue

            channel_embeddings = [items[ch] for ch in range(num_ch)]
            self.data.append({
                "image_id": image_id,
                "z_per_channel": torch.stack(channel_embeddings, dim=0),
            })

        print(
            f"[INFO] Loaded {len(self.data)} complete precomputed time-series samples "
            f"from {len(self.shard_paths)} shards under {self.precomputed_dir}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def prepare_latent_dataloader(
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    # dataset = TimeSeriesCaptionDataset(image_path, jsonl_path, vlm_name)
    dataset = LatentDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True if shuffle else False,
        # collate_fn=collate_fn,
    )
    return loader, sampler


def collate_precomputed_one_per_channel(batch):
    z = torch.stack([sample["z_per_channel"] for sample in batch], dim=0)
    image_ids = [sample["image_id"] for sample in batch]
    num_ch = z.shape[1]
    return {
        "z_per_channel": z,
        "image_ids": image_ids,
        "batch_size_ts": len(batch),
        "num_ch": num_ch,
    }


def prepare_dataloader(
    image_path: str,
    jsonl_path: str,
    vlm_name: str,
    num_seg: int,
    num_ch: int,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    dataset = TimeSeriesImageDataset(
        image_path, jsonl_path,
        num_seg, num_ch
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    collate_fn_ = build_timeseries_collate_fn(
        processor=Qwen3VLProcessor.from_pretrained(
            vlm_name,
            images_kwargs={
        "min_pixels": 0,
        "max_pixels": 64 * 14 * 14}),
        num_seg=4,
        num_ch=2,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True if shuffle else False,
        collate_fn=collate_fn_,
    )
    return loader, sampler



def prepare_dataloader_one_per_channel(
    image_path: str,
    vlm_name: str,
    num_ch: int,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    dataset = TimeSeriesImageDatasetOnePerChannel(
        image_path, num_ch
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    collate_fn_ = build_timeseries_collate_fn_one_per_channel(
        processor=Qwen3VLProcessor.from_pretrained(
            vlm_name,
            images_kwargs={
        "min_pixels": 0,
        "max_pixels": 32 * 14 * 14}),
        num_ch=num_ch,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True if shuffle else False,
        collate_fn=collate_fn_,
    )
    return loader, sampler


def prepare_precomputed_dataloader_one_per_channel(
    precomputed_dir: str,
    num_ch: int,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    dataset = PrecomputedVisionEmbeddingDatasetOnePerChannel(
        precomputed_dir=precomputed_dir,
        num_ch=num_ch,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True if shuffle else False,
        collate_fn=collate_precomputed_one_per_channel,
    )
    return loader, sampler


def get_autocast_scaler(args):
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)
    
    return scaler, autocast_kwargs



if __name__ == "__main__":
    # data = TimeSeriesCaptionDataset(
    #     "/Users/zhc/Documents/PhD/projects/TimeSeriesUnifiedModel/VerbalTSDatasets/image_synthetic_u",
    #     "/Users/zhc/Documents/PhD/projects/LITS/step_1_dataset_construction/synthetic_u_caption/time_series_caps_5_8.jsonl",
    #     vlm_name="Qwen/Qwen3-VL-8B-Instruct")
    #
    # print(data[0])


    loader, sampler = prepare_dataloader(
            image_path="/Users/zhc/Documents/PhD/projects/TimeSeriesUnifiedModel/VerbalTSDatasets/image_synthetic_u",
            jsonl_path="/Users/zhc/Documents/PhD/projects/LITS/step_1_dataset_construction/synthetic_u_caption/time_series_caps_5_8.jsonl",
            vlm_name="Qwen/Qwen3-VL-8B-Instruct",
            batch_size=4,
            workers=4,
            rank=0,
            world_size=1
    )

    batch = next(iter(loader))
    print(batch)
