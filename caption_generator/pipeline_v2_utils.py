import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def _flatten_caption_array(captions: np.ndarray) -> List[str]:
    flattened: List[str] = []
    for item in captions:
        if isinstance(item, np.ndarray):
            flattened.append(str(item.reshape(-1)[0]))
        elif isinstance(item, (list, tuple)):
            flattened.append(str(item[0]))
        else:
            flattened.append(str(item))
    return flattened


def load_split_arrays(dataset_root: str, split: str):
    ts = np.load(os.path.join(dataset_root, f"{split}_ts.npy"), allow_pickle=True).astype(np.float32)
    captions = np.load(
        os.path.join(dataset_root, f"{split}_text_caps.npy"),
        allow_pickle=True,
    )
    attrs_path = os.path.join(dataset_root, f"{split}_attrs_idx.npy")
    attrs = np.load(attrs_path, allow_pickle=True) if os.path.exists(attrs_path) else None
    return ts, _flatten_caption_array(captions), attrs


def compute_global_stats(train_ts: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(train_ts.mean()),
        "std": float(train_ts.std() + 1e-6),
    }


def save_json(data: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


class DeterministicTimeSeriesRenderer:
    def __init__(
        self,
        image_size: int = 100,
        background_color: int = 255,
        line_color: int = 0,
        line_width: int = 2,
        padding: int = 8,
        panel_gap: int = 4,
        normalization: str = "per_sample_minmax",
        fixed_y_min: float = -3.0,
        fixed_y_max: float = 3.0,
    ) -> None:
        self.image_size = int(image_size)
        self.background_color = int(background_color)
        self.line_color = int(line_color)
        self.line_width = int(line_width)
        self.padding = int(padding)
        self.panel_gap = int(panel_gap)
        self.normalization = normalization
        self.fixed_y_min = float(fixed_y_min)
        self.fixed_y_max = float(fixed_y_max)

    def _normalize_series(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        if self.normalization == "per_sample_minmax":
            v_min = float(values.min())
            v_max = float(values.max())
            if abs(v_max - v_min) < 1e-8:
                return np.full_like(values, 0.5)
            return (values - v_min) / (v_max - v_min)
        if self.normalization == "fixed_range":
            clipped = np.clip(values, self.fixed_y_min, self.fixed_y_max)
            denom = max(self.fixed_y_max - self.fixed_y_min, 1e-8)
            return (clipped - self.fixed_y_min) / denom
        raise ValueError(f"Unsupported renderer normalization mode: {self.normalization}")

    def render(self, ts: np.ndarray) -> Image.Image:
        if ts.ndim == 1:
            ts = ts[:, None]
        if ts.ndim != 2:
            raise ValueError(f"Expected time series with shape (T,) or (T, C), got {ts.shape}")

        length, channels = ts.shape
        if length < 2:
            raise ValueError(f"Expected at least 2 time steps, got {length}")

        canvas = Image.new("RGB", (self.image_size, self.image_size), (self.background_color,) * 3)
        draw = ImageDraw.Draw(canvas)

        usable_w = self.image_size - 2 * self.padding
        usable_h = self.image_size - 2 * self.padding
        if usable_w <= 1 or usable_h <= 1:
            raise ValueError("Renderer image_size/padding combination leaves no drawable area.")

        total_gap = self.panel_gap * max(channels - 1, 0)
        panel_h = max((usable_h - total_gap) // channels, 1)

        for channel_idx in range(channels):
            top = self.padding + channel_idx * (panel_h + self.panel_gap)
            bottom = min(top + panel_h - 1, self.image_size - self.padding - 1)
            normalized = self._normalize_series(ts[:, channel_idx])

            points = []
            for step_idx, value in enumerate(normalized):
                x = self.padding + round(step_idx * (usable_w - 1) / (length - 1))
                y = top + round((1.0 - float(value)) * max(bottom - top, 1))
                points.append((x, y))

            if len(points) >= 2:
                draw.line(points, fill=(self.line_color,) * 3, width=self.line_width)

        return canvas


class TimeSeriesImageCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        prompt_text: str,
        renderer: DeterministicTimeSeriesRenderer,
        normalization_stats: Optional[Dict[str, float]] = None,
        normalize_before_render: bool = False,
    ) -> None:
        ts, captions, attrs = load_split_arrays(dataset_root, split)
        self.ts = ts
        self.captions = [caption.strip() for caption in captions]
        self.attrs = attrs
        self.prompt_text = prompt_text
        self.renderer = renderer
        self.normalize_before_render = bool(normalize_before_render)
        self.normalization_stats = normalization_stats

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        ts = self.ts[index]
        if ts.ndim == 1:
            ts = ts[:, None]
        if ts.ndim != 2:
            raise ValueError(f"Expected per-sample time series shape (T, C), got {ts.shape}")

        render_ts = ts
        if self.normalize_before_render:
            if self.normalization_stats is None:
                raise ValueError("normalize_before_render=True requires normalization_stats.")
            render_ts = (render_ts - self.normalization_stats["mean"]) / self.normalization_stats["std"]

        sample = {
            "image": self.renderer.render(render_ts),
            "caption_text": self.captions[index],
            "prompt_text": self.prompt_text,
            "sample_index": index,
        }
        if self.attrs is not None:
            sample["attrs_idx"] = torch.as_tensor(self.attrs[index], dtype=torch.long)
        return sample


class PreRenderedTimeSeriesImageCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        image_root: str,
        split: str,
        prompt_text: str,
    ) -> None:
        _, captions, attrs = load_split_arrays(dataset_root, split)
        split_dir = os.path.join(image_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Pre-rendered image directory does not exist for split '{split}': {split_dir}"
            )

        self.image_paths = [os.path.join(split_dir, f"{idx:06d}.png") for idx in range(len(captions))]
        missing = [path for path in self.image_paths if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} pre-rendered images under {split_dir}. "
                f"First missing file: {missing[0]}"
            )

        self.captions = [caption.strip() for caption in captions]
        self.attrs = attrs
        self.prompt_text = prompt_text

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = {
            "image": Image.open(self.image_paths[index]).convert("RGB"),
            "caption_text": self.captions[index],
            "prompt_text": self.prompt_text,
            "sample_index": index,
            "image_path": self.image_paths[index],
        }
        if self.attrs is not None:
            sample["attrs_idx"] = torch.as_tensor(self.attrs[index], dtype=torch.long)
        return sample


def render_and_save_split(
    dataset_root: str,
    output_root: str,
    split: str,
    renderer: DeterministicTimeSeriesRenderer,
    normalization_stats: Optional[Dict[str, float]] = None,
    normalize_before_render: bool = False,
) -> Dict[str, object]:
    ts, captions, _ = load_split_arrays(dataset_root, split)
    split_dir = os.path.join(output_root, split)
    os.makedirs(split_dir, exist_ok=True)

    for index, series in enumerate(ts):
        if series.ndim == 1:
            series = series[:, None]
        render_ts = series
        if normalize_before_render:
            if normalization_stats is None:
                raise ValueError("normalize_before_render=True requires normalization_stats.")
            render_ts = (render_ts - normalization_stats["mean"]) / normalization_stats["std"]
        image = renderer.render(render_ts)
        image.save(os.path.join(split_dir, f"{index:06d}.png"))

    metadata = {
        "split": split,
        "num_samples": int(len(captions)),
        "image_dir": split_dir,
    }
    save_json(metadata, os.path.join(split_dir, "metadata.json"))
    return metadata


@dataclass
class PipelineV2Collator:
    processor: object
    instruction_prompt: str

    def _build_messages(self) -> List[Dict[str, object]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.instruction_prompt},
                ],
            }
        ]

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        captions = [item["caption_text"] for item in batch]

        prompt_text = self.processor.apply_chat_template(
            self._build_messages(),
            tokenize=False,
            add_generation_prompt=True,
        )
        full_texts = [prompt_text + caption for caption in captions]

        prompt_inputs = self.processor(
            text=[prompt_text] * len(batch),
            images=images,
            padding=True,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[row_idx, :prompt_len] = -100

        output = {
            "input_ids": full_inputs["input_ids"],
            "attention_mask": full_inputs["attention_mask"],
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"],
            "labels": labels,
            "raw_captions": captions,
            "prompt_text": prompt_text,
            "sample_indices": torch.tensor([item["sample_index"] for item in batch], dtype=torch.long),
        }
        if "attrs_idx" in batch[0]:
            output["attrs_idx"] = torch.stack([item["attrs_idx"] for item in batch], dim=0)
        return output
