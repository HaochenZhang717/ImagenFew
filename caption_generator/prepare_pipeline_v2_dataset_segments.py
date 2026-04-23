import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Optional

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from omegaconf import OmegaConf

try:
    from caption_generator.pipeline_v2_utils import (
        compute_global_stats,
        load_split_arrays,
        save_json,
    )
except ImportError:
    from pipeline_v2_utils import (
        compute_global_stats,
        load_split_arrays,
        save_json,
    )


_RENDERER = None
_NORMALIZATION_STATS = None
_NORMALIZE_BEFORE_RENDER = False


class FourSegmentTimeSeriesRenderer:
    def __init__(
        self,
        segment_size: int = 100,
        num_segments: int = 4,
        background_color: int = 255,
        line_color: int = 0,
        line_width: int = 2,
        separator_width: int = 2,
        padding: int = 8,
        panel_gap: int = 4,
        normalization: str = "per_sample_minmax",
        fixed_y_min: float = -3.0,
        fixed_y_max: float = 3.0,
    ) -> None:
        self.segment_size = int(segment_size)
        self.num_segments = int(num_segments)
        self.background_color = int(background_color)
        self.line_color = int(line_color)
        self.line_width = int(line_width)
        self.separator_width = int(separator_width)
        self.padding = int(padding)
        self.panel_gap = int(panel_gap)
        self.normalization = normalization
        self.fixed_y_min = float(fixed_y_min)
        self.fixed_y_max = float(fixed_y_max)

    @property
    def image_width(self) -> int:
        return self.segment_size * self.num_segments

    @property
    def image_height(self) -> int:
        return self.segment_size

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
        if length < self.num_segments:
            raise ValueError(f"Expected at least {self.num_segments} time steps, got {length}")

        normalized_ts = np.empty_like(ts, dtype=np.float32)
        for channel_idx in range(channels):
            normalized_ts[:, channel_idx] = self._normalize_series(ts[:, channel_idx])

        segment_indices = np.array_split(np.arange(length), self.num_segments)
        dpi = 100
        figsize = (self.image_width / dpi, self.image_height / dpi)
        fig, axes = plt.subplots(
            1,
            self.num_segments,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
            gridspec_kw={"wspace": 0.0},
        )
        fig.patch.set_facecolor((self.background_color / 255.0,) * 3)

        line_color = (self.line_color / 255.0,) * 3
        for ax, indices in zip(axes[0], segment_indices):
            segment = normalized_ts[indices]
            x = np.arange(len(indices), dtype=np.float32)
            for channel_idx in range(channels):
                ax.plot(
                    x,
                    segment[:, channel_idx],
                    color=line_color,
                    linewidth=self.line_width,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )
            ax.set_xlim(0, max(len(indices) - 1, 1))
            ax.set_ylim(0.0, 1.0)
            ax.set_facecolor((self.background_color / 255.0,) * 3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.margins(x=0.0, y=0.05)

        left = self.padding / self.image_width
        right = 1.0 - self.padding / self.image_width
        bottom = self.padding / self.image_height
        top = 1.0 - self.padding / self.image_height
        wspace = self.panel_gap / max(self.segment_size, 1)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
        plt.close(fig)
        half_width = max(self.separator_width // 2, 0)
        for boundary in range(self.segment_size, self.image_width, self.segment_size):
            start = max(boundary - half_width, 0)
            end = min(start + self.separator_width, self.image_width)
            image[:, start:end, :] = np.array([255, 0, 0], dtype=np.uint8)
        return Image.fromarray(image)


def _init_render_worker(
    renderer_kwargs: Dict[str, object],
    normalization_stats: Optional[Dict[str, float]],
    normalize_before_render: bool,
) -> None:
    global _RENDERER, _NORMALIZATION_STATS, _NORMALIZE_BEFORE_RENDER
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    _RENDERER = FourSegmentTimeSeriesRenderer(**renderer_kwargs)
    _NORMALIZATION_STATS = normalization_stats
    _NORMALIZE_BEFORE_RENDER = normalize_before_render


def _render_one_sample_to_path(task):
    index, series, output_path = task

    if series.ndim == 1:
        series = series[:, None]
    render_ts = series
    if _NORMALIZE_BEFORE_RENDER:
        if _NORMALIZATION_STATS is None:
            raise ValueError("normalize_before_render=True requires normalization_stats.")
        render_ts = (render_ts - _NORMALIZATION_STATS["mean"]) / _NORMALIZATION_STATS["std"]

    image = _RENDERER.render(render_ts)
    image.save(output_path)
    return index


def render_and_save_split(
    dataset_root: str,
    output_root: str,
    split: str,
    renderer: FourSegmentTimeSeriesRenderer,
    normalization_stats: Optional[Dict[str, float]] = None,
    normalize_before_render: bool = False,
    num_workers: int = 1,
    overwrite: bool = False,
    chunksize: int = 16,
    parallel_backend: str = "auto",
) -> Dict[str, object]:
    ts, captions, _ = load_split_arrays(dataset_root, split)
    split_dir = os.path.join(output_root, split)
    os.makedirs(split_dir, exist_ok=True)
    renderer_kwargs = {
        "segment_size": renderer.segment_size,
        "num_segments": renderer.num_segments,
        "background_color": renderer.background_color,
        "line_color": renderer.line_color,
        "line_width": renderer.line_width,
        "separator_width": renderer.separator_width,
        "padding": renderer.padding,
        "panel_gap": renderer.panel_gap,
        "normalization": renderer.normalization,
        "fixed_y_min": renderer.fixed_y_min,
        "fixed_y_max": renderer.fixed_y_max,
    }

    tasks = []
    skipped = 0
    for index, series in enumerate(ts):
        output_path = os.path.join(split_dir, f"{index:06d}.png")
        if not overwrite and os.path.exists(output_path):
            skipped += 1
            continue
        tasks.append((index, series, output_path))

    backend_used = "none"
    if tasks:
        if num_workers <= 1:
            _init_render_worker(renderer_kwargs, normalization_stats, normalize_before_render)
            iterator = map(_render_one_sample_to_path, tasks)
            for _ in tqdm(iterator, total=len(tasks), desc=f"render {split}", leave=False):
                pass
            backend_used = "single"
        else:
            selected_backend = parallel_backend
            if selected_backend not in {"auto", "process", "thread"}:
                raise ValueError(f"Unsupported parallel_backend: {parallel_backend}")

            if selected_backend in {"auto", "process"}:
                try:
                    with ProcessPoolExecutor(
                        max_workers=num_workers,
                        initializer=_init_render_worker,
                        initargs=(renderer_kwargs, normalization_stats, normalize_before_render),
                    ) as executor:
                        iterator = executor.map(_render_one_sample_to_path, tasks, chunksize=max(int(chunksize), 1))
                        for _ in tqdm(iterator, total=len(tasks), desc=f"render {split}", leave=False):
                            pass
                    backend_used = "process"
                except (PermissionError, OSError):
                    if selected_backend == "process":
                        raise
                    selected_backend = "thread"

            if selected_backend == "thread":
                _init_render_worker(renderer_kwargs, normalization_stats, normalize_before_render)
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    iterator = executor.map(_render_one_sample_to_path, tasks)
                    for _ in tqdm(iterator, total=len(tasks), desc=f"render {split}", leave=False):
                        pass
                backend_used = "thread"

    metadata = {
        "split": split,
        "num_samples": int(len(captions)),
        "image_dir": split_dir,
        "num_rendered": int(len(tasks)),
        "num_skipped_existing": int(skipped),
        "parallel_backend": backend_used,
        "num_workers": int(num_workers),
        "image_width": int(renderer.image_width),
        "image_height": int(renderer.image_height),
        "num_segments": int(renderer.num_segments),
        "separator_width": int(renderer.separator_width),
    }
    save_json(metadata, os.path.join(split_dir, "metadata.json"))
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-render time-series images for pipeline_v2 training.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--parallel-backend", type=str, default=None, choices=["auto", "process", "thread"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    output_root = cfg["data"].get("prerendered_image_root")
    if not output_root:
        raise ValueError("Config must define data.prerendered_image_root for offline image caching.")
    os.makedirs(output_root, exist_ok=True)

    train_ts, _, _ = load_split_arrays(cfg["data"]["dataset_root"], "train")
    stats = compute_global_stats(train_ts)
    renderer_cfg = dict(cfg["renderer"])
    if "image_size" in renderer_cfg and "segment_size" not in renderer_cfg:
        renderer_cfg["segment_size"] = renderer_cfg.pop("image_size")
    renderer_cfg.setdefault("segment_size", 100)
    renderer_cfg.setdefault("num_segments", 4)
    renderer = FourSegmentTimeSeriesRenderer(**renderer_cfg)
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["data"].get("prerender_num_workers", 1))
    chunksize = args.chunksize if args.chunksize is not None else int(cfg["data"].get("prerender_chunksize", 16))
    parallel_backend = args.parallel_backend or cfg["data"].get("prerender_parallel_backend", "auto")

    split_summaries = []
    for split in args.splits:
        split_summaries.append(
            render_and_save_split(
                dataset_root=cfg["data"]["dataset_root"],
                output_root=output_root,
                split=split,
                renderer=renderer,
                normalization_stats=stats,
                normalize_before_render=cfg["data"].get("normalize_before_render", False),
                num_workers=num_workers,
                overwrite=args.overwrite,
                chunksize=chunksize,
                parallel_backend=parallel_backend,
            )
        )

    manifest = {
        "dataset_root": cfg["data"]["dataset_root"],
        "prerendered_image_root": output_root,
        "normalize_before_render": bool(cfg["data"].get("normalize_before_render", False)),
        "prerender_num_workers": int(num_workers),
        "prerender_chunksize": int(chunksize),
        "prerender_parallel_backend": parallel_backend,
        "renderer": cfg["renderer"],
        "rendered_image_width": int(renderer.image_width),
        "rendered_image_height": int(renderer.image_height),
        "rendered_num_segments": int(renderer.num_segments),
        "train_stats": stats,
        "splits": split_summaries,
    }
    save_json(manifest, os.path.join(output_root, "manifest.json"))


if __name__ == "__main__":
    main()
