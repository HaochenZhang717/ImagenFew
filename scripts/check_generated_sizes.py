import argparse
import os
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def summarize_npy(path: Path) -> str:
    arr = np.load(path, allow_pickle=True)
    preview = [
        f"shape={getattr(arr, 'shape', None)}",
        f"dtype={getattr(arr, 'dtype', None)}",
        f"file_size={format_bytes(path.stat().st_size)}",
    ]
    if getattr(arr, "dtype", None) == object and arr.size > 0:
        try:
            first = arr.reshape(-1)[0]
            preview.append(f"first_item_type={type(first).__name__}")
        except Exception:
            pass
    return ", ".join(preview)


def summarize_pt(path: Path) -> str:
    if torch is None:
        return f"file_size={format_bytes(path.stat().st_size)}, torch=unavailable"

    obj = torch.load(path, map_location="cpu", weights_only=False)
    preview = [f"file_size={format_bytes(path.stat().st_size)}", f"object_type={type(obj).__name__}"]

    if torch.is_tensor(obj):
        preview.insert(0, f"shape={tuple(obj.shape)}")
        preview.insert(1, f"dtype={obj.dtype}")
    elif isinstance(obj, dict):
        preview.append(f"keys={sorted(obj.keys())[:10]}")
        for key in ("shape", "embeddings", "train_latents", "valid_latents", "real_ts", "sampled_ts"):
            if key in obj and torch.is_tensor(obj[key]):
                tensor = obj[key]
                preview.append(f"{key}.shape={tuple(tensor.shape)}")
                preview.append(f"{key}.dtype={tensor.dtype}")
                break
    return ", ".join(preview)


def iter_targets(root: Path):
    patterns = ["generated_text_caps.npy", "*.pt"]
    seen = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def main():
    parser = argparse.ArgumentParser(description="Inspect generated caption/result files under VerbalTSDatasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets"),
        help="Root directory to scan.",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    found_any = False
    for path in sorted(iter_targets(root)):
        found_any = True
        rel_path = path.relative_to(root)
        suffix = path.suffix.lower()
        try:
            if suffix == ".npy":
                summary = summarize_npy(path)
            elif suffix == ".pt":
                summary = summarize_pt(path)
            else:
                summary = f"file_size={format_bytes(path.stat().st_size)}"
        except Exception as exc:
            summary = f"error={type(exc).__name__}: {exc}"
        print(f"{rel_path}: {summary}")

    if not found_any:
        print(f"No generated_text_caps.npy or .pt files found under {root}")


if __name__ == "__main__":
    main()
