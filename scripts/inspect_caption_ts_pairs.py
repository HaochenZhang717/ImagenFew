import argparse
import os
import random
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEGMENT_PATTERN = re.compile(
    r"\[Segment\s+([1-4])\]\s*:?\s*(.*?)(?=\n\s*\[Segment\s+[1-4]\]\s*:|\Z)",
    flags=re.DOTALL,
)


def flatten_caption_array(captions):
    return [str(item) for item in captions.reshape(-1)]


def parse_index_list(index_text):
    indices = []
    for item in index_text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(item))
    return indices


def split_caption_segments(caption):
    segments = {}
    for match in SEGMENT_PATTERN.finditer(caption.strip()):
        segments[int(match.group(1))] = " ".join(match.group(2).strip().split())
    return segments


def save_plot(output_path, series, caption):
    indices_by_segment = np.array_split(np.arange(len(series)), 4)
    caption_segments = split_caption_segments(caption)

    fig, axes = plt.subplots(4, 1, figsize=(12, 7), dpi=140, sharey=True)
    for segment_id, (ax, indices) in enumerate(zip(axes, indices_by_segment), start=1):
        values = series[indices]
        ax.plot(indices, values, linewidth=1.8)
        ax.axvline(indices[0], color="gray", linewidth=0.6, alpha=0.4)
        ax.grid(alpha=0.2)
        ax.set_title(f"Segment {segment_id}", fontsize=9, loc="left")
        text = caption_segments.get(segment_id, "")
        ax.text(
            0.01,
            0.04,
            text,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
        )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def print_one(index, series, caption, print_values):
    print("=" * 100)
    print(f"Index: {index}")
    print("- Caption:")
    print(caption)
    print("- Time series summary:")
    print(
        f"  length={len(series)} min={series.min():.6g} max={series.max():.6g} "
        f"mean={series.mean():.6g} std={series.std():.6g}"
    )
    for segment_id, indices in enumerate(np.array_split(np.arange(len(series)), 4), start=1):
        values = series[indices]
        print(
            f"  Segment {segment_id}: start={values[0]:.6g} end={values[-1]:.6g} "
            f"delta={values[-1] - values[0]:.6g} min={values.min():.6g} max={values.max():.6g}"
        )
        if print_values:
            print("    values:", np.array2string(values, precision=4, separator=", "))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Print caption/time-series pairs for manual inspection."
    )
    parser.add_argument("--dataset-root", type=str, default="../data/VerbalTSDatasets/istanbul_traffic")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ts-path", type=str, default=None)
    parser.add_argument("--caps-path", type=str, default=None)
    parser.add_argument(
        "--indices",
        type=str,
        default="0-9",
        help="Comma-separated indices/ranges, e.g. '0,5,10-15'. Ignored when --random is set.",
    )
    parser.add_argument("--random", type=int, default=0, help="Randomly inspect this many samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-values", action="store_true", help="Print raw segment values.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Optional directory to save paired plots.")
    args = parser.parse_args()

    ts_path = args.ts_path or os.path.join(args.dataset_root, f"{args.split}_ts.npy")
    caps_path = args.caps_path or os.path.join(args.dataset_root, f"{args.split}_my_text_caps.npy")
    ts = np.load(ts_path, allow_pickle=True)
    captions = flatten_caption_array(np.load(caps_path, allow_pickle=True))

    if ts.ndim != 3:
        raise ValueError(f"Expected ts shape (N, T, C), got {ts.shape}")
    if ts.shape[0] != len(captions):
        raise ValueError(f"Count mismatch: ts N={ts.shape[0]}, captions N={len(captions)}")
    if ts.shape[2] != 1:
        raise ValueError(f"Expected one channel, got C={ts.shape[2]}")

    if args.random > 0:
        rng = random.Random(args.seed)
        indices = rng.sample(range(ts.shape[0]), k=min(args.random, ts.shape[0]))
    else:
        indices = parse_index_list(args.indices)

    print(f"ts_path: {ts_path}")
    print(f"caps_path: {caps_path}")
    print(f"ts_shape: {ts.shape}")
    print(f"caption_count: {len(captions)}")
    print(f"indices: {indices}")
    print()

    for index in indices:
        if index < 0 or index >= ts.shape[0]:
            print(f"[WARN] skip out-of-range index: {index}")
            continue
        series = ts[index, :, 0]
        caption = captions[index]
        print_one(index, series, caption, args.print_values)
        if args.plot_dir:
            save_plot(os.path.join(args.plot_dir, f"{index:06d}.png"), series, caption)

    if args.plot_dir:
        print(f"Saved plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()
