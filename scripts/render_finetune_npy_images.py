import argparse
import os
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def infer_segment_length(seq_len, num_segments, segment_len):
    if segment_len is not None:
        total = segment_len * num_segments
        if total > seq_len:
            raise ValueError(
                f"segment_len * num_segments = {total} exceeds sequence length {seq_len}."
            )
        return segment_len

    if seq_len % num_segments != 0:
        raise ValueError(
            f"Sequence length {seq_len} is not divisible by num_segments={num_segments}. "
            "Please pass --segment-len explicitly."
        )
    return seq_len // num_segments


def plot_one(task):
    sample_idx, channel_idx, segments, save_dir, dpi, line_width, x_label, x_tick_step, image_size = task

    num_segments = len(segments)
    segment_len = len(segments[0])
    t = np.arange(segment_len)

    fig, axes = plt.subplots(
        1,
        num_segments,
        figsize=(image_size / dpi, image_size / dpi),
        dpi=dpi,
    )

    if num_segments == 1:
        axes = [axes]

    for seg_idx, ax in enumerate(axes):
        ax.plot(t, segments[seg_idx], linewidth=line_width)
        ax.margins(x=0.08, y=0.12)
        ax.set_yticks([])
        if x_tick_step and x_tick_step > 0:
            ax.set_xticks(np.arange(0, segment_len, x_tick_step))
            ax.tick_params(axis="x", labelsize=8, width=1, length=3, pad=1)
        elif not x_label:
            ax.set_xticks([])
        if x_label:
            ax.set_xlabel(x_label)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")

    plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.22, wspace=0.02, hspace=0)
    plt.savefig(
        os.path.join(save_dir, f"image{sample_idx}_ch{channel_idx}.png"),
        dpi=dpi,
        pad_inches=0,
    )
    plt.close(fig)


def build_tasks(ts_array, save_dir, num_segments, segment_len, dpi, line_width, x_label, x_tick_step, image_size):
    num_samples, _, num_channels = ts_array.shape
    for sample_idx in range(num_samples):
        for channel_idx in range(num_channels):
            segments = []
            for seg_idx in range(num_segments):
                start = segment_len * seg_idx
                end = start + segment_len
                segments.append(ts_array[sample_idx, start:end, channel_idx])
            yield (
                sample_idx,
                channel_idx,
                segments,
                save_dir,
                dpi,
                line_width,
                x_label,
                x_tick_step,
                image_size,
            )


def save_simple(
    save_dir,
    ts_path,
    num_segments=1,
    segment_len=None,
    workers=None,
    dpi=100,
    line_width=1.0,
    x_label="time",
    x_tick_step=6,
    image_size=100,
):
    os.makedirs(save_dir, exist_ok=True)

    train_ts = np.load(ts_path, allow_pickle=True)
    print(f"Loaded {ts_path} with shape {train_ts.shape}")

    if train_ts.ndim != 3:
        raise ValueError(f"Expected a (N, T, C) array, but got shape {train_ts.shape}")

    seq_len = train_ts.shape[1]
    segment_len = infer_segment_length(seq_len, num_segments, segment_len)
    workers = cpu_count() if workers is None else max(1, int(workers))
    total_images = train_ts.shape[0] * train_ts.shape[2]

    print(
        f"Rendering to {save_dir}: N={train_ts.shape[0]}, T={seq_len}, C={train_ts.shape[2]}, "
        f"num_segments={num_segments}, segment_len={segment_len}, workers={workers}, total_images={total_images}"
    )

    tasks = build_tasks(
        train_ts,
        save_dir,
        num_segments,
        segment_len,
        dpi,
        line_width,
        x_label,
        x_tick_step,
        image_size,
    )
    with Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(plot_one, tasks), total=total_images))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts-path", type=str, required=True, help="Path to input .npy file with shape (N, T, C)")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save rendered images")
    parser.add_argument("--num-segments", type=int, default=4, help="Number of horizontal segments per image")
    parser.add_argument("--segment-len", type=int, default=None, help="Optional manual segment length")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--x-label", type=str, default="", help="X-axis label text; pass empty string to hide")
    parser.add_argument("--x-tick-step", type=int, default=6, help="Spacing between x-axis ticks")
    parser.add_argument("--image-size", type=int, default=100, help="Output image size in pixels (square)")
    args = parser.parse_args()

    save_simple(
        save_dir=args.save_dir,
        ts_path=args.ts_path,
        num_segments=args.num_segments,
        segment_len=args.segment_len,
        workers=args.workers,
        dpi=args.dpi,
        line_width=args.line_width,
        x_label=args.x_label,
        x_tick_step=args.x_tick_step,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
