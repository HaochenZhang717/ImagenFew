import argparse
import csv
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

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

RISE_WORDS = (
    "rise",
    "rises",
    "rising",
    "increase",
    "increases",
    "increasing",
    "climb",
    "climbs",
    "climbing",
    "upward",
    "goes up",
    "trend up",
)
FALL_WORDS = (
    "decline",
    "declines",
    "declining",
    "drop",
    "drops",
    "dropping",
    "decrease",
    "decreases",
    "decreasing",
    "fall",
    "falls",
    "falling",
    "downward",
    "goes down",
    "trend down",
)
FLAT_WORDS = (
    "flat",
    "stable",
    "stabilizes",
    "stabilize",
    "steady",
    "level",
    "levels off",
    "little overall change",
    "minor fluctuations",
)
SHARP_WORDS = ("sharp", "sharply", "steep", "steeply", "abrupt", "rapid", "rapidly")
PEAK_WORDS = ("peak", "peaks", "high", "highest")
LOW_WORDS = ("low", "lower", "minimum", "minima", "trough")


def flatten_caption_array(captions: np.ndarray) -> List[str]:
    if captions.ndim == 0:
        raise ValueError(f"Expected caption array, got scalar shape={captions.shape}")
    return [str(item) for item in captions.reshape(-1)]


def parse_segments(caption: str) -> Dict[int, str]:
    segments = {}
    for match in SEGMENT_PATTERN.finditer(caption.strip()):
        segment_id = int(match.group(1))
        segment_text = " ".join(match.group(2).strip().split())
        segments[segment_id] = segment_text
    if sorted(segments) != [1, 2, 3, 4]:
        raise ValueError(f"Expected four [Segment i] entries, got ids={sorted(segments)}")
    return segments


def contains_any(text: str, words: Sequence[str]) -> bool:
    text = text.lower()
    return any(word in text for word in words)


def classify_segment(values: np.ndarray, global_std: float, trend_threshold: float) -> Dict[str, float]:
    values = values.astype(np.float64, copy=False)
    delta = float(values[-1] - values[0])
    delta_z = float(delta / max(global_std, 1e-8))
    value_range = float(values.max() - values.min())
    range_z = float(value_range / max(global_std, 1e-8))
    max_pos = int(np.argmax(values))
    min_pos = int(np.argmin(values))

    if delta_z > trend_threshold:
        trend = "rising"
    elif delta_z < -trend_threshold:
        trend = "falling"
    else:
        trend = "flat"

    return {
        "start": float(values[0]),
        "end": float(values[-1]),
        "delta": delta,
        "delta_z": delta_z,
        "range": value_range,
        "range_z": range_z,
        "max": float(values.max()),
        "min": float(values.min()),
        "max_pos_frac": float(max_pos / max(len(values) - 1, 1)),
        "min_pos_frac": float(min_pos / max(len(values) - 1, 1)),
        "trend": trend,
    }


def check_description_against_stats(
    description: str,
    stats: Dict[str, float],
    trend_threshold: float,
    flat_range_threshold: float,
    sharp_delta_threshold: float,
) -> List[str]:
    issues = []
    says_rise = contains_any(description, RISE_WORDS)
    says_fall = contains_any(description, FALL_WORDS)
    says_flat = contains_any(description, FLAT_WORDS)
    says_sharp = contains_any(description, SHARP_WORDS)
    says_peak = contains_any(description, PEAK_WORDS)
    says_low = contains_any(description, LOW_WORDS)

    if says_rise and stats["delta_z"] < -trend_threshold:
        issues.append("caption_says_rise_but_delta_is_negative")
    if says_fall and stats["delta_z"] > trend_threshold:
        issues.append("caption_says_fall_but_delta_is_positive")
    if says_flat and abs(stats["delta_z"]) > trend_threshold and stats["range_z"] > flat_range_threshold:
        issues.append("caption_says_flat_but_segment_changes")
    if says_sharp and abs(stats["delta_z"]) < sharp_delta_threshold:
        issues.append("caption_says_sharp_but_delta_is_small")
    if says_peak and stats["range_z"] < flat_range_threshold:
        issues.append("caption_mentions_peak_but_range_is_small")
    if says_low and stats["range_z"] < flat_range_threshold:
        issues.append("caption_mentions_low_but_range_is_small")
    if says_rise and says_fall and abs(stats["delta_z"]) > trend_threshold:
        issues.append("caption_has_both_rise_and_fall_keywords")

    return issues


def plot_sample(
    output_path: str,
    series: np.ndarray,
    caption_segments: Dict[int, str],
    segment_stats: Dict[int, Dict[str, float]],
    segment_issues: Dict[int, List[str]],
) -> None:
    indices_by_segment = np.array_split(np.arange(len(series)), 4)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), dpi=140, sharey=True)
    if len(axes.shape) == 0:
        axes = [axes]

    for segment_id, (ax, indices) in enumerate(zip(axes, indices_by_segment), start=1):
        values = series[indices]
        color = "tab:red" if segment_issues[segment_id] else "tab:blue"
        ax.plot(indices, values, color=color, linewidth=1.7)
        ax.axhline(values[0], color="gray", linewidth=0.8, alpha=0.4)
        stats = segment_stats[segment_id]
        issue_text = ", ".join(segment_issues[segment_id]) if segment_issues[segment_id] else "ok"
        title = (
            f"Segment {segment_id} | trend={stats['trend']} | "
            f"delta_z={stats['delta_z']:.2f} | range_z={stats['range_z']:.2f} | {issue_text}"
        )
        ax.set_title(title, fontsize=8, loc="left")
        ax.text(
            0.01,
            0.04,
            caption_segments[segment_id],
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )
        ax.grid(alpha=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Check whether *_my_text_caps.npy segment captions roughly match time-series trends."
    )
    parser.add_argument("--dataset-root", type=str, default="data/VerbalTSDatasets/istanbul_traffic")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ts-path", type=str, default=None)
    parser.add_argument("--caps-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="logs/caption_ts_alignment/istanbul_traffic")
    parser.add_argument("--trend-threshold", type=float, default=0.25)
    parser.add_argument("--flat-range-threshold", type=float, default=0.35)
    parser.add_argument("--sharp-delta-threshold", type=float, default=0.7)
    parser.add_argument("--max-plots", type=int, default=50)
    parser.add_argument("--plot-ok-samples", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ts_path = args.ts_path or os.path.join(args.dataset_root, f"{args.split}_ts.npy")
    caps_path = args.caps_path or os.path.join(args.dataset_root, f"{args.split}_my_text_caps.npy")
    os.makedirs(args.output_dir, exist_ok=True)

    ts = np.load(ts_path, allow_pickle=True)
    captions = flatten_caption_array(np.load(caps_path, allow_pickle=True))
    if ts.ndim != 3:
        raise ValueError(f"Expected time series shape (N, T, C), got {ts.shape} from {ts_path}")
    if ts.shape[2] != 1:
        raise ValueError(f"This checker expects one channel, got C={ts.shape[2]} from {ts_path}")
    if ts.shape[0] != len(captions):
        raise ValueError(
            f"Count mismatch: ts has N={ts.shape[0]}, captions has N={len(captions)}. "
            f"ts_path={ts_path}, caps_path={caps_path}"
        )

    total = ts.shape[0] if args.limit is None else min(ts.shape[0], args.limit)
    global_std = float(ts[:total, :, 0].std() + 1e-8)
    rows = []
    bad_caption_parse = []
    suspicious_sample_indices = set()

    for sample_idx in range(total):
        series = ts[sample_idx, :, 0]
        try:
            caption_segments = parse_segments(captions[sample_idx])
        except ValueError as exc:
            bad_caption_parse.append({"index": sample_idx, "error": str(exc), "caption": captions[sample_idx]})
            suspicious_sample_indices.add(sample_idx)
            continue

        indices_by_segment = np.array_split(np.arange(len(series)), 4)
        for segment_id, indices in enumerate(indices_by_segment, start=1):
            stats = classify_segment(series[indices], global_std, args.trend_threshold)
            issues = check_description_against_stats(
                caption_segments[segment_id],
                stats,
                trend_threshold=args.trend_threshold,
                flat_range_threshold=args.flat_range_threshold,
                sharp_delta_threshold=args.sharp_delta_threshold,
            )
            if issues:
                suspicious_sample_indices.add(sample_idx)
            rows.append(
                {
                    "index": sample_idx,
                    "segment": segment_id,
                    "caption": caption_segments[segment_id],
                    "trend": stats["trend"],
                    "delta": stats["delta"],
                    "delta_z": stats["delta_z"],
                    "range": stats["range"],
                    "range_z": stats["range_z"],
                    "max_pos_frac": stats["max_pos_frac"],
                    "min_pos_frac": stats["min_pos_frac"],
                    "issues": ";".join(issues),
                }
            )

    csv_path = os.path.join(args.output_dir, f"{args.split}_my_text_caps_alignment.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "segment",
                "caption",
                "trend",
                "delta",
                "delta_z",
                "range",
                "range_z",
                "max_pos_frac",
                "min_pos_frac",
                "issues",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    parse_error_path = os.path.join(args.output_dir, f"{args.split}_caption_parse_errors.json")
    with open(parse_error_path, "w", encoding="utf-8") as f:
        json.dump(bad_caption_parse, f, indent=2, ensure_ascii=False)

    plotted = 0
    plot_indices = list(sorted(suspicious_sample_indices))
    if args.plot_ok_samples and plotted < args.max_plots:
        ok_indices = [idx for idx in range(total) if idx not in suspicious_sample_indices]
        plot_indices.extend(ok_indices)

    plot_dir = os.path.join(args.output_dir, f"{args.split}_plots")
    for sample_idx in plot_indices:
        if plotted >= args.max_plots:
            break
        try:
            caption_segments = parse_segments(captions[sample_idx])
        except ValueError:
            continue
        series = ts[sample_idx, :, 0]
        segment_stats = {}
        segment_issues = {}
        for segment_id, indices in enumerate(np.array_split(np.arange(len(series)), 4), start=1):
            stats = classify_segment(series[indices], global_std, args.trend_threshold)
            segment_stats[segment_id] = stats
            segment_issues[segment_id] = check_description_against_stats(
                caption_segments[segment_id],
                stats,
                trend_threshold=args.trend_threshold,
                flat_range_threshold=args.flat_range_threshold,
                sharp_delta_threshold=args.sharp_delta_threshold,
            )
        plot_sample(
            output_path=os.path.join(plot_dir, f"{sample_idx:06d}.png"),
            series=series,
            caption_segments=caption_segments,
            segment_stats=segment_stats,
            segment_issues=segment_issues,
        )
        plotted += 1

    issue_rows = [row for row in rows if row["issues"]]
    summary = {
        "ts_path": ts_path,
        "caps_path": caps_path,
        "ts_shape": list(ts.shape),
        "caption_count": len(captions),
        "checked_samples": total,
        "checked_segments": len(rows),
        "caption_parse_errors": len(bad_caption_parse),
        "suspicious_segments": len(issue_rows),
        "suspicious_samples": len(suspicious_sample_indices),
        "csv_path": csv_path,
        "parse_error_path": parse_error_path,
        "plot_dir": plot_dir,
        "num_plots": plotted,
        "thresholds": {
            "trend_threshold": args.trend_threshold,
            "flat_range_threshold": args.flat_range_threshold,
            "sharp_delta_threshold": args.sharp_delta_threshold,
            "global_std": global_std,
        },
    }
    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if issue_rows:
        print("\nFirst suspicious segments:")
        for row in issue_rows[:10]:
            print(
                f"idx={row['index']} seg={row['segment']} trend={row['trend']} "
                f"delta_z={float(row['delta_z']):.2f} issues={row['issues']} | {row['caption']}"
            )


if __name__ == "__main__":
    main()
