#!/usr/bin/env python
import argparse
import json
from collections import Counter

import numpy as np


def normalize_text(text):
    return " ".join(str(text).strip().split()).lower()


def row_key(row):
    return tuple(normalize_text(segment) for segment in np.asarray(row).reshape(-1))


def analyze(path, examples):
    caps = np.load(path, allow_pickle=True)
    if caps.ndim < 3:
        raise ValueError(f"Expected paraphrased captions shaped like [N, variants, segments], got {caps.shape}")

    n_samples, n_variants, n_segments = caps.shape[:3]

    unique_variant_counts = []
    all_variants_identical = []
    has_any_rephrase = []
    variant_duplicate_hist = Counter()
    segment_unique_counts = []
    segment_all_identical_counts = Counter()

    for idx in range(n_samples):
        sample = caps[idx]
        keys = [row_key(sample[variant_id]) for variant_id in range(n_variants)]
        unique_count = len(set(keys))
        unique_variant_counts.append(unique_count)
        all_same = unique_count == 1
        all_variants_identical.append(all_same)
        has_any_rephrase.append(unique_count > 1)
        variant_duplicate_hist[unique_count] += 1

        identical_segments = 0
        for segment_id in range(n_segments):
            segment_values = [
                normalize_text(sample[variant_id, segment_id])
                for variant_id in range(n_variants)
            ]
            seg_unique = len(set(segment_values))
            segment_unique_counts.append(seg_unique)
            if seg_unique == 1:
                identical_segments += 1
        segment_all_identical_counts[identical_segments] += 1

    repeated_indices = [idx for idx, same in enumerate(all_variants_identical) if same]
    rephrased_indices = [idx for idx, flag in enumerate(has_any_rephrase) if flag]

    summary = {
        "path": path,
        "shape": list(caps.shape),
        "n_samples": n_samples,
        "n_variants": n_variants,
        "n_segments": n_segments,
        "samples_with_all_variants_identical": len(repeated_indices),
        "samples_with_any_rephrase": len(rephrased_indices),
        "sample_percent_all_variants_identical": 100.0 * len(repeated_indices) / n_samples,
        "sample_percent_any_rephrase": 100.0 * len(rephrased_indices) / n_samples,
        "unique_variant_count_histogram": dict(sorted(variant_duplicate_hist.items())),
        "segments_total": len(segment_unique_counts),
        "segments_with_all_variants_identical": sum(count == 1 for count in segment_unique_counts),
        "segments_with_any_rephrase": sum(count > 1 for count in segment_unique_counts),
        "segment_percent_all_variants_identical": 100.0 * sum(count == 1 for count in segment_unique_counts) / len(segment_unique_counts),
        "segment_percent_any_rephrase": 100.0 * sum(count > 1 for count in segment_unique_counts) / len(segment_unique_counts),
        "identical_segment_count_per_sample_histogram": dict(sorted(segment_all_identical_counts.items())),
        "first_repeated_indices": repeated_indices[:examples],
        "first_rephrased_indices": rephrased_indices[:examples],
    }

    return caps, summary


def print_examples(caps, indices, title):
    print(f"\n{title}")
    if not indices:
        print("  <none>")
        return

    for idx in indices:
        print("=" * 80)
        print(f"idx={idx}")
        for variant_id, segments in enumerate(caps[idx]):
            print(f"  variant {variant_id}:")
            for segment_id, text in enumerate(segments, start=1):
                print(f"    Segment {segment_id}: {text}")


def main():
    parser = argparse.ArgumentParser(description="Analyze duplicated paraphrased caption variants.")
    parser.add_argument(
        "--path",
        default="../data/VerbalTSDatasets/istanbul_traffic/train_my_text_caps_paraphrased.npy",
        help="Path to *_my_text_caps_paraphrased.npy",
    )
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Print only JSON summary.")
    args = parser.parse_args()

    caps, summary = analyze(args.path, args.examples)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(json.dumps(summary, indent=2))
    print_examples(caps, summary["first_repeated_indices"], "Examples: all variants identical")
    print_examples(caps, summary["first_rephrased_indices"], "Examples: has real rephrasing")


if __name__ == "__main__":
    main()
