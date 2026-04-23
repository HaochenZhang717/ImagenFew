import argparse
import os

from omegaconf import OmegaConf

try:
    from caption_generator.pipeline_v2_utils import (
        DeterministicTimeSeriesRenderer,
        compute_global_stats,
        load_split_arrays,
        render_and_save_split,
        save_json,
    )
except ImportError:
    from pipeline_v2_utils import (
        DeterministicTimeSeriesRenderer,
        compute_global_stats,
        load_split_arrays,
        render_and_save_split,
        save_json,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-render time-series images for pipeline_v2 training.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
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
    renderer = DeterministicTimeSeriesRenderer(**cfg["renderer"])

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
            )
        )

    manifest = {
        "dataset_root": cfg["data"]["dataset_root"],
        "prerendered_image_root": output_root,
        "normalize_before_render": bool(cfg["data"].get("normalize_before_render", False)),
        "renderer": cfg["renderer"],
        "train_stats": stats,
        "splits": split_summaries,
    }
    save_json(manifest, os.path.join(output_root, "manifest.json"))


if __name__ == "__main__":
    main()
