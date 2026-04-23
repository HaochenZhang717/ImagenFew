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
    renderer = DeterministicTimeSeriesRenderer(**cfg["renderer"])
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
        "train_stats": stats,
        "splits": split_summaries,
    }
    save_json(manifest, os.path.join(output_root, "manifest.json"))


if __name__ == "__main__":
    main()
