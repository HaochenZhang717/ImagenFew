import argparse
import json

import torch
from omegaconf import OmegaConf

try:
    from caption_generator.pipeline_v2_model import PipelineV2CaptionModel
    from caption_generator.pipeline_v2_utils import (
        DeterministicTimeSeriesRenderer,
        PreRenderedTimeSeriesImageCaptionDataset,
        TimeSeriesImageCaptionDataset,
        compute_global_stats,
        load_split_arrays,
    )
except ImportError:
    from pipeline_v2_model import PipelineV2CaptionModel
    from pipeline_v2_utils import (
        DeterministicTimeSeriesRenderer,
        PreRenderedTimeSeriesImageCaptionDataset,
        TimeSeriesImageCaptionDataset,
        compute_global_stats,
        load_split_arrays,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions with the pipeline_v2 model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def resolve_device(cfg):
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def main():
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    device = resolve_device(cfg)

    model = PipelineV2CaptionModel(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    train_ts, _, _ = load_split_arrays(cfg["data"]["dataset_root"], "train")
    stats = compute_global_stats(train_ts)
    image_root = cfg["data"].get("prerendered_image_root")
    use_prerendered = bool(cfg["data"].get("use_prerendered_images", True)) and bool(image_root)
    if use_prerendered:
        dataset = PreRenderedTimeSeriesImageCaptionDataset(
            dataset_root=cfg["data"]["dataset_root"],
            image_root=image_root,
            split=args.split,
            prompt_text=cfg["data"]["instruction_prompt"],
        )
    else:
        renderer = DeterministicTimeSeriesRenderer(**cfg["renderer"])
        dataset = TimeSeriesImageCaptionDataset(
            dataset_root=cfg["data"]["dataset_root"],
            split=args.split,
            prompt_text=cfg["data"]["instruction_prompt"],
            renderer=renderer,
            normalization_stats=stats,
            normalize_before_render=cfg["data"].get("normalize_before_render", False),
        )

    sample = dataset[args.index]
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": cfg["data"]["instruction_prompt"]},
            ],
        }
    ]
    prompt_text = model.processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = model.processor(
        text=[prompt_text],
        images=[sample["image"]],
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    generated_ids = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        pixel_values=encoded["pixel_values"],
        image_grid_thw=encoded["image_grid_thw"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=model.processor.tokenizer.eos_token_id,
    )
    prompt_len = int(encoded["attention_mask"].sum(dim=1).item())
    new_token_ids = generated_ids[:, prompt_len:]
    prediction = model.processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print(
        json.dumps(
            {
                "split": args.split,
                "index": args.index,
                "prompt": cfg["data"]["instruction_prompt"],
                "prediction": prediction,
                "ground_truth": sample["caption_text"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
