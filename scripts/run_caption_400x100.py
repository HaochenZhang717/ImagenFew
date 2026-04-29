import os
import json
import glob
import math
import argparse
import re
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==============================
# Config
# ==============================
model_name = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_NEW_TOKENS = 512

# MIN_PIXELS = 16 * 14 * 14
MIN_PIXELS = 0
MAX_PIXELS = 32 * 14 * 14


TIME_SERIES_CAPTION_PROMPT_SINGLE_CHANNEL = r"""
You are given one 400x100 time-series image for a single channel.
The image is split by red vertical lines into four chronological segments from left to right.

Describe the time series in exactly four segments:
1. segment 1
2. segment 2
3. segment 3
4. segment 4

For each segment, Use one sentence to describe the behavior of time series.

Rules:
- Keep each line concise and factual.
- Description of each segment should be a short sentence.

Use exactly the following output format:
Output exactly:
[Segment 1]: <description> 
[Segment 2]: <description>
[Segment 3]: <description>
[Segment 4]: <description>
""".strip()


# ==============================
# Load model + processor
# ==============================
print(f"[INFO] Loading model: {model_name}")


model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=True,
    device_map=None,
).eval().to(device)

processor = AutoProcessor.from_pretrained(model_name)

print("[INFO] Model loaded successfully.")


def get_generation_kwargs(max_new_tokens: int, do_sample: bool):
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = 1.0
        generation_kwargs["top_p"] = 1.0
    return generation_kwargs


def load_existing_image_set(jsonl_path):
    existing = set()
    if not os.path.exists(jsonl_path):
        return existing

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "image" in obj:
                    existing.add(obj["image"])
            except Exception:
                continue
    return existing


def image_index_from_name(image_name: str) -> int:
    match = re.fullmatch(r"(\d+)\.png", image_name)
    if match is None:
        raise ValueError(f"Expected image filename like 000000.png, got: {image_name}")
    return int(match.group(1))


def validate_image_size(image_paths, expected_width: int, expected_height: int) -> None:
    for image_path in image_paths:
        with Image.open(image_path) as image:
            if image.size != (expected_width, expected_height):
                raise ValueError(
                    f"Expected {expected_width}x{expected_height} image, got {image.size} at {image_path}"
                )


def collect_caption_records(save_dir: str, split: str, num_parts: int):
    index_to_caption = {}
    for part_id in range(num_parts):
        jsonl_path = os.path.join(save_dir, f"{split}_caps_{part_id}_{num_parts}.jsonl")
        if not os.path.exists(jsonl_path):
            continue
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_name = row["image"]
                caption = row["caption"]
                index_to_caption[image_index_from_name(image_name)] = caption
    return index_to_caption


def save_verbalts_caption_npy(
    save_dir: str,
    split: str,
    num_parts: int,
    output_npy: str,
    expected_count: int,
    require_complete: bool,
) -> bool:
    index_to_caption = collect_caption_records(save_dir, split, num_parts)
    missing = [idx for idx in range(expected_count) if idx not in index_to_caption]
    if missing and require_complete:
        print(
            f"[WARN] Not saving npy yet: {len(missing)} captions are missing. "
            f"First missing index: {missing[0]:06d}.png"
        )
        return False

    available_indices = range(expected_count) if not missing else sorted(index_to_caption)
    captions = [index_to_caption[idx] for idx in available_indices]
    caption_array = np.asarray(captions, dtype=str).reshape(-1, 1)
    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
    np.save(output_npy, caption_array)
    print(f"[DONE] Saved VerbalTSDatasets caption npy: {output_npy} shape={caption_array.shape}")
    return True


def build_message(image_path: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": TIME_SERIES_CAPTION_PROMPT_SINGLE_CHANNEL,
                },
            ],
        }
    ]


@torch.inference_mode()
def caption_one_image(
    dataset_name: str,
    image_path: str,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    messages = build_message(image_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device, non_blocking=True)
    generated_ids = model.generate(**inputs, **get_generation_kwargs(max_new_tokens, do_sample))

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()


@torch.inference_mode()
def caption_image_batch(image_paths, max_new_tokens: int, do_sample: bool):
    messages_batch = [build_message(image_path) for image_path in image_paths]
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device, non_blocking=True)

    generated_ids = model.generate(**inputs, **get_generation_kwargs(max_new_tokens, do_sample))

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [text.strip() for text in output_texts]


def split_range(total, part_id, num_parts):
    """
    Split [0, total) into num_parts nearly-equal contiguous chunks.
    Return (start, end) for chunk part_id.
    """
    chunk_size = math.ceil(total / num_parts)
    start = part_id * chunk_size
    end = min((part_id + 1) * chunk_size, total)
    return start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part_id", type=int, default=0, help="which split to run (0-based)")
    parser.add_argument("--num_parts", type=int, default=1, help="total number of splits")
    parser.add_argument("--dataset-root", type=str, default="data/VerbalTSDatasets/synthetic_u")
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_name", type=str, default="synthetic_u")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--output-npy",
        type=str,
        default=None,
        help=(
            "Path to save captions in VerbalTSDatasets format. "
            "Defaults to <dataset-root>/<split>_text_caps.npy."
        ),
    )
    parser.add_argument(
        "--allow-partial-npy",
        action="store_true",
        help="Save output npy even if not all caption parts are finished.",
    )
    parser.add_argument("--expected-width", type=int, default=400)
    parser.add_argument("--expected-height", type=int, default=100)
    parser.add_argument("--quiet", action="store_true", help="Reduce per-image stdout logging")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of images to caption per forward pass")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of newly generated tokens per caption",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use stochastic decoding. Default is greedy decoding for faster, more stable captions.",
    )

    args = parser.parse_args()
    if args.image_root is None:
        args.image_root = os.path.join(args.dataset_root, "pipeline_v2_images_400x100_segments")
    if args.image_folder is None:
        args.image_folder = os.path.join(args.image_root, args.split)
    if args.save_dir is None:
        args.save_dir = os.path.join("logs", "finetune_captions_400x100", args.dataset_name)
    if args.output_npy is None:
        args.output_npy = os.path.join(args.dataset_root, f"{args.split}_my_text_caps.npy")

    os.makedirs(args.save_dir, exist_ok=True)
    output_jsonl = f"{args.save_dir}/{args.split}_caps_{args.part_id}_{args.num_parts}.jsonl"
    error_jsonl = f"{args.save_dir}/{args.split}_errors_{args.part_id}_{args.num_parts}.jsonl"

    assert 0 <= args.part_id < args.num_parts, "part_id must be in [0, num_parts)"

    png_files = sorted(glob.glob(os.path.join(args.image_folder, "*.png")))
    total = len(png_files)
    if total == 0:
        raise FileNotFoundError(f"No PNG files found under image_folder: {args.image_folder}")
    validate_image_size(png_files, args.expected_width, args.expected_height)

    start_idx, end_idx = split_range(total, args.part_id, args.num_parts)
    png_files = png_files[start_idx:end_idx]

    print(f"[INFO] Total PNG files: {total}")
    print(f"[INFO] Running part {args.part_id}/{args.num_parts} -> range [{start_idx}, {end_idx})")
    print(f"[INFO] This part contains {len(png_files)} images")

    processed_images = load_existing_image_set(output_jsonl)
    print(f"[INFO] Already processed {len(processed_images)} images (resume enabled).")

    fout = open(output_jsonl, "a", encoding="utf-8")
    ferr = open(error_jsonl, "a", encoding="utf-8")

    num_success = 0
    num_fail = 0

    pending_png_files = []
    for img_path in png_files:
        img_name = os.path.basename(img_path)
        if img_name not in processed_images:
            pending_png_files.append(img_path)

    for start in tqdm(range(0, len(pending_png_files), args.batch_size), desc=f"Captioning part {args.part_id}"):
        batch_paths = pending_png_files[start:start + args.batch_size]
        batch_names = [os.path.basename(path) for path in batch_paths]

        try:
            captions = caption_image_batch(
                batch_paths,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
            for img_path, img_name, caption in zip(batch_paths, batch_names, captions):
                if not args.quiet:
                    print(f"[INFO] Processing {img_path}")
                    print(caption)
                record = {
                    "image": img_name,
                    "image_path": img_path,
                    "caption": caption,
                    "model": model_name,
                    "part_id": args.part_id,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_success += 1
            fout.flush()

        except Exception as batch_error:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for img_path, img_name in zip(batch_paths, batch_names):
                try:
                    caption = caption_one_image(
                        dataset_name=args.dataset_name,
                        image_path=img_path,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                    )
                    # if not args.quiet:
                    print(f"[INFO] Processing {img_path}")
                    print(caption)
                    record = {
                        "image": img_name,
                        "image_path": img_path,
                        "caption": caption,
                        "model": model_name,
                        "part_id": args.part_id,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    num_success += 1
                except Exception as image_error:
                    err_record = {
                        "image": img_name,
                        "image_path": img_path,
                        "error": f"batch_error={str(batch_error)} | image_error={str(image_error)}",
                        "part_id": args.part_id,
                    }
                    ferr.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                    num_fail += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            ferr.flush()

    fout.close()
    ferr.close()

    print(f"[DONE] Success: {num_success}, Failed: {num_fail}")
    print(f"[DONE] Output saved to: {output_jsonl}")
    print(f"[DONE] Errors saved to: {error_jsonl}")
    if args.output_npy:
        save_verbalts_caption_npy(
            save_dir=args.save_dir,
            split=args.split,
            num_parts=args.num_parts,
            output_npy=args.output_npy,
            expected_count=total,
            require_complete=not args.allow_partial_npy,
        )


if __name__ == "__main__":
    main()


