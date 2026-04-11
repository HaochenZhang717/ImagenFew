import argparse
import os

import torch
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info

from stage1_model import Qwen3VisionEncoder, Stage1_Qwen3VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MIN_PIXELS = 0
MAX_PIXELS = 32 * 14 * 14

PROMPT_TEXT = r"""
You are given one time-series image for a single channel.

Describe the time series in exactly three stages:
1. early stage
2. middle stage
3. late stage

For each stage, briefly describe the trend, level change, or stability pattern.
If there is a clear local event such as a spike, dip, abrupt rise, abrupt drop, or sharp fluctuation, explicitly mention it.

Rules:
- Keep each line concise and factual.
- Description of each stage should be a short sentence.
- Local event description should also be a short sentence.
- If clear local event does not exist, just put <not exist>

Use exactly the following output format:
Output exactly:
Early: <description>; Middle: <description>; Late: <description>; Local Event: <description>
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare direct image captioning vs GT-embedding decode captioning."
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to one image like image0_ch0.png")
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default=None,
        help="Optional directory containing precomputed shards like train_rank0.pt/train_rank1.pt. If absent, encode image online.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def build_message(image_path: str):
    return [[
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
                    "text": PROMPT_TEXT,
                },
            ],
        }
    ]]


def build_generation_inputs(
    processor: Qwen3VLProcessor,
    image_path: str,
    device: torch.device,
):
    prompt_messages = build_message(image_path)
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in prompt_messages
    ]
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)


def caption_from_image(
    model: Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    image_path: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = build_message(image_path)
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def load_precomputed_latent(precomputed_dir: str, image_name: str) -> torch.Tensor:
    shard_paths = sorted(
        os.path.join(precomputed_dir, name)
        for name in os.listdir(precomputed_dir)
        if name.endswith(".pt") and "_rank" in name
    )
    if not shard_paths:
        raise ValueError(f"No precomputed shards found in {precomputed_dir}")

    found = []
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        if image_name in shard:
            found.append((shard_path, shard[image_name]))

    if not found:
        raise KeyError(f"{image_name} was not found in any precomputed shard under {precomputed_dir}")

    latent = found[0][1]
    for shard_path, candidate in found[1:]:
        if candidate.shape != latent.shape or not torch.equal(candidate, latent):
            raise ValueError(
                f"Found non-identical duplicated latent for {image_name} across precomputed shards."
            )
    return latent


def encode_image_online(
    image_path: str,
    processor: Qwen3VLProcessor,
    vision_encoder: Qwen3VisionEncoder,
    device: torch.device,
) -> torch.Tensor:
    messages = build_message(image_path)
    image_inputs, video_inputs = process_vision_info(messages)
    proc = processor(
        text=[""],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    pixel_values = proc["pixel_values"].to(device)
    image_grid_thw = proc["image_grid_thw"].to(device)
    with torch.no_grad():
        latent = vision_encoder.encode_images(pixel_values, image_grid_thw)
    return latent.squeeze(0).cpu()


def decode_from_latent(
    stage1_model: Stage1_Qwen3VLForConditionalGeneration,
    processor: Qwen3VLProcessor,
    image_path: str,
    latent_2d: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    c, h, w = latent_2d.shape
    latent_tokens = latent_2d.unsqueeze(0).permute(0, 2, 3, 1).reshape(h * w, c).to(device)
    generation_inputs = build_generation_inputs(
        processor=processor,
        image_path=image_path,
        device=device,
    )
    with torch.no_grad():
        generated_ids = stage1_model.generate(
            **generation_inputs,
            shallow_visual_latent=latent_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(generation_inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def main():
    args = parse_args()
    image_name = os.path.basename(args.image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Qwen3VLProcessor.from_pretrained(MODEL_NAME)
    direct_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        device_map=None,
    ).eval().to(device)

    stage1_model = Stage1_Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        device_map=None,
    ).eval().to(device)

    direct_text = caption_from_image(
        model=direct_model,
        processor=processor,
        image_path=args.image_path,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    if args.precomputed_dir:
        latent = load_precomputed_latent(args.precomputed_dir, image_name)
        latent_source = f"precomputed:{args.precomputed_dir}"
    else:
        vision_encoder = Qwen3VisionEncoder(MODEL_NAME).eval().to(device)
        latent = encode_image_online(
            image_path=args.image_path,
            processor=processor,
            vision_encoder=vision_encoder,
            device=device,
        )
        latent_source = "online-encoded"

    decoded_text = decode_from_latent(
        stage1_model=stage1_model,
        processor=processor,
        image_path=args.image_path,
        latent_2d=latent,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    exact_match = direct_text == decoded_text
    normalized_match = normalize_text(direct_text) == normalize_text(decoded_text)

    print(f"image_path: {args.image_path}")
    print(f"image_name: {image_name}")
    print(f"latent_source: {latent_source}")
    print(f"latent_shape: {tuple(latent.shape)} dtype={latent.dtype}")
    print(f"exact_match: {exact_match}")
    print(f"normalized_match: {normalized_match}")
    print("\n=== Direct image caption ===")
    print(direct_text)
    print("\n=== GT embedding decode caption ===")
    print(decoded_text)


if __name__ == "__main__":
    main()
