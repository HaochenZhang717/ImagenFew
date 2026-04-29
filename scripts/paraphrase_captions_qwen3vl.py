import argparse
import json
import math
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
NUM_PARAPHRASES = 3
DEFAULT_MAX_NEW_TOKENS = 128


SEGMENT_PARAPHRASE_PROMPT = r"""
Paraphrase the following single time-series segment description into exactly three different sentences while preserving its meaning.

Requirements:
- Produce exactly three paraphrases.
- Do not add new observations that are not in the original caption.
- Do not mention any other segment.
- Each paraphrase must be one concise and factual.
- The three paraphrases should use different wording.
- Do not include segment labels.

Use exactly this output format:
[Paraphrase 1]: <sentence>
[Paraphrase 2]: <sentence>
[Paraphrase 3]: <sentence>

Original segment description:
{segment_text}
""".strip()

SEGMENT_PATTERN = re.compile(
    r"\[Segment\s+([1-4])\]\s*:?\s*(.*?)(?=\n\s*\[Segment\s+[1-4]\]\s*:|\Z)",
    flags=re.DOTALL,
)
PARAPHRASE_PATTERN = re.compile(
    r"\[Paraphrase\s+([1-3])\]\s*:?\s*(.*?)(?=\n\s*\[Paraphrase\s+[1-3]\]\s*:|\Z)",
    flags=re.DOTALL | re.I,
)


def get_generation_kwargs(max_new_tokens: int, do_sample: bool) -> Dict:
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = 0.8
        generation_kwargs["top_p"] = 0.95
    return generation_kwargs


def split_range(total: int, part_id: int, num_parts: int) -> Tuple[int, int]:
    chunk_size = math.ceil(total / num_parts)
    start = part_id * chunk_size
    end = min((part_id + 1) * chunk_size, total)
    return start, end


def flatten_caption_array(caption_array: np.ndarray) -> List[str]:
    if caption_array.ndim == 0:
        raise ValueError("Expected an array of captions, got a scalar npy.")
    if caption_array.ndim > 2:
        raise ValueError(
            f"Expected a 1D or 2D caption array, got shape={caption_array.shape}."
        )
    return [str(item) for item in caption_array.reshape(-1)]


def parse_segments(caption: str) -> Dict[int, str]:
    segments = {}
    for match in SEGMENT_PATTERN.finditer(caption.strip()):
        segment_id = int(match.group(1))
        segment_text = " ".join(match.group(2).strip().split())
        segments[segment_id] = segment_text
    if sorted(segments) != [1, 2, 3, 4]:
        raise ValueError(f"Expected four [Segment i] entries, got segment ids={sorted(segments)}")
    return segments


def make_messages(segment_text: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": SEGMENT_PARAPHRASE_PROMPT.format(segment_text=segment_text),
                }
            ],
        }
    ]


def normalize_paraphrase(text: str) -> str:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return text


def clean_paraphrase_sentence(text: str) -> str:
    text = normalize_paraphrase(text)
    text = re.sub(r"^\s*\[?Paraphrase\s+[1-3]\]?\s*:?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:[1-3][\).\:\-]|[-*])\s*", "", text)
    text = re.sub(r"^\s*(Paraphrased|Rewritten)\s*(sentence|description)?\s*:?\s*", "", text, flags=re.I)
    return " ".join(text.split())


def parse_segment_paraphrases(text: str) -> List[str]:
    text = normalize_paraphrase(text)
    paraphrases = {}
    for match in PARAPHRASE_PATTERN.finditer(text):
        paraphrase_id = int(match.group(1))
        paraphrases[paraphrase_id] = clean_paraphrase_sentence(match.group(2))

    if sorted(paraphrases) == [1, 2, 3]:
        return [paraphrases[idx] for idx in range(1, 4)]

    lines = [clean_paraphrase_sentence(line) for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if line]
    if len(lines) == 3:
        return lines

    raise ValueError(f"Expected exactly three paraphrases, got {len(lines)} lines: {text!r}")


def original_variant_matrix(caption: str) -> List[List[str]]:
    segments = parse_segments(caption)
    original_segments = [segments[idx] for idx in range(1, 5)]
    return [original_segments.copy() for _ in range(NUM_PARAPHRASES + 1)]


def is_valid_variant_matrix(matrix: Sequence[Sequence[str]]) -> bool:
    if len(matrix) != NUM_PARAPHRASES + 1:
        return False
    return all(len(row) == 4 and all(str(item).strip() for item in row) for row in matrix)


@torch.inference_mode()
def paraphrase_segment_texts(
    segment_texts: Sequence[str],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
) -> List[str]:
    messages_batch = [make_messages(segment_text) for segment_text in segment_texts]
    texts = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in messages_batch
    ]
    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    ).to(device, non_blocking=True)

    generated_ids = model.generate(
        **inputs,
        **get_generation_kwargs(max_new_tokens=max_new_tokens, do_sample=do_sample),
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [normalize_paraphrase(text) for text in output_texts]


def paraphrase_captions_by_segment(
    captions: Sequence[str],
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
) -> List[List[List[str]]]:
    parsed_caption_segments = [parse_segments(caption) for caption in captions]
    flat_segment_refs = []
    flat_segment_texts = []
    for caption_idx, segments in enumerate(parsed_caption_segments):
        for segment_id in range(1, 5):
            flat_segment_refs.append((caption_idx, segment_id))
            flat_segment_texts.append(segments[segment_id])

    flat_paraphrases = paraphrase_segment_texts(
        flat_segment_texts,
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    caption_variant_matrices = []
    for segments in parsed_caption_segments:
        original_segments = [segments[segment_id] for segment_id in range(1, 5)]
        caption_variant_matrices.append(
            [original_segments.copy()] + [["" for _ in range(4)] for _ in range(NUM_PARAPHRASES)]
        )

    for (caption_idx, segment_id), paraphrase_text in zip(flat_segment_refs, flat_paraphrases):
        segment_paraphrases = parse_segment_paraphrases(paraphrase_text)
        for paraphrase_idx, sentence in enumerate(segment_paraphrases, start=1):
            caption_variant_matrices[caption_idx][paraphrase_idx][segment_id - 1] = sentence

    return caption_variant_matrices


def load_existing_records(jsonl_path: str) -> Dict[int, List[List[str]]]:
    records = {}
    if not os.path.exists(jsonl_path):
        return records
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                matrix = obj["caption_variants"]
                if is_valid_variant_matrix(matrix):
                    records[int(obj["index"])] = [
                        [str(item) for item in row] for row in matrix
                    ]
            except Exception:
                continue
    return records


def save_output_npy(
    input_array: np.ndarray,
    records: Dict[int, List[List[str]]],
    output_npy: str,
    require_complete: bool,
) -> bool:
    flat_input = input_array.reshape(-1)
    total = len(flat_input)
    missing = [idx for idx in range(total) if idx not in records]
    if missing and require_complete:
        print(
            f"[WARN] Not saving npy yet: {len(missing)} captions are missing. "
            f"First missing index: {missing[0]}."
        )
        return False

    output_array = np.empty((total, NUM_PARAPHRASES + 1, 4), dtype=object)
    for idx, original_caption in enumerate(flat_input):
        matrix = records.get(idx)
        if matrix is None:
            matrix = original_variant_matrix(str(original_caption))
        output_array[idx] = np.asarray(matrix, dtype=object)

    output_array = output_array.astype(str)
    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
    np.save(output_npy, output_array)
    print(f"[DONE] Saved paraphrased caption npy: {output_npy} shape={output_array.shape}")
    return True


def iter_batches(items: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase VerbalTSDatasets caption npy files with Qwen3-VL."
    )
    parser.add_argument(
        "--input-npy",
        type=str,
        default="data/VerbalTSDatasets/istanbul_traffic/train_my_text_caps.npy",
        help="Input caption npy path.",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        default=None,
        help="Output paraphrased caption npy path. Defaults to <input stem>_paraphrased.npy.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="logs/paraphrased_captions_qwen3vl",
        help="Directory for resumable jsonl outputs.",
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--part-id",
        "--part_id",
        dest="part_id",
        type=int,
        default=0,
        help="Which split to run (0-based).",
    )
    parser.add_argument(
        "--num-parts",
        "--num_parts",
        dest="num_parts",
        type=int,
        default=1,
        help="Total number of splits.",
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use stochastic decoding for more diverse paraphrases.",
    )
    parser.add_argument(
        "--allow-partial-npy",
        action="store_true",
        help=(
            "Save npy even when other parts are not finished; missing entries "
            "are filled with original segments in all variant slots."
        ),
    )
    parser.add_argument(
        "--fallback-original-on-bad-format",
        action="store_true",
        help=(
            "If parsing generated paraphrases fails, fill that index with "
            "original segments in all variant slots."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick debugging within the selected part.",
    )
    args = parser.parse_args()

    assert 0 <= args.part_id < args.num_parts, "part_id must be in [0, num_parts)"
    if args.output_npy is None:
        base, ext = os.path.splitext(args.input_npy)
        args.output_npy = f"{base}_paraphrased{ext or '.npy'}"

    caption_array = np.load(args.input_npy, allow_pickle=True)
    captions = flatten_caption_array(caption_array)
    total = len(captions)
    start_idx, end_idx = split_range(total, args.part_id, args.num_parts)
    part_indices = list(range(start_idx, end_idx))
    if args.limit is not None:
        part_indices = part_indices[:args.limit]

    os.makedirs(args.save_dir, exist_ok=True)
    output_jsonl = os.path.join(
        args.save_dir,
        f"{os.path.basename(args.output_npy)}.part{args.part_id}_{args.num_parts}.jsonl",
    )
    error_jsonl = os.path.join(
        args.save_dir,
        f"{os.path.basename(args.output_npy)}.part{args.part_id}_{args.num_parts}.errors.jsonl",
    )

    existing_records = load_existing_records(output_jsonl)
    pending_indices = [idx for idx in part_indices if idx not in existing_records]

    print(f"[INFO] Loading input npy: {args.input_npy} shape={caption_array.shape}")
    print(f"[INFO] Running part {args.part_id}/{args.num_parts} -> range [{start_idx}, {end_idx})")
    print(f"[INFO] Pending captions in this run: {len(pending_indices)}")

    num_success = 0
    num_fail = 0

    if pending_indices:
        print(f"[INFO] Loading model: {args.model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            device_map=None,
        ).eval().to(device)
        processor = AutoProcessor.from_pretrained(args.model_name)
        print(f"[INFO] Model loaded on {device}.")

        fout = open(output_jsonl, "a", encoding="utf-8")
        ferr = open(error_jsonl, "a", encoding="utf-8")

        for batch_indices in tqdm(
            iter_batches(pending_indices, args.batch_size),
            total=math.ceil(len(pending_indices) / args.batch_size),
            desc=f"Paraphrasing part {args.part_id}",
        ):
            batch_captions = [captions[idx] for idx in batch_indices]
            try:
                caption_variant_matrices = paraphrase_captions_by_segment(
                    batch_captions,
                    model=model,
                    processor=processor,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                )
            except Exception as batch_error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for idx, caption in zip(batch_indices, batch_captions):
                    try:
                        caption_variant_matrix = paraphrase_captions_by_segment(
                            [caption],
                            model=model,
                            processor=processor,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                        )[0]
                    except Exception as item_error:
                        if args.fallback_original_on_bad_format:
                            caption_variant_matrix = original_variant_matrix(caption)
                        else:
                            ferr.write(
                                json.dumps(
                                    {
                                        "index": idx,
                                        "error": (
                                            f"batch_error={str(batch_error)} | "
                                            f"item_error={str(item_error)}"
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            ferr.flush()
                            num_fail += 1
                            continue

                    if args.fallback_original_on_bad_format and not is_valid_variant_matrix(
                        caption_variant_matrix
                    ):
                        caption_variant_matrix = original_variant_matrix(caption)
                    record = {
                        "index": idx,
                        "caption": caption,
                        "caption_variants": caption_variant_matrix,
                        "model": args.model_name,
                        "part_id": args.part_id,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    existing_records[idx] = caption_variant_matrix
                    num_success += 1
                continue

            for idx, caption, caption_variant_matrix in zip(
                batch_indices, batch_captions, caption_variant_matrices
            ):
                if args.fallback_original_on_bad_format and not is_valid_variant_matrix(
                    caption_variant_matrix
                ):
                    caption_variant_matrix = original_variant_matrix(caption)
                record = {
                    "index": idx,
                    "caption": caption,
                    "caption_variants": caption_variant_matrix,
                    "model": args.model_name,
                    "part_id": args.part_id,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_records[idx] = caption_variant_matrix
                num_success += 1
            fout.flush()

        fout.close()
        ferr.close()
    else:
        print("[INFO] No pending captions for this part; skipping model loading.")

    print(f"[DONE] Success: {num_success}, Failed: {num_fail}")
    print(f"[DONE] Part jsonl saved to: {output_jsonl}")
    print(f"[DONE] Errors saved to: {error_jsonl}")

    merged_records = {}
    for part_id in range(args.num_parts):
        part_jsonl = os.path.join(
            args.save_dir,
            f"{os.path.basename(args.output_npy)}.part{part_id}_{args.num_parts}.jsonl",
        )
        merged_records.update(load_existing_records(part_jsonl))

    save_output_npy(
        input_array=caption_array,
        records=merged_records,
        output_npy=args.output_npy,
        require_complete=not args.allow_partial_npy,
    )


if __name__ == "__main__":
    main()
