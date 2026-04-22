import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


RE_VAR = re.compile(r"This sequence is\s+([A-Z]+)\.")
RE_SEASON = re.compile(r"The main season cycles is around\s+([^\.]+)\.")
RE_OVERALL = re.compile(r"For the ovearll shape,\s*([^\.]+)\.")
RE_DIST = re.compile(r"The distribution of the value in time series is\s+([^\.]+)\.")
RE_SENTENCE = re.compile(r"[^.]+\.")
RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align captions with attrs_idx, mine reusable sentence templates, "
            "and export suggested special tokens for template-aware prompting/training."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset directory containing train/valid/test *_text_caps.npy and *_attrs_idx.npy files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save alignment, template statistics, and special token suggestions.",
    )
    parser.add_argument(
        "--min-template-count",
        type=int,
        default=20,
        help="Minimum occurrence count for a template to be assigned a special token.",
    )
    parser.add_argument(
        "--token-prefix",
        type=str,
        default="<CAPTPL_",
        help="Prefix of generated special token names.",
    )
    parser.add_argument(
        "--token-suffix",
        type=str,
        default=">",
        help="Suffix of generated special token names.",
    )
    parser.add_argument(
        "--max-token-templates",
        type=int,
        default=256,
        help="Maximum number of templates to convert into special tokens.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_meta(dataset_dir: str) -> Dict:
    meta_path = os.path.join(dataset_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_caption_item(item) -> str:
    if isinstance(item, np.ndarray):
        flat = item.reshape(-1)
        if len(flat) == 0:
            return ""
        return str(flat[0]).strip()
    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return ""
        return str(item[0]).strip()
    return str(item).strip()


def parse_caption_fields(caption: str) -> Dict[str, Optional[str]]:
    var_match = RE_VAR.search(caption)
    season_match = RE_SEASON.search(caption)
    overall_match = RE_OVERALL.search(caption)
    dist_match = RE_DIST.search(caption)
    return {
        "var_phrase": var_match.group(1).strip() if var_match else None,
        "season_phrase": season_match.group(1).strip() if season_match else None,
        "overall_phrase": overall_match.group(1).strip() if overall_match else None,
        "distribution_phrase": dist_match.group(1).strip() if dist_match else None,
    }


def split_sentences(caption: str) -> List[str]:
    sentences = []
    for raw in RE_SENTENCE.findall(caption.replace("\n", " ")):
        cleaned = " ".join(raw.split())
        if cleaned:
            sentences.append(cleaned)
    if not sentences:
        fallback = " ".join(caption.replace("\n", " ").split())
        if fallback:
            sentences.append(fallback)
    return sentences


def normalize_sentence_template(sentence: str) -> str:
    s = " ".join(sentence.split())
    if s.startswith("This sequence is "):
        return "This sequence is <VAR_NAME>."
    if s.startswith("The main season cycles is around "):
        return "The main season cycles is around <SEASON_CYCLE>."
    if s.startswith("For the ovearll shape,"):
        return "For the ovearll shape, <OVERALL_TREND>."
    if s.startswith("The distribution of the value in time series is "):
        return "The distribution of the value in time series is <SKEWNESS_KURTOSIS>."
    s = RE_NUMBER.sub("<NUM>", s)
    return s


def dominant_mapping(counter: Counter) -> Tuple[object, int, float]:
    total = int(sum(counter.values()))
    if total == 0:
        return -1, 0, 0.0
    value, count = counter.most_common(1)[0]
    return value, int(count), float(count / total)


def summarize_phrase_mapping(records: Sequence[Dict], attr_names: Sequence[str]) -> Dict:
    by_field = {
        "var_phrase": 0,
        "overall_phrase": 1,
        "season_phrase": 2,
        "distribution_phrase": (3, 4),
    }
    output: Dict[str, Dict] = {}

    for field, target_dim in by_field.items():
        phrase_to_counter = defaultdict(Counter)
        for r in records:
            phrase = r["parsed_fields"].get(field)
            if phrase is None:
                continue
            attrs_idx = r["attrs_idx"]
            if isinstance(target_dim, tuple):
                value = tuple(int(attrs_idx[d]) for d in target_dim)
            else:
                value = int(attrs_idx[target_dim])
            phrase_to_counter[phrase][value] += 1

        phrase_stats = []
        for phrase, cnt in sorted(phrase_to_counter.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
            total = int(sum(cnt.values()))
            top_value, top_count = cnt.most_common(1)[0]
            phrase_stats.append(
                {
                    "phrase": phrase,
                    "total": total,
                    "top_value": list(top_value) if isinstance(top_value, tuple) else int(top_value),
                    "top_count": int(top_count),
                    "purity": float(top_count / total),
                    "value_histogram": {
                        ",".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v)
                        for k, v in cnt.items()
                    },
                }
            )

        if isinstance(target_dim, tuple):
            target_name = [attr_names[d] if d < len(attr_names) else str(d) for d in target_dim]
        else:
            target_name = attr_names[target_dim] if target_dim < len(attr_names) else str(target_dim)

        output[field] = {
            "target_attr": target_name,
            "num_phrases": len(phrase_to_counter),
            "num_ambiguous_phrases": sum(1 for x in phrase_stats if x["purity"] < 1.0),
            "phrases": phrase_stats,
        }
    return output


def mine_templates(records: Sequence[Dict], attr_names: Sequence[str]) -> Dict:
    template_counter = Counter()
    template_examples: Dict[str, str] = {}
    template_attrs = defaultdict(Counter)

    for r in records:
        attrs_tuple = tuple(int(x) for x in r["attrs_idx"])
        for sentence in split_sentences(r["caption"]):
            template = normalize_sentence_template(sentence)
            template_counter[template] += 1
            template_attrs[template][attrs_tuple] += 1
            template_examples.setdefault(template, sentence)

    templates = []
    for template, count in template_counter.most_common():
        dom_attr, dom_count, dom_ratio = dominant_mapping(template_attrs[template])
        templates.append(
            {
                "template": template,
                "count": int(count),
                "num_distinct_attrs_idx": int(len(template_attrs[template])),
                "dominant_attrs_idx": list(dom_attr) if isinstance(dom_attr, tuple) else dom_attr,
                "dominant_ratio": float(dom_ratio),
                "example_sentence": template_examples[template],
            }
        )

    return {
        "num_unique_templates": len(template_counter),
        "templates": templates,
        "attr_names": list(attr_names),
    }


def build_special_tokens(
    template_stats: Dict,
    min_template_count: int,
    max_token_templates: int,
    token_prefix: str,
    token_suffix: str,
) -> Dict:
    selected = []
    for item in template_stats["templates"]:
        if item["count"] < min_template_count:
            continue
        selected.append(item)
        if len(selected) >= max_token_templates:
            break

    tokens = []
    for i, item in enumerate(selected, start=1):
        token = f"{token_prefix}{i:03d}{token_suffix}"
        tokens.append(
            {
                "token": token,
                "template": item["template"],
                "count": item["count"],
                "example_sentence": item["example_sentence"],
            }
        )

    return {
        "num_selected_templates": len(tokens),
        "min_template_count": int(min_template_count),
        "max_token_templates": int(max_token_templates),
        "tokens": tokens,
        "additional_special_tokens": [t["token"] for t in tokens],
    }


def write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    dataset_dir = os.path.abspath(args.dataset_dir)
    meta = load_meta(dataset_dir)
    attr_names = meta.get("attr_list", [])

    all_records = []
    split_stats = {}

    for split in args.splits:
        caps_path = os.path.join(dataset_dir, f"{split}_text_caps.npy")
        attrs_path = os.path.join(dataset_dir, f"{split}_attrs_idx.npy")
        if not os.path.exists(caps_path):
            raise FileNotFoundError(f"Missing file: {caps_path}")
        if not os.path.exists(attrs_path):
            raise FileNotFoundError(f"Missing file: {attrs_path}")

        caps = np.load(caps_path, allow_pickle=True)
        attrs = np.load(attrs_path, allow_pickle=True)
        if len(caps) != len(attrs):
            raise ValueError(f"Split {split}: caps length {len(caps)} != attrs length {len(attrs)}")

        records = []
        for i in range(len(caps)):
            caption = normalize_caption_item(caps[i])
            attrs_idx = [int(x) for x in np.asarray(attrs[i]).tolist()]
            parsed_fields = parse_caption_fields(caption)
            record = {
                "split": split,
                "index": i,
                "sample_id": f"{split}:{i}",
                "caption": caption,
                "attrs_idx": attrs_idx,
                "parsed_fields": parsed_fields,
            }
            records.append(record)

        all_records.extend(records)
        split_stats[split] = {
            "num_samples": len(records),
            "caps_path": os.path.abspath(caps_path),
            "attrs_path": os.path.abspath(attrs_path),
        }
        write_jsonl(os.path.join(args.output_dir, f"aligned_{split}.jsonl"), records)

    phrase_mapping = summarize_phrase_mapping(all_records, attr_names)
    template_stats = mine_templates(all_records, attr_names)
    special_tokens = build_special_tokens(
        template_stats=template_stats,
        min_template_count=args.min_template_count,
        max_token_templates=args.max_token_templates,
        token_prefix=args.token_prefix,
        token_suffix=args.token_suffix,
    )

    summary = {
        "dataset_dir": dataset_dir,
        "splits": args.splits,
        "split_stats": split_stats,
        "num_total_samples": len(all_records),
        "attr_names": attr_names,
        "output_files": {
            "aligned_jsonl_per_split": [f"aligned_{s}.jsonl" for s in args.splits],
            "phrase_mapping_json": "phrase_attr_mapping.json",
            "template_stats_json": "template_stats.json",
            "special_tokens_json": "special_tokens.json",
            "special_tokens_txt": "special_tokens.txt",
        },
    }

    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    write_json(os.path.join(args.output_dir, "phrase_attr_mapping.json"), phrase_mapping)
    write_json(os.path.join(args.output_dir, "template_stats.json"), template_stats)
    write_json(os.path.join(args.output_dir, "special_tokens.json"), special_tokens)

    with open(os.path.join(args.output_dir, "special_tokens.txt"), "w", encoding="utf-8") as f:
        for tok in special_tokens["additional_special_tokens"]:
            f.write(tok + "\n")

    print(f"Saved outputs to: {os.path.abspath(args.output_dir)}")
    print(f"Total records: {len(all_records)}")
    print(f"Unique templates: {template_stats['num_unique_templates']}")
    print(f"Selected special tokens: {special_tokens['num_selected_templates']}")


if __name__ == "__main__":
    main()
