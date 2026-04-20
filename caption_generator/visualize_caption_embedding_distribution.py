import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed generated and real captions with Qwen embeddings and visualize their distribution."
    )
    parser.add_argument("--generated-path", type=str, required=True, help="Path to generated_text_caps.npy")
    parser.add_argument("--real-path", type=str, required=True, help="Path to real/test_text_caps.npy")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save plots and stats")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        type=str,
        default="Given a time-series caption, embed it for semantic comparison with other time-series captions",
    )
    parser.add_argument("--use-instruct", action="store_true", help="Wrap each caption with Qwen instruct formatting")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--save-embeddings", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def flatten_caption_array(path: str) -> List[str]:
    arr = np.load(path, allow_pickle=True)
    flattened = []
    for item in arr:
        if isinstance(item, np.ndarray):
            flattened.append(str(item.reshape(-1)[0]))
        elif isinstance(item, (list, tuple)):
            flattened.append(str(item[0]))
        else:
            flattened.append(str(item))
    return flattened


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def fit_pca(x: np.ndarray, n_components: int = 2):
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:n_components]
    transformed = x_centered @ components.T
    explained_variance_ratio = (s[:n_components] ** 2) / np.sum(s ** 2)
    return transformed, explained_variance_ratio


def fit_tsne(x: np.ndarray, perplexity: float, seed: int):
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, max(5.0, len(x) - 1.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(x), perplexity


def cosine_similarity_stats(generated: np.ndarray, real: np.ndarray):
    cross = generated @ real.T
    gen_self = generated @ generated.T
    real_self = real @ real.T
    np.fill_diagonal(gen_self, np.nan)
    np.fill_diagonal(real_self, np.nan)
    return {
        "generated_to_real_mean": float(np.mean(cross)),
        "generated_to_real_std": float(np.std(cross)),
        "generated_self_mean": float(np.nanmean(gen_self)),
        "generated_self_std": float(np.nanstd(gen_self)),
        "real_self_mean": float(np.nanmean(real_self)),
        "real_self_std": float(np.nanstd(real_self)),
        "generated_to_real_top1_mean": float(np.mean(np.max(cross, axis=1))),
    }


def plot_scatter(points: np.ndarray, labels: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    colors = {"generated": "#d55e00", "real": "#0072b2"}
    for label in ["generated", "real"]:
        mask = labels == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=16,
            alpha=0.65,
            c=colors[label],
            label=label,
        )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_similarity_hist(generated: np.ndarray, real: np.ndarray, out_path: Path):
    cross = (generated @ real.T).reshape(-1)
    gen_self = (generated @ generated.T)
    real_self = (real @ real.T)
    gen_self = gen_self[~np.eye(gen_self.shape[0], dtype=bool)]
    real_self = real_self[~np.eye(real_self.shape[0], dtype=bool)]

    plt.figure(figsize=(8, 6))
    plt.hist(gen_self, bins=60, alpha=0.5, label="generated self-similarity", density=True)
    plt.hist(real_self, bins=60, alpha=0.5, label="real self-similarity", density=True)
    plt.hist(cross, bins=60, alpha=0.5, label="generated vs real", density=True)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_texts = flatten_caption_array(args.generated_path)
    real_texts = flatten_caption_array(args.real_path)

    if args.use_instruct:
        generated_inputs = [get_detailed_instruct(args.task, text) for text in generated_texts]
        real_inputs = [get_detailed_instruct(args.task, text) for text in real_texts]
    else:
        generated_inputs = generated_texts
        real_inputs = real_texts

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    generated_embeddings = encode_texts(
        generated_inputs, tokenizer, model, batch_size=args.batch_size, max_length=args.max_length
    ).numpy()
    real_embeddings = encode_texts(
        real_inputs, tokenizer, model, batch_size=args.batch_size, max_length=args.max_length
    ).numpy()

    all_embeddings = np.concatenate([generated_embeddings, real_embeddings], axis=0)
    labels = np.array(["generated"] * len(generated_embeddings) + ["real"] * len(real_embeddings))

    pca_points, explained = fit_pca(all_embeddings, n_components=2)
    plot_scatter(
        pca_points,
        labels,
        output_dir / "pca_scatter.png",
        title=f"PCA of caption embeddings (explained={explained[0]:.3f}, {explained[1]:.3f})",
    )

    tsne_points, used_perplexity = fit_tsne(all_embeddings, perplexity=args.tsne_perplexity, seed=args.seed)
    plot_scatter(
        tsne_points,
        labels,
        output_dir / "tsne_scatter.png",
        title=f"t-SNE of caption embeddings (perplexity={used_perplexity:.1f})",
    )

    plot_similarity_hist(generated_embeddings, real_embeddings, output_dir / "cosine_similarity_hist.png")

    stats = {
        "model_name": args.model_name,
        "generated_path": os.path.abspath(args.generated_path),
        "real_path": os.path.abspath(args.real_path),
        "num_generated": len(generated_texts),
        "num_real": len(real_texts),
        "embedding_dim": int(generated_embeddings.shape[1]),
        "pca_explained_variance_ratio": [float(explained[0]), float(explained[1])],
        "tsne_perplexity": float(used_perplexity),
    }
    stats.update(cosine_similarity_stats(generated_embeddings, real_embeddings))

    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if args.save_embeddings:
        torch.save(
            {
                "generated_texts": generated_texts,
                "real_texts": real_texts,
                "generated_embeddings": torch.from_numpy(generated_embeddings),
                "real_embeddings": torch.from_numpy(real_embeddings),
            },
            output_dir / "caption_embeddings.pt",
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
