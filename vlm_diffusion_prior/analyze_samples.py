import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA


def embed_jsonl_to_pt(
    input_jsonl: str,
    output_pt: str,
    text_key: str = "caption",
    batch_size: int = 32,
    prompt_name: str = None,
    device: str = "cuda"
):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    model.eval().to(device)

    all_embeddings = []
    all_ids = []

    buffer_texts = []
    buffer_ids = []

    with open(input_jsonl, "r") as f:
        for line_idx, line in enumerate(tqdm(f, desc="Reading JSONL")):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            text = obj.get(text_key, None)
            if text is None:
                continue

            sample_id = obj.get("sample_id", line_idx)

            buffer_texts.append(text)
            buffer_ids.append(sample_id)

            if len(buffer_texts) >= batch_size:
                emb = model.encode(
                    buffer_texts,
                    prompt_name=prompt_name,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

                all_embeddings.append(emb.cpu())
                all_ids.extend(buffer_ids)

                buffer_texts = []
                buffer_ids = []

        # last batch
        if len(buffer_texts) > 0:
            emb = model.encode(
                buffer_texts,
                prompt_name=prompt_name,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            all_embeddings.append(emb.cpu())
            all_ids.extend(buffer_ids)

    embeddings = torch.cat(all_embeddings, dim=0)  # shape: (N, D)
    ids = torch.tensor(all_ids)

    save_obj = {
        "embeddings": embeddings,   # torch.FloatTensor [N, D]
        "sample_ids": ids,          # torch.LongTensor [N]
        "text_key": text_key,
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
    }

    torch.save(save_obj, output_pt)
    print(f"Saved embeddings to {output_pt}")
    print("embeddings shape:", embeddings.shape)




def visualize_umap(train_path, gen_path, save_name):

    train = torch.load(train_path)
    gen = torch.load(gen_path)

    train_np = train["embeddings"].detach().cpu().numpy()
    gen_np   = gen["embeddings"].detach().cpu().numpy()

    X = np.concatenate([train_np, gen_np], axis=0)
    labels = np.array([0] * len(train_np) + [1] * len(gen_np))

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1)
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(Z[labels==0, 0], Z[labels==0, 1], s=10, alpha=0.5, label="Train")
    plt.scatter(Z[labels==1, 0], Z[labels==1, 1], s=10, alpha=0.5, label="Generated")
    plt.legend()
    plt.title("UMAP visualization")
    plt.grid(True)
    # plt.show()
    plt.savefig(save_name)





def compute_fid(train_path, gen_path, eps=1e-6):
    """
    train: (N, d) torch.Tensor
    gen:   (M, d) torch.Tensor
    Returns: fid score (float)
    """
    train = torch.load(train_path)
    gen = torch.load(gen_path)

    X = train["embeddings"].detach().cpu().numpy()
    Y = gen["embeddings"].detach().cpu().numpy()

    mu_x = np.mean(X, axis=0)
    mu_y = np.mean(Y, axis=0)

    sigma_x = np.cov(X, rowvar=False)
    sigma_y = np.cov(Y, rowvar=False)

    diff = mu_x - mu_y

    # sqrt of product
    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)

    # numerical stability: handle imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_x + sigma_y - 2 * covmean)

    # avoid negative due to numerical error
    if fid < 0:
        fid = 0.0

    return float(fid)


def compute_fid_pca(train_path, gen_path, pca_dim=64, eps=1e-6):
    """
    Compute FID between train and generated embeddings, after PCA reduction.

    train_path: path to torch file containing {"embeddings": Tensor(N, d)}
    gen_path:   path to torch file containing {"embeddings": Tensor(M, d)}
    pca_dim:    reduced dimension for PCA
    eps:        small regularization term for numerical stability

    Returns:
        fid (float)
    """

    train = torch.load(train_path, map_location="cpu")
    gen   = torch.load(gen_path, map_location="cpu")

    X = train["embeddings"].detach().cpu().numpy()
    Y = gen["embeddings"].detach().cpu().numpy()

    # -----------------------------
    # 1) PCA on combined embeddings
    # -----------------------------
    Z = np.concatenate([X, Y], axis=0)

    pca_dim = min(pca_dim, Z.shape[1])  # cannot exceed original dim
    pca = PCA(n_components=pca_dim, svd_solver="auto", random_state=0)
    Zp = pca.fit_transform(Z)

    Xp = Zp[: len(X)]
    Yp = Zp[len(X):]

    # -----------------------------
    # 2) compute mean and covariance
    # -----------------------------
    mu_x = np.mean(Xp, axis=0)
    mu_y = np.mean(Yp, axis=0)

    sigma_x = np.cov(Xp, rowvar=False)
    sigma_y = np.cov(Yp, rowvar=False)

    # regularization for stability
    sigma_x += np.eye(sigma_x.shape[0]) * eps
    sigma_y += np.eye(sigma_y.shape[0]) * eps

    diff = mu_x - mu_y

    # -----------------------------
    # 3) sqrt of product of covariances
    # -----------------------------
    covmean, _ = linalg.sqrtm(sigma_x @ sigma_y, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_x + sigma_y - 2 * covmean)

    # numerical precision sometimes makes fid slightly negative
    if fid < 0:
        fid = 0.0

    return float(fid)




if __name__ == "__main__":

    # embed_jsonl_to_pt(
    #     input_jsonl="decoded_samples.jsonl",
    #     output_pt="../../decoded_samples_emb.pt",
    #     text_key="caption",
    #     batch_size=32,
    #     prompt_name=None,
    #     device="cuda:2"
    # )
    #
    # embed_jsonl_to_pt(
    #     input_jsonl="../step_1_dataset_construction/synthetic_u_caption/time_series_caps_3072.jsonl",
    #     output_pt="../../training_samples_emb_3072.pt",
    #     text_key="caption",
    #     batch_size=32,
    #     prompt_name=None,
    #     device="cuda:2"
    # )

    embed_jsonl_to_pt(
        input_jsonl="../../DiTDH-XL-samples.jsonl",
        output_pt="../../DiTDH-XL-samples-emb.pt",
        text_key="caption",
        batch_size=32,
        prompt_name=None,
        device="cuda:2"
    )

    visualize_umap(
        train_path="../../training_samples_emb_3072.pt",
        gen_path="../../DiTDH-XL-samples-emb.pt",
        save_name="DiTDH-XL-UMAP-06B.png"
    )

    fid64 = compute_fid_pca(
        "../../training_samples_emb_3072.pt",
        "../../DiTDH-XL-samples-emb.pt",
        pca_dim=64
    )
    fid128 = compute_fid_pca(
        "../../training_samples_emb_3072.pt",
        "../../DiTDH-XL-samples-emb.pt",
        pca_dim=128
    )
    fid = compute_fid(
        train_path="../../training_samples_emb_3072.pt",
        gen_path="../../DiTDH-XL-samples-emb.pt",
    )
    print("XL FID: ")
    print("FID@PCA64:", fid64)
    print("FID@PCA128:", fid128)
    print("FID:", fid)



    fid64 = compute_fid_pca(
        "../../training_samples_emb_3072.pt",
        "../../decoded_samples_emb.pt",
        pca_dim=64
    )
    fid128 = compute_fid_pca(
        "../../training_samples_emb_3072.pt",
        "../../decoded_samples_emb.pt",
        pca_dim=128
    )
    fid = compute_fid(
        train_path="../../training_samples_emb_3072.pt",
        gen_path="../../decoded_samples_emb.pt",
    )
    print("S FID: ")
    print("FID@PCA64:", fid64)
    print("FID@PCA128:", fid128)
    print("FID:", fid)

