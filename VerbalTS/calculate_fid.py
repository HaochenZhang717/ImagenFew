import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fid_vae import FIDVAE

from scipy.linalg import sqrtm
from metrics.discriminative_torch import discriminative_score_metrics


def compute_fid(real, fake):
    """
    real, fake: (N, D)
    """

    mu_r = np.mean(real, axis=0)
    mu_f = np.mean(fake, axis=0)

    sigma_r = np.cov(real, rowvar=False)
    sigma_f = np.cov(fake, rowvar=False)

    covmean = sqrtm(sigma_r @ sigma_f)

    # 数值稳定（必须）
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_r - mu_f) ** 2) + np.trace(
        sigma_r + sigma_f - 2 * covmean
    )

    return float(fid)

# =========================
# KID (Kernel Inception Distance)
# =========================
def compute_kid(real, fake):
    """
    real, fake: (N, D)
    unbiased MMD^2 with polynomial kernel
    """

    def polynomial_kernel(a, b):
        return ((a @ b.T) / a.shape[1] + 1) ** 3

    K_XX = polynomial_kernel(real, real)
    K_YY = polynomial_kernel(fake, fake)
    K_XY = polynomial_kernel(real, fake)

    n = real.shape[0]
    m = fake.shape[0]

    kid = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        - 2 * K_XY.mean()
    )

    return float(kid)


# =========================
# CMMD (Conditional MMD / Context MMD)
# =========================
def compute_cmmd(real, fake):
    """
    real, fake: (N, D)
    使用 RBF kernel 的 MMD（更稳定）
    """

    def rbf_kernel(a, b, sigma=1.0):
        a_norm = (a ** 2).sum(axis=1).reshape(-1, 1)
        b_norm = (b ** 2).sum(axis=1).reshape(1, -1)
        dist = a_norm + b_norm - 2 * a @ b.T
        return np.exp(-dist / (2 * sigma ** 2))

    K_XX = rbf_kernel(real, real)
    K_YY = rbf_kernel(fake, fake)
    K_XY = rbf_kernel(real, fake)

    cmmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

    return float(cmmd)
# =========================
# Args
# =========================
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--real_path", type=str, required=True)
    parser.add_argument("--fake_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)

    # model config（必须一致）
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)
    # parser.add_argument("--num_samples", type=int, default=64)

    # embedding 类型
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default="mean",
    #     choices=["mean", "full", "flatten"]
    # )

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


# =========================
# Dataset
# =========================
def load_dataset(dict_path, dict_key, idx=-1):
    data = torch.load(dict_path, weights_only=False)
    data = data[dict_key]
    # data = data[:args.num_samples]
    if dict_key == "sampled_ts":
        if idx > -1:
            data = data[idx]
    if data.shape[1] > data.shape[2]:
        data = data.permute(0,2,1)
    # print(f"Loaded {dict_path}, key: {dict_key}: {data.shape}")
    # print(f"max: {data.max().item()}, min: {data.min().item()}")
    return TensorDataset(data.float())


# =========================
# Extract Embedding
# =========================
@torch.no_grad()
def extract_embeddings(model, dataloader, device):

    model.eval()

    all_embeddings = []

    for batch in dataloader:
        x = batch[0].to(device)
        with torch.no_grad():
            out = model(x)
        mu = out["mu"]
        emb = mu
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).cpu().numpy()


# =========================
# Main
# =========================
def main(args):

    device = args.device if torch.cuda.is_available() else "cpu"

    real_dataset = load_dataset(args.real_path, "real_ts", idx=-1)
    real_dataloader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

    sample = real_dataset[0][0]
    C, T = sample.shape

    # ===== load model =====
    model = FIDVAE(
        input_dim=C,
        output_dim=C,
        seq_len=T,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device).eval()

    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    print(f"Loaded checkpoint from {args.ckpt_path}")

    real_embeddings = extract_embeddings(
        model,
        real_dataloader,
        device,
    )


    fid_list = []
    kid_list = []
    cmmd_list = []
    for i in range(1):
        fake_dataset = load_dataset(args.fake_path, "sampled_ts",idx=i)
        # fake_dataset = load_dataset(args.fake_path, "sampled_ts",idx=i)
        fake_dataloader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False)

        # ===== extract =====
        fake_embeddings = extract_embeddings(
            model,
            fake_dataloader,
            device,
        ) # (N, seq_len, dim)

        # print("Real embeddings:", real_embeddings.shape)
        # print("Fake embeddings:", fake_embeddings.shape)
        num_embeds = real_embeddings.shape[0]
        fake_embeddings = fake_embeddings[:num_embeds]
        # ===== compute FID =====
        fid = compute_fid(real_embeddings, fake_embeddings)
        kid = compute_kid(real_embeddings, fake_embeddings)
        cmmd = compute_cmmd(real_embeddings, fake_embeddings)

        fid_list.append(fid)
        kid_list.append(kid)
        cmmd_list.append(cmmd)

    fid_array = np.array(fid_list)
    kid_array = np.array(kid_list)
    cmmd_array = np.array(cmmd_list)

    print("\n==========================")
    print("real_path: {}".format(args.real_path))
    print("fake_path: {}".format(args.fake_path))
    print(f"FID:  ${fid_array.mean():.4f} \\pm {fid_array.std():.4f}$")
    print(f"KID:  ${kid_array.mean():.6f} \\pm {kid_array.std():.6f}$")
    print(f"CMMD: ${cmmd_array.mean():.6f} \\pm {cmmd_array.std():.6f}$")
    print("==========================\n")



if __name__ == "__main__":
    args = get_args()
    main(args)