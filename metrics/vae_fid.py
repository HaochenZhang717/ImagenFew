import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm
import torch
import torch.nn as nn
import torch.nn.functional as F

def _load_fid_vae_class():
    # module_path = Path(__file__).resolve().parents[1] / "VerbalTS" / "fid_vae.py"
    # spec = importlib.util.spec_from_file_location("verbalts_fid_vae", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load FIDVAE from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FIDVAE


# FIDVAE = _load_fid_vae_class()
# from fid_vae import FIDVAE


class NormCausalAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        mlp_ratio = 4.0

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = NormCausalAttention(hidden_size, num_heads=num_heads)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(
            hidden_size,
            int(2 / 3 * mlp_hidden_dim),
            hidden_size
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FIDEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.to_mu = nn.Linear(hidden_size, latent_dim)
        self.to_logvar = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        """
        x: (B, C, T)
        return:
            mu:     (B, latent_dim)
            logvar: (B, latent_dim)
        """
        x = self.conv(x)              # (B, hidden, T')
        x = x.permute(0, 2, 1)        # (B, T', hidden)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # global pooling over time
        x = x.mean(dim=1)             # (B, hidden)

        mu = self.to_mu(x)            # (B, latent_dim)
        logvar = self.to_logvar(x)    # (B, latent_dim)

        # 可选：数值稳定
        logvar = torch.clamp(logvar, min=-6.0, max=6.0)

        return mu, logvar


class FIDDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        output_dim,
        hidden_size=128,
        seq_len=128,
    ):
        super().__init__()

        assert seq_len % 4 == 0, "seq_len must be divisible by 4"
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.base_len = seq_len // 4

        self.fc = nn.Linear(latent_dim, hidden_size * self.base_len)

        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv1d(hidden_size // 2, output_dim, kernel_size=5, padding=2),
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        return: (B, C, T)
        """
        B = z.shape[0]

        x = self.fc(z)                                # (B, hidden * T/4)
        x = x.view(B, self.hidden_size, self.base_len)
        x = self.net(x)                               # (B, C, T)

        return x


class FIDVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        seq_len,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        latent_dim=64,
        beta=0.001,
    ):
        super().__init__()

        self.beta = beta

        self.encoder = FIDEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
        )

        self.decoder = FIDDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            seq_len=seq_len,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        x: (B, C, T)
        returns:
            mu, logvar, z with shape (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)

    def get_embedding(self, x, use_mu=True):
        """
        x: (B, C, T)
        return: (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        if use_mu:
            return mu
        return self.reparameterize(mu, logvar)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()

        loss = recon_loss + self.beta * kl_loss

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }



_DEFAULT_CKPT_DIR_CANDIDATES = (
    os.getenv("FID_VAE_CKPT_ROOT"),
    "./fid_vae_ckpts",
    "../fid_vae_ckpts",
    "/playpen-shared/haochenz/ImagenFew/fid_vae_ckpts",
)

_DATASET_TO_CKPT_DIR = {
    "synthetic_u": "vae_synth_u",
    "synthetic_m": "vae_synth_m",
    "ETTm1": "vae_ettm1",
    "ettm1": "vae_ettm1",
    "istanbul_traffic": "vae_istanbul_traffic",
    "Weather": "vae_weather",
    "weather": "vae_weather",
    "glucose": "vae_glucose",
    "glucose_daily": "vae_glucose_daily",
}


def _compute_fid(real_embeddings, fake_embeddings):
    mu_r = np.mean(real_embeddings, axis=0)
    mu_f = np.mean(fake_embeddings, axis=0)

    sigma_r = np.cov(real_embeddings, rowvar=False)
    sigma_f = np.cov(fake_embeddings, rowvar=False)

    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu_r - mu_f) ** 2) + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)


def _resolve_ckpt_root(ckpt_root=None):
    if ckpt_root:
        return ckpt_root
    for candidate in _DEFAULT_CKPT_DIR_CANDIDATES:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def _resolve_ckpt_path(dataset, ckpt_root=None, ckpt_name="best.pt"):
    ckpt_root = _resolve_ckpt_root(ckpt_root)
    if ckpt_root is None:
        raise FileNotFoundError(
            "Could not find fid_vae_ckpts root. Set FID_VAE_CKPT_ROOT or pass vae_ckpt_root explicitly."
        )

    # 1) Prefer direct folder naming: <ckpt_root>/<dataset>/<ckpt_name>
    direct_path = os.path.join(ckpt_root, dataset, ckpt_name)
    if os.path.exists(direct_path):
        return direct_path

    # 2) Backward-compatible mapped folder naming
    ckpt_dir_name = _DATASET_TO_CKPT_DIR.get(dataset)
    if ckpt_dir_name is not None:
        mapped_path = os.path.join(ckpt_root, ckpt_dir_name, ckpt_name)
        if os.path.exists(mapped_path):
            return mapped_path

    raise FileNotFoundError(
        f"Could not find FID-VAE checkpoint for dataset '{dataset}'. "
        f"Tried: {direct_path}"
        + (
            f", {os.path.join(ckpt_root, ckpt_dir_name, ckpt_name)}"
            if ckpt_dir_name is not None
            else ""
        )
    )


def _to_bct(data):
    if torch.is_tensor(data):
        tensor = data.detach().cpu()
    else:
        tensor = torch.as_tensor(data)
    tensor = tensor.to(torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D time-series tensor/array, got shape {tuple(tensor.shape)}")
    if tensor.shape[1] > tensor.shape[2]:
        tensor = tensor.permute(0, 2, 1)
    return tensor.contiguous()


def _extract_embeddings(model, data, device, batch_size=128):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, data.shape[0], batch_size):
            batch = data[start:start + batch_size].to(device)
            out = model(batch)
            embeddings.append(out["mu"].detach().cpu())
    return torch.cat(embeddings, dim=0).numpy()


def _load_model_state_dict(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return state


def _infer_ckpt_input_dim(state_dict):
    # encoder.conv.0.weight shape: (hidden_size//2, input_dim, kernel)
    return int(state_dict["encoder.conv.0.weight"].shape[1])


def _infer_vae_hparams_from_state(state_dict):
    # encoder.to_mu.weight: (latent_dim, hidden_size)
    to_mu_w = state_dict["encoder.to_mu.weight"]
    latent_dim = int(to_mu_w.shape[0])
    hidden_size = int(to_mu_w.shape[1])

    # Infer encoder depth from encoder.layers.<idx>.*
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.append(int(parts[2]))
    num_layers = (max(layer_indices) + 1) if layer_indices else 2

    # Heuristic for heads: prefer 8 when divisible, then 4/2/1
    if hidden_size % 8 == 0:
        num_heads = 8
    elif hidden_size % 4 == 0:
        num_heads = 4
    elif hidden_size % 2 == 0:
        num_heads = 2
    else:
        num_heads = 1

    # decoder.fc.weight: (hidden_size * (seq_len // 4), latent_dim)
    fc_w = state_dict["decoder.fc.weight"]
    seq_len = int((fc_w.shape[0] // hidden_size) * 4)
    return {
        "latent_dim": latent_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
    }


def _load_model_state(model, state_dict):
    model.load_state_dict(state_dict, strict=True)
    return model


def VAE_FID(
    ori_data,
    generated_data,
    dataset,
    device,
    vae_ckpt_root=None,
    vae_ckpt_name="best.pt",
    batch_size=128,
):
    real_tensor = _to_bct(ori_data)
    fake_tensor = _to_bct(generated_data)
    channels = real_tensor.shape[1]

    dataset_candidates = [dataset]
    if channels == 1 and not str(dataset).endswith("_one_channel"):
        dataset_candidates.insert(0, f"{dataset}_one_channel")

    selected = None
    tried_msgs = []
    for dataset_name in dataset_candidates:
        try:
            ckpt_path = _resolve_ckpt_path(dataset_name, vae_ckpt_root, ckpt_name=vae_ckpt_name)
        except FileNotFoundError as exc:
            tried_msgs.append(str(exc))
            continue

        state_dict = _load_model_state_dict(ckpt_path, device)
        ckpt_in_dim = _infer_ckpt_input_dim(state_dict)
        if ckpt_in_dim != channels:
            tried_msgs.append(
                f"Found checkpoint at {ckpt_path}, but input_dim={ckpt_in_dim} "
                f"does not match evaluated channels={channels}."
            )
            continue
        selected = (dataset_name, ckpt_path, state_dict)
        break

    if selected is None:
        raise RuntimeError(
            "No compatible FID-VAE checkpoint was found for evaluation.\n"
            + "\n".join(tried_msgs)
        )

    _, ckpt_path, state_dict = selected
    hp = _infer_vae_hparams_from_state(state_dict)

    model = FIDVAE(
        input_dim=channels,
        output_dim=channels,
        seq_len=hp["seq_len"],
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        num_heads=hp["num_heads"],
        latent_dim=hp["latent_dim"],
    ).to(device).eval()
    model = _load_model_state(model, state_dict)
    print("real_tensor size:", real_tensor.shape)
    print("fake_tensor size:", fake_tensor.shape)
    real_embeddings = _extract_embeddings(model, real_tensor, device=device, batch_size=batch_size)
    fake_embeddings = _extract_embeddings(model, fake_tensor, device=device, batch_size=batch_size)
    fake_embeddings = fake_embeddings[: real_embeddings.shape[0]]

    return _compute_fid(real_embeddings, fake_embeddings)
