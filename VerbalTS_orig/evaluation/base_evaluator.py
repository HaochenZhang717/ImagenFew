import os
import time
import torch
import numpy as np
from models.cttp.cttp_model import CTTP
import yaml
import tqdm
import numpy as np
from scipy import linalg
import random
import textwrap
from metrics import evaluate_model_uncond


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class BaseEvaluator:
    def __init__(self, configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_data(dataset)
        if "clip_config_path" in configs.keys():
            self._init_clip(configs)

    def _init_clip(self, configs):
        model_dict = {
            "clip_patchtst": CTTP,
        }
        clip_configs = yaml.safe_load(open(configs["clip_config_path"]))
        self.clip = model_dict[clip_configs["clip_type"]](clip_configs)
        self.clip.load_state_dict(torch.load(configs["clip_model_path"]))
        self.clip = self.clip.to(self.clip.device)

        fid_mean_cache_path = os.path.join(configs["cache_folder"], "fid_mean.npy")
        fid_cov_cache_path = os.path.join(configs["cache_folder"], "fid_cov.npy")
        jftsd_mean_cache_path = os.path.join(configs["cache_folder"], "jftsd_mean.npy")
        jftsd_cov_cache_path = os.path.join(configs["cache_folder"], "jftsd_cov.npy")
        print("cache_folder: ", configs["cache_folder"])
        if os.path.exists(fid_mean_cache_path) and os.path.exists(fid_cov_cache_path) and os.path.exists(jftsd_mean_cache_path) and os.path.exists(jftsd_cov_cache_path):
            self.ts_mean = np.load(fid_mean_cache_path)
            self.ts_cov = np.load(fid_cov_cache_path)
            self.joint_mean = np.load(jftsd_mean_cache_path)
            self.joint_cov = np.load(jftsd_cov_cache_path)
        else:
            train_loader = self.dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=False, include_self=False)
            all_ts_emb, all_joint_emb = [], []
            with torch.no_grad():
                print("calc the ts mean and cov")
                for batch in tqdm.tqdm(train_loader):
                    ts = batch["ts"].to(self.clip.device).float()
                    ts_len = batch["ts_len"].to(self.clip.device).int()
                    cap = batch["cap"]
                    ts_emb = self.clip.get_ts_coemb(ts, ts_len)
                    cap_emb = self.clip.get_text_coemb(cap, None)
                    all_ts_emb.append(ts_emb)
                    all_joint_emb.append(torch.cat([ts_emb,cap_emb], dim=-1))

            all_ts_emb = torch.cat(all_ts_emb, dim=0)
            all_ts_emb = all_ts_emb.cpu().numpy()
            self.ts_mean = np.mean(all_ts_emb, axis=0)
            self.ts_cov = np.cov(all_ts_emb, rowvar=False)
            all_joint_emb = torch.cat(all_joint_emb, dim=0)
            all_joint_emb = all_joint_emb.cpu().numpy()
            self.joint_mean = np.mean(all_joint_emb, axis=0)
            self.joint_cov = np.cov(all_joint_emb, rowvar=False)

            os.makedirs(configs["cache_folder"], exist_ok=True)
            np.save(fid_mean_cache_path, self.ts_mean)
            np.save(fid_cov_cache_path, self.ts_cov)
            np.save(jftsd_mean_cache_path, self.joint_mean)
            np.save(jftsd_cov_cache_path, self.joint_cov)

    def _init_cfgs(self, configs):
        self.configs = configs
        self.batch_size = self.configs["batch_size"]
        self.n_samples = self.configs["n_samples"]
        self.display_epoch_interval = self.configs["display_interval"]
        self.model_path = self.configs["model_path"]
        self.visual_debug_dir = os.path.abspath(
            self.configs.get(
                "visual_debug_dir",
                "verbal_conditional_ts_debug",
            )
        )
        self.n_visual_debug_samples = self.configs.get("n_visual_debug_samples", 12)
        self.visual_debug_inverse_transform = self.configs.get("visual_debug_inverse_transform", True)

    def _inverse_transform_for_visual_debug(self, ts):
        if not self.visual_debug_inverse_transform:
            return ts
        if hasattr(self.dataset, "dataset") and hasattr(self.dataset.dataset, "inverse_transform"):
            return self.dataset.dataset.inverse_transform(ts)
        if hasattr(self.dataset, "inverse_transform"):
            return self.dataset.inverse_transform(ts)
        return ts

    def _visualize_verbal_conditional_generation(self, all_real, all_samples, all_caps):
        if self.n_visual_debug_samples <= 0:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.visual_debug_dir, exist_ok=True)

        real = all_real.detach().cpu().numpy()
        samples = all_samples.detach().cpu().numpy()
        real = self._inverse_transform_for_visual_debug(real)
        samples = self._inverse_transform_for_visual_debug(samples)

        n_items = min(len(real), len(samples), len(all_caps), self.n_visual_debug_samples)
        if n_items == 0:
            print("[Visual Debug] No samples available for visualization.")
            return

        caption_path = os.path.join(self.visual_debug_dir, "caption_index.txt")
        with open(caption_path, "w") as f:
            for idx in range(n_items):
                f.write(f"[{idx}] {all_caps[idx]}\n\n")

        for idx in range(n_items):
            real_i = np.asarray(real[idx])
            sample_i = np.asarray(samples[idx])
            if real_i.ndim == 1:
                real_i = real_i[:, None]
            if sample_i.ndim == 1:
                sample_i = sample_i[:, None]

            mean_real = real_i.mean(axis=-1)
            mean_sample = sample_i.mean(axis=-1)
            n_vars = real_i.shape[-1]
            time_axis = np.arange(real_i.shape[0])

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5),
                dpi=140,
                sharey=True,
            )
            wrapped_cap = textwrap.fill(str(all_caps[idx]), width=120)
            fig.suptitle(f"Verbal condition #{idx}\n{wrapped_cap}", fontsize=10, y=1.04)

            for var_idx in range(n_vars):
                label = f"var {var_idx}" if n_vars <= 8 else None
                axes[0].plot(time_axis, real_i[:, var_idx], linewidth=1.6, alpha=0.85, label=label)
                axes[1].plot(time_axis, sample_i[:, var_idx], linewidth=1.6, alpha=0.85, label=label)
                axes[2].plot(
                    time_axis,
                    real_i[:, var_idx],
                    color="tab:blue",
                    linewidth=1.0,
                    alpha=0.25 if n_vars > 1 else 0.75,
                )
                axes[2].plot(
                    time_axis,
                    sample_i[:, var_idx],
                    color="tab:orange",
                    linewidth=1.0,
                    alpha=0.25 if n_vars > 1 else 0.75,
                )

            if n_vars > 1:
                axes[0].plot(time_axis, mean_real, color="black", linewidth=2.4, label="mean")
                axes[1].plot(time_axis, mean_sample, color="black", linewidth=2.4, label="mean")
                axes[2].plot(time_axis, mean_real, color="tab:blue", linewidth=2.4, label="real mean")
                axes[2].plot(time_axis, mean_sample, color="tab:orange", linewidth=2.4, label="generated mean")
            else:
                axes[2].plot(time_axis, mean_real, color="tab:blue", linewidth=2.4, label="real TS")
                axes[2].plot(time_axis, mean_sample, color="tab:orange", linewidth=2.4, label="generated TS")

            axes[0].set_title("real TS")
            axes[0].set_xlabel("time")
            axes[0].set_ylabel("value")
            axes[1].set_title("generated TS")
            axes[1].set_xlabel("time")
            axes[2].set_title("real vs generated")
            axes[2].set_xlabel("time")

            for ax in axes:
                ax.grid(alpha=0.25)
            if n_vars <= 8:
                axes[0].legend(loc="best")
                axes[1].legend(loc="best")
            axes[2].legend(loc="best")

            fig.tight_layout()
            out_path = os.path.join(self.visual_debug_dir, f"sample_{idx:03d}.png")
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        print(f"[Visual Debug] Saved {n_items} verbal-conditional TS plots to {self.visual_debug_dir}")
        print(f"[Visual Debug] Caption index: {caption_path}")

    def _init_model(self, model):
        self.model = model
        if self.model_path != "":
            print("Loading pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path)['ema_model'])

    def _init_data(self, dataset):
        self.dataset = dataset
        self.test_loader = dataset.get_loader(split="test", batch_size=self.batch_size, shuffle=False, include_self=False)

    """
    Evaluate.
    """
    def evaluate(self, mode="cond_gen", sampler="ddpm", save_pred=False):
        """
        Args:
            mode: cond_gen or edit.
            sampler: ddpm or ddim.
        """
        print("\n-------------------------------")
        print(f"Evaluating the model with mode={mode} and sampler={sampler}")
        self.model.eval()
        all_tsgen_emb = []
        all_joint_emb = []
        cttp = 0
        sample_num = 0


        all_real = []
        all_samples = []
        all_caps = []

        with torch.no_grad():
            for batch_no, batch in enumerate(self.test_loader):
                start_time = time.time()
                multi_preds = self.model.generate(batch, self.n_samples, sampler)
                multi_preds = multi_preds.permute(0,1,3,2)
                pred = multi_preds.median(dim=0).values

                ts = batch["ts"].to(self.model.device).float()
                # ts_len = batch["ts_len"].to(self.model.device).int()
                # cap_tokens = batch["cap"]

                all_real.append(ts.detach().cpu())
                # all_samples.append(pred.detach().cpu())
                all_samples.append(multi_preds[0].detach().cpu())
                all_caps.extend([str(cap) for cap in batch["cap"]])

                end_time = time.time()
                if (batch_no+1)%self.display_epoch_interval == 0:
                    print("Batch", batch_no, 
                        "Batch Time {:.2f}s".format(end_time-start_time))


        print("Done!")
        res_dict = {
            "tensorboard":{},
            "df":{},
        }

        all_real = torch.cat(all_real)
        all_samples = torch.cat(all_samples)
        print(f"all_samples max, min: {all_samples.max(), all_samples.min()}")
        print(f"all real max, min: {all_real.max(), all_real.min()}")

        print("real mean/std:", all_real.mean(), all_real.std())
        print("sample mean/std:", all_samples.mean(), all_samples.std())
        self._visualize_verbal_conditional_generation(all_real, all_samples, all_caps)
        breakpoint()
        
        flat_real = all_real.flatten()
        flat_sample = all_samples.flatten()
        print("real quantiles:", torch.quantile(flat_real, torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999])))
        print("sample quantiles:", torch.quantile(flat_sample, torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999])))

        metrics = evaluate_model_uncond(
            real_sig=all_real,
            gen_sig=all_samples,
            dataset="istanbul_traffic",
            device="cuda" if torch.cuda.is_available() else "cpu",
            # eval_metrics=['disc', 'vaeFID'],
            eval_metrics=['vaeFID'],
            metric_iteration=10,
            base_path=None,
            vae_ckpt_root="/playpen-shared/haochenz/ImagenFew/fid_vae_ckpts"
        )

        for k, v in metrics.items():
            print(f"{k}: {v}")

        return metrics
