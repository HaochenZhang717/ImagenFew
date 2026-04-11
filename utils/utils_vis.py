import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path

from scipy.spatial import distance


def prepare_data(ori_sig, gen_sig):
    # Analysis sample size (for faster computation)
    sample_num = min([1000, len(ori_sig)])
    idx = np.random.permutation(len(ori_sig))[:sample_num]

    # Data preprocessing
    # ori_ssig = np.asarray(ori_sig)
    # generated_data = np.asarray(gen_sig)

    ori_sig = ori_sig[idx]
    gen_sig = gen_sig[idx]
    no, seq_len, dim = ori_sig.shape
    prep_ori = np.reshape(np.mean(ori_sig[0, :, :], 1), [1, seq_len])
    prep_gen = np.reshape(np.mean(gen_sig[0, :, :], 1), [1, seq_len])
    for i in range(1, sample_num):
        prep_ori = np.concatenate((prep_ori,
                                    np.reshape(np.mean(ori_sig[i, :, :], 1), [1, seq_len])))
        prep_gen = np.concatenate((prep_gen,
                                        np.reshape(np.mean(gen_sig[i, :, :], 1), [1, seq_len])))
    return prep_ori, prep_gen, sample_num


def PCA_plot(prep_ori, prep_gen, anal_sample_no, logger, args):
    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(prep_ori)
    pca_results = pca.transform(prep_ori)
    pca_hat_results = pca.transform(prep_gen)

    # Plotting
    # plt.ion()
    f, ax = plt.subplots(1)
    plt.scatter(pca_results[:, 0], pca_results[:, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    logger.log_fig(f'{args.dataset}_PCA', f)
    plt.close()


def TSNE_plot(prep_ori, prep_gen, anal_sample_no, logger, args):
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    prep_data_final = np.concatenate((prep_ori, prep_gen), axis=0)

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()
    logger.log_fig(f'{args.dataset}_TSNE', f)

    plt.close()


def density_plot(prep_ori, prep_gen, logger, args):
    f, ax = plt.subplots(1)

    sns.distplot(prep_ori, hist=False, kde=True, label='Original')
    sns.distplot(prep_gen, hist=False, kde=True, kde_kws={'linestyle': '--'}, label='Model')
    # Plot formatting
    plt.legend()
    plt.xlabel('Data Value')
    plt.ylabel('Data Density Estimate')
    plt.rcParams['pdf.fonttype'] = 42
    plt.title(args.dataset)
    plt.show()
    logger.log_fig(f'{args.dataset}_density', f)
    plt.close()


def jensen_shannon_divergence(prep_ori, prep_gen,logger):
    """
    method to compute the Jenson-Shannon Divergence of two probability distributions
    """
    p_ = sns.histplot(prep_ori.flatten(), label='Original', stat='probability', bins=200).patches
    plt.close()
    q_ = sns.histplot(prep_gen.flatten(), label='Model', stat='probability', bins=200).patches
    plt.close()
    p = np.array([h.get_height() for h in p_])
    q = np.array([h.get_height() for h in q_])
    if p.shape[0] < q.shape[0]:
        q = q[:p.shape[0]]
    else:
        p = p[:q.shape[0]]
    logger.log('JSD', distance.jensenshannon(p, q))


def save_sample_channel_plots(
    data,
    save_dir,
    max_samples=None,
    dpi=100,
    figsize=(2, 2),
    prefix="sample",
):
    """
    Save one figure per sample/channel pair.

    Expected sample shape is (seq_len, channels) or (channels, seq_len).
    Supported inputs:
        - torch.Tensor / np.ndarray with shape (N, L, C) or (N, C, L)
        - dataset object where dataset[idx] returns a sample tensor/array or (sample, label)

    Files are saved as:
        save_dir/
            sample_00000/
                channel_000.png
                channel_001.png
                ...
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _to_numpy(sample):
        if isinstance(sample, (tuple, list)):
            sample = sample[0]
        if torch is not None and isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()
        else:
            sample = np.asarray(sample)
        if sample.ndim != 2:
            raise ValueError(f"Each sample must be 2D, but got shape {sample.shape}.")
        return sample

    def _normalize_shape(sample):
        # We expect (seq_len, channels); if channels-first is more likely, transpose it.
        if sample.shape[0] < sample.shape[1]:
            sample = sample
        else:
            sample = sample.T
        return sample

    if torch is not None and isinstance(data, torch.Tensor):
        total_samples = data.shape[0]
        get_sample = lambda idx: data[idx]
    elif isinstance(data, np.ndarray):
        total_samples = data.shape[0]
        get_sample = lambda idx: data[idx]
    else:
        total_samples = len(data)
        get_sample = lambda idx: data[idx]

    if max_samples is not None:
        total_samples = min(total_samples, int(max_samples))

    for sample_idx in range(total_samples):
        sample = _normalize_shape(_to_numpy(get_sample(sample_idx)))
        sample_dir = save_dir / f"{prefix}_{sample_idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        seq_len, channels = sample.shape
        x_axis = np.arange(seq_len)

        for channel_idx in range(channels):
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(x_axis, sample[:, channel_idx], linewidth=1.2)
            # ax.set_title(f"{prefix} {sample_idx} | channel {channel_idx}")
            ax.set_xlabel("time")
            ax.set_ylabel("value")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(sample_dir / f"channel_{channel_idx:03d}.png", dpi=dpi)
            plt.close(fig)


def sample_to_numpy_2d(sample):
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if isinstance(sample, (tuple, list)):
        sample = sample[0]
    if torch is not None and isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
    else:
        sample = np.asarray(sample)
    if sample.ndim != 2:
        raise ValueError(f"Each sample must be 2D, but got shape {sample.shape}.")
    if sample.shape[0] >= sample.shape[1]:
        return sample
    return sample.T


def save_one_sample_channel_plots(
    sample,
    sample_idx,
    save_dir,
    dpi=150,
    figsize=(8, 3),
    prefix="sample",
):
    sample = sample_to_numpy_2d(sample)
    save_dir = Path(save_dir)
    sample_dir = save_dir / f"{prefix}_{sample_idx:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    seq_len, channels = sample.shape
    x_axis = np.arange(seq_len)

    for channel_idx in range(channels):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_axis, sample[:, channel_idx], linewidth=1.2)
        ax.set_title(f"{prefix} {sample_idx} | channel {channel_idx}")
        ax.set_xlabel("time")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(sample_dir / f"channel_{channel_idx:03d}.png", dpi=dpi)
        plt.close(fig)


