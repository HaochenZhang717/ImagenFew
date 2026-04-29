# import os
# import json
# import numpy as np
# import random
# from torch.utils.data import Dataset
# import time
#
# class CustomDataset:
#     def __init__(self, folder, **kwargs):
#         super().__init__()
#         self.folder = folder
#         self._load_meta()
#
#     def _load_meta(self):
#         self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
#         self.attr_list = self.meta["attr_list"]
#         n_attr = len(self.attr_list)
#         self.attr_ids = np.arange(n_attr)
#         self.attr_n_ops = np.array(self.meta["attr_n_ops"])
#
#     def get_split(self, split, *args):
#         return CustomSplit(self.folder, split)
#
#
# class CustomSplit(Dataset):
#     def __init__(self, folder, split="train"):
#         super().__init__()
#         assert split in ("train", "valid", "test"), "Please specify a valid split."
#         self.split = split
#         self.folder = folder
#         self._load_data()
#
#         print(f"Split: {self.split}, total samples {self.n_samples}.")
#
#     def _load_data(self):
#         ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
#         attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]
#         caps = np.load(os.path.join(self.folder, self.split+fr"_my_text_caps.npy"), allow_pickle=True)
#         self.ts, self.attrs, self.caps = ts, attrs, caps
#         self.n_samples = self.ts.shape[0]
#         self.n_steps = self.ts.shape[1]
#         self.n_attrs = self.attrs.shape[1]
#         self.time_point = np.arange(self.n_steps)
#
#     def __getitem__(self, idx):
#         cap_id = random.randint(0, len(self.caps[idx])-1)
#         tmp_ts = self.ts[idx]
#         if len(tmp_ts.shape) == 1:
#             tmp_ts = tmp_ts[...,np.newaxis]
#         return {"ts": tmp_ts,
#                 "ts_len": tmp_ts.shape[0],
#                 "attrs": self.attrs[idx],
#                 "cap": self.caps[idx][cap_id],
#                 "tp": self.time_point}
#
#     def __len__(self):
#         return self.n_samples


import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CustomDataset:
    def __init__(self, folder, **kwargs):
        super().__init__()
        self.folder = folder
        self.scale = True
        self.scaler_path = None
        self.scaler = None
        self._load_meta()
        self._init_scaler()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    # =========================
    # ⭐ 核心：初始化 scaler（不依赖 split 顺序）
    # =========================
    def _init_scaler(self):

        # 1️⃣ 如果有保存的 scaler → 直接加载
        if self.scaler_path is not None and os.path.exists(self.scaler_path):
            stats = np.load(self.scaler_path)
            scaler = StandardScaler()
            scaler.mean_ = stats["mean"]
            scaler.scale_ = stats["scale"]
            scaler.var_ = stats["var"]
            self.scaler = scaler
            print(f"[Scaler] Loaded from {self.scaler_path}")
            return

        # 2️⃣ 否则自动用 train 数据 fit
        train_path = os.path.join(self.folder, "train_ts.npy")
        assert os.path.exists(train_path), "train_ts.npy not found!"

        ts = np.load(train_path)

        if ts.ndim == 2:
            ts = ts[:, :, None]

        ts = ts.astype(np.float32)

        scaler = StandardScaler()
        scaler.fit(ts.reshape(-1, ts.shape[-1]))

        self.scaler = scaler

        print("[Scaler] Fitted from train_ts.npy")

        # 可选保存
        if self.scaler_path is not None:
            np.savez(
                self.scaler_path,
                mean=scaler.mean_.astype(np.float32),
                scale=scaler.scale_.astype(np.float32),
                var=scaler.var_.astype(np.float32),
            )
            print(f"[Scaler] Saved to {self.scaler_path}")

    def transform(self, ts):
        if not self.scale or self.scaler is None:
            return ts

        original_shape = ts.shape
        ts = self.scaler.transform(
            ts.reshape(-1, original_shape[-1])
        ).reshape(original_shape)

        return ts.astype(np.float32)

    def inverse_transform(self, ts):
        if not self.scale or self.scaler is None:
            return ts

        original_shape = ts.shape
        ts = ts.reshape(-1, original_shape[-1])
        ts = ts * self.scaler.scale_ + self.scaler.mean_
        return ts.reshape(original_shape)

    def get_split(self, split, *args):
        return CustomSplit(self, self.folder, split)


class CustomSplit(Dataset):
    def __init__(self, dataset, folder, split="train"):
        super().__init__()
        assert split in ("train", "valid", "test")

        self.dataset = dataset
        self.split = split
        self.folder = folder

        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split + "_ts.npy"))
        attrs = np.load(os.path.join(self.folder, self.split + "_attrs_idx.npy"))
        caps = np.load(
            os.path.join(self.folder, self.split + "_my_text_caps.npy"),
            allow_pickle=True
        )

        # [N, T] → [N, T, C]
        if ts.ndim == 2:
            ts = ts[:, :, None]

        ts = ts.astype(np.float32)

        # ⭐ 直接用 dataset 的 scaler（无顺序依赖）
        ts = self.dataset.transform(ts)

        self.ts = ts
        self.attrs = attrs
        self.caps = caps

        self.n_samples = ts.shape[0]
        self.n_steps = ts.shape[1]
        self.n_attrs = attrs.shape[1]

        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        cap_id = random.randint(0, len(self.caps[idx]) - 1)

        tmp_ts = self.ts[idx]

        return {
            "ts": tmp_ts,
            "ts_len": tmp_ts.shape[0],
            "attrs": self.attrs[idx],
            "cap": self.caps[idx][cap_id],
            "tp": self.time_point
        }

    def __len__(self):
        return self.n_samples


class ParaphrasedCaptionDataset(CustomDataset):
    def get_split(self, split, *args):
        return ParaphrasedCaptionSplit(self, self.folder, split)


class ParaphrasedCaptionSplit(CustomSplit):
    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split + "_ts.npy"))
        attrs = np.load(os.path.join(self.folder, self.split + "_attrs_idx.npy"))

        caps = np.load(
            os.path.join(self.folder, self.split + "_my_text_caps_paraphrased.npy"),
            allow_pickle=True
        )

        if ts.ndim == 2:
            ts = ts[:, :, None]

        ts = ts.astype(np.float32)
        ts = self.dataset.transform(ts)

        self.ts = ts
        self.attrs = attrs
        self.caps = caps

        self.n_samples = ts.shape[0]
        self.n_steps = ts.shape[1]
        self.n_attrs = attrs.shape[1]
        breakpoint()
        self.time_point = np.arange(self.n_steps)

    def _select_caption(self, cap_item):
        cap_item = np.asarray(cap_item)
        n_variants, n_segments = cap_item.shape
        segments = [
            cap_item[random.randint(0, n_variants - 1), segment_id]
            for segment_id in range(n_segments)
        ]
        return "\n".join(
            f"[Segment {idx + 1}]: {str(segment).strip()}"
            for idx, segment in enumerate(segments)
        )

    def __getitem__(self, idx):
        tmp_ts = self.ts[idx]

        return {
            "ts": tmp_ts,
            "ts_len": tmp_ts.shape[0],
            "attrs": self.attrs[idx],
            "cap": self._select_caption(self.caps[idx]),
            "tp": self.time_point
        }
