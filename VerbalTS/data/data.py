import os
import json
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import time

class CustomDataset:
    def __init__(self, folder, scale=True, **kwargs):
        super().__init__()
        self.folder = folder
        self.scale = scale
        self._load_meta()
        self._fit_scaler()

    def _load_meta(self):
        self.meta = json.load(open(os.path.join(self.folder, "meta.json")))
        self.attr_list = self.meta["attr_list"]
        n_attr = len(self.attr_list)
        self.attr_ids = np.arange(n_attr)
        self.attr_n_ops = np.array(self.meta["attr_n_ops"])

    def _fit_scaler(self):
        self.scaler = None
        if not self.scale:
            return

        train_ts = np.load(os.path.join(self.folder, "train_ts.npy"))
        train_ts_3d, _ = self._ensure_3d(train_ts)

        self.scaler = StandardScaler()
        self.scaler.fit(train_ts_3d.reshape(-1, train_ts_3d.shape[-1]))

    @staticmethod
    def _ensure_3d(ts):
        squeezed = False
        if ts.ndim == 2:
            ts = ts[..., np.newaxis]
            squeezed = True
        return ts, squeezed

    def get_split(self, split, *args):
        return CustomSplit(self.folder, split, scaler=self.scaler)

    def inverse_transform(self, ts):
        if self.scaler is None:
            return ts

        is_2d = ts.ndim == 2
        if is_2d:
            ts = ts[..., np.newaxis]
        original_shape = ts.shape
        ts = self.scaler.inverse_transform(ts.reshape(-1, original_shape[-1])).reshape(original_shape)
        if is_2d:
            ts = ts.squeeze(-1)
        return ts


class CustomSplit(Dataset):
    def __init__(self, folder, split="train", scaler=None):
        super().__init__()
        assert split in ("train", "valid", "test"), "Please specify a valid split."
        self.split = split            
        self.folder = folder
        self.scaler = scaler
        self._load_data()

        print(f"Split: {self.split}, total samples {self.n_samples}.")

    def _load_data(self):
        ts = np.load(os.path.join(self.folder, self.split+"_ts.npy"))     # [n_samples, n_steps]
        attrs = np.load(os.path.join(self.folder, self.split+"_attrs_idx.npy"))  # [n_samples, n_attrs]
        caps = np.load(os.path.join(self.folder, self.split+fr"_text_caps.npy"), allow_pickle=True)
        ts, squeezed = CustomDataset._ensure_3d(ts)
        if self.scaler is not None:
            original_shape = ts.shape
            ts = self.scaler.transform(ts.reshape(-1, original_shape[-1])).reshape(original_shape)
        if squeezed:
            ts = ts.squeeze(-1)
        self.ts, self.attrs, self.caps = ts, attrs, caps
        self.n_samples = self.ts.shape[0]
        self.n_steps = self.ts.shape[1]
        self.n_attrs = self.attrs.shape[1]
        self.time_point = np.arange(self.n_steps)

    def __getitem__(self, idx):
        cap_id = random.randint(0, len(self.caps[idx])-1)
        tmp_ts = self.ts[idx]
        if len(tmp_ts.shape) == 1:
            tmp_ts = tmp_ts[...,np.newaxis]
        return {"ts": tmp_ts,
                "ts_len": tmp_ts.shape[0],
                "attrs": self.attrs[idx],
                "cap": self.caps[idx][cap_id],
                "tp": self.time_point,
                "indices": idx
                }

    def __len__(self):
        return self.n_samples

    def inverse_transform(self, ts):
        if self.scaler is None:
            return ts

        is_2d = ts.ndim == 2
        if is_2d:
            ts = ts[..., np.newaxis]
        original_shape = ts.shape
        ts = self.scaler.inverse_transform(ts.reshape(-1, original_shape[-1])).reshape(original_shape)
        if is_2d:
            ts = ts.squeeze(-1)
        return ts
