import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def AIREADIGlucose(**config):
    return AIREADIGlucoseDataset(
        root_path=config["datasets_dir"],
        rel_path=config.get("rel_path", "AI-READI-processed"),
        flag=config["flag"],
        seq_len=config["seq_len"],
        scale=config.get("scale", True),
        stride=config.get("window_stride", 1),
        min_seq_len=config.get("min_seq_len", None),
        drop_nan=config.get("drop_nan", True),
        return_metadata=config.get("return_metadata", False),
    )


class AIREADIGlucoseDataset(Dataset):
    SPLIT_TO_FILE = {
        "train": "glucose_train.parquet",
        "val": "glucose_valid.parquet",
        "valid": "glucose_valid.parquet",
        "test": "glucose_test.parquet",
    }

    def __init__(
        self,
        root_path,
        rel_path="AI-READI-processed",
        flag="train",
        seq_len=24,
        scale=True,
        stride=1,
        min_seq_len=None,
        drop_nan=True,
        return_metadata=False,
    ):
        super().__init__()
        if flag not in self.SPLIT_TO_FILE:
            raise ValueError(f"Unsupported flag {flag}. Expected one of {sorted(self.SPLIT_TO_FILE)}")

        self.root_path = root_path
        self.rel_path = rel_path
        self.flag = flag
        self.seq_len = int(seq_len)
        self.scale = scale
        self.stride = int(stride)
        self.min_seq_len = int(min_seq_len) if min_seq_len is not None else self.seq_len
        self.drop_nan = drop_nan
        self.return_metadata = return_metadata

        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.min_seq_len < self.seq_len:
            self.min_seq_len = self.seq_len

        self.scaler = StandardScaler()
        self.samples = []
        self.sample_index = []

        self._build_dataset()

    def _resolve_path(self, filename):
        return os.path.join(self.root_path, self.rel_path, filename)

    def _load_frame(self, filename):
        path = self._resolve_path(filename)
        return pd.read_parquet(path)

    def _fit_scaler(self):
        if not self.scale:
            return

        train_df = self._load_frame(self.SPLIT_TO_FILE["train"])
        all_values = []
        for values in train_df["glucose"]:
            arr = np.asarray(values, dtype=np.float32)
            if self.drop_nan:
                arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                all_values.append(arr)

        if not all_values:
            raise ValueError("No valid glucose values found in AI-READI train split.")

        train_values = np.concatenate(all_values, axis=0).reshape(-1, 1)
        self.scaler.fit(train_values)

    def _normalize(self, values):
        if not self.scale:
            return values
        return self.scaler.transform(values.reshape(-1, 1)).reshape(-1).astype(np.float32)

    def _build_dataset(self):
        self._fit_scaler()

        df = self._load_frame(self.SPLIT_TO_FILE[self.flag])
        for row_idx, row in enumerate(df.itertuples(index=False)):
            values = np.asarray(row.glucose, dtype=np.float32)
            if self.drop_nan:
                values = values[np.isfinite(values)]

            if values.size < self.min_seq_len:
                continue

            values = self._normalize(values)
            patient_id = getattr(row, "patient_id", None)

            max_start = values.shape[0] - self.seq_len
            for start in range(0, max_start + 1, self.stride):
                end = start + self.seq_len
                window = torch.from_numpy(values[start:end]).unsqueeze(-1)
                self.samples.append(window)
                self.sample_index.append(
                    {
                        "row_idx": row_idx,
                        "patient_id": patient_id[start] if isinstance(patient_id, np.ndarray) else patient_id,
                        "start": start,
                        "end": end,
                    }
                )

        if not self.samples:
            raise ValueError(
                f"No windows were created for split={self.flag}. "
                f"Check seq_len={self.seq_len} and stride={self.stride}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.return_metadata:
            return sample, self.sample_index[index]
        return sample

