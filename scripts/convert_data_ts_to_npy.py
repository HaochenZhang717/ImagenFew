#!/usr/bin/env python3
"""Convert .ts datasets under data/ into train_ts.npy and test_ts.npy.

Usage:
  python scripts/convert_data_ts_to_npy.py
  python scripts/convert_data_ts_to_npy.py --data-dir data/UEA --out-dir npy_data
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe


def _series_to_float_array(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(arr).any():
        s = pd.Series(arr)
        s = s.interpolate(limit_direction="both")
        s = s.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
        arr = s.to_numpy(dtype=np.float32)
    return arr


def _nested_df_to_3d(df: pd.DataFrame) -> np.ndarray:
    n_samples, n_dims = df.shape

    # Find maximum sequence length across all dimensions/samples.
    max_len = 0
    for i in range(n_samples):
        for j in range(n_dims):
            max_len = max(max_len, len(df.iat[i, j]))

    data = np.zeros((n_samples, max_len, n_dims), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_dims):
            arr = _series_to_float_array(df.iat[i, j])
            end = min(len(arr), max_len)
            data[i, :end, j] = arr[:end]
    return data


def convert_one(train_ts_path: Path, test_ts_path: Path, out_dir: Path) -> None:
    train_df, _ = load_from_tsfile_to_dataframe(
        str(train_ts_path), return_separate_X_and_y=True, replace_missing_vals_with="NaN"
    )
    test_df, _ = load_from_tsfile_to_dataframe(
        str(test_ts_path), return_separate_X_and_y=True, replace_missing_vals_with="NaN"
    )

    train_ts = _nested_df_to_3d(train_df)
    test_ts = _nested_df_to_3d(test_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train_ts.npy.shape = {train_ts.npy.shape}")
    print(f"test_ts.npy.shape = {test_ts.npy.shape}")
    np.save(out_dir / "train_ts.npy", train_ts)
    np.save(out_dir / "test_ts.npy", test_ts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("npy_data"))
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir = args.out_dir.resolve()

    train_files = sorted(data_dir.rglob("*_TRAIN.ts"))
    converted = 0

    for train_file in train_files:
        stem = train_file.name[:-len("_TRAIN.ts")]
        test_file = train_file.with_name(f"{stem}_TEST.ts")
        if not test_file.exists():
            continue

        dataset_name = train_file.parent.name
        target_dir = out_dir / dataset_name
        convert_one(train_file, test_file, target_dir)
        print(f"[OK] {dataset_name} -> {target_dir}")
        converted += 1

    if converted == 0:
        raise SystemExit(f"No dataset pairs '*_TRAIN.ts'/'*_TEST.ts' found under {data_dir}")

    print(f"Done. Converted {converted} datasets to: {out_dir}")


if __name__ == "__main__":
    main()
