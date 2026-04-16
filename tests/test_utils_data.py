import numpy as np
import pandas as pd
import torch

from utils.utils_data import (
    fit_aireadi_signal_scaler,
    fit_aireadi_glucose_scaler,
    load_aireadi_windows,
    load_aireadi_glucose_windows,
    normalize_aireadi_signal_values,
    normalize_aireadi_glucose_values,
    resolve_aireadi_signal_path,
    resolve_aireadi_glucose_path,
)


def _write_glucose_split(base_dir, split, glucose_rows, patient_prefix="patient"):
    rows = []
    filename = f"glucose_{'valid' if split in {'val', 'valid'} else split}.parquet"
    for idx, values in enumerate(glucose_rows):
        n = len(values)
        rows.append({
            "glucose": np.array(values, dtype=np.float32),
            "unit": np.array(["mg/dL"] * n, dtype=object),
            "event_type": np.array(["EGV"] * n, dtype=object),
            "source_device_id": np.array([f"device-{idx}"] * n, dtype=object),
            "transmitter_id": np.array([f"tx-{idx}"] * n, dtype=object),
            "transmitter_time": np.arange(n, dtype=np.int64),
            "patient_id": np.array([f"{patient_prefix}-{idx}"] * n, dtype=object),
            "time_utc": np.array([f"2024-01-01T00:{m:02d}:00" for m in range(n)], dtype=object),
            "time_local": np.array([f"2024-01-01T00:{m:02d}:00" for m in range(n)], dtype=object),
        })
    pd.DataFrame(rows).to_parquet(base_dir / filename)


def _write_calorie_split(base_dir, split, calorie_rows, patient_prefix="patient"):
    rows = []
    filename = f"calorie_{'valid' if split in {'val', 'valid'} else split}.parquet"
    for idx, values in enumerate(calorie_rows):
        n = len(values)
        rows.append({
            "calorie": np.array(values, dtype=np.float32),
            "unit": np.array(["kcal"] * n, dtype=object),
            "event_type": np.array(["calorie"] * n, dtype=object),
            "source_device_id": np.array([f"device-{idx}"] * n, dtype=object),
            "patient_id": np.array([f"{patient_prefix}-{idx}"] * n, dtype=object),
            "is_missing": False,
            "time_utc": np.array([f"2024-01-01T00:{m:02d}:00" for m in range(n)], dtype=object),
            "time_local": np.array([f"2024-01-01T00:{m:02d}:00" for m in range(n)], dtype=object),
        })
    pd.DataFrame(rows).to_parquet(base_dir / filename)


def test_load_aireadi_glucose_windows_scaling_and_metadata(tmp_path):
    data_dir = tmp_path / "AI-READI-processed"
    data_dir.mkdir()
    _write_glucose_split(data_dir, "train", [[1.0, 2.0, 3.0, 4.0], [10.0, np.nan, 12.0, 13.0]])
    _write_glucose_split(data_dir, "test", [[4.0, 5.0, 6.0, 7.0]])

    samples, sample_index, scaler = load_aireadi_glucose_windows(
        root_path=str(tmp_path),
        rel_path="AI-READI-processed",
        split="test",
        seq_len=3,
        scale=True,
        stride=1,
        drop_nan=True,
    )

    assert len(samples) == 2
    assert all(isinstance(sample, torch.Tensor) for sample in samples)
    assert samples[0].shape == (3, 1)
    assert scaler is not None
    assert sample_index[0]["patient_id"] == "patient-0"
    assert sample_index[0]["start"] == 0
    assert sample_index[0]["end"] == 3

    expected = scaler.transform(np.array([[4.0], [5.0], [6.0]], dtype=np.float32)).reshape(-1)
    np.testing.assert_allclose(samples[0].squeeze(-1).numpy(), expected.astype(np.float32))


def test_normalize_and_min_seq_len_filtering(tmp_path):
    data_dir = tmp_path / "AI-READI-processed"
    data_dir.mkdir()
    _write_glucose_split(data_dir, "train", [[1.0, np.nan, 3.0, 5.0], [2.0, 4.0, 6.0]])
    _write_glucose_split(data_dir, "valid", [[9.0, np.nan, 11.0], [7.0, 8.0, 9.0, 10.0]])

    path = resolve_aireadi_glucose_path(str(tmp_path), split="valid")
    assert path.endswith("glucose_valid.parquet")

    scaler = fit_aireadi_glucose_scaler(str(tmp_path))
    normalized = normalize_aireadi_glucose_values([1.0, np.nan, 3.0], scaler=scaler, drop_nan=True)
    assert normalized.shape == (2,)

    samples, sample_index, _ = load_aireadi_glucose_windows(
        root_path=str(tmp_path),
        rel_path="AI-READI-processed",
        split="valid",
        seq_len=3,
        scale=False,
        stride=1,
        min_seq_len=4,
        drop_nan=True,
    )

    assert len(samples) == 2
    np.testing.assert_allclose(samples[0].squeeze(-1).numpy(), np.array([7.0, 8.0, 9.0], dtype=np.float32))
    assert sample_index[0]["row_idx"] == 1


def test_generic_signal_utils_support_calorie(tmp_path):
    data_dir = tmp_path / "AI-READI-processed"
    data_dir.mkdir()
    _write_calorie_split(data_dir, "train", [[1.0, 2.0, 3.0], [10.0, np.nan, 12.0, 14.0]])
    _write_calorie_split(data_dir, "test", [[4.0, 5.0, 6.0, 7.0]])

    path = resolve_aireadi_signal_path(str(tmp_path), signal="calorie", split="test")
    assert path.endswith("calorie_test.parquet")

    scaler = fit_aireadi_signal_scaler(str(tmp_path), signal="calorie")
    normalized = normalize_aireadi_signal_values([1.0, np.nan, 3.0], scaler=scaler, drop_nan=True)
    assert normalized.shape == (2,)

    samples, sample_index, _ = load_aireadi_windows(
        root_path=str(tmp_path),
        signal="calorie",
        rel_path="AI-READI-processed",
        split="test",
        seq_len=3,
        scale=True,
        stride=1,
        drop_nan=True,
    )

    assert len(samples) == 2
    assert samples[0].shape == (3, 1)
    assert sample_index[0]["patient_id"] == "patient-0"
