import json
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "data_provider" / "datasets" / "verbal_ts.py"
SPEC = importlib.util.spec_from_file_location("verbal_ts", MODULE_PATH)
verbal_ts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verbal_ts)
Dataset_VerbalTS = verbal_ts.Dataset_VerbalTS


def _write_split(base_dir, split, ts_shape):
    split_name = "valid" if split in {"val", "valid"} else split
    ts = np.arange(np.prod(ts_shape), dtype=np.float32).reshape(ts_shape)
    attrs = np.zeros((ts_shape[0], 2), dtype=np.int64)
    caps = np.full((ts_shape[0], 1), "caption", dtype=object)
    np.save(base_dir / f"{split_name}_ts.npy", ts)
    np.save(base_dir / f"{split_name}_attrs_idx.npy", attrs)
    np.save(base_dir / f"{split_name}_text_caps.npy", caps)


def test_verbal_ts_dataset_loads_train_and_test_splits(tmp_path):
    dataset_dir = tmp_path / "VerbalTSDatasets" / "ToySet"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "meta.json").write_text(json.dumps({"attr_list": ["trend"]}), encoding="utf-8")

    _write_split(dataset_dir, "train", (3, 12, 2))
    _write_split(dataset_dir, "valid", (2, 12, 2))
    _write_split(dataset_dir, "test", (4, 12, 2))

    train_ds = Dataset_VerbalTS(root_path=str(tmp_path), rel_path="VerbalTSDatasets/ToySet", flag="train")
    test_ds = Dataset_VerbalTS(root_path=str(tmp_path), rel_path="VerbalTSDatasets/ToySet", flag="test")
    valid_ds = Dataset_VerbalTS(root_path=str(tmp_path), rel_path="VerbalTSDatasets/ToySet", flag="val")

    assert len(train_ds) == 3
    assert len(test_ds) == 4
    assert len(valid_ds) == 2
    assert train_ds[0].shape == (12, 2)
    assert str(train_ds[0].dtype) == "torch.float32"
    assert train_ds.meta["attr_list"] == ["trend"]
    assert train_ds.attrs_idx.shape == (3, 2)
    assert train_ds.text_caps.shape == (3, 1)


def test_verbal_ts_dataset_scales_with_train_statistics(tmp_path):
    dataset_dir = tmp_path / "VerbalTSDatasets" / "ToySet"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "meta.json").write_text(json.dumps({"attr_list": ["trend"]}), encoding="utf-8")

    train_ts = np.array(
        [
            [[1.0], [3.0]],
            [[5.0], [7.0]],
        ],
        dtype=np.float32,
    )
    test_ts = np.array(
        [
            [[9.0], [11.0]],
        ],
        dtype=np.float32,
    )
    np.save(dataset_dir / "train_ts.npy", train_ts)
    np.save(dataset_dir / "valid_ts.npy", train_ts[:1])
    np.save(dataset_dir / "test_ts.npy", test_ts)
    np.save(dataset_dir / "train_attrs_idx.npy", np.zeros((2, 1), dtype=np.int64))
    np.save(dataset_dir / "valid_attrs_idx.npy", np.zeros((1, 1), dtype=np.int64))
    np.save(dataset_dir / "test_attrs_idx.npy", np.zeros((1, 1), dtype=np.int64))
    np.save(dataset_dir / "train_text_caps.npy", np.full((2, 1), "caption", dtype=object))
    np.save(dataset_dir / "valid_text_caps.npy", np.full((1, 1), "caption", dtype=object))
    np.save(dataset_dir / "test_text_caps.npy", np.full((1, 1), "caption", dtype=object))

    train_ds = Dataset_VerbalTS(root_path=str(tmp_path), rel_path="VerbalTSDatasets/ToySet", flag="train")
    test_ds = Dataset_VerbalTS(root_path=str(tmp_path), rel_path="VerbalTSDatasets/ToySet", flag="test")

    train_values = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    mean = train_values.mean()
    std = train_values.std(ddof=0)
    expected_test = ((test_ts - mean) / std).reshape(2)

    np.testing.assert_allclose(train_ds.inverse_transform(train_ds[0]).reshape(-1), train_ts[0].reshape(-1), rtol=1e-5)
    np.testing.assert_allclose(test_ds[0].numpy().reshape(-1), expected_test, rtol=1e-5)
