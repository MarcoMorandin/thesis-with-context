from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from common import config
from tier6 import uk_multimodal
from tier6.uk_multimodal import UKMultimodalDataset


def _runner_module():
    fake_modeling = types.ModuleType("aurora.modeling_aurora")
    fake_modeling.AuroraForPrediction = object
    fake_package = types.ModuleType("aurora")
    sys.modules.setdefault("aurora", fake_package)
    sys.modules.setdefault("aurora.modeling_aurora", fake_modeling)
    path = Path(__file__).parents[1] / "tier5" / "vendor" / "aurora" / "run_ukpv.py"
    spec = importlib.util.spec_from_file_location("aurora_ukpv_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latest_real_frames_selects_last_valid_frame_and_makes_rgb_uint8():
    runner = _runner_module()
    vision = np.zeros((2, 3, 1, 2, 2), dtype=np.float32)
    vision[0, 0] = 0.2
    vision[0, 2] = 0.8
    vision[1, 1] = 0.4
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)

    frames, available = runner.latest_real_frames(vision, mask)

    assert frames.shape == (2, 3, 2, 2)
    assert frames.dtype == torch.uint8
    assert available.tolist() == [True, True]
    assert torch.all(frames[0] == 204)
    assert torch.all(frames[1] == 102)


def test_forecast_batch_retains_samples_and_uses_real_frames():
    runner = _runner_module()

    class Model:
        def generate(self, *, inputs, vision_inputs=None, **kwargs):
            value = 3.0 if vision_inputs is not None else 1.0
            return torch.full(
                (len(inputs), kwargs["num_samples"], kwargs["max_output_length"]), value
            )

    samples = runner.forecast_batch(
        Model(),
        torch.zeros((2, 8)),
        torch.ones((2, 3, 4, 4), dtype=torch.uint8),
        torch.ones(2, dtype=torch.bool),
        vision_mode="real",
        pred_len=3,
        inference_token_len=4,
        num_samples=5,
    )

    assert samples.shape == (2, 5, 3)
    assert torch.all(samples == 3)


def test_real_mode_falls_back_to_native_pseudo_image_when_frame_is_missing():
    runner = _runner_module()

    class Model:
        def generate(self, *, inputs, vision_inputs=None, **kwargs):
            value = 3.0 if vision_inputs is not None else 1.0
            return torch.full((len(inputs), 2, 2), value)

    samples = runner.forecast_batch(
        Model(),
        torch.zeros((2, 8)),
        torch.ones((2, 3, 4, 4), dtype=torch.uint8),
        torch.tensor([True, False]),
        vision_mode="real",
        pred_len=2,
        inference_token_len=4,
        num_samples=2,
    )

    assert torch.all(samples[0] == 3)
    assert torch.all(samples[1] == 1)


def test_validation_score_is_macro_averaged_by_plant():
    runner = _runner_module()
    pred = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    true = np.zeros_like(pred)
    valid = np.ones_like(pred)
    sites = np.array(["small", "large", "large"])

    score = runner.macro_mae(pred, true, valid, sites)

    assert score == pytest.approx(0.5)


def test_visual_history_steps_loads_only_latest_frame():
    ds = UKMultimodalDataset.__new__(UKMultimodalDataset)
    ds.win = [
        {
            "dataset": "uk_pv",
            "site_id": "site",
            "timestamps": np.array([10, 20, 30, 40]),
        }
    ]
    ds.history = 3
    ds.visual_history_steps = 1
    ds.img_size = 2
    ds.to_gray = True
    ds.channels = 1
    ds.coords = {}
    ds.frame_maps = {("uk_pv", "site"): {10: 0, 20: 1, 30: 2}}
    images = np.stack(
        [
            np.full((2, 2), 10, dtype=np.uint8),
            np.full((2, 2), 20, dtype=np.uint8),
            np.full((2, 2), 30, dtype=np.uint8),
        ]
    )
    ds._h5_group = lambda *_: {"images": images}
    ds._png = False

    item = ds[0]

    assert item["V"].shape == (1, 1, 2, 2)
    assert item["mask_visual"].tolist() == [1.0]
    assert np.allclose(item["V"], 30 / 255.0)


def test_png_encoded_frames_are_decoded():
    """data_v2 stores each frame as a 1-D PNG blob, not a raw (H, W) array."""
    import io

    from PIL import Image

    from tier6.uk_multimodal import _decode_frame

    def as_png_bytes(arr: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return np.frombuffer(buf.getvalue(), dtype=np.uint8)

    raw = np.full((2, 2), 30, dtype=np.uint8)
    blob = as_png_bytes(raw)
    assert blob.ndim == 1  # what h5py hands back for v2

    assert np.array_equal(_decode_frame(blob, png=True), raw)
    assert np.array_equal(_decode_frame(raw, png=False), raw)

    ds = UKMultimodalDataset.__new__(UKMultimodalDataset)
    ds.win = [
        {
            "dataset": "uk_pv",
            "site_id": "site",
            "timestamps": np.array([10, 20]),
        }
    ]
    ds.history = 1
    ds.visual_history_steps = 1
    ds.img_size = 2
    ds.to_gray = True
    ds.channels = 1
    ds.coords = {}
    ds.frame_maps = {("uk_pv", "site"): {10: 0}}
    ds._h5_group = lambda *_: {"images": [blob]}
    ds._png = True

    item = ds[0]

    assert item["V"].shape == (1, 1, 2, 2)
    assert np.allclose(item["V"], 30 / 255.0)


def test_numerical_windows_keep_rows_without_satellite_frames(monkeypatch):
    import pandas as pd

    frame = pd.DataFrame(
        {
            config.DATASET_COL: ["uk_pv"] * 3,
            config.SITE_COL: ["site"] * 3,
            config.TIME_COL: pd.date_range(
                "2025-01-01", periods=3, freq="30min", tz="UTC"
            ),
            config.TARGET_COL: [0.0, 0.1, 0.2],
            config.CAPACITY_COL: [1000.0] * 3,
            config.CLEARSKY_COL: [0.0, 1.0, 1.0],
            config.FRAME_INDEX_COL: [np.nan, 0.0, 1.0],
            "latitude": [50.0] * 3,
            "longitude": [-1.0] * 3,
            **{name: [0.0] * 3 for name in config.COV_COLS},
        }
    )
    captured = {}

    def capture_windows(df, *args, **kwargs):
        captured["rows"] = df.copy()
        return []

    monkeypatch.setattr(uk_multimodal.pd, "read_parquet", lambda *args, **kwargs: frame)
    monkeypatch.setattr(uk_multimodal, "dataset_for_sites", capture_windows)

    UKMultimodalDataset(site_ids=["site"], data_path="unused", h5_path="unused")

    assert len(captured["rows"]) == 3


def test_real_image_processing_returns_pixels_to_input_device():
    try:
        from tier5.vendor.aurora.aurora.real_image import process_real_images
    except ImportError as exc:
        pytest.fail(f"real-image device bridge is missing: {exc}")

    class Processor:
        def __call__(self, *, images, return_tensors):
            assert images.device.type == "cpu"
            assert return_tensors == "pt"
            return {"pixel_values": images.float() / 255.0}

    frames = torch.full((2, 3, 4, 4), 255, dtype=torch.uint8)
    pixels = process_real_images(Processor(), frames)

    assert pixels.device == frames.device
    assert pixels.dtype == torch.float32
    assert torch.all(pixels == 1)
