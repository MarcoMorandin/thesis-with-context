"""The Aurora fine-tune path: target masking and the vision-group split."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

AURORA = Path(__file__).parents[1] / "tier5" / "vendor" / "aurora"


def _ft_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "aurora_ukpv_ft_dataset", AURORA / "ukpv_ft_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flow_loss_module():
    """Import `aurora.flow_loss` without the package's heavy transformers imports."""
    package = types.ModuleType("aurora")
    package.__path__ = [str(AURORA / "aurora")]
    sys.modules.setdefault("aurora", package)
    spec = importlib.util.spec_from_file_location(
        "aurora.flow_loss", AURORA / "aurora" / "flow_loss.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["aurora.flow_loss"] = module
    spec.loader.exec_module(module)
    return module


def _window_batch(n=4, history=6, horizon=3, side=2):
    """A minimal batch shaped like `UKMultimodalDataset.batch`."""
    return {
        "y_hist": np.random.rand(n, history).astype(np.float32),
        "y_future": np.random.rand(n, horizon).astype(np.float32),
        "mask_future": np.ones((n, horizon), dtype=np.float32),
        "V": np.zeros((n, 2, 1, side, side), dtype=np.float32),
        "mask_visual": np.zeros((n, 2), dtype=np.float32),
    }


def _latest_real_frames(vision, mask):
    """Stand-in for the runner helper; same contract, no torch/hydra import."""
    available = mask.astype(bool).any(axis=1)
    latest = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    frames = vision[np.arange(len(vision)), latest]
    if frames.shape[1] == 1:
        frames = np.repeat(frames, 3, axis=1)
    frames = np.rint(np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    return torch.from_numpy(frames), torch.from_numpy(available)


def test_unscored_windows_are_dropped_before_the_forward():
    """A window with no observed horizon step would make an empty forecast term."""
    module = _ft_dataset_module()
    batch = _window_batch(n=4)
    batch["mask_future"][1] = 0.0  # nothing observed in this window's horizon
    batch["mask_visual"][:] = 0.0  # all pseudo, so one group

    groups = list(
        module.aurora_subbatches(
            batch,
            "cpu",
            vision_mode="real",
            latest_real_frames=_latest_real_frames,
        )
    )

    assert len(groups) == 1
    kwargs, count = groups[0]
    assert count == 3
    assert kwargs["input_ids"].shape[0] == 3
    assert kwargs["vision_ids"] is None


def test_real_and_pseudo_windows_go_through_separate_forwards():
    """`forward` takes one `vision_ids` per call, so the batch must be split."""
    module = _ft_dataset_module()
    batch = _window_batch(n=4)
    batch["mask_visual"][0, 1] = 1.0
    batch["mask_visual"][2, 0] = 1.0

    groups = dict(
        (kwargs["vision_ids"] is not None, count)
        for kwargs, count in module.aurora_subbatches(
            batch,
            "cpu",
            vision_mode="real",
            latest_real_frames=_latest_real_frames,
        )
    )

    assert groups == {True: 2, False: 2}


def test_pseudo_mode_never_passes_frames_even_when_they_exist():
    module = _ft_dataset_module()
    batch = _window_batch(n=3)
    batch["mask_visual"][:] = 1.0

    groups = list(
        module.aurora_subbatches(
            batch,
            "cpu",
            vision_mode="pseudo",
            latest_real_frames=_latest_real_frames,
        )
    )

    assert len(groups) == 1
    kwargs, count = groups[0]
    assert count == 3
    assert kwargs["vision_ids"] is None


def test_loss_masks_are_carried_through_as_mask_future():
    module = _ft_dataset_module()
    batch = _window_batch(n=2, horizon=3)
    batch["mask_future"][0] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    batch["mask_visual"][:] = 0.0

    ((kwargs, _),) = module.aurora_subbatches(
        batch, "cpu", vision_mode="real", latest_real_frames=_latest_real_frames
    )

    assert torch.equal(kwargs["loss_masks"][0], torch.tensor([1.0, 0.0, 1.0]))


def test_flow_loss_survives_a_fully_masked_row():
    """A window whose steps are all masked would otherwise divide by zero."""
    flow = _flow_loss_module()
    loss_fn = flow.FlowLoss(
        target_channels=4, z_channels=4, depth=1, width=8, num_sampling_steps=2
    )
    target = torch.randn(3, 4)
    z = torch.randn(3, 4)
    mask = torch.ones(3, 4)
    mask[1] = 0.0

    loss = loss_fn(target=target, z=z, mask=mask)

    assert torch.isfinite(loss)


def _batchnorm_decoder_norm(channels=8):
    """Shaped like Aurora's `norm_mode: 'batch'` decoder norm."""

    class Transpose(torch.nn.Module):
        def forward(self, x):
            return x.transpose(1, 2)

    return torch.nn.Sequential(Transpose(), torch.nn.BatchNorm1d(channels), Transpose())


def test_finetune_mode_leaves_batchnorm_on_running_statistics():
    """The vision-group split yields single-window forwards; batch stats can't."""
    module = _ft_dataset_module()
    model = torch.nn.Sequential(_batchnorm_decoder_norm(8), torch.nn.Linear(8, 8))

    module.set_finetune_mode(model)

    assert model.training  # the model at large still trains
    bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert bn and not any(m.training for m in bn)
    # one window, one token -- the shape that crashed job 57823466
    assert model(torch.randn(1, 1, 8)).shape == (1, 1, 8)


def test_finetune_mode_keeps_batchnorm_affine_trainable():
    """Frozen statistics still leave weight/bias to absorb the shift."""
    module = _ft_dataset_module()
    model = _batchnorm_decoder_norm(8)

    module.set_finetune_mode(model)

    assert all(p.requires_grad for p in module.trainable_parameters(model))
    assert len(module.trainable_parameters(model)) == 2  # weight, bias
