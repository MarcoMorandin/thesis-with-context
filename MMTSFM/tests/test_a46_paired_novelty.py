"""A46: budget-matched temporal-pair selection for multi-anchor S2E."""

from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from mmtsfm.models.vision.patch_projector import evs_select


CONFIG_DIR = str(Path(__file__).parents[1] / "configs")


def test_a46_composes_as_budget_matched_s2e_training_arm():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=vision_chronos2_s2e",
                "model.vision_cfg.n_visual_context_steps=5",
                "+ablation=A46",
            ],
        )

    assert cfg.seed == 42
    assert cfg.model.vision_cfg.fusion_mode == "interleaved_raw"
    assert cfg.model.vision_cfg.n_visual_context_steps == 5
    assert cfg.model.vision_cfg.visual_evs_keep == 100
    assert cfg.model.vision_cfg.visual_evs_mode == "paired_novelty"
    assert cfg.data.video_frames == 20
    assert cfg.data.visual_anchor_stride_hours == 24.0
    assert cfg.train is True
    assert cfg.test is True


def test_paired_novelty_keeps_both_endpoints_of_each_selected_trajectory():
    """Catch a selector that spends all 20 slots on an anchor's first 49 cells."""
    torch.manual_seed(46)
    tokens = torch.randn(2, 10, 49, 8, requires_grad=True)

    kept, frame_idx, cell_idx = evs_select(
        tokens,
        keep=100,
        mode="paired_novelty",
        n_groups=5,
    )

    assert kept.shape == (2, 100, 8)
    for anchor in range(5):
        block = slice(anchor * 20, (anchor + 1) * 20)
        assert torch.equal(
            torch.bincount(frame_idx[0, block], minlength=10),
            torch.tensor(
                [10 if i in (2 * anchor, 2 * anchor + 1) else 0 for i in range(10)]
            ),
        )
        assert torch.equal(
            cell_idx[0, block][:10],
            cell_idx[0, block][10:],
        )

    kept.sum().backward()
    assert tokens.grad is not None
    assert (tokens.grad[:, 1::2].abs().sum(dim=(0, 2, 3)) > 0).all()


def test_paired_novelty_rejects_non_pair_groups():
    tokens = torch.randn(1, 12, 4, 3)

    try:
        evs_select(tokens, keep=12, mode="paired_novelty", n_groups=4)
    except ValueError as error:
        assert "two latent frames per anchor" in str(error)
    else:
        raise AssertionError(
            "three-frame anchors must not silently change A46 semantics"
        )


def test_paired_novelty_rejects_odd_per_anchor_budget():
    tokens = torch.randn(1, 10, 4, 3)

    try:
        evs_select(tokens, keep=25, mode="paired_novelty", n_groups=5)
    except ValueError as error:
        assert "even token budget" in str(error)
    else:
        raise AssertionError("paired trajectories require two tokens per selected cell")
