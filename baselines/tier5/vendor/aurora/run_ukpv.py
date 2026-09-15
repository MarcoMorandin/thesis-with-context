"""Protocol-aligned zero-shot Aurora evaluation on UK-PV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

BASELINES = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASELINES))

from aurora.modeling_aurora import AuroraForPrediction  # noqa: E402

from tier6.uk_multimodal import UKMultimodalDataset, sites_for_split  # noqa: E402


def latest_real_frames(
    vision: np.ndarray, mask: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select each window's latest observed frame and convert it to RGB uint8."""
    available = mask.astype(bool).any(axis=1)
    latest_idx = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    frames = vision[np.arange(len(vision)), latest_idx]
    if frames.shape[1] == 1:
        frames = np.repeat(frames, 3, axis=1)
    frames = np.rint(np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    return torch.from_numpy(frames), torch.from_numpy(available)


def forecast_batch(
    model,
    histories,
    frames,
    available,
    *,
    vision_mode,
    pred_len,
    inference_token_len,
    num_samples,
):
    """Generate samples, using pseudo-images when real frames are missing."""
    kwargs = {
        "max_output_length": pred_len,
        "inference_token_len": inference_token_len,
        "num_samples": num_samples,
    }
    if vision_mode == "pseudo":
        return model.generate(inputs=histories, **kwargs)
    output = torch.empty(
        (len(histories), num_samples, pred_len), device=histories.device
    )
    for use_real in (True, False):
        keep = available == use_real
        if not keep.any():
            continue
        frame_mask = keep.detach().cpu()
        images = frames[frame_mask].to(histories.device) if use_real else None
        output[keep] = model.generate(
            inputs=histories[keep], vision_inputs=images, **kwargs
        )
    return output


def macro_mae(
    pred: np.ndarray, true: np.ndarray, valid: np.ndarray, sites: np.ndarray
) -> float:
    """Return the per-plant macro MAE over valid daylight targets."""
    scores = []
    for site in np.unique(sites):
        take, weight = sites == site, valid[sites == site]
        if weight.sum() == 0:
            continue
        scores.append(
            float((np.abs(pred[take] - true[take]) * weight).sum() / weight.sum())
        )
    if not scores:
        raise ValueError("no valid daylight targets")
    return float(np.mean(scores))


def _run_split(
    model, cfg: DictConfig, split: str, vision_mode: str, out: Path
) -> float:
    device = next(model.parameters()).device
    totals = {"pred": [], "true": [], "valid": [], "site": []}
    out.mkdir(parents=True, exist_ok=True)
    for site in sites_for_split(split, dataset=cfg.dataset):
        ds = UKMultimodalDataset(
            site_ids=[site],
            data_path=cfg.data_path,
            h5_path=cfg.h5_path,
            history=cfg.history,
            horizon=cfg.horizon,
            stride=cfg.stride,
            img_size=cfg.image_size,
            datasets=[cfg.dataset],
            to_gray=True,
            visual_history_steps=cfg.visual_history_steps,
        )
        sample_parts, true_parts, valid_parts = [], [], []
        history_parts, history_valid_parts = [], []
        target_valid_parts, daylight_parts = [], []
        for batch in ds.iter_batches(cfg.batch_size):
            histories = torch.from_numpy(batch["y_hist"]).float().to(device)
            frames, available = latest_real_frames(batch["V"], batch["mask_visual"])
            samples = (
                forecast_batch(
                    model,
                    histories,
                    frames,
                    available.to(device),
                    vision_mode=vision_mode,
                    pred_len=cfg.horizon,
                    inference_token_len=cfg.inference_token_len,
                    num_samples=cfg.num_samples,
                )
                .float()
                .cpu()
                .numpy()
            )
            sample_parts.append(samples)
            true_parts.append(batch["y_future"])
            valid_parts.append(batch["mask_future"] * batch["daylight_future"])
            history_parts.append(batch["y_hist"][:, -1])
            history_valid_parts.append(batch["mask_hist"][:, -1])
            target_valid_parts.append(batch["mask_future"])
            daylight_parts.append(batch["daylight_future"])
        samples, true, valid = map(
            np.concatenate, (sample_parts, true_parts, valid_parts)
        )
        last_history, last_history_valid, target_valid, daylight = map(
            np.concatenate,
            (history_parts, history_valid_parts, target_valid_parts, daylight_parts),
        )
        pred = np.clip(samples.mean(axis=1), 0.0, 1.0).astype(np.float32)
        np.savez(
            out / f"aurora_{site}_pred.npz",
            pred=pred,
            true=true,
            samples=samples.astype(np.float32),
            valid=valid,
            last_history=last_history,
            last_history_valid=last_history_valid,
            target_valid=target_valid,
            daylight=daylight,
            protocol_aligned=np.array(True),
        )
        totals["pred"].append(pred)
        totals["true"].append(true)
        totals["valid"].append(valid)
        totals["site"].append(np.full(len(pred), site))
    return macro_mae(
        *(np.concatenate(totals[key]) for key in ("pred", "true", "valid", "site"))
    )


@hydra.main(
    version_base=None, config_path="../../../configs/tier5", config_name="aurora"
)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AuroraForPrediction.from_pretrained(cfg.ckpt_path).to(device).eval()
    root = Path(cfg.out)
    if cfg.vision_mode == "auto":
        scores = {}
        for mode in ("pseudo", "real"):
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            scores[mode] = _run_split(
                model, cfg, "val", mode, root / "validation" / mode
            )
        selected = min(scores, key=scores.get)
    else:
        selected, scores = cfg.vision_mode, {}
    test_score = _run_split(model, cfg, "test", selected, root)
    summary = {
        "selected_vision_mode": selected,
        "validation_macro_mae": scores,
        "test_macro_mae": test_score,
        "num_samples": cfg.num_samples,
    }
    (root / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
