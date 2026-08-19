"""G1 localization probes.

Reads only artifacts that already exist: the per-site prediction dumps written by
ProtocolEvaluator (`results/predictions/<tag>_<site>_pred.npz`) and the covariate
table. Distinguishes C5 (horizon mismatch / dilution) from a uniform effect.
"""

from __future__ import annotations

import numpy as np


def decompose_by_horizon(
    pred_on: np.ndarray,
    pred_off: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Per-horizon NMAE with vision on vs off, and their difference.

    A positive `delta[h]` means vision helped at horizon step h. The aggregate
    marginal gain is a mask-weighted average of these, so a strong early effect
    can be invisible once averaged over 12 steps.
    """
    h = y.shape[1]
    nmae_on, nmae_off = [], []
    for i in range(h):
        m = mask[:, i]
        if not m.any():
            nmae_on.append(float("nan"))
            nmae_off.append(float("nan"))
            continue
        nmae_on.append(float(np.abs(pred_on[m, i] - y[m, i]).mean()))
        nmae_off.append(float(np.abs(pred_off[m, i] - y[m, i]).mean()))
    delta = [
        (nmae_off[i] - nmae_on[i])
        if not (np.isnan(nmae_on[i]) or np.isnan(nmae_off[i]))
        else float("nan")
        for i in range(h)
    ]
    return {"nmae_on": nmae_on, "nmae_off": nmae_off, "delta": delta}


def stratify_by_variability(
    delta_per_sample: np.ndarray,
    csi_var: np.ndarray,
    n_bins: int = 3,
) -> dict:
    """Group the per-sample vision benefit by within-window sky variability.

    If vision only pays on variable-sky windows, the aggregate is diluted by the
    clear and fully-overcast majority — a reporting problem, not a model problem.
    """
    order = np.argsort(csi_var)
    bins = np.array_split(order, n_bins)
    return {
        "mean_delta": [float(delta_per_sample[b].mean()) for b in bins],
        "counts": [int(len(b)) for b in bins],
        "var_edges": [float(csi_var[b].min()) for b in bins]
        + [float(csi_var[order[-1]])],
    }


def gate_stats(ckpt_path: str) -> dict:
    """Fusion-gate and modality-bias statistics from a trained checkpoint.

    Tests C4 (modality laziness). Two readings, both against a known-good
    contrast: modality_pair_bias was EXACTLY 0.0 in all 12 blocks through s1 and
    s2a, because that pathway only becomes active under interleaved fusion. A
    still-zero bias after interleaved training means the pathway received no
    usable gradient.

    Returns per-block: mean |modality_pair_bias|, and W_gate bias mean (a proxy
    for the resting alpha, since alpha = sigmoid(W_gate(u)) and a large positive
    bias pins alpha toward the numeric residual, closing the visual path).
    """
    import torch

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    blocks: dict[int, dict[str, float]] = {}
    for k, v in sd.items():
        if ".encoder.block." not in k:
            continue
        idx = int(k.split(".encoder.block.")[1].split(".")[0])
        e = blocks.setdefault(idx, {})
        if k.endswith("modality_pair_bias"):
            e["modality_pair_bias_absmean"] = float(v.float().abs().mean())
        elif k.endswith("layer.0.W_gate.bias"):
            e["w_gate_bias_mean"] = float(v.float().mean())
    return {
        "per_block": {str(i): blocks[i] for i in sorted(blocks)},
        "n_blocks_with_zero_modality_bias": sum(
            1
            for e in blocks.values()
            if e.get("modality_pair_bias_absmean", 1.0) == 0.0
        ),
    }
