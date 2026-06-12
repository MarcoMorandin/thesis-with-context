"""Efficiency table (§4.6): params, single-window latency, peak memory.

Fits nothing heavyweight here — trained models are loaded/fit with the small
synthetic frame just to materialize their parameters; the table reports
trainable/total params, single-window CPU latency (median of N runs) and,
on CUDA, peak VRAM. GPU-hours-to-train must be filled from the actual run
logs (W&B), not from this script.

    uv run python scripts/efficiency.py --models persistence dlinear patchtst
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.base import build  # noqa: E402
from tests.conftest import make_frame, windows_for  # noqa: E402


def count_params(model) -> dict[str, int]:
    torch_module = None
    for attr in ("_model", "_adapter"):
        candidate = getattr(model, attr, None)
        if candidate is not None and hasattr(candidate, "parameters"):
            torch_module = candidate
            break
    if torch_module is None:
        return {"total": 0, "trainable": 0}
    total = sum(p.numel() for p in torch_module.parameters())
    trainable = sum(
        p.numel() for p in torch_module.parameters() if p.requires_grad
    )
    return {"total": total, "trainable": trainable}


def measure_latency(model, batch, n_runs: int = 50) -> float:
    """Median single-window forward latency in milliseconds."""
    single = {
        k: (v[:1] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
        for k, v in batch.items()
    }
    model.predict(single)  # warmup
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(single)
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", default="results/efficiency.json")
    args = parser.parse_args()

    df = make_frame(n_sites=3, days=6)
    sites = lambda *ids: df[df.site_id.isin(ids)]  # noqa: E731
    train = windows_for(sites("site_0", "site_1"))
    val = windows_for(sites("site_2"))
    batch = val.batch(list(range(8)))

    rows = {}
    for name in args.models:
        model = build(name)
        if model.requires_fit:
            model.fit(train, val)
        rows[name] = {
            "params": count_params(model),
            "latency_ms_cpu": measure_latency(model, batch),
            "zero_shot": not model.requires_fit,
        }
        try:
            import torch

            if torch.cuda.is_available():
                rows[name]["peak_vram_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
        except ImportError:
            pass
        print(f"{name}: {rows[name]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
