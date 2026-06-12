"""Full evaluation suite orchestrator (BASELINE_COMPARISON.md §8 + §4.1).

Maps scenarios S1-S6, the §5 robustness controls and the A15 RAG sweep onto
run_eval.py invocations. Default is --dry-run: print the exact commands so
they can be reviewed / dispatched to SLURM; --execute runs them sequentially.

    uv run python scripts/run_suite.py                  # print the plan
    uv run python scripts/run_suite.py --execute        # run everything
    uv run python scripts/run_suite.py --only s2 s6     # subset
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

BASE = "uv run python run_eval.py"
SEEDS = "--seeds 42 43 44"

ZS_MODELS = "persistence seasonal_naive climatology_hourly chronos2_zs timesfm_zs tirex_zs"
TRAINED_MODELS = "lightgbm mlp dlinear patchtst itransformer tft"
TIER4_MODELS = "ts_rag cross_rag cora"
ALL_MODELS = f"{ZS_MODELS} {TRAINED_MODELS} {TIER4_MODELS}"

# scenario id → list of run_eval.py argument strings
SUITE: dict[str, list[str]] = {
    # S1 sanity / upper bound
    "s1": [f"--model {ALL_MODELS} --in-domain --tag s1 {SEEDS}"],
    # S2 headline cross-plant (S6 ramp columns are always included)
    "s2": [f"--model {ALL_MODELS} --tag s2 {SEEDS}"],
    # S3 cross-dataset transfer, both directions
    "s3": [
        f"--model {ALL_MODELS} --train-datasets uk_pv "
        f"--eval-datasets goes_pvdaq --tag s3_uk2goes {SEEDS}",
        f"--model {ALL_MODELS} --train-datasets goes_pvdaq "
        f"--eval-datasets uk_pv --tag s3_goes2uk {SEEDS}",
    ],
    # S4 long horizon decay
    "s4": [
        f"--model {ALL_MODELS} --horizon 24 --eval-stride 24 --tag s4_h24 {SEEDS}",
        f"--model {ALL_MODELS} --horizon 48 --eval-stride 48 --tag s4_h48 {SEEDS}",
    ],
    # S5 data efficiency curves
    "s5": [
        f"--model {TRAINED_MODELS} {TIER4_MODELS} --train-fraction {frac} "
        f"--tag s5_f{int(frac * 100):03d} {SEEDS}"
        for frac in (0.10, 0.25, 0.50, 1.00)
    ],
    # §5 robustness battery (numerical controls)
    "controls": [
        f"--model {ALL_MODELS} --control zero_cov --tag ctl_zerocov",
        f"--model {ALL_MODELS} --control low_history_4 --tag ctl_hist4",
        f"--model {ALL_MODELS} --control low_history_8 --tag ctl_hist8",
        f"--model {ALL_MODELS} --control low_history_12 --tag ctl_hist12",
    ],
    # A15: RAG datastore size / top-k sweep (fairness: tuned, not strawmanned)
    "a15": [
        f"--model ts_rag --model-kwargs '{{\"top_k\": {k}}}' --tag a15_k{k}"
        for k in (2, 4, 8, 16, 32)
    ] + [
        f"--model ts_rag --model-kwargs '{{\"max_datastore\": {m}}}' "
        f"--tag a15_store{m}"
        for m in (10_000, 50_000, 200_000)
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=sorted(SUITE),
                        choices=sorted(SUITE))
    parser.add_argument("--execute", action="store_true",
                        help="run the commands (default: dry-run print)")
    args = parser.parse_args()

    baselines_dir = Path(__file__).resolve().parents[1]
    commands = [
        f"{BASE} {run_args}"
        for scenario in args.only
        for run_args in SUITE[scenario]
    ]
    print(f"# {len(commands)} run_eval invocations "
          f"({'EXECUTING' if args.execute else 'dry-run'})\n")
    for cmd in commands:
        print(cmd)
        if args.execute:
            result = subprocess.run(
                shlex.split(cmd), cwd=baselines_dir,
            )
            if result.returncode != 0:
                sys.exit(f"command failed (exit {result.returncode}): {cmd}")


if __name__ == "__main__":
    main()
