# Conventions — layout, naming, config, git

Single source of truth for *how code is organized*. `AGENTS.md` states the rules an agent
must obey; this file carries the detail behind them.

---

## 1. Repository layout (as it actually is)

```
MMTSFM/                      # the model — main package `mmtsfm`
  src/mmtsfm/
    train.py                 # Hydra entrypoint (`uv run python -m mmtsfm.train`)
    data/                    # dataset.py, datamodule.py, pv_record.py
    models/
      base.py
      chronos2/              # vision_chronos2.py, model.py, grassmann.py, layers.py,
                             # lightning_module.py, pipeline.py, config.py
      vision/                # visual_encoder.py, latent_summarizer.py,
                             # cross_modal_adapter.py, vidtok_encoder.py
  src/eval/                  # metrics.py, evaluator.py, protocol_eval.py
  configs/                   # Hydra groups: model/ data/ trainer/ stage/
  tests/                     # mirrors src/mmtsfm/
  scripts/                   # slurm_curriculum.sh, precache_login.sh, extract_vjepa.sbatch

baselines/                   # the comparison suite
  tier0/ … tier6/            # one dir per tier (tier*/vendor/ = vendored third-party)
  common/                    # config.py, windows.py, splits.py, metrics.py, runner.py
  run_eval.py                # canonical runner
  results/                   # result JSONs + ALL_RESULTS.md (generated)
  scripts/                   # SLURM submitters, aggregate_all.py

knowledge/                   # ALL project prose + papers (this tree) — Graphify domain
report/                      # ongoing results: EDA report, baseline test report
manuscript/                  # thesis LaTeX (chapters, biblio, front matter)
dataset_exploration/         # EDA notebooks/scripts
scripts/                     # dataset build + verification utilities
```

> `src/pvtsfm/` does not exist and never did. Any doc describing a `pvtsfm` package or a
> "port from MMTSFM" is stale — the MMTSFM package **is** the model of record.

---

## 2. File naming

| Pattern | Contains | Example |
|---|---|---|
| `{component}.py` | exactly one `nn.Module` class | `grassmann.py` → `CausalGrassmannMixing` |
| `{verb}_{noun}.py` | exactly one pure function | `build_batch.py` |
| `lightning_{stage}.py` | one Lightning module variant | `lightning_module.py` |
| `test_{module}.py` | shape + gradient smoke tests, mirroring the source path | `tests/test_vision_chronos2.py` |
| `scripts/{verb}_{noun}.{py,sh}` | one script capability | `scripts/extract_video_embeddings.py` |

Class name = PascalCase of the file name. Modules are `snake_case`.
Target **< 150 lines per file**; one class or one script capability per file.

---

## 3. Imports

- Main package is `mmtsfm`, rooted at `MMTSFM/src/mmtsfm/`. Run as `python -m mmtsfm.train`.
- Relative imports **within** the `mmtsfm` package only.
- No circular imports — specifically between `models/chronos2/` and `models/vision/`.
- Shared types live in one module (e.g. `types.py`), never duplicated across files.

---

## 4. Hydra

- Every hyperparameter lives in `MMTSFM/configs/`. **No magic numbers in model code.**
- Config path mirrors the module path: `configs/model/vision_chronos2_grassmann.yaml`.
- Complex submodules use the `@dataclass` + `instantiate` pattern.
- Hydra only — no `argparse`, no `yaml.load` outside Hydra.
- Baseline configs are **self-contained per baseline codebase** (`baselines/configs/`).

Experiment run names: `{ablation}_{variant}_{date}` — e.g. `fusion_interleaved_crossplant_2026-06-12`.

---

## 5. Git

| Prefix | Use |
|---|---|
| `exp/<name>` | experiments / ablations |
| `feat/<name>` | features |
| `fix/<name>` | bugfixes |

- **Never work on `main`.** Branch first, merge locally, push `main` only.
- Commit message: `exp(<name>): …` / `feat(<name>): …` / `fix(<name>): …`, one logical
  change per commit. Micro-commit after each verified sub-task; do not accumulate.
- If tests fail and the fix is not obvious, `git checkout` / `git reset --hard` back to the
  last verified commit rather than stacking untested workarounds.
- Never commit data, checkpoints, logs, or large binaries.
- Before claiming done: `git diff HEAD`, review every line, strip debug prints.

---

## 6. Generated artifacts — never hand-edit

| Path | Regenerate with |
|---|---|
| `graphify-out/` | `graphify update knowledge/` |
| `.gitnexus/` | `node .gitnexus/run.cjs analyze` |
| `baselines/results/ALL_RESULTS.md` | `python baselines/scripts/aggregate_all.py` |
| `report/REPORT.pdf`, `manuscript/*.pdf` | `latexmk` |
