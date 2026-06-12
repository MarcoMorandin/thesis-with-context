# PVTSFM  -  PV Temporal Spatiotemporal Foundation Model

This repository is the **single source of truth for the entire research project**, housing the code for the main foundation model, all baselines, documentation, and evaluation infrastructure. The goal is to build a research-grade multimodal foundation model for **PV power forecasting** by injecting numerical data, covariates, and visual frames (satellite PNG images) using foundation models (Chronos-2, V-JEPA 2.1).

## Research Parameters & Decisions
* **Repo Strategy**: A clean research repository containing everything. Baselines requiring GPU training will have SLURM scripts configured to run on the cluster.
* **Cross-Plant Protocol**: Held-out plants are used as a disjoint test set. Plants will have a small history, and the model must generalize based on what it learned from other plants (zero-shot generalization, not few-shot context injection).
* **Baselines**: Multiple baselines will be compared against, including:
  * TSFM zero-shot
  * TSFM finetuning
  * TSFM + RAG (or similar memory/retrieval approaches)
  * Deep Learning baselines
  * Classical Machine Learning baselines
* **Visual Modality**: Satellite PNG images.
* **Horizon**: Intra-hour granularity with a hopefully long horizon (to be verified).
* **Configuration**: Configuration for each baseline will be self-contained within each baseline's module inside this bigger codebase.

## Quick start

```bash
# Install (uv only)
uv sync --dev

# Train (main model example)
uv run pvtsfm-train trainer.max_epochs=1

# Test architecture pieces
uv run pytest tests/ -k "not integration"
```

## Context Engineering & Agent Skills

To maintain a clean and highly effective agentic coding environment, we utilize specific skills and knowledge graph tools:

### Skills to Install / Enable
| Skill | Why |
|-------|-----|
| `lean-ctx` | Token-efficient reads (mandatory over native reads) |
| `academic-research-skills` | For deep-research, academic-paper writing, and reviewer pipelines |
| `graphify` | Managing the research knowledge graph |
| `systematic-debugging` | Resolving training convergence bugs |
| `verification-before-completion` | Never claiming a fix works without pytest/logs |
| `gitnexus-*` | Exploring, debugging, impact-analysis, and refactoring |

### Knowledge Graphs
We use two systems to separate code understanding from research understanding:
1. **GitNexus** (`npx gitnexus analyze && npx gitnexus setup`): Primary tool for the codebase graph. It analyzes code dependencies, call chains, and blast radius.
2. **Graphify** (`/graphify knowledge/ --wiki --update`): Primary tool for the research corpus. It tracks papers, proposals, cross-doc synthesis, and the audit trail. 
*Note: Make sure to place papers in `knowledge/papers/baselines` and `knowledge/papers/related`, and internal documents in `knowledge/docs/` before updating the Graphify wiki.*

## Research Documents to Write (TODOs)
The following documents still need to be written to complete the context engineering:
- `docs/context/DATASET_CONTRACT.md` — Schema of `standardized-dataset` (columns, splits, no ETL code).
- `docs/experiments/BASELINE_PROTOCOL.md` — Fair comparison rules (same horizon, same plants, same metrics).
- `docs/architecture/CROSS_PLANT_CONTEXT.md` — How the plant descriptor / small history is ingested without few-shot.

## Dataset Exploration
Exploratory Data Analysis (EDA) and data curation scripts are located in [dataset_exploration/](file:///Users/marcomorandin/Desktop/thesis-with-context/dataset_exploration):
- `run_eda.py`: Runs exploratory analysis over target PV datasets.
- `curate_dataset.py`: Filters and curates subsets of solar plant telemetry.
- `pack_images.py`: Packages raw images into standardized spatial-temporal structures.
- `plots/`: Contains 35 diagnostic plots (brightness vs. power, diurnal profiles, capacity distribution, etc.).
- `dataset_exploration_report.md`: Detailed exploration analysis and findings report.

## External resources

| Resource | Path |
|----------|------|
| Prepared dataset | `/Volumes/SSD/standardized-dataset` |
| Prior implementation (MMTSFM) | `/Users/marcomorandin/Desktop/MMTSFM` |
| SolarVLM baseline | `/Users/marcomorandin/Desktop/Code-Thesis/Solar-VLM-original` |
| Baseline papers | `baselines/` |
| Related work | `solar-related-work/` |

## Stack

- **Python**: uv only
- **Config**: Hydra only (contained per baseline)
- **Training**: PyTorch Lightning
- **Backbones**: Chronos-2 (TS), V-JEPA 2.1 (vision)
