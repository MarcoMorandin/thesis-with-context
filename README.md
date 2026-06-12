# PVTSFM  -  PV Temporal Spatiotemporal Foundation Model Research Repository

This repository is the **single source of truth for the entire research project**. It houses the research documentation, dataset exploration scripts, the prior MMTSFM proposal framework, and the external baselines (e.g., SolarVLM) used for benchmarking zero-shot cross-plant solar power forecasting.

---

## Repository Structure

```
thesis-with-context/
├── MMTSFM/                      # Full MMTSFM proposal codebase (configs, src, tests, scripts)
├── baselines/
│   └── solar_vlm/               # Clean port of the SolarVLM baseline & configs
├── dataset_exploration/         # Standalone EDA code, report, and plots
├── docs/                        # Research documentation
│   ├── architecture/            # Architecture overviews & roadmaps
│   ├── context/                 # Context engineering guidelines & data contracts
│   └── experiments/             # Ablation registry & baseline protocols
└── knowledge/                   # Research literature and proposals (Graphify input)
```

---

## Research Parameters & Decisions
* **Repo Strategy**: Single source of truth containing everything. GPU baselines are configured with SLURM scripts for cluster runs.
* **Cross-Plant Protocol**: Zero-shot cross-plant generalization evaluated on disjoint test plants using their small history, rather than few-shot in-context learning.
* **Visual Modality**: Satellite PNG images.
* **Horizon**: Intra-hour granularity with long-horizon forecasting.
* **Configuration**: Configuration for each baseline is self-contained within its respective directory.

---

## Quick Start Guide

### 1. Dataset Exploration (EDA)
Explore the telemetry and visual datasets using the standalone tools:
```bash
cd dataset_exploration
# Run the EDA script
uv run run_eda.py
```
Detailed findings are logged in [dataset_exploration_report.md](file:///Users/marcomorandin/Desktop/thesis-with-context/dataset_exploration/dataset_exploration_report.md).

### 2. Prior Proposal Framework (MMTSFM)
Run the Chronos-2 + V-JEPA + Grassmann temporal-mixing framework:
```bash
cd MMTSFM
# Install dependencies
uv sync --dev
# Run a training smoke-test on synthetic data
uv run python -m mmtsfm.train
```

### 3. SolarVLM Baseline
Run the multimodal LLM-driven baseline:
```bash
cd baselines/solar_vlm
# Set up the environment
source setup_env.sh
# Train on SKIPP'D (using precomputed offline features)
python run_skippd.py --is_training 1 --use_offline_vision --vision_feat_dir /path/to/feats
```

---

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

---

## Research Documents & Guidelines

* [AGENTS.md](file:///Users/marcomorandin/Desktop/thesis-with-context/AGENTS.md) — Unified instruction file mandating Git workflows, naming conventions, and constraints.
* [CLAUDE.md](file:///Users/marcomorandin/Desktop/thesis-with-context/CLAUDE.md) — Claude Code specific settings and commands.
* [docs/context/DATASET_CONTRACT.md](file:///Users/marcomorandin/Desktop/thesis-with-context/docs/context/DATASET_CONTRACT.md) — Standardized-dataset schema, splits, and tensor output contracts.
* [docs/experiments/BASELINE_PROTOCOL.md](file:///Users/marcomorandin/Desktop/thesis-with-context/docs/experiments/BASELINE_PROTOCOL.md) — Fair comparison protocols and metrics.
* **TODO**: `docs/architecture/CROSS_PLANT_CONTEXT.md` — How the plant descriptor / small history is ingested without few-shot.

---

## External Resources

| Resource | Path |
|----------|------|
| Prepared dataset | `/Volumes/SSD/standardized-dataset` |
| Prior implementation (MMTSFM) | `/Users/marcomorandin/Desktop/MMTSFM` |
| SolarVLM baseline | `/Users/marcomorandin/Desktop/Code-Thesis/Solar-VLM-original` |
| Baseline papers | `knowledge/papers/baselines/` |
| Related work | `knowledge/papers/related/` |

---

## Stack
- **Python**: uv only (isolated environment per baseline package)
- **Config**: Hydra composed hierarchies (contained per baseline)
- **Backbones**: Chronos-2 (TS), V-JEPA 2.1 (vision), Qwen-VL (SolarVLM)
