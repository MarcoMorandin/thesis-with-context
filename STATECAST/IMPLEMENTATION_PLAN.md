# StateCast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the StateCast package (learned data assimilation for asynchronous multimodal forecasting, design of record [STATECAST.md](../STATECAST.md)) as a standalone project folder mirroring [MMTSFM](../MMTSFM), through Stage-0 (AsyncBench synthetic) training, state-recovery probes, the attention twin, the G4 stress harness, and the uk_pv protocol adapter.

**Architecture:** All sensor streams are noisy observations of one latent state `s` (8 tokens × 256 dims). Each stream has an encoder `E_m`, a predicted-observation readout `h_m`, and an innovation-gated update `a_m` (learned Kalman-gain analog). A Δt-conditioned transition `f(s, Δt)` rolls the state between observations and across the horizon; a conditional flow-matching head samples stochastic transitions for ensembles. The only entity-specific component is ψ (~96 dims), amortized from history by a set encoder and consumed by the quantile readout `g_ψ`. A matched-parameter attention twin plus a stress harness implement the formulation-vs-mechanism study (G4).

**Tech Stack:** PyTorch ≥2.0, Lightning ≥2.1, Hydra ≥1.3, uv, numpy, pytest. No new heavy deps — frozen perception encoders (V-JEPA) stay outside the model boundary (observations arrive pre-embedded, as in MMTSFM's latent cache).

## Global Constraints

(from [STATECAST.md](../STATECAST.md) and repo rules in [CLAUDE.md](../CLAUDE.md) / [AGENTS.md](../AGENTS.md))

- Python via `uv` only — every command is `uv run ...`; deps via `uv add` / `uv sync`. Run all commands from `STATECAST/` (it is its own uv project, like `MMTSFM/`).
- Hydra only for config — no argparse in training entrypoints; all hyperparameters in `STATECAST/configs/`, no magic numbers in model code.
- One class or one script capability per file; target < 150 lines per file.
- File naming: `{component}.py` for a single `nn.Module`, `{verb}_{noun}.py` for a pure function, `lightning_{stage}.py` for Lightning modules.
- Relative imports inside the `statecast` package; shared types in `statecast/types.py`; no circular imports.
- Tests mirror `src/statecast/` under `tests/`; every module file gets `test_<module>.py` with shape + gradient smoke tests. Verify with `uv run pytest` before claiming completion.
- Latent state: **8 tokens × 256 dims**. Quantiles: **9** levels (0.1 … 0.9). ψ: **~10² dims** (96 here). Ensemble: **K = 8** flow samples. Budget ≤ **30M** trainable params.
- All prediction losses in **latent space** (JEPA-style) — no pixel decoding anywhere.
- No physics heuristics (CSI conversion, irradiance formulas) — the synthetic "sun/cloud" structure lives only in the AsyncBench *generator*, never in the model.
- Never commit data, checkpoints, logs, or large binaries. Branch: `feat/statecast`. Commit format `feat(statecast): <what and why>`, micro-commits after each verified sub-task.
- Do not modify `/leonardo_scratch/fast/IscrC_MTSFM/data` or `baselines/` (read/import only).
- Time is always **float minutes** in model + data code; Δt is always non-negative.

**Batching convention (load-bearing, used by every data/model task):** within one batch, all samples share the same observation *schedule* (event times per stream). AsyncBench samples a random schedule per dataset instance; schedules vary across instances/epochs, not within a batch. This keeps tensors rectangular — `t: (E,)` shared, `v: (B, E, d)`, `mask: (B, E)` — while still exercising asynchrony. (uk_pv is grid-regular per dataset, so this also matches the real testbed.)

**Batch schema (produced by Task 3, consumed by Tasks 9–12, 14–16):**

```python
batch = {
    "streams": {                      # every observation stream, history + horizon
        name: {
            "t":    FloatTensor (E,),      # event times, minutes, ascending (shared in batch)
            "v":    FloatTensor (B, E, d), # raw values (d = stream dim)
            "mask": FloatTensor (B, E),    # 1 = observed, 0 = missing
        }
    },
    "future_known": tuple[str, ...],  # streams assimilable during the horizon ("sun", "nwp")
    "t0": FloatTensor (1,),           # forecast origin, minutes
    "t_future": FloatTensor (H,),     # horizon step times (> t0)
    "y_future": FloatTensor (B, H),   # target over horizon
    "psi_true": FloatTensor (B, 3),   # generator entity params (loc, amp, tilt_shift)
    "true_cloud_t0": FloatTensor (B,),# ground-truth latent variable at t0 (probe target)
}
```

---

### Task 1: Project scaffold

**Files:**
- Create: `STATECAST/pyproject.toml`
- Create: `STATECAST/README.md`
- Create: `STATECAST/src/statecast/__init__.py`
- Create: `STATECAST/tests/__init__.py`
- Create: `STATECAST/tests/test_scaffold.py`

**Interfaces:**
- Produces: importable `statecast` package; `uv run pytest` green; branch `feat/statecast`.

- [ ] **Step 1: Create branch**

```bash
cd /Users/marcomorandin/Desktop/thesis-with-context
git checkout -b feat/statecast
```

- [ ] **Step 2: Write pyproject**

`STATECAST/pyproject.toml` (trimmed from MMTSFM's; same cu121 pin so Leonardo works):

```toml
[project]
name = "statecast"
version = "0.1.0"
description = "StateCast: learned data assimilation for asynchronous multimodal forecasting"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0",
    "lightning>=2.1.0",
    "hydra-core>=1.3.2",
    "omegaconf>=2.3.0",
    "wandb>=0.16.0",
    "python-dotenv>=1.0.0",
    "pyrootutils>=1.0.4",
    "einops>=0.7.0",
    "numpy>=1.26.0",
    "pandas>=2.0.0",   # uk_pv adapter (dataset-of-record parquet)
    "pyarrow>=15.0.0",
    "h5py>=3.10.0",    # uk_pv adapter (images_all.h5 pointers; frames stay outside model)
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
pythonpath = ["src"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = ["pytest>=8.0.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/statecast"]

# Leonardo compute-node driver is CUDA 12.2 → pin cu121 wheels on Linux (no-op on macOS).
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu121", marker = "sys_platform == 'linux'" }]
```

- [ ] **Step 3: Write README stub**

`STATECAST/README.md`:

```markdown
# StateCast — learned data assimilation for asynchronous multimodal forecasting

Implementation of [STATECAST.md](../STATECAST.md) (MMTSFM v5, design of record).
One latent state per entity; every sensor stream enters through a learned
observation operator with an innovation gate; forecasting = rolling a
Δt-conditioned (flow-matching) transition; per-entity readout ψ is amortized
in-context. See IMPLEMENTATION_PLAN.md for the build order.

```bash
# Stage-0 smoke train (AsyncBench synthetic)
uv run python -m statecast.train

# tests
uv run pytest
```
```

- [ ] **Step 4: Write package init + scaffold test**

`STATECAST/src/statecast/__init__.py`:

```python
"""StateCast: learned data assimilation for asynchronous multimodal forecasting."""

__version__ = "0.1.0"
```

`STATECAST/tests/__init__.py`: empty file.

`STATECAST/tests/test_scaffold.py`:

```python
def test_package_imports():
    import statecast

    assert statecast.__version__ == "0.1.0"
```

- [ ] **Step 5: Sync and run tests**

```bash
cd STATECAST && uv sync && uv run pytest tests/test_scaffold.py -v
```
Expected: `test_package_imports PASSED`

- [ ] **Step 6: Commit**

```bash
git add STATECAST/pyproject.toml STATECAST/README.md STATECAST/src STATECAST/tests STATECAST/uv.lock
git commit -m "feat(statecast): scaffold standalone uv project mirroring MMTSFM layout"
```

---

### Task 2: AsyncBench latent causal process

**Files:**
- Create: `STATECAST/src/statecast/data/__init__.py` (empty)
- Create: `STATECAST/src/statecast/data/asyncbench_process.py`
- Test: `STATECAST/tests/data/__init__.py` (empty), `STATECAST/tests/data/test_asyncbench_process.py`

**Interfaces:**
- Produces: `LatentProcess(field_len, n_steps, dt_minutes, advection, noise, seed)` with `simulate() -> {"field": (T, L) float32 in [0,1], "t_minutes": (T,) float32}`, `LatentProcess.local_cloud(field, loc, width) -> (T,)`, `LatentProcess.daylight(t_minutes) -> (T,) in [0,1]`, `target(field, t_minutes, loc, amp, tilt_shift) -> (T,) float32`. Consumed by Task 3 renderer and Task 13 probes (ground-truth state).

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/data/test_asyncbench_process.py`:

```python
import numpy as np

from statecast.data.asyncbench_process import LatentProcess


def test_simulate_shapes_and_range():
    proc = LatentProcess(field_len=32, n_steps=128, dt_minutes=15.0, seed=0)
    sim = proc.simulate()
    assert sim["field"].shape == (128, 32)
    assert sim["t_minutes"].shape == (128,)
    assert sim["field"].min() >= 0.0 and sim["field"].max() <= 1.0
    assert np.all(np.diff(sim["t_minutes"]) == 15.0)


def test_advection_moves_field():
    proc = LatentProcess(field_len=32, n_steps=64, advection=1, noise=0.0, seed=1)
    sim = proc.simulate()
    # with zero noise, frame t+1 is a smoothed roll of frame t: correlation with
    # the rolled previous frame beats correlation with the unrolled one
    a, b = sim["field"][10], sim["field"][11]
    corr_rolled = np.corrcoef(np.roll(a, 1), b)[0, 1]
    corr_static = np.corrcoef(a, b)[0, 1]
    assert corr_rolled > corr_static


def test_target_is_zero_at_night_and_entity_dependent():
    proc = LatentProcess(field_len=32, n_steps=192, dt_minutes=15.0, seed=2)
    sim = proc.simulate()
    y1 = proc.target(sim["field"], sim["t_minutes"], loc=3, amp=1.0, tilt_shift=0.0)
    y2 = proc.target(sim["field"], sim["t_minutes"], loc=3, amp=2.0, tilt_shift=90.0)
    night = LatentProcess.daylight(sim["t_minutes"]) == 0.0
    assert np.all(y1[night] == 0.0)
    assert not np.allclose(y1, y2)
    assert y1.shape == (192,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_asyncbench_process.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'statecast.data'`

- [ ] **Step 3: Implement the process**

`STATECAST/src/statecast/data/asyncbench_process.py`:

```python
"""AsyncBench ground truth: advecting latent field -> local cloudiness -> entity target.

Causal chain (STATECAST.md Stage 0): an advecting 1-D field (the "sky") drives a
local intermediate variable (cloud over the entity), which — modulated by a
deterministic diurnal signal and an entity transfer function (loc, amp,
tilt_shift) — produces the target. Ground-truth state is kept for probes.
"""
from __future__ import annotations

import numpy as np


class LatentProcess:
    def __init__(
        self,
        field_len: int = 32,
        n_steps: int = 512,
        dt_minutes: float = 15.0,
        advection: int = 1,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.field_len = field_len
        self.n_steps = n_steps
        self.dt_minutes = dt_minutes
        self.advection = advection
        self.noise = noise

    def simulate(self) -> dict:
        L, T = self.field_len, self.n_steps
        field = np.empty((T, L), dtype=np.float32)
        kernel = np.array([0.25, 0.5, 0.25])
        row = np.clip(self.rng.normal(0.5, 0.25, size=L), 0.0, 1.0)
        for t in range(T):
            field[t] = row
            row = np.roll(row, self.advection)
            row = np.convolve(np.pad(row, 1, mode="wrap"), kernel, mode="valid")
            row = np.clip(row + self.rng.normal(0.0, self.noise, size=L), 0.0, 1.0)
        t_minutes = np.arange(T, dtype=np.float32) * self.dt_minutes
        return {"field": field, "t_minutes": t_minutes}

    @staticmethod
    def local_cloud(field: np.ndarray, loc: int, width: int = 3) -> np.ndarray:
        idx = (np.arange(-width, width + 1) + loc) % field.shape[1]
        return field[:, idx].mean(axis=1).astype(np.float32)

    @staticmethod
    def daylight(t_minutes: np.ndarray, day_minutes: float = 1440.0) -> np.ndarray:
        phase = 2.0 * np.pi * (np.asarray(t_minutes) % day_minutes) / day_minutes
        return np.clip(np.sin(phase), 0.0, None).astype(np.float32)

    def target(
        self,
        field: np.ndarray,
        t_minutes: np.ndarray,
        loc: int,
        amp: float,
        tilt_shift: float,
    ) -> np.ndarray:
        cloud = self.local_cloud(field, loc)
        sun = self.daylight(t_minutes + tilt_shift)
        return (amp * sun * (1.0 - 0.8 * cloud)).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_asyncbench_process.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/data STATECAST/tests/data
git commit -m "feat(statecast): AsyncBench latent causal process with ground-truth state"
```

---

### Task 3: AsyncBench renderer + torch Dataset

**Files:**
- Create: `STATECAST/src/statecast/data/asyncbench_render.py`
- Create: `STATECAST/src/statecast/data/asyncbench_dataset.py`
- Test: `STATECAST/tests/data/test_asyncbench_render.py`, `STATECAST/tests/data/test_asyncbench_dataset.py`

**Interfaces:**
- Consumes: `LatentProcess` (Task 2).
- Produces:
  - `render_streams(sim, loc, amp, tilt_shift, schedule, rng) -> dict[str, dict]` with keys `"target"(d=1), "aux"(d=2), "vision"(d=8), "sun"(d=1), "nwp"(d=1)`; each value is per-stream numpy (single entity): `{"t": (E,) float32, "v": (E, d) float32, "mask": (E,) float32}`.
  - `Schedule` dataclass: `cadence: dict[str, int]` (base steps between events), `offset: dict[str, int]`, `p_missing: dict[str, float]`, `noise: dict[str, float]`, `vis_window_steps: int`.
  - `AsyncBenchDataset(n_entities, windows_per_entity, hist_steps, horizon_steps, field_len, dt_minutes, schedule_seed, entity_seed, cadence_scale=1)` (`cadence_scale=2` ⇒ all cadences halved, the Task-15 stress axis) — `__getitem__` returns the **batch schema** dict from Global Constraints (single-sample; default collate stacks it). `STREAM_DIMS = {"target": 1, "aux": 2, "vision": 8, "sun": 1, "nwp": 1}` and `FUTURE_KNOWN = ("sun", "nwp")` exported as module constants (Tasks 10, 12, 14 read them).

- [ ] **Step 1: Write the failing tests**

`STATECAST/tests/data/test_asyncbench_render.py`:

```python
import numpy as np

from statecast.data.asyncbench_process import LatentProcess
from statecast.data.asyncbench_render import Schedule, default_schedule, render_streams


def test_render_stream_shapes_and_cadence():
    proc = LatentProcess(n_steps=256, seed=0)
    sim = proc.simulate()
    sched = default_schedule(seed=0)
    rng = np.random.default_rng(0)
    streams = render_streams(sim, loc=5, amp=1.0, tilt_shift=0.0, schedule=sched, rng=rng)
    assert set(streams) == {"target", "aux", "vision", "sun", "nwp"}
    for name, s in streams.items():
        assert s["t"].ndim == 1 and s["v"].shape[0] == s["t"].shape[0]
        assert s["mask"].shape == s["t"].shape
    assert streams["vision"]["v"].shape[1] == 8
    assert streams["aux"]["v"].shape[1] == 2
    # cadence respected: spacing between target events = cadence * dt
    dt = np.diff(streams["target"]["t"])
    assert np.allclose(dt, dt[0])


def test_sun_is_noise_free_and_nwp_is_noisy():
    proc = LatentProcess(n_steps=256, seed=1)
    sim = proc.simulate()
    sched = default_schedule(seed=1)
    rng = np.random.default_rng(1)
    streams = render_streams(sim, loc=5, amp=1.0, tilt_shift=0.0, schedule=sched, rng=rng)
    sun_true = LatentProcess.daylight(streams["sun"]["t"])
    assert np.allclose(streams["sun"]["v"][:, 0], sun_true)   # exact observation
    cloud_true = LatentProcess.local_cloud(sim["field"], 5)
    idx = (streams["nwp"]["t"] / sim["t_minutes"][1]).astype(int)
    assert not np.allclose(streams["nwp"]["v"][:, 0], cloud_true[idx])  # noisy
```

`STATECAST/tests/data/test_asyncbench_dataset.py`:

```python
import torch
from torch.utils.data import DataLoader

from statecast.data.asyncbench_dataset import FUTURE_KNOWN, STREAM_DIMS, AsyncBenchDataset


def test_item_matches_batch_schema_and_collates():
    ds = AsyncBenchDataset(n_entities=4, windows_per_entity=2, hist_steps=96,
                           horizon_steps=24, schedule_seed=0, entity_seed=0)
    assert len(ds) == 8
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    H = 24
    assert batch["y_future"].shape == (4, H)
    assert batch["t_future"].shape[-1] == H
    assert batch["psi_true"].shape == (4, 3)
    assert batch["true_cloud_t0"].shape == (4,)
    for name, d in STREAM_DIMS.items():
        s = batch["streams"][name]
        assert s["v"].shape[0] == 4 and s["v"].shape[2] == d
        assert s["mask"].shape == s["v"].shape[:2]
    # history events end at/before t0 for non-future-known streams' history slice
    t0 = batch["t0"].reshape(-1)[0]
    vis_t = batch["streams"]["vision"]["t"].reshape(-1, batch["streams"]["vision"]["t"].shape[-1])[0]
    assert (vis_t <= t0).all()  # vision is history-only (dense recent window)
    for name in FUTURE_KNOWN:
        t = batch["streams"][name]["t"].reshape(-1, batch["streams"][name]["t"].shape[-1])[0]
        assert (t > t0).any()   # future-known streams extend into the horizon
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_asyncbench_render.py tests/data/test_asyncbench_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError` on both new modules

- [ ] **Step 3: Implement the renderer**

`STATECAST/src/statecast/data/asyncbench_render.py`:

```python
"""Render one LatentProcess simulation into asynchronous, noisy, gappy streams."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .asyncbench_process import LatentProcess


@dataclass
class Schedule:
    cadence: dict = field(default_factory=dict)     # base steps between events
    offset: dict = field(default_factory=dict)      # base-step offset of first event
    p_missing: dict = field(default_factory=dict)   # per-event missingness prob
    noise: dict = field(default_factory=dict)       # additive Gaussian sigma
    vis_window_steps: int = 16                      # vision = dense recent window only


def default_schedule(seed: int = 0) -> Schedule:
    rng = np.random.default_rng(seed)
    names = ("target", "aux", "vision", "sun", "nwp")
    base = {"target": 2, "aux": 4, "vision": 1, "sun": 2, "nwp": 4}
    return Schedule(
        cadence={n: base[n] for n in names},
        offset={n: int(rng.integers(0, base[n])) for n in names},
        p_missing={"target": 0.05, "aux": 0.1, "vision": 0.1, "sun": 0.0, "nwp": 0.0},
        noise={"target": 0.02, "aux": 0.05, "vision": 0.05, "sun": 0.0, "nwp": 0.15},
    )


def render_streams(sim: dict, loc: int, amp: float, tilt_shift: float,
                   schedule: Schedule, rng: np.random.Generator) -> dict:
    field_, t_min = sim["field"], sim["t_minutes"]
    T = t_min.shape[0]
    cloud = LatentProcess.local_cloud(field_, loc)
    sun = LatentProcess.daylight(t_min)
    y = amp * sun * (1.0 - 0.8 * cloud)
    win = (np.arange(-4, 4) + loc) % field_.shape[1]           # vision: 8-px field crop
    raw = {
        "target": y[:, None],
        "aux": np.stack([cloud, sun], axis=1),
        "vision": field_[:, win],
        "sun": sun[:, None],                                    # deterministic, exact
        "nwp": cloud[:, None],                                  # forecastable but noisy
    }
    out = {}
    for name, vals in raw.items():
        idx = np.arange(schedule.offset[name], T, schedule.cadence[name])
        v = vals[idx].astype(np.float32)
        if schedule.noise[name] > 0:
            v = v + rng.normal(0.0, schedule.noise[name], size=v.shape).astype(np.float32)
        mask = (rng.random(len(idx)) >= schedule.p_missing[name]).astype(np.float32)
        out[name] = {"t": t_min[idx].copy(), "v": v, "mask": mask}
    return out
```

- [ ] **Step 4: Implement the dataset**

`STATECAST/src/statecast/data/asyncbench_dataset.py`:

```python
"""Windowed torch Dataset over AsyncBench renders (batch schema of the plan).

One Schedule per dataset instance -> rectangular tensors within any batch.
Vision events are clipped to the last `vis_window_steps` of history (dense
recent window); "sun"/"nwp" extend into the horizon (future-known streams).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .asyncbench_process import LatentProcess
from .asyncbench_render import default_schedule, render_streams

STREAM_DIMS = {"target": 1, "aux": 2, "vision": 8, "sun": 1, "nwp": 1}
FUTURE_KNOWN = ("sun", "nwp")


class AsyncBenchDataset(Dataset):
    def __init__(self, n_entities: int = 8, windows_per_entity: int = 4,
                 hist_steps: int = 96, horizon_steps: int = 24,
                 field_len: int = 32, dt_minutes: float = 15.0,
                 schedule_seed: int = 0, entity_seed: int = 0,
                 cadence_scale: int = 1) -> None:
        self.hist_steps, self.horizon_steps = hist_steps, horizon_steps
        self.dt = dt_minutes
        self.schedule = default_schedule(seed=schedule_seed)
        if cadence_scale != 1:   # stress axis: denser/sparser event schedules
            for name, c in self.schedule.cadence.items():
                self.schedule.cadence[name] = max(1, c // cadence_scale)
        ent_rng = np.random.default_rng(entity_seed)
        n_steps = hist_steps + horizon_steps * (windows_per_entity + 1)
        self.entities = []
        for e in range(n_entities):
            proc = LatentProcess(field_len=field_len, n_steps=n_steps,
                                 dt_minutes=dt_minutes, seed=entity_seed * 1000 + e)
            sim = proc.simulate()
            psi = (int(ent_rng.integers(0, field_len)),
                   float(ent_rng.uniform(0.5, 2.0)),
                   float(ent_rng.uniform(0.0, 360.0)))
            streams = render_streams(sim, *psi, schedule=self.schedule,
                                     rng=np.random.default_rng(entity_seed * 77 + e))
            cloud = LatentProcess.local_cloud(sim["field"], psi[0])
            self.entities.append((sim, psi, streams, cloud))
        self.index = [(e, w) for e in range(n_entities) for w in range(windows_per_entity)]

    def __len__(self) -> int:
        return len(self.index)

    def _slice(self, s: dict, lo: float, hi: float) -> dict:
        keep = (s["t"] > lo) & (s["t"] <= hi)
        return {"t": torch.from_numpy(s["t"][keep]),
                "v": torch.from_numpy(s["v"][keep]),
                "mask": torch.from_numpy(s["mask"][keep])}

    def __getitem__(self, idx: int) -> dict:
        e, w = self.index[idx]
        sim, psi, streams, cloud = self.entities[e]
        t0 = (self.hist_steps + w * self.horizon_steps) * self.dt
        t_end = t0 + self.horizon_steps * self.dt
        out_streams = {}
        for name, s in streams.items():
            hi = t_end if name in FUTURE_KNOWN else t0
            lo = t0 - self.schedule.vis_window_steps * self.dt if name == "vision" else -1.0
            out_streams[name] = self._slice(s, lo, hi)
        t_future = torch.arange(1, self.horizon_steps + 1, dtype=torch.float32) * self.dt + t0
        y_idx = np.searchsorted(sim["t_minutes"], t_future.numpy())
        proc_y = LatentProcess.daylight(t_future.numpy() + psi[2]) * psi[1] \
            * (1.0 - 0.8 * cloud[np.clip(y_idx, 0, len(cloud) - 1)])
        return {
            "streams": out_streams,
            "future_known": FUTURE_KNOWN,
            "t0": torch.tensor([t0], dtype=torch.float32),
            "t_future": t_future,
            "y_future": torch.from_numpy(proc_y.astype(np.float32)),
            "psi_true": torch.tensor(psi, dtype=torch.float32),
            "true_cloud_t0": torch.tensor(
                cloud[min(int(t0 / self.dt), len(cloud) - 1)], dtype=torch.float32),
        }
```

Note: `future_known` is a tuple of str — default collate keeps it as a list of tuples; the model reads `FUTURE_KNOWN` from the module constant instead of the batch when convenient (Task 10 does exactly that).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/data/ -v`
Expected: all PASS. If the collate of `future_known` errors under `default_collate` on your torch version, drop the key from `__getitem__` (the constant is authoritative) and delete the corresponding assert — keep the rest.

- [ ] **Step 6: Commit**

```bash
git add STATECAST/src/statecast/data STATECAST/tests/data
git commit -m "feat(statecast): AsyncBench renderer + windowed Dataset (shared-schedule batching)"
```

---

### Task 4: Shared types + learned initial state

**Files:**
- Create: `STATECAST/src/statecast/types.py`
- Create: `STATECAST/src/statecast/models/__init__.py` (empty)
- Create: `STATECAST/src/statecast/models/state_init.py`
- Test: `STATECAST/tests/models/__init__.py` (empty), `STATECAST/tests/models/test_state_init.py`

**Interfaces:**
- Produces:
  - `statecast.types.Dims` frozen dataclass: `state_tokens=8, state_dim=256, obs_dim=128, psi_dim=96, n_quantiles=9, k_samples=8, dt_feat_dim=16`.
  - `statecast.types.QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)`.
  - `StateInit(dims).forward(batch_size:int) -> (B, state_tokens, state_dim)` learned parameter expanded per batch. Every later model task consumes `Dims`.

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/models/test_state_init.py`:

```python
import torch

from statecast.models.state_init import StateInit
from statecast.types import QUANTILE_LEVELS, Dims


def test_dims_defaults():
    d = Dims()
    assert (d.state_tokens, d.state_dim, d.psi_dim, d.n_quantiles) == (8, 256, 96, 9)
    assert len(QUANTILE_LEVELS) == d.n_quantiles


def test_state_init_shape_and_grad():
    d = Dims(state_tokens=4, state_dim=32)
    init = StateInit(d)
    s = init(batch_size=3)
    assert s.shape == (3, 4, 32)
    s.sum().backward()
    assert init.s0.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_state_init.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`STATECAST/src/statecast/types.py`:

```python
"""Shared shapes and constants for the statecast package."""
from __future__ import annotations

from dataclasses import dataclass

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class Dims:
    state_tokens: int = 8
    state_dim: int = 256
    obs_dim: int = 128
    psi_dim: int = 96
    n_quantiles: int = 9
    k_samples: int = 8
    dt_feat_dim: int = 16
```

`STATECAST/src/statecast/models/state_init.py`:

```python
"""Learned initial latent state, expanded per batch."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims


class StateInit(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.s0 = nn.Parameter(torch.randn(dims.state_tokens, dims.state_dim) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.s0.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_state_init.py -v` — Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/types.py STATECAST/src/statecast/models STATECAST/tests/models
git commit -m "feat(statecast): shared Dims/quantile constants + learned StateInit"
```

---

### Task 5: Observation encoder E_m and predicted-observation readout h_m

**Files:**
- Create: `STATECAST/src/statecast/models/obs_encoder.py`
- Create: `STATECAST/src/statecast/models/obs_readout.py`
- Test: `STATECAST/tests/models/test_obs_encoder.py`, `STATECAST/tests/models/test_obs_readout.py`

**Interfaces:**
- Consumes: `Dims` (Task 4).
- Produces:
  - `ObsEncoder(d_in:int, dims).forward(v: (B, d_in)) -> (B, obs_dim)` — one per stream; for real testbeds `d_in` = the frozen-encoder latent dim (e.g. cached V-JEPA), keeping heavy perception outside the model boundary.
  - `ObsReadout(dims).forward(s: (B, Tk, D)) -> (B, obs_dim)` — `h_m(s)`: "what should this sensor read given my state"; also the JEPA-style latent-prediction head.

- [ ] **Step 1: Write the failing tests**

`STATECAST/tests/models/test_obs_encoder.py`:

```python
import torch

from statecast.models.obs_encoder import ObsEncoder
from statecast.types import Dims


def test_obs_encoder_shape_and_grad():
    d = Dims(obs_dim=16)
    enc = ObsEncoder(d_in=8, dims=d)
    v = torch.randn(5, 8, requires_grad=True)
    o = enc(v)
    assert o.shape == (5, 16)
    o.sum().backward()
    assert v.grad is not None
```

`STATECAST/tests/models/test_obs_readout.py`:

```python
import torch

from statecast.models.obs_readout import ObsReadout
from statecast.types import Dims


def test_obs_readout_shape_and_grad():
    d = Dims(state_tokens=4, state_dim=32, obs_dim=16)
    ro = ObsReadout(d)
    s = torch.randn(5, 4, 32, requires_grad=True)
    o = ro(s)
    assert o.shape == (5, 16)
    o.sum().backward()
    assert s.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_obs_encoder.py tests/models/test_obs_readout.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement both modules**

`STATECAST/src/statecast/models/obs_encoder.py`:

```python
"""Per-stream observation encoder E_m: raw (pre-embedded) values -> obs latent."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims


class ObsEncoder(nn.Module):
    def __init__(self, d_in: int, dims: Dims) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, dims.obs_dim), nn.GELU(),
            nn.Linear(dims.obs_dim, dims.obs_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v)
```

`STATECAST/src/statecast/models/obs_readout.py`:

```python
"""Predicted-observation readout h_m(s): state -> expected sensor reading (latent)."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims


class ObsReadout(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.pool = nn.Linear(dims.state_dim, 1)   # attention-style token pooling
        self.net = nn.Sequential(
            nn.LayerNorm(dims.state_dim),
            nn.Linear(dims.state_dim, dims.obs_dim), nn.GELU(),
            nn.Linear(dims.obs_dim, dims.obs_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.pool(s), dim=1)      # (B, Tk, 1)
        pooled = (w * s).sum(dim=1)                  # (B, D)
        return self.net(pooled)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_obs_encoder.py tests/models/test_obs_readout.py -v` — Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/models/obs_encoder.py STATECAST/src/statecast/models/obs_readout.py STATECAST/tests/models
git commit -m "feat(statecast): per-stream ObsEncoder + ObsReadout (h_m) heads"
```

---

### Task 6: Innovation gate G_m + assimilation update a_m

**Files:**
- Create: `STATECAST/src/statecast/models/innovation_gate.py`
- Create: `STATECAST/src/statecast/models/assimilation_update.py`
- Test: `STATECAST/tests/models/test_innovation_gate.py`, `STATECAST/tests/models/test_assimilation_update.py`

**Interfaces:**
- Consumes: `Dims`.
- Produces:
  - `InnovationGate(dims).forward(innov: (B, obs_dim)) -> (B, 1, state_dim)` in (0,1) — diagonal (per-channel) gate, shared over tokens: this is the scan-compatibility constraint.
  - `AssimilationUpdate(dims).forward(s, o_enc, o_pred, mask) -> (s_new, gate_mean, innov_norm)` with `s: (B,Tk,D)`, `o_enc/o_pred: (B,obs_dim)`, `mask: (B,)`; `mask=0` ⇒ `s_new == s` exactly. Consumed by `StateCast` (Task 10); `gate_mean (B,)` feeds the innovation regularizer (Task 11) and the gate-trace audit.

- [ ] **Step 1: Write the failing tests**

`STATECAST/tests/models/test_innovation_gate.py`:

```python
import torch

from statecast.models.innovation_gate import InnovationGate
from statecast.types import Dims


def test_gate_shape_and_range():
    d = Dims(state_dim=32, obs_dim=16)
    g = InnovationGate(d)
    out = g(torch.randn(5, 16))
    assert out.shape == (5, 1, 32)
    assert (out > 0).all() and (out < 1).all()
```

`STATECAST/tests/models/test_assimilation_update.py`:

```python
import torch

from statecast.models.assimilation_update import AssimilationUpdate
from statecast.types import Dims


def _setup():
    d = Dims(state_tokens=4, state_dim=32, obs_dim=16)
    upd = AssimilationUpdate(d)
    s = torch.randn(5, 4, 32)
    o_enc, o_pred = torch.randn(5, 16), torch.randn(5, 16)
    return upd, s, o_enc, o_pred


def test_update_shapes_and_grad():
    upd, s, o_enc, o_pred = _setup()
    s.requires_grad_(True)
    s_new, gate_mean, innov_norm = upd(s, o_enc, o_pred, mask=torch.ones(5))
    assert s_new.shape == s.shape and gate_mean.shape == (5,) and innov_norm.shape == (5,)
    s_new.sum().backward()
    assert s.grad is not None


def test_masked_event_is_identity():
    upd, s, o_enc, o_pred = _setup()
    s_new, _, _ = upd(s, o_enc, o_pred, mask=torch.zeros(5))
    assert torch.allclose(s_new, s)


def test_zero_innovation_moves_state_little():
    upd, s, o_enc, _ = _setup()
    s_same, _, _ = upd(s, o_enc, o_enc, mask=torch.ones(5))           # innovation = 0
    s_diff, _, _ = upd(s, o_enc, o_enc + 5.0, mask=torch.ones(5))     # big surprise
    assert (s_same - s).norm() < (s_diff - s).norm()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_innovation_gate.py tests/models/test_assimilation_update.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement both modules**

`STATECAST/src/statecast/models/innovation_gate.py`:

```python
"""Learned Kalman-gain analog: innovation magnitude -> per-channel state gate."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims


class InnovationGate(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims.obs_dim, dims.state_dim), nn.GELU(),
            nn.Linear(dims.state_dim, dims.state_dim),
        )

    def forward(self, innov: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(innov)).unsqueeze(1)   # (B, 1, D), diagonal over D
```

`STATECAST/src/statecast/models/assimilation_update.py`:

```python
"""Innovation-gated state update a_m: s+ = s- + mask * G(innov) * Delta(innov)."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims
from .innovation_gate import InnovationGate


class AssimilationUpdate(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.gate = InnovationGate(dims)
        self.delta = nn.Sequential(
            nn.Linear(dims.obs_dim, dims.state_dim), nn.GELU(),
            nn.Linear(dims.state_dim, dims.state_tokens * dims.state_dim),
        )
        self.tokens, self.dim = dims.state_tokens, dims.state_dim
        # near-zero init: an untrained filter should barely move the state
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, s, o_enc, o_pred, mask):
        innov = o_enc - o_pred                                     # (B, obs_dim)
        gate = self.gate(innov)                                    # (B, 1, D)
        delta = self.delta(innov).view(-1, self.tokens, self.dim)  # (B, Tk, D)
        m = mask.view(-1, 1, 1)
        s_new = s + m * gate * delta
        gate_mean = (gate.squeeze(1).mean(dim=-1)) * mask
        innov_norm = innov.norm(dim=-1) * mask
        return s_new, gate_mean, innov_norm
```

Note on `test_zero_innovation_moves_state_little`: with `delta` zero-initialized both updates are zero — the test compares norms with `<`, so seed the layers away from zero inside the test if it flakes: call `torch.nn.init.normal_(upd.delta[-1].weight, std=0.02)` after `_setup()`. Put that line in the test itself, not in the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_innovation_gate.py tests/models/test_assimilation_update.py -v` — Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/models/innovation_gate.py STATECAST/src/statecast/models/assimilation_update.py STATECAST/tests/models
git commit -m "feat(statecast): innovation-gated assimilation update (learned Kalman gain)"
```

---

### Task 7: Δt-conditioned transition f(s, Δt)

**Files:**
- Create: `STATECAST/src/statecast/models/dt_features.py`
- Create: `STATECAST/src/statecast/models/transition.py`
- Test: `STATECAST/tests/models/test_transition.py`

**Interfaces:**
- Consumes: `Dims`.
- Produces:
  - `dt_features(dt: (B,) minutes, dim:int) -> (B, dim)` — Fourier features of `log1p(dt)` (pure function).
  - `Transition(dims).forward(s: (B,Tk,D), dt: (B,)) -> (B,Tk,D)` — residual, FiLM-conditioned on Δt features; `dt=0` ⇒ near-identity. This is the deterministic mean-mode; the flow head (Task 8) conditions on its output.

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/models/test_transition.py`:

```python
import torch

from statecast.models.dt_features import dt_features
from statecast.models.transition import Transition
from statecast.types import Dims


def test_dt_features_shape():
    f = dt_features(torch.tensor([0.0, 15.0, 360.0]), dim=16)
    assert f.shape == (3, 16)
    assert torch.isfinite(f).all()


def test_transition_shape_grad_and_dt_sensitivity():
    d = Dims(state_tokens=4, state_dim=32, dt_feat_dim=8)
    f = Transition(d)
    s = torch.randn(5, 4, 32, requires_grad=True)
    s15 = f(s, torch.full((5,), 15.0))
    s360 = f(s, torch.full((5,), 360.0))
    assert s15.shape == s.shape
    assert not torch.allclose(s15, s360)   # Δt actually conditions the step
    s15.sum().backward()
    assert s.grad is not None


def test_zero_dt_is_near_identity_at_init():
    d = Dims(state_tokens=4, state_dim=32, dt_feat_dim=8)
    f = Transition(d)
    s = torch.randn(5, 4, 32)
    assert (f(s, torch.zeros(5)) - s).norm() < 0.5 * s.norm()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_transition.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`STATECAST/src/statecast/models/dt_features.py`:

```python
"""Fourier features of log-compressed time gaps (minutes)."""
from __future__ import annotations

import math

import torch


def dt_features(dt: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(1000.0), half, device=dt.device))
    x = torch.log1p(dt.clamp(min=0.0)).unsqueeze(-1) * freqs   # (B, half)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
```

`STATECAST/src/statecast/models/transition.py`:

```python
"""Continuous-time transition f(s, dt): the world model. Residual + FiLM on dt."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims
from .dt_features import dt_features


class Transition(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.dims = dims
        self.norm = nn.LayerNorm(dims.state_dim)
        self.mix = nn.MultiheadAttention(dims.state_dim, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dims.state_dim, 2 * dims.state_dim), nn.GELU(),
            nn.Linear(2 * dims.state_dim, dims.state_dim),
        )
        self.film = nn.Linear(dims.dt_feat_dim, 2 * dims.state_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, s: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(dt_features(dt, self.dims.dt_feat_dim)).chunk(2, dim=-1)
        h = self.norm(s)
        h, _ = self.mix(h, h, h, need_weights=False)
        h = self.mlp(h) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        scale = torch.log1p(dt.clamp(min=0.0)).view(-1, 1, 1) / 5.0  # dt=0 => no drift
        return s + scale * h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_transition.py -v` — Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/models/dt_features.py STATECAST/src/statecast/models/transition.py STATECAST/tests/models
git commit -m "feat(statecast): dt-conditioned residual transition (deterministic world model)"
```

---

### Task 8: Conditional flow-matching transition head

**Files:**
- Create: `STATECAST/src/statecast/models/flow_transition.py`
- Test: `STATECAST/tests/models/test_flow_transition.py`

**Interfaces:**
- Consumes: `Dims`, `Transition` (composition: caller passes the deterministic mean `f(s,dt)` as conditioning), `dt_features`.
- Produces: `FlowTransition(dims)` with
  - `velocity(x: (B,Tk,D), tau: (B,), cond: (B,Tk,D), dt: (B,)) -> (B,Tk,D)` — the CFM vector field (trained by Task 11's `cfm_loss`).
  - `sample(cond: (B,Tk,D), dt: (B,), n_steps:int=4, generator=None) -> (B,Tk,D)` — Euler integration from noise, τ 0→1. Ensemble = call K times (Task 10).

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/models/test_flow_transition.py`:

```python
import torch

from statecast.models.flow_transition import FlowTransition
from statecast.types import Dims


def _d():
    return Dims(state_tokens=4, state_dim=32, dt_feat_dim=8)


def test_velocity_shape_and_grad():
    flow = FlowTransition(_d())
    x = torch.randn(5, 4, 32, requires_grad=True)
    v = flow.velocity(x, tau=torch.rand(5), cond=torch.randn(5, 4, 32),
                      dt=torch.full((5,), 30.0))
    assert v.shape == x.shape
    v.sum().backward()
    assert x.grad is not None


def test_sample_shape_and_stochasticity():
    flow = FlowTransition(_d())
    cond, dt = torch.randn(5, 4, 32), torch.full((5,), 30.0)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    s1 = flow.sample(cond, dt, n_steps=4, generator=g1)
    s2 = flow.sample(cond, dt, n_steps=4, generator=g2)
    assert s1.shape == (5, 4, 32)
    assert not torch.allclose(s1, s2)   # different noise draws -> different futures
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_flow_transition.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`STATECAST/src/statecast/models/flow_transition.py`:

```python
"""Conditional flow-matching transition: sample s_{t+dt} ~ p(. | s_t, dt).

The velocity field is conditioned on the deterministic transition output
(mean-mode) and dt; sampling integrates from Gaussian noise, tau 0 -> 1.
"""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims
from .dt_features import dt_features


class FlowTransition(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.dims = dims
        d = dims.state_dim
        self.net = nn.Sequential(
            nn.Linear(2 * d + dims.dt_feat_dim + 1, 2 * d), nn.GELU(),
            nn.Linear(2 * d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def velocity(self, x, tau, cond, dt):
        B, Tk, _ = x.shape
        dtf = dt_features(dt, self.dims.dt_feat_dim)
        feat = torch.cat(
            [x, cond,
             dtf.unsqueeze(1).expand(B, Tk, -1),
             tau.view(B, 1, 1).expand(B, Tk, 1)], dim=-1)
        return self.net(feat)

    @torch.no_grad()
    def sample(self, cond, dt, n_steps: int = 4, generator=None):
        x = torch.randn(cond.shape, generator=generator, device=cond.device)
        step = 1.0 / n_steps
        for i in range(n_steps):
            tau = torch.full((cond.shape[0],), i * step, device=cond.device)
            x = x + step * self.velocity(x, tau, cond, dt)
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_flow_transition.py -v` — Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/models/flow_transition.py STATECAST/tests/models
git commit -m "feat(statecast): conditional flow-matching transition head (generative ensembles)"
```

---

### Task 9: ψ set encoder + quantile readout g_ψ

**Files:**
- Create: `STATECAST/src/statecast/models/psi_encoder.py`
- Create: `STATECAST/src/statecast/models/quantile_readout.py`
- Test: `STATECAST/tests/models/test_psi_encoder.py`, `STATECAST/tests/models/test_quantile_readout.py`

**Interfaces:**
- Consumes: `Dims`, `dt_features`.
- Produces:
  - `PsiSetEncoder(dims).forward(t: (B,E), v: (B,E,1), mask: (B,E), t0: (B,)) -> (B, psi_dim)` — permutation-invariant DeepSet over the entity's target-history events; robust to short histories (masked mean).
  - `QuantileReadout(dims).forward(s: (B,Tk,D), psi: (B,psi_dim)) -> (B, n_quantiles)` — FiLM(ψ)-conditioned pooling; output **monotone non-decreasing** across quantile levels by construction (cumulative softplus).

- [ ] **Step 1: Write the failing tests**

`STATECAST/tests/models/test_psi_encoder.py`:

```python
import torch

from statecast.models.psi_encoder import PsiSetEncoder
from statecast.types import Dims


def test_psi_shape_grad_and_permutation_invariance():
    d = Dims(psi_dim=24, dt_feat_dim=8)
    enc = PsiSetEncoder(d)
    t = torch.linspace(0, 900, 12).unsqueeze(0).repeat(3, 1)
    v = torch.randn(3, 12, 1, requires_grad=True)
    mask = torch.ones(3, 12)
    t0 = torch.full((3,), 1000.0)
    psi = enc(t, v, mask, t0)
    assert psi.shape == (3, 24)
    perm = torch.randperm(12)
    psi_p = enc(t[:, perm], v[:, perm], mask[:, perm], t0)
    assert torch.allclose(psi, psi_p, atol=1e-5)
    psi.sum().backward()
    assert v.grad is not None


def test_psi_handles_empty_history():
    d = Dims(psi_dim=24, dt_feat_dim=8)
    enc = PsiSetEncoder(d)
    psi = enc(torch.zeros(2, 5), torch.zeros(2, 5, 1), torch.zeros(2, 5),
              torch.full((2,), 100.0))
    assert torch.isfinite(psi).all()
```

`STATECAST/tests/models/test_quantile_readout.py`:

```python
import torch

from statecast.models.quantile_readout import QuantileReadout
from statecast.types import Dims


def test_quantiles_shape_monotone_and_psi_dependent():
    d = Dims(state_tokens=4, state_dim=32, psi_dim=24, n_quantiles=9)
    ro = QuantileReadout(d)
    s = torch.randn(5, 4, 32)
    q1 = ro(s, torch.randn(5, 24))
    q2 = ro(s, torch.randn(5, 24))
    assert q1.shape == (5, 9)
    assert (q1[:, 1:] >= q1[:, :-1]).all()          # monotone by construction
    assert not torch.allclose(q1, q2)                # psi actually conditions output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_psi_encoder.py tests/models/test_quantile_readout.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement both modules**

`STATECAST/src/statecast/models/psi_encoder.py`:

```python
"""Amortized entity operator: DeepSet over target-history events -> psi (~10^2 dims)."""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims
from .dt_features import dt_features


class PsiSetEncoder(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.dims = dims
        self.phi = nn.Sequential(
            nn.Linear(1 + dims.dt_feat_dim, dims.psi_dim), nn.GELU(),
            nn.Linear(dims.psi_dim, dims.psi_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(dims.psi_dim, dims.psi_dim), nn.GELU(),
            nn.Linear(dims.psi_dim, dims.psi_dim),
        )

    def forward(self, t, v, mask, t0):
        age = (t0.unsqueeze(1) - t).clamp(min=0.0)              # (B, E)
        feat = torch.cat(
            [v, dt_features(age.reshape(-1), self.dims.dt_feat_dim)
                 .view(*age.shape, -1)], dim=-1)
        h = self.phi(feat) * mask.unsqueeze(-1)                  # (B, E, psi)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.rho(h.sum(dim=1) / denom)
```

`STATECAST/src/statecast/models/quantile_readout.py`:

```python
"""g_psi: state -> monotone target quantiles, FiLM-conditioned on the entity psi."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..types import Dims


class QuantileReadout(nn.Module):
    def __init__(self, dims: Dims) -> None:
        super().__init__()
        self.pool = nn.Linear(dims.state_dim, 1)
        self.film = nn.Linear(dims.psi_dim, 2 * dims.state_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dims.state_dim),
            nn.Linear(dims.state_dim, dims.state_dim), nn.GELU(),
            nn.Linear(dims.state_dim, dims.n_quantiles),
        )

    def forward(self, s: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.pool(s), dim=1)
        pooled = (w * s).sum(dim=1)                              # (B, D)
        gamma, beta = self.film(psi).chunk(2, dim=-1)
        raw = self.head(pooled * (1.0 + gamma) + beta)           # (B, Q)
        base, deltas = raw[:, :1], F.softplus(raw[:, 1:])
        return torch.cat([base, base + deltas.cumsum(dim=-1)], dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_psi_encoder.py tests/models/test_quantile_readout.py -v` — Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/models/psi_encoder.py STATECAST/src/statecast/models/quantile_readout.py STATECAST/tests/models
git commit -m "feat(statecast): amortized psi set encoder + monotone quantile readout"
```

---

### Task 10: StateCast core — the assimilate/roll/decode loop

**Files:**
- Create: `STATECAST/src/statecast/models/statecast_core.py`
- Test: `STATECAST/tests/models/test_statecast_core.py`

**Interfaces:**
- Consumes: everything from Tasks 4–9; `STREAM_DIMS`, `FUTURE_KNOWN` (Task 3).
- Produces: `StateCast(dims, stream_dims: dict[str,int], future_known: tuple[str,...])` with
  `forward(batch, use_flow: bool = False, generator=None) -> dict`:
  - `"quantiles"`: `(B, H, Q)`
  - `"gate_traces"`: `dict[str, list[Tensor(B,)]]` per-event gate means (audit instrument + regularizer input)
  - `"latent_pairs"`: `list[(pred (B,obs_dim), target (B,obs_dim), mask (B,))]` — h_m rolled-state predictions vs encoded actual future observations (JEPA loss input)
  - `"fm_pairs"`: `list[(s_prev (B,Tk,D), dt (B,), s_target (B,Tk,D))]` — transition training pairs, `s_target` detached (CFM loss input)
  - `"psi"`: `(B, psi_dim)`
  - `param_count() -> int` (twin matching, Task 14).
- v0 runs the event loop sequentially (correctness first); the diagonal gate keeps the later associative-scan rewrite possible — that rewrite is out of scope here.

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/models/test_statecast_core.py`:

```python
import torch
from torch.utils.data import DataLoader

from statecast.data.asyncbench_dataset import FUTURE_KNOWN, STREAM_DIMS, AsyncBenchDataset
from statecast.models.statecast_core import StateCast
from statecast.types import Dims


def _tiny():
    dims = Dims(state_tokens=4, state_dim=32, obs_dim=16, psi_dim=24, dt_feat_dim=8)
    model = StateCast(dims, STREAM_DIMS, FUTURE_KNOWN)
    ds = AsyncBenchDataset(n_entities=2, windows_per_entity=1, hist_steps=48,
                           horizon_steps=8, schedule_seed=0, entity_seed=0)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    return model, batch


def test_forward_shapes():
    model, batch = _tiny()
    out = model(batch)
    B, H, Q = 2, 8, 9
    assert out["quantiles"].shape == (B, H, Q)
    assert out["psi"].shape == (B, 24)
    assert set(out["gate_traces"]) <= set(STREAM_DIMS)
    assert len(out["fm_pairs"]) > 0 and len(out["latent_pairs"]) > 0


def test_backward_through_full_loop():
    model, batch = _tiny()
    out = model(batch)
    out["quantiles"].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_flow_mode_runs():
    model, batch = _tiny()
    out = model(batch, use_flow=True, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out["quantiles"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_statecast_core.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the core**

`STATECAST/src/statecast/models/statecast_core.py`:

```python
"""StateCast forward pass: assimilate history in time order, roll the horizon
drip-feeding future-known streams, decode quantiles through g_psi.

Schedules are shared within a batch (see plan Global Constraints), so event
times are read from row 0 of each stream's time tensor.
"""
from __future__ import annotations

import torch
from torch import nn

from ..types import Dims
from .assimilation_update import AssimilationUpdate
from .obs_encoder import ObsEncoder
from .obs_readout import ObsReadout
from .psi_encoder import PsiSetEncoder
from .quantile_readout import QuantileReadout
from .flow_transition import FlowTransition
from .state_init import StateInit
from .transition import Transition


class StateCast(nn.Module):
    def __init__(self, dims: Dims, stream_dims: dict, future_known: tuple) -> None:
        super().__init__()
        self.dims, self.future_known = dims, tuple(future_known)
        self.state_init = StateInit(dims)
        self.transition = Transition(dims)
        self.flow = FlowTransition(dims)
        self.psi_encoder = PsiSetEncoder(dims)
        self.quantile_readout = QuantileReadout(dims)
        self.encoders = nn.ModuleDict({m: ObsEncoder(d, dims) for m, d in stream_dims.items()})
        self.readouts = nn.ModuleDict({m: ObsReadout(dims) for m in stream_dims})
        self.updates = nn.ModuleDict({m: AssimilationUpdate(dims) for m in stream_dims})

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _events(self, streams: dict, lo: float, hi: float, names) -> list:
        ev = []
        for m in names:
            t_row = streams[m]["t"][0] if streams[m]["t"].ndim == 2 else streams[m]["t"]
            for j, t in enumerate(t_row.tolist()):
                if lo < t <= hi:
                    ev.append((t, m, j))
        return sorted(ev)

    def _step(self, s, t_cur, t, use_flow, generator):
        dt = torch.full((s.shape[0],), max(t - t_cur, 0.0), device=s.device)
        mean = self.transition(s, dt)
        if use_flow:
            return self.flow.sample(mean, dt, generator=generator)
        return mean

    def _assimilate(self, s, streams, t, m, j, out):
        o_enc = self.encoders[m](streams[m]["v"][:, j])
        o_pred = self.readouts[m](s)
        s_new, gate_mean, _ = self.updates[m](s, o_enc, o_pred, streams[m]["mask"][:, j])
        out["gate_traces"].setdefault(m, []).append(gate_mean)
        return s_new

    def forward(self, batch, use_flow: bool = False, generator=None):
        streams = batch["streams"]
        t0 = float(batch["t0"].reshape(-1)[0])
        t_future = batch["t_future"][0] if batch["t_future"].ndim == 2 else batch["t_future"]
        B = batch["y_future"].shape[0]
        out = {"gate_traces": {}, "latent_pairs": [], "fm_pairs": []}

        tg = streams["target"]
        out["psi"] = self.psi_encoder(
            tg["t"].expand(B, -1) if tg["t"].ndim == 1 else tg["t"],
            tg["v"], tg["mask"], batch["t0"].reshape(-1).expand(B))

        s, t_cur = self.state_init(B), None
        for t, m, j in self._events(streams, -1.0, t0, streams.keys()):
            if t_cur is not None and t > t_cur:
                s_prev = s
                s = self._step(s, t_cur, t, False, None)
                out["fm_pairs"].append(
                    (s_prev, torch.full((B,), t - t_cur, device=s.device), s.detach()))
            s = self._assimilate(s, streams, t, m, j, out)
            t_cur = t if t_cur is None else max(t_cur, t)
        t_cur = t0 if t_cur is None else t_cur

        future_ev = self._events(streams, t0, float(t_future[-1]), self.future_known)
        quantiles = []
        for h, t_h in enumerate(t_future.tolist()):
            while future_ev and future_ev[0][0] <= t_h:
                t, m, j = future_ev.pop(0)
                s = self._step(s, t_cur, t, use_flow, generator)
                pred, target = self.readouts[m](s), self.encoders[m](streams[m]["v"][:, j])
                out["latent_pairs"].append((pred, target.detach(), streams[m]["mask"][:, j]))
                s = self._assimilate(s, streams, t, m, j, out)
                t_cur = t
            s = self._step(s, t_cur, t_h, use_flow, generator)
            t_cur = t_h
            quantiles.append(self.quantile_readout(s, out["psi"]))
        out["quantiles"] = torch.stack(quantiles, dim=1)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_statecast_core.py -v` — Expected: 3 PASSED
(`test_flow_mode_runs` exercises `flow.sample` under `no_grad` — that is fine; FM training gradients flow through `velocity` in the loss, not through sampling.)

- [ ] **Step 5: Run the whole suite + commit**

```bash
uv run pytest
git add STATECAST/src/statecast/models/statecast_core.py STATECAST/tests/models
git commit -m "feat(statecast): full assimilate/roll/decode event loop (StateCast core)"
```

---

### Task 11: Losses

**Files:**
- Create: `STATECAST/src/statecast/losses/__init__.py` (empty)
- Create: `STATECAST/src/statecast/losses/pinball_loss.py`
- Create: `STATECAST/src/statecast/losses/latent_prediction_loss.py`
- Create: `STATECAST/src/statecast/losses/flow_matching_loss.py`
- Create: `STATECAST/src/statecast/losses/innovation_regularizer.py`
- Test: `STATECAST/tests/losses/__init__.py` (empty), `STATECAST/tests/losses/test_losses.py`

**Interfaces:**
- Consumes: `QUANTILE_LEVELS`; `FlowTransition.velocity`; core outputs (`latent_pairs`, `fm_pairs`, `gate_traces`).
- Produces (all pure functions returning scalar tensors):
  - `ramp_weighted_pinball(q_pred (B,H,Q), y (B,H), mask (B,H)|None, ramp_lambda=2.0) -> ()` — weight `1 + λ·|Δy|/mean|Δy|` upweights sharp transitions (S6 alignment).
  - `latent_prediction_loss(pairs) -> ()` — masked MSE over `(pred, target, mask)` triples.
  - `cfm_loss(flow, pairs, rng=None) -> ()` — conditional flow matching on `(s_prev, dt, s_target)`: `x_τ = (1−τ)ε + τ·s_target`, regress `velocity(x_τ, τ, cond=transition-mean-free s_prev, dt)` onto `s_target − ε`. Conditioning uses `s_prev` directly (the flow net learns the drift; the deterministic mean stays a separate mode).
  - `gate_band_penalty(gate_traces, low=0.05, high=0.95) -> ()` — hinge on per-stream mean gate: punish always-ignore and always-parrot.

- [ ] **Step 1: Write the failing tests**

`STATECAST/tests/losses/test_losses.py`:

```python
import torch

from statecast.losses.flow_matching_loss import cfm_loss
from statecast.losses.innovation_regularizer import gate_band_penalty
from statecast.losses.latent_prediction_loss import latent_prediction_loss
from statecast.losses.pinball_loss import ramp_weighted_pinball
from statecast.models.flow_transition import FlowTransition
from statecast.types import Dims


def test_pinball_scalar_and_ramp_weighting():
    q = torch.randn(4, 6, 9).sort(dim=-1).values
    y_flat = torch.zeros(4, 6)
    y_ramp = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expand(4, 6)
    l_flat = ramp_weighted_pinball(q, y_flat, None)
    l_ramp = ramp_weighted_pinball(q, y_ramp, None)
    assert l_flat.ndim == 0 and torch.isfinite(l_flat)
    assert l_ramp != l_flat


def test_latent_prediction_masked():
    pairs = [(torch.randn(3, 8), torch.randn(3, 8), torch.tensor([1.0, 0.0, 1.0]))]
    loss = latent_prediction_loss(pairs)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert latent_prediction_loss([]) == 0.0


def test_cfm_loss_backprops_into_flow():
    d = Dims(state_tokens=2, state_dim=16, dt_feat_dim=8)
    flow = FlowTransition(d)
    pairs = [(torch.randn(3, 2, 16), torch.full((3,), 15.0), torch.randn(3, 2, 16))]
    loss = cfm_loss(flow, pairs)
    loss.backward()
    assert any(p.grad is not None for p in flow.parameters())


def test_gate_band_penalty_direction():
    mid = {"a": [torch.full((4,), 0.5)]}
    stuck = {"a": [torch.full((4,), 0.001)]}
    assert gate_band_penalty(stuck) > gate_band_penalty(mid)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/losses/test_losses.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the four loss files**

`STATECAST/src/statecast/losses/pinball_loss.py`:

```python
"""Ramp-weighted pinball loss on target quantiles."""
from __future__ import annotations

import torch

from ..types import QUANTILE_LEVELS


def ramp_weighted_pinball(q_pred, y, mask=None, ramp_lambda: float = 2.0):
    levels = torch.tensor(QUANTILE_LEVELS, device=q_pred.device).view(1, 1, -1)
    err = y.unsqueeze(-1) - q_pred                              # (B, H, Q)
    pin = torch.maximum(levels * err, (levels - 1.0) * err)
    dy = torch.zeros_like(y)
    dy[:, 1:] = (y[:, 1:] - y[:, :-1]).abs()
    w = 1.0 + ramp_lambda * dy / dy.mean().clamp(min=1e-8)      # (B, H)
    if mask is not None:
        w = w * mask
    return (pin * w.unsqueeze(-1)).sum() / w.sum().clamp(min=1e-8) / len(QUANTILE_LEVELS)
```

`STATECAST/src/statecast/losses/latent_prediction_loss.py`:

```python
"""JEPA-style latent prediction: h_m(rolled state) vs encoded real future obs."""
from __future__ import annotations

import torch


def latent_prediction_loss(pairs):
    if not pairs:
        return torch.tensor(0.0)
    num, den = 0.0, 0.0
    for pred, target, mask in pairs:
        se = ((pred - target) ** 2).mean(dim=-1)    # (B,)
        num = num + (se * mask).sum()
        den = den + mask.sum()
    return num / torch.clamp(torch.as_tensor(den, dtype=torch.float32), min=1.0)
```

`STATECAST/src/statecast/losses/flow_matching_loss.py`:

```python
"""Conditional flow matching on transition pairs (s_prev, dt, s_target)."""
from __future__ import annotations

import torch


def cfm_loss(flow, pairs, rng=None):
    if not pairs:
        return torch.tensor(0.0)
    total = 0.0
    for s_prev, dt, s_target in pairs:
        eps = torch.randn(s_target.shape, generator=rng, device=s_target.device)
        tau = torch.rand(s_target.shape[0], generator=rng, device=s_target.device)
        x_tau = (1.0 - tau.view(-1, 1, 1)) * eps + tau.view(-1, 1, 1) * s_target
        v_target = s_target - eps
        v_pred = flow.velocity(x_tau, tau, cond=s_prev.detach(), dt=dt)
        total = total + ((v_pred - v_target) ** 2).mean()
    return total / len(pairs)
```

`STATECAST/src/statecast/losses/innovation_regularizer.py`:

```python
"""Keep the filter honest: mean gate per stream must stay inside (low, high)."""
from __future__ import annotations

import torch


def gate_band_penalty(gate_traces, low: float = 0.05, high: float = 0.95):
    if not gate_traces:
        return torch.tensor(0.0)
    total = 0.0
    for traces in gate_traces.values():
        g = torch.stack(traces).mean()
        total = total + torch.relu(low - g) + torch.relu(g - high)
    return total / len(gate_traces)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/losses/test_losses.py -v` — Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/losses STATECAST/tests/losses
git commit -m "feat(statecast): pinball/latent-prediction/CFM losses + innovation regularizer"
```

---

### Task 12: Stage-0 Lightning module, Hydra configs, train entrypoint

**Files:**
- Create: `STATECAST/src/statecast/lightning_stage0.py`
- Create: `STATECAST/src/statecast/train.py`
- Create: `STATECAST/configs/config.yaml`, `STATECAST/configs/model/statecast.yaml`, `STATECAST/configs/data/asyncbench.yaml`, `STATECAST/configs/trainer/default.yaml`
- Test: `STATECAST/tests/test_training_loop.py`

**Interfaces:**
- Consumes: `StateCast`, all losses, `AsyncBenchDataset`.
- Produces:
  - `StateCastStage0(model_cfg: dict, loss_weights: dict, lr: float)` LightningModule; `training_step` returns total loss `pinball + w_latent*latent + w_fm*cfm + w_innov*gate_band`; logs each component.
  - `uv run python -m statecast.train` trains a smoke config end-to-end. Task 14's twin reuses this module via `model_cfg["arch"] = "twin"`.

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/test_training_loop.py`:

```python
import lightning as L
import torch
from torch.utils.data import DataLoader

from statecast.data.asyncbench_dataset import AsyncBenchDataset
from statecast.lightning_stage0 import StateCastStage0


def _module():
    return StateCastStage0(
        model_cfg={"arch": "statecast", "state_tokens": 4, "state_dim": 32,
                   "obs_dim": 16, "psi_dim": 24, "dt_feat_dim": 8},
        loss_weights={"latent": 1.0, "fm": 0.1, "innov": 0.1},
        lr=1e-3,
    )


def test_two_optimizer_steps_reduce_or_keep_finite_loss():
    torch.manual_seed(0)
    module = _module()
    ds = AsyncBenchDataset(n_entities=2, windows_per_entity=2, hist_steps=48,
                           horizon_steps=8, schedule_seed=0, entity_seed=0)
    trainer = L.Trainer(max_epochs=1, limit_train_batches=2, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        accelerator="cpu")
    trainer.fit(module, DataLoader(ds, batch_size=2))
    assert torch.isfinite(trainer.callback_metrics["train/loss"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training_loop.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the Lightning module**

`STATECAST/src/statecast/lightning_stage0.py`:

```python
"""Stage 0: train StateCast (or the attention twin) on AsyncBench synthetic."""
from __future__ import annotations

import lightning as L
import torch

from .data.asyncbench_dataset import FUTURE_KNOWN, STREAM_DIMS
from .losses.flow_matching_loss import cfm_loss
from .losses.innovation_regularizer import gate_band_penalty
from .losses.latent_prediction_loss import latent_prediction_loss
from .losses.pinball_loss import ramp_weighted_pinball
from .types import Dims


def build_model(model_cfg: dict):
    arch = model_cfg.get("arch", "statecast")
    dims = Dims(**{k: v for k, v in model_cfg.items()
                   if k in Dims.__dataclass_fields__})
    if arch == "statecast":
        from .models.statecast_core import StateCast
        return StateCast(dims, STREAM_DIMS, FUTURE_KNOWN)
    if arch == "twin":
        from .models.attention_twin import AttentionTwin
        return AttentionTwin(dims, STREAM_DIMS)
    raise ValueError(f"unknown arch: {arch}")


class StateCastStage0(L.LightningModule):
    def __init__(self, model_cfg: dict, loss_weights: dict, lr: float = 3e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(dict(model_cfg))
        self.w = dict(loss_weights)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        mask = torch.ones_like(batch["y_future"])
        pin = ramp_weighted_pinball(out["quantiles"], batch["y_future"], mask)
        # aux losses may return fresh CPU scalars (empty inputs) — align devices
        lat = latent_prediction_loss(out.get("latent_pairs", [])).to(pin.device)
        fm = (cfm_loss(self.model.flow, out["fm_pairs"])
              if hasattr(self.model, "flow") else torch.tensor(0.0)).to(pin.device)
        innov = gate_band_penalty(out.get("gate_traces", {})).to(pin.device)
        loss = (pin + self.w.get("latent", 0.0) * lat
                + self.w.get("fm", 0.0) * fm + self.w.get("innov", 0.0) * innov)
        self.log_dict({"train/loss": loss, "train/pinball": pin, "train/latent": lat,
                       "train/fm": fm, "train/innov": innov}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training_loop.py -v` — Expected: PASSED
(The twin branch of `build_model` imports lazily, so it does not break before Task 14.)

- [ ] **Step 5: Write configs + hydra entrypoint**

`STATECAST/configs/config.yaml`:

```yaml
defaults:
  - trainer: default
  - model: statecast
  - data: asyncbench
  - _self_

seed: 42
project_name: statecast
run_name: ${now:%Y-%m-%d_%H-%M-%S}_run

hydra:
  run:
    dir: logs/experiments/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}
```

`STATECAST/configs/model/statecast.yaml`:

```yaml
arch: statecast
state_tokens: 8
state_dim: 256
obs_dim: 128
psi_dim: 96
n_quantiles: 9
k_samples: 8
dt_feat_dim: 16
loss_weights:
  latent: 1.0
  fm: 0.1
  innov: 0.1
lr: 3.0e-4
```

`STATECAST/configs/data/asyncbench.yaml`:

```yaml
n_entities: 16
windows_per_entity: 8
hist_steps: 96
horizon_steps: 24
batch_size: 8
num_workers: 0
```

`STATECAST/configs/trainer/default.yaml`:

```yaml
max_epochs: 3
accelerator: auto
devices: 1
log_every_n_steps: 5
reload_dataloaders_every_n_epochs: 1   # new AsyncBench schedule each epoch
```

`STATECAST/src/statecast/train.py`:

```python
"""Hydra entrypoint: uv run python -m statecast.train [overrides]."""
from __future__ import annotations

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .data.asyncbench_dataset import AsyncBenchDataset
from .lightning_stage0 import StateCastStage0


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    module = StateCastStage0(model_cfg=model_cfg,
                             loss_weights=model_cfg.pop("loss_weights"),
                             lr=model_cfg.pop("lr"))
    epoch = {"n": -1}

    def loader():
        epoch["n"] += 1
        ds = AsyncBenchDataset(
            n_entities=cfg.data.n_entities,
            windows_per_entity=cfg.data.windows_per_entity,
            hist_steps=cfg.data.hist_steps, horizon_steps=cfg.data.horizon_steps,
            schedule_seed=cfg.seed + epoch["n"], entity_seed=cfg.seed)
        return DataLoader(ds, batch_size=cfg.data.batch_size,
                          num_workers=cfg.data.num_workers)

    trainer = L.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer.fit(module, train_dataloaders=loader)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run training**

```bash
uv run python -m statecast.train trainer.max_epochs=1 data.n_entities=4 data.windows_per_entity=2
```
Expected: 1 epoch completes, `train/loss` finite in the console log. If `trainer.fit(..., train_dataloaders=loader)` rejects the callable on your Lightning version, wrap it in a `LightningDataModule` with `train_dataloader = loader` — keep the per-epoch schedule reseed.

- [ ] **Step 7: Run full suite + commit**

```bash
uv run pytest
git add STATECAST/src/statecast/lightning_stage0.py STATECAST/src/statecast/train.py STATECAST/configs STATECAST/tests/test_training_loop.py
git commit -m "feat(statecast): Stage-0 Lightning module + Hydra configs + train entrypoint"
```

---

### Task 13: State-recovery probe (Stage-0 audit instrument)

**Files:**
- Create: `STATECAST/src/statecast/eval/__init__.py` (empty)
- Create: `STATECAST/src/statecast/eval/state_probe.py`
- Create: `STATECAST/scripts/probe_state.py`
- Test: `STATECAST/tests/eval/__init__.py` (empty), `STATECAST/tests/eval/test_state_probe.py`

**Interfaces:**
- Consumes: `StateCast` (needs the state at t0 → add nothing to the core: the probe re-runs `forward` and reads the state via a small hook exposed here), `AsyncBenchDataset` (`true_cloud_t0`).
- Produces:
  - `collect_states(model, loader) -> (X (N, Tk*D), y (N,))` — pooled/flattened state at forecast origin vs ground-truth cloud. Implemented by monkeypatch-free wrapping: run the history walk with `t_future` truncated to 1 step and grab the state just before decode via `model.quantile_readout` pre-hook.
  - `ridge_r2(X_train, y_train, X_test, y_test, l2=1.0) -> float` — closed-form ridge, returns test R².
  - `scripts/probe_state.py --ckpt <path>`: loads a Stage-0 checkpoint, prints R². **Pass criterion (post-training, run manually): R² ≥ 0.5** — the degenerate-state early alarm from STATECAST.md §I.2.

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/eval/test_state_probe.py`:

```python
import torch
from torch.utils.data import DataLoader

from statecast.data.asyncbench_dataset import FUTURE_KNOWN, STREAM_DIMS, AsyncBenchDataset
from statecast.eval.state_probe import collect_states, ridge_r2
from statecast.models.statecast_core import StateCast
from statecast.types import Dims


def test_collect_states_shapes():
    dims = Dims(state_tokens=4, state_dim=32, obs_dim=16, psi_dim=24, dt_feat_dim=8)
    model = StateCast(dims, STREAM_DIMS, FUTURE_KNOWN)
    ds = AsyncBenchDataset(n_entities=2, windows_per_entity=2, hist_steps=48,
                           horizon_steps=8, schedule_seed=0, entity_seed=0)
    X, y = collect_states(model, DataLoader(ds, batch_size=2))
    assert X.shape == (4, 4 * 32) and y.shape == (4,)


def test_ridge_r2_recovers_linear_signal():
    torch.manual_seed(0)
    X = torch.randn(200, 16)
    w = torch.randn(16)
    y = X @ w + 0.01 * torch.randn(200)
    r2 = ridge_r2(X[:150], y[:150], X[150:], y[150:], l2=1e-3)
    assert r2 > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_state_probe.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement probe + script**

`STATECAST/src/statecast/eval/state_probe.py`:

```python
"""Linear probes on the latent state (Stage-0 audit: is cloud cover readable?)."""
from __future__ import annotations

import torch


@torch.no_grad()
def collect_states(model, loader):
    xs, ys = [], []
    captured = {}

    def hook(_mod, args):
        captured["s"] = args[0]          # (B, Tk, D) just before quantile decode

    handle = model.quantile_readout.register_forward_pre_hook(hook)
    try:
        for batch in loader:
            batch = dict(batch)
            batch["t_future"] = batch["t_future"][..., :1]   # decode once, at ~t0
            model(batch)
            xs.append(captured["s"].flatten(1).cpu())
            ys.append(batch["true_cloud_t0"].reshape(-1).cpu())
    finally:
        handle.remove()
    return torch.cat(xs), torch.cat(ys)


def ridge_r2(X_train, y_train, X_test, y_test, l2: float = 1.0) -> float:
    d = X_train.shape[1]
    A = X_train.T @ X_train + l2 * torch.eye(d)
    w = torch.linalg.solve(A, X_train.T @ y_train)
    pred = X_test @ w
    ss_res = ((y_test - pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum().clamp(min=1e-8)
    return float(1.0 - ss_res / ss_tot)
```

`STATECAST/scripts/probe_state.py`:

```python
"""Report cloud-cover probe R^2 for a Stage-0 checkpoint.

Usage: uv run python scripts/probe_state.py <ckpt_path>
Pass criterion (trained model): R^2 >= 0.5.
"""
from __future__ import annotations

import sys

import torch
from torch.utils.data import DataLoader

from statecast.data.asyncbench_dataset import AsyncBenchDataset
from statecast.eval.state_probe import collect_states, ridge_r2
from statecast.lightning_stage0 import StateCastStage0


def main(ckpt_path: str) -> None:
    module = StateCastStage0.load_from_checkpoint(ckpt_path, map_location="cpu")
    mk = lambda seed: AsyncBenchDataset(n_entities=8, windows_per_entity=8,
                                        hist_steps=96, horizon_steps=24,
                                        schedule_seed=seed, entity_seed=seed)
    Xtr, ytr = collect_states(module.model, DataLoader(mk(0), batch_size=8))
    Xte, yte = collect_states(module.model, DataLoader(mk(1), batch_size=8))
    r2 = ridge_r2(Xtr, ytr, Xte, yte)
    print(f"cloud-cover probe R^2 = {r2:.3f}  (pass >= 0.5)")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_state_probe.py -v` — Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/eval STATECAST/scripts/probe_state.py STATECAST/tests/eval
git commit -m "feat(statecast): state-recovery linear probe (Stage-0 audit instrument)"
```

---

### Task 14: Attention twin (matched-parameter fusion baseline)

**Files:**
- Create: `STATECAST/src/statecast/models/attention_twin.py`
- Test: `STATECAST/tests/models/test_attention_twin.py`

**Interfaces:**
- Consumes: `Dims`, `dt_features`, `QuantileReadout`-free (the twin has **no ψ** — entity identity stays entangled, per STATECAST.md §3); batch schema.
- Produces: `AttentionTwin(dims, stream_dims)` with `forward(batch) -> {"quantiles": (B,H,Q), "latent_pairs": [], "fm_pairs": [], "gate_traces": {}}` — same output contract as `StateCast` so `StateCastStage0` and the stress harness run it unchanged (auxiliary losses are empty ⇒ pinball-only, which is the matched-loss condition on the target term; document this asymmetry in the paper, not here), and `param_count() -> int`. **Matched-parameter test:** within ±15% of a `StateCast` built with the same `Dims`.

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/models/test_attention_twin.py`:

```python
import torch
from torch.utils.data import DataLoader

from statecast.data.asyncbench_dataset import FUTURE_KNOWN, STREAM_DIMS, AsyncBenchDataset
from statecast.models.attention_twin import AttentionTwin
from statecast.models.statecast_core import StateCast
from statecast.types import Dims


def _dims():
    return Dims(state_tokens=4, state_dim=32, obs_dim=16, psi_dim=24, dt_feat_dim=8)


def test_twin_forward_contract():
    twin = AttentionTwin(_dims(), STREAM_DIMS)
    ds = AsyncBenchDataset(n_entities=2, windows_per_entity=1, hist_steps=48,
                           horizon_steps=8, schedule_seed=0, entity_seed=0)
    batch = next(iter(DataLoader(ds, batch_size=2)))
    out = twin(batch)
    assert out["quantiles"].shape == (2, 8, 9)
    assert (out["quantiles"][:, :, 1:] >= out["quantiles"][:, :, :-1]).all()
    out["quantiles"].sum().backward()
    assert any(p.grad is not None for p in twin.parameters())


def test_parameter_matching():
    dims = _dims()
    sc = StateCast(dims, STREAM_DIMS, FUTURE_KNOWN).param_count()
    tw = AttentionTwin(dims, STREAM_DIMS).param_count()
    assert abs(tw - sc) / sc < 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_attention_twin.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the twin**

`STATECAST/src/statecast/models/attention_twin.py`:

```python
"""Matched-parameter attention-fusion twin: every event is a token; a
transformer encoder fuses them; horizon queries cross-attend to decode
quantiles. No latent state, no psi, no assimilation — the strongest member
of the crowded fusion class (STATECAST.md section 3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..types import Dims
from .dt_features import dt_features


class AttentionTwin(nn.Module):
    def __init__(self, dims: Dims, stream_dims: dict, n_layers: int = 4) -> None:
        super().__init__()
        self.dims = dims
        d = dims.state_dim
        self.value_proj = nn.ModuleDict(
            {m: nn.Linear(di, d) for m, di in stream_dims.items()})
        self.stream_emb = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(d) * 0.02) for m in stream_dims})
        self.time_proj = nn.Linear(dims.dt_feat_dim, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query_time = nn.Linear(dims.dt_feat_dim, d)
        self.cross = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, dims.n_quantiles))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _tokens(self, batch):
        streams, toks, masks = batch["streams"], [], []
        t0 = batch["t0"].reshape(-1)[0]
        for m, s in streams.items():
            t = s["t"][0] if s["t"].ndim == 2 else s["t"]        # shared schedule
            age = (t0 - t).abs()
            tok = (self.value_proj[m](s["v"])
                   + self.time_proj(dt_features(age, self.dims.dt_feat_dim)).unsqueeze(0)
                   + self.stream_emb[m])
            toks.append(tok)
            masks.append(s["mask"])
        return torch.cat(toks, dim=1), torch.cat(masks, dim=1)

    def forward(self, batch):
        tokens, mask = self._tokens(batch)
        enc = self.encoder(tokens, src_key_padding_mask=(mask == 0))
        t_future = batch["t_future"][0] if batch["t_future"].ndim == 2 else batch["t_future"]
        lead = (t_future - batch["t0"].reshape(-1)[0]).clamp(min=0.0)
        q = self.query_time(dt_features(lead, self.dims.dt_feat_dim))
        q = q.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        dec, _ = self.cross(q, enc, enc, key_padding_mask=(mask == 0),
                            need_weights=False)
        raw = self.head(dec)                                     # (B, H, Q)
        base, deltas = raw[..., :1], F.softplus(raw[..., 1:])
        quantiles = torch.cat([base, base + deltas.cumsum(dim=-1)], dim=-1)
        return {"quantiles": quantiles, "latent_pairs": [], "fm_pairs": [],
                "gate_traces": {}}
```

- [ ] **Step 4: Run tests; tune n_layers if parameter matching fails**

Run: `uv run pytest tests/models/test_attention_twin.py -v`
Expected: 2 PASSED. If `test_parameter_matching` fails, adjust `n_layers` (or `dim_feedforward`) until within ±15% — record the final value in `configs/model/twin.yaml` below.

- [ ] **Step 5: Add twin config**

`STATECAST/configs/model/twin.yaml`:

```yaml
arch: twin
state_tokens: 8
state_dim: 256
obs_dim: 128
psi_dim: 96
n_quantiles: 9
k_samples: 8
dt_feat_dim: 16
loss_weights: {latent: 0.0, fm: 0.0, innov: 0.0}
lr: 3.0e-4
```

Smoke: `uv run python -m statecast.train model=twin trainer.max_epochs=1 data.n_entities=4 data.windows_per_entity=2` — completes with finite loss.

- [ ] **Step 6: Commit**

```bash
git add STATECAST/src/statecast/models/attention_twin.py STATECAST/tests/models/test_attention_twin.py STATECAST/configs/model/twin.yaml
git commit -m "feat(statecast): matched-parameter attention-fusion twin (G4 counterpart)"
```

---

### Task 15: G4 stress harness (cadence shift / dropout / horizon extrapolation)

**Files:**
- Create: `STATECAST/src/statecast/eval/stress_eval.py`
- Create: `STATECAST/scripts/g4_stress.py`
- Test: `STATECAST/tests/eval/test_stress_eval.py`

**Interfaces:**
- Consumes: any model with the Task-10/14 forward contract; `AsyncBenchDataset`, `default_schedule`.
- Produces:
  - `pinball_score(model, loader) -> float` (mean unweighted pinball — comparable across models) and `coverage_80(model, loader) -> float` (empirical coverage of the [q0.1, q0.9] band; calibration instrument).
  - `run_conditions(model, base_cfg: dict) -> dict[str, dict[str, float]]` over conditions: `"base"`, `"cadence_shift"` (all cadences halved ⇒ 2× denser events), `"drop_vision"` (vision mask forced to 0), `"horizon_x2"` (horizon_steps doubled — train-short/test-long extrapolation).
  - `scripts/g4_stress.py --statecast <ckpt> --twin <ckpt>` writes `STATECAST/results/g4_stress.json`: `{model: {condition: {"pinball": float, "coverage80": float}}}`. **G4 criterion (manual, post-training): StateCast within parity on `base` AND better on ≥ 1 stress axis, with `coverage80` closer to 0.8 under `drop_vision`.**

- [ ] **Step 1: Write the failing test**

`STATECAST/tests/eval/test_stress_eval.py`:

```python
from statecast.data.asyncbench_dataset import FUTURE_KNOWN, STREAM_DIMS
from statecast.eval.stress_eval import run_conditions
from statecast.models.statecast_core import StateCast
from statecast.types import Dims


def test_run_conditions_produces_all_cells():
    dims = Dims(state_tokens=4, state_dim=32, obs_dim=16, psi_dim=24, dt_feat_dim=8)
    model = StateCast(dims, STREAM_DIMS, FUTURE_KNOWN)
    base_cfg = {"n_entities": 2, "windows_per_entity": 1, "hist_steps": 48,
                "horizon_steps": 8, "schedule_seed": 3, "entity_seed": 3,
                "batch_size": 2}
    table = run_conditions(model, base_cfg)
    assert set(table) == {"base", "cadence_shift", "drop_vision", "horizon_x2"}
    for cell in table.values():
        assert set(cell) == {"pinball", "coverage80"}
        assert all(isinstance(v, float) for v in cell.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_stress_eval.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the harness**

`STATECAST/src/statecast/eval/stress_eval.py`:

```python
"""G4 stress harness: score a model across formulation-separating conditions."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..data.asyncbench_dataset import AsyncBenchDataset
from ..losses.pinball_loss import ramp_weighted_pinball


def _make_loader(cfg: dict, cadence_scale: int = 1, horizon_mult: int = 1,
                 drop_stream: str | None = None) -> DataLoader:
    ds = AsyncBenchDataset(
        n_entities=cfg["n_entities"], windows_per_entity=cfg["windows_per_entity"],
        hist_steps=cfg["hist_steps"],
        horizon_steps=cfg["horizon_steps"] * horizon_mult,
        schedule_seed=cfg["schedule_seed"], entity_seed=cfg["entity_seed"],
        cadence_scale=cadence_scale)
    loader = DataLoader(ds, batch_size=cfg["batch_size"])
    loader.drop_stream = drop_stream
    return loader


@torch.no_grad()
def _score(model, loader) -> dict:
    pins, hits, n = [], 0.0, 0.0
    for batch in loader:
        if getattr(loader, "drop_stream", None):
            batch["streams"][loader.drop_stream]["mask"] = torch.zeros_like(
                batch["streams"][loader.drop_stream]["mask"])
        out = model(batch)
        q, y = out["quantiles"], batch["y_future"]
        pins.append(float(ramp_weighted_pinball(q, y, None, ramp_lambda=0.0)))
        inside = (y >= q[..., 0]) & (y <= q[..., -1])
        hits += float(inside.float().sum())
        n += float(inside.numel())
    return {"pinball": sum(pins) / len(pins), "coverage80": hits / max(n, 1.0)}


def run_conditions(model, base_cfg: dict) -> dict:
    model.eval()
    return {
        "base": _score(model, _make_loader(base_cfg)),
        "cadence_shift": _score(model, _make_loader(base_cfg, cadence_scale=2)),
        "drop_vision": _score(model, _make_loader(base_cfg, drop_stream="vision")),
        "horizon_x2": _score(model, _make_loader(base_cfg, horizon_mult=2)),
    }
```

(`cadence_scale` is the `AsyncBenchDataset` constructor arg added in Task 3 — it halves the base-step cadences before entities are rendered, so the shifted schedule is real, not cosmetic.)

`STATECAST/scripts/g4_stress.py`:

```python
"""G4: score StateCast vs the attention twin across stress conditions.

Usage: uv run python scripts/g4_stress.py <statecast.ckpt> <twin.ckpt>
Writes results/g4_stress.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from statecast.eval.stress_eval import run_conditions
from statecast.lightning_stage0 import StateCastStage0

BASE_CFG = {"n_entities": 16, "windows_per_entity": 8, "hist_steps": 96,
            "horizon_steps": 24, "schedule_seed": 99, "entity_seed": 99,
            "batch_size": 8}


def main(statecast_ckpt: str, twin_ckpt: str) -> None:
    table = {}
    for name, ckpt in (("statecast", statecast_ckpt), ("twin", twin_ckpt)):
        module = StateCastStage0.load_from_checkpoint(ckpt, map_location="cpu")
        table[name] = run_conditions(module.model, BASE_CFG)
    out = Path(__file__).resolve().parents[1] / "results" / "g4_stress.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_stress_eval.py -v` — Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add STATECAST/src/statecast/eval/stress_eval.py STATECAST/scripts/g4_stress.py STATECAST/tests/eval
git commit -m "feat(statecast): G4 stress harness (cadence shift, sensor dropout, horizon extrapolation)"
```

---

### Task 16: uk_pv protocol adapter (Stage-2 entry point)

**Files:**
- Create: `STATECAST/src/statecast/data/pv_events.py`
- Test: `STATECAST/tests/data/test_pv_events.py`

**Interfaces:**
- Consumes: `mmtsfm.data.pv_record.PVRecordDataset` (import via explicit `sys.path` insertion — same pattern PVRecordDataset itself uses for `baselines/common`). Its item schema (verified at [pv_record.py:337-409](../MMTSFM/src/mmtsfm/data/pv_record.py)): `Y (N,T,1)`, `Y_future (N,H,1)`, `X_cov (N,T+H,C)`, `Z` (cached V-JEPA latents, optional) / `V` (frames), `mask_target (N,T,1)`, `mask_future (N,H,1)`, `mask_visual (N,Tv)`, `video_delta_t (N,Tv)`, `hist_delta_t (N,T)`, `daylight_future (N,H,1)`, `timestamps (T+H,) int64 seconds`, `site_id`.
- Produces: `PVEventDataset(**pv_record_kwargs)` — wraps `PVRecordDataset(num_entities=1)` and converts each item into the **batch schema**: streams `"target"` (Y, d=1), `"weather"` (X_cov history rows, d=C), `"nwp"` (X_cov future rows — known-future weather per protocol, future-known), `"vision"` (V-JEPA latents `Z` flattened per frame, d = latent dim; only emitted when `Z` present), plus `"t_future"`, `"y_future"`, `"daylight_future"`, `"mask_future"`, `"site_id"`. Times = `timestamps` converted to float minutes. Stage-2 Lightning module + `ProtocolEvaluator` wiring (see [MMTSFM protocol_eval](../MMTSFM/src/eval/protocol_eval.py)) is deferred to the Stage-2 plan — this task delivers and tests the data contract only.

- [ ] **Step 1: Write the failing test (synthetic PVRecord item — no dataset of record needed on dev machines)**

`STATECAST/tests/data/test_pv_events.py`:

```python
import torch

from statecast.data.pv_events import item_to_events


def _fake_pv_item(T=8, H=4, C=3, Tv=2, Zd=16):
    ts = torch.arange(T + H, dtype=torch.int64) * 1800   # 30-min grid, seconds
    return {
        "Y": torch.rand(1, T, 1), "Y_future": torch.rand(1, H, 1),
        "X_cov": torch.rand(1, T + H, C),
        "mask_target": torch.ones(1, T, 1), "mask_future": torch.ones(1, H, 1),
        "daylight_future": torch.ones(1, H, 1),
        "mask_visual": torch.tensor([[0.0, 1.0]]),
        "video_delta_t": torch.tensor([[0.0, 1800.0]]),
        "hist_delta_t": (ts[T - 1] - ts[:T]).float().view(1, T),
        "timestamps": ts, "site_id": "site_a",
        "Z": torch.rand(1, Tv, Zd),
    }


def test_item_to_events_schema():
    item = _fake_pv_item()
    sample = item_to_events(item)
    assert set(sample["streams"]) == {"target", "weather", "nwp", "vision"}
    tgt = sample["streams"]["target"]
    assert tgt["v"].shape == (8, 1) and tgt["t"].shape == (8,)
    assert sample["streams"]["nwp"]["v"].shape == (4, 3)      # future weather rows
    assert sample["streams"]["vision"]["v"].shape == (2, 16)
    assert sample["y_future"].shape == (4,)
    t0 = float(sample["t0"][0])
    assert (sample["streams"]["target"]["t"] <= t0).all()
    assert (sample["streams"]["nwp"]["t"] > t0).all()
    # times are minutes: 30-min grid => spacing 30
    dt = sample["streams"]["target"]["t"][1] - sample["streams"]["target"]["t"][0]
    assert float(dt) == 30.0


def test_vision_mask_respects_missing_frames():
    sample = item_to_events(_fake_pv_item())
    assert sample["streams"]["vision"]["mask"].tolist() == [0.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_pv_events.py -v` — Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the adapter**

`STATECAST/src/statecast/data/pv_events.py`:

```python
"""uk_pv / goes_pvdaq adapter: PVRecordDataset items -> StateCast event streams.

`item_to_events` is a pure converter (unit-testable without the dataset of
record); `PVEventDataset` wires it to MMTSFM's PVRecordDataset for real runs.
Times are float minutes; NWP = known-future weather rows per the protocol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

PV_FUTURE_KNOWN = ("nwp",)
_SEC_TO_MIN = 1.0 / 60.0


def item_to_events(item: dict) -> dict:
    T = item["Y"].shape[1]
    H = item["Y_future"].shape[1]
    t_all = item["timestamps"].float() * _SEC_TO_MIN          # (T+H,) minutes
    t_hist, t_fut = t_all[:T], t_all[T:]
    t0 = t_hist[-1]
    streams = {
        "target": {"t": t_hist.clone(), "v": item["Y"][0],
                   "mask": item["mask_target"][0, :, 0]},
        "weather": {"t": t_hist.clone(), "v": item["X_cov"][0, :T],
                    "mask": torch.ones(T)},
        "nwp": {"t": t_fut.clone(), "v": item["X_cov"][0, T:],
                "mask": torch.ones(H)},
    }
    if "Z" in item:
        z = item["Z"][0]                                       # (Tv, Zd) frame latents
        t_vis = t0 - item["video_delta_t"][0] * _SEC_TO_MIN
        streams["vision"] = {"t": t_vis, "v": z.flatten(1) if z.ndim > 2 else z,
                             "mask": item["mask_visual"][0]}
    return {
        "streams": streams,
        "future_known": PV_FUTURE_KNOWN,
        "t0": t0.view(1),
        "t_future": t_fut.clone(),
        "y_future": item["Y_future"][0, :, 0],
        "daylight_future": item["daylight_future"][0, :, 0],
        "mask_future": item["mask_future"][0, :, 0],
        "site_id": item["site_id"],
    }


class PVEventDataset(Dataset):
    def __init__(self, **pv_kwargs) -> None:
        repo = Path(__file__).resolve().parents[4]
        for p in (repo / "MMTSFM" / "src", repo / "baselines"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from mmtsfm.data.pv_record import PVRecordDataset

        self.inner = PVRecordDataset(num_entities=1, **pv_kwargs)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        return item_to_events(self.inner[idx])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_pv_events.py -v` — Expected: 2 PASSED

- [ ] **Step 5: Full suite, diff audit, commit**

```bash
uv run pytest
git diff HEAD   # review: no debug prints, no stray files
git add STATECAST/src/statecast/data/pv_events.py STATECAST/tests/data/test_pv_events.py
git commit -m "feat(statecast): uk_pv event-stream adapter over PVRecordDataset (Stage-2 data contract)"
```

---

## Out of scope for this plan (next plans, in order)

Deliberately excluded so every task here is runnable on a laptop with no cluster or dataset-of-record access:

1. **Stage-2 uk_pv training + ProtocolEvaluator wiring** (`lightning_stage2.py`, per-site batching across the shared-schedule constraint, results into `baselines/results` via MMTSFM's `ProtocolEvaluator`) — needs cluster + dataset of record; blocked on this plan's Task 16.
2. **Gates G1–G3** (latent forecastability probe on cached V-JEPA latents, NWP information probe through existing baselines, synthetic-transfer probe) — run against *existing* MMTSFM/baselines code per STATECAST.md §5; zero new model code, so they belong to the experiment registry (`docs/experiments/ABLATION_REGISTRY.md`), not this build plan. G2 is "next action (1)" in STATECAST.md §9 and can run in parallel with Tasks 2–15.
3. **Associative-scan parallelization** of the assimilation loop (the diagonal gate keeps it possible; v0 is sequential).
4. **ψ-swap counterfactual (A8), amortized-vs-oracle ψ (A7), ablation arms A1–A13, Stage 1 public-corpora pretraining, MIMIC-IV testbed, AsyncBench public release packaging.**

Each experiment run from this code must still follow the repo's experiment workflow: hypothesis + config diff under `configs/` + registry entry + baseline comparison.

## Execution notes

- Work happens on `feat/statecast` (created in Task 1); micro-commit after each task's green step, message format `feat(statecast): ...`.
- Every task is CPU-runnable in minutes; no task touches `/leonardo_scratch`, `baselines/` internals, or MMTSFM code (Task 16 imports MMTSFM read-only).
- After the final task: `uv run pytest` from `STATECAST/` must be fully green; then run `graphify update .` and re-index gitnexus per repo rules.
