# PROTEUS — the forecaster that never stops learning: streaming fast-weight adaptation as the architecture

**(Proposal v1, AI-first framing. Orthogonal to MMTSFM (fusion mechanism), [STATECAST.md](STATECAST.md) (inductive bias), [SIBYL.md](SIBYL.md) (training distribution), [GEPPETTO.md](GEPPETTO.md) (parameter space), and [SCOUT.md](SCOUT.md) (interaction policy): this one moves the question to *time* — a model is not a fixed artifact but a trajectory. Written 2026-07-15, grounded in a last-30-days trend scan.)**

Named for Proteus, the shape-shifting sea god: pin him down and he has already become something else. Every model on the current board is frozen at submission; the atmosphere it forecasts is not.

---

## 0. Paper identity

**Research question (the paper's first sentence):** *for non-stationary multimodal sensor streams, does in-weights streaming memory — fast weights updated by a self-supervised loss at every step — beat in-context memory, and where is the crossover?*

**Model (the vehicle):** PROTEUS — a multimodal forecaster whose memory *is* test-time training. Slow weights (shared, trained offline across plants) carry general competence; **fast-weight modules** (TTT layers) are updated online, at every incoming observation, by a self-supervised next-observation objective on the deployment stream itself — constant per-step latency regardless of how long the model has been deployed (the property NVIDIA's RoboTTT demonstrated this month for robotic memory at 8,000 timesteps). Plants have no IDs and no ψ vectors: **an entity is its fast-weight trajectory.** Zero-shot at a new plant = start from the slow init and stream the 14-day history through the update rule — exactly the "history-only gradients" definition declared for this thesis's protocol, now load-bearing rather than an ablation.

**Contributions (AI-community form):**
1. **TTT as the memory architecture for multimodal forecasting:** first forecaster where continual self-supervised weight updates are the *mechanism* — not offline fine-tuning, not a bolt-on adapter — with constant-latency streaming updates over heterogeneous asynchronous sensors (power, weather, cached V-JEPA satellite latents, NWP). The TTT line (TTT layers; RoboTTT, July 2026; video-TTT) has not touched multimodal geophysical streams.
2. **The in-weights vs in-context controlled study:** same backbone, matched parameters and compute — (a) fast-weight memory (PROTEUS), (b) frozen weights + history in context (ICL), (c) hybrid. Stressed along deployment length, distribution drift, sensor swap, and cadence shift. Prediction stated up front: (b) wins short and stationary, (a) wins long and drifting, and **the crossover point is the paper** — the forecasting analog of the "ICL vs fine-tuning" question the LLM community keeps re-litigating without ground truth.
3. **A new evaluation axis the board doesn't have: adaptation curves.** Skill as a function of hours-since-deployment; seasonal-drift tracking (soiling, panel degradation, season change — real signals in the 106-site record); stability/forgetting metrics (backward transfer when a plant returns to a previous regime). Static models draw flat lines here by definition.
4. **Honest placement of the online-adaptation wave:** the 2026 TSFM-adaptation line (AdapTS, arXiv 2502.12920; black-box error-context adaptation, arXiv 2606.14222; TimesFM 2.5's LoRA tooling) adapts *around* frozen unimodal models. PROTEUS is the in-architecture, multimodal end of that spectrum, and those methods are its baselines — a comparison neither side has run.

**Target venue:** ICLR/NeurIPS main track. **Thesis compatibility:** protocol, splits, metrics, board unchanged; adaptation curves are additive; the streaming pass over the 14-day history is protocol-legal by the declared zero-shot definition.

---

## Intuition — plain-language walkthrough

### I.1 The core idea

Every board model is trained, frozen, and then judged on months of held-out time — as if the world signed a stationarity contract. It didn't: panels soil, seasons turn, sensors drift, inverters get replaced. The standard answers are retraining schedules (operational, ugly) or longer contexts (pay quadratically to *re-read* what you could have *learned*). PROTEUS takes the third answer seriously: keep learning, forever, from the stream itself — which is possible precisely because forecasting is self-supervised: the stream continuously delivers its own labels (the observation that arrives IS the target the model just implicitly predicted). No labels to wait for, no retraining pipeline; the deployment *is* the training loop.

### I.2 Fast weights vs a longer prompt

Both are memory. A prompt is *episodic* memory — exact, expensive, evicted when the window slides. Fast weights are *consolidated* memory — lossy, cheap, cumulative. The 14-day protocol history is small enough that ICL can hold all of it, which is exactly why the comparison is honest: in-context has its best case here, and if in-weights still wins on drift and long deployments, the result is sharp. The constant-latency property is what makes the architecture deployable — step cost never grows with deployment age, unlike context length.

### I.3 What the update rule must survive (the actual research)

Naive online SGD on a stream dies three ways: catastrophic forgetting of slow-weight competence (answer: only fast modules update; slow weights frozen at deployment), instability under outliers (answer: learned per-module update gates and norm bounds — the update rule itself is *meta-trained*, so "how to learn online" is learned offline across plants), and self-poisoning during sensor faults (answer: update gated by prediction-error credibility — an implausible innovation updates nothing; kinship with StateCast's innovation gate, acknowledged, but acting on *weights* not state).

### I.4 A forecast, step by step

New plant, day 0: slow init + streaming pass over the provided 14-day history — the fast weights absorb the plant's transfer function (capacity, orientation show up as consistent prediction-error structure, which the update rule consumes). Forecast normally. Day 30: the model is no longer the day-0 model — it has silently tracked a soiling trend the frozen board models will misread for months. Every step: one forward + one cheap fast-weight update; latency flat.

### I.5 Why it should win both boards

**Intra-plant** is where PROTEUS is structurally favored and the board is structurally weak: held-out-time evaluation *is* a drift setting, and every competitor is frozen. Adaptation is a monotone accumulating edge over the test horizon. **Cross-plant:** the streaming pass over 14 days of history is a stronger conditioning mechanism than a prompt read — it performs actual credit assignment — and the meta-trained update rule has practiced fast entity acquisition across 75 plants (that is what meta-training is for). Prediction (a)-vs-(b) may be close at day 0 and diverge from there; the divergence rate is a headline figure.

---

## 1. Why this is an AI paper in 2026

| 2026 AI agenda | Where PROTEUS sits |
|---|---|
| **Test-time training** (TTT layers; RoboTTT constant-latency memory, July 2026; video-TTT) | Extends TTT from tokens/robot trajectories to heterogeneous asynchronous geophysical sensors — and gives it ground-truth non-stationarity to earn its keep on |
| **ICL vs fine-tuning debate** | The controlled (a)/(b)/(c) study with a physical testbed, drift you can point at, and a measurable crossover |
| **Continual learning renaissance** ("learning never stops" FM agenda) | Continual learning with a live leaderboard consequence instead of Split-CIFAR |
| **Online TSFM adaptation** (AdapTS 2502.12920; error-context black-box adaptation 2606.14222; TimesFM 2.5 LoRA/PEFT) | That line wraps frozen unimodal models; PROTEUS is the in-architecture multimodal endpoint, with the wrappers as baselines |
| **Inference-time compute** | Update-compute is a dial: steps-per-observation trades accuracy for cost — an inference-scaling curve where the compute buys *learning*, not sampling |

**What is honestly NOT novel:** fast weights are Schmidhuber-old; online learning and concept drift have decades. Novelty located in: the multimodal asynchronous instantiation with credibility-gated meta-trained updates, the matched in-weights/in-context/hybrid evidence, the adaptation-curve evaluation axis on a public board, and constant-latency streaming at deployment scale.

---

## 2. Model

```
 asynchronous multimodal stream (power, weather, V-JEPA sat latents, NWP)
   ── o_t ── o_t+1 ── o_t+2 ──►  arrives forever; the deployment IS the training loop
        │
        ▼ (event tokens)
 ┌────────────────────────────────────────────────────────────────┐
 │ BACKBONE                                                       │
 │                                                                │
 │   SLOW WEIGHTS = FROZEN CHRONOS-2 (PRIMARY substrate)          │
 │   pretrained TSFM: general temporal competence already paid    │
 │   for; weather/NWP via covariates; V-JEPA sat latents via a    │
 │   once-trained frozen projection adapter                       │
 │   (native ~30M event-token backbone = control arm)             │
 │                                                                │
 │   FAST MODULES (TTT layers replacing some MLP blocks)          │
 │   ┌──────────────────────────────────────────────┐             │
 │   │ inner model  W_fast   ◄── 1 inner step per   │             │
 │   │                            observation        │             │
 │   │ inner loss: predict next observation embed    │  constant   │
 │   │ meta-learned: inner LR, per-module gates,     │  latency,   │
 │   │              norm bounds                      │  ∀ deploy   │
 │   └──────────────▲───────────────────────────────┘  age        │
 │                  │                                              │
 │        CREDIBILITY GATE: update × plausibility(o_t)             │
 │        (sensor fault / outlier ⇒ learn ≈ nothing)               │
 └───────────────────────────────┬────────────────────────────────┘
                                 ▼
                    ramp-weighted quantile forecast

 ENTITY = FAST-WEIGHT TRAJECTORY (no plant IDs, no ψ):
   day 0 zero-shot  = slow init + streaming pass over 14-day history
   day 30           = has silently tracked soiling/seasonal drift
 META-TRAINING (offline, 75 plants): outer quantile loss AFTER inner
   updates ⇒ the online update rule itself is learned; drift augmentation
   (gain/soiling ramps, sensor swaps, blackouts) injected into episodes
```

- **Backbone (PRIMARY = frozen Chronos-2 as the slow weights):** the slow weights are a frozen pretrained TSFM — general temporal competence rented, not trained. Multimodal inputs enter via the covariates channel (weather/NWP) and a once-trained frozen projection adapter (cached V-JEPA satellite latents). Only the fast modules, the meta-learned update rule, and that adapter are ever trained. The native ~30M event-token decoder (shared skeleton with the siblings) is retained as the control arm to show the mechanism is not substrate-bound and to keep satellite fusion first-class.
- **Fast modules:** TTT layers replacing a subset of MLP blocks — each holds a small inner model updated by an inner self-supervised loss (reconstruct/predict the next observation embedding) with meta-learned inner learning rates, per-module gates, and norm constraints. One inner step per observation ⇒ constant latency.
- **Credibility gate:** inner update scaled by the plausibility of the triggering observation (predictive-likelihood based); sensor faults and outliers learn ≈ nothing.
- **Arms:** (a) PROTEUS (fast weights on, short context); (b) ICL twin (fast weights off, full 14-day context, parameter-matched by widening); (c) hybrid (both). All decode ramp-weighted quantiles.
- **Rejected machinery (decision hygiene):** full-model online SGD (forgetting, latency, and it's "just fine-tuning" — the review death); replay buffers (violates streaming/constant-memory claim; kept only as a diagnostic upper bound); per-plant LoRA banks (that is GEPPETTO's parameter-space axis); making the update rule a Kalman-style state update (that is StateCast — here memory lives in weights, and the fence between "state" and "weights" is exactly contribution 2's subject).

---

## 3. The experiment that carries the paper

**In-weights vs in-context vs hybrid**, matched parameters/compute/data, crossed with:

| Stress axis | Prediction if in-weights memory matters |
|---|---|
| Deployment length (day 0 → month 3 of test time) | (a) accumulates edge; (b) flat; crossover measured |
| Seasonal / degradation drift (real, in the record) | (a) tracks; (b) and the whole frozen board misread systematically |
| Sensor swap / recalibration mid-deployment | (a) re-adapts in hours; (b) contaminated context |
| Cross-plant day-0 (zero-shot) | near-parity; divergence rate thereafter is the figure |
| Stationary easy plants | (b) may win — stated up front; deployment-regime map, not a sweep, is the expected outcome |

**Baselines beyond the twins:** frozen board models (adaptation curves = flat lines drawn from existing results, nearly free); AdapTS and error-context black-box adaptation wrapped around the same backbone (the 2026 wrapper line, finally compared to in-architecture TTT); periodic offline refits (the operational status quo, compute-matched).

**Audit instruments:** fast-weight drift trajectories vs known plant events (probe: does the fast-weight state encode the soiling trend?); update-gate traces (gate → 0 at night / during faults — learned, not coded); forgetting matrix (skill on regime A after adapting through regime B).

---

## 4. Training

**Meta-training (offline, 75 plants):** episodes = simulated deployments (sample plant, sample multi-week window, stream it); outer loss = forecast quantile loss *after* inner updates ⇒ the update rule is learned end-to-end (learning-to-learn-online). Drift augmentation: synthetic gain/soiling ramps, sensor swaps, blackout gaps injected into training streams so adaptation is practiced, not hoped for. **Deployment:** slow weights frozen; fast weights stream. No stage-0 corpus needed — this proposal is deliberately the cheapest of the family (no zoo, no prior, no RL).

---

## 5. Testbed — PV (= thesis chain, unchanged)

Dataset of record, 75/16/15, SS/R²/CRPS + S6 ramps, full board. Board entries: PROTEUS (streaming) and its ICL twin (static control). Adaptation curves and the forgetting matrix are additive results sections. PV is the right domain: genuine slow drift (soiling, seasonal sun geometry, hardware aging) at known timescales, plus fast regime shifts (weather) — the two-timescale structure the fast/slow split is built for, and *checkable* because the drivers are physical.

---

## 6. Gates, ablations, risks

**Go/no-go (ordered, cheap-first):**
- **G1 — drift-exists gate (~week 1, zero new model code):** measure existing baselines' error as a function of test-time distance from their training cutoff, per plant. No systematic degradation ⇒ the record has no exploitable non-stationarity ⇒ kill (and the measurement itself is a useful protocol note).
- **G2 — stability gate:** on held-in plants, streaming for 3 simulated months must never fall below the frozen twin's skill (the "do no harm" bar). Fails ⇒ fix gates/bounds before any claim.
- **G3 — day-0 parity gate:** streaming-pass conditioning must match ICL conditioning zero-shot cross-plant. Fails badly ⇒ the entity-acquisition half of the story dies; drift-tracking half may survive as a smaller paper.

**Ablations:** A1 arms a/b/c (the paper) → A2 credibility gate off → A3 meta-learned vs fixed inner LR → A4 inner steps per observation ∈ {0,1,4} (the learning-as-inference-compute curve) → A5 fast-module placement/capacity → A6 drift augmentation off → A7 wrapper baselines (AdapTS, error-context) on same backbone → A8 replay-buffer diagnostic ceiling → A9 forgetting matrix → A10 probe fast weights for physical drift variables.

**Risks:**

| Risk | Mitigation |
|---|---|
| Record's drift too weak to reward adaptation (G1) | week-1 kill, zero model code; sensor-swap and long-deployment axes can be synthetically injected as a declared secondary claim |
| Online instability / self-poisoning | credibility gate + meta-trained bounds + G2 do-no-harm bar; rollback-free by construction (fast weights resettable) |
| "It's just fine-tuning" review | fenced: constant-latency architectural memory, meta-trained update rule, matched ICL twin, forgetting/adaptation metrics — none of which describe fine-tuning |
| ICL twin wins everywhere | the crossover-map negative result, with drift ground truth — directly useful to the ICL-vs-FT debate; thesis chain unharmed |
| Streaming evaluation cost | fast-weight updates are tiny; evaluation windows subsampled; cheapest proposal of the family regardless |
| Zero-shot definition disputes | pinned to the declared history-only-gradients protocol arm; a forward-pass-only variant (inner steps = 0 at day 0) reported alongside for the strict definition |

---

## 7. Prior-art fence

| Nearest work | Why PROTEUS is outside it |
|---|---|
| **TTT line** (TTT layers, RoboTTT, video-TTT) | tokens/robot memory/video; no heterogeneous asynchronous sensors, no entity transfer, no drift ground truth, no leaderboard |
| **Online TSFM adaptation** (AdapTS, error-context 2606.14222) | wrappers around frozen unimodal FMs; PROTEUS is in-architecture and multimodal — and runs them as baselines |
| **Concept-drift / online-learning classics** | shallow models, drift detectors as plugins; here the update rule is meta-learned end-to-end inside a multimodal FM |
| **Continual-learning benchmarks** | task boundaries and toy streams; here boundaryless physical drift with money-metric consequences |
| **Internal siblings** | StateCast updates a *state*, GEPPETTO generates *static* weights once, SIBYL conditions in-context; PROTEUS is the only one whose parameters move during deployment — the time axis, orthogonal to all |

**Exact claim:** *first multimodal forecaster with constant-latency test-time-training memory — a meta-learned, credibility-gated streaming update rule over heterogeneous asynchronous sensors — evaluated by a matched in-weights vs in-context study with adaptation curves, forgetting matrices, and a measured ICL/TTT crossover on a public benchmark with physical drift ground truth.*

---

## 8. Foundation-model substrate analysis: introducing Chronos-2 / TabFM

**Chronos-2 as the slow weights — adopted as the primary engineering path.** PROTEUS's slow/fast split maps perfectly onto a frozen TSFM: Chronos-2 *is* the slow weights (generic temporal competence, already paid for, zero training), and only the fast TTT adapters + the meta-learned update rule + a once-trained multimodal input adapter (covariates channel for weather/NWP; a projection for cached V-JEPA latents) are learned. The already-cheapest proposal of the family gets cheaper: meta-training reduces to learning *how to adapt* a frozen generalist, not learning the generalist. And the adaptation-cost ladder becomes the pitch: zero-shot Chronos-2 (day-0 floor, free) → +streaming fast-weight updates (tiny per-step gradient on ~1% of parameters, constant latency) → full fine-tune (the expensive ceiling nobody deploys). PROTEUS-on-Chronos-2 sits at the sweet spot: near-fine-tune skill at near-zero-shot cost, which is exactly the generalization-per-adaptation-dollar argument.

**Why this substrate also sharpens the science:** the 2026 wrapper line (AdapTS, error-context black-box adaptation) adapts *around* frozen TSFMs — usually Chronos-family. Putting PROTEUS's fast modules *inside* the same frozen Chronos-2 makes A7 a perfectly controlled comparison: identical substrate, identical stream, only the adaptation mechanism differs (outside-wrapper vs in-architecture TTT). No published work has run that comparison; it is the cleanest evidence slot in the proposal.

**Costs, honestly:** Chronos-2's representations were never trained to be TTT-friendly — the inner self-supervised loss operates on hidden states of a model that didn't expect online updates, so adapter placement and inner-loss design need a search (G2's do-no-harm bar gates it); multimodal fusion through a covariates channel is weaker than native event tokens, so the native 30M backbone stays as a control arm to show the mechanism isn't substrate-bound and to keep the satellite stream first-class.

**TabFM as the in-context arm's cheapest member — adopted as a baseline.** TabFM's only adaptation mechanism *is* a growing context: featurize the stream, append rows, forecast by tabular ICL. That makes it a natural, zero-training member of arm (b) (in-context memory) — the in-weights vs in-context study gains a pretrained-generalist column on both sides (Chronos-2+TTT vs TabFM-ICL) without training anything. Its expected failure under drift (context fills with stale regime rows) vs PROTEUS's consolidation is precisely the crossover the paper is about.

**Net rule:** frozen-FM PROTEUS is the recommended primary instantiation — lowest adaptation cost in the family, strongest baseline symmetry — with the native backbone as control. Same v2 caveat as everywhere: the meta-learned update rule and the crossover study are the paper; "TTT adapters on Chronos-2" alone is glue.

---

## 9. Decision record

Rejected framings: full-model online fine-tuning (forgetting + review death); replay-based continual learning (breaks the streaming claim); folding this into SIBYL's TTT ablation (there it is a few polish steps on a prompt; here continual adaptation is the architecture and the deployment model — different claim, different evidence, different risk). Axis taxonomy, final: MMTSFM = fusion mechanism; StateCast = computation structure; SIBYL = training distribution; GEPPETTO = parameter space; SCOUT = interaction policy; **PROTEUS = adaptation over time**.

**Next actions (ordered):** (1) G1 drift measurement on existing baseline results — zero new model code, days; (2) TTT-layer module + meta-training loop on the shared event-token skeleton; (3) G2/G3 on a 10-plant subset; (4) wrapper baselines (AdapTS, error-context) harnessed for A7; (5) open `exp/proteus` after G1+G2 pass.
