# StateCast — learned data assimilation for asynchronous multimodal forecasting

**(MMTSFM v5, AI-first framing — design of record.** Supersedes MMTSFM_V4_DESIGN.md (not retained in this repo). Written 2026-07-15, reframed same day: the contribution is an AI model + a controlled inductive-bias study; PV power forecasting is the *primary testbed*, not the subject.)

---

## 0. Paper identity

**Research question (the paper's first sentence):** *when does a state-estimation inductive bias beat attention fusion for forecasting from heterogeneous, asynchronous, unreliable sensors?*

**Model (the vehicle):** StateCast — all input streams are treated as noisy, irregularly-timed *observations of one latent state*; each stream enters through a learned per-modality observation operator that updates the state at its own timestamp (an amortized, learned analog of Kalman assimilation); forecasting = rolling a learned continuous-time transition model; per-entity readout parameters are amortized in-context from history.

**Contributions (AI-community form):**
1. **A formulation, not a fusion module:** learned assimilation with factorized observation operators unifies — in one mechanism — multimodal fusion, multi-rate asynchrony, missing sensors, calibrated uncertainty, and zero-shot transfer to unseen entities. No pairwise modality attention exists in the model.
2. **A controlled formulation-vs-mechanism study:** matched-parameter attention-fusion twins evaluated across stress axes (cadence shift, sensor dropout, out-of-distribution entities, horizon length) — an answer about *inductive bias*, publishable in either direction.
3. **In-context learning made auditable:** the entity readout operator ψ is an explicit, low-dimensional, physically interpretable ICL target; the operator-swap counterfactual verifies *what* in-context inference actually estimated — connecting to the implicit-Bayes / task-vector ICL literature with a falsifiable instrument instead of probing.
4. **Testbeds + artifact:** PV power (primary — real multimodal sensors, an existing 30-model leaderboard, physically auditable gates), a second domain (§6), and **AsyncBench** — a released synthetic generator with controllable asynchrony, missingness, noise, and cross-modal causal structure, filling a benchmark gap the IMTS literature keeps papering over with medical data.

**Target venue:** ICLR/NeurIPS main track. **Thesis compatibility:** the PV testbed section *is* the thesis experiment chain (protocol, gates, leaderboard unchanged); the AI framing costs one extra testbed and the generator release.

---

## Intuition — plain-language walkthrough (read this first)

*Unnumbered on purpose: this section explains the concepts; the numbered sections state the claims. It maps 1:1 onto §2 (model), §4 (training), §3 (twin study), §5–7 (testbeds/gates).*

### I.1 The core idea in one analogy

Think of how a weather service works. It never "fuses" a satellite image with a thermometer reading directly. Instead it maintains a **best estimate of the state of the atmosphere** — one internal picture of reality. Every time a new measurement arrives (a satellite pass, a weather balloon, a ground station), it asks: *does this measurement match what my current picture predicts I should see?* If yes, tiny correction. If no — surprise — big correction. This is **data assimilation**, how every operational weather center has worked for 60 years (the Kalman filter is its simplest mathematical form).

StateCast does exactly this, but **learns every component with neural networks**, at the scale of a single PV plant instead of the globe. The satellite crop, the temperature reading, the inverter's own power output — these are not three "modalities" to be mixed by attention. They are three **different sensors looking at the same physical reality**: the sky over that plant. So the model maintains one latent picture of that reality and lets each sensor correct it.

This is why there is no fusion module. Fusion *is* the act of updating a shared state. We didn't pick a better quadrant of the early/late-fusion taxonomy — we removed the question.

### I.2 The latent state `s_t`

A small set of learned vectors — 8 tokens × 256 dims — representing "everything the model currently believes about the atmosphere and conditions over this plant at time t": cloud cover, cloud motion, haze, whatever matters. No dimension is hand-defined; the training losses force it to contain whatever is needed to (a) predict power and (b) predict future observations. And it is **checkable**: in Stage 0 the synthetic generator's true cloud field is known, so linear probes verify cloud cover is readable from `s_t`; on real data we probe for clear-sky index. Degenerate state ⇒ caught early.

### I.3 Observation operators and the innovation gate (the heart)

Each sensor type `m` (power, weather, satellite, NWP) has two small learned networks:

- **`h_m(s)` — "what should this sensor read, given my state?"** If the state believes "thick cloud approaching from west," `h_sky(s)` should output a V-JEPA latent that looks like that frame.
- **`a_m(s, o)` — "given what the sensor actually read, correct the state."**

The update is **innovation-gated**. Innovation = (actual − predicted) observation; the learned gate `G_m` scales the state correction by the surprise:

- Satellite frame at night → predicted and actual both ~black → innovation ≈ 0 → state barely moves. The model *learns* that night frames are uninformative — nothing is hard-coded.
- Clear morning, frame suddenly shows a cloud front → large innovation → big update → power forecast 90 minutes out drops.

This is the learned analog of the **Kalman gain**. In a real Kalman filter the gain comes from covariance algebra under linear-Gaussian assumptions; here it is a network trained end-to-end, so it works for pixels and nonlinear dynamics. The lineage is stated openly — the novelty is not the math but making it work with heterogeneous real sensors and proving where it beats attention.

The "diagonal-gated" constraint is engineering with a purpose: it keeps updates associative-scan-compatible, so training parallelizes like a modern SSM instead of crawling like an RNN — killing the classic "filters train too slowly" objection.

### I.4 The transition model `f(s, Δt)` — the world model

Between observations, and across the whole forecast horizon, nothing arrives; the state must **evolve on its own**. `f` takes the state and a time gap Δt and returns the state Δt later — clouds advect, the sun moves. This *is* the world model: in v3/v4 it was an auxiliary head bolted onto a transformer; here it is the load-bearing engine of the forecast. Δt-conditioning means steps compose — f(f(s, 1h), 1h) ≈ f(s, 2h) — which is also what makes the model cadence-agnostic: a 15-min plant and a 30-min plant are different observation *schedules*, not different problems.

### I.5 A forecast, step by step (concrete)

UK plant, forecast origin 12:00, horizon 6 h:

1. **Assimilate history in time order.** From a learned initial state, walk the last 14 days: at every power reading (30-min), every hourly weather reading — roll `f` to that timestamp, apply that sensor's update. Over the last 2–4 h, dense satellite frames (15–30-min) join in. By 12:00 the state is a filtered summary of everything all sensors said, weighted by surprise.
2. **Roll across the horizon, drip-feeding known future information.** Roll `f` by 30 min → 12:30 state; assimilate **solar geometry** at 12:30 (known exactly ⇒ treated as a perfect observation) and the **NWP forecast** for that step (a *noisy* observation whose trust the model has learned as a function of lead time — NWP is blurry at 30 min, decent at 4 h; the model discovers the crossover itself, which in v4 required a dedicated fusion gate). Repeat to 18:00.
3. **Decode power.** At each step, `g_ψ(s)` maps state → 9 quantiles through the plant's operator ψ (I.6).

The transition is **generative**: rolling forward means *sampling* the next state from a learned distribution (conditional flow matching — same family of method behind GenCast, DeepMind's ensemble weather model). Run K = 8 sampled trajectories in parallel: they fan out with rollout length and tighten after informative observations; the fan *is* the uncertainty, and genuinely ambiguous futures ("clearing vs staying overcast") survive as distinct trajectory modes rather than being averaged away. Unplug the satellite → fewer corrections → wider fan, automatically: calibration is structural, not a bolt-on layer.

### I.6 ψ and the factorization — why zero-shot cross-plant works

Physics fact: power = (site-agnostic weather) passed through (site-specific hardware: capacity, orientation, tilt, soiling, inverter clipping). The architecture bakes this in:

- `f` and all observation operators: **shared across every plant** — they model the sky, and the sky doesn't know what plant sits under it.
- `g_ψ`: the **only** plant-specific component, parameterized by ψ — ~100 numbers.
- ψ is never trained per plant. A **set encoder** reads the 14-day history and emits ψ in **one forward pass** (amortized inference), having learned across the 75 training plants — fed deliberately shortened histories too — how to read a plant's transfer function off its recent behavior (e.g., output peaking at 14:00 instead of noon ⇒ west-facing panels).

**Unseen test plant:** feed its 14-day history (which the protocol gives every model anyway) → set encoder emits ψ → forecast. No gradient step, no fine-tuning, no retrieval — the same zero-shot definition as chronos2_zs on the board.

Why this targets the board's diagnosed failure: cross-plant error is dominated by **bias/scale** (REPORT concl. 7 — shape tracked, level wrong). Attention models smear plant identity through the whole representation, so transfer breaks everywhere at once, undebuggably. Here plant identity is *architecturally imprisoned* in 100 numbers: cross-plant generalization can only fail inside ψ, and A7 (amortized vs oracle-fitted ψ) plus A8 (the ψ-swap counterfactual — decode plant A's state through plant B's ψ and check the error matches the known hardware difference) measure and *demonstrate* the factorization. Degradation cases are quantified by the same mechanism: thin history ⇒ wide ψ posterior ⇒ wide quantiles (in-distribution via short-history sampling); true cold start ⇒ ψ from prior + static metadata, widest quantiles, still forecasts; missing sensors ⇒ fewer updates ⇒ wider fan.

### I.7 What the losses teach

| Loss | Teaches |
|---|---|
| Ramp-weighted pinball on power quantiles | the task; ramp-weighting upweights the sharp transitions the S6 metric scores |
| Latent prediction — `h_m(rolled state)` vs encoded real future observations (e.g., the V-JEPA latent of the frame that actually arrived 2 h later) | forces `f` to genuinely model cloud dynamics, not interpolate power; JEPA-style — latent space only, no pixels, no decoder |
| Flow-matching loss on the transition | teaches `f` the *distribution* of next states, so the sampled ensemble covers real alternative futures — no hand-tuned diversity tricks needed |
| Innovation regularizer | keeps the filter honest: state can neither ignore sensors (updates ≈ 0 always) nor parrot them (memorize the observation, forget dynamics) |

Curriculum, in intuition form: **Stage 0** puts the causal chain "dark blob upstream ⇒ power dip in 90 min" into the weights on unlimited synthetic data (with known ground-truth state for probing) before any real frame is seen; **Stage 1** (public corpora, power-only) buys generic temporal competence so the model is a foundation model rather than a PV-idiot-savant, and proves graceful degradation to unimodal; **Stage 2** trains the full loop on the 75 real training plants.

### I.8 Why the twin experiment is the paper

Pure leaderboard accuracy = energy paper. The AI contribution is a **controlled experiment about inductive bias**: build the strongest attention-fusion model (v4's dual-axis trunk — same parameters, data, losses) as a **twin**, then stress both where the formulations should differ. Parity is *expected* in-distribution and stated up front (attention is a universal approximator). The predicted separations: cadence shift (events vs learned grids), test-time sensor dropout (calibrated widening vs unpredictable degradation), unseen plants (ψ vs entangled identity), horizon extrapolation (composable `f` vs fixed decoder), data efficiency (structure wins at few plants; where attention catches up is itself a finding). Either outcome publishes: wins on some axes ⇒ "inductive bias buys robustness at zero accuracy cost"; a twin sweep ⇒ a rigorous negative result on a hypothesis much of the world-model community holds.

### I.9 The mental model to carry

StateCast = a learned Kalman filter whose state is a neural embedding of the local sky; each sensor gets a learned "what should I see / how do I correct" pair; forecasting = rolling the state forward while drip-feeding the future things we genuinely know (sun position exactly, NWP with learned distrust); and the *only* component that knows which plant we are on is a 100-number vector estimated on the fly from two weeks of history. Decoupled resolutions, missing frames, calibrated uncertainty, zero-shot new plants — all consequences of that one loop, instead of five modules.

---

## 1. Why this is an AI paper in 2026

| 2026 AI agenda | Where StateCast sits |
|---|---|
| **World models** (JEPA line, V-JEPA-2, action-conditioned agents, video WMs) | The unoccupied corner: **observation-driven filtering world models** — state estimated from heterogeneous real sensors, no actions, no pixel generation, evaluated by downstream decision-relevant forecasts. JEPA-orthodox (all losses in latent space). |
| **Structure vs scale / attention** | The twin study is a clean data point in the "when does inductive bias matter" debate — the reviewer-magnet axis, independent of domain. |
| **ICL theory** (implicit Bayesian inference, task vectors) | ψ-amortization is ICL where the task parameter is explicit and auditable; operator-swap is a counterfactual test no LLM-ICL setup can run. |
| **Reliable ML / calibrated uncertainty** | Posterior widening under sensor removal is a *structural* property, measured with reliability diagrams per degradation condition — not a post-hoc calibration layer. |
| **Generative ensembles / flow matching** (GenCast beat ECMWF ENS with diffusion ensembles, Nature 2025; FM is 2026's default generative objective) | The transition is a conditional flow-matching model — GenCast-style generative ensembles brought to *site scale, inside a learned filter*, an unoccupied combination. |
| **Irregular multivariate TS** (active 2025–26 subfield: ASTGI, recursive multi-scale IMTS, IMTS-MAE) | That literature handles asynchrony *within* one modality family and stops at accuracy; StateCast extends to heterogeneous exogenous modalities and makes degradation/transfer first-class. Their benchmarks become our baselines' home turf. |
| **TSFM wave** (TiRex-2, Toto-2.0, Chronos-2) | With only the target-observation operator active, StateCast *is* a state-space TSFM (Stage 1). TSFMs are baselines and one diagnostic arm — the paper is not competing on generic pretraining scale. |

**What is honestly NOT novel — and how the paper handles it:** the math lineage (deep Kalman filters, Kalman-VAE, latent SDEs, neural processes, 2015–2020) is mature. The paper says so in paragraph 2 and locates novelty where it actually is: the *multimodal heterogeneous observation-operator instantiation*, the *entity factorization with auditable ICL*, and the *controlled evidence* — not the filtering math. Papers that hide their lineage die in review; papers that weaponize it ("60 years of assimilation practice, finally end-to-end learnable at sensor level") survive.

---

## 2. Model

General statement: entities e (PV plants / patients / stations) each emit M observation streams o^m at arbitrary timestamps. A shared latent state s_t (8 tokens × 256, K particles) explains all streams of an entity.

```
observations (asynchronous, native cadence, timestamped, per entity)
  stream 1 ... stream M          future-known streams (NWP, solar geometry, schedules,
     │              │             frozen-TSFM forecast — Chronos-2 assimilated as a sensor, §9)
     │              │                       │
  [obs op a_1] ... [obs op a_M]      [obs ops a_f, a_tsfm: trust conditioned on lead time]
     └──────┬───────┘                       │
            ▼                               ▼
  ════════ LATENT STATE s_t  — updated at observation times only ════════
            │                               ▲
            │   TRANSITION f(s, Δt): continuous-time, Δt-conditioned, composable
            └── rolls state between observations and across the forecast horizon ──┘

  readouts: g_ψ(s) → target quantiles   (ψ = entity operator, amortized from history by set encoder)
            h_m(s) → predicted stream m (training losses; = world-model objective for the vision stream)
```

**Assimilation step (the learned gain):** at time t, prior s⁻ → predicted observation ô = h_m(s⁻); innovation (o − ô) drives a diagonal-gated cross-attention update s⁺ = s⁻ + G_m(innovation, noise embedding) · Δ. Surprise ⇒ large update; expected observation ⇒ near-identity. Modalities interact *only through the state*. Diagonal gating keeps updates associative-scan-compatible ⇒ chunked-parallel training (the classic "filters train slow" objection, answered by construction).

**Entity factorization (the ICL claim):** transition f and all observation operators are shared across entities — they model the environment. Only g_ψ is entity-specific, with ψ (~10² dims) amortized from the history window. Zero-shot transfer to an unseen entity = estimate ψ in-context, reuse everything else. **Auditable:** swap ψ between entities and check the error decomposes as claimed; probe ψ against known entity metadata (PV: capacity/orientation; ICU: age/weight/condition class).

**Generative transition (flow matching) + uncertainty:** f is not deterministic — it is a **conditional flow-matching transition**, s_{t+Δt} ~ p_θ(· | s_t, Δt), learned as a stochastic interpolant in latent space. The forecast ensemble = K samples from this transition rolled through the horizon (GenCast's diffusion-ensemble result — Nature 2025, beat ECMWF ENS — validates generative ensembles for weather; nobody has run them *at site scale inside a learned filter*). This replaces the older particles-plus-noise-injection heuristic and the K-hypothesis diversity regularizer with one principled objective: distributional coverage ("clearing vs staying overcast" = actual modes of p_θ) comes from the FM loss by construction, CRPS becomes the ensemble's native score, and spread still grows with rollout and shrinks with each assimilated observation (missing sensor ⇒ no update ⇒ quantified widening). Sampling cost handled by few-step shortcut/consistency distillation of the flow; a deterministic mean-mode remains for cheap point forecasts.

**Machinery choices considered and rejected (decision hygiene):** pixel-space diffusion/video generation for the sky stream (violates the latent-space stance; generation quality ≠ forecast value; 100× compute); Mamba-3/xLSTM as the f backbone (legitimate engineering choice, zero claim value — implementation detail); hypernetwork-generated LoRA in place of the ψ vector (destroys ψ's auditability — the swap test and metadata probes are the ICL claim; kept only as optional ablation A13); "in-context filtering" via a plain transformer over observation tokens (that is *literally the attention twin* — making it the model would delete the paper's central experiment).

**Budget:** ~30M trainable. Heavy perception encoders (V-JEPA for satellite) frozen, cached, outside the model boundary — observations arrive already embedded.

---

## 3. The experiment that carries the paper: formulation vs mechanism

**Twin protocol:** StateCast vs a matched-parameter, matched-data, matched-loss attention-fusion twin (dual-axis time×track transformer with continuous-time embeddings — i.e., v4's trunk, the strongest member of the crowded class). Both trained identically. Then stress:

| Axis | Prediction if the formulation matters |
|---|---|
| **In-distribution accuracy** | parity (attention is a universal approximator; no win expected here — say so up front) |
| **Cadence shift** (train 30-min schedules, test 15-min / jittered) | filter degrades gracefully (updates are timestamped events); attention twin's learned positional structure breaks |
| **Sensor dropout at test time** (remove modalities, vary history density) | filter: monotone, *calibrated* widening (reliability diagrams); twin: uncalibrated degradation |
| **OOD entities** (zero-shot cross-plant / cross-patient) | factorized ψ transfers; twin's entangled entity information does not |
| **Horizon extrapolation** (train 6 h, test 12 h) | composable transition extrapolates; decoder-bound twin cannot |
| **Data efficiency** (subsample training entities) | structure pays at low data; attention catches up with scale — *the crossover point is itself a result* |

Either outcome publishes: structure wins somewhere ⇒ "inductive bias buys robustness axes X, Y at zero accuracy cost"; twin sweeps ⇒ a rigorous negative result on the assimilation hypothesis with unusual experimental hygiene (rarer and more citable than another +0.3% fusion paper).

**Physics-audit instruments (only this formulation affords):** gate traces G_m over conditions (satellite gate → 0 at night/clear-sky, spikes at cloud fronts); linear probes for known state variables (cloud cover, clear-sky index) on the latent state; ψ-swap matrices. These make Figures 2–4 and are unreproducible by the attention twin — the asymmetry is part of the argument.

---

## 4. Training

**Losses:** ramp-weighted pinball on target + conditional flow-matching loss on the transition (distributional rollout; subsumes the old particle-diversity/energy-score patches) + latent prediction losses on h_m readouts vs encoded real future observations (JEPA-style, no reconstruction) + innovation regularizer (state must neither ignore nor memorize observations).

**Curriculum:**
- **Stage 0 — AsyncBench synthetic.** Generator: latent causal process (e.g., advecting field → intermediate variable → entity-transformed target) rendered into M streams with randomized cadences, offsets, noise levels, missingness, and entity transfer functions. Ground-truth state known ⇒ *state-recovery probes* validate the latent semantics before real data. This generator, parameterized and released, is the benchmark artifact.
- **Stage 1 — public TS corpora,** target-stream-only (StateCast as a state-space TSFM; generic temporal competence; proves the degradation path).
- **Stage 2 — real testbeds** (per-domain, below).
- Throughout: observation-blackout sampling (contiguous gaps — teaches coasting), short-history sampling (trains ψ amortization for the thin-history regime), randomized schedules (exercises asynchrony).

---

## 5. Testbed 1 — multimodal PV forecasting (primary; = the thesis chain, unchanged)

Everything from v4 carries over verbatim as the domain instantiation: dataset of record (106 sites, uk_pv 30-min + goes_pvdaq 15-min, satellite crops, weather covariates), decoupled-resolution input contract (14-day native-cadence power/weather history; dense 2–4 h satellite window at intra-hour cadence; NWP past+future as observations with lead-time trust; deterministic solar geometry as exact observations), cross-plant protocol (75/16/15), intra-site held-out-time reported first-class, cross-region matrix, SS/R²/CRPS + S6 ramps, full baseline board (iTransformer 0.552 to beat, time_vlm, chronos2_oracle 0.474 ceiling reference, TiRex-2, Toto-2.0, PARA-PV, FusionSF, SolarCrossFormer).

Why PV is the *right* AI testbed (stated in the paper): genuinely heterogeneous sensors (pixels, scalars, forecasts) with real asynchrony; a physical ground-truth state (the cloud field) enabling audit; an existing 30-model leaderboard with measured oracle gaps (+0.14 SS for future weather — the information-value budget is *known*); and a diagnosed failure mode (bias/scale dominates cross-entity error) that the ψ factorization targets by construction.

**Domain gates (ordered, from v4, unchanged):** G1 latent forecastability probe on cached V-JEPA latents (both cadences); G2 NWP information probe through existing baselines (zero new model code; requires the historical-forecast-archive fetch — v4 §7 data work carries over, with the synthetic-NWP fallback); G3 synthetic-transfer probe.

---

## 6. Testbed 2 — the anti-"application-paper" insurance

**Requirement:** irregular, multimodal, missing-heavy, entity-structured, with established baselines. **Choice: ICU physiological forecasting (PhysioNet/MIMIC-IV):** vitals (dense, regular-ish) + labs (sparse, event-driven) + interventions (point events) → forecast vitals; entities = patients; ψ = patient physiology operator; OOD axis = held-out patients (and held-out care units for the cross-region analog). The IMTS SOTA line (ASTGI, recursive multi-scale IMTS, latent-ODE family, neural processes) supplies credible baselines on their home turf.

Scope discipline: testbed 2 runs the *twin protocol + degradation axes only* — no domain SOTA chase, no clinical claims. One table + one figure. Its job is to show the formulation's advantages are not solar-shaped. (Fallback if MIMIC access stalls: USHCN/air-quality station forecasting — weaker but zero-friction.)

---

## 7. Gates, ablations, risks

**Go/no-go (ordered):** G1, G2, G3 (domain, §5) → **G4 twin test** on Stage-0 synthetic + uk_pv subset (~week 2 of model work): StateCast must reach accuracy parity AND win ≥ 1 structural axis, else the assimilation story dies cheaply and the negative result is documented with the same rigor.

**Ablations:** A1 target-only filter (TSFM floor) → A2 +weather obs → A3 NWP-as-observation vs NWP-as-concatenated-covariate (*formulation ablation* — isolates "observations vs inputs") → A4 +satellite obs → A5 innovation gating off (plain additive updates) → A6 FM-generative transition vs deterministic-f+noise-particles (*the machinery ablation*; ensemble size K ∈ {1,4,8} within each) → A7 ψ amortized vs oracle-fitted (ICL headroom) → A8 ψ-swap matrix → A9 Stage 0 on/off → A10 oracle future observations (diagnostic ceiling) → A11 cadence-shift stress → A12 cross-region/cross-unit matrices → A13 (optional) hypernet-LoRA ψ vs vector ψ.

**Risks:**

| Risk | Mitigation |
|---|---|
| State bottleneck loses to full attention everywhere | G4 catches in week 2; negative result publishable under this protocol; M, d are dials |
| "Kalman filter with extra steps" review | leaned into explicitly (§1); novelty located in instantiation + evidence + audit instruments, never in the math |
| Twin parity on all robustness axes too | the crossover/data-efficiency curves still constitute the inductive-bias result; thesis (PV chain) unharmed either way |
| Second testbed doubles the work | scope-fenced to twin protocol only; fallback dataset named; cut = last resort (paper drops to borderline at AI venues without it — this is the one non-negotiable addition) |
| Recursive training slow/unstable | diagonal-gated scan-compatible updates; chunked teacher forcing; Stage 0 as cheap debugging ground with known true state |
| FM sampling cost per transition step blows up rollout latency | few-step shortcut/consistency distillation; deterministic mean-mode for point metrics; A6 quantifies the accuracy/cost trade directly |
| Gates learn to ignore a sensor (v1 pathology) | innovation regularizer + gate-trace audit as first-class instrument; G1 de-risks the satellite signal itself |
| NWP archive gaps 2019–2020 | v4 fallback: declared synthetic-NWP proxy; deployable claim scoped to ClimateHackAI-covered subset |

---

## 8. Prior-art fence

| Nearest work | Why StateCast is outside it |
|---|---|
| **Aardvark Weather** (Nature 2025) — end-to-end obs→forecast | global gridded NWP replacement; no entity factorization, no ICL claim, no multimodal site sensors, encoder-decoder not recursive assimilation |
| **Latent assimilation** (FengWu-DA, DiffDA) | assimilates into weather-model latents for the weather task; no heterogeneous sensor operators, no downstream entity targets |
| **Deep Kalman / Kalman-VAE / latent SDE / neural processes** | the machinery; contribution = heterogeneous-multimodal instantiation + factorized auditable ICL + controlled twin evidence at leaderboard scale |
| **IMTS line** (ASTGI, 2602.21498, 2505.22815) | asynchrony within one modality family, accuracy-only evaluation; here: exogenous heterogeneous modalities, robustness/transfer/calibration as primary axes |
| **Attention-fusion multimodal TS** (FusionSF, M3S-Net, MATE, crossvivit, Time-VLM, Solar-VLM) | the crowded taxonomy this design exits; represented by the strongest twin |
| **TSFMs** (Chronos-2, TiRex-2, Toto-2.0) | covariates as inputs not observations; no state semantics; baselines + Stage-1 degenerate case |

**Exact claim (every noun load-bearing):** *first learned-assimilation model for forecasting from heterogeneous asynchronous multimodal sensors with factorized, in-context-amortized entity operators — evaluated by a matched-twin inductive-bias study across robustness axes on two real domains and a released controllable benchmark.*

---

## 9. Foundation-model substrate analysis: introducing Chronos-2 / TabFM

**Chronos-2 as the trunk — rejected, uniquely here.** In the sibling proposals the backbone is swappable plumbing, so a frozen TSFM can be the primary model. StateCast is the one design where that move deletes the paper: the claim *is* the mechanism (learned assimilation vs attention), and replacing the recurrent filter with a transformer decoder produces… the attention twin (§3), i.e., the control arm. FM-as-primary is structurally impossible without becoming the thing the paper argues against.

**Chronos-2 as a sensor — adopted, and it is the formulation's party trick.** StateCast's premise is that every information source is a noisy observation of the latent state — and a frozen Chronos-2 zero-shot forecast of the plant's power trajectory is exactly that: one more future-known stream, assimilated through its own observation operator `a_tsfm` with learned lead-time trust, precisely like NWP (§I.5 step 2). The innovation gate then *learns* when the TSFM is credible (clear-sky persistence regimes: high trust; cloud-front hours where the satellite disagrees: down-weighted). This buys pretrained cross-domain generalization at zero training cost, with calibrated trust and graceful removal for free — no architecture change, one new operator pair. No other proposal can ingest a foundation model as *data*.

**Twin discipline preserved:** the attention twin receives the same Chronos-2 forecast as a concatenated covariate — the existing A3 formulation ablation (observations vs inputs) extends to the FM stream verbatim, so matched-information comparison survives.

**Stage-1 economics:** Chronos-2 already paid for generic temporal competence, so Stage 1 (public-corpora pretraining of the filter) shrinks to a short run that teaches filter *dynamics*, not competence — the assimilated TSFM stream carries the rest. Cheaper training, same claims.

**TabFM:** featurized 14-day history → plant metadata regression via ICL, as a floor/sanity bound for the ψ set-encoder (what must be recoverable from history at all). Baseline instrument, not a component.

**Net rule:** in StateCast the foundation model enters as an *observation stream* (new ablation A3′′: TSFM-as-observation vs TSFM-as-covariate vs no-TSFM), never as the trunk. The one proposal where "FM inside" and "FM beside" have a third option: "FM as sensor."

---

## 10. Decision record (v1 → v5)

v1: frozen FMs + attention fusion, measured lift ≈ 0. v2: better glue onto a frozen backbone; interface commoditized by TiRex-2 within weeks. v3: native multimodal training, fusion still attention. v4: right information (multi-rate inputs, dual-source future), still the crowded mechanism. v5: mechanism replaced by state estimation; **then reframed AI-first** — the model is a general asynchronous-multimodal forecaster, the paper is an inductive-bias study with auditable ICL, PV is the primary testbed (and remains the complete thesis chain), a second domain and a released benchmark generator purchase the AI-venue claim.

**Next actions (ordered):** (1) G2 NWP probe — zero new model code; (2) historical NWP archive fetch; (3) G1 latent probe on cached latents; (4) AsyncBench generator v0 + G4 twin scaffolding; (5) MIMIC-IV access request (long lead time — start now); (6) open `exp/statecast` branch after G1+G4 pass.
