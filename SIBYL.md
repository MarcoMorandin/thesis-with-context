# SIBYL — forecasting as amortized inference over a written world prior

**(Proposal v1, AI-first framing. A deliberately different bet from [STATECAST.md](STATECAST.md) and the MMTSFM line: not a fusion mechanism, not a filtering mechanism — a training-distribution paper. Written 2026-07-15.)**

Named for the Sibyl, who never saw the future — she inferred it from signs. SIBYL never sees the test plant's future either; it infers it in-context, because it has already lived a million synthetic lifetimes that contained plants like it.

---

## 0. Paper identity

**Research question (the paper's first sentence):** *is zero-shot generalization to unseen entities better bought by training on real entities, or by amortizing Bayesian inference over an explicit, procedurally-written prior of synthetic ones?*

**Model (the vehicle):** SIBYL — a generic decoder-only transformer over interleaved multimodal event tokens, trained almost entirely on samples from a **procedural world prior**: a generative program that writes synthetic sites (latent sky process → heterogeneous asynchronous sensors → entity transfer function → power). Every training example is a *new plant*; cross-entity transfer is therefore **in-distribution by construction**. At test time the real plant's 14-day multimodal history is the prompt, and the forecast is (approximately) the **posterior predictive under the prior** — the prior-fitted-network (PFN) argument, lifted from tabular rows to multimodal asynchronous streams.

**Contributions (AI-community form):**
1. **The prior is the model.** First prior-fitted network for multimodal asynchronous forecasting: architecture deliberately generic (the contribution is *what it is trained on*, not a mechanism), trained on an explicit, seeded, fully reproducible generative program. A foundation model whose entire pretraining distribution ships in the supplementary material.
2. **A controlled real-vs-simulated-experience study:** identical architecture, identical compute, three data arms — real-only, prior-only, prior→real — crossed with intra-plant, cross-plant, cross-region, sensor-dropout, and cadence-shift evaluation. "Where does simulated experience beat real experience" is the headline result, publishable in any direction.
3. **Generalization causally attributed to prior components:** knock out ingredients of the generative program (cloud advection, entity randomization, asynchrony randomization, sensor-noise diversity) and measure which generalization axis each one buys. No real-data-trained model can run this experiment — the training distribution is not an artifact here, it is an *instrument*.
4. **Inference-time scaling for forecasting:** accuracy/CRPS as a function of test-time compute along four axes — sample count K, prompt-resampling ensembles, iterative self-conditioning, and self-supervised test-time training on the target history (the *in-context vs in-weights adaptation* comparison, compute-matched). Inference scaling curves, 2026's favorite figure, drawn for a non-LLM domain with verifiable ground truth.
5. **Auditable Bayes:** on tractable synthetic tasks the exact posterior is computable, so the gap to Bayes-optimality is *measured*, not assumed — and calibration decay is charted as a function of distance from the prior (a quantitative misspecification meter). This grounds the ICL-as-implicit-Bayesian-inference literature with actual ground truth.

**Target venue:** ICLR/NeurIPS main track. **Thesis compatibility:** protocol, splits, metrics, baseline board unchanged; the generator doubles as the AsyncBench release already planned in StateCast §4.

---

## Intuition — plain-language walkthrough (read this first)

### I.1 The core idea in one analogy

TabPFN (Nature 2025) did something strange to tabular ML: instead of training on real datasets, it trained a transformer on **millions of fake datasets sampled from a prior**, and at test time it reads your real dataset as a prompt and predicts — no gradient steps — because predicting-given-a-prompt *is* Bayesian inference when the training distribution is a prior. It beat tuned gradient boosting.

SIBYL asks: what is the equivalent move for our problem? Our problem is *entities* (plants) that each emit heterogeneous asynchronous sensor streams, and the hard axis is **generalizing to entities never seen in training**. Notice the trap every real-data-trained model is in: it saw 75 plants. Seventy-five. Whatever it learned about "what plants are like" comes from 75 samples of plant-space. But we *know* what plant-space looks like — capacity, orientation, tilt, clipping, soiling are physics, and cloud fields are advecting stochastic processes. So instead of hoping 75 real plants teach the model plant-space, **we write plant-space down as a generative program and sample it a million times**. A model trained that way has met a million plants. The 76th — a real one — is not out-of-distribution. It is Tuesday.

### I.2 Why this is "completely different" from MMTSFM and StateCast

MMTSFM asked: *what mechanism mixes the modalities?* (attention fusion). StateCast asked: *what inductive bias structures the computation?* (learned assimilation). Both are **mechanism papers** — the training data is real and fixed, the architecture carries the claim. SIBYL inverts this: the architecture is the most boring one available on purpose — a plain decoder over event tokens, the very thing StateCast §2 dismisses as "literally the attention twin" — and the **training distribution carries the claim**. If SIBYL wins, it wins *because of what it was trained on*, and the prior-knockout ablations prove that causally. Mechanism and data-distribution are orthogonal research axes; this proposal occupies the one the other two left empty.

### I.3 The generative program (the actual research object)

A hierarchical sampler, four levels, all parameters randomized per task:

1. **World:** a latent sky process — advecting cloud fields with sampled velocity, diffusion, and convective pop-up dynamics; diurnal solar geometry from sampled latitude/longitude/season. Several dynamics *classes*, so the model cannot memorize one physics.
2. **Sensor suite:** which modalities exist this time (target power always; weather, satellite, NWP each present with some probability — teaching graceful degradation *in the prior*), each with sampled cadence, phase offset, noise model, dropout process, and miscalibration.
3. **Entity:** a transfer function from sky state to power — capacity, orientation/tilt (geometric projection), inverter clipping, soiling drift, curtailment events. ~10² interpretable parameters, sampled from wide physical ranges.
4. **Render:** streams materialized at native cadences. The "satellite" stream is rendered as short cloud-field videos and pushed through the **same frozen V-JEPA encoder used for real satellite crops** — sim and real meet at the latent interface, so the pixel-level sim2real gap is absorbed by a perception model that never trains on either.

Everything is generated **on the fly inside the dataloader from a seed** — no dataset on disk, which the Leonardo quota will appreciate, and which makes the pretraining data literally reproducible from the paper.

### I.4 A forecast, step by step

Unseen UK plant, forecast origin 12:00, horizon 6 h:

1. Serialize the 14-day history into one time-ordered token stream: every power reading, weather reading, V-JEPA satellite latent, past NWP value becomes a token = [modality embedding | continuous-time embedding | value projection]. Older history enters through coarse learned-pooled summary tokens; the recent window at native cadence. Missing sensor ⇒ tokens simply absent — asynchrony and missingness need no mechanism at all in a decoder over timestamped events.
2. Append the future-known tokens: solar geometry (exact), NWP forecasts tagged with lead time.
3. Decode the target autoregressively as a distribution (quantile head; K samples for CRPS). No gradient step — the prompt *is* the plant identification. During its million synthetic lifetimes the model has learned that "output peaking at 14:00" means west-facing, that a clipped plateau means an undersized inverter — because those correlations were *in the prior* and in-context inference recovers them.
4. Optionally spend more inference compute: more samples, prompt-resampled ensemble members, refinement passes, or a few LoRA gradient steps on the history itself (self-supervised — forecasting needs no labels beyond the stream). Each axis has a measured accuracy-per-FLOP exchange rate; the paper reports which adaptation currency is cheapest.

### I.5 Why it should win both boards

**Cross-plant:** the board's diagnosed failure is bias/scale transfer (REPORT concl. 7). For a real-data model, a new plant's transfer function is an extrapolation from 75 examples. For SIBYL it is an interpolation over ~10⁶ examples whose generative ranges were *chosen to cover reality*. This is the one axis where the argument is structural, not hopeful.

**Intra-plant:** three stacked advantages — (a) unlimited pretraining data means model scale is free, so SIBYL trains at 100–200M where the mechanism papers sit at 30M; (b) the prior→real fine-tuned arm C keeps every bit of leaderboard sharpness real data buys; (c) inference-time compute is a dial nobody else on the board has.

### I.6 What could kill it, honestly

One thing: **prior misspecification** — reality contains plant behaviors the generative program never wrote (weird curtailment policies, snow, sensor pathologies). Three answers, all first-class in the design: the misspecification meter (I.7) *measures* the gap instead of hoping; arm C fine-tuning *closes* it with real data; and gate G1 (prior-only model must already beat persistence and approach chronos2_zs zero-shot on real uk_pv) kills the project in week 2 if the prior is fantasy. Cheap death is a feature.

### I.7 The audit instruments (what makes it science, not augmentation)

- **Bayes-gap:** on prior tasks with tractably few latent parameters, compare SIBYL's predictive to the exact/particle posterior. Reviewers of "ICL is implicit Bayes" papers never get ground truth; we manufacture it.
- **Misspecification meter:** calibration and skill charted against a measured distance between a real plant's history statistics and the prior's typical set — the first quantitative answer to "how wrong can the prior be before the model is."
- **Entity-parameter probes:** the synthetic entity parameters are known by construction; probe them from the model's activations over the prompt to show *what* in-context inference recovered — the same auditable-ICL move as StateCast's ψ, but with exact ground truth instead of metadata proxies.

---

## 1. Why this is an AI paper in 2026

| 2026 AI agenda | Where SIBYL sits |
|---|---|
| **Prior-fitted networks / amortized inference** (TabPFN Nature 2025, PFN line, simulation-based inference) | The unoccupied extension: from i.i.d. tabular rows to multimodal asynchronous *streams* with entity structure — posterior *predictive* over trajectories, not posterior over parameters |
| **Synthetic-data pretraining** (Chronos/KernelSynth, TiRex synthetic mixes, synthetic-data debates in LLMs) | Those use synthetic as *augmentation* with no semantics attached; here the synthetic distribution is an explicit prior, and that semantics is exploited (Bayes-gap audit, prior-knockout attribution) |
| **Inference-time scaling** (reasoning models; test-time compute as the new scaling axis) | Scaling curves for a verifiable non-LLM domain, four compute currencies compared, including the in-context vs in-weights (TTT) exchange rate |
| **World models** | Inverted stance: don't *learn* the world model — *write* it, and distill it into an inference network. The generative program is a white-box world model with knockout switches |
| **ICL theory** (implicit Bayes, task vectors) | The one setting where the prior is known exactly, so "ICL approximates the posterior" is a measurement, not an interpretation |
| **Data-centric AI / open science** | Pretraining data reproducible from a seed; no proprietary corpus; the generator is the released benchmark (AsyncBench, promoted from curriculum aid to central object) |
| **TSFM wave** (Chronos-2, TiRex-2, Toto-2.0) | They are baselines; none is multimodal-native, none makes a posterior claim, none ships its training distribution |

**What is honestly NOT novel — and how the paper handles it:** PFNs exist (2021–), ForecastPFN exists (univariate, weak), synthetic pretraining for TSFMs exists (Chronos). Paragraph 2 says so, then locates the novelty where it is: the *multimodal asynchronous entity-structured instantiation*, the *controlled real-vs-sim experience study*, the *prior-attribution and Bayes-gap instruments*, and inference-time scaling in this domain. Same survival rule as StateCast: weaponize the lineage, never hide it.

---

## 2. Model

```
 TRAINING: sample a NEW synthetic plant per example, forever (seeded, no disk)
 ─────────────────────────────────────────────────────────────────────────────
 ┌ PROCEDURAL WORLD PRIOR (the generative program) ──────────────────────────┐
 │ 1. WORLD    sample sky process: advection/diffusion/convective dynamics,  │
 │             velocity field, lat/lon/season geometry                       │
 │ 2. SENSORS  sample suite: which modalities exist, cadence, phase, noise,  │
 │             dropout, miscalibration    (absence trained-in ⇒ degradation) │
 │ 3. ENTITY   sample transfer fn: capacity, orientation/tilt, clipping,     │
 │             soiling drift, curtailment (~10² interpretable params)        │
 │ 4. RENDER   streams at native cadence; "satellite" = cloud-field clips    │
 │             through the SAME frozen V-JEPA as real crops (G4 gate)        │
 └──────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼ ~10⁶–10⁷ synthetic site-tasks
 INFERENCE (real unseen plant):
   14-day history ──► event tokens [modality emb | cont-time emb | value proj]
   old history → learned-pool summary tokens; recent window native (~4–8k tok)
   + future-known tokens (solar geometry exact; NWP with lead-time emb)
                                    │
                                    ▼
        ┌──────────────────────────────────────────────┐
        │ PLAIN DECODER-ONLY TRANSFORMER (100–200M)    │   boring on purpose:
        │ no towers, no cross-attn, no state, no filter│   the PRIOR carries
        │ init: scratch  → arms A/B/C (the science)    │   the claim
        │ init: Chronos-2 + LoRA → arm B★ (leaderboard)│
        └──────────────────────┬───────────────────────┘
                               ▼
        autoregressive distributional head (9 quantiles; K samples → CRPS)
                               │
        prompt ≈ posterior conditioning  ⇒  forecast ≈ posterior predictive
        optional test-time compute: +K samples, prompt-resample ensembles,
        refinement passes, LoRA TTT steps on history (separate arm, matched)
```

**Tokenization:** every observation, any stream, one token: [modality-type embedding | continuous-time embedding (learned Fourier features on absolute time-of-day/day-of-year + relative Δt) | value projection]. Satellite = frozen V-JEPA latent, projected. NWP future tokens carry a lead-time embedding. Solar geometry tokens are deterministic inputs. Missing data = absent tokens; cadence = token spacing. Multi-resolution context: history beyond the recent window compressed by learned pooling into summary tokens (14 days fits in ~4–8k tokens).

**Backbone:** plain decoder-only transformer, 100–200M parameters (data is unlimited; scale is free — and a clean scaling-law figure on an explicit prior is a bonus result). No per-modality towers, no cross-attention modules, no state, no filter. Boring on purpose: any architectural cleverness would contaminate the data-distribution claim.

**Head:** autoregressive distributional decoding of the target stream — 9 quantiles for point metrics, sampled trajectories for CRPS. K, ensemble size, refinement passes are inference-time dials.

**Test-time adaptation (permitted by protocol decision):** optional LoRA gradient steps minimizing self-supervised next-window prediction over the prompt's own history — touches only data the protocol already provides. Reported always as a *separate arm*, compute-matched against pure in-context ensembles.

**Machinery choices considered and rejected (decision hygiene):** any bespoke fusion/filtering mechanism (would contaminate the claim; that axis belongs to StateCast); pixel-space simulation-to-real for satellite (frozen-encoder latent interface instead); training a learned world model to generate the synthetic data (circular — the point is the prior is *written*, inspectable, knockout-able); retrieval over real plants at test time (changes the zero-shot definition; belongs to the TS-RAG baselines).

---

## 3. The experiment that carries the paper: real vs simulated experience

**Data-distribution twins.** One architecture, one compute budget, three training arms:

| Arm | Training data | What it isolates |
|---|---|---|
| **A — real-only** | 75 real training plants (the standard supervised regime; = what every board model does) | the value of real experience |
| **B — prior-only** | generative program only; zero real samples ever | the value of written experience; the pure zero-shot claim |
| **C — prior → real** | B's checkpoint fine-tuned on the 75 plants | whether the two compose |

Crossed with evaluation axes: intra-plant held-out time, cross-plant (primary), cross-region matrix, test-time sensor dropout, cadence shift (30-min→15-min), horizon extrapolation. **Predictions, stated up front:** A edges intra-plant; B beats A cross-plant (the headline if true); C dominates everywhere. Every outcome publishes: B ≥ A cross-plant ⇒ "written priors beat 75 real plants for transfer"; A sweeps ⇒ a rigorous negative on the PFN hypothesis outside tabular data — rarer and more citable than another fusion delta.

**Prior-attribution study (the experiment only this formulation affords):** retrain B with components of the generative program knocked out — no cloud advection (static skies), no entity randomization (one fixed transfer function), no asynchrony randomization (regular grids), no sensor-noise diversity. Measure which knockout destroys which evaluation axis. Result: a *causal map from training-distribution ingredients to generalization capabilities* — the figure that makes this a science paper.

**Inference-compute study:** accuracy/CRPS vs test-time FLOPs for: sample count K ∈ {1, 8, 32}; prompt-resampling ensembles; refinement passes; TTT steps ∈ {0, 10, 100}. All on the same checkpoints, log-compute-matched. The in-context-vs-in-weights exchange rate is a standalone contribution.

---

## 4. Training

**Objective:** quantile (pinball) loss + trajectory-level CRPS on sampled decodings; ramp-weighted exactly as the S6 protocol metric demands. No auxiliary losses — the prior does the work the mechanism papers do with regularizers (graceful degradation is *trained in* by sampling sensor suites with modalities absent; coasting is trained in by sampling blackout gaps).

**Curriculum on the realism dial:** early training samples the prior wide (fast randomized dynamics, coarse rendering); later training narrows toward the measured statistics of the real corpus (cadences 15/30-min, UK/US geometry ranges, V-JEPA latent statistics matched via G4). The dial position is logged, making "how much realism was needed" itself reportable.

**Infrastructure notes:** generation is procedural and CPU-cheap → generated on the fly in the dataloader, seeded, nothing stored (1 TB quota untouched). V-JEPA encoding of synthetic videos is the one GPU-side generation cost; amortized by rendering short clips at low resolution and caching per-epoch latents in RAM.

---

## 5. Testbed — multimodal PV forecasting (= the thesis chain, unchanged)

Everything carries over verbatim: dataset of record (106 sites, uk_pv 30-min + goes_pvdaq 15-min, satellite crops via frozen V-JEPA, weather covariates, NWP with lead-time trust), 75/16/15 cross-plant protocol, intra-site held-out-time first-class, cross-region matrix, SS/R²/CRPS + S6 ramps, full baseline board (iTransformer 0.552 to beat, chronos2_oracle 0.474 ceiling, chronos2_zs as the zero-shot reference definition, TiRex-2, Toto-2.0, Solar-VLM, FusionSF, SolarCrossFormer). Arm C is the leaderboard entry; arm B is the zero-shot entry.

Why PV is the right testbed *for this formulation specifically*: the entity transfer function is genuinely low-dimensional physics (capacity/orientation/clipping) — exactly the situation where a written prior can cover entity-space; the latent world (cloud fields) has known dynamics classes; and the board provides 30 real-data-trained competitors to run arm-A-vs-B against.

*(A second testbed is deliberately NOT in v1 scope: the prior-attribution + Bayes-gap + inference-scaling instruments carry the AI claim without one. If review pressure demands it, the generative program generalizes to station-network forecasting (USHCN/air quality) with a different world level and the same sensor/entity levels — the modular prior makes the port cheap. Decision recorded, cost acknowledged.)*

---

## 6. Gates, ablations, risks

**Go/no-go (ordered, all cheap-first):**
- **G1 — prior-realism gate (~week 2):** train a small (≤30M) arm-B model, evaluate zero-shot on a uk_pv subset. Must beat persistence + climatology and land within a declared margin of chronos2_zs. Fails ⇒ the prior is fantasy; fix the generative program or kill the project before real compute.
- **G2 — Bayes-gap gate (synthetic-only):** on low-dimensional prior tasks, predictive must approach the particle-filter posterior as training scales. Fails ⇒ the architecture can't amortize this inference class; adjust context length/scale before blaming the prior.
- **G3 — prior-diversity gate:** performance must improve from 10⁴ → 10⁶ sampled tasks. Flat curve ⇒ the prior has low effective diversity; widen before scaling up.
- **G4 — latent-interface gate:** encoded synthetic satellite latents must overlap real V-JEPA latent statistics (MMD + cross-domain linear-probe transfer). Fails ⇒ switch fallback: fit a lightweight generative model to real latent statistics and sample the "satellite" stream directly in latent space.

**Ablations:** A1 arms A/B/C (the paper) → A2 prior-component knockouts × evaluation axes → A3 inference-compute axes, compute-matched (incl. TTT-vs-ICL) → A4 realism-dial curriculum on/off → A5 model scale ∈ {30M, 100M, 200M} on the fixed prior (scaling law) → A6 context length / summary-token compression → A7 entity-parameter probes (synthetic exact; real vs metadata) → A8 misspecification meter vs per-plant skill (does the meter predict where the model fails?) → A9 modality-absence in prior on/off (does trained-in degradation beat post-hoc dropout robustness?).

**Risks:**

| Risk | Mitigation |
|---|---|
| Prior misspecification: real plants outside the written distribution | G1 week-2 kill; misspecification meter localizes it; realism-dial curriculum; arm C closes residual gap with real data |
| Satellite sim2real gap survives the frozen encoder | G4 gate + declared latent-space-prior fallback (no pixels rendered at all) |
| "It's just data augmentation" review | fenced by the instruments no augmentation paper has: Bayes-gap measurement, prior-knockout causal attribution, seeded reproducible prior |
| "ForecastPFN/TabPFN-TS already did PFN for time series" | both univariate/featurized, no multimodality, no asynchrony, no entity structure, no attribution study — stated in the fence table |
| Arm B loses to arm A everywhere | negative result on the PFN hypothesis under unusually clean controls; arm C still contends on the board; thesis chain unharmed |
| 100–200M training cost on Leonardo | data generated on-the-fly (no I/O bottleneck, no storage); G1/G2/G3 all run at ≤30M before any large run is approved |
| TTT arm muddies the zero-shot claim | reported strictly as a separate arm under the declared "history-only gradients" definition; pure arm B keeps the forward-pass-only claim intact |
| Long multimodal context blows memory | summary-token compression (A6); 14 days ≈ 4–8k tokens after pooling — LLM-standard territory |

---

## 7. Prior-art fence

| Nearest work | Why SIBYL is outside it |
|---|---|
| **TabPFN / PFN line** (Nature 2025) | i.i.d. tabular rows; no time, no streams, no modalities, no entities; posterior over labels, not trajectories |
| **ForecastPFN, TabPFN-TS** | univariate or feature-hacked; no multimodal sensors, no asynchrony, no entity transfer functions, no attribution or Bayes-gap instruments |
| **Chronos (KernelSynth), TiRex synthetic mixes** | synthetic as anonymous augmentation for univariate TSFMs; no prior semantics, no posterior claim, no knockout science, not multimodal |
| **Simulation-based inference (SBI/NPE)** | amortized posteriors over *simulator parameters* for scientists; here posterior-*predictive* over multimodal trajectories at leaderboard scale, with entity structure |
| **StateCast (internal sibling)** | mechanism paper (learned assimilation, real-data-trained); SIBYL holds architecture fixed and moves the training distribution — orthogonal axis, mutually citable, potentially combinable later |
| **TSFMs (Chronos-2, TiRex-2, Toto-2.0)** | real/mixed-corpus pretraining, covariates-as-inputs, unimodal-native; baselines here |
| **World-model line (JEPA, GenCast)** | learn the world model from data; SIBYL writes it and distills it — white-box, knockout-able |

**Exact claim (every noun load-bearing):** *first prior-fitted network for multimodal asynchronous entity-structured forecasting — a generic transformer trained on an explicit, seeded, procedurally-written world prior that performs amortized posterior-predictive inference in-context, transfers zero-shot to unseen real plants, scales with inference-time compute, and whose generalization is causally attributed to named components of its training distribution.*

---

## 8. Foundation-model substrate analysis: introducing Chronos-2 / TabFM

**Chronos-2 as warm-start init of the decoder — rejected for the main arms.** Tempting (temporal competence free; Chronos-2's own KernelSynth lineage is philosophically adjacent), but it destroys the two properties the paper sells: arm B's pretraining distribution stops being seeded and reproducible (Chronos-2's corpus is not ours to publish), and the posterior-predictive interpretation dies — the model's predictive is no longer the posterior under *the written prior*, so the Bayes-gap audit (contribution 5) becomes uninterpretable. A prior-fitted network with a contaminated prior is just a fine-tuned TSFM.

**Chronos-2 as a fourth arm — adopted.** Arm **B★: Chronos-2-init → prior training**, evaluated alongside A/B/C. It answers a question the clean arms cannot: *once an explicit prior exists, does real-corpus pretraining still buy anything?* B★ ≈ B strengthens the "written experience suffices" headline; B★ > B quantifies exactly what reality-pretrained weights add. Either way it converts the contamination into evidence. B★ is also the engineering fallback if the scratch 100–200M run underfits the compute budget.

**TabFM as a PFN-family baseline — adopted, never a component.** TabPFN-TS-style featurization (calendar features, target lags, weather/NWP values as columns; one ICL regression per horizon step) gives the tabular-PFN answer to this task in days of work. Its structural blindness — no native asynchrony, no satellite stream, no entity factorization beyond feature engineering — is precisely the gap §1 claims; a measured TabFM row on the board turns that claim from assertion into number. Folding TabFM *into* SIBYL (e.g., as the readout) would re-featurize the event-token stream and forfeit the asynchrony story.

**Net rule (v2 lesson from the decision record lineage):** a foundation model *beside* SIBYL is evidence; a foundation model *inside* SIBYL is a contaminant — **for the scientific arms**. For raw generalization-per-compute, B★ is the *recommended board entry*: Chronos-2 already paid for generic temporal competence, so the prior only has to teach multimodality and entity structure — and B★ can be run as LoRA/PEFT over the frozen backbone instead of full training, collapsing adaptation cost to a fraction of the scratch run while inheriting the TSFM's cross-domain generalization. Purity arms (A/B/C) carry the paper; B★ carries the leaderboard.

---

## 9. Decision record

**Why not the other 2026-fancy directions (considered and rejected):**
- **Reasoning-RL forecaster** (multimodal LLM + GRPO on forecast reward): maximal fashion, minimal science — the reward is a scalar metric, the "reasoning" unverifiable, compute enormous, and reviewers increasingly hostile to RL-stunt papers outside language.
- **Pure test-time-training paper:** clean but small; TTT survives here as a first-class inference-compute axis (A3) rather than the thesis.
- **Any new fusion/filtering mechanism:** that axis is occupied internally by StateCast and externally by a crowded taxonomy; adding a third mechanism paper dilutes both.

**Relation to the lineage:** MMTSFM v1–v4 asked *which mechanism fuses*; StateCast asked *which inductive bias structures computation*; SIBYL asks *which experience distribution to train on* — the third orthogonal axis, and the only one where cross-entity transfer is in-distribution by construction. The two proposals are competitors for the same thesis slot but not for the same claim; a future combination (StateCast architecture trained on the SIBYL prior) is an obvious follow-up, deliberately out of scope.

**Next actions (ordered):** (1) generative program v0 — world/sensor/entity levels, seeded, dataloader-native; (2) G4 latent-interface check on a first batch of rendered clips vs cached real V-JEPA latents (zero model training); (3) G1 small-model prior-realism gate on uk_pv subset; (4) G2/G3 on synthetic; (5) open `exp/sibyl` branch after G1 passes; (6) large arm-B run, then arms A and C under matched compute.
