# GEPPETTO — entities as weights: generative weight-space forecasting

**(Proposal v1, AI-first framing. Orthogonal to MMTSFM (fusion mechanism), [STATECAST.md](STATECAST.md) (inductive bias), and [SIBYL.md](SIBYL.md) (training distribution): this one moves the question to *parameter space*. Written 2026-07-15, grounded in a last-30-days trend scan.)**

Named for the puppet-maker: GEPPETTO does not *condition* one forecaster on a new plant — it *carves* a new forecaster for it, by generating the weights.

---

## 0. Paper identity

**Research question (the paper's first sentence):** *where should entity-specific knowledge live — in activations (in-context conditioning) or in parameters (generated weights) — and which choice generalizes better to unseen entities?*

**Model (the vehicle):** GEPPETTO — a two-level system. Level 1: a population of small per-plant expert forecasters (multimodal, LoRA-parameterized over a shared frozen backbone, ~10⁵–10⁶ trainable dims each). Level 2: a **weight-space learner** — a permutation-safe weight encoder producing hyper-representations, a history encoder over the plant's 14-day multimodal stream, a CLIP-style contrastive alignment between the two (data ↔ weights), and a conditional flow-matching generator in the weight latent. **Zero-shot cross-plant = weight generation:** feed the unseen plant's history, sample its expert's LoRA weights, forecast with a specialist that never existed a second ago. No gradient step (optional streaming polish permitted under the declared history-only-gradients protocol).

**Contributions (AI-community form):**
1. **Weight generation as the transfer mechanism for forecasting:** first system where cross-entity generalization is performed by *synthesizing the model* rather than conditioning it — weight-space learning (a 2026 growth area: dedicated ICLR workshop; survey arXiv 2603.10090; WeightCLIP, arXiv 2607.03551, July 2026) applied to a live forecasting leaderboard instead of model-zoo analysis.
2. **The in-context vs in-weights controlled study:** same backbone, same data, three matched arms — (a) history as prompt (ICL), (b) history → generated LoRA (GEPPETTO), (c) history → gradient-fine-tuned LoRA (the oracle every arm chases). Gap (c)−(a) is the known ICL ceiling; gap (c)−(b) measures generation fidelity; (b) vs (a) is the paper's headline. Publishable in every direction.
3. **Weight space made legible:** the expert population is entity-structured, so the weight latent can be audited against physics — capacity/orientation directions in weight space, interpolation between two plants' weights yielding plausible intermediate plants, metadata probes on hyper-representations. Weight-space papers audit against architecture; we audit against *reality*.
4. **A released entity-structured model zoo:** thousands of trained expert checkpoints with aligned histories and metadata — the benchmark artifact the weight-space community currently lacks (their zoos are CIFAR classifiers).

**Target venue:** ICLR/NeurIPS main track. **Thesis compatibility:** PV protocol, splits, metrics, board unchanged; arm (c) doubles as the strongest intra-plant entry.

---

## Intuition — plain-language walkthrough

### I.1 The core idea

Every model on the current board is one network asked to be all 106 plants at once. GEPPETTO asks the heretical question: why share? Train a *specialist* per plant — small, cheap, unashamedly overfitted to that plant's hardware and microclimate — and specialists beat generalists at home; that is the oldest result in ML. The problem was always the 76th plant: no data (protocol gives 14 days), no time to train. Weight-space learning dissolves it: if you have a population of (history, weights) pairs, "which weights fit this history" is just another supervised mapping — learn it, and the new plant's expert is a *single forward pass of a generator*.

### I.2 Why LoRA kills the classic objection

Naive weight-space learning fights permutation symmetry (the same function has combinatorially many weight encodings, so the mapping history→weights is ill-posed). We sidestep it structurally: every expert is a **LoRA delta on one shared frozen multimodal backbone, from one shared init**. All experts live in the *same aligned subspace* — low-dimensional, symmetry-broken, and generation-friendly. This is also what makes a 10⁵–10⁶-dim generation target tractable for flow matching where full 30M-weight generation would be fantasy.

### I.3 Growing the zoo (the real work)

75 real plants is not a population; weight-space learning wants thousands of points. Three multipliers: (i) window-level experts — one expert per plant×season×history-slice, each a distinct valid (history, weights) pair; (ii) sensor-suite variants — experts trained with modalities masked, teaching the generator how weights change when a sensor is absent; (iii) synthetic plants from a procedural generator (the SIBYL prior, reused as a zoo factory — the proposals share tooling without sharing claims). Zoo training is embarrassingly parallel and each expert is minutes of GPU: population scale is cheap by construction.

### I.4 A forecast, step by step

Unseen UK plant, origin 12:00: (1) encode its 14-day multimodal history → z_h; (2) sample LoRA weights from the flow-matching generator conditioned on z_h (K samples ⇒ an *ensemble of models*, not an ensemble of trajectories — weight-space uncertainty, a qualitatively different and rarely-measured kind); (3) attach each LoRA to the shared backbone, forecast, aggregate quantiles. Total cost: one generator pass + K small forward passes. The deployed artifact per plant is ~1 MB of LoRA — fleet-scale personalization at negligible cost, a deployment story no board model has.

### I.5 Why it should win both boards

**Intra-plant:** arm (c) — the oracle-fine-tuned expert — is the natural intra-plant champion (specialists at home), and it is a *first-class entry*, not just a diagnostic. **Cross-plant:** the board's diagnosed failure is bias/scale transfer; a generated specialist gets its bias/scale *in its weights*, fitted by a generator that has seen thousands of (history, weights) pairs. And the (c)−(b) gap is a measured, improvable quantity — the paper knows exactly how far it is from its own ceiling, which no conditioning-based model can say.

---

## 1. Why this is an AI paper in 2026

| 2026 AI agenda | Where GEPPETTO sits |
|---|---|
| **Weight-space learning** (ICLR workshop; survey 2603.10090; WeightCLIP 2607.03551; weight-space world models 2605.06298) | The field analyzes static zoos of toy classifiers; GEPPETTO gives it a live task, an entity-structured zoo, and a leaderboard consequence |
| **Generative models everywhere** (flow matching as default) | Generation target = *models*, conditioned on multimodal streams — an unoccupied conditioning regime |
| **ICL theory** | The in-context vs in-weights arms are the cleanest available instantiation of "prompt vs parameters" outside language |
| **Personalization / fleet ML** | Per-entity specialists at ~1 MB each, synthesized not trained — the deployment economics section writes itself |
| **TSFM wave** | The shared backbone can *be* a frozen TSFM (TimesFM 2.5 ships LoRA/PEFT support as of April 2026 — the tooling is mainstream); TSFMs become substrate, not competitor |

**What is honestly NOT novel:** hypernetworks (2016–), LoRA generation, weight diffusion (p-diff) exist. Novelty located in: the *entity-structured multimodal instantiation*, the matched three-arm study, the physics-auditable weight latent, and the zoo release — stated in paragraph 2, per house rule.

---

## 2. Model

```
TRAINING (build + learn the zoo)                      INFERENCE (unseen plant)
────────────────────────────────                      ────────────────────────
 75 plants × windows × sensor-suites                   14-day multimodal history
  (+ synthetic plants, optional)                                │
        │  train per-expert (minutes each)                      ▼
        ▼                                              [history encoder = backbone, pooled]
 ┌─────────────────────────────┐                                │ z_h
 │ EXPERT ZOO (~2–5k experts)  │                                ▼
 │ each = LoRA Δ on the SAME   │                       ┌─────────────────────┐
 │ frozen init  ⇒ one aligned, │                       │ CFM WEIGHT GENERATOR│──► K weight samples
 │ symmetry-free subspace      │                       │  p(w | z_h)         │    (model ensemble)
 └──────┬──────────────────────┘                       └─────────────────────┘
        │ (history, weights) pairs                              │  LoRA Δ  (~1 MB/plant)
        ▼                                                       ▼
 [weight encoder] ──► z_w                              ┌──────────────────────┐
 [history encoder] ─► z_h                              │ FROZEN CHRONOS-2     │
        │                                              │ (PRIMARY substrate)  │──► quantile forecast
        ▼                                              │ weather/NWP → covar  │    (aggregate over K)
 InfoNCE align z_h ↔ z_w (WeightCLIP-style)            │ V-JEPA sat → frozen  │
 + CFM loss on generator                               │ shared proj adapter  │
                                                       └──────────────────────┘
                                        (native 30M event-token backbone = control substrate, A3′)
```

- **Shared backbone (frozen; PRIMARY = Chronos-2):** the LoRA substrate is a frozen pretrained TSFM — Chronos-2, whose pipeline this repo already ships — giving every expert a pretrained-generalization floor at zero training cost. Weather/NWP enter through the covariates channel; cached V-JEPA satellite latents through a projection adapter trained once and frozen (shared across all experts, so weight space stays LoRA-only). A native ~30M event-token backbone trained on the 75 plants is retained as the *control substrate* (A3′) to show the weight-space claims are not Chronos-2-specific and to keep the satellite stream first-class.
- **Experts:** LoRA (rank r ∈ {4–16}) over attention + readout, per plant×window×sensor-suite. Trained with the standard ramp-weighted pinball loss.
- **Weight encoder:** set/graph encoder over LoRA factors → hyper-representation z_w (256-d). **History encoder:** the backbone itself, pooled → z_h. **Alignment:** InfoNCE between z_h and z_w over the zoo (WeightCLIP-style, data↔weights).
- **Generator:** conditional flow matching in LoRA-parameter space (or in a learned weight-autoencoder latent if raw dims resist), conditioned on z_h. K weight samples = model ensemble.
- **Rejected machinery (decision hygiene):** full-weight generation (symmetry + dimension); hypernetwork direct regression history→weights (mode-averaging over valid experts; the generative formulation keeps multimodality); per-plant gradient TTT as the *main* mechanism (that is PROTEUS's axis; here it is only the (c) oracle and an optional polish).

---

## 3. The experiment that carries the paper

**Three matched arms, one backbone, one data budget:**

| Arm | Entity knowledge lives in | Cost at new plant |
|---|---|---|
| (a) ICL | activations (history as prompt) | 1 forward pass |
| (b) GEPPETTO | generated parameters | 1 generator pass + K forwards |
| (c) Oracle FT | gradient-fitted parameters | minutes of GPU (protocol-legal: history-only gradients) |

Crossed with: intra-plant, cross-plant, cross-region, sensor dropout, thin-history (2-day) prompts. **Predictions up front:** (c) wins intra; (b) closes most of the (c)−(a) gap cross-plant; under thin history (b) degrades gracefully (generator has seen thin-history pairs) while (c) overfits. Any ordering publishes — including "(a) matches (b) everywhere," which would be a sharp negative result on weight generation's practical value.

**Audit instruments:** metadata probes on z_w (capacity, orientation, tilt readable from weights alone); weight interpolation plants (decode midpoints, check monotone transfer-function morphing); zoo heritage analysis (window-experts of one plant cluster). These figures exist for no attention/conditioning model.

---

## 4. Training

Stage 1: shared backbone on 75 plants (standard). Stage 2: zoo — ~2–5k experts (plants × windows × sensor-suites × optional synthetic plants), parallel, cheap. Stage 3: weight-space learner (alignment + generator) on the zoo; held-out plants' experts excluded by construction. Losses: pinball (experts), InfoNCE (alignment), CFM (generator).

---

## 5. Testbed — PV (= thesis chain, unchanged)

Dataset of record (106 sites, uk_pv + goes_pvdaq), 75/16/15 cross-plant protocol, SS/R²/CRPS + S6 ramps, full board (iTransformer 0.552 to beat, chronos2_oracle 0.474 ceiling reference). Arm (c) = intra-plant entry; arm (b) = zero-shot cross-plant entry.

---

## 6. Gates, ablations, risks

**Go/no-go (ordered, cheap-first):**
- **G1 — headroom gate (~week 1):** per-plant oracle LoRA experts must beat the shared backbone intra-plant by a declared margin. If specialists don't beat the generalist here, the whole premise dies for this domain — cheapest possible kill.
- **G2 — weight-latent legibility:** plant metadata linearly decodable from z_w. Fails ⇒ LoRA subspace too entangled; raise rank / restrict LoRA placement.
- **G3 — generation fidelity:** on held-in plants, generated weights reach ≥95% of that plant's oracle-expert skill. Fails ⇒ generator or alignment problem, fix before any cross-plant claim.

**Ablations:** A1 arms a/b/c (the paper) → A2 zoo size scaling (10² → 10³·⁵ experts; flat curve = diversity problem) → A3 LoRA rank / placement → A4 K weight-samples (weight-space uncertainty calibration) → A5 alignment on/off (generator conditioned on raw z_h) → A6 synthetic-plant zoo fraction → A7 generated-then-polished (b + few TTT steps, compute-matched vs c) → A8 interpolation/probe audits.

**Risks:**

| Risk | Mitigation |
|---|---|
| Specialists don't beat generalist (G1 fails) | week-1 kill; negative result cheap; backbone work reusable by sibling proposals |
| Zoo too small/homogeneous for generation | window/sensor-suite multipliers + synthetic plants; A2 measures directly |
| Generated weights unstable outliers | generate in weight-AE latent; reject-sample by self-consistency (predicted-vs-actual history fit) |
| "Hypernetworks exist" review | fenced: generative not regressive, entity-structured zoo, matched 3-arm evidence, physics audit — none in prior weight-space work |
| Permutation symmetry | dissolved by construction (shared-init LoRA subspace) |

---

## 7. Prior-art fence

| Nearest work | Why GEPPETTO is outside it |
|---|---|
| **Weight-space learning line** (survey 2603.10090, WeightCLIP, hyper-representations) | analyzes/generates toy-classifier zoos; no forecasting task, no entity structure, no downstream leaderboard, no multimodal conditioning |
| **Hypernetworks / p-diff weight diffusion** | unconditional or class-conditioned; here condition = multimodal sensor history, target = entity expert, evaluation = live board |
| **LoRA-personalization line** (per-user adapters) | adapters *trained* per entity; here they are *generated* zero-shot — the (b) vs (c) arms measure exactly this distinction |
| **StateCast ψ / SIBYL ICL (internal siblings)** | both keep one network and condition it (state readout / prompt); GEPPETTO changes the *artifact* per entity — parameter-space axis, orthogonal and mutually citable |
| **TSFMs** | substrate (frozen backbone option) and baselines, not competitors |

**Exact claim:** *first generative weight-space method for multimodal forecasting — per-entity expert weights synthesized in one pass from the entity's sensor history, evaluated by a matched in-context vs in-weights vs oracle study on a public leaderboard, with a physics-auditable weight latent and a released entity-structured model zoo.*

---

## 8. Foundation-model substrate analysis: introducing Chronos-2 / TabFM

**Chronos-2 as the shared frozen LoRA substrate — adopted as the PRIMARY substrate (design of record; native backbone demoted to control).** GEPPETTO's claims are explicitly backbone-agnostic (the contribution is weight *generation*, not the backbone), so the shared frozen init can be Chronos-2 instead of the native 30M event-token model. What it buys: a higher expert ceiling (stronger substrate ⇒ stronger specialists ⇒ more headroom for G1); a community-legible artifact (a zoo of thousands of Chronos-2 LoRAs with aligned histories is directly useful to the HF ecosystem, unlike LoRAs on a bespoke backbone); mainstream tooling (TimesFM 2.5 shipping LoRA/PEFT in April 2026 shows the pattern is standard); and near-zero integration cost — the chronos2 pipeline already lives in this repo (`src/mmtsfm/models/chronos2/`). What it costs: Chronos-2 is unimodal-native, so weather/NWP ride the covariates channel and V-JEPA satellite latents need a projection adapter — trained once, *frozen, and shared across all experts* (a per-expert adapter would inflate the weight-space dimension and re-entangle it); LoRA over a 100M+ backbone is a larger generation target, so placement is restricted (attention only) and G3 (generation fidelity) must be re-passed per substrate.

**The substrate-specific risk:** on Chronos-2, expert skill may come from *covariate exploitation* (learning to read weather better) rather than transfer-function specialization (learning the plant). The G2 metadata probe arbitrates: if capacity/orientation stop being decodable from z_w on the Chronos-2 zoo, the weight latent has lost its physics and the native backbone remains the paper's spine.

**TabFM as a probe baseline — adopted, small.** TabFM regressing plant metadata (capacity, orientation) from featurized 14-day histories via ICL is a days-of-work sanity bound on what z_h must contain: if tabular ICL recovers the metadata, the history encoder has no excuse; if it can't, the metadata probes get a meaningful floor. Not a component — TabFM has no role in weight space.

**The adaptation-cost story (why the FM substrate is more than plumbing here):** frozen Chronos-2 + generated LoRA is *zero-gradient specialization of a TSFM* — the new plant pays one generator forward pass, no fine-tuning, and inherits the backbone's pretrained generalization as its floor. That is the cheapest specialization mechanism on the whole adaptation spectrum (cheaper than LoRA fine-tuning, which is arm (c); cheaper than any TTT), and it makes GEPPETTO-on-Chronos-2 the arm most likely to win cross-plant per unit of deployment compute.

**Net rule:** frozen Chronos-2 is the primary substrate; the native 30M backbone runs as the control arm (A3′); report both zoos. But the v2 lesson stands — if the paper collapses to "a LoRA zoo on Chronos-2," it is glue on a frozen backbone and will be commoditized within weeks; the weight-space learner and the three-arm study are the paper, the substrate is plumbing.

---

## 9. Decision record

Rejected sibling framings: direct hypernetwork regression (mode-averaging, kills the generative-uncertainty story); full-weight generation (symmetry, dimension); making TTT the mechanism (PROTEUS owns that axis). Axis taxonomy across the proposal family: MMTSFM = fusion mechanism; StateCast = computation structure; SIBYL = training distribution; **GEPPETTO = parameter space**; SCOUT = interaction policy; PROTEUS = adaptation over time.

**Next actions (ordered):** (1) G1 headroom gate — LoRA experts on 10 plants vs shared backbone (uses existing training code, days); (2) zoo pipeline (plants × windows), embarrassingly parallel; (3) G2 probe on first 500 experts; (4) weight-AE + CFM generator v0; (5) open `exp/geppetto` after G1+G3 pass.
