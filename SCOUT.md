# SCOUT — forecasting as active perception: a learned acquisition policy with verifiable rewards

**(Proposal v1, AI-first framing. Orthogonal to MMTSFM (fusion mechanism), [STATECAST.md](STATECAST.md) (inductive bias), [SIBYL.md](SIBYL.md) (training distribution), and [GEPPETTO.md](GEPPETTO.md) (parameter space): this one moves the question to the *interaction policy* — not how to use observations, but which observations to buy. Written 2026-07-15, grounded in a last-30-days trend scan.)**

Named for the scout: sent ahead not to see everything, but to decide what is worth looking at.

---

## 0. Paper identity

**Research question (the paper's first sentence):** *given a fixed sensing/compute budget, is forecast skill better bought by a learned observation-acquisition policy than by any fixed sensing pipeline — and what is the value of each sensor, measured rather than assumed?*

**Model (the vehicle):** SCOUT — a base multimodal forecaster (deliberately swappable; the claim is about the policy, not the backbone) wrapped by an **acquisition policy** π. At each decision point π chooses from a costed action set: pull a satellite crop (at chosen resolution/extent), pull a NWP variable subset, extend the history window, densify the recent window, spend K more forecast samples, or *stop and answer*. Cost = declared per-action price (bandwidth/latency/FLOPs); reward = realized reduction in forecast error — a **verifiable reward**, computed against ground truth during training, no proxy, no judge.

**Contributions (AI-community form):**
1. **Forecasting as budgeted active perception:** first formulation of multimodal PV forecasting where the sensing pipeline is a *decision variable*. Every model on the current board consumes a fixed input contract; SCOUT makes the contract itself learned — unifying "which sensor," "at what fidelity," and "how much test-time compute" into one budget.
2. **RL with genuinely verifiable rewards:** the 2026 RLVR wave runs on proxy verifiers; here the reward is the actual scoring metric against actual ground truth. A clean, unusually honest testbed for policy-learning-for-perception (lineage: RL acquisition policies for inverse problems, arXiv 2407.07794; non-greedy cost-tradeoff acquisition NOCTA, arXiv 2507.12412; budget-constrained station selection in environmental monitoring — none at multimodal forecasting-leaderboard scale).
3. **The accuracy–cost Pareto study:** matched-backbone comparison of learned policy vs all-sensors vs fixed heuristics vs random dropping, across the full budget axis. The claim is *dominance of the frontier*, not a single point — and the frontier is a result no fixed-input model can even plot.
4. **Value-of-information as science:** the trained policy is an instrument — VOI maps over conditions (when does the satellite pay? at what horizon does NWP overtake it? which plants never need the dense window?) quantify, for the first time on this testbed, the marginal worth of each modality. This answers the oracle-gap question (+0.14 SS for future weather, known from the board) *causally* and per-condition.

**Target venue:** ICLR/NeurIPS main track (agentic planning + inference-time efficiency are named ICLR 2026 themes). **Thesis compatibility:** full-budget SCOUT is a standard board entry; the budget axis is additive to the protocol, not a change to it.

---

## Intuition — plain-language walkthrough

### I.1 The core idea

Every proposal so far — and every model on the board — answers "here are all the sensors, forecast." Real deployments never look like that: satellite crops cost bandwidth, NWP variables cost API calls, dense history costs latency, ensembles cost FLOPs. The interesting question a fielded forecaster faces is *what to look at next*. A weather-desk human does this instinctively: clear stable morning → glance at the power curve, done; convective afternoon → pull the satellite loop, check two NWP runs. SCOUT learns that instinct. The forecast model stays a forecast model; the intelligence being studied is the *policy around it*.

### I.2 Why this is not a systems paper

Because the deliverable is not a cheaper pipeline — it is three pieces of science: (i) whether sequential, state-dependent acquisition beats any static contract at matched cost (an exploration/exploitation question, testable); (ii) the measured VOI structure of a real multimodal domain (the satellite is worth ~nothing at night and ~everything during morning cloud fronts — SCOUT puts numbers and confidence intervals on folklore); (iii) a clean study of RL-with-verifiable-reward on a task where the verifier is physics, not another model. Robustness falls out for free: a missing sensor is just an action whose cost became infinite — the policy re-plans, where fixed-input models break.

### I.3 A forecast, step by step

Unseen plant, origin 12:00, budget B: (1) free context first — power history and solar geometry (cost 0 by declaration); base model produces a draft forecast *with uncertainty*; (2) π reads the draft's uncertainty profile and the cheap context, and buys the action with the highest predicted error-reduction per unit cost — say, the recent satellite window at half resolution; (3) forecast updates; repeat until marginal predicted VOI < marginal cost or B exhausted; (4) answer. Uncertain situations soak up budget; easy ones exit in one step. **Adaptive computation, but over *sensors*, not just layers.**

### I.4 Why it should win both boards

At **full budget** SCOUT is the base model with everything — parity by construction, entered on the board as usual (any sibling backbone slots in; the policy is backbone-agnostic and that generality is ablation A6). The wins are: (a) *every constrained regime* — and the protocol's own sensor-dropout stress tests are constrained regimes, where the board's fixed-input models degrade unpredictably while SCOUT re-plans; (b) **cross-plant**, where the policy transfers because it reads state (uncertainty, sky condition), never plant identity — and thin-history/cold-start plants are, again, just expensive-information regimes it was trained for.

---

## 1. Why this is an AI paper in 2026

| 2026 AI agenda | Where SCOUT sits |
|---|---|
| **Agentic planning** (named ICLR 2026 theme) | An agent whose tools are sensors and whose environment is the atmosphere — long-horizon value-of-information, no LLM cosplay |
| **Test-time compute / inference-time efficiency** (named ICLR 2026 theme) | Generalizes "how much compute" to "which information": one budget, one frontier |
| **RL with verifiable rewards** | Reward = ground-truth metric; the rare RLVR setting with a perfect verifier |
| **Active sensing / acquisition** (NOCTA 2507.12412; RL acquisition 2407.07794; monitoring surveys) | That line stops at toy/medical/station-selection tasks with unimodal costs; SCOUT: heterogeneous multimodal actions, entity transfer, public leaderboard |
| **Reliable ML** | Graceful degradation as *policy behavior* (re-planning) rather than architectural hope |

**What is honestly NOT novel:** active perception is 40 years old; adaptive computation and acquisition-under-budget both have literatures. Novelty located in: the multimodal-forecasting instantiation with heterogeneous real sensor costs, the frontier-dominance evidence at leaderboard scale, VOI-as-science on a physically auditable domain, and entity transfer of the policy.

---

## 2. Model

```
                    free context (cost 0): power history + solar geometry
                                        │
                                        ▼
                        ┌───────────────────────────────┐
              ┌────────►│ FROZEN CHRONOS-2 (PRIMARY)    │──► draft forecast
              │         │ zero-shot base forecaster;    │    + uncertainty profile
              │         │ weather/NWP via covariates    │      (quantile spread)
              │         │ (swappable backbone — A6)     │
              │         └───────────────────────────────┘
              │                         │
   new observation                      ▼
   (re-forecast per acquisition;
    recompute FLOPs priced
    into the action costs)    ┌───────────────────────────────┐
              │               │ ACQUISITION POLICY π          │
              │               │ inputs: uncertainty profile,  │
              │               │ cheap context, budget left,   │
              │               │ actions already taken         │
              │               └───────────────┬───────────────┘
              │                               │ argmax predicted ΔError / cost
              │        ┌──────────────────────┼──────────────────────────┐
              │        ▼            ▼         ▼            ▼             ▼
              │   [satellite    [NWP vars  [history    [+K ensemble   [STOP]──► final
              │    crop res×     × lead     extend/     samples]               quantile
              │    extent×win]   range]     densify]                           forecast
              │        │            │         │            │
              │        └──── each action has a DECLARED cost (bandwidth/FLOPs) ────┐
              └────────────────────────────────────────────────────────────────────┘
                       loop until: marginal VOI < marginal cost, or budget B spent

   TRAINING SIGNAL: reward = realized error reduction vs ground truth (verifiable)
                    (i) amortized VOI regression → greedy floor
                    (ii) RL / GRPO on episode return → non-greedy sequencing
```

- **Base forecaster (PRIMARY = frozen Chronos-2):** the base model is a frozen zero-shot TSFM — nothing below the policy trains. Weather/NWP ride the covariates channel; calibrated quantiles come out of the box; the repo's existing chronos2 pipeline makes integration days, not weeks. Chronos-2 has no incremental KV-append, so each purchased observation triggers a full re-forecast — that recompute is *priced into the declared action costs* rather than engineered away (deployments pay re-inference too; the frontier stays honest). The event-token decoder shared with the sibling proposals is the A6 swap backbone (it does support KV append, giving the frontier a cheap-conditioning comparison point).
- **Action set (costed, declared):** satellite crop {resolution × extent × window length}, NWP {variable subset × lead range}, history {extension × densification}, ensemble {+K samples}, STOP. Costs from measured bandwidth/FLOPs, published with the paper.
- **Policy π:** compact network over (draft-forecast uncertainty profile, cheap context, remaining budget, actions already taken). Two training modes: (i) **amortized VOI** — regress each action's realized error-reduction, act greedily (stable, the floor); (ii) **RL** — policy-gradient/GRPO on episode return = final-error reduction minus λ·cost, non-greedy (NOCTA-style sequencing where early cheap looks inform later expensive ones). (ii) vs (i) is itself a headline ablation: *does non-greedy sequencing matter for sensing?*
- **Rejected machinery (decision hygiene):** LLM-agent tool-calling wrapper (adds tokens, subtracts rigor); learned costs (costs are measured facts; learning them contaminates the frontier claim); acquisition inside the backbone via per-layer gating (entangles policy with mechanism — the separation is what makes A6 portability an experiment).

---

## 3. The experiment that carries the paper

**Frontier protocol:** for budgets B ∈ {0, ¼, ½, 1, 2}× the standard contract's cost, compare: SCOUT-RL, SCOUT-greedy-VOI, all-sensors-truncated, expert heuristic (daylight-gated satellite + always-NWP), random acquisition, power-only floor. Same backbone everywhere. **Predictions up front:** parity at B=1 vs all-sensors (information-equivalent); strict dominance for B<1 (the claim); at B>2 the frontier saturates at the oracle gap — and *where* it saturates measures how much of the +0.14 SS future-weather oracle headroom is purchasable.

Crossed with the protocol's stress axes: sensor dropout (= infinite-cost action — SCOUT's home turf), cross-plant zero-shot (policy fixed, no per-plant tuning), cadence shift, thin history.

**VOI instruments (Figures 2–4):** acquisition heat-maps over (time-of-day × sky condition × horizon); per-plant modality budgets (which plants "need" the satellite); marginal-value curves per modality. Unreproducible by any fixed-input board model.

---

## 4. Training

Stage A: base forecaster with **randomized observation subsets** (so it is well-defined under any acquisition state — the same masking trick every sibling proposal uses, here load-bearing). Stage B: roll acquisition episodes on training plants, log realized error-reductions → amortized VOI regressor. Stage C: RL fine-tune π (reward verifiable by construction; short episodes ⇒ stable). Policy never sees plant IDs.

---

## 5. Testbed — PV (= thesis chain, unchanged)

Dataset of record, 75/16/15 protocol, SS/R²/CRPS + S6 ramps, full board. Board entry = SCOUT at B=1 (and the base model all-sensors, as its own control). The budget-frontier table is an *added* results section; nothing in the existing protocol moves. PV is the right domain: real heterogeneous per-modality costs (satellite imagery is genuinely the expensive stream), a known oracle information budget to saturate against, and physically checkable VOI (satellite value should track cloud dynamics — an audit, not a hope).

---

## 6. Gates, ablations, risks

**Go/no-go (ordered, cheap-first):**
- **G1 — VOI-exists gate (~week 1, zero new model code):** using an *existing* trained baseline, measure error deltas from ablating each modality per condition. If deltas are uniform across conditions (no state-dependence), a policy has nothing to exploit — kill before any RL.
- **G2 — greedy-suffices probe:** amortized-VOI greedy policy on frozen backbone must beat random and heuristic acquisition at B=½. Fails ⇒ the uncertainty signal doesn't predict VOI; fix calibration first.
- **G3 — RL-adds-value gate:** SCOUT-RL must beat SCOUT-greedy somewhere on the frontier, else ship the greedy paper (still publishable; smaller).

**Ablations:** A1 frontier protocol (the paper) → A2 greedy vs non-greedy (RL) → A3 action-set granularity (coarse binary sensors vs fidelity-graded) → A4 uncertainty input off (policy blind to draft confidence) → A5 cost-vector sensitivity (FLOPs- vs bandwidth-priced) → A6 backbone swap (policy portability) → A7 cross-plant policy transfer vs per-plant-tuned oracle policy → A8 VOI maps vs physics (satellite VOI ↔ measured cloud variability).

**Risks:**

| Risk | Mitigation |
|---|---|
| No state-dependent VOI in the data (G1) | week-1 kill with zero model code; the G1 measurement is itself a small publishable analysis |
| RL unstable / not better than greedy | greedy-VOI floor is the fallback paper; G3 decides honestly |
| "Systems paper" desk-reject smell | claims framed as exploration/VOI science + verifiable-reward RL study; frontier + VOI figures carry it; costs published as measured facts |
| Base-model dependence | A6 dual-backbone; the policy, not the backbone, is the contribution |
| Full-budget parity read as "no win" | stated as prediction up front; the frontier and stress axes are the win condition, and the board's own dropout stress is a constrained regime |

---

## 7. Prior-art fence

| Nearest work | Why SCOUT is outside it |
|---|---|
| **Acquisition-under-budget line** (NOCTA, RL acquisition for inverse problems, PiCSRL station selection) | unimodal or homogeneous costs, toy/medical scale, no entity transfer, no public leaderboard; SCOUT: heterogeneous sensor actions, cross-plant policy transfer, frontier at board scale |
| **Adaptive computation / early-exit / test-time-compute scaling** | budget over *internal* compute only; SCOUT unifies compute with external information in one costed action space |
| **Sensor-selection in energy forecasting** | offline, static feature selection; SCOUT is sequential, state-dependent, per-decision |
| **LLM tool-use agents** | learned *when* to call tools but reward-hacked proxies; here the verifier is ground truth physics |
| **Internal siblings** | StateCast/SIBYL/GEPPETTO all consume a fixed contract; SCOUT is complementary — any of them can be its backbone (A6), which is a feature of the axis taxonomy, not an overlap |

**Exact claim:** *first budgeted active-perception formulation of multimodal forecasting — a backbone-agnostic acquisition policy trained with verifiable rewards that dominates fixed sensing pipelines across the accuracy–cost frontier, transfers zero-shot across entities, and yields the first measured value-of-information maps for a public multimodal forecasting benchmark.*

---

## 8. Foundation-model substrate analysis: introducing Chronos-2 / TabFM

**Chronos-2 as the base forecaster — adopted, and it is the cheapest full SCOUT that exists.** SCOUT is backbone-agnostic by design (A6), and a frozen zero-shot TSFM is the limiting case that makes the adaptation-cost story sharpest: *nothing is trained except the policy.* Chronos-2 supplies pretrained cross-domain generalization and calibrated quantiles out of the box (chronos2_zs is already a board row, and the pipeline already lives in `src/mmtsfm/models/chronos2/`); weather/NWP enter through its covariates channel; the acquisition policy is a small network trained on logged error-reductions. Total gradient budget of the whole system ≈ the policy — days, not weeks. Cross-plant generalization comes bundled: the backbone is zero-shot by construction and the policy reads state (uncertainty, sky condition, budget), never plant identity, so the *entire system* transfers to an unseen plant with zero adaptation cost.

**The real cost it introduces:** Chronos-2 has no incremental KV-append across acquisition steps — every purchased observation forces a full re-forecast. This is not a dealbreaker; it is a *price change*: the recompute FLOPs go into the declared action-cost vector, which actually makes the frontier more honest (real deployments pay re-inference too). The frontier claim is unchanged; only the price table moves. Second cost: the VOI predictor leans on the backbone's uncertainty calibration — G2 already gates on exactly this, so the risk is measured before RL is attempted.

**TabFM as the VOI estimator — a genuinely interesting zero-training variant.** Stage B logs (state features, action, realized Δerror) rows — which is a *tabular dataset*. Hand those rows to TabFM as ICL context and predicted VOI becomes a zero-shot tabular regression: no trained value head at all. If TabFM-VOI matches the trained amortized regressor (new ablation A2′), the greedy tier of SCOUT becomes fully **trainingless** — frozen TSFM + frozen TabFM + greedy selection — a system with zero gradient steps end-to-end that still adapts its sensing per situation. That row on the frontier table is a headline-adjacent result about how far pretrained generalists go when composed.

**Net rule:** the FM substrate doesn't dilute SCOUT — it *is* the strongest version of its thesis: generalization rented from pretrained models, intelligence added as a near-free decision layer. Keep one trained-backbone arm (A6) to show the policy also composes with specialists; keep the v2 lesson in view — the policy and the frontier are the paper, not the wrapper.

---

## 9. Decision record

Rejected framings: LLM-agent wrapper (fashion, no rigor); making acquisition an internal gating mechanism (collapses into the fusion-mechanism axis MMTSFM already occupies); pure compute-scaling without sensors (SIBYL already carries inference-compute curves). Axis taxonomy: MMTSFM = fusion mechanism; StateCast = computation structure; SIBYL = training distribution; GEPPETTO = parameter space; **SCOUT = interaction policy**; PROTEUS = adaptation over time.

**Next actions (ordered):** (1) G1 VOI-existence measurement on an existing baseline's per-condition ablation deltas — zero new model code, days; (2) cost vector measurement + publication draft of the action-price table; (3) masked-subset backbone training (shared with siblings); (4) G2 greedy-VOI probe; (5) open `exp/scout` after G1+G2 pass.
