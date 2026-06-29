# Tested-Baselines Report — `uk_pv` cross-plant (S2)

**Status**: Living synthesis. Generated from the live results
([`baselines/results/ALL_RESULTS.md`](../../baselines/results/ALL_RESULTS.md),
`ALL_RESULTS.json`), the qualitative forecast plots and the per-model
actual-vs-predicted scatter under [`baselines/plots/`](../../baselines/plots/)
(`plots/scatter/` covers all 28 models incl. the fixed time_vlm). Companion to
[BASELINE_PROTOCOL.md](BASELINE_PROTOCOL.md) (fairness rules),
[BASELINE_COMPARISON.md](BASELINE_COMPARISON.md) (suite design) and
[BASELINE_RESULTS_UKPV.md](BASELINE_RESULTS_UKPV.md) (the superseded laptop-era snapshot).

**Purpose**: one place that says *what has actually been run, what it scored, what the
plots show, and what the next steps are* — so the PVTSFM target bar and the open gaps
are unambiguous.

> **How to read this doc (meeting version).** Start with the **TL;DR** below for the
> story in three numbers and the one decision we need to make. The **Discussion agenda**
> (before §6) lists the choices to settle together, each with a recommendation. §1–§5 are
> the supporting evidence; §6 is the proposed plan, open to re-prioritization.

---

## TL;DR — for our meeting

**What this is.** 28 forecasting baselines run end-to-end on the `uk_pv` cross-plant
benchmark (14 disjoint test plants, 14-day history → 6-hour horizon), scored by **Skill
Score (SS)** over Smart Persistence. This is the bar the thesis model (PVTSFM) must clear.

**Where we landed — the three numbers that matter:**

1. **The bar to beat is SS ≈ 0.55.** Best supervised model = iTransformer **0.552**; best
   multimodal = time_vlm **0.540**. PVTSFM must clear ~0.55 to justify itself — *not*
   Smart Persistence and *not* the published multimodal SOTA.
2. **Vision now looks worth it.** time_vlm is essentially tied with the best supervised
   model **and** has the best ramp-regime accuracy of the whole suite (Ramp NRMSE 0.172).
   This is the first concrete evidence that satellite vision converts into a
   cloud-transition advantage — the core thesis hypothesis.
3. **Retrieval beats fine-tuning.** Frozen-backbone RAG (≈0.477) is far ahead of
   fine-tuning the same backbone (chronos2_ft 0.331). Adaptation is mandatory — zero-shot
   foundation models alone do not even clear the classical-ML tier.

**Honesty caveats (read before over-claiming):**

- **Single dataset.** Everything here is `uk_pv`. No cross-dataset (`goes_pvdaq`, the US
  set) evidence yet → generalization is *unproven*, not disproven.
- **Two results are harness-limited, not architecture verdicts.** time_vlm runs on
  non-aligned eval windows (needs an aligned re-run before we headline it); unicast is
  genuinely weak but for window-alignment reasons (§3.3).
- **Oracle rows are upper bounds, not competitors** (they use privileged information).

**The decision we need to make together.** Is the current evidence strong enough to commit
the thesis to a **multimodal + retrieval** PVTSFM design, or do we first close the two
credibility gaps (cross-dataset run + aligned time_vlm) before locking the direction? The
Discussion agenda frames this and three related choices.

---

## 0. Scope of this report

* **Protocol**: disjoint cross-plant, `uk_pv` only (14 test plants, 69/15/14 split,
  30-min cadence). History **T = 14 days (672 steps)**, horizon **H = 6 h (12 steps)**.
* **Reference**: Smart Persistence — Skill Score `SS = 1 − NRMSE/NRMSE_SP ≡ 0`.
  Every `SS` below is *within its own scenario tag* (each model carries a matched
  `smart_persistence_<tag>` reference; see the per-tag tables in `ALL_RESULTS.md` §4.4).
* **Not yet covered**: `goes_pvdaq` (downloaded, not run — split still lists 2
  `bad_site_flag` sites, must be reconciled first), low-history robustness sweep,
  long-horizon S4, day-ahead decay curve.

---

## 1. Headline leaderboard (sorted by SS ↓)

Pulled from `ALL_RESULTS.md`. ⚠ flags = read §3 before quoting the number.

| Rank | Tier | Model | NMAE ↓ | NRMSE ↓ | **SS ↑** | CRPS ↓ | Ramp NRMSE ↓ | Note |
|---:|---|---|---|---|---|---|---|---|
| 1 | T2 | **iTransformer** | 0.0699 | 0.1032 | **0.552** | — | 0.1769 | best overall, supervised |
| 2 | **T5** | **time_vlm** | 0.0692 | 0.1061 | **0.540** | — | **0.1720** | **best multimodal; best ramp** (fixed, §3.0) |
| 3 | T3 | chronos2_oracle_ft | 0.0808 | 0.1142 | 0.504 | 0.0630 | 0.1824 | ⚠ oracle |
| 4 | T4 | ts_rag | 0.0705 | 0.1203 | 0.478 | — | 0.2004 | retrieval-augmented |
| 5 | T4 | cross_rag | 0.0726 | 0.1206 | 0.477 | — | 0.1969 | retrieval-augmented |
| 6 | T3 | chronos2_oracle | 0.0817 | 0.1213 | 0.474 | 0.0635 | 0.1915 | ⚠ oracle |
| 7 | T2 | PatchTST | 0.0886 | 0.1249 | 0.458 | — | 0.1888 | supervised |
| 8 | T6 | solar_vlm | 0.0955 | 0.1283 | 0.443 | — | 0.1849 | multimodal (PV-specialized) |
| 9 | T2 | TFT | 0.0889 | 0.1330 | 0.423 | 0.0689 | 0.2001 | quantile |
| 10 | T2 | MLP | 0.0958 | 0.1352 | 0.413 | — | 0.1981 | supervised |
| 11 | T1 | LightGBM | 0.1000 | 0.1419 | 0.384 | 0.0768 | 0.2024 | classical ML |
| 12 | T4 | cora | 0.1025 | 0.1444 | 0.374 | 0.0816 | 0.2021 | frozen-TSFM adapt |
| 13 | T3 | ttm_ft | 0.1029 | 0.1465 | 0.364 | — | 0.2062 | tiny TSFM, fine-tuned |
| 14 | T6 | crossvivit | 0.1112 | 0.1500 | 0.349 | — | 0.2100 | multimodal |
| 15 | T1 | TabPFN | 0.1076 | 0.1524 | 0.339 | 0.0815 | 0.2050 | tabular FM |
| 16 | T3 | chronos2_ft | 0.1120 | 0.1543 | 0.331 | 0.0855 | 0.2115 | our backbone, FT |
| 17 | T2 | DLinear | 0.1131 | 0.1556 | 0.325 | — | 0.2092 | linear check |
| 18 | T3 | tirex_zs | 0.1145 | 0.1642 | 0.287 | 0.0892 | 0.2233 | zero-shot |
| 19 | T3 | timesfm_zs | 0.1172 | 0.1680 | 0.271 | 0.0923 | 0.2319 | zero-shot |
| 20 | T0 | climatology_hourly | 0.1353 | 0.1766 | 0.234 | — | 0.2037 | reference |
| 21 | T5 | aurora | 0.1280 | 0.1769 | 0.232 | — | 0.2516 | multimodal (weak) |
| 22 | T6 | sunset | 0.1384 | 0.1806 | 0.216 | — | 0.2177 | multimodal |
| 23 | T3 | chronos2_zs | 0.1376 | 0.1873 | 0.187 | 0.1072 | 0.2335 | our backbone, ZS |
| 24 | T5 | unicast | 0.1433 | 0.2025 | 0.121 | — | 0.2814 | ⚠ genuinely weak (§3.3) |
| 25 | T0 | seasonal_naive | 0.1419 | 0.2058 | 0.107 | — | 0.2575 | reference |
| 26 | T5 | visionts_pp | 0.1690 | 0.2266 | 0.017 | — | 0.2515 | ≈ persistence |
| 27 | T0 | persistence | 0.1643 | 0.2272 | 0.014 | — | 0.2990 | floor |
| — | T0 | smart_persistence | 0.1593 | 0.2304 | 0.000 | — | 0.2806 | **reference (SS≡0)** |
| ✗ | T3 | ttm_zs | 0.1704 | 0.2490 | **−0.081** | — | 0.3414 | worse than SP |

> **Ramp NRMSE** now populated for every row (previously `—` for T4–T6) — see §3.2.
> For T4–T6 it is computed on each harness's native windows (proxy mask), so compare
> ramp *within tier / by rank*, not pooled across tiers.

---

## 2. Reading by tier

* **T0 reference** — Smart Persistence is the bar. Hourly climatology (SS 0.234) is
  the strongest naive, confirming a strong diurnal-seasonality signal that any learned
  model must clear. Plain persistence and seasonal-naive barely beat zero.
* **T1 classical ML** — LightGBM (0.384) > TabPFN (0.339). Both beat every zero-shot
  TSFM. Tabular models on flattened `(Y, X_cov)` remain a strong, cheap baseline.
* **T2 supervised deep TS** — **the tier to beat.** iTransformer (0.552) is the best
  model in the entire suite; PatchTST (0.458), TFT (0.423), MLP (0.413) all rank high.
  Channel-attention over covariates is clearly the strongest *learned* signal so far.
  DLinear (0.325) is the lowest T2 — the linear check no longer wins, which is healthy.
* **T3 TS foundation** — **zero-shot is weak**: chronos2_zs 0.187, timesfm_zs 0.271,
  tirex_zs 0.287, and ttm_zs is *below* the reference (−0.081). Fine-tuning recovers a
  lot (chronos2_ft 0.331, ttm_ft 0.364). The **oracle** variants (0.474 / 0.504) are
  *upper bounds, not deployable* — see §3.
* **T4 frozen-TSFM adaptation** — ts_rag / cross_rag (≈0.477) are the best non-oracle
  TSFM-based results and beat fine-tuning by a wide margin; retrieval is the most
  effective frozen-backbone adaptation tested. cora (0.374) trails the RAG pair.
* **T5/T6 multimodal** — after the time_vlm fix (§3.0), **time_vlm (0.540) is the best
  multimodal model and is essentially tied with iTransformer for best-overall**, and it
  has the **lowest Ramp NRMSE of the entire suite (0.172)** — vision paying off exactly
  on the cloud-transition regime. solar_vlm (0.443) is the best PV-specialized
  multimodal, then crossvivit (0.349). aurora (0.232), sunset (0.216), unicast (0.121)
  remain weak; visionts_pp ≈ persistence.

---

## 3. Caveats that change interpretation

0. **time_vlm fixed (was SS −1.64, now 0.540).** The old catastrophic row was a *stale
   result*: `import_predictions.py` had re-saved the inverse-scaled (de-standardized)
   predictions to `results/predictions/time_vlm_*_pred.npz`, but the `time_vlm_s2_ukpv.json`
   was never regenerated, so the table still showed the pre-inverse NRMSE 0.61. Re-scoring
   from the corrected npz (no retraining) gives NMAE 0.069 / NRMSE 0.106 / **SS 0.540**.
   The other 8 externals re-score *bit-identical* to their published rows, so the fix is
   isolated to time_vlm.
1. **Oracle ≠ deployable.** `chronos2_oracle` / `chronos2_oracle_ft` consume
   privileged information (oracle context/clear-sky selection) and are an *upper bound*,
   not a competitor. The honest TSFM number is `chronos2_ft` = **0.331**. The
   oracle→ft gap (0.504 − 0.331 = **0.173 SS**) measures the headroom available from
   better context/conditioning — a direct motivation for the PVTSFM design.
2. **Ramp now covered (proxy).** Ramp NMAE/NRMSE are now computed for T4–T6 by re-scoring
   the saved predictions (`import_predictions.py` → `ramp_mask_from_true`). The threshold
   is each site's top-decile |Δtrue| on its *native* windows (daylight proxy `true>0`,
   step 0 excluded for lack of dumped history) — **not bit-aligned** with the T0–T3 ramp
   subset, so read ramp within-tier / by rank. CRPS is still only available for
   quantile-native models (LightGBM, TFT, TabPFN, the chronos2 family).
3. **`unicast` is genuinely weak, not a stale bug.** Its npz re-scores bit-identical to
   its JSON (SS 0.121). Diagnosis from the saved predictions: amplitude under-prediction
   (`pred ≈ 0.61·true + 0.063`, corr 0.61) plus a late-day skew in its native eval
   windows (per-horizon mean of `true` falls monotonically 0.22→0.08, i.e. forecast
   origins cluster near the daily peak). Both are model/harness properties, **not
   recoverable from the computed predictions** without retraining or an illegitimate
   affine recalibration on the targets — flagged for P0 below, *not* fixed here.
4. **Single dataset.** Everything here is `uk_pv`. No cross-dataset (`goes_pvdaq`)
   evidence yet, so generalization claims are unsupported.

---

## 4. What the plots show (qualitative — test site 10793)

Two plot families, both on test site 10793. **§4.1 forecast traces** (per-window
overlays) read the *temporal* behaviour; **§4.2 actual-vs-predicted scatter** reads the
*calibration* behaviour pooled over all of the site's windows. ⚠ The NMAE / Ramp NRMSE
printed in each scatter title are **site-10793-only** — they are *not* the pooled
14-plant numbers in §1 and will not match them (e.g. iTransformer site NMAE 0.065 vs
pooled 0.070; time_vlm site 0.082 vs pooled 0.069). Read scatter ranks within this
section, not against the leaderboard.

### 4.1 Forecast traces (per-window overlays)

Plots live in `baselines/plots/<group>/plot_site_10793_w{52,229,502,564,1310}.png`, one
file per window. Five groups: `classical_naive`, `deep_ts`, `ts_foundation`,
`multimodal_vision`, and `comparison` (best-of-each-cluster). All overlay each model's
12-step forecast against True Future for the same window. Reading window 52 (a sharp
cloud-driven ramp peaking near step 9):

* **classical_naive** — smart_persistence and climatology track the morning ramp shape
  best; lightgbm/tabpfn systematically *under-shoot* the peak (smooth, capped curves).
* **deep_ts** — iTransformer / TFT follow the rising limb closely but, like all models,
  miss the abrupt step-9 spike and over-shoot the step-11 drop. Confirms T2 strength is
  *trend tracking*, not transient/ramp capture.
* **ts_foundation** — dense spread; ZS models are visibly biased low early in the
  horizon; the RAG/oracle traces hug the truth better than raw ZS.
* **multimodal_vision** — **solar_vlm tracks the truth most closely** of the plotted
  vision models on this window (rides the ramp up to step 9); crossvivit captures an
  early bump then collapses; **unicast outputs a flat ≈0 line on this window** (its
  late-day window skew, §3.3); sunset is roughly flat/noisy. *(time_vlm is not in the
  plotted groups — its fix, §3.0, post-dates these PNGs; regenerate plots to include it.)*
* **comparison** — best-of-clusters confirms the ordering: no model captures the
  step-9 transient; smart_persistence is a deceptively strong shape-matcher; learned
  models trade peak accuracy for smoothness.

**Takeaway from the plots**: the unmodeled signal is the *cloud-induced transient /
ramp*. The newly-computed ramp metrics (§3.2) now quantify it: **time_vlm (0.172) and
iTransformer (0.177) own the best Ramp NRMSE**, confirming vision *can* convert into a
ramp advantage. Regenerate the plot set to surface the fixed time_vlm trace.

### 4.2 Actual-vs-predicted scatter (calibration, site 10793)

One file per model, `baselines/plots/scatter/scatter_<model>.png`: every (true, pred)
pair for site 10793 against the `y = x` perfect-forecast line. This view exposes
*calibration* (conditional-mean bias, variance, value quantization) that the SS table and
the trace plots hide. Sorted by **site NMAE ↓** (site-10793-only — see §4 warning):

| Model | Tier | site NMAE | site Ramp NRMSE | Scatter signature |
|---|---|---:|---:|---|
| iTransformer | T2 | 0.065 | 0.190 | tightest cloud on the diagonal; mild high-end saturation (caps ≈0.8) |
| chronos2_oracle(_ft) | T3 | 0.071 | 0.181 | tightest of the TSFM family — confirms the oracle upper bound (§3.1) |
| patchtst | T2 | 0.080 | 0.192 | clean diagonal; vertical spray at true≈0 |
| tft | T2 | 0.081 | 0.205 | diagonal but wider spread than PatchTST |
| lightgbm | T1 | 0.081 | 0.196 | smooth, peak-capped (≈0.7) — under-shoots high output |
| **time_vlm** | T5 | 0.082 | **0.169** | **best site ramp**; under-predicts peaks but tightest ramp regime |
| mlp | T2 | 0.083 | 0.196 | diagonal, moderate spread |
| tabpfn | T1 | 0.083 | 0.190 | like LightGBM, peak-capped |
| ttm_ft | T3 | 0.083 | 0.197 | FT recovers a near-diagonal cloud |
| cora | T4 | 0.084 | 0.189 | strong true≈0 vertical band, otherwise diagonal |
| dlinear | T2 | 0.087 | 0.190 | noisier diagonal |
| chronos2_ft | T3 | 0.088 | 0.189 | FT pulls the ZS cloud back onto the diagonal |
| tirex_zs | T3 | 0.090 | 0.202 | under-prediction + true≈0 band |
| cross_rag | T4 | 0.091 | 0.189 | **vertical quantization stripes** — discrete retrieved-neighbour values |
| solar_vlm | T6 | 0.092 | 0.177 | systematic under-prediction (sits below diagonal) yet good ramp |
| ts_rag | T4 | 0.093 | 0.193 | same retrieval quantization stripes as cross_rag, milder |
| timesfm_zs | T3 | 0.094 | 0.205 | broad under-prediction |
| aurora | T5 | 0.098 | 0.215 | very high vertical variance — poorly calibrated |
| crossvivit | T6 | 0.106 | 0.212 | peak-capped under-prediction |
| climatology_hourly | T0 | 0.116 | 0.184 | **discretized horizontal bands** + hard ceiling ≈0.45 (can't reach high output) |
| chronos2_zs | T3 | 0.117 | 0.231 | strong under-prediction, predictions capped low |
| seasonal_naive | T0 | 0.121 | 0.252 | unstructured noise |
| unicast | T5 | 0.135 | 0.291 | wide, under-predicting cloud (consistent with §3.3) |
| ttm_zs | T3 | 0.137 | 0.289 | severe under-prediction — points hug the x-axis |
| sunset | T6 | 0.139 | 0.236 | predictions compressed into a low band — heavy under-prediction |
| visionts_pp | T5 | 0.145 | 0.233 | near-persistence: big true≈0 spray, little diagonal structure |
| persistence | T0 | 0.150 | 0.254 | unstructured noise around the diagonal |
| smart_persistence | T0 | 0.186 | 0.279 | largest true≈0 streak; *worst raw NMAE at this site* (see note) |

**What the scatter adds beyond the SS table:**

* **Universal regression-to-the-mean at high output.** Almost every model sits *below*
  `y = x` for high actual power: the conditional-mean forecast under-shoots bright/clear
  peaks. Worst in `climatology_hourly`, `chronos2_zs`, `ttm_zs`, `sunset`. This is the
  same peak-under-shoot the traces show (§4.1), now visible as a global bias.
* **true≈0 vertical spray.** Many models (chronos2, cora, ts_rag, time_vlm, visionts_pp,
  smart_persistence) emit non-zero power when actual ≈ 0 (night / heavy overcast) — a
  vertical band on the y-axis. A cheap deployable win is a daylight/zero gate.
* **Retrieval leaves a fingerprint.** `ts_rag` and especially `cross_rag` show discrete
  **vertical stripes** — the forecast collapses onto a finite set of retrieved-neighbour
  values. Visual confirmation that the RAG mechanism is doing nearest-neighbour copying,
  not smooth regression.
* **Climatology is a lookup table.** Horizontal banding + a ≈0.45 ceiling makes explicit
  why its SS (0.234) plateaus: it structurally cannot represent high-output windows.
* **time_vlm vs iTransformer, same site.** iTransformer has the tighter point cloud
  (site NMAE 0.065 < 0.082) but time_vlm owns the **best site Ramp NRMSE (0.169)** — the
  vision signal buys ramp calibration, not point accuracy, exactly as the pooled story
  (§2 T5/T6, §5.2) claims.
* **Note — SP is a weak *raw* reference at site 10793.** Here smart_persistence has the
  highest raw NMAE (0.186), worse than plain persistence (0.150). SS stays meaningful
  (it is relative within each tag), but do not read the scatter NMAE as a skill ranking —
  it is raw per-site error, and this site happens to be one where the clear-sky
  normalization hurts SP.

---

## 5. Implications for PVTSFM

1. **The bar to beat is SS ≈ 0.55 (iTransformer 0.552 / time_vlm 0.540)**, not Smart
   Persistence and not the multimodal SOTA. A frozen multimodal stack below ~0.55 does
   **not** yet support the core research claim.
2. **The multimodal claim is now *plausible* (was unsupported).** After the time_vlm fix,
   the best multimodal (time_vlm 0.540) is essentially tied with the best supervised
   model (iTransformer 0.552) and **leads the suite on Ramp NRMSE (0.172)** — the first
   evidence that vision converts into a real ramp/cloud-transition advantage. Caveat:
   time_vlm runs on native, non-aligned eval windows, so confirm with an aligned re-run
   before headlining it.
3. **Retrieval is the most effective frozen-backbone lever tested** (ts_rag/cross_rag
   ≈ 0.477 vs chronos2_ft 0.331). The PVTSFM design should treat RAG-style conditioning
   as a strong component, not an afterthought.
4. **Zero-shot TSFMs are insufficient alone** — adaptation (FT / RAG / fusion) is
   mandatory to clear even the classical-ML tier.

---

## 5b. Discussion agenda — what to decide together

Four decisions for the meeting. Each carries my recommendation; the supporting evidence
is in §1–§5.

**D1 — Is the multimodal direction proven enough to commit to?**
* *For:* time_vlm 0.540 ≈ iTransformer 0.552, and it owns the best ramp (§5.2).
* *Risk:* it runs on non-aligned windows and on a single dataset, so the tie may not
  survive a fair re-run.
* *Recommendation:* commit **provisionally**, but gate the headline claim on (a) an
  aligned time_vlm re-run and (b) the first `goes_pvdaq` cross-dataset result. Decide
  together whether that gate blocks the thesis framing or is just a robustness footnote.

**D2 — Which adaptation lever does PVTSFM build on?**
* Options: retrieval / RAG (≈0.477) · fine-tuning (0.331) · deep vision fusion (the core
  hypothesis).
* *Recommendation:* treat **retrieval as a first-class component**, not an add-on — it is
  the strongest frozen-backbone lever measured. Open question: combine RAG *and* vision
  fusion, or test them as competing arms first?

**D3 — Breadth vs depth for the next sprint?**
* *Breadth:* run the full suite on `goes_pvdaq` → first cross-dataset generalization
  evidence.
* *Depth:* decompose the oracle gap (0.173 SS of headroom), low-history + long-horizon
  sweeps.
* *Recommendation:* **breadth first** — a cross-dataset number changes the thesis story
  more than another within-`uk_pv` ablation. Confirm priority together.

**D4 — How much to invest in harness fairness before moving on?**
* unicast / time_vlm window alignment, CRPS for T4–T6, plot regeneration, significance
  tests.
* *Recommendation:* do only what protects a headline claim now (aligned time_vlm, T4–T6
  significance); defer the rest. Decide what rigour level the committee will expect.

---

## 6. Proposed next steps (to confirm in the meeting)

*Prioritized plan below — open to re-ordering per the Discussion agenda (§5b).*

**P0 — unblock fair comparison**
* ✅ **Done:** `time_vlm` re-scored from corrected predictions (SS −1.64 → 0.540) and
  **ramp NMAE/NRMSE added to all T4–T6** (re-scored from saved npz; `ALL_RESULTS.md`
  regenerated). No retraining used.
* `unicast` (SS 0.121) is **not** a stale-result bug (§3.3) — it needs a *retrain /
  window-alignment*, not a re-score: align its eval windows to the canonical origins
  (remove the late-day skew) and check the amplitude under-prediction. Until then keep it
  flagged as harness-limited, not architecture evidence.
* ✅ **Partly done:** per-model actual-vs-predicted **scatter** plots regenerated for all
  28 models incl. the **fixed time_vlm** (`plots/scatter/`, see §4.2). Still pending: the
  per-window **trace** overlays (`multimodal_vision` / `comparison` groups) predate the
  time_vlm fix and should be regenerated to surface its trace.
* CRPS for T4–T6: dump quantile forecasts from the vendored harnesses where supported, so
  the probabilistic column is no longer reference-only.
* Reconcile the `goes_pvdaq` split (drop the 2 `bad_site_flag` sites → 8 usable) and run
  the full suite there for the first cross-dataset generalization evidence.

**P1 — quantify headroom & robustness**
* Decompose the **oracle gap** (chronos2_oracle_ft 0.504 vs chronos2_ft 0.331): what
  exactly does "oracle" supply, and how much is recoverable by learned conditioning?
* Run the low-history robustness sweep and the day-ahead decay curve
  (`DECAY_HORIZONS_HOURS = (1, 6, 24)`) — zero-shot FMs may close the gap at longer
  horizons where supervised models over-fit short transients.

**P2 — strengthen the multimodal case**
* Investigate why most vision baselines (sunset, aurora, visionts_pp) sit near the
  naive floor while solar_vlm clears 0.44 — fusion depth vs late fusion is the likely
  axis (the core PVTSFM hypothesis).
* Add seed replicates / significance tests (DM + block bootstrap) for the T4–T6 models
  to match the rigor already applied to T0–T2 in `BASELINE_RESULTS_UKPV.md` §3.

---

*Source of truth for numbers: `baselines/results/ALL_RESULTS.md` (regenerated by
`baselines/scripts/aggregate_all.py` each cluster sweep). Re-sync this report after the
next sweep — the live table moves.*
