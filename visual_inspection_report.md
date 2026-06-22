# PV Forecast Visual Inspection Report

This report presents a visual inspection of the PV power forecasts across the baselines at site **10793** (UK PV test dataset).

## Architectural Clusters
The models are grouped into four architectural categories:
1. **Classical / Naive**: `persistence`, `smart_persistence`, `climatology_hourly`, `seasonal_naive`, `lightgbm`.
2. **Deep Time-Series**: `mlp`, `dlinear`, `patchtst`, `itransformer`, `tft`.
3. **TS Foundation Models**: `timesfm_zs`, `ttm_zs`, `ttm_ft`, `ts_rag_orig`, `cross_rag_orig`.
4. **Multimodal / Vision**: `aurora`, `unicast`, `sunset`, `solar_vlm`.

For each category, a folder is created under `plots/` containing 5 sample windows showing predictions from only that group's models.

## Best Model per Cluster (Site 10793)
To compare architectures, the plotting tool automatically loads the plant-specific metrics from the results JSON files and identifies the best model from each category:
* **Classical / Naive**: `smart_persistence` (NMAE: `0.1630`)
* **Deep Time-Series**: `patchtst` (NMAE: `0.0886`)
* **TS Foundation**: `cross_rag_orig` (NMAE: `0.0772`)
* **Multimodal / Vision**: `sunset` (NMAE: `0.0963`)

---

## Overall Architecture Comparison
The following plots show the predictions of the **best model from each cluster** compared against ground truth and `smart_persistence`:

### Window 52
![Window 52](file:///Users/marcomorandin/Desktop/thesis-with-context/baselines/plots/comparison/plot_site_10793_w52.png)

---

### Window 229
![Window 229](file:///Users/marcomorandin/Desktop/thesis-with-context/baselines/plots/comparison/plot_site_10793_w229.png)

---

### Window 502
![Window 502](file:///Users/marcomorandin/Desktop/thesis-with-context/baselines/plots/comparison/plot_site_10793_w502.png)

---

### Window 564
![Window 564](file:///Users/marcomorandin/Desktop/thesis-with-context/baselines/plots/comparison/plot_site_10793_w564.png)

---

### Window 1310
![Window 1310](file:///Users/marcomorandin/Desktop/thesis-with-context/baselines/plots/comparison/plot_site_10793_w1310.png)

---

## Directory Structure of Generated Plots
The generated plots are organized locally under the `baselines/plots/` directory:
```
baselines/plots/
├── classical_naive/      # 5 plots showing persistence, lightgbm, etc.
├── deep_ts/              # 5 plots showing MLP, DLinear, PatchTST, iTransformer, TFT
├── ts_foundation/        # 5 plots showing TimesFM, TTM, TS-RAG, Cross-RAG
├── multimodal_vision/    # 5 plots showing Aurora, Unicast, Sunset, Solar-VLM
└── comparison/           # 5 plots showing the best model of each category (cross_rag_orig, patchtst, sunset, smart_persistence)
```
