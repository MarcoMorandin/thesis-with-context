"""Deep EDA for the standardized PV + satellite-image dataset.

Produces a markdown report (dataset_exploration_report.md) and plots/ directory.
Designed to answer, before modeling:
  1. Is the data physically coherent (power vs irradiance, capacity bounds)?
  2. Which sites/rows are broken and need fixing or exclusion?
  3. What drives production (feature importance) and how similar are sites
     (relevant for cross-site zero-shot transfer)?
  4. Are the images aligned, readable and informative?
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_DIR = "/leonardo_scratch/fast/IscrC_MTSFM/data"
NUMERIC_PATH = os.path.join(DATA_DIR, "dataset_all.parquet")
IMAGES_H5 = os.path.join(DATA_DIR, "images_all.h5")
OUTPUT_DIR = "."


# Frames live in images_all.h5 (per-site group <dataset>_<site>), addressed by
# the canonical image_h5_index pointer — no PNG tree. Lazy, single open.
_H5_HANDLE = {"f": None}


def _h5():
    import h5py

    if _H5_HANDLE["f"] is None:
        _H5_HANDLE["f"] = h5py.File(IMAGES_H5, "r")
    return _H5_HANDLE["f"]


def open_frame(row):
    """Return a PIL.Image for a row's frame from images_all.h5, or None."""
    try:
        grp = _h5()[f"{row.dataset}_{row.site_id}"]
        arr = np.asarray(grp["images"][int(row.image_h5_index)])
        return Image.fromarray(arr)          # mode 'L' (HxW) or 'RGB' (HxWx3)
    except Exception:
        return None


def frame_timestamp(row):
    """ISO timestamp string stored in the h5 group for this row's frame, or ''."""
    try:
        grp = _h5()[f"{row.dataset}_{row.site_id}"]
        ts = grp["timestamps"][int(row.image_h5_index)]
        return ts.decode() if isinstance(ts, bytes) else str(ts)
    except Exception:
        return ""
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RNG = 42
WEATHER_COLS = [
    "temperature_2m", "shortwave_radiation", "direct_radiation",
    "diffuse_radiation", "direct_normal_irradiance", "cloudcover",
    "windspeed_10m", "precipitation",
]

report_lines = []
issues = []  # auto-collected data-quality findings


def add_md(text=""):
    report_lines.append(text)


def add_issue(severity, text):
    issues.append((severity, text))


def save_fig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    add_md(f"![{name}](plots/{name})")
    add_md()


# ---------------------------------------------------------------- load

print("Loading data...")
df = pd.read_parquet(NUMERIC_PATH)
df = df.sort_values(["site_id", "timestamp_utc"]).reset_index(drop=True)
df["norm_power"] = df["power_w"] / df["installed_power_w"]
df["hour"] = df["timestamp_utc"].dt.hour
df["month"] = df["timestamp_utc"].dt.month
df["date"] = df["timestamp_utc"].dt.date
# approximate local solar hour from longitude (15 deg per hour)
df["solar_hour"] = (df["hour"] + df["longitude"] / 15.0) % 24

add_md("# Deep Exploratory Data Analysis Report")
add_md()
add_md("Cross-site PV power dataset with satellite/sky image augmentation. "
       "This report covers data integrity, physical coherence of production, "
       "temporal structure, feature importance, cross-site similarity "
       "(zero-shot transfer relevance) and image quality.")
add_md()

# ---------------------------------------------------------------- 1. overview

add_md("## 1. Dataset Overview")
add_md()
add_md(f"- **Total records:** {len(df):,}")
add_md(f"- **Columns:** {len(df.columns)} (raw: 18 + derived norm_power/hour/month/date/solar_hour)")
add_md(f"- **Time range:** {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")
add_md(f"- **Unique sites:** {df['site_id'].nunique()}")
add_md()

per_ds = []
for ds, g in df.groupby("dataset"):
    one_site = g[g.site_id == g.site_id.iloc[0]].sort_values("timestamp_utc")
    freq = one_site["timestamp_utc"].diff().mode().iloc[0]
    per_ds.append({
        "dataset": ds,
        "rows": len(g),
        "sites": g.site_id.nunique(),
        "sampling": str(freq),
        "first_ts": g.timestamp_utc.min(),
        "last_ts": g.timestamp_utc.max(),
        "median_rows_per_site": g.groupby("site_id").size().median(),
        "capacity_range_W": f"{g.installed_power_w.min():,.0f} – {g.installed_power_w.max():,.0f}",
    })
per_ds_df = pd.DataFrame(per_ds).set_index("dataset")
add_md(per_ds_df.to_markdown())
add_md()

for ds, row in per_ds_df.iterrows():
    span_days = (row.last_ts - row.first_ts).days
    if span_days < 90:
        add_issue("warn", f"`{ds}` covers only {span_days} days "
                          f"({row.first_ts:%Y-%m-%d} → {row.last_ts:%Y-%m-%d}) — "
                          "no seasonal diversity; cross-dataset evaluation on it tests "
                          "a single season only.")

add_md("### Missing values")
add_md()
missing = df.isnull().sum()
missing_df = pd.DataFrame({
    "missing": missing,
    "%": (missing / len(df) * 100).round(3),
})
missing_df = missing_df[missing_df.missing > 0]
if missing_df.empty:
    add_md("No missing values in any column.")
else:
    add_md(missing_df.to_markdown())
    for col, row in missing_df.iterrows():
        add_issue("warn", f"`{col}` has {row['missing']:,} missing values ({row['%']}%)")
add_md()

add_md("### Descriptive statistics (numeric columns)")
add_md()
desc_cols = ["power_w", "installed_power_w", "norm_power"] + WEATHER_COLS
add_md(df[desc_cols].describe().round(3).T.to_markdown())
add_md()

# ---------------------------------------------------------------- 2. integrity

print("Running integrity checks...")
add_md("## 2. Data Integrity & Physical Coherence Checks")
add_md()

checks = []

dup = df.duplicated(subset=["site_id", "timestamp_utc"]).sum()
checks.append(("Duplicate (site, timestamp) rows", dup, "error" if dup else None))

neg_power = (df.power_w < 0).sum()
checks.append(("Negative power values", neg_power, "warn" if neg_power else None))

over_cap = (df.power_w > df.installed_power_w).sum()
checks.append(("Power exceeding installed capacity", over_cap, "warn" if over_cap else None))

over_cap_5pct = (df.power_w > 1.05 * df.installed_power_w).sum()
checks.append(("Power exceeding capacity by >5%", over_cap_5pct, "error" if over_cap_5pct else None))

neg_rad = (df[["shortwave_radiation", "direct_radiation", "diffuse_radiation",
               "direct_normal_irradiance"]] < 0).sum().sum()
checks.append(("Negative radiation values", neg_rad, "error" if neg_rad else None))

nan_power = df.power_w.isna().sum()
checks.append(("NaN power values", nan_power, "warn" if nan_power else None))

zero_power = (df.power_w == 0).sum()
checks.append((f"Zero power rows ({zero_power / len(df):.1%} of data)", zero_power, None))

# suspicious zeros: zero power while irradiance is strong -> likely outage / metering fault
susp_zero = df[(df.power_w == 0) & (df.shortwave_radiation > 200)]
checks.append(("Zero power with shortwave_radiation > 200 W/m² (suspected outages)",
               len(susp_zero), "warn" if len(susp_zero) else None))

# production at night (no irradiance) -> metering noise
night_prod = df[(df.shortwave_radiation == 0) & (df.norm_power > 0.01)]
checks.append(("Power > 1% capacity with zero irradiance (night production)",
               len(night_prod), "warn" if len(night_prod) else None))

# capacity sanity: one value per site
cap_var = (df.groupby("site_id")["installed_power_w"].nunique() > 1).sum()
checks.append(("Sites with non-constant installed capacity", cap_var, "error" if cap_var else None))

# coordinates sanity: one location per site
loc_var = (df.groupby("site_id")[["latitude", "longitude"]].nunique() > 1).any(axis=1).sum()
checks.append(("Sites with non-constant coordinates", loc_var, "error" if loc_var else None))

checks_df = pd.DataFrame(checks, columns=["check", "count", "severity"])
add_md(checks_df[["check", "count"]].to_markdown(index=False))
add_md()
for _, row in checks_df.iterrows():
    if row.severity:
        add_issue(row.severity, f"{row['check']}: {row['count']:,} rows")

# stuck-sensor detection: same nonzero power repeated >= 6 consecutive steps in daytime
print("  stuck-sensor scan...")
d = df[df.shortwave_radiation > 50]
stuck_counts = {}
for site, s in d.groupby("site_id")["power_w"]:
    runs = (s != s.shift()).cumsum()
    run_len = s.groupby(runs).transform("size")
    stuck = int(((run_len >= 6) & (s > 0)).sum())
    if stuck:
        stuck_counts[site] = stuck
stuck_total = sum(stuck_counts.values())
add_md(f"**Stuck-sensor heuristic** (identical non-zero power for ≥6 consecutive daytime "
       f"steps): {stuck_total:,} rows across {len(stuck_counts)} sites.")
add_md()
if stuck_total > 0.005 * len(df):
    top_stuck = sorted(stuck_counts.items(), key=lambda x: -x[1])[:5]
    add_issue("warn", f"Stuck-sensor rows: {stuck_total:,} across {len(stuck_counts)} sites "
                      f"(top: {top_stuck})")

# timestamp gap / coverage analysis
print("  coverage analysis...")
cov_rows = []
for ds, g in df.groupby("dataset"):
    step = pd.Timedelta("15min") if ds == "goes_pvdaq" else pd.Timedelta("30min")
    for site, sg in g.groupby("site_id"):
        ts = sg.timestamp_utc.sort_values()
        diffs = ts.diff().dropna()
        # only count gaps within the same day (data is daytime-only by design)
        same_day = ts.dt.date.values[1:] == ts.dt.date.values[:-1]
        intra_gaps = diffs[same_day & (diffs > step)]
        n_days = sg["date"].nunique()
        span_days = (ts.iloc[-1] - ts.iloc[0]).days + 1
        cov_rows.append({
            "dataset": ds, "site_id": site, "rows": len(sg),
            "days_with_data": n_days, "span_days": span_days,
            "day_coverage_%": round(100 * n_days / span_days, 1),
            "intra_day_gaps": len(intra_gaps),
        })
cov_df = pd.DataFrame(cov_rows)

add_md("### Temporal coverage per site")
add_md()
add_md("Data is daytime-only by design (no night rows), so coverage is measured as "
       "days-with-data over the site's active span, plus missing steps *within* days.")
add_md()
cov_summary = cov_df.groupby("dataset")[["rows", "days_with_data", "day_coverage_%",
                                         "intra_day_gaps"]].agg(["median", "min", "max"]).round(1)
add_md(cov_summary.to_markdown())
add_md()

worst = cov_df.nsmallest(10, "day_coverage_%")
add_md("Worst 10 sites by day coverage:")
add_md()
add_md(worst.to_markdown(index=False))
add_md()
low_cov = cov_df[cov_df["day_coverage_%"] < 70]
if len(low_cov):
    add_issue("warn", f"{len(low_cov)} sites have <70% day coverage "
                      f"(e.g. {low_cov.site_id.head(5).tolist()})")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (ds, g) in zip(axes, cov_df.groupby("dataset")):
    ax.hist(g["day_coverage_%"], bins=30, color="steelblue", edgecolor="white")
    ax.set_title(f"{ds}: day coverage per site")
    ax.set_xlabel("% of span days with data")
    ax.set_ylabel("sites")
save_fig("coverage_per_site.png")

# per-site data availability timeline (which months each site has data)
print("  availability timeline...")
avail = df.groupby(["site_id", pd.Grouper(key="timestamp_utc", freq="MS")]).size().unstack(fill_value=0)
avail_norm = (avail > 0).astype(int)
plt.figure(figsize=(14, max(6, len(avail_norm) * 0.12)))
sns.heatmap(avail_norm, cmap=["#f0f0f0", "#2a76b8"], cbar=False,
            yticklabels=(len(avail_norm) <= 60))
plt.title("Site availability by month (blue = has data)")
plt.xlabel("month")
plt.ylabel("site")
plt.xticks(np.arange(len(avail_norm.columns)) + 0.5,
           [c.strftime("%Y-%m") for c in avail_norm.columns], rotation=90, fontsize=7)
save_fig("site_availability_timeline.png")

# ---------------------------------------------------------------- 3. sites

print("Site-level analysis...")
add_md("## 3. Site-Level Analysis")
add_md()

site_meta = df.groupby("site_id").agg(
    dataset=("dataset", "first"),
    lat=("latitude", "first"),
    lon=("longitude", "first"),
    capacity_w=("installed_power_w", "first"),
    rows=("power_w", "size"),
    mean_norm_power=("norm_power", "mean"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, ds in zip(axes, sorted(site_meta.dataset.unique())):
    g = site_meta[site_meta.dataset == ds]
    sc = ax.scatter(g.lon, g.lat, s=np.sqrt(g.capacity_w) / 2 + 20,
                    c=g.mean_norm_power, cmap="viridis", edgecolor="k", linewidth=0.3)
    ax.set_title(f"{ds} sites (size ~ sqrt(capacity), color = mean capacity factor)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.colorbar(sc, ax=ax, label="mean norm power")
save_fig("site_map.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (ds, g) in zip(axes, site_meta.groupby("dataset")):
    ax.hist(g.capacity_w / 1000, bins=25, color="darkorange", edgecolor="white")
    ax.set_title(f"{ds}: installed capacity distribution")
    ax.set_xlabel("capacity (kW)")
    ax.set_ylabel("sites")
save_fig("capacity_distribution.png")

add_md("Capacity spans multiple orders of magnitude across datasets — power must be "
       "normalized by installed capacity for any cross-site model.")
add_md()

# daytime capacity factor per site
day_df = df[df.shortwave_radiation > 10]
cf = day_df.groupby("site_id").agg(
    dataset=("dataset", "first"),
    cf_mean=("norm_power", "mean"),
    cf_p95=("norm_power", lambda s: s.quantile(0.95)),
).reset_index()
corrs = day_df.groupby("site_id").apply(
    lambda g: g["norm_power"].corr(g["shortwave_radiation"]), include_groups=False)
cf["pw_rad_corr"] = cf.site_id.map(corrs)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.histplot(data=cf, x="cf_mean", hue="dataset", bins=30, ax=axes[0])
axes[0].set_title("Daytime mean capacity factor per site")
axes[0].set_xlabel("mean norm power (irradiance > 10 W/m²)")
sns.histplot(data=cf, x="pw_rad_corr", hue="dataset", bins=30, ax=axes[1])
axes[1].set_title("Per-site corr(norm power, shortwave radiation)")
axes[1].set_xlabel("Pearson r")
save_fig("site_capacity_factor_and_corr.png")

add_md("### Production coherence per site")
add_md()
add_md("Per-site correlation between normalized power and shortwave radiation is the "
       "primary coherence check: a healthy PV site should have r ≳ 0.7. Low values "
       "indicate metering faults, wrong capacity, tracker issues or bad weather joins.")
add_md()
bad_sites = cf[cf.pw_rad_corr < 0.6].sort_values("pw_rad_corr")
if len(bad_sites):
    add_md(f"**{len(bad_sites)} sites with corr < 0.6:**")
    add_md()
    add_md(bad_sites.round(3).to_markdown(index=False))
    add_issue("error", f"{len(bad_sites)} sites have power–radiation corr < 0.6: "
                       f"{bad_sites.site_id.tolist()}")
else:
    add_md("All sites have power–radiation correlation ≥ 0.6. ✓")
add_md()

odd_cf = cf[(cf.cf_p95 < 0.1) | (cf.cf_p95 > 1.0)]
if len(odd_cf):
    add_issue("warn", f"{len(odd_cf)} sites with implausible p95 capacity factor "
                      f"(<0.1 or >1.0): {odd_cf.site_id.tolist()}")

# normalized power boxplot for a sample of sites
sample_sites = []
for ds, g in site_meta.groupby("dataset"):
    sample_sites += g.site_id.sample(min(15, len(g)), random_state=RNG).tolist()
bx = day_df[day_df.site_id.isin(sample_sites)]
plt.figure(figsize=(14, 5))
order = bx.groupby("site_id")["norm_power"].median().sort_values().index
sns.boxplot(data=bx, x="site_id", y="norm_power", order=order, hue="dataset",
            showfliers=False)
plt.xticks(rotation=90, fontsize=7)
plt.title("Daytime normalized power by site (random sample of sites)")
save_fig("site_norm_power_boxplot.png")

# ---------------------------------------------------------------- 4. target

print("Target variable analysis...")
add_md("## 4. Target Variable: PV Power")
add_md()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].hist(df.norm_power.clip(0, 1.2), bins=60, color="royalblue", edgecolor="white")
axes[0].set_title("Normalized power (all rows)")
axes[0].set_xlabel("power / capacity")
pos = df[df.norm_power > 0.001]
axes[1].hist(pos.norm_power.clip(0, 1.2), bins=60, color="royalblue", edgecolor="white")
axes[1].set_title("Normalized power (production > 0.1% only)")
axes[1].set_xlabel("power / capacity")
axes[2].hist(np.log10(pos.power_w.clip(lower=1)), bins=60, color="seagreen", edgecolor="white")
axes[2].set_title("log10 absolute power (W), production only")
axes[2].set_xlabel("log10 power_w")
save_fig("power_distributions.png")

add_md(f"- Zero-power rows: {zero_power:,} ({zero_power / len(df):.1%}) — mostly dawn/dusk "
       "edges of the daytime window; verify they are real zeros, not sentinel fills.")
add_md(f"- Normalized power > 1: {(df.norm_power > 1).sum():,} rows "
       f"(max = {df.norm_power.max():.3f}).")
add_md()

# diurnal profile by solar hour
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (ds, g) in zip(axes, df.groupby("dataset")):
    prof = g.groupby(g.solar_hour.round())["norm_power"].agg(
        ["mean", lambda s: s.quantile(0.1), lambda s: s.quantile(0.9)])
    prof.columns = ["mean", "p10", "p90"]
    ax.plot(prof.index, prof["mean"], color="crimson", label="mean")
    ax.fill_between(prof.index, prof.p10, prof.p90, alpha=0.25, color="crimson",
                    label="p10–p90")
    ax.set_title(f"{ds}: diurnal profile (approx. solar hour)")
    ax.set_xlabel("solar hour")
    ax.set_ylabel("norm power")
    ax.legend()
save_fig("diurnal_profile.png")

# month x hour heatmap
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (ds, g) in zip(axes, df.groupby("dataset")):
    pivot = g.pivot_table(index="month", columns=g.solar_hour.round(),
                          values="norm_power", aggfunc="mean")
    sns.heatmap(pivot, cmap="inferno", ax=ax, cbar_kws={"label": "mean norm power"})
    ax.set_title(f"{ds}: mean norm power by month × solar hour")
save_fig("month_hour_heatmap.png")

add_md("Diurnal and seasonal structure should look like a clean solar geometry "
       "surface; horizontal stripes or holes indicate data problems for specific months.")
add_md()

# monthly production by dataset
plt.figure(figsize=(13, 4.5))
monthly = df.groupby([pd.Grouper(key="timestamp_utc", freq="MS"), "dataset"])["norm_power"].mean().unstack()
monthly.plot(ax=plt.gca(), marker="o")
plt.title("Mean normalized power by month")
plt.ylabel("mean norm power")
save_fig("monthly_production.png")

# ramp rates (step-to-step change), per dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (ds, g) in zip(axes, day_df.groupby("dataset")):
    ramps = g.groupby("site_id")["norm_power"].diff().dropna()
    ax.hist(ramps.clip(-0.5, 0.5), bins=80, color="slateblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_title(f"{ds}: step-to-step ramp of norm power")
    ax.set_xlabel("Δ norm power per step")
save_fig("ramp_rates.png")

add_md("Ramp-rate tails capture cloud-driven variability — the signal satellite imagery "
       "should help predict. Heavy tails = more value from the vision modality.")
add_md()

# example week per dataset: power vs radiation, twin axis
for ds, g in df.groupby("dataset"):
    site = g.groupby("site_id")["norm_power"].mean().idxmax()
    sg = g[g.site_id == site].sort_values("timestamp_utc")
    mid = len(sg) // 2
    week = sg.iloc[mid:mid + (7 * (96 if ds == "goes_pvdaq" else 48))]
    fig, ax1 = plt.subplots(figsize=(14, 4.5))
    ax1.plot(week.timestamp_utc, week.norm_power, color="tab:red", lw=1)
    ax1.set_ylabel("norm power", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(week.timestamp_utc, week.shortwave_radiation, color="tab:blue", lw=1, alpha=0.6)
    ax2.set_ylabel("shortwave radiation (W/m²)", color="tab:blue")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.title(f"{ds} / {site}: one week, power vs radiation")
    save_fig(f"week_{ds}.png")

# autocorrelation of normalized power (intraday persistence)
add_md("### Intraday persistence (autocorrelation)")
add_md()
acf_fig, ax = plt.subplots(figsize=(10, 4.5))
for ds, g in day_df.groupby("dataset"):
    site = g.site_id.value_counts().index[0]
    s = g[g.site_id == site].set_index("timestamp_utc")["norm_power"]
    step = "15min" if ds == "goes_pvdaq" else "30min"
    s = s.resample(step).mean().interpolate(limit=4)
    lags = range(1, 25)
    acf = [s.autocorr(l) for l in lags]
    hours = [l * (0.25 if ds == "goes_pvdaq" else 0.5) for l in lags]
    ax.plot(hours, acf, marker="o", ms=3, label=f"{ds} ({site})")
ax.set_xlabel("lag (hours)")
ax.set_ylabel("autocorrelation")
ax.set_title("Norm power autocorrelation, busiest site per dataset")
ax.legend()
save_fig("power_acf.png")
add_md("High short-lag autocorrelation sets the persistence baseline any forecast "
       "model must beat.")
add_md()

# ---------------------------------------------------------------- 5. features

print("Feature analysis...")
add_md("## 5. Weather Features vs Production")
add_md()

feat_cols = WEATHER_COLS
corr_p = day_df[feat_cols + ["norm_power"]].corr(method="pearson")
corr_s = day_df[feat_cols + ["norm_power"]].corr(method="spearman")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(corr_p, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
axes[0].set_title("Pearson (daytime rows)")
sns.heatmap(corr_s, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title("Spearman (daytime rows)")
save_fig("correlation_matrices.png")

add_md("Correlation with normalized power (daytime rows):")
add_md()
corr_table = pd.DataFrame({
    "pearson": corr_p["norm_power"].drop("norm_power"),
    "spearman": corr_s["norm_power"].drop("norm_power"),
}).sort_values("pearson", ascending=False).round(3)
add_md(corr_table.to_markdown())
add_md()

# hexbin scatter of key drivers
key_feats = ["shortwave_radiation", "direct_radiation", "diffuse_radiation",
             "direct_normal_irradiance", "cloudcover", "temperature_2m"]
sample = day_df.sample(min(150_000, len(day_df)), random_state=RNG).copy()
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, key_feats):
    ax.hexbin(sample[col], sample["norm_power"], gridsize=60, cmap="viridis",
              mincnt=1, bins="log")
    ax.set_xlabel(col)
    ax.set_ylabel("norm power")
fig.suptitle("Weather drivers vs normalized power (150k daytime sample, log density)")
save_fig("feature_vs_power_hexbin.png")

add_md("The power–shortwave relation should be tightly linear with a flat saturation "
       "near the inverter limit; wide vertical spread at high irradiance means "
       "curtailment, soiling, shading or capacity errors.")
add_md()

# cloudcover effect on the irradiance-conditioned residual
sample["rad_bin"] = pd.cut(sample.shortwave_radiation, bins=[0, 100, 250, 450, 700, 1100])
sample["cc_bin"] = pd.cut(sample.cloudcover, bins=[-0.1, 25, 75, 100],
                          labels=["clear (0-25)", "mixed (25-75)", "overcast (75-100)"])
plt.figure(figsize=(12, 5))
sns.boxplot(data=sample, x="rad_bin", y="norm_power", hue="cc_bin", showfliers=False)
plt.title("Norm power by irradiance bin × cloud cover — residual cloud signal")
plt.xlabel("shortwave radiation bin (W/m²)")
save_fig("cloudcover_residual_effect.png")

add_md("If cloud cover still separates power *within* an irradiance bin, images carry "
       "information beyond the reanalysis weather — the core hypothesis of the project.")
add_md()

# feature importance: RF + permutation + mutual information
print("  feature importance (RF on 150k sample)...")
add_md("### Feature importance for predicting normalized power")
add_md()
X = sample[feat_cols + ["solar_hour", "month", "latitude", "longitude"]].fillna(0)
y = sample["norm_power"].clip(0, 1.2)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RNG)
rf = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1, random_state=RNG)
rf.fit(Xtr, ytr)
r2 = rf.score(Xte, yte)
Xperm = Xte.sample(min(20_000, len(Xte)), random_state=RNG)
perm = permutation_importance(rf, Xperm, yte.loc[Xperm.index],
                              n_repeats=5, random_state=RNG, n_jobs=-1)
Xmi = Xtr.sample(min(40_000, len(Xtr)), random_state=RNG)
mi = mutual_info_regression(Xmi, ytr.loc[Xmi.index], random_state=RNG)
imp = pd.DataFrame({
    "rf_gini": rf.feature_importances_,
    "permutation": perm.importances_mean,
    "mutual_info": mi,
}, index=X.columns).sort_values("permutation", ascending=False).round(4)

add_md(f"Random-forest tabular baseline on weather + calendar + location features: "
       f"**R² = {r2:.3f}** on held-out 25% (random split — optimistic vs a true "
       f"site-held-out split, but a useful upper reference for tabular-only signal).")
add_md()
add_md(imp.to_markdown())
add_md()

imp_plot = imp / imp.max()
plt.figure(figsize=(10, 5.5))
imp_plot.plot(kind="barh", ax=plt.gca())
plt.gca().invert_yaxis()
plt.title("Feature importance (each metric max-normalized)")
save_fig("feature_importance.png")

# leave-one-dataset-out probe: train uk -> test goes (zero-shot signal)
print("  cross-dataset transfer probe...")
uk_all = day_df[day_df.dataset == "uk_pv"]
uk = uk_all.sample(min(120_000, len(uk_all)), random_state=RNG)
goes = day_df[day_df.dataset == "goes_pvdaq"]
fcols = feat_cols + ["solar_hour", "month"]
rf2 = RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1, random_state=RNG)
rf2.fit(uk[fcols].fillna(0), uk["norm_power"].clip(0, 1.2))
pred = rf2.predict(goes[fcols].fillna(0))
true = goes["norm_power"].clip(0, 1.2)
ss_res = ((true - pred) ** 2).sum()
ss_tot = ((true - true.mean()) ** 2).sum()
r2_transfer = 1 - ss_res / ss_tot
mae_transfer = (true - pred).abs().mean()
add_md(f"**Cross-dataset zero-shot probe** (RF trained on uk_pv weather features, "
       f"evaluated on goes_pvdaq): R² = {r2_transfer:.3f}, MAE = {mae_transfer:.3f} "
       f"normalized power. This is the tabular transfer floor your foundation-model + "
       f"vision approach should beat.")
add_md()
if r2_transfer < 0:
    add_issue("warn", f"Naive cross-dataset transfer R² is negative ({r2_transfer:.3f}) — "
                      "strong domain shift between uk_pv and goes_pvdaq.")

# ---------------------------------------------------------------- 6. cross-site

print("Cross-site similarity...")
add_md("## 6. Cross-Site Structure (Zero-Shot Transfer Relevance)")
add_md()

# daily energy per site -> site x day matrix -> correlation
daily = day_df.groupby(["site_id", "date"])["norm_power"].mean().unstack()
# goes_pvdaq spans a single month (June 2019, <=30 days/site), so thresholds
# must stay low or every goes site is silently dropped
keep = daily.index[daily.notna().sum(axis=1) >= 20]
daily_k = daily.loc[keep]
site_corr = daily_k.T.corr(min_periods=20)
ds_lookup = site_meta.set_index("site_id")["dataset"]
row_colors = ds_lookup.loc[site_corr.index].map(
    {"uk_pv": "#2a76b8", "goes_pvdaq": "#e07b39"})
cg = sns.clustermap(site_corr.fillna(0), cmap="vlag", center=0, figsize=(12, 12),
                    row_colors=row_colors, col_colors=row_colors,
                    xticklabels=False, yticklabels=False)
cg.fig.suptitle("Site × site correlation of daily mean norm power (clustered)", y=1.0)
cg.savefig(os.path.join(PLOTS_DIR, "site_similarity_clustermap.png"))
plt.close("all")
add_md("![site_similarity_clustermap.png](plots/site_similarity_clustermap.png)")
add_md()

is_uk_r = ds_lookup.loc[site_corr.index] == "uk_pv"
is_uk_c = ds_lookup.loc[site_corr.columns] == "uk_pv"
intra_uk = site_corr.loc[is_uk_r.values, is_uk_c.values]
vals_uk = intra_uk.values[np.triu_indices_from(intra_uk, 1)]
intra_goes = site_corr.loc[~is_uk_r.values, ~is_uk_c.values]
vals_goes = intra_goes.values[np.triu_indices_from(intra_goes, 1)]
cross = site_corr.loc[is_uk_r.values, ~is_uk_c.values].values.ravel()
add_md(f"- Median intra-uk_pv site correlation: **{np.nanmedian(vals_uk):.3f}**")
add_md(f"- Median intra-goes_pvdaq site correlation: **{np.nanmedian(vals_goes):.3f}**")
add_md(f"- Median cross-dataset site correlation: **{np.nanmedian(cross):.3f}**")
add_md()
add_md("High intra-dataset correlation means weather regimes are shared and "
       "leave-site-out splits within a region are *not* independent; cross-dataset "
       "(UK ↔ US) evaluation is the honest zero-shot test.")
add_md()

# dataset shift: feature distributions uk vs goes (KS statistics)
ks_rows = []
for col in feat_cols + ["norm_power"]:
    a = day_df.loc[day_df.dataset == "uk_pv", col].sample(20_000, random_state=RNG)
    b = day_df.loc[day_df.dataset == "goes_pvdaq", col]
    ks = stats.ks_2samp(a, b)
    ks_rows.append({"feature": col, "KS_stat": round(ks.statistic, 3)})
ks_df = pd.DataFrame(ks_rows).sort_values("KS_stat", ascending=False)
add_md("### Distribution shift between datasets (KS statistic, daytime rows)")
add_md()
add_md(ks_df.to_markdown(index=False))
add_md()

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for ax, col in zip(axes.flat, feat_cols):
    for ds, g in day_df.groupby("dataset"):
        sns.kdeplot(g[col].sample(min(30_000, len(g)), random_state=RNG), ax=ax,
                    label=ds, fill=True, alpha=0.3)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
save_fig("dataset_shift_kde.png")

# ---------------------------------------------------------------- 7. images

print("Image analysis...")
add_md("## 7. Image Modality")
add_md()

# integrity: index validity + readability on a large sample (images_all.h5)
n_img_check = 3000
img_sample = df.sample(n_img_check, random_state=RNG)
missing_imgs, corrupt_imgs = [], []
for _, row in img_sample.iterrows():
    im = open_frame(row)
    if im is None:
        missing_imgs.append((row.dataset, row.site_id, row.image_h5_index))
        continue
    try:
        im.verify()
    except Exception:
        corrupt_imgs.append((row.dataset, row.site_id, row.image_h5_index))
add_md(f"- Index/readability check on {n_img_check:,} random rows: "
       f"**{len(missing_imgs)} missing**, **{len(corrupt_imgs)} corrupt**.")
if missing_imgs:
    add_issue("error", f"{len(missing_imgs)}/{n_img_check} sampled frames unreadable "
                       f"(e.g. {missing_imgs[:3]})")
if corrupt_imgs:
    add_issue("error", f"{len(corrupt_imgs)}/{n_img_check} sampled images corrupt")

# timestamp alignment: the frame stored at image_h5_index matches the row time
mis_aligned = 0
for _, row in img_sample.head(500).iterrows():
    expected = row.timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if expected != frame_timestamp(row):
        mis_aligned += 1
add_md(f"- image_h5_index↔timestamp alignment check on 500 rows: **{mis_aligned} mismatches**.")
add_md()
if mis_aligned:
    add_issue("error", f"{mis_aligned}/500 image filenames do not match row timestamp")

# image properties per dataset
prop_rows = []
for ds, g in df.groupby("dataset"):
    s = g.sample(min(400, len(g)), random_state=RNG)
    for _, row in s.iterrows():
        im = open_frame(row)
        if im is None:
            continue
        try:
            arr = np.array(im.convert("L"), dtype=np.float32)
            size, mode = im.size, im.mode
            prop_rows.append({
                "dataset": ds, "site_id": row.site_id, "size": size,
                "mode": mode, "brightness": arr.mean(), "contrast": arr.std(),
                "norm_power": row.norm_power,
                "shortwave_radiation": row.shortwave_radiation,
                "cloudcover": row.cloudcover, "hour": row.hour,
            })
        except Exception:
            pass
props = pd.DataFrame(prop_rows)

add_md("### Image formats")
add_md()
fmt = props.groupby("dataset").agg(
    sizes=("size", lambda s: s.value_counts().to_dict()),
    modes=("mode", lambda s: s.value_counts().to_dict()),
)
add_md(fmt.to_markdown())
add_md()
add_md("Heterogeneous image sizes/modes across datasets — the video encoder pipeline "
       "needs an explicit resize + channel policy per source.")
add_md()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.histplot(data=props, x="brightness", hue="dataset", bins=40, ax=axes[0])
axes[0].set_title("Image mean brightness")
sns.histplot(data=props, x="contrast", hue="dataset", bins=40, ax=axes[1])
axes[1].set_title("Image contrast (pixel std)")
save_fig("image_brightness_contrast.png")

# does the image signal track power / clouds?
add_md("### Do images carry production-relevant signal?")
add_md()
sig_rows = []
for ds, g in props.groupby("dataset"):
    sig_rows.append({
        "dataset": ds,
        "corr(brightness, norm_power)": round(g.brightness.corr(g.norm_power), 3),
        "corr(brightness, shortwave_rad)": round(g.brightness.corr(g.shortwave_radiation), 3),
        "corr(brightness, cloudcover)": round(g.brightness.corr(g.cloudcover), 3),
        "corr(contrast, norm_power)": round(g.contrast.corr(g.norm_power), 3),
        "n": len(g),
    })
add_md(pd.DataFrame(sig_rows).to_markdown(index=False))
add_md()
add_md("Even a trivial brightness statistic correlating with power/cloud confirms the "
       "imagery is informative; the video encoder should extract far more (cloud "
       "morphology, motion).")
add_md()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (ds, g) in zip(axes, props.groupby("dataset")):
    sc = ax.scatter(g.brightness, g.norm_power, c=g.cloudcover, cmap="Blues_r",
                    alpha=0.7, s=18)
    ax.set_xlabel("image mean brightness")
    ax.set_ylabel("norm power")
    ax.set_title(f"{ds}")
    plt.colorbar(sc, ax=ax, label="cloudcover")
save_fig("brightness_vs_power.png")

# sample grids: low vs high production for each dataset
for ds, g in df.groupby("dataset"):
    gg = g[g.shortwave_radiation > 50]
    low = gg.nsmallest(2000, "norm_power").sample(4, random_state=RNG)
    high = gg.nlargest(2000, "norm_power").sample(4, random_state=RNG)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (_, row) in zip(axes.flat, pd.concat([high, low]).iterrows()):
        im = open_frame(row)
        if im is not None:
            ax.imshow(im, cmap="gray" if im.mode == "L" else None)
        else:
            ax.text(0.5, 0.5, "missing", ha="center")
        ax.set_title(f"{row.site_id}\nnp={row.norm_power:.2f} cc={row.cloudcover:.0f}%",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{ds}: top row = high production, bottom row = low production "
                 "(daytime only)")
    save_fig(f"image_samples_{ds}.png")

# intra-day image sequence for one site (video-encoder sanity check)
for ds, g in df.groupby("dataset"):
    site = g.site_id.value_counts().index[0]
    sg = g[g.site_id == site]
    best_day = sg.groupby("date")["norm_power"].mean().idxmax()
    day_rows = sg[sg.date == best_day].sort_values("timestamp_utc")
    step = max(1, len(day_rows) // 8)
    day_rows = day_rows.iloc[::step][:8]
    fig, axes = plt.subplots(1, len(day_rows), figsize=(16, 3))
    for ax, (_, row) in zip(np.atleast_1d(axes), day_rows.iterrows()):
        im = open_frame(row)
        if im is not None:
            ax.imshow(im, cmap="gray" if im.mode == "L" else None)
        else:
            ax.text(0.5, 0.5, "missing", ha="center")
        ax.set_title(f"{row.timestamp_utc:%H:%M}\nnp={row.norm_power:.2f}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"{ds} / {site}: image sequence on {best_day} (video-encoder input view)")
    save_fig(f"image_sequence_{ds}.png")

# ---------------------------------------------------------------- 8. issues

add_md("## 8. Auto-Detected Issues & Recommendations")
add_md()
if not issues:
    add_md("No blocking issues detected. ✓")
else:
    sev_order = {"error": 0, "warn": 1}
    for sev, text in sorted(issues, key=lambda x: sev_order.get(x[0], 2)):
        marker = "🔴" if sev == "error" else "🟡"
        add_md(f"- {marker} **{sev.upper()}**: {text}")
add_md()
add_md("### Recommendations for the modeling phase")
add_md()
add_md("1. **Normalize power by installed capacity** everywhere; clip to [0, 1] after "
       "verifying >1 rows are inverter-rating artifacts and not capacity errors.")
add_md("2. **Mask or drop suspected-outage rows** (zero power under strong irradiance) "
       "and stuck-sensor runs — they corrupt both training and evaluation.")
add_md("3. **Exclude or down-weight sites** with power–radiation correlation < 0.6 "
       "until their metering is explained.")
add_md("4. Use **cross-dataset (UK↔US) splits** as the primary zero-shot benchmark; "
       "intra-region leave-site-out is contaminated by shared weather.")
add_md("5. Handle the **sampling-rate mismatch** (15 vs 30 min) and **image "
       "heterogeneity** (32×32 L vs 256×256 RGB) explicitly in the data pipeline.")
add_md("6. The tabular RF baselines above (random-split R² and UK→US transfer R²) are "
       "the numbers the foundation-model + vision approach must beat.")

report_path = os.path.join(OUTPUT_DIR, "dataset_exploration_report.md")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"\nReport: {report_path}")
print(f"Plots:  {PLOTS_DIR}")
print(f"Issues: {len(issues)} flagged")
