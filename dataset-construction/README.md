# dataset-construction/ — provenance of the dataset of record

The ETL that produced `thesis-dataset/dataset_all.parquet` + `images_all.h5`.

**Rescued 2026-08-20 from `/Volumes/SSD`, where all three source repos had no git
remote and zero commits — every file untracked.** `report/REPORT.md` §1.3
*describes* this pipeline; until now the code that implements it existed in
exactly one place, on a disk about to be reformatted. It is checked in verbatim,
unmodified, so the numbers in the thesis stay traceable to the code that made
them.

Nothing here is on the training path. It is documentation-by-source plus the
starting point for any re-extraction.

| dir | from | what it does |
|---|---|---|
| `builder/` | `/Volumes/SSD/dataset_builder` | merges the numerical tracks, joins weather/solar covariates, stamps site metadata |
| `exploration/` | `/Volumes/SSD/dataset-exploration` | curation, the unified merge, `pack_images.py` (frames → HDF5), GOES extension, EDA behind `report/REPORT.md` |
| `uk_satellite/` | `/Volumes/SSD/useless-stuff-dataset/uk-data` | UK satellite crops: `main.py` (32×32, superseded), `extract_128.py` (128×128, current) |

Each dir keeps its own `pyproject.toml` + `uv.lock`; they were separate projects
with separate environments and are not merged.

## Known defects in this code — read before re-running any of it

**`uk_satellite/extract_128.py` — the source is visible-only.** It reads
`SEVIRI_RSS/v4/{year}_hrv.zarr`. HRV is High Resolution *Visible*: reflected
sunlight, black at night by physics. Line ~159 then drops what it produces:

```python
valid = ~np.isnan(patches).all(axis=(1, 2))   # night HRV is all-NaN -> dropped
arr = np.nan_to_num(patches[valid], nan=0.0)
```

Consequence, measured 2026-08-20 (SLURM job 53012660): cache windows sit at
stride 12 on a 30-minute grid, so uk_pv has exactly two origins per site per day,
07:30 and 13:30 UTC. The visual window looks 6 h *backward*, so 13:30 sees
07:30–13:30 (daylight) and 07:30 sees 01:30–07:30 (dark most of the UK year).
Max per-column std of the pooled V-JEPA latent: **4.39** for 13:30 origins,
**0.0038** train / **0.00023** test for 07:30. Four orders of magnitude — those
are embeddings of a blank frame, and *non-zero*, so an all-zero check reports
0.0 % and catches nothing. Half the visual channel is blank, in the probe and in
s2a/s2b training alike. See `knowledge/specs/2026-08-19-visual-fusion-diagnosis.md`.

**The fix is in the same bucket.** `{year}_nonhrv.zarr` exists at the identical
path for 2019 and 2020, with the identical 104,807 timesteps — the night scans
were always there. 11 channels; 8 of them emissive (`IR_039 IR_087 IR_097
IR_108 IR_120 IR_134 WV_062 WV_073`) and therefore alive 24 h. Open Climate
Fix's own PVNet uses the 11 non-HRV channels and not HRV.

Two things to know before using it:

- **Values are normalised to [0,1] float16, not kelvin.** Recover with the affine
  from `openclimatefix/Satip`, `satip/scale_to_zero_to_one.py` (`K = v*(max-min)+min`).
  Verified: `IR_087/108/120` then agree at 268–271 K on the same pixel.
- **Do not build RGB from channel *differences*.** The float16 squeeze quantises
  `IR_120` at 0.325 K and `IR_087` at 0.270 K, against the 6 K beam width the
  standard 24h-Microphysics recipe uses — several percent of full scale, which
  shows up as heavy speckle (measured speckle index 0.451 day / 0.313 night vs
  0.129 / 0.057 for a difference-free recipe). Prefer three emissive channels
  taken straight: `IR_108` (235–300 K), `WV_073` (240–275 K), `WV_062` (225–245 K).

**Crop footprint was also too small.** 128×128 at HRV's ~1 km is ~128 km of
context against a 6 h lookback, but clouds advect 120–360 km in 6 h — the source
region was mostly outside the frame. Non-HRV at ~3 km makes the same 128×128
cover ~384 km, which matches. This is an independent bound on the horizon at
which vision can help, and it lines up with the measured signal peaking at h=3
and dying by h=5.

**Filesystem.** These scripts were written for an exFAT volume with a 131,072-byte
allocation block, where macOS also writes a `._` AppleDouble sidecar per file —
so every PNG cost 256 KiB on disk regardless of its real size, and
`extract_128.py`'s header comment rules out per-frame files for that reason. On
APFS (4 KiB blocks, native xattrs) the same 3.5 M frames cost ~112 GB instead of
~918 GB, and per-frame PNG becomes practical.
