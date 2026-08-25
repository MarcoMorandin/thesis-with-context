# 12 — Was the V-JEPA cache built from v1 HRV or v2 non-HRV frames?

Type: task
Status: resolved

## Question

`curriculum_stage.sbatch` guards against a cache/data mismatch by comparing
`_extract_meta.txt`'s recorded `data_dir=` against `DATA_DIR`. That catches a cache built
from a *different directory*. It does **not** catch a cache built from *different imagery
at the same path* — which is exactly what happens if `images_all_v2.h5` replaced the v1
archive in place.

The stakes are larger than wave 1. If the 139k-file cache at
`/leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224` predates v2, then the
latents are **v1 HRV single-channel daylight-only** frames, and:

- `mmtsfm_s2b_ukpv.json` and its +2.7% vision marginal describe imagery the model never saw
- `knowledge/dataset.md` §2's rewrite — three distinct IR bands, night frames carrying
  structure, 96.1% window coverage — describes the *dataset*, not the *cache the model
  consumed*, and the two would be silently different
- the ~2 h advection decorrelation, measured on v2 frames, would not bound what the model
  actually had access to

Verify before wave 1, because a stale cache means re-extraction (expensive, ~139k files)
and a re-read of every vision number already reported.

```bash
# what the cache says about itself
cat /leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224/_extract_meta.txt

# cache mtime vs imagery mtime — the decisive comparison
stat -c '%y %n' /leonardo_work/IscrC_MTSFM/vjepa_cache/uk_pv/vit_large_f8_s224/*.pt \
  | sort | head -3
stat -c '%y %n' $DATA_DIR/images_all*.h5 $DATA_DIR/dataset_all*.parquet

# which filenames Leonardo actually exposes (pv_record hard-codes the un-suffixed names)
ls -la $DATA_DIR/
```

Any cache file older than the h5 → the cache is stale. Report the `_extract_meta.txt`
contents and both mtimes; the answer records which imagery every existing vision number
was actually computed on.

## Answer — the cache is v2, and every vision number stands

Verified on Leonardo 2026-08-25.

```
_extract_meta.txt: data_dir=/leonardo_scratch/fast/IscrC_MTSFM/data_v2
                   arch=vit_large|frames=8|img=224|window_h=6.0
                   spacing_min=auto|train_stride=12
cache .pt mtime  : 2026-08-21 11:16
images_all.h5    : 2026-08-20 20:27
s2b run consumed : .../uk_pv/vit_large_f8_s224_nonhrv_sp45
```

Cache is ~15 h newer than the imagery, records the v2 data dir, and the s2b run of record
was explicitly pointed at it. **`mmtsfm_s2b_ukpv.json`, the +2.7% vision marginal, and
`knowledge/dataset.md` §2 all describe the imagery the model actually consumed.** No
re-extraction, no re-reading of results.

Two side facts, both closing open questions:

- **Filenames.** `data_v2/` contains `dataset_all.parquet` + `images_all.h5` — un-suffixed.
  The `_v2` is the *directory*. `pv_record.py`'s hard-coded names are correct as written;
  only `DATA_DIR` changes. Sizes match the local sync byte-for-byte.
- **Spacing.** `spacing_min=auto` with `window_h=6.0, frames=8` resolves to 45 min, which is
  the slot rule the 96.1% window-coverage measurement in `dataset.md` §2.2 assumed. The
  measurement describes the frames the cache encoded.

### The trap this exposed, now closed

The obsolete v1 HRV cache **still exists** at `.../uk_pv/vit_large_f8_s224` — which was the
value `VJEPA_CACHE_VER` defaulted to. `-d` would have succeeded, and the mismatch guard only
fires when `_extract_meta.txt` is present, so a cache predating provenance would have been
consumed in silence. Three changes:

1. `VJEPA_CACHE_VER` now defaults to `vit_large_f8_s224_nonhrv_sp45`, the cache of record.
2. A requested-but-absent cache is **fatal** instead of a WARN + silent fall back to live
   encoding (~10x cost, discovered only after the job finishes).
3. A cache with **no `_extract_meta.txt` is fatal**. Unknown provenance is not something to
   train a wave on, and it is precisely the hole the stale directory sat in.

Four tests in `test_curriculum_runner_wave_safety.py` cover absent / no-provenance /
matching / mismatched. 249 tests green.

**Facts later tickets depend on:**

- `DATA_DIR=/leonardo_scratch/fast/IscrC_MTSFM/data_v2`
- `VJEPA_CACHE_VER=vit_large_f8_s224_nonhrv_sp45` (now the default; no need to pass it)
- Deleting `.../uk_pv/vit_large_f8_s224` would remove the trap entirely. Left in place —
  that is a data-deletion call for the user, and the guards now make it loud rather than
  silent.
