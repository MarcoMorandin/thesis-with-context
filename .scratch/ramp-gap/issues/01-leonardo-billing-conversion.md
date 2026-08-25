# 01 — Does wave 1 fit the monthly cap?

Type: task
Status: resolved

## Question

Wave 1 is five curriculum chains, each `s1 → s2a → s2b`. From the measured per-epoch times
in `slurm_curriculum.sh` (s2a 69 min/epoch, EarlyStopping at 13; s2b 77 min/epoch, ~20
epochs) a chain is roughly **62 GPU-hours**, so five chains is roughly **310 GPU-hours**.

`saldo -b` reports the account in **local hours**, not GPU-hours:

```
IscrC_MTSFM   total 60000   consumed 14261 (23.8%)   monthTotal 6545   monthConsumed 1343
```

The conversion from a `boost_usr_prod` GPU-hour to a billed local hour is unknown and must
not be guessed — it decides whether wave 1 fits August's remaining 5,202 h, spills into
September's fresh 6,545, or does not fit a single month at all.

Resolve by measuring, not by reading a table: submit one short job to `boost_usr_prod` with
`--gres=gpu:1`, note `saldo -b` before and after, and derive the factor. Then state
explicitly whether five chains fit one month, and if not, how the wave splits across the
Aug/Sep boundary.

The answer records the conversion factor and the go/no-go for five chains, since every later
sizing decision depends on it.

## Answer — the monthly cap covers wave 1

Decided by the user 2026-08-25: the 6,545 local-h monthly cap is sufficient for the six
chains, so the conversion factor does not need measuring before launch.

Sizing that stands behind it: ~62 GPU-h per chain (s1 ~20 h + s2a ~16 h + s2b ~26 h, s3
dropped per standing decision 4) x 6 chains = **~370 GPU-h**. Against 5,202 h remaining in
August and a fresh 6,545 on 1 September, with a 45,739 h balance and the account open to
2026-12-02. Compute is not the binding constraint on this map; wall-clock and queue are.

The local-h per GPU-h factor is therefore still **unmeasured**. It only matters if a later
wave is large enough to approach a monthly cap — measure it then with one short
`boost_usr_prod` job and a `saldo -b` before/after, rather than assuming.

**Facts later tickets depend on:**

- Wave 1 is cleared to launch on budget grounds. [Launch wave 1](08-launch-wave-1.md) has no
  remaining blockers.
- If a wave-2 arm set is materially larger than six chains, size it against the cap before
  submitting.
