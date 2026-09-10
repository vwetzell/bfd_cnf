# NEXT SESSION — take `gauss2_v3d` to σ(m1) = 1e-3

Written 2026-09-07, after the timing work that settled the device path.

---

## READ THIS BEFORE LAUNCHING — what this run does and does not give you

**It delivers σ(m1) = 1e-3. It does NOT promise |m1| < 1e-3.**

`[[road-to-1e-3]]` is explicit that statistics is the *last* lever: the known
floor is the centroid layer's spin-2 ellipticity-response undershoot (25–30% on
`copies_gauss2_deep`), and no amount of target count touches it. That memory
also says this run "only pays once the centroid term is understood."

The reason to do it anyway: the best current number is

> `gauss2_v3d` 500k, window (2.2, 3.2), terms at 2^26:
> **m1 = −0.0030 ± 0.0031** — 1.0σ, consistent with zero

The floor was measured at ~5e-3 *before* the size-ceiling fix and has not been
re-confirmed since. So −0.0030 is either a real residual sitting right at the
old floor, or nothing. **At σ = 1e-3 that becomes a 3σ detection or a clean
null.** Either outcome is decisive and neither is available today. That is the
justification — not the hope that the number will land under 1e-3.

If you want |m1| < 1e-3 rather than σ(m1) = 1e-3, do the centroid spin-2 work
first and don't spend 36 GPU-h here.

---

## The arithmetic

Measured from the 500k run, not assumed:

| quantity | value |
|---|---|
| in-window targets | 117,247 |
| its bar | ±0.00303 |
| **k = σ·√N_in** | **1.038** (memory's 1.03 ✓) |

So σ = 1e-3 needs **N_in = 1.08M**. At the 500k run's ratios (in-window 23.5% of
catalog, pre-filter integrates 50.9%):

| | |
|---|---|
| catalog targets | **4.6M** |
| integrated (2 arms) | **2.33M** |
| GPU time, device path @ 46.64 ms | **30.2 h** |
| GPU time, host path @ 52.08 ms | 33.7 h |
| render | ~5.4 h CPU |
| disk | ~6.8 GB (3 arms × 2.25 GB) — 80 GB free ✓ |

---

## BLOCKER: the catalog on disk is 500k

`CATALOGS["gauss2_v3d"]` points at `targets_g2v3d_g1{p,m}02_500k` /
`..._g0_500k`. There is no 4.6M catalog. **You must render one first.** The
population is file-backed, not generated on the fly.

### Step 1 — render (CPU, ~5.4 h)

`rebuild_v3.sh` is the provenance record and already parameterised for this:

```bash
setsid env SIGMA=1.86 POP=gauss2_fwd TAG=g2v3e NTARGET=4600000 SIZE=4600k \
    bash rebuild_v3.sh render
```

* **`SIGMA=1.86` is mandatory** — that is the deep arm. The 0.93 default is the
  shallow one and would silently build the wrong population.
* **`TAG=g2v3e`, a NEW tag.** Do not reuse `g2v3d`; the script's own comment
  warns an override must carry its own tag so it cannot overwrite a 0.93
  catalog. Same logic protects the existing 500k set.
* Render all three arms even though the zero arm is not integrated —
  `bias.py:3158` still reads `m_raw["zero"]` for the ESS diagnostic.
* `[[jax-threadpool-starves-renders]]`: this is CPU and parallel. Do not run
  anything else on the box, JAX included.

### Step 2 — register the catalog (2 small edits in `bias.py`)

```python
# CATALOGS, next to "gauss2_v3d"
"gauss2_v3e": {"plus":  "targets_g2v3e_g1p02_4600k",
               "minus": "targets_g2v3e_g1m02_4600k",
               "zero":  "targets_g2v3e_g0_4600k"},

# TRAIN_DATA — SAME prior file as g2v3d, deliberately
"gauss2_v3e": "moments_gauss2_fwd_g2v3d.fits",
```

**Do not retrain the flows and do not re-render the prior.**
`RawMomentStandardize` is frozen at training time, so the new targets must be
standardised against the file the flows were trained on. This is the same
population at a larger target count, so `[[no-cross-population-transfer]]` is
satisfied — it would be violated by pointing at a *different* population's
moments, which is the exact bug that memory records.

Verify before the long run: a 5k smoke test should reproduce the 500k run's
pre-filter fractions (~50.9% integrated, ~23.5% in window). If they differ, the
render used the wrong `SIGMA` or `POP`.

---

## Step 3 — make it resumable (recommended, ~10 lines)

A single 30 h job that dies at hour 29 loses everything. `bias.py:2990` is

```python
n = slice(None) if a.n_targets is None else slice(a.n_targets)
```

Add `--target-offset` (default 0) and make it `slice(off, off + a.n_targets)`.
Then run **5 slices of 920k targets** (~6 h each), each with its own
`--save-pqr`, and concatenate the PQR files offline. Benefits beyond crash
safety: you can inspect m1 drift across slices as a free consistency check, and
stop early if something is wrong.

If you skip this, run the 4.6M in one go and accept the risk.

---

## Step 4 — the run

Per slice (drop `--target-offset` and set `--n-targets 4600000` for one-shot):

```bash
setsid env BFD_DEVICE_PQR=1 JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax \
python -u bias.py \
    --pop gauss2_v3e --flow flows/centroid_g2v3d.eqx \
    --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 65536 \
    --n-targets 920000 --target-offset 0 \
    --gauge auto --no-zero-arm \
    --prefilter-pad 0.2 --prefilter-sample 0.15 \
    --window-size 2.2 3.2 --window-flux 2500 50000 \
    --window-terms score --window-draws 16777216 \
    --floor-eps 1e-3 --window-guard 100 \
    --save-pqr pqr/g2v3e_slice0.npz 2>&1 | tee logs/bias_g2v3e_s0.log
```

### Every speedup, and what each is worth

| flag | worth | evidence |
|---|---|---|
| `BFD_DEVICE_PQR=1` | 26.04 → **23.32 ms**/target/arm (10.4%, 5.0σ) | 4-point fits both paths, 2026-09-07 |
| `--prefilter-pad 0.2 --prefilter-sample 0.15` | ~2× — integrates 50.9% not 100% | `[[prefilter-the-window]]` |
| `--no-zero-arm` | 30%, m1/c bit-identical | `[[zero-arm-is-optional]]` — this is your "no shear arm" |
| `JAX_COMPILATION_CACHE_DIR` | ~267 s/run; outputs **identical** | measured 2026-09-07, both passes |
| already in the code | `make_psi_ld` 2.6%, on-device `kernel_draws` 3%, direct Q/R read 5% | `dev/g2v3d_500k.sh` header |

`BFD_DEVICE_PQR=1` is above its break-even here by 45× — it only wins past
~52k integrated targets, and this run integrates 2.33M.

---

## Step 5 — the number you quote

`ghat` is affine in (P_s, Q_s, R_s), so the selection terms re-solve off the
saved PQR with **no target integration**:

```bash
for seed in 0 7; do
  JAX_PLATFORMS=cpu python -u dev/window_scan.py --pop gauss2_v3e \
      --flow flows/centroid_g2v3d.eqx --pqr pqr/g2v3e_all.npz \
      --log2-draws 28 --no-scan --seed $seed | tee logs/resolve_g2v3e_s$seed.log
done
```

**2^28, not 2^26.** `[[draw-noise-not-negligible-at-depth]]`: the seed spread is
2.4e-4 at 2^26, which is 24% of a 1e-3 bar — too close to ignore. 2^28 puts it
at ~1.2e-4. Cost is ~2.7 h per measurement, offline, no GPU integration.

The two-seed spread **is** the selection MC error; `R_s12` and the traceless
part are forced zeros, so their magnitude is that error without differencing.

---

## Traps — all of these have bitten before

* **ONE JAX job at a time — GPU *and* CPU.** A `JAX_PLATFORMS=cpu` probe run
  beside a GPU job crashed this box into a swap spiral and a reboot.
  `[[no-jax-cpu-alongside-gpu]]`
* **`setsid`, and poll log files, not `pgrep -f`** — `pgrep -f` matches the
  waiter's own command line. `[[background-hang-is-pgrep-selfmatch]]`
* **`--floor-eps 1e-3 --window-guard 100` are mandatory on gauss2.** Without
  them the deep arm returned m1 = −5.75 ± 116.64.
* **`--window-terms score`, never `templates`.** `[[flow-only-selection-correction]]`
* **`--gauge auto`** — gauge P's Var[score] is infinite. `[[r-is-an-is-artifact]]`
* **The size window ceiling 3.2 is not optional.** Every number taken at
  `2.2 1e9` is void. `[[draw-noise-not-negligible-at-depth]]`
* **`prefilter_w` must be in the PQR** or `window_scan.py` under-counts `N_ns`
  by 64% and you get +0.0121 instead of −0.0028. `save_pqr` writes it now —
  confirm the key is present before re-solving. `[[window-scan-n-ns-fixed]]`
* **The device path uses a different RNG stream.** Statistically consistent
  (m1 moves 5.6e-03 against ~1.2e-02 draw noise) but **not paired** — never
  difference a device-path PQR against a host-path one.
* **`--batch-budget` bounds memory**; omitting it makes the batch larger and
  the card OOMs. A single run sits near 14 GB on a 16 GB card.
* **`bc` is not installed on this box.** Use `$((...))` in shell timing.
* Compilation-cache timings are noisy: one run in six came in 400 s slow and
  halved on repeat. Never conclude a cached timing from a single run.

---

## Risks specific to this run

1. **Draw noise.** `[[draw-noise-not-negligible-at-depth]]` — the deep arm's
   per-target draw noise does *not* fully converge in S; ~1% of targets carry
   ~55% of the variance, and they sit at the `POINT_SOURCE` ceilings. The (2.2,
   3.2) window excludes most of them (0.36% near-ceiling vs 8.05% without), and
   the residual integrates down with target count to ~6e-5 here. **Check, don't
   assume:** run a `--draw-seed` null on one slice. If the spread exceeds
   ~3e-4, raise `--samples` or revisit the stability guard that memory flags as
   an open lead.
2. **The selection correction may not be the limit but is not free** — at 1e-3
   its 2^28 error (~1.2e-4) is fine, but `R_s12`/traceless are 12–13% of
   `R_s11` at the ceiling window vs 2.4% without. Watch them.
3. **30 h is long enough for the box to be disturbed.** Slices (Step 3) are the
   mitigation.

---

## Definition of done

* `m1 ± σ` with σ ≤ 1.1e-3, window (2.2, 3.2), flux (2500, 50000)
* selection terms re-solved at 2^28 on two seeds, spread quoted
* a draw-seed null on one slice showing < 3e-4
* the result written into `[[road-to-1e-3]]` either as "the −0.0030 is real at
  Nσ" or "consistent with zero at 1e-3", and the centroid-floor attribution
  updated accordingly
