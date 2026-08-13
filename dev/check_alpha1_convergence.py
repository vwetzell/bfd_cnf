"""Does raising S fix the alpha = 1 blowup, or is it unbounded weights?

The full deep run (S = 32768, alpha = 1) puts Mf-q2 at m1 = -0.55; dropping ONE
target takes it to -0.053, which is what alpha = 0.5 says at a quarter the
draws.  Two readings:

  (a) the kernel proposal is merely under-sampled there, so S buys convergence;
  (b) the kernel's weights are UNBOUNDED for that target -- its C_M-ball barely
      overlaps the prior's support, one draw carries all the weight, and the
      estimator has no finite variance to converge to.  Then no S is enough.

(b) is what `mixture_draws` exists for: with alpha < 1, q >= alpha * L(M_i - m)
bounds the weight by sup_m P(m|g)/alpha regardless of the target.

This runs the offenders (and a control sample of ordinary targets) up a ladder
of S at alpha = 1 and asks which reading holds.  If Q is still moving by orders
of magnitude at the top of the ladder while the control is flat, it is (b).
"""

import sys

import jax.random as jr
import equinox as eqx
import fitsio
import numpy as np

# Repo root only.  NOT "..": /home/vwetzell/gitrepos/bfd would then shadow the
# installed `bfd` package that bias.py imports.
sys.path.insert(0, ".")

import bias  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402

DATA = "../bfd_cnf_imsims/data"
FLOW = "flows/centroid_deep.eqx"
LADDER = [8192, 32768, 131072, 524288]
SEED = 1001                      # bias.py's noise_seed + 1000
REF = "/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/c3167564-9dce-4c7a-b465-37816d87f103/scratchpad"


def load():
    """Reproduce `bias.main`'s load for --pop bulgedisc_deep, exactly."""
    cat = bias.CATALOGS["bulgedisc_deep"]
    path = lambda v: f"{DATA}/{v}.fits"
    rows = {k: fitsio.read(path(v))[:20000] for k, v in cat.items()}
    keep = np.ones(len(rows["zero"]), dtype=bool)
    for v in rows.values():
        keep &= ~v["badcenter"]
    rows = {k: v[keep] for k, v in rows.items()}
    m = {k: np.asarray(v["moments"], dtype=np.float64) for k, v in rows.items()}
    sigma_x = np.asarray(rows["zero"]["cov_odd"], dtype=np.float64)

    m_train = shear.load(f"{DATA}/moments.fits")[0]      # no 0.9 slice: centroid on
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)
    return flow, m, sigma_x, bias.load_cov(path(cat["zero"]))


def pick(n_ctrl=64):
    """The targets where alpha = 1 and alpha = 0.5 disagree, plus a control."""
    full = np.load(f"{REF}/full_with.npz")
    a05 = np.load(f"{REF}/a05_with.npz")
    rel = (np.abs(full["plus_q"] - a05["plus_q"]).max(1)
           / np.maximum(np.abs(a05["plus_q"]).max(1), 1.0))
    bad = np.argsort(-np.nan_to_num(rel, nan=np.inf))[:32]
    # Control: ordinary targets at the SAME flux, so any S-dependence seen in
    # the offenders cannot be blamed on their being faint.
    flux = full["flux"]
    rng = np.random.default_rng(0)
    lo, hi = np.percentile(flux[bad], [5, 95])
    pool = np.flatnonzero((flux >= lo) & (flux <= hi) & (rel < 0.05))
    return bad, rng.choice(pool, n_ctrl, replace=False), a05


def main():
    flow, m, sigma_x, cov = load()
    bad, ctrl, a05 = pick()
    idx = np.concatenate([bad, ctrl])
    print(f"{len(bad)} offenders + {len(ctrl)} controls at matched flux\n")

    out = {}
    for s in LADDER:
        chunk = min(s, 4096)
        q, r = bias.pqr_streamed(flow, m["plus"][idx], cov, s, 1.0, SEED,
                                 sigma_x[idx], batch=max(1, 131072 // chunk),
                                 chunk=chunk)
        out[s] = (np.asarray(q), np.asarray(r))
        print(f"  S = {s:7d} done", flush=True)

    qa = a05["plus_q"][idx]
    print(f"\n{'target':>8s} {'flux':>9s} " +
          " ".join(f"{'Q1 @ ' + str(s):>12s}" for s in LADDER) +
          f" {'Q1 a=0.5':>12s}")

    def show(name, sl):
        print(f"-- {name}")
        for j in sl:
            row = " ".join(f"{out[s][0][j, 0]:>12.3f}" for s in LADDER)
            print(f"{idx[j]:>8d} {a05['flux'][idx[j]]:>9.0f} {row} "
                  f"{qa[j, 0]:>12.3f}")

    show("offenders (worst 12)", range(12))
    show("controls (first 6)", range(len(bad), len(bad) + 6))

    # The summary number: how much does Q still move over the last 4x of S?
    print("\nmedian |Q(S) - Q(S/4)| / |Q(a=0.5)| over the top rung:")
    for nm, sl in (("offenders", slice(0, len(bad))),
                   ("controls", slice(len(bad), None))):
        a, b = out[LADDER[-1]][0][sl, 0], out[LADDER[-2]][0][sl, 0]
        d = np.abs(a - b) / np.maximum(np.abs(qa[sl, 0]), 1.0)
        print(f"  {nm:>10s}  {np.median(d):.3f}   (worst {np.max(d):.1f})")
    np.savez("/tmp/alpha1_ladder.npz", idx=idx, n_bad=len(bad),
             **{f"q{s}": out[s][0] for s in LADDER},
             **{f"r{s}": out[s][1] for s in LADDER})


if __name__ == "__main__":
    main()
