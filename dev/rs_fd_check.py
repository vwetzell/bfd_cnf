"""Minimal, from-scratch check of the selection terms P_s, Q_s, R_s at g=0 --
built to be independently readable in one sitting, not trusted from bias.py's
larger (and, this session found, buggy) estimator machinery.

MATH (paper eq. 30/40/45-46).  P_s(g) = E_{m ~ P(.|g)}[F(m)], where `F(m)` is
the C_M-SMOOTHED probability that a noiseless moment `m`'s NOISY measurement
lands in the selection window (eq. 30's closed-form Gauss-Legendre quadrature
over C_M -- see `bias.window_prob`).  Q_s = dP_s/dg, R_s = d2P_s/dg2 at g=0.

WHY THIS IS SIMPLE AND TRUSTWORTHY.  `F` is a PROBABILITY, bounded in [0, 1]
BY CONSTRUCTION -- it cannot blow up no matter how pathological the flow's
fitted density is anywhere.  That is what this session's failure mode was:
autodiffing a per-draw score/Hessian of `log P_flow` at the SAME draws (what
`bias.selection_terms_score` does) hits genuine spikes in the fitted density
(`dev/rs_census.py`: 10 in-window, in-support draws carrying 50% of the raw
`R_s11` sum) and every attempted fix (a magnitude guard, C_M-kernel smoothing
via importance sampling) either needed a tunable factor or introduced its OWN
instability (`dev/rs_smooth_check.py`: pure-kernel smoothing made the BULK
worse than raw, correlation ~0).  None of that is possible here:

  1. Draw N samples ONCE from the flow's base distribution.
  2. Push the SAME z through the flow's transform at 9 fixed g points (a
     standard 2-D central-difference stencil) -- common random numbers, so
     the differences are paired and most of the per-draw noise cancels.
  3. Average the BOUNDED F(m(g)) at each stencil point.  A mean of bounded
     terms has ordinary finite variance; CLT applies; the printed standard
     error means something.  No score, no Hessian, no kernel draws, no ESS,
     no guard factor.
  4. Central-difference the 9 numbers for P_s, Q_s, R_s.

C_M is `cov` below -- loaded straight from the target catalog's own FITS
header (`bias.load_cov`, eq. 9) and passed into `window_prob` exactly as the
rest of the pipeline does.  It enters ONLY through `F`; the flow still draws
NOISELESS moments at g=0, and C_M's job is solely to smooth the window edge
by the measurement noise, which is what makes F bounded instead of a sharp
step in the first place.

The step `h` matches `ghat`'s own operating point (|g| ~ 0.02) rather than
being tuned toward zero: eq. (45)-(46) is already a quadratic stand-in for
P_s(g) over that range, not a claim about an infinitesimal derivative.

`bias.load_cov`, `bias.window_prob` and `bias.condition` are reused here --
they are plain, deterministic, already-checked utilities (a FITS read, a 1-D
quadrature, a concatenation), not the estimator machinery this session found
bugs in.  Everything else in this file is new and self-contained.

    python -u dev/rs_fd_check.py --flow flows/centroid_g2v3d.eqx
"""
import argparse
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B  # noqa: E402
import bulk  # noqa: E402
import shear  # noqa: E402


def load_flow_f64(pop, flow_path, data_dir):
    """(flow, cov, sigma_x) -- flow built float64, checkpoint loaded float32
    first (its own dtype) then upcast, so `eqx.tree_deserialise_leaves`'s
    dtype assert against the template does not fire."""
    cat = B.CATALOGS[pop]
    targets = f"{data_dir}/{cat['zero']}.fits"
    cov = B.load_cov(targets)                                    # C_M, eq. (9)
    sigma_x = np.asarray(fitsio.read(targets)["cov_odd"], dtype=np.float64)[0]

    m_train = shear.load(f"{data_dir}/{B.TRAIN_DATA[pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(flow_path, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    flow = bulk.SupportedFlow(flow, eps=0.0, support=True)
    return flow, cov, sigma_x


def fd_selection_terms(flow, cov, sigma_x, window_size, window_flux,
                       n, h=0.02, chunk=16384, seed=0):
    """(grand, per_chunk) -- P_s/Q_s/R_s at g=0 by the bounded-F finite-
    difference method (module docstring), plus a per-chunk breakdown for
    error bars.  `grand` and each entry of `per_chunk` are dicts with keys
    P_s, Q_s1, Q_s2, R_s11, R_s22, R_s12, iso_diff (= R_s11 - R_s22)."""
    sx = jnp.asarray(sigma_x, dtype=jnp.float64)
    stencil = {"00": (0.0, 0.0), "p0": (h, 0.0), "m0": (-h, 0.0),
              "0p": (0.0, h), "0m": (0.0, -h),
              "pp": (h, h), "pm": (h, -h), "mp": (-h, h), "mm": (-h, -h)}
    names = list(stencil)

    # One jitted transform per stencil point, closed over its own fixed g --
    # `g=jnp.array(gg)` as a default argument binds it at definition time, not
    # at call time, so the nine closures don't all end up pointing at the
    # loop's final value.
    tr = {name: eqx.filter_jit(jax.vmap(
        lambda z1, g=jnp.array(gg): flow.bijection.transform(
            z1, B.condition(g, sx))))
        for name, gg in stencil.items()}
    fprob = jax.jit(lambda mm: B.window_prob(mm, cov, window_size, window_flux))

    # Per-CHUNK stencil means, not just a running sum: chunks are independent
    # draws (only the 9 points WITHIN a chunk share common random numbers), so
    # treating each chunk's own derived P_s/Q_s/R_s as one iid sample gives a
    # real chunk-to-chunk standard error on every quantity below.
    chunk_P = []      # list of {name: mean F at this stencil point}, per chunk
    counts = 0
    for i in range(0, n, chunk):
        k = min(chunk, n - i)
        z = flow.base_dist.sample(jr.fold_in(jr.key(seed), i),
                                  (k,)).astype(jnp.float64)
        ms = {name: tr[name](z) for name in names}
        # A draw is kept only if EVERY stencil point is finite -- otherwise
        # dropping it at some points but not others would bias the paired
        # differences (same reasoning bias.py's own `--window-fd` uses).
        ok = jnp.all(jnp.stack(
            [jnp.isfinite(ms[name]).all(-1) for name in names]), axis=0)
        Fs = {name: jnp.where(ok, fprob(ms[name]), 0.0) for name in names}
        kk = int(ok.sum())
        counts += kk
        chunk_P.append({name: float(jnp.sum(Fs[name])) / max(kk, 1)
                        for name in names})

    def stat(P):
        p00 = P["00"]
        q1 = (P["p0"] - P["m0"]) / (2 * h)
        q2 = (P["0p"] - P["0m"]) / (2 * h)
        r11 = (P["p0"] - 2 * p00 + P["m0"]) / h ** 2
        r22 = (P["0p"] - 2 * p00 + P["0m"]) / h ** 2
        r12 = (P["pp"] - P["pm"] - P["mp"] + P["mm"]) / (4 * h ** 2)
        return dict(P_s=p00, Q_s1=q1, Q_s2=q2, R_s11=r11, R_s22=r22, R_s12=r12,
                   iso_diff=r11 - r22)

    # The GRAND estimate: derive P_s/Q_s/R_s from the mean stencil values
    # pooled over every chunk (equivalent to pooling all n draws directly).
    grand_P = {name: np.mean([c[name] for c in chunk_P]) for name in names}
    grand = stat(grand_P)
    grand["n_kept"] = counts
    per_chunk = [stat(c) for c in chunk_P]
    return grand, per_chunk


def se(per_chunk, key):
    vals = np.array([c[key] for c in per_chunk])
    n = len(vals)
    return vals.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", default="gauss2_v3e", choices=sorted(B.CATALOGS))
    p.add_argument("--flow", required=True)
    p.add_argument("--data-dir", default="../bfd_cnf_imsims/data")
    p.add_argument("--window-size", type=float, nargs=2, default=(2.2, 3.2))
    p.add_argument("--window-flux", type=float, nargs=2, default=(2500.0, 50000.0))
    p.add_argument("--n", type=int, default=1 << 20, help="prior draws")
    p.add_argument("--h", type=float, default=0.02,
                   help="finite-difference step, matched to ghat's |g| "
                        "operating point -- see module docstring")
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    flow, cov, sigma_x = load_flow_f64(a.pop, a.flow, a.data_dir)
    grand, per_chunk = fd_selection_terms(
        flow, cov, sigma_x, a.window_size, a.window_flux, a.n, a.h, a.chunk,
        a.seed)
    nchunks = len(per_chunk)

    print(f"flow {a.flow}   pop {a.pop}   {grand['n_kept']}/{a.n} draws kept "
          f"(finite at all 9 stencil points), {nchunks} chunks")
    print(f"C_M = {np.array2string(cov, precision=3)}")
    print(f"window size {a.window_size}  flux {a.window_flux}  h = {a.h}\n")

    for key, label in [("P_s", "P_s"), ("Q_s1", "Q_s1"), ("Q_s2", "Q_s2"),
                       ("R_s11", "R_s11"), ("R_s22", "R_s22"), ("R_s12", "R_s12"),
                       ("iso_diff", "R_s11 - R_s22")]:
        s = se(per_chunk, key)
        print(f"  {label:>14} = {grand[key]:+.5e}  +/- {s:.2e}  "
              f"({abs(grand[key]) / s if s else float('nan'):.1f} sigma from 0)")

    print(f"\nisotropy check (spin-0 window: R_s11 should equal R_s22, "
          f"R_s12 should be 0):")
    print(f"  R_s11 - R_s22 = {grand['iso_diff']:+.4e}  "
          f"(consistent with 0 at "
          f"{abs(grand['iso_diff']) / se(per_chunk, 'iso_diff'):.1f} sigma)")


if __name__ == "__main__":
    main()
