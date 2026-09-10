"""Logit chart vs bare-ratio chart, on the same data.

Three questions, in the order they decide anything:

1. **Is the fit better or worse?**  The flow's `log_prob` includes the chart's
   own log-det, so the NLL is a density on RAW MOMENTS and is directly
   comparable between charts.  Reported on the training prior and, more
   importantly, on the g=0 TARGETS (held out from training, and noisy).

2. **Is the extrapolation cliff gone?**  With the logit, moving 0.1 in slot 1
   past the data envelope sent the base coordinate to 1.9e7 and log p to
   -1.8e14.  Same probe here.

3. **How many kernel draws does the estimator lose?**  With the logit, 26% of
   draws were poisoned (log p < -1e4) even for targets far from the ceiling,
   and the poisoned set moved with g -- so Q and R differentiated a moving
   mask.  Recount.

Evaluates ONE flow, because a checkpoint only means anything under the chart
CODE it was trained with: the old checkpoint restores its own mean/std but
would be pushed through the new `_forward_transform`.  So run this twice --
here for the bare-ratio chart, and inside a `git worktree` at the pre-change
commit for the logit one -- and compare the printed numbers.

    PYTHONPATH=. python dev/chart_compare.py flows/bulk_v3.eqx
"""
import os
import sys

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias                                              # noqa: E402
import bulk                                              # noqa: E402
import shear as sm                                       # noqa: E402
from models.bijections import in_domain                  # noqa: E402

D = "../bfd_cnf_imsims/data"
FLOW = sys.argv[1] if len(sys.argv) > 1 else "flows/bulk_v3.eqx"
TAG = os.environ.get("TAG", os.path.basename(FLOW))
N = 20000


def load(path, m_train):
    flow = bulk.build_flow(jr.key(0), m_train)
    return eqx.tree_deserialise_leaves(path, flow)


def main():
    m, _, _ = sm.load(f"{D}/moments_bulgedisc_v3.fits")
    m = np.asarray(m, np.float64)
    tg = fitsio.read(f"{D}/targets_v3_g0_200k.fits", rows=np.arange(N))
    M = np.asarray(tg["moments"], np.float64)
    cov = np.asarray(bias.load_cov(f"{D}/targets_v3_g0_200k.fits"), np.float64)
    if cov.ndim == 3:
        cov = cov[0]

    print("NLL on RAW MOMENTS (chart log-det included), lower is better:")
    print(f"  {'chart':>12s} {'prior (train)':>15s} {'g=0 targets':>14s} "
          f"{'frac poisoned':>14s}")
    flows = {}
    for lab, path in ((TAG, FLOW),):
        f = load(path, m)
        flows[lab] = f
        out = []
        for x in (m[:N], M):
            xx = jnp.asarray(x, jnp.float32)
            ok = np.asarray(in_domain(xx))
            lp = np.asarray(jax.vmap(f.log_prob)(xx))
            good = ok & np.isfinite(lp) & (lp > -1e4)
            out.append((-lp[good].mean(), 1.0 - good.mean()))
        print(f"  {lab:>12s} {out[0][0]:15.4f} {out[1][0]:14.4f} "
              f"{out[1][1]:14.2%}")

    # 2. the extrapolation probe: walk slot 1 out past the data
    print("\nextrapolation probe -- log p and |base| walking slot 1 outward\n"
          "from the most extreme training point:")
    for lab, f in flows.items():
        # Perturb in CHART space but hand the STACK raw moments -- its first
        # element IS the chart, so feeding it standardised coordinates is a
        # category error (it reads slot 0 as log10 Mf and slot 1 as a ratio).
        chart = f.bijection.bijection.bijections[0]
        size = m[:, 1] / m[:, 0]
        i = int(np.argmax(size))
        z_i = chart.transform(jnp.asarray(m[i], jnp.float32))
        print(f"  --- {lab} (most extreme training row: Mr/Mf = {size[i]:.4f}, "
              f"z1 = {float(z_i[1]):.3f})")
        for d in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            z = z_i.at[1].add(d)
            m_pert = chart.inverse(z)
            u = np.asarray(f.bijection.transform(m_pert))
            lp = float(f.log_prob(m_pert))
            print(f"      z1 +{d:4.2f} (Mr/Mf {float(m_pert[1]/m_pert[0]):6.3f})"
                  f"  |base| {np.linalg.norm(u):11.4g}   log p {lp:13.5g}")

    # 3. poisoned kernel draws
    print("\npoisoned kernel draws (log p < -1e4 among in_domain draws), and\n"
          "whether that set moves with g:")
    a = np.zeros(5)
    from models.bijections import POINT_SOURCE
    a[0], a[1] = POINT_SOURCE, -1.0
    t = M @ a / np.sqrt(a @ cov @ a)
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(0)
    S = 2048
    for lab, f in flows.items():
        print(f"  --- {lab}")
        for lo, hi, name in ((-9, 1, "< 1 sigma"), (1, 3, "1-3"), (3, 99, "> 3")):
            idx = np.where((t >= lo) & (t < hi))[0][:40]
            if len(idx) == 0:
                continue
            dom, pois, dp = [], [], []
            for i in idx:
                dr = jnp.asarray(M[i] + rng.standard_normal((S, 5)) @ L.T,
                                 jnp.float32)
                ok = np.asarray(in_domain(dr))
                # bulk alone is g-independent, so one evaluation is enough;
                # the g-dependence of the mask is a property of the full chain
                # and is measured there, not here.
                lp = np.asarray(jax.vmap(f.log_prob)(dr))
                dom.append(ok.sum()); pois.append(int(((lp < -1e4) & ok).sum()))
            dom, pois = np.array(dom), np.array(pois)
            print(f"      {name:>10s}  in_domain {dom.mean()/S:6.2%}   "
                  f"poisoned {pois.mean()/S:6.2%}   usable "
                  f"{(dom-pois).mean()/S:6.2%}")


if __name__ == "__main__":
    main()
