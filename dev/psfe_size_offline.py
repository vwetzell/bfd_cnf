"""Does tightening the Mr/Mf size ceiling shrink the PSF-ellipticity c-leak,
the same way it shrinks the ordinary m1 size-edge bias
(residual-is-the-size-edge memory, dev/size_offline.py)?

Reuses each mcfix config's saved --save-pqr file (per-target Q,R,obs do not
depend on the window) and computes FRESH selection terms at a few Mr/Mf
ceilings via B.selection_terms_score's own multi-window pass (one set of
prior draws serves every ceiling). Matched-pairs aligns each config against
psfe00 (as in psfe_joint_slope_mcfix.py) and reports the dc1/d(psf_e1),
dc2/d(psf_e2) slope at each ceiling.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.spatial import cKDTree

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk
import shear

FLOW_PATH = "flows/centroid_g2v3d_sigmaxblock_mc_lr3e4_s0.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
FLUX = (2500.0, 50000.0)
CEILINGS = [3.2, 3.1, 3.0, 2.9]   # 3.2 = nominal
LOG2_DRAWS = 22
G = 0.02
N_BOOT = 200

CONFIGS = {
    "psfe1p05": (0.05, 0.0), "psfe2p05": (0.0, 0.05), "psfe1m05": (-0.05, 0.0),
}
ALL_TAGS = ["psfe00"] + list(CONFIGS)


def build_flow_once():
    """`jax_enable_x64` is a sticky, process-global JAX setting -- rebuilding
    +deserialising the flow more than once per process (once per config, as
    an earlier version of this script did) breaks the SECOND call onward,
    since `build_flow`'s `like` template is already float64 by then while
    the on-disk checkpoint is float32. Build once, reuse across every
    config -- correct AND cheaper, since it's the same checkpoint each time,
    only Sigma_X/cov (loaded per-config in the caller) differ."""
    pop = f"gauss2_v3d_{ALL_TAGS[0]}"
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW_PATH, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    return bulk.SupportedFlow(flow, eps=0.0, support=True)


def selection_terms_all_ceilings(flow, tag):
    """(cov, {ceiling: (ps, qs, rs)}) for one config, one GPU pass over
    LOG2_DRAWS prior draws shared across every ceiling in CEILINGS."""
    pop = f"gauss2_v3d_{tag}"
    targets = f"{DATA_DIR}/{B.CATALOGS[pop]['zero']}.fits"
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets, rows=[0])["cov_odd"], dtype=np.float64)[0])

    n = 1 << LOG2_DRAWS
    tr = eqx.filter_jit(jax.vmap(
        lambda z1: flow.bijection.transform(z1, B.condition(jnp.zeros(2), sigma_x))))
    zs = flow.base_dist.sample(jr.key(31), (n,)).astype(jnp.float64)
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384])) for i in range(0, n, 16384)])
    del zs

    windows = [((2.2, hi), FLUX) for hi in CEILINGS]
    terms = B.selection_terms_score(flow, m0, cov, None, None, sigma_x=sigma_x,
                                    windows=windows, guard=100)
    del m0
    return cov, {hi: (float(t[0]), t[1], t[2]) for hi, t in zip(CEILINGS, terms)}


def pairwise_match(base_moments, other_moments):
    """base_idx -> other_idx dict, built ONCE (mirrors
    dev/psfe_joint_slope_mcfix.py's function of the same purpose -- an
    earlier version of this script rebuilt the equivalent set/dict on every
    single bootstrap draw instead of once outside the loop, which is why it
    ran for 90+ minutes without finishing even the first ceiling)."""
    tree = cKDTree(base_moments)
    dist, j = tree.query(other_moments)
    self_dist, _ = cKDTree(other_moments).query(other_moments, k=2)
    scale = np.median(self_dist[:, 1])
    ok = dist < 0.5 * scale
    uniq, counts = np.unique(j[ok], return_counts=True)
    keep_base = set(uniq[counts == 1])
    return {int(j[k]): k for k in np.flatnonzero(ok) if int(j[k]) in keep_base}


def main():
    pqr = {t: np.load(f"pqr/g2v3d_{t}_mcfix.npz") for t in ALL_TAGS}
    print("computing fresh selection terms at each ceiling (one GPU pass per config)...")
    flow = build_flow_once()
    terms_by_tag = {}
    for t in ALL_TAGS:
        print(f"  {t} ...", flush=True)
        _, terms_by_tag[t] = selection_terms_all_ceilings(flow, t)

    matches = {tag: pairwise_match(pqr["psfe00"]["moments"], pqr[tag]["moments"])
               for tag in CONFIGS}
    print("pairwise match counts:", {t: len(m) for t, m in matches.items()})

    def obs_and_arrays(d, idx, size):
        sel = (B.window_mask(d["obs_plus"][idx], size, FLUX),
               B.window_mask(d["obs_minus"][idx], size, FLUX))
        n_out = ((~sel[0]).sum(), (~sel[1]).sum())
        return (d["plus_q"][idx], d["plus_r"][idx], d["minus_q"][idx], d["minus_r"][idx]), sel, n_out

    def wls_slope(xs, ys):
        xs, ys = np.asarray(xs), np.asarray(ys)
        return np.sum(xs * ys) / np.sum(xs * xs)

    x_c1 = {t: CONFIGS[t][0] for t in CONFIGS}
    x_c2 = {t: CONFIGS[t][1] for t in CONFIGS}

    print(f"\n{'ceiling':>8s} {'dc1/d(psf_e1)':>22s} {'dc2/d(psf_e2)':>22s}")
    for hi in CEILINGS:
        size = (2.2, hi)

        def diffs_for(base_idx_full):
            out = {}
            for tag, m in matches.items():
                base_ok = np.array([i for i in base_idx_full if i in m])
                other_ok = np.array([m[i] for i in base_ok])
                b_arrs, b_sel, b_nout = obs_and_arrays(pqr["psfe00"], base_ok, size)
                bp, bq, br = terms_by_tag["psfe00"][hi]
                b_ns = ((float(b_nout[0]), bp, bq, br), (float(b_nout[1]), bp, bq, br))
                arrs, sel, nout = obs_and_arrays(pqr[tag], other_ok, size)
                p, q, r = terms_by_tag[tag][hi]
                ns = ((float(nout[0]), p, q, r), (float(nout[1]), p, q, r))
                b_base = B.bias(*b_arrs, G, sel=b_sel, ns=b_ns)
                b_other = B.bias(*arrs, G, sel=sel, ns=ns)
                out[tag] = (b_other[1] - b_base[1], b_other[2] - b_base[2])
            return out

        n_base = len(pqr["psfe00"]["moments"])
        t0 = time.time()
        d0 = diffs_for(np.arange(n_base))
        print(f"  [{hi:.2f}] point estimate took {time.time()-t0:.1f}s", flush=True)
        y_c1_0 = [d0[t][0] for t in CONFIGS]
        y_c2_0 = [d0[t][1] for t in CONFIGS]
        s_c1 = wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                         [y for t, y in zip(CONFIGS, y_c1_0) if x_c1[t] != 0])
        s_c2 = wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                         [y for t, y in zip(CONFIGS, y_c2_0) if x_c2[t] != 0])

        rng = np.random.default_rng(0)
        s_c1_boot, s_c2_boot = [], []
        t0 = time.time()
        for bi in range(N_BOOT):
            idx = rng.choice(n_base, n_base)
            d = diffs_for(idx)
            if bi == 4:
                print(f"  [{hi:.2f}] {bi+1}/{N_BOOT} boot draws in "
                      f"{time.time()-t0:.1f}s -> est. total {N_BOOT*(time.time()-t0)/(bi+1):.0f}s",
                      flush=True)
            y1 = [d[t][0] for t in CONFIGS]
            y2 = [d[t][1] for t in CONFIGS]
            s_c1_boot.append(wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                                       [y for t, y in zip(CONFIGS, y1) if x_c1[t] != 0]))
            s_c2_boot.append(wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                                       [y for t, y in zip(CONFIGS, y2) if x_c2[t] != 0]))
        sd1, sd2 = np.std(s_c1_boot), np.std(s_c2_boot)
        print(f"{hi:8.2f}  {s_c1:+.5f} +/- {sd1:.5f} ({s_c1/sd1:+.2f}s)  "
              f"{s_c2:+.5f} +/- {sd2:.5f} ({s_c2/sd2:+.2f}s)", flush=True)


if __name__ == "__main__":
    main()
