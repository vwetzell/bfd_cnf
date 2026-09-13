"""Red-team check on dev/psfe_size_offline.py: that script computes all 4
ceilings from ONE shared prior-draw stream per config (deliberately paired,
per its own docstring), but selection_terms_score's own diagnostics showed
R_s11's outlier concentration shrinking smoothly with the ceiling (top draw
2.3% -> 0.4%) -- so some of the measured dc1 improvement could be that one
high-leverage draw's contribution dropping out of the quadrature, not a
real reduction in the population's leak. This reruns each ceiling with an
INDEPENDENT draw seed (not shared) to see if the same trend survives.
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

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk
import shear

FLOW_PATH = "flows/centroid_g2v3d_sigmaxblock_mc_lr3e4_s0.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
FLUX = (2500.0, 50000.0)
CEILINGS = [3.2, 3.1, 3.0, 2.9]
LOG2_DRAWS = 22
G = 0.02
N_BOOT = 200

CONFIGS = {
    "psfe1p05": (0.05, 0.0), "psfe2p05": (0.0, 0.05), "psfe1m05": (-0.05, 0.0),
}
ALL_TAGS = ["psfe00"] + list(CONFIGS)


def build_flow_once():
    pop = f"gauss2_v3d_{ALL_TAGS[0]}"
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW_PATH, flow)
    jax.config.update("jax_enable_x64", True)
    flow = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, flow)
    return bulk.SupportedFlow(flow, eps=0.0, support=True)


def selection_terms_one_ceiling(flow, tag, hi, seed):
    """INDEPENDENT prior draws per (tag, ceiling), unlike
    psfe_size_offline.py's one-shared-stream-per-config design -- the point
    of this script."""
    pop = f"gauss2_v3d_{tag}"
    targets = f"{DATA_DIR}/{B.CATALOGS[pop]['zero']}.fits"
    cov = B.load_cov(targets)
    sigma_x = jnp.asarray(
        np.asarray(fitsio.read(targets, rows=[0])["cov_odd"], dtype=np.float64)[0])

    n = 1 << LOG2_DRAWS
    tr = eqx.filter_jit(jax.vmap(
        lambda z1: flow.bijection.transform(z1, B.condition(jnp.zeros(2), sigma_x))))
    zs = flow.base_dist.sample(jr.key(seed), (n,)).astype(jnp.float64)
    m0 = np.concatenate([np.asarray(tr(zs[i:i + 16384])) for i in range(0, n, 16384)])
    del zs

    ps, qs, rs, _ = B.selection_terms_score(flow, m0, cov, (2.2, hi), FLUX,
                                            sigma_x=sigma_x, guard=100)
    del m0
    return cov, (float(ps), qs, rs)


def pairwise_match(base_moments, other_moments):
    from scipy.spatial import cKDTree
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
    flow = build_flow_once()

    matches = {tag: pairwise_match(pqr["psfe00"]["moments"], pqr[tag]["moments"])
               for tag in CONFIGS}

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

    print(f"\n{'ceiling':>8s} {'seed':>6s} {'dc1/d(psf_e1)':>22s} {'dc2/d(psf_e2)':>22s}")
    # Two independent seeds per ceiling, so we can see draw-to-draw scatter
    # directly, not just compare against the shared-stream run.
    for seed_tag, seed_base in [("A", 101), ("B", 202)]:
        for hi in CEILINGS:
            size = (2.2, hi)
            t0 = time.time()
            terms_by_tag = {}
            for i, t in enumerate(ALL_TAGS):
                _, terms_by_tag[t] = selection_terms_one_ceiling(flow, t, hi, seed_base + i)
            print(f"  [{seed_tag} {hi:.2f}] terms computed in {time.time()-t0:.1f}s", flush=True)

            def diffs_for(base_idx_full):
                out = {}
                for tag, m in matches.items():
                    base_ok = np.array([i for i in base_idx_full if i in m])
                    other_ok = np.array([m[i] for i in base_ok])
                    b_arrs, b_sel, b_nout = obs_and_arrays(pqr["psfe00"], base_ok, size)
                    bp, bq, br = terms_by_tag["psfe00"]
                    b_ns = ((float(b_nout[0]), bp, bq, br), (float(b_nout[1]), bp, bq, br))
                    arrs, sel, nout = obs_and_arrays(pqr[tag], other_ok, size)
                    p, q, r = terms_by_tag[tag]
                    ns = ((float(nout[0]), p, q, r), (float(nout[1]), p, q, r))
                    b_base = B.bias(*b_arrs, G, sel=b_sel, ns=b_ns)
                    b_other = B.bias(*arrs, G, sel=sel, ns=ns)
                    out[tag] = (b_other[1] - b_base[1], b_other[2] - b_base[2])
                return out

            n_base = len(pqr["psfe00"]["moments"])
            d0 = diffs_for(np.arange(n_base))
            y_c1_0 = [d0[t][0] for t in CONFIGS]
            y_c2_0 = [d0[t][1] for t in CONFIGS]
            s_c1 = wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                             [y for t, y in zip(CONFIGS, y_c1_0) if x_c1[t] != 0])
            s_c2 = wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                             [y for t, y in zip(CONFIGS, y_c2_0) if x_c2[t] != 0])

            rng = np.random.default_rng(0)
            s_c1_boot, s_c2_boot = [], []
            for _ in range(N_BOOT):
                idx = rng.choice(n_base, n_base)
                d = diffs_for(idx)
                y1 = [d[t][0] for t in CONFIGS]
                y2 = [d[t][1] for t in CONFIGS]
                s_c1_boot.append(wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                                           [y for t, y in zip(CONFIGS, y1) if x_c1[t] != 0]))
                s_c2_boot.append(wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                                           [y for t, y in zip(CONFIGS, y2) if x_c2[t] != 0]))
            sd1, sd2 = np.std(s_c1_boot), np.std(s_c2_boot)
            print(f"{hi:8.2f} {seed_tag:>6s}  {s_c1:+.5f} +/- {sd1:.5f} ({s_c1/sd1:+.2f}s)  "
                  f"{s_c2:+.5f} +/- {sd2:.5f} ({s_c2/sd2:+.2f}s)", flush=True)


if __name__ == "__main__":
    main()
