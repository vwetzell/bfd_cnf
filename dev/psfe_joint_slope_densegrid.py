"""Same joint-bootstrap slope fit as psfe_joint_slope.py, restricted to the
3 amplitude-0.05 configs and pointed at the densegrid-checkpoint PQR/terms
files -- validates whether densifying ANISO_TRAIN_POINTS near the actually-
probed e_Sigma_X range shrinks the c1/c2 PSF leak found on the baseline
multiscale checkpoint. See PSFE_PROVENANCE.md #8d for the baseline slope
this compares against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

import bias as B

SIZE, FLUX = (2.2, 3.2), (2500.0, 50000.0)
G = 0.02
N_BOOT = 400
SUFFIX = "_densegrid"

CONFIGS = {
    "psfe1p05": (0.05, 0.0), "psfe2p05": (0.0, 0.05), "psfe1m05": (-0.05, 0.0),
}


def load(tag):
    d = np.load(f"pqr/g2v3d_{tag}{SUFFIX}.npz")
    terms = np.load(f"logs/phase_a/terms_gauss2_v3d_{tag}_24_s0_1w_g100{SUFFIX}.npz")
    return d, (float(terms["ps"][0]), terms["qs"][0], terms["rs"][0])


def pairwise_match(base_moments, other_moments):
    tree = cKDTree(base_moments)
    dist, j = tree.query(other_moments)
    self_dist, _ = cKDTree(other_moments).query(other_moments, k=2)
    scale = np.median(self_dist[:, 1])
    ok = dist < 0.5 * scale
    uniq, counts = np.unique(j[ok], return_counts=True)
    keep_base = set(uniq[counts == 1])
    return {int(j[k]): k for k in np.flatnonzero(ok) if int(j[k]) in keep_base}


def main():
    base, base_terms = load("psfe00")
    others = {tag: load(tag) for tag in CONFIGS}
    n_base = len(base["moments"])

    matches = {tag: pairwise_match(base["moments"], d["moments"])
               for tag, (d, _) in others.items()}
    print("pairwise match counts:", {t: len(m) for t, m in matches.items()})

    def obs_and_arrays(d, idx):
        sel = (B.window_mask(d["obs_plus"][idx], SIZE, FLUX),
               B.window_mask(d["obs_minus"][idx], SIZE, FLUX))
        n_out = ((~sel[0]).sum(), (~sel[1]).sum())
        return (d["plus_q"][idx], d["plus_r"][idx], d["minus_q"][idx], d["minus_r"][idx]), sel, n_out

    def biasf(arrs, sel, ns):
        return B.bias(*arrs, G, sel=sel, ns=ns)

    def wls_slope(xs, ys):
        xs, ys = np.asarray(xs), np.asarray(ys)
        return np.sum(xs * ys) / np.sum(xs * xs)

    x_c1 = {t: CONFIGS[t][0] for t in CONFIGS}
    x_c2 = {t: CONFIGS[t][1] for t in CONFIGS}

    def diffs_for(base_idx_full):
        out = {}
        for tag, (d, terms) in others.items():
            m = matches[tag]
            base_ok = np.array([i for i in base_idx_full if i in m])
            other_ok = np.array([m[i] for i in base_ok])
            b_arrs, b_sel, b_nout = obs_and_arrays(base, base_ok)
            b_ns = ((float(b_nout[0]), *base_terms), (float(b_nout[1]), *base_terms))
            arrs, sel, nout = obs_and_arrays(d, other_ok)
            ns = ((float(nout[0]), *terms), (float(nout[1]), *terms))
            b_base_i = biasf(b_arrs, b_sel, b_ns)
            b = biasf(arrs, sel, ns)
            out[tag] = (b[1] - b_base_i[1], b[2] - b_base_i[2])
        return out

    diffs0 = diffs_for(np.arange(n_base))
    y_c1_0 = [diffs0[t][0] for t in CONFIGS]
    y_c2_0 = [diffs0[t][1] for t in CONFIGS]
    slope_c1_0 = wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                           [y for t, y in zip(CONFIGS, y_c1_0) if x_c1[t] != 0])
    slope_c2_0 = wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                           [y for t, y in zip(CONFIGS, y_c2_0) if x_c2[t] != 0])
    print(f"\npoint estimate: dc1/d(psf_e1) = {slope_c1_0:+.5f}   "
          f"dc2/d(psf_e2) = {slope_c2_0:+.5f}")

    rng = np.random.default_rng(0)
    slope_c1_boot, slope_c2_boot = [], []
    for _ in range(N_BOOT):
        idx = rng.choice(n_base, n_base)
        diffs = diffs_for(idx)
        y_c1 = [diffs[t][0] for t in CONFIGS]
        y_c2 = [diffs[t][1] for t in CONFIGS]
        slope_c1_boot.append(wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                                       [y for t, y in zip(CONFIGS, y_c1) if x_c1[t] != 0]))
        slope_c2_boot.append(wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                                       [y for t, y in zip(CONFIGS, y_c2) if x_c2[t] != 0]))

    sd_c1 = np.std(slope_c1_boot)
    sd_c2 = np.std(slope_c2_boot)
    print(f"joint-bootstrap error: dc1/d(psf_e1) = {slope_c1_0:+.5f} +/- {sd_c1:.5f} "
          f"({slope_c1_0/sd_c1:+.2f} sigma)")
    print(f"joint-bootstrap error: dc2/d(psf_e2) = {slope_c2_0:+.5f} +/- {sd_c2:.5f} "
          f"({slope_c2_0/sd_c2:+.2f} sigma)")


if __name__ == "__main__":
    main()
