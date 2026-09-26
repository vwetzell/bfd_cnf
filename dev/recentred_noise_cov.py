"""Is the real targets' measurement noise the C_M the model assumes?

The model (flow + bias.py's convolution, and the closed loop by construction)
treats a recentred target's moments as the centroid layer's point plus
N(0, C_M), C_M being bfd's analytic covariance at fixed centre.  Real
targets are rendered with pixel noise and RECENTRED on that same noise, so
their moment noise is whatever recentring makes of it.  Here: in-window real
target galaxies, re-rendered with many independent noise fields exactly as
imsims.sim does (draw -> + pixel noise -> _measure with recenter), and the
empirical covariance compared with C_M along each galaxy's own shear
response v = dm/dg:  r = v^T Cov_emp v / v^T C_M v  (1 = model right).  A
~3% excess would match the real-vs-closed-loop score-variance excess.
"""
import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
sys.path.insert(0, "../bfd_cnf_imsims")
import fitsio
import numpy as np

import bias as B
from imsims import sim

D = "../bfd_cnf_imsims/data"
N_GAL, N_REAL = int(sys.argv[1]) if len(sys.argv) > 1 else 48, 300


def one(args):
    g, seed = args
    import bfd
    _, draw = sim.POPULATIONS["gauss2_fwd"]
    wt = bfd.KBlackmanHarris(weightSigma=sim.WEIGHT_SIGMA)
    img = draw(g)
    rng = np.random.default_rng(seed)
    out, bad = [], 0
    for _ in range(N_REAL):
        mc = sim._measure(img + rng.normal(0.0, sim.NOISE_SIGMA, img.shape), wt, sim.NOISE_SIGMA)
        row = np.empty(1, dtype=sim.ROW_DTYPE)[0]
        sim.fill_row(row, mc)
        if row["badcenter"]:
            bad += 1
            continue
        out.append(row["moments"])
    return np.array(out), bad


def main():
    path = f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits"
    t = fitsio.read(path)
    pop = fitsio.read(path, ext="POPULATION")
    win = np.flatnonzero(B.window_mask(t["moments"], (2.2, 3.2), (3000.0, 20000.0)) & ~t["badcenter"])
    pick = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0).choice(win, N_GAL, replace=False)
    CM = B.load_cov(path)
    with Pool(12) as p:
        res = p.map(one, [(pop[i], 1000 + k) for k, i in enumerate(pick)])
    rows = []
    for (ms, bad), i in zip(res, pick):
        C = np.cov(ms.T)
        v = t["dm_dg"][i]                                    # (2,5)
        r = np.einsum("ai,ij,aj->", v, C, v) / np.einsum("ai,ij,aj->", v, CM, v)
        e = np.hypot(*t["moments"][i, 2:4]) / t["moments"][i, 1]
        rows.append((t["moments"][i, 0], e, r, *(np.diag(C) / np.diag(CM)), bad))
    rows = np.array(rows)
    print(f"{N_GAL} in-window real target galaxies x {N_REAL} noise fields, recentred as imsims does")
    print("per-galaxy  Mf     |e|   v^T C v / v^T C_M v   diag Cov/C_M (Mf Mr M1 M2 Mc)   bad")
    for r in rows[np.argsort(rows[:, 1])]:
        print(f"  {r[0]:8.0f} {r[1]:.3f}   {r[2]:.3f}                " + " ".join(f"{x:.3f}" for x in r[3:8])
              + f"   {int(r[8])}")
    # sampling error of a variance ratio from N_REAL draws ~ sqrt(2/N_REAL) per galaxy
    se = np.sqrt(2 / N_REAL) / np.sqrt(N_GAL)
    print(f"\nmean response-direction ratio {rows[:, 2].mean():.4f} +/- {rows[:, 2].std() / np.sqrt(N_GAL):.4f}"
          f"   (pure-sampling floor {se:.4f})")
    print("mean diag ratios " + " ".join(f"{x:.4f}" for x in rows[:, 3:8].mean(0)))


if __name__ == "__main__":
    main()
