"""Is m1 flat in flux and size, once the near-unresolved tail is cut?

Both axes are NOISE-FREE.  The targets' own moments share their noise draw with
Q (all three deep catalogs are SEED 1), so binning on them manufactures
opposite-sign structure -- that is what produced the spurious +4%/-49% pattern.
`dm_dg` / `d2m_dg2` on disk are exact per-galaxy derivatives, so a regression of
a moment onto them is a prediction with no target noise in it at all.

The g=0 arm is carried through every cell as a null: its truth is exactly zero
regardless of shear response, so a cell that fails it has a wrong density, not
a wrong calibration.
"""
import sys

import fitsio
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, ".")

import bias                                          # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SP = ("/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/"
      "c729e54d-31f4-414c-8c5d-6d6a496207d0/scratchpad")
RCUT, NBOOT, NF, NR = 3.5, 300, 4, 4


def noise_free(z, y, sel):
    """E[y | noiseless per-galaxy features], fitted out-of-fold."""
    F = np.concatenate([np.asarray(z["dm_dg"], np.float64).reshape(len(z), -1),
                        np.asarray(z["d2m_dg2"], np.float64).reshape(len(z), -1)], 1)
    mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                        max_depth=6, l2_regularization=1.0,
                                        random_state=0)
    out = np.full(len(z), np.nan)
    out[sel] = cross_val_predict(mdl, F[sel], y[sel], cv=5)
    return out


def main():
    cat = bias.CATALOGS["bulgedisc_deep"]
    rows = {k: fitsio.read(f"{DATA}/{v}.fits")[:20000] for k, v in cat.items()}
    keep = np.ones(len(rows["zero"]), bool)
    for v in rows.values():
        keep &= ~v["badcenter"]
    z = rows["zero"][keep]
    M0 = np.asarray(z["moments"], np.float64)
    C = bias.load_cov(f"{DATA}/{cat['zero']}.fits")
    sel = M0[:, 0] > 2800

    fhat = noise_free(z, M0[:, 0], sel)
    rhat = noise_free(z, M0[:, 1] / M0[:, 0], sel)
    for nm, y, yh, s2 in (("Mf", M0[:, 0], fhat, C[0, 0]),
                          ("Mr/Mf", M0[:, 1] / M0[:, 0], rhat, None)):
        tot = y[sel].var()
        if s2 is None:
            J = np.stack([-M0[:, 1] / M0[:, 0] ** 2, 1 / M0[:, 0]], 1)
            s2 = float(np.median((J @ C[:2, :2] * J).sum(1)[sel]))
        print(f"  noise-free {nm:6s}: R^2={1 - ((y - yh)[sel] ** 2).mean() / tot:.4f} "
              f"(ceiling {1 - s2 / tot:.4f}), keeps "
              f"{yh[sel].std() / np.sqrt(tot - s2):.2f} of the intrinsic spread")

    # The default is the baseline run; pass another --save-pqr npz to compare a
    # retrained flow against it on the SAME noise-free binning.
    d = np.load(sys.argv[1] if len(sys.argv) > 1 else f"{SP}/paired_with.npz")
    print(f"\nPQR from {sys.argv[1] if len(sys.argv) > 1 else 'paired_with.npz'}")
    rng = np.random.default_rng(0)

    def est(s):
        gp, gm = bias.ghat(d["plus_q"], d["plus_r"], s), bias.ghat(d["minus_q"], d["minus_r"], s)
        return (gp[0] - gm[0]) / 0.04 - 1, bias.ghat(d["zero_q"], d["zero_r"], s)[0]

    def boot(s):
        i0 = np.flatnonzero(s)
        v = np.array([est(rng.choice(i0, len(i0))) for _ in range(NBOOT)])
        return v.std(0)

    base = sel & (rhat < RCUT)
    m, z0 = est(base)
    sm, sz = boot(base)
    print(f"\nbase selection: Mf>2800 and noise-free Mr/Mf<{RCUT}, N={base.sum()}")
    print(f"  m1 = {m:+.4f} +/- {sm:.4f}     ghat(g=0) = {z0:+.4f} +/- {sz:.4f}")

    fe = np.quantile(fhat[base], np.linspace(0, 1, NF + 1)); fe[0], fe[-1] = -np.inf, np.inf
    re = np.quantile(rhat[base], np.linspace(0, 1, NR + 1)); re[0], re[-1] = -np.inf, np.inf
    cells = []
    print(f"\nm1 on a {NF}x{NR} noise-free grid  (rows: flux quartile, cols: Mr/Mf quartile)")
    hdr = "".join(f"{f'{re[j]:.2f}-{re[j + 1]:.2f}':>19s}" for j in range(NR))
    print(f"{'flux (Mf)':>17s}" + hdr)
    for i in range(NF):
        line = f"{f'{fe[i]:.0f}-{fe[i + 1]:.0f}':>17s}"
        for j in range(NR):
            s = base & (fhat >= fe[i]) & (fhat < fe[i + 1]) & (rhat >= re[j]) & (rhat < re[j + 1])
            if s.sum() < 80:
                line += f"{'--':>19s}"; continue
            v, _ = est(s); e = boot(s)[0]
            cells.append((v, e, s.sum()))
            line += f"{f'{v:+.4f}+/-{e:.4f}':>19s}"
        print(line)

    v = np.array([c[0] for c in cells]); e = np.array([c[1] for c in cells])
    chi2 = float((((v - m) / e) ** 2).sum())
    print(f"\nflatness: chi2 = {chi2:.1f} on {len(cells) - 1} dof "
          f"(chi2/dof = {chi2 / (len(cells) - 1):.2f}); "
          f"cell spread {v.std():.4f} vs median cell error {np.median(e):.4f}")

    for nm, hat, edges in (("flux Mf", fhat, fe), ("Mr/Mf", rhat, re)):
        print(f"\nmarginal in {nm}")
        print(f"{'bin':>16s} {'N':>6s} {'m1':>19s} {'ghat(g=0)':>19s}")
        for k in range(len(edges) - 1):
            s = base & (hat >= edges[k]) & (hat < edges[k + 1])
            (a, b), (sa, sb) = est(s), boot(s)
            print(f"{f'{edges[k]:.2f}-{edges[k + 1]:.2f}':>16s} {s.sum():6d} "
                  f"{f'{a:+.4f} +/- {sa:.4f}':>19s} {f'{b:+.4f} +/- {sb:.4f}':>19s}")


if __name__ == "__main__":
    main()
