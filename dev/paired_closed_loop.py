"""Paired real-vs-closed-loop comparison on two saved --save-pqr runs of the
SAME flow: real targets, and the closed loop (targets drawn from that flow,
dev/closed_loop.py catalogs), where the model is exact by construction.

1. m1(real) - m1(closed), matched pairs.  Each real target is paired with its
   nearest closed-loop target in shape space (g = 0 moments: log Mf, Mr/Mf,
   e1, e2, Mc/Mr, standardised), and the difference is bootstrapped over
   PAIRS, so the shape noise the twins share cancels.  Selection terms are the
   flow's CRN finite-difference truth (the same flow -> the same P_s, R_s for
   both); N_ns is each catalog's own population count.
2. Fisher identity per flux quintile: sum q q^T / sum(-r), which is exactly 1
   in expectation when the model is right (the closed loop), for real and
   closed, and their paired difference.  A per-target Q/R misfit shows here
   even when the population-mean response is right.
"""
import argparse
import sys
sys.path.insert(0, ".")

import fitsio
import numpy as np
from scipy.spatial import cKDTree

import bias as B

D = "../bfd_cnf_imsims/data"
SIZE, FLUX, G = (2.2, 3.2), (3000.0, 20000.0), 0.02


def feats(m):
    return np.stack([np.log(m[:, 0]), m[:, 1] / m[:, 0], m[:, 2] / m[:, 1],
                     m[:, 3] / m[:, 1], m[:, 4] / m[:, 1]], 1)


def load(path, npop):
    d = dict(np.load(path))
    d["sp"] = B.window_mask(d["obs_plus"], SIZE, FLUX)
    d["sm"] = B.window_mask(d["obs_minus"], SIZE, FLUX)
    d["nsp"], d["nsm"] = npop - d["sp"].sum(), npop - d["sm"].sum()
    return d


def m1(d, idx, ns_scale, rs):
    """Corrected and uncorrected m1 on the rows `idx` (with repeats)."""
    R = np.diag([rs, rs]); z = np.zeros(2)
    sp, sm = d["sp"][idx], d["sm"][idx]
    args = (d["plus_q"][idx], d["plus_r"][idx], d["minus_q"][idx], d["minus_r"][idx], G)
    unc = B.bias(*args, sel=(sp, sm))[0]
    ns = ((d["nsp"] * ns_scale, PS, z, R), (d["nsm"] * ns_scale, PS, z, R))
    cor = B.bias(*args, sel=(sp, sm), ns=ns)[0]
    return unc, cor


def fisher(d, idx):
    """sum |q|^2 / sum(-tr r) over both arms' in-window rows (both components)."""
    num = den = 0.0
    for arm, s in (("plus", "sp"), ("minus", "sm")):
        k = idx[d[s][idx]]
        num += (d[f"{arm}_q"][k] ** 2).sum()
        den += (-d[f"{arm}_r"][k, 0, 0] - d[f"{arm}_r"][k, 1, 1]).sum()
    return num / den


def fisher_split(R, C, j, var, name, rng, nq=5):
    """Paired Fisher excess (real - closed) by quantile of `var` over in-window real rows."""
    w = B.window_mask(R["moments"], SIZE, FLUX)
    q = np.quantile(var[w], np.linspace(0, 1, nq + 1)); q[-1] += 1e-9
    print(f"\n   by {name}:")
    for lo, hi in zip(q[:-1], q[1:]):
        k = np.flatnonzero((var >= lo) & (var < hi))
        d0 = fisher(R, k) - fisher(C, j[k])
        bs = [fisher(R, b) - fisher(C, j[b]) for b in (rng.choice(k, len(k)) for _ in range(300))]
        print(f"     {lo:9.4g}-{hi:<9.4g} excess {d0:+.4f}+/-{np.std(bs):.4f}")


def c2st_obs(rng):
    """Classifier: real vs closed-loop NOISY g=0 observations in the window, with a
    closed-vs-closed (disjoint halves of the iid flow population) baseline."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rd = fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n']['zero']}.fits")["moments"]
    cd = fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n_closed']['zero']}.fits")["moments"]
    rd, cd = rd[B.window_mask(rd, SIZE, FLUX)], cd[B.window_mask(cd, SIZE, FLUX)]
    cd = cd[rng.permutation(len(cd))]
    half = len(cd) // 2
    n = min(len(rd), half)

    def auc(a, b):
        X = np.concatenate([feats(a), feats(b)]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        aucs = []
        for tr, te in ((slice(0, None, 2), slice(1, None, 2)), (slice(1, None, 2), slice(0, None, 2))):
            p = rng.permutation(len(y)); Xp, yp = X[p], y[p]
            pr = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15).fit(Xp[tr], yp[tr]).predict_proba(Xp[te])[:, 1]
            aucs.append(roc_auc_score(yp[te], pr))
        return np.mean(aucs)

    base = [auc(cd[:half][rng.choice(half, n, False)], cd[half:][rng.choice(len(cd) - half, n, False)]) for _ in range(5)]
    real = [auc(rd[rng.choice(len(rd), n, False)], cd[rng.choice(len(cd), n, False)]) for _ in range(5)]
    print(f"\n3. C2ST on noisy in-window g=0 observations ({n} vs {n}): "
          f"real-vs-flow AUC {np.mean(real):.4f}+/-{np.std(real):.4f}   "
          f"flow-vs-flow baseline {np.mean(base):.4f}+/-{np.std(base):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", default="pqr/g2v4n_K_20k.npz")
    p.add_argument("--closed", default="pqr/closed_g2v4n_K.npz")
    p.add_argument("--rs", type=float, default=0.2337, help="R_s11 = R_s22 (CRN FD truth)")
    p.add_argument("--ps", type=float, default=0.17435)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    global PS
    PS = a.ps
    rng = np.random.default_rng(a.seed)

    npop_real = fitsio.read_header(f"{D}/{B.CATALOGS['gauss2_v4n']['plus']}.fits", ext=1)["NPOP"]
    npop_closed = fitsio.read_header(f"{D}/{B.CATALOGS['gauss2_v4n_closed']['plus']}.fits", ext=1).get(
        "NPOP", len(fitsio.read(f"{D}/{B.CATALOGS['gauss2_v4n_closed']['plus']}.fits", columns=["nda"])))
    R, C = load(a.real, npop_real), load(a.closed, npop_closed)

    # pairs: every real target -> nearest closed target (with replacement)
    fr, fc = feats(R["moments"]), feats(C["moments"])
    mu, sd = fc.mean(0), fc.std(0)
    _, j = cKDTree((fc - mu) / sd).query((fr - mu) / sd, workers=-1)
    n = len(fr)
    print(f"{n} real targets paired to {len(np.unique(j))} distinct closed targets "
          f"(real N_pop {npop_real}, closed N_pop {npop_closed})")

    # N_ns is a population count; a paired resample of n real rows stands for
    # the real population, and its partner set for the closed one at the
    # SAME size, so the closed count is scaled to the real population size.
    ir = np.arange(n)
    base_r = m1(R, ir, 1.0, a.rs)
    base_c = m1(C, j, npop_real / npop_closed, a.rs)
    boots = []
    for _ in range(a.boot):
        b = rng.integers(0, n, n)
        ur, cr = m1(R, b, 1.0, a.rs)
        uc, cc = m1(C, j[b], npop_real / npop_closed, a.rs)
        boots.append((ur, cr, uc, cc, ur - uc, cr - cc))
    boots = np.array(boots); err = boots.std(0)
    print("\n1. m1, paired over matched pairs (bootstrap over pairs)")
    print(f"   {'':12s} {'real':>18s} {'closed (matched)':>18s} {'real - closed':>18s}  unpaired err of diff")
    for i, name in enumerate(("uncorrected", "corrected")):
        d = base_r[i] - base_c[i]
        print(f"   {name:12s} {base_r[i]:+.4f}+/-{err[i]:.4f}   {base_c[i]:+.4f}+/-{err[2 + i]:.4f}   "
              f"{d:+.4f}+/-{err[4 + i]:.4f}   {np.hypot(err[i], err[2 + i]):.4f}")

    print("\n2. Fisher identity sum |q|^2 / sum(-tr r), in-window rows, by real g=0 flux quintile")
    flux = R["moments"][:, 0]
    w = B.window_mask(R["moments"], SIZE, FLUX)
    q = np.quantile(flux[w], np.linspace(0, 1, 6)); q[-1] += 1
    print(f"   {'flux':>14s} {'real':>15s} {'closed':>15s} {'real - closed':>15s}")
    for lo, hi in list(zip(q[:-1], q[1:])) + [(FLUX[0], FLUX[1])]:
        k = np.flatnonzero((flux >= lo) & (flux < hi))
        vals = [fisher(R, k), fisher(C, j[k])]
        bs = []
        for _ in range(300):
            b = rng.choice(k, len(k))
            bs.append((fisher(R, b), fisher(C, j[b])))
        bs = np.array(bs); e = bs.std(0); ed = (bs[:, 0] - bs[:, 1]).std()
        print(f"   {lo:6.0f}-{hi:<7.0f} {vals[0]:.4f}+/-{e[0]:.4f} {vals[1]:.4f}+/-{e[1]:.4f} "
              f"{vals[0] - vals[1]:+.4f}+/-{ed:.4f}")

    m = R["moments"]
    print("\n2b. paired Fisher excess (real - closed) localised")
    fisher_split(R, C, j, m[:, 1] / m[:, 0], "size Mr/Mf", rng)
    fisher_split(R, C, j, np.hypot(m[:, 2], m[:, 3]) / m[:, 1], "|e|", rng)
    fisher_split(R, C, j, m[:, 4] / m[:, 1], "concentration Mc/Mr", rng)
    c2st_obs(rng)


if __name__ == "__main__":
    main()
