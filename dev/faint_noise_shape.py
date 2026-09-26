"""Recentred noise beyond its variance: shape and g-dependence.

Reads dev/faint_response_obs.npz (dev/faint_response.py).  The model convolves
each target with a FIXED Gaussian C_M.  Two ways real recentred noise can
differ without moving the mean response or the variance along v:

1. shape at g = 0: skewness / excess kurtosis of the score-direction projection
   y = v^T C_M^-1 (M - <M>),  v = the flow's R (2x5, one y per shear axis).
   Gaussian: skew 0, excess kurtosis 0.
2. dCov/dg: C' = (Cov(+g) - Cov(-g)) / 2 delta, CRN so the difference is
   precise.  The model has C' = 0.  Its Fisher information relative to the
   mean term,  F_cov / F_mean = 1/2 tr(C^-1 C' C^-1 C') / (v^T C^-1 v),
   is the score-variance excess it would make -- compare the ~3% real-vs-closed
   Fisher excess.  Also tr(C^-1 C'), the log-det's g-slope.
"""
import numpy as np

d = np.load("dev/faint_response_obs.npz")
nf, delta, CM, Rfl, Mf = int(d["n_faint"]), float(d["delta"]), d["CM"], d["R_fl"], d["Mf"]
CMi = np.linalg.inv(CM)
out = []
for k in range(len(Mf)):
    o = d[f"obs{k}"]                                    # (n, 5 arms, 5)
    r = o[:, 0] - o[:, 0].mean(0)
    y = r @ CMi @ Rfl[k].T                              # (n, 2)
    ys = y / y.std(0)
    skew, kurt = (ys ** 3).mean(0), (ys ** 4).mean(0) - 3
    C0 = np.cov(o[:, 0].T)
    row = [skew.mean(), kurt.mean(), np.trace(C0 @ CMi) / 5]
    for a, (p, m) in enumerate(((1, 2), (3, 4))):
        Cp = (np.cov(o[:, p].T) - np.cov(o[:, m].T)) / (2 * delta)
        Ci = np.linalg.inv(C0)
        v = Rfl[k][a]
        row += [0.5 * np.trace(Ci @ Cp @ Ci @ Cp) / (v @ Ci @ v), np.trace(Ci @ Cp)]
    out.append(row)
out = np.array(out)

for name, sl in (("faint", slice(0, nf)), ("bright", slice(nf, None))):
    x = out[sl]
    n = len(x)
    se = lambda c: x[:, c].std() / np.sqrt(n)
    print(f"\n{name}: {n} galaxies, median Mf {np.median(Mf[sl]):.0f}, {len(d['obs0'])} noise fields")
    print(f"  score-direction skew        {x[:, 0].mean():+.4f} +/- {se(0):.4f}   (Gaussian 0)")
    print(f"  score-direction excess kurt {x[:, 1].mean():+.4f} +/- {se(1):.4f}   (Gaussian 0)")
    print(f"  tr(Cov C_M^-1)/5            {x[:, 2].mean():.4f} +/- {se(2):.4f}")
    print(f"  F_cov/F_mean  g1 {x[:, 3].mean():.5f}  g2 {x[:, 5].mean():.5f}   (vs ~0.03 Fisher excess)")
    print(f"  tr(C^-1 dC/dg)  g1 {x[:, 4].mean():+.3f} +/- {se(4):.3f}   g2 {x[:, 6].mean():+.3f} +/- {se(6):.3f}")
    print(f"  |tr(C^-1 dC/dg)| rms  g1 {np.sqrt((x[:, 4] ** 2).mean()):.3f}  g2 {np.sqrt((x[:, 6] ** 2).mean()):.3f}")
