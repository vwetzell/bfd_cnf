#!/usr/bin/env python
"""Is the centroid (MX,MY) shift grid too coarse for the last-layer marginalisation?

The ELBO marginalises each template over its shifted copies by the discrete sum
    p(M|C_X) ~ sum_i softmax(log L(X_i|C_X)) ,  L(X|C_X)=N(X;0,C_X)   (flows.py:741-759)
on the copies' first-order centroid moments X=[MX,MY] (the FITS `centroid` col).
C_X is drawn over log_scale_range=(10.5,13.0); true var is sigma_X^2 = exp(log_scale).

This script asks, per template, how well that DISCRETE sum reproduces the
CONTINUOUS Gaussian integral as a function of log_scale, using the real X grid:
  - ESS(ls)      = (sum L)^2 / sum L^2      -> # copies actually integrated
  - var_eff(ls)  = <|X|^2>_L / 2            -> recovered sigma_X^2 (want = exp(ls))
If var_eff falls below exp(ls) and ESS->1 as ls shrinks, the integral is
undersampled in a log_scale-DEPENDENT way -> predicts m(log_scale).
e=0 (isotropic C_X), matching the e=0 question.
"""
import numpy as np
from astropy.io import fits

LOG_SCALES = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 11.98, 12.5, 13.0])
MIN_COPIES = 20          # templates with a real fan
N_PARENTS = 8000         # sample size

def main():
    h = fits.open('data/templates_train.fits', memmap=True)
    d = h[1].data
    # rows are contiguous per parent; read a chunk holding many complete parents
    n = min(4_000_000, h[1].header['NAXIS2'])
    ids = np.asarray(d['id'][:n])
    X = np.asarray(d['centroid'][:n], dtype=np.float64)   # (n,2) [MX,MY]
    Mf = np.asarray(d['moments'][:n, 0], dtype=np.float64)
    h.close()

    # group by parent (contiguous); drop the last (possibly truncated) group
    bnd = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], bnd])
    ends = np.concatenate([bnd, [len(ids)]])
    groups = [(s, e) for s, e in zip(starts[:-1], ends[:-1]) if e - s >= MIN_COPIES]
    rng = np.random.default_rng(0)
    if len(groups) > N_PARENTS:
        groups = [groups[i] for i in rng.choice(len(groups), N_PARENTS, replace=False)]
    print(f'parents used: {len(groups)} (>= {MIN_COPIES} copies)')

    # X-grid spacing: median nearest-neighbour |dX| within a few parents
    nn = []
    for s, e in groups[:200]:
        Xi = X[s:e]
        for j in range(len(Xi)):
            dd = np.hypot(Xi[:, 0] - Xi[j, 0], Xi[:, 1] - Xi[j, 1])
            dd[j] = np.inf
            nn.append(dd.min())
    dx = np.median(nn)
    print(f'X-grid spacing  Delta ~ {dx:.1f} (moment units)   [sigma_xy ref ~400]')

    print(f'\n{"log_scale":>9} {"sigmaX":>7} {"sigX/Δ":>6} '
          f'{"ESS_med":>8} {"var_eff/true":>13} {"(16,50,84 pct)":>22}')
    for ls in LOG_SCALES:
        var_true = np.exp(ls)                 # sigma_X^2
        sigX = np.sqrt(var_true)
        ess_all, ratio_all = [], []
        inv2var = 1.0 / (2.0 * var_true)
        for s, e in groups:
            Xi = X[s:e]
            r2 = Xi[:, 0] ** 2 + Xi[:, 1] ** 2
            logL = -r2 * inv2var              # e=0 isotropic, drop const norm
            logL -= logL.max()
            L = np.exp(logL)
            sumL = L.sum()
            ess = sumL * sumL / (L * L).sum()
            var_eff = (r2 * L).sum() / sumL / 2.0     # recovered sigma_X^2
            ess_all.append(ess)
            ratio_all.append(var_eff / var_true)
        ess_all = np.array(ess_all)
        ratio = np.array(ratio_all)
        p = np.percentile(ratio, [16, 50, 84])
        print(f'{ls:9.2f} {sigX:7.0f} {sigX/dx:6.2f} '
              f'{np.median(ess_all):8.1f} {p[1]:13.3f} '
              f'({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})')

    # self-check: recovered variance must be monotone in log_scale and <= true
    # (discrete sum on a fixed grid cannot over-recover an isotropic Gaussian var)
    print('\n[check] var_eff/true should rise toward 1 as log_scale grows;'
          ' a shortfall at small log_scale = the undersampling the hypothesis predicts.')

if __name__ == '__main__':
    main()
