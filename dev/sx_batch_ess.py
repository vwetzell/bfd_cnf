#!/usr/bin/env python
"""Effective sample size of the SNIS marginalisation, per gradient step.

The loss draws batch_size=2048 copies at random from the full pool and weights
them w_b = softmax(log L(X_b|C_X)+log nda+log detj) (flows.py:633,755-758).  The
X-integral each step actually sees is carried by ESS = 1/sum(w_b^2) of those 2048
copies.  This is NOT the per-template grid quadrature; it's importance sampling
with the copy population as the proposal.  Narrow L (low log_scale) -> few batch
survivors -> noisy, finite-sample-biased SNIS gradients.  Measure ESS vs log_scale.
"""
import numpy as np
import fitsio

BATCH = 2048
N_TRIALS = 60
LOG_SCALES = [10.5, 11.0, 11.5, 11.98, 12.5, 13.0]

def main():
    h = fitsio.FITS('data/templates_train.fits')[1]
    N = h.get_nrows()
    # 3 contiguous chunks (start/mid/end) span different id/flux regions -> pool proxy
    per = 120_000
    rows = np.concatenate([np.arange(per),
                           np.arange(N // 2, N // 2 + per),
                           np.arange(N - per, N)])
    blk = h.read(rows=rows, columns=["moments", "centroid", "nda"])
    m = blk["moments"].astype(np.float64)
    X = blk["centroid"][:, :2].astype(np.float64)
    nda = blk["nda"].astype(np.float64)
    detj = np.maximum(0.25 * (m[:, 1] ** 2 - m[:, 2] ** 2 - m[:, 3] ** 2), 1e-30)
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    base = np.log(nda) + np.log(detj)            # log_scale-independent part
    P = len(r2)
    rng = np.random.default_rng(0)

    print(f"pool proxy: {P:,} copies   batch={BATCH}   trials={N_TRIALS}\n")
    print(f"{'log_scale':>9} {'sigmaX':>7} {'ESS_med':>8} {'ESS_10/90':>14} "
          f"{'top1_w':>7} {'frac<1σ':>8}")
    for ls in LOG_SCALES:
        inv = 1.0 / np.exp(ls)
        ess, top1 = [], []
        for _ in range(N_TRIALS):
            b = rng.integers(0, P, BATCH)
            logw = -0.5 * r2[b] * inv + base[b]
            logw -= logw.max()
            w = np.exp(logw); w /= w.sum()
            ess.append(1.0 / (w * w).sum())
            top1.append(w.max())
        sig = np.exp(ls / 2.0)
        frac_in = np.mean(r2 < sig * sig)        # pool fraction within 1σ_X
        e = np.array(ess)
        print(f"{ls:9.2f} {sig:7.0f} {np.median(e):8.1f} "
              f"({np.percentile(e,10):.0f},{np.percentile(e,90):.0f})".ljust(24)
              + f"{np.median(top1):7.3f} {frac_in:8.3f}")

    # self-check: ESS must be <= BATCH and rise with log_scale (wider L = more survivors)
    print(f"\n[check] ESS << {BATCH} means each step's X-integral is carried by few "
          f"copies; lower log_scale -> fewer -> the SNIS concern you raised.")

if __name__ == "__main__":
    main()
