"""Per-flux-bin m1 against the Fisher metric, offline from one --save-pqr run.

The interior bright bins sit far above any flux window boundary, so they carry
NO selection term -- their m1 is the residual bias with the selection machinery
taken out of the argument entirely.  For each bin:

    num = (SUM q1(+g) - SUM q1(-g)) / 2g        first order, survives vs BFD
    den = mean over arms of SUM -r11            Newton metric, the suspect
    denF = mean over arms of SUM q1^2           Fisher metric, same root

`m1 = num/den - 1` is what bias.py reports; `m1F = num/denF - 1` is the same
estimate with the second-order term replaced by the outer product.  If m1F is
clean where m1 is not, the residual lives in R (curvature + score variance);
if both are equally off, it lives in Q or in the population.

`fisher = denF/den` is the health check of [[fisher-identity-is-the-health-check]].
"""
import os, sys
import numpy as np

PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
G = float(os.environ.get("G", 0.02))
NB = int(os.environ.get("NB", 5))
BOOT = int(os.environ.get("BOOT", 200))


def load(path):
    d = np.load(path)
    ok = np.ones(len(d["plus_q"]), dtype=bool)
    for k in ("plus", "minus", "zero"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    return {k: d[k][ok] for k in d.files}, ok


def stats(d, idx):
    """(m1, m1F, fisher, opg_top1pct_share) over the rows in `idx`."""
    qp, qm = d["plus_q"][idx, 0], d["minus_q"][idx, 0]
    rp, rm = d["plus_r"][idx, 0, 0], d["minus_r"][idx, 0, 0]
    num = (qp.sum() - qm.sum()) / (2 * G)
    den = 0.5 * (-rp.sum() + -rm.sum())
    denF = 0.5 * (np.sum(qp ** 2) + np.sum(qm ** 2))
    return num / den - 1, num / denF - 1, denF / den


def boot(d, idx, n=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    out = np.array([stats(d, rng.choice(idx, len(idx))) for _ in range(n)])
    return out.std(axis=0, ddof=1)


if __name__ == "__main__":
    d, ok = load(PQR)
    print(f"{len(d['flux'])} finite targets ({int((~ok).sum())} dropped)")
    f = d["flux"]
    edges = np.percentile(f, np.linspace(0, 100, NB + 1))
    print(f"\n{'bin':>6s}{'Mf lo':>10s}{'n':>8s}{'m1 (Newton)':>22s}"
          f"{'m1 (Fisher/OPG)':>24s}{'denF/den':>10s}")
    rows = [("all", np.arange(len(f)))]
    rows += [(f"q{i+1}", np.flatnonzero((f >= edges[i]) & (f <= edges[i + 1])))
             for i in range(NB)]
    for name, idx in rows:
        m1, m1F, fis = stats(d, idx)
        s = boot(d, idx)
        print(f"{name:>6s}{f[idx].min():10.0f}{len(idx):8d}"
              f"{m1:+13.4f} +/- {s[0]:.4f}{m1F:+15.4f} +/- {s[1]:.4f}"
              f"{fis:10.3f}")
