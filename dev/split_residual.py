"""Where does the interior-bin residual m1 live?  Offline from one saved run.

Bins that sit entirely above the flux window's boundary carry no selection
term, so their m1 is the residual bias with the whole selection machinery out
of the argument.  This slices those bins by the other coordinates the density
is suspected in -- the point-source ceiling `v = Mc/(POINT_SOURCE_MC Mr)`, the
resolution `Mr/Mf`, and the ellipticity -- to see whether the residual is a
sub-population or spread over everything.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.bijections import POINT_SOURCE_MC  # noqa: E402

PQR = os.environ.get("PQR", "dev/pqr_200k.npz")
G = float(os.environ.get("G", 0.02))
BOOT = int(os.environ.get("BOOT", 200))
FLUX_LO = float(os.environ.get("FLUX_LO", 0))


def m1(d, idx):
    qp, qm = d["plus_q"][idx, 0], d["minus_q"][idx, 0]
    rp, rm = d["plus_r"][idx, 0, 0], d["minus_r"][idx, 0, 0]
    num = (qp.sum() - qm.sum()) / (2 * G)
    return num / (0.5 * (-rp.sum() - rm.sum())) - 1


def row(d, name, idx, rng):
    v = m1(d, idx)
    s = np.std([m1(d, rng.choice(idx, len(idx))) for _ in range(BOOT)], ddof=1)
    print(f"{name:>22s}{len(idx):9d}{v:+11.4f} +/- {s:.4f}")


if __name__ == "__main__":
    d = dict(np.load(PQR))
    ok = np.ones(len(d["flux"]), bool)
    for k in ("plus", "minus"):
        ok &= np.isfinite(d[f"{k}_q"]).all(1)
        ok &= np.isfinite(d[f"{k}_r"]).reshape(len(ok), -1).all(1)
    d = {k: v[ok] for k, v in d.items()}
    m = d["moments"]                       # clean [Mf, Mr, M1, M2, Mc]
    sel = d["flux"] >= FLUX_LO
    d = {k: v[sel] for k, v in d.items()}
    m = m[sel]
    coords = {
        "v = Mc/(PSMc.Mr)": m[:, 4] / (POINT_SOURCE_MC * m[:, 1]),
        "Mr/Mf": m[:, 1] / m[:, 0],
        "|e| = |M1,M2|/Mr": np.hypot(m[:, 2], m[:, 3]) / m[:, 1],
        "Mf": m[:, 0],
    }
    rng = np.random.default_rng(0)
    print(f"{len(m)} targets, Mf >= {FLUX_LO:g}")
    for label, c in coords.items():
        print(f"\n  by {label}"
              f"\n{'bin':>22s}{'n':>9s}{'m1':>11s}")
        e = np.percentile(c, np.linspace(0, 100, 6))
        for i in range(5):
            idx = np.flatnonzero((c >= e[i]) & (c <= e[i + 1]))
            row(d, f"[{e[i]:.4g}, {e[i+1]:.4g}]", idx, rng)
