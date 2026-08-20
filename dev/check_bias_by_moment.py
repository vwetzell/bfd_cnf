"""m1 profile along each of the five raw moment axes [Mf, Mr, M1, M2, Mc].

Reads a `--save-pqr` npz from `bias.py` (needs the `moments` column, added
alongside `flux` for this) and bins the +/-g pair's paired m1 estimator along
each axis in turn, marginalising over the other four -- the same construction
as `dev/check_size_oscillation.py`'s `profile()`, generalised from one axis
(Mr/Mf) to all five.

Binning is on the catalog's own clean unsheared moments (`bias.py`'s `truth`,
saved before noise/centroid), so a bin boundary cannot induce a selection term
(same reasoning as `dev/check_bias_by_selection.py`).

    python bias.py --flow flows/shear.eqx --pop bulgedisc --samples 8192 \\
        --n-targets 20000 --save-pqr dev/pqr_bulgedisc.npz
    python dev/check_bias_by_moment.py dev/pqr_bulgedisc.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402

G, NBIN, NBOOT = 0.02, 12, 200
AXES = ["Mf", "Mr", "M1", "M2", "Mc"]


def profile(y, qp, rp, qm, rm, nbin, rng):
    edges = np.quantile(y, np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    out = []
    for k in range(nbin):
        s = (y >= edges[k]) & (y < edges[k + 1])
        if s.sum() < 50:
            continue
        idx0 = np.flatnonzero(s)
        v = bias.bias(qp[s], rp[s], qm[s], rm[s], G)[0]
        boot = [bias.bias(qp[i], rp[i], qm[i], rm[i], G)[0]
                for i in (rng.choice(idx0, len(idx0)) for _ in range(NBOOT))]
        out.append((float(y[s].mean()), v, float(np.std(boot)), int(s.sum())))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pqr", help="--save-pqr npz from bias.py")
    p.add_argument("--nbin", type=int, default=NBIN)
    a = p.parse_args()

    d = np.load(a.pqr)
    if "moments" not in d:
        raise SystemExit(f"{a.pqr} has no 'moments' column -- re-run bias.py "
                          f"--save-pqr with the version that writes it")
    m = d["moments"]
    qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
    if "keep" in d:
        # check_selfconsistency_noisy.py's finite/|R|-guard mask, computed but
        # not applied before saving -- apply it here.
        keep = d["keep"]
        m, qp, rp, qm, rm = m[keep], qp[keep], rp[keep], qm[keep], rm[keep]
    rng = np.random.default_rng(0)

    overall = bias.bias(qp, rp, qm, rm, G)[0]
    print(f"{a.pqr}: {len(m)} targets, overall m1 = {overall:+.4f}\n")

    for i, name in enumerate(AXES):
        print(f"=== m1 vs {name}, {a.nbin} quantile bins ===")
        rows = profile(m[:, i], qp, rp, qm, rm, a.nbin, rng)
        for c, v, e, n in rows:
            bar = "*" * int(round(abs(v) / 0.002)) if np.isfinite(v) else ""
            print(f"  {name}={c:12.4g}  n={n:6d}  m1={v:+.4f} +/- {e:.4f}  {bar}")
        print()


if __name__ == "__main__":
    main()
