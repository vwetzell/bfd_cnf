"""Check 1: is a copy's centroid shear response dX/dg = K(M) X, with K a
binned function of the copy's own moments?

Fits, per (flux, size, e1, e2) bin, dX/dg_a = K_a X and d2X/dg_ab = K_ab X
(2x2 each) by least squares on one slab of copies and scores held-out
copies from a disjoint slab.  Reports held-out R^2 of dX, d2X and of the
quantity the eq.-36 weight actually uses, l_a = -X^T Sinv dX/dg_a.
Writes the table to logs/centroid_K.npz for template_sum_reweight --kfit.
"""
import sys
sys.path.insert(0, ".")
import fitsio
import numpy as np

COPIES = "../bfd_cnf_imsims/data/copies_gauss2_fwd_g2v4n.fits"
TARGETS = "../bfd_cnf_imsims/data/targets_g2v4_g1p02_22k.fits"
NB = dict(flux=8, size=6, e=7)


def feats(c):
    m = np.asarray(c["moments"], np.float64)
    return m[:, 0], m[:, 1] / m[:, 0], m[:, 2] / m[:, 1], m[:, 3] / m[:, 1]


def targets_Y(c):
    dX = np.asarray(c["dxy_dg"], np.float64)      # (n, g, comp)
    d2X = np.asarray(c["d2xy_dg2"], np.float64)   # (n, 3, comp)
    return np.concatenate([dX, d2X], 1)           # (n, 5, 2)


def bin_index(f, edges):
    idx = np.zeros(len(f[0]), np.int64)
    for v, e in zip(f, edges):
        idx = idx * (len(e) + 1) + np.searchsorted(e, v)
    return idx


def fit(c, edges):
    X = np.asarray(c["xy"], np.float64)
    Y = targets_Y(c).reshape(len(X), 10)
    b = bin_index(feats(c), edges)
    nb = np.prod([len(e) + 1 for e in edges])
    XtX = np.zeros((nb, 2, 2)); XtY = np.zeros((nb, 2, 10))
    np.add.at(XtX, b, X[:, :, None] * X[:, None, :])
    np.add.at(XtY, b, X[:, :, None] * Y[:, None, :])
    XtX += 1e-9 * np.eye(2)
    return np.linalg.solve(XtX, XtY)              # (nb, 2, 10)


def predict(c, edges, coef):
    X = np.asarray(c["xy"], np.float64)
    b = bin_index(feats(c), edges)
    return np.einsum("nk,nkj->nj", X, coef[b]).reshape(len(X), 5, 2)


def r2(y, p, axis=None):
    return 1 - ((y - p) ** 2).sum(axis) / ((y - y.mean(0)) ** 2).sum(axis)


def main():
    F = fitsio.FITS(COPIES)["COPIES"]
    tr, te = F[0:4_000_000], F[60_000_000:61_000_000]
    f = feats(tr)
    edges = [np.quantile(f[0], np.linspace(0, 1, NB["flux"] + 1)[1:-1]),
             np.quantile(f[1], np.linspace(0, 1, NB["size"] + 1)[1:-1]),
             np.quantile(f[2], np.linspace(0, 1, NB["e"] + 1)[1:-1]),
             np.quantile(f[3], np.linspace(0, 1, NB["e"] + 1)[1:-1])]
    coef = fit(tr, edges)
    Y, P = targets_Y(te), predict(te, edges, coef)

    # baseline: one global K for every copy (pure affine shear of X)
    P0 = predict(te, [np.array([])], fit(tr, [np.array([])]))

    sx = fitsio.read(TARGETS, rows=[0])["cov_odd"][0]
    sinv = np.linalg.inv(np.array([[sx[0], sx[1]], [sx[1], sx[2]]]))
    X = np.asarray(te["xy"], np.float64)
    sX = X @ sinv
    la = lambda D: -np.einsum("nk,nak->na", sX, D[:, :2])

    flux = feats(te)[0]
    print(f"train 4M rows, held-out 1M rows, {coef.shape[0]} bins")
    print(f"{'':22s}{'global K':>10s}{'binned K':>10s}")
    for name, sl in (("dX/dg", slice(0, 2)), ("d2X/dg2", slice(2, 5))):
        print(f"R^2 {name:18s}{r2(Y[:, sl], P0[:, sl]):10.3f}{r2(Y[:, sl], P[:, sl]):10.3f}")
    print(f"R^2 {'l_a (weight term)':18s}{r2(la(Y), la(P0)):10.3f}{r2(la(Y), la(P)):10.3f}")
    print("binned R^2 of l_a by flux bin:")
    for lo, hi in zip([0, 3000, 8000, 20000], [3000, 8000, 20000, np.inf]):
        k = (flux >= lo) & (flux < hi)
        print(f"  flux {lo:>6}-{hi:<6}  n={k.sum():7d}  R^2={r2(la(Y)[k], la(P)[k]):.3f}")
    np.savez("logs/centroid_K.npz", coef=coef, **{f"edge{i}": e for i, e in enumerate(edges)})
    print("saved logs/centroid_K.npz")


if __name__ == "__main__":
    main()
