"""Test 1: does dropping the odd-moment (centroid) shear derivative from BFD's
exact PQR reproduce the flow's +0.18 m-bias?

BFD's _prob_scalar_scalar (probabilities_jax.py:195-200):
    chiE = (M_t_even - M_copy_even)·invCovE·(...)        [5-d even]
    chiO = X_copy·invCovO·X_copy                          [2-d odd centroid]
    prob = detj_target · exp(-0.5(chiE+chiO))             detj_target const -> cancels
    dP/dg = Σ nda·dp/dM·dM/dg ;  d2P/dg2 = Σ nda·[dMdg·d2p/dM2·dMdg + dp/dM·d2Mdg2]
with dM/dg over ALL 7 moments (5 even + 2 odd MX,MY).  The flow drops moments 5:7.

We compute BFD PQR analytically (closed-form derivs of prob) on grid ± targets vs a
template bank (templates_train.fits), once with the full 7-moment dm_dg and once with
the odd part (cols 5,6) zeroed -- the flow's omission.  KD-tree prunes to k nearest
templates (prob is sharply peaked).  Report aggregate m for both.
"""
import sys
import numpy as np
import fitsio
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/vwetzell/gitrepos/bfd_cnf")
import bfd_cnf.config as C
import bfd
from bfd_cnf.statistics import pqr2g

TMPL = "data/templates_train.fits"
N_TMPL = 4_000_000      # template bank size (random rows; raise for coverage)
N_TARG = 40_000         # targets per arm
K = 3000                # nearest templates per target
DG = 0.04               # +/- shear separation
rng = np.random.default_rng(0)


def load_templates(n):
    h = fitsio.FITS(TMPL)[1]
    ntot = h.get_nrows()
    rows = np.sort(rng.choice(ntot, size=min(n, ntot), replace=False))
    d = h.read(rows=rows, columns=["moments", "centroid", "dm_dg", "d2m_dg2", "nda"])
    me = np.asarray(d["moments"][:, :5], np.float64)        # (n,5) even
    xo = np.asarray(d["centroid"], np.float64)              # (n,2) odd MX,MY
    m7 = np.concatenate([me, xo], axis=1)                   # (n,7)
    rdm = np.asarray(d["dm_dg"], np.float64)                # (n,2,7)[shear,moment]
    rd2 = np.asarray(d["d2m_dg2"], np.float64)              # (n,3,7)
    dmdg = np.transpose(rdm, (0, 2, 1))                     # (n,7,2)[moment,shear]
    d2 = np.zeros((len(me), 7, 2, 2))
    d2[:, :, 0, 0] = rd2[:, 0, :]
    d2[:, :, 0, 1] = rd2[:, 1, :]; d2[:, :, 1, 0] = rd2[:, 1, :]
    d2[:, :, 1, 1] = rd2[:, 2, :]
    nda = np.asarray(d["nda"], np.float64)
    return m7, dmdg, d2, nda


def load_targets(path, n):
    a = np.load(path, allow_pickle=True)
    mom = np.asarray(a["moments"], np.float64)              # (N,5)
    mf, mr = mom[:, 0], mom[:, 1]
    pqr0 = np.asarray(a["pqr"][:, 0], np.float64)           # catalog BFD P (validity)
    sel = (mf > 1500) & (mf < 90000) & (mr / mf > 2.2) & (mr / mf < 3.5) & (pqr0 > 0)
    idx = np.where(sel)[0]
    idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
    cov = bfd.MomentCovariance.bulkUnpack(a["covariance"][idx])[:, :5, :5].astype(np.float64)
    return mom[idx], cov


def covO_from_covE(covE):
    # _prep_target lines 161-165
    cO = np.zeros((covE.shape[0], 2, 2))
    cO[:, 0, 0] = 0.5 * (covE[:, 0, 1] + covE[:, 0, 2])
    cO[:, 1, 1] = 0.5 * (covE[:, 0, 1] - covE[:, 0, 2])
    cO[:, 0, 1] = cO[:, 1, 0] = 0.5 * covE[:, 0, 3]
    return cO


def bfd_pqr_for_targets(mom_t, cov_t, tmpl, tree, tw_scale, drop_odd):
    """Return (N,6) pqr [P,Q1,Q2,R11,R22,R12] for targets via KD-pruned BFD sum."""
    m7, dmdg, d2m, nda = tmpl
    if drop_odd:
        dmdg = dmdg.copy(); dmdg[:, 5:7, :] = 0.0
        d2m = d2m.copy();  d2m[:, 5:7, :, :] = 0.0
    invCovE = np.linalg.inv(cov_t)                          # (N,5,5)
    invCovO = np.linalg.inv(covO_from_covE(cov_t))          # (N,2,2)
    Kfac = 0.25 * (mom_t[:, 1] ** 2 - mom_t[:, 2] ** 2 - mom_t[:, 3] ** 2)
    out = np.zeros((len(mom_t), 6))
    qpts = mom_t[:, :4] @ tw_scale.T                        # whiten by Cholesky^-1
    for s in range(0, len(mom_t), 2000):
        e = min(s + 2000, len(mom_t))
        _, nbr = tree.query(qpts[s:e], k=K, workers=-1)     # (b,K)
        for j in range(s, e):
            ii = nbr[j - s]
            mc = m7[ii]                                     # (K,7)
            dE = mom_t[j] - mc[:, :5]                       # (K,5)
            iCE, iCO = invCovE[j], invCovO[j]
            chiE = np.einsum("ki,ij,kj->k", dE, iCE, dE)
            Xc = mc[:, 5:7]
            chiO = np.einsum("ki,ij,kj->k", Xc, iCO, Xc)
            prob = Kfac[j] * np.exp(-0.5 * (chiE + chiO))   # (K,)
            # sc_i = dlogprob/dM_copy_i : even = invCovE·dE ; odd = -invCovO·Xc
            sc = np.zeros((len(ii), 7))
            sc[:, :5] = dE @ iCE
            sc[:, 5:7] = -(Xc @ iCO)
            w = nda[ii] * prob
            P = w.sum()
            dmg = dmdg[ii]                                  # (K,7,2)
            sg = np.einsum("ki,kim->km", sc, dmg)          # (K,2)  sc·dM/dg
            Q = np.einsum("k,km->m", w, sg)                # (2,)
            # H = blockdiag(-invCovE, -invCovO); dm·H·dm
            HE = np.einsum("kim,ij,kjn->kmn", dmg[:, :5], -iCE, dmg[:, :5])
            HO = np.einsum("kim,ij,kjn->kmn", dmg[:, 5:7], -iCO, dmg[:, 5:7])
            quad = sg[:, :, None] * sg[:, None, :] + HE + HO        # (K,2,2)
            lin = np.einsum("ki,kimn->kmn", sc, d2m[ii])          # sc·d2M/dg2
            R = np.einsum("k,kmn->mn", w, quad + lin)              # (2,2)
            out[j] = [P, Q[0], Q[1], R[0, 0], R[1, 1], R[0, 1]]
    return out


def m_from_pqr(pqr_p, pqr_m):
    gp, gm = np.asarray(pqr2g(pqr_p)), np.asarray(pqr2g(pqr_m))
    return gp, gm, (gp[0] - gm[0]) / DG - 1.0


def main():
    print(f"loading {N_TMPL:,} templates ...", flush=True)
    tmpl = load_templates(N_TMPL)
    m7 = tmpl[0]
    print("KD-tree built. loading targets ...", flush=True)
    momp, covp = load_targets(C.GRID_P_PATH, N_TARG)
    momm, covm = load_targets(C.GRID_M_PATH, N_TARG)
    # whiten the 4 even moments by a typical target covariance so Euclidean KD
    # neighbours approximate BFD's Mahalanobis (per-target invCovE) neighbours
    med_cov = np.median(np.concatenate([covp, covm], 0)[:, :4, :4], axis=0)
    W = np.linalg.inv(np.linalg.cholesky(med_cov))        # (4,4) whitening
    tw_scale = W
    tree = cKDTree(m7[:, :4] @ W.T)
    print(f"targets: +{len(momp)}  -{len(momm)};  integrating (k={K}) ...", flush=True)

    for drop in (False, True):
        tag = "ODD-ZEROED (flow's omission)" if drop else "FULL BFD (with centroid deriv)"
        pp = bfd_pqr_for_targets(momp, covp, tmpl, tree, tw_scale, drop)
        pm = bfd_pqr_for_targets(momm, covm, tmpl, tree, tw_scale, drop)
        ok = np.isfinite(pp).all(1) & np.isfinite(pm).all(1) & (pp[:, 0] > 0) & (pm[:, 0] > 0)
        gp, gm, m = m_from_pqr(pp[ok], pm[ok])
        print(f"\n=== {tag} ===")
        print(f"  g(+)={gp}  g(-)={gm}")
        print(f"  m = {m:+.4f}   (n={ok.sum()})")


if __name__ == "__main__":
    main()
