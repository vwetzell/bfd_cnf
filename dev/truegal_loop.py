"""True galaxies, model noise: the rung between the closed loop and the sims. (2026-10-01)

The closed loop (dev/closed_loop.py --jacobian, m1 +0.0002) draws the noiseless
moments from the FLOW; the sims (m1 ~ +0.01) use real galaxies AND real pixel
noise + recentring.  This catalog takes the sims' OWN galaxies (copies built at
seed 2 = the sn8r targets, row for row) and gives them the paper's noise model:
  detection   keep galaxy i if v_i < sum_u w(u; g)    (eq. 36 weights sum to P(detect))
  centroid    one copy u ~ w(u; g) = da |J(u_g)| N(X_g(u); 0, Sigma_X), copy lensed by its
              own dm_dg/d2m_dg2 and |J| at the lensed moments ([[copies-g-weights-freeze-J]])
  noise       M = m + n, n ~ N(0, C_M) tilted by J(M)/J(m)  (as closed_loop --jacobian)
All uniforms and noise candidates are common to the three arms (paired).  m1 ~ 0
here => the flow's density is right and the excess is pixel noise / recentring
beyond the first-order J model; m1 ~ +0.01 => it is the flow's density.
  python dev/truegal_loop.py --copies ../bfd_cnf_imsims/data/copies_bulgedisc_g2_bdg2n_sn8_seed2.fits
"""
import argparse, os, sys
import numpy as np, fitsio
sys.path.insert(0, "."); sys.path.insert(0, "../bfd_cnf_imsims")
import bias as B
from imsims.copies import log_weights, log_jacobian

D = "../bfd_cnf_imsims/data"
COLS = ["gal", "moments", "xy", "da", "dm_dg", "d2m_dg2", "dxy_dg", "d2xy_dg2"]
ARMS = (("plus", 0.02), ("minus", -0.02), ("zero", 0.0))


def pick(c, sx, u, v, out, det):
    """For the whole galaxies in block `c`: per arm, detection and one copy ~ w(u; g)."""
    gal = c["gal"]
    g0, g1 = gal[0], gal[-1] + 1
    mc = np.asarray(c["moments"], np.float64)
    for a, (arm, g) in enumerate(ARMS):
        ml = mc + g * c["dm_dg"][:, 0] + 0.5 * g * g * c["d2m_dg2"][:, 0]
        lw = log_weights(c, sx, g=np.array([g, 0.0])) + log_jacobian(ml) - log_jacobian(mc)
        w = np.where(np.isfinite(lw), np.exp(lw), 0.0)
        cs = np.cumsum(w)
        tot = np.bincount(gal - g0, w, g1 - g0)
        first = np.searchsorted(gal, np.arange(g0, g1))          # group start rows (gal sorted)
        base = np.concatenate([[0.0], cs])[first]
        has = np.bincount(gal - g0, minlength=g1 - g0) > 0
        ids = np.arange(g0, g1)[has]
        k = np.minimum(np.searchsorted(cs, base[has] + u[ids] * tot[has], side="right"), len(cs) - 1)
        out[a, ids] = ml[k]
        det[a, ids] = v[ids] < tot[has]
        det[a, np.arange(g0, g1)[~has]] = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--copies", required=True)
    p.add_argument("--real", default="bulgedisc_g2n_176k_sn8r", help="C_M, Sigma_X, layout")
    p.add_argument("--pop", default="bdg2n_sn8r_truegal", help="bias.py CATALOGS entry to write")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--step", type=int, default=3_000_000)
    a = p.parse_args()
    REAL = B.CATALOGS[a.real]
    real = {k: fitsio.read(f"{D}/{v}.fits", ext=1) for k, v in REAL.items()}
    hdr = fitsio.read_header(f"{D}/{REAL['plus']}.fits", ext=1)
    sx = real["plus"]["cov_odd"][0]
    cm = B.load_cov(f"{D}/{REAL['plus']}.fits")
    f = fitsio.FITS(a.copies)
    n = f["GALAXIES"].get_nrows()
    assert n == len(real["plus"]), (n, len(real["plus"]))
    rng = np.random.default_rng(a.seed)
    u, v = rng.random(n), rng.random(n)
    m = np.zeros((3, n, 5)); det = np.zeros((3, n), bool)
    nr, carry = f["COPIES"].get_nrows(), None
    for i in range(0, nr, a.step):
        c = f["COPIES"].read(columns=COLS, rows=np.arange(i, min(i + a.step, nr)))
        if carry is not None:
            c = np.concatenate([carry, c])
        if i + a.step < nr:      # the last galaxy may continue in the next block
            cut = np.searchsorted(c["gal"], c["gal"][-1])
            c, carry = c[:cut], c[cut:]
        pick(c, sx, u, v, m, det)
        print(f"  {min(i + a.step, nr)}/{nr} copies", flush=True)
    print("detected fraction per arm:", det.mean(1))

    # noise: paper eq. pMsG2, rejection as closed_loop --jacobian, candidates common to arms
    cand = rng.multivariate_normal(np.zeros(5), cm, (n, 48)).astype(np.float32)
    ur = rng.random((n, 48), dtype=np.float32)
    jac = lambda d: 0.25 * (d[..., 1] ** 2 - d[..., 2] ** 2 - d[..., 3] ** 2)
    for j, (arm, g1) in enumerate(ARMS):
        r = np.maximum(jac(m[j][:, None] + cand) / jac(m[j])[:, None], 0.0)
        r = np.where(np.isfinite(r), r, 1.0)
        ok = ur < r / 3.0
        noise = cand[np.arange(n), ok.argmax(1)]
        print(f"  {arm}: ratio > 3 in {(r > 3).mean():.2e} of candidates; "
              f"{(~ok.any(1)).sum()} rows with no acceptance (kept candidate 0)")
        mm = m[j] + noise
        bad = ~det[j] | ~np.isfinite(mm).all(1)
        mm[bad] = 0.0
        out = np.zeros(n, dtype=real[arm].dtype)
        out["moments"] = mm
        for col in ("cov", "cov_odd", "nda"):
            out[col] = real[arm][col][0]
        out["badcenter"] = bad
        h = {k: hdr[k] for k in ("PIXSCALE", "WTSIGMA", "PSFSIGMA", "NOISESIG",
                                 "PSFE1", "PSFE2", "POPKIND", "IMGNOISE", "SIG_XY")}
        h.update(G1=g1, G2=0.0, SEED=a.seed, TRUEGAL=os.path.basename(a.copies), NPOP=n)
        path = f"{D}/{B.CATALOGS[a.pop][arm]}.fits"
        fitsio.write(path, out, header=h, clobber=True)
        print(f"{arm:5s} g1={g1:+.2f}  wrote {path}  ({bad.sum()} undetected/non-finite)")


if __name__ == "__main__":
    main()
