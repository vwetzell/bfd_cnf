"""template_sum_reweight.py
=========================
BFD's exact template-sum shear log-derivatives, straight from a shifted-copies
catalog -- no flow, no training, just eq. (36)'s sum evaluated at g=0 by
autodiff of each copy's OWN exact shear response.

Two variants of the same sum:

  * "frozen"   -- the eq.-36 detection weight L(X(u); 0, Sigma_X) is held at
    its g=0 value.  This is what every existing template-sum check in this
    repo does.
  * "reweight" -- the weight is ALSO differentiated with respect to g, via
    each copy's own dxy_dg/d2xy_dg2 (see imsims/copies.py's log_weights
    docstring: a lensed galaxy's centroid response shifts with g exactly like
    its even moments do).

Both share one forward pass over the copies (the kernel term Cinv-weighted
residual is identical), so their paired difference isolates exactly the
centroid-reweight term with none of the population noise that would come from
two independent runs.

Run: `python dev/template_sum_reweight.py --selftest`
     `python dev/template_sum_reweight.py --n-targets 2000 --n-rows 1000000`
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import fitsio
import numpy as np

sys.path.insert(0, ".")
import bias as B
sys.path.insert(0, "dev")
import fit_centroid_K as FK

sys.path.insert(0, "../bfd_cnf_imsims")
from imsims.copies import log_weights

import jax
import jax.numpy as jnp

KFIT = None  # set by --kfit: binned centroid response table (dev/fit_centroid_K.py)
COPIES_PATH = "../bfd_cnf_imsims/data/copies_gauss2_fwd_g2v4n.fits"
TARGETS_PLUS = "../bfd_cnf_imsims/data/targets_g2v4_g1p02_22k.fits"
TARGETS_MINUS = "../bfd_cnf_imsims/data/targets_g2v4_g1m02_22k.fits"


def load_copies(path, n_rows):
    c = fitsio.FITS(path)["COPIES"][0:n_rows]
    c = c[c["gal"] < c["gal"][-1]]
    return c


def sinv_of(sigma_x):
    c00, c01, c11 = sigma_x
    det = c00 * c11 - c01 * c01
    return np.array([[c11, -c01], [-c01, c00]], dtype=np.float64) / det


def precompute_copy_arrays(c, cinv, sigma_x):
    """Copy-only quantities (independent of the target being scored):
    the base log weight, the two kernel-derivative ingredients (`m_a`,
    `m_ab`, and the target-independent `m_a^T Cinv m_b` correction), and the
    two weight-derivative ingredients `l_a`, `l_ab`.  All float32 -- the
    "f32 evaluation" side of the accumulation.
    """
    mom = np.asarray(c["moments"], dtype=np.float32)
    dm_dg = np.asarray(c["dm_dg"], dtype=np.float64)   # (n,2,5)
    d2m = np.asarray(c["d2m_dg2"], dtype=np.float64)   # (n,3,5)
    X = np.asarray(c["xy"], dtype=np.float64)          # (n,2)
    dX = np.asarray(c["dxy_dg"], dtype=np.float64)     # (n,2,2)
    d2X = np.asarray(c["d2xy_dg2"], dtype=np.float64)  # (n,3,2)
    if KFIT is not None:  # check 2: swap in the binned K(M) X prediction
        P = FK.predict(c, KFIT["edges"], KFIT["coef"])
        dX, d2X = P[:, :2], P[:, 2:]

    lw = log_weights(c, sigma_x).astype(np.float32)    # (n,)

    # m_a^T Cinv m_b, in float64 -- a copy-only, target-independent scalar per
    # pair, cheap relative to the O(n_targets x n_copies) kernel term below.
    tmp = np.einsum("nam,ml->nal", dm_dg, cinv)  # (n,2,5)
    macb = np.empty((len(c), 3), dtype=np.float64)
    macb[:, 0] = np.einsum("nm,nm->n", tmp[:, 0], dm_dg[:, 0])
    macb[:, 1] = np.einsum("nm,nm->n", tmp[:, 0], dm_dg[:, 1])
    macb[:, 2] = np.einsum("nm,nm->n", tmp[:, 1], dm_dg[:, 1])

    sinv = sinv_of(sigma_x)
    sinvX = X @ sinv.T   # (n,2)
    la = np.empty((len(c), 2), dtype=np.float64)
    la[:, 0] = -np.einsum("nm,nm->n", sinvX, dX[:, 0])
    la[:, 1] = -np.einsum("nm,nm->n", sinvX, dX[:, 1])

    sinvdX = np.einsum("nal,lk->nak", dX, sinv)  # (n,2,2)
    lab = np.empty((len(c), 3), dtype=np.float64)
    lab[:, 0] = -np.einsum("nm,nm->n", dX[:, 0], sinvdX[:, 0]) - np.einsum(
        "nm,nm->n", sinvX, d2X[:, 0])
    lab[:, 1] = -np.einsum("nm,nm->n", dX[:, 0], sinvdX[:, 1]) - np.einsum(
        "nm,nm->n", sinvX, d2X[:, 1])
    lab[:, 2] = -np.einsum("nm,nm->n", dX[:, 1], sinvdX[:, 1]) - np.einsum(
        "nm,nm->n", sinvX, d2X[:, 2])

    return dict(
        mom=mom, ma=dm_dg.astype(np.float32), mab=d2m.astype(np.float32),
        macb=macb.astype(np.float32), lw=lw,
        la=la.astype(np.float32), lab=lab.astype(np.float32))


@jax.jit
def _process_chunk(Mt, cinv, mom, ma, mab, macb, lw, la, lab):
    """One (target-batch x copy-chunk) block.  Returns this chunk's own
    running max and its sum-over-copies-in-this-chunk (both still float32);
    the caller folds them into the float64 running accumulators."""
    diff = Mt[:, None, :] - mom[None, :, :]           # (Bt,Bc,5)
    rc = jnp.einsum("tcm,ml->tcl", diff, cinv)         # (Bt,Bc,5)
    logN = -0.5 * jnp.sum(diff * rc, axis=-1)          # (Bt,Bc)
    logf = lw[None, :] + logN                          # (Bt,Bc)

    kl_a = jnp.einsum("tcm,cam->tca", rc, ma)          # (Bt,Bc,2)
    kl_ab = jnp.einsum("tcm,ckm->tck", rc, mab) - macb[None, :, :]

    def hstack(s):
        return jnp.stack(
            [s[..., 0] * s[..., 0], s[..., 0] * s[..., 1],
             s[..., 1] * s[..., 1]], axis=-1)

    s_frz = kl_a
    h_frz = hstack(s_frz) + kl_ab
    s_rew = kl_a + la[None, :, :]
    h_rew = hstack(s_rew) + kl_ab + lab[None, :, :]

    cmax = jnp.max(logf, axis=1)
    w = jnp.exp(logf - cmax[:, None])
    P = jnp.sum(w, axis=1)
    Sa_frz = jnp.einsum("tc,tca->ta", w, s_frz)
    Hab_frz = jnp.einsum("tc,tck->tk", w, h_frz)
    Sa_rew = jnp.einsum("tc,tca->ta", w, s_rew)
    Hab_rew = jnp.einsum("tc,tck->tk", w, h_rew)
    return cmax, P, Sa_frz, Hab_frz, Sa_rew, Hab_rew


def _finish(P, Sa, Hab):
    q = Sa / P[:, None]
    r = np.empty((len(P), 2, 2), dtype=np.float64)
    r[:, 0, 0] = Hab[:, 0] / P - q[:, 0] * q[:, 0]
    r[:, 0, 1] = r[:, 1, 0] = Hab[:, 1] / P - q[:, 0] * q[:, 1]
    r[:, 1, 1] = Hab[:, 2] / P - q[:, 1] * q[:, 1]
    return q, r


def _new_acc(n_t):
    return dict(M=np.full(n_t, -np.inf), P=np.zeros(n_t),
                Sf=np.zeros((n_t, 2)), Hf=np.zeros((n_t, 3)),
                Sr=np.zeros((n_t, 2)), Hr=np.zeros((n_t, 3)))


def accumulate(acc, c, targets, cinv, sigma_x, target_batch, copy_chunk):
    """Streams one block of copies against `targets`, folding into `acc`
    (float64 running log-sum-exp accumulators from `_new_acc`)."""
    arrs = precompute_copy_arrays(c, cinv, sigma_x)
    n_t, n_c = len(targets), len(c)
    Mt_all = jnp.asarray(targets, dtype=jnp.float32)
    dev = {k: jnp.asarray(v) for k, v in arrs.items()}
    cinv_d = jnp.asarray(cinv, dtype=jnp.float32)
    M = acc["M"]

    for c0 in range(0, n_c, copy_chunk):
        c1 = min(c0 + copy_chunk, n_c)
        chunk = {k: v[c0:c1] for k, v in dev.items()}
        for t0 in range(0, n_t, target_batch):
            t1 = min(t0 + target_batch, n_t)
            out = _process_chunk(
                Mt_all[t0:t1], cinv_d, chunk["mom"], chunk["ma"],
                chunk["mab"], chunk["macb"], chunk["lw"], chunk["la"],
                chunk["lab"])
            cmax, Pc, Sfc, Hfc, Src, Hrc = (np.asarray(x, dtype=np.float64)
                                             for x in out)
            sl = slice(t0, t1)
            new_max = np.maximum(M[sl], cmax)
            so = np.exp(M[sl] - new_max)
            sn = np.exp(cmax - new_max)
            acc["P"][sl] = acc["P"][sl] * so + Pc * sn
            for k, v in (("Sf", Sfc), ("Hf", Hfc), ("Sr", Src), ("Hr", Hrc)):
                acc[k][sl] = acc[k][sl] * so[:, None] + v * sn[:, None]
            M[sl] = new_max


def finish_acc(acc):
    q_frz, r_frz = _finish(acc["P"], acc["Sf"], acc["Hf"])
    q_rew, r_rew = _finish(acc["P"], acc["Sr"], acc["Hr"])
    return q_frz, r_frz, q_rew, r_rew


def run_arm(c, targets, cinv, sigma_x, target_batch, copy_chunk):
    """Whole in-memory catalog in one block -- returns
    (q_frz, r_frz, q_rew, r_rew) for every target."""
    acc = _new_acc(len(targets))
    accumulate(acc, c, targets, cinv, sigma_x, target_batch, copy_chunk)
    return finish_acc(acc)


def brute_force(c, targets, cinv, sigma_x):
    """Same formulas, no chunking, plain float64 numpy -- for --selftest."""
    arrs = precompute_copy_arrays(c, cinv, sigma_x)
    arrs = {k: v.astype(np.float64) for k, v in arrs.items()}
    diff = targets[:, None, :].astype(np.float64) - arrs["mom"][None, :, :]
    rc = np.einsum("tcm,ml->tcl", diff, cinv)
    logN = -0.5 * np.sum(diff * rc, axis=-1)
    logf = arrs["lw"][None, :] + logN
    kl_a = np.einsum("tcm,cam->tca", rc, arrs["ma"])
    kl_ab = np.einsum("tcm,ckm->tck", rc, arrs["mab"]) - arrs["macb"][None, :, :]

    def hstack(s):
        return np.stack([s[..., 0] * s[..., 0], s[..., 0] * s[..., 1],
                          s[..., 1] * s[..., 1]], axis=-1)

    s_frz = kl_a
    h_frz = hstack(s_frz) + kl_ab
    s_rew = kl_a + arrs["la"][None, :, :]
    h_rew = hstack(s_rew) + kl_ab + arrs["lab"][None, :, :]

    cmax = logf.max(axis=1)
    w = np.exp(logf - cmax[:, None])
    P = w.sum(axis=1)
    q_frz, r_frz = _finish(P, np.einsum("tc,tca->ta", w, s_frz),
                            np.einsum("tc,tck->tk", w, h_frz))
    q_rew, r_rew = _finish(P, np.einsum("tc,tca->ta", w, s_rew),
                            np.einsum("tc,tck->tk", w, h_rew))
    return q_frz, r_frz, q_rew, r_rew


def selftest():
    c = load_copies(COPIES_PATH, 200_000)
    targets = fitsio.read(TARGETS_PLUS, rows=range(3))
    M = np.asarray(targets["moments"], dtype=np.float64)
    sigma_x = targets["cov_odd"][0]
    cinv = np.linalg.inv(B.load_cov(TARGETS_PLUS))

    q_frz_s, r_frz_s, q_rew_s, r_rew_s = run_arm(
        c, M, cinv, sigma_x, target_batch=3, copy_chunk=50_000)
    q_frz_b, r_frz_b, q_rew_b, r_rew_b = brute_force(c, M, cinv, sigma_x)

    def reldiff(a, b):
        return np.max(np.abs(a - b) / (np.abs(b) + 1e-30))

    d = max(reldiff(q_frz_s, q_frz_b), reldiff(r_frz_s, r_frz_b),
            reldiff(q_rew_s, q_rew_b), reldiff(r_rew_s, r_rew_b))
    print(f"selftest max rel diff = {d:.3e}")
    print("PASS" if d < 1e-4 else "FAIL")
    return d < 1e-4


def paired_bootstrap(qp_f, rp_f, qm_f, rm_f, qp_r, rp_r, qm_r, rm_r,
                      n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(qp_f)
    point_f = np.array(B.bias(qp_f, rp_f, qm_f, rm_f))
    point_r = np.array(B.bias(qp_r, rp_r, qm_r, rm_r))
    point_diff = point_r - point_f
    diffs = np.empty((n_boot, 3))
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        bf = B.bias(qp_f[idx], rp_f[idx], qm_f[idx], rm_f[idx])
        br = B.bias(qp_r[idx], rp_r[idx], qm_r[idx], rm_r[idx])
        diffs[i] = np.array(br) - np.array(bf)
    err = diffs.std(axis=0, ddof=1)
    return point_f, point_r, point_diff, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", default=COPIES_PATH)
    ap.add_argument("--targets-plus", default=TARGETS_PLUS)
    ap.add_argument("--targets-minus", default=TARGETS_MINUS)
    ap.add_argument("--n-rows", type=int, default=4_000_000)
    ap.add_argument("--n-targets", type=int, default=None)
    ap.add_argument("--target-batch", type=int, default=256)
    ap.add_argument("--copy-chunk", type=int, default=32_768)
    ap.add_argument("--block", type=int, default=10_000_000)
    ap.add_argument("--flux-max", type=float, default=None,
                    help="window targets on flux (M[:,0]) < this, per arm")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", default="logs/template_sum_reweight.npz")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--kfit", default=None,
                    help="npz from dev/fit_centroid_K.py: use K(M) X for dX, d2X")
    args = ap.parse_args()

    if args.kfit:
        global KFIT
        z = np.load(args.kfit)
        KFIT = dict(coef=z["coef"], edges=[z[f"edge{i}"] for i in range(4)])
    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    # ponytail: copies streamed from disk in --block rows; 115M rows won't fit
    # in RAM or on the GPU at once.  Log-sum-exp sums don't care about the
    # block split, only the last (possibly partial) galaxy is dropped.
    F = fitsio.FITS(args.copies)["COPIES"]
    n_rows = min(args.n_rows, F.get_nrows())
    last_gal = F[n_rows - 1:n_rows]["gal"][0] if n_rows < F.get_nrows() else None

    setup = {}
    for name, path in (("plus", args.targets_plus), ("minus", args.targets_minus)):
        n = args.n_targets
        t = fitsio.read(path, rows=range(n) if n else None)
        M = np.asarray(t["moments"], dtype=np.float64)
        sigma_x = t["cov_odd"]
        assert np.allclose(sigma_x, sigma_x[0]), f"{name}: cov_odd varies"
        setup[name] = (M, np.linalg.inv(B.load_cov(path)), sigma_x[0],
                       _new_acc(len(M)))

    t0 = time.time()
    n_used = 0
    for b0 in range(0, n_rows, args.block):
        c = F[b0:min(b0 + args.block, n_rows)]
        if last_gal is not None:
            c = c[c["gal"] != last_gal]
        n_used += len(c)
        for name, (M, cinv, sx, acc) in setup.items():
            accumulate(acc, c, M, cinv, sx, args.target_batch, args.copy_chunk)
        print(f"  rows {b0 + len(c)}/{n_rows}  {time.time() - t0:.0f}s", flush=True)
        del c
    print(f"copies used: {n_used} rows")

    arms = {}
    for name, (M, cinv, sx, acc) in setup.items():
        q_frz, r_frz, q_rew, r_rew = finish_acc(acc)
        arms[name] = dict(q_frz=q_frz, r_frz=r_frz, q_rew=q_rew, r_rew=r_rew,
                          flux=M[:, 0])

    if args.flux_max is not None:
        for name, a in arms.items():
            keep = a["flux"] < args.flux_max
            print(f"{name}: window flux<{args.flux_max}: {keep.sum()}/{len(keep)}")
            arms[name] = {k: v[keep] for k, v in a.items()}

    p, m = arms["plus"], arms["minus"]
    m1_f, c1_f, c2_f = B.bias(p["q_frz"], p["r_frz"], m["q_frz"], m["r_frz"])
    m1_r, c1_r, c2_r = B.bias(p["q_rew"], p["r_rew"], m["q_rew"], m["r_rew"])
    print(f"frozen:   m1={m1_f:.5f} c1={c1_f:.3e} c2={c2_f:.3e}")
    print(f"reweight: m1={m1_r:.5f} c1={c1_r:.3e} c2={c2_r:.3e}")

    pf, pr, diff, err = paired_bootstrap(
        p["q_frz"], p["r_frz"], m["q_frz"], m["r_frz"],
        p["q_rew"], p["r_rew"], m["q_rew"], m["r_rew"], n_boot=args.n_boot)
    print(f"paired diff (reweight - frozen): "
          f"dm1={diff[0]:+.5f}+/-{err[0]:.5f}  "
          f"dc1={diff[1]:+.3e}+/-{err[1]:.3e}  "
          f"dc2={diff[2]:+.3e}+/-{err[2]:.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        q_frz_plus=p["q_frz"], r_frz_plus=p["r_frz"],
        q_rew_plus=p["q_rew"], r_rew_plus=p["r_rew"],
        q_frz_minus=m["q_frz"], r_frz_minus=m["r_frz"],
        q_rew_minus=m["q_rew"], r_rew_minus=m["r_rew"],
        flux_plus=p["flux"], flux_minus=m["flux"])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
