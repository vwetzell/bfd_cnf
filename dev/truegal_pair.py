"""sims - truegal, paired galaxy by galaxy (dev/truegal_loop.py). (2026-10-01)

Both catalogs hold the SAME 520k galaxies row for row (seed-2 copies = the sn8r
targets); they differ only in how noise/recentring made the measured moments.
Same flow on both sides, so the selection terms are shared and the uncorrected
difference IS the corrected one.  Saved PQR files carry no row index: rows are
recovered by matching each run's obs_plus to its catalog's plus-arm moments.
Paired bootstrap over galaxy rows; per Mf bin (each arm's own observed Mf) the
contribution w_b (sims_b - truegal_b) as in dev/real_closed_bins.py.
  python dev/truegal_pair.py pqr/cv2_s0_sn8r_J.npz bulgedisc_g2n_176k_sn8r pqr/truegal_cv2s0.npz bdg2n_sn8r_truegal
"""
import sys
import numpy as np, fitsio
sys.path.insert(0, ".")
import bias as B

D = "../bfd_cnf_imsims/data"
S, F = (2.2, 3.2), (3000, 20000)
EDGES = [3000, 4000, 5000, 7000, 10000, 20000]


def load(npz, pop):
    d = np.load(npz)
    cat = fitsio.read(f"{D}/{B.CATALOGS[pop]['plus']}.fits", columns=["moments"])["moments"]
    key = {r.tobytes(): i for i, r in enumerate(np.asarray(cat, np.float32))}
    rows = np.array([key[r.tobytes()] for r in np.asarray(d["obs_plus"], np.float32)])
    n = len(cat)
    out = {}
    for a in ("plus", "minus"):
        q, r, o = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 5))
        q[rows], r[rows], o[rows] = d[a + "_q"], d[a + "_r"], d["obs_" + a]
        ok = np.zeros(n, bool)
        ok[rows] = np.isfinite(d[a + "_q"]).all(1) & np.isfinite(d[a + "_r"]).all((1, 2))
        out[a] = (np.where(ok[:, None], q, 0), np.where(ok[:, None, None], r, 0),
                  B.window_mask(o, S, F) & ok, o[:, 0])
    return out


def m1(k, i, lo=0, hi=np.inf):
    (qp, rp, sp, fp), (qm, rm, sm, fm) = k["plus"], k["minus"]
    sp = sp & (fp >= lo) & (fp < hi)
    sm = sm & (fm >= lo) & (fm < hi)
    w = -(rp[i][sp[i], 0, 0].sum() + rm[i][sm[i], 0, 0].sum())
    return B.bias(qp[i], rp[i], qm[i], rm[i], 0.02, sel=(sp[i], sm[i]))[0], w


sims, tg = load(sys.argv[1], sys.argv[2]), load(sys.argv[3], sys.argv[4])
n = len(sims["plus"][0])
rng = np.random.default_rng(0)
boot = [rng.integers(0, n, n) for _ in range(200)]
allr = np.arange(n)
print(f"{sys.argv[1]} - {sys.argv[3]}: uncorrected window m1 (same flow => = corrected difference)")
tot_s, tot_t = m1(sims, allr)[1], m1(tg, allr)[1]
for lo, hi in [(0, np.inf)] + list(zip(EDGES[:-1], EDGES[1:])):
    (a, wa), (b, wb) = m1(sims, allr, lo, hi), m1(tg, allr, lo, hi)
    d = [m1(sims, i, lo, hi)[0] - m1(tg, i, lo, hi)[0] for i in boot]
    print(f"Mf {lo:>6g}-{hi:<6g} sims {a:+.4f} truegal {b:+.4f}  diff {a - b:+.4f}+/-{np.std(d):.4f}"
          f"   w sims {wa / tot_s:.3f} truegal {wb / tot_t:.3f}  contribution {wa / tot_s * a - wb / tot_t * b:+.4f}")
