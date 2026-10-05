"""Paired m1 shift from the per-target 1/J(m) draw weight (--target-jacobian).

Compares pqr/cv2_s0_tj_sub.npz (first 130000 catalog rows, --target-jacobian)
against the SAME targets in pqr/cv2_s0.npz (full run, no target J).  Same seed,
batch and row order, so the IS draws are common and only the weights differ.
Windowed m1 at the base window, uncorrected and corrected with the J-weighted
selection terms (logs/wshift_full/base.log), N_ns counted over the 130000-row
slice.  Numpy only; run after the GPU job.
"""
import os, re, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias import bias, window_mask

NROW, SIZE, FLUX, G = 130000, (2.2, 3.2), (3000, 20000), 0.02
D = "../bfd_cnf_imsims/data/targets_bdg2_{}_176k_varobs_sn8.fits"
full, sub = np.load("pqr/cv2_s0.npz"), np.load(sys.argv[1] if len(sys.argv) > 1 else "pqr/cv2_s0_tj_sub.npz")

# align rows by the exact observed moments of both arms
key = lambda d: [tuple(r) for r in np.concatenate([d["obs_plus"], d["obs_minus"]], 1)]
pos = {k: i for i, k in enumerate(key(full))}
idx = np.array([pos.get(k, -1) for k in key(sub)])
assert (idx >= 0).all(), f"{(idx < 0).sum()} subset rows not in the full run"
print(f"{len(idx)} subset targets aligned; corr(q1) = "
      f"{np.corrcoef(full['plus_q'][idx, 0], sub['plus_q'][:, 0])[0, 1]:.4f}")

t = open("logs/wshift_full/base.log").read()
ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
rs = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]",
                        t).groups(), float).reshape(2, 2)

cat = {k: fitsio.read(D.format(a), columns=["moments", "badcenter"], rows=np.arange(NROW))
       for k, a in (("p", "g1p02"), ("m", "g1m02"), ("z", "g0"))}
keep = ~np.any([cat[k]["badcenter"].astype(bool) for k in cat], axis=0)
nin = {k: window_mask(np.asarray(cat[k]["moments"], float)[keep], SIZE, FLUX).sum() for k in "pm"}

sp = window_mask(sub["obs_plus"], SIZE, FLUX)
sm = window_mask(sub["obs_minus"], SIZE, FLUX)
assert abs(sp.sum() - nin["p"]) < 50, (sp.sum(), nin["p"])
A = [full[k][idx] for k in ("plus_q", "plus_r", "minus_q", "minus_r")]
B = [sub[k] for k in ("plus_q", "plus_r", "minus_q", "minus_r")]


def m1(qr, i=slice(None), corr=False):
    ns = None
    if corr:
        ns = tuple((NROW - s[i].sum(), ps, qs, rs) for s in (sp, sm))
    return bias(*[x[i] for x in qr], G, sel=(sp[i], sm[i]), ns=ns)[0]


rng = np.random.default_rng(0)
boots = [rng.integers(0, len(sp), len(sp)) for _ in range(300)]
for corr in (False, True):
    a, b = m1(A, corr=corr), m1(B, corr=corr)
    d = [m1(B, i, corr) - m1(A, i, corr) for i in boots]
    print(f"{'corrected  ' if corr else 'uncorrected'}  no target J {a:+.5f}   target J {b:+.5f}   "
          f"shift {b - a:+.5f} +/- {np.std(d):.5f}")
