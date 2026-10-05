"""Where in flux does the consistent-J windowed m1 excess live? (2026-09-29)

Offline, on pqr/cv2_s0.npz (plain) and pqr/cv2_s0_J.npz (consistent J), same
seed/draws, rows aligned by observed moments.  Per base-window flux band:
  - hybrid: m1 with J Q/R only for that band's targets (plain elsewhere), so
    the per-band pieces of the target-J shift add up;
  - Fisher ratio sum(q^2)/sum(-r) (health check, [[fisher-identity-is-the-health-check]]).
Selection terms: the J model's base window (logs/wshift_full/base.log).
"""
import os, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias import bias, window_mask
os.environ["JAC"] = "full"
import dev.window_shift as ws

S, F = ws.WINDOWS["base"]
A, B = np.load("pqr/cv2_s0.npz"), np.load("pqr/cv2_s0_J.npz")
key = lambda d: [tuple(r) for r in np.concatenate([d["obs_plus"], d["obs_minus"]], 1)]
pos = {k: i for i, k in enumerate(key(A))}
ia = np.array([pos.get(k, -1) for k in key(B)]); assert (ia >= 0).all()
K = ("plus_q", "plus_r", "minus_q", "minus_r")
a, b = [A[k][ia] for k in K], [B[k] for k in K]
op, om = B["obs_plus"], B["obs_minus"]
sp, sm = window_mask(op, S, F), window_mask(om, S, F)

full = {k: fitsio.read(ws.D + v + ".fits", columns=["moments", "badcenter"]) for k, v in ws.ARMS.items()}
keep = ~np.any([full[k]["badcenter"].astype(bool) for k in full], axis=0)
npop = fitsio.read_header(ws.D + ws.ARMS["plus"] + ".fits", ext=1)["NPOP"]
ps, qs, rs = ws.sel_terms("base")


def m1(qr, i=slice(None)):
    ns = tuple((npop - s[i].sum(), ps, qs, rs) for s in (sp, sm))
    return bias(*[x[i] for x in qr], 0.02, sel=(sp[i], sm[i]), ns=ns)[0]


def mix(band):   # band: bool per row -> J values there, plain elsewhere
    return [np.where(band.reshape((-1,) + (1,) * (x.ndim - 1)), y, x) for x, y in zip(a, b)]


rng = np.random.default_rng(0)
boots = [rng.integers(0, len(sp), len(sp)) for _ in range(200)]
m0, mJ = m1(a), m1(b)
print(f"base window corrected m1: plain targets {m0:+.5f}, J targets {mJ:+.5f}  (J selection terms both)")
edges = [3000, 3500, 4000, 5000, 6000, 8000, 12000, 20000]
fl = 0.5 * (op[:, 0] + om[:, 0])
print(f"{'Mf band':>13s} {'N_in':>6s} {'dm1 from J':>18s} {'Fisher plain':>12s} {'Fisher J':>9s}")
for lo, hi in zip(edges[:-1], edges[1:]):
    band = (fl >= lo) & (fl < hi)
    h = mix(band)
    d = m1(h) - m0
    db = np.std([m1(h, i) - m1(a, i) for i in boots])
    w = band & sp
    fr = lambda q, r: (q[w] ** 2).sum(0) / (-np.einsum("nii->ni", r[w])).sum(0)
    print(f"{lo:>6d}-{hi:<6d} {w.sum():>6d} {d:>+10.5f}+/-{db:.5f} "
          f"{fr(a[0], a[1])[0]:>12.3f} {fr(b[0], b[1])[0]:>9.3f}")
