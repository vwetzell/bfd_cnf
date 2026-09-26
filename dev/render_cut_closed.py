"""Does the real catalog's render cut explain real-minus-closed integrated m1?

The real g2v4 targets were only RENDERED where the NOISELESS (Mf, Mr/Mf) lies
in (2000, 21000) x (2.0, 3.4) (imsims --region, pad 1000 / 0.2).  bias.py's
--prefilter-pad 0.2 integrates OBSERVED (2400, 24000) x (1.76, 3.84), so the
closed loop (no render cut) integrates galaxies the real catalog never had.
Apply the same noiseless cut to the closed loop (noise regenerated from its
seed) and recompute bias.py's integrated m1 on the 4x PQR.
"""
import sys
sys.path.insert(0, ".")
import fitsio
import numpy as np

import bias as B

D = "../bfd_cnf_imsims/data"
C = B.CATALOGS["gauss2_v4n_closed"]
z = fitsio.read(f"{D}/{C['zero']}.fits", columns=["moments"])["moments"]
pl = fitsio.read(f"{D}/{C['plus']}.fits", columns=["moments"])["moments"]
seed = fitsio.read_header(f"{D}/{C['zero']}.fits", 1)["SEED"]
cm = B.load_cov(f"{D}/{B.CATALOGS['gauss2_v4n']['plus']}.fits")
nl = z - np.random.default_rng(seed).multivariate_normal(np.zeros(5), cm, len(z))
r = nl[:, 1] / nl[:, 0]
keep_all = (nl[:, 0] > 2000) & (nl[:, 0] < 21000) & (r > 2.0) & (r < 3.4)

d = np.load("pqr/closed_g2v4n_Ke_4x.npz")
pl = pl.astype("<f8")
row = {m.tobytes(): i for i, m in enumerate(pl)}
idx = np.array([row[m.tobytes()] for m in d["obs_plus"].astype("<f8")])
keep = keep_all[idx]
q = (d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"])
print(f"closed integrated set {len(idx)}, inside real render cut {keep.sum()} ({keep.mean():.1%})")
print(f"  per 65372-galaxy population: closed {len(idx) / 4:.0f} -> {keep.sum() / 4:.0f}  (real integrates 18608)")
for name, s in (("all integrated", None), ("render cut", keep), ("outside cut", ~keep)):
    m1 = B.bias(*q, 0.02, s)[0]
    dm1 = B.bootstrap(*q, 0.02, 200, 0, s)[0]
    print(f"  {name:15s} m1 = {m1:+.5f} +/- {dm1:.5f}")
flux = d["moments"][:, 0]
e = np.percentile(flux, [0, 20, 40, 60, 80, 100])
print("  quintile   all      cut    (real: +0.039 +0.012 +0.002 +0.003 -0.001)")
for i in range(5):
    s = (flux >= e[i]) & (flux <= e[i + 1])
    print(f"  q{i + 1}      {B.bias(*q, 0.02, s)[0]:+.4f}  {B.bias(*q, 0.02, s & keep)[0]:+.4f}")

# does the render cut reach the WINDOW (which the corrected m1 lives in)?
w = lambda m: B.window_mask(m, (2.2, 3.2), (3000.0, 20000.0))
sp, sm = w(d["obs_plus"]), w(d["obs_minus"])
inw = sp | sm
print(f"\nin-window (either arm) {inw.sum()}, outside render cut {(inw & ~keep).sum()} ({(inw & ~keep).mean() / inw.mean():.3%})")
for name, s in (("all", np.ones_like(keep)), ("render cut", keep)):
    m1 = B.bias(*q, 0.02, sel=(sp & s, sm & s))[0]
    print(f"  windowed uncorrected, {name:10s} m1 = {m1:+.5f}")
