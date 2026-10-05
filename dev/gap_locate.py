"""Where does the varobs - sn8r m1 gap live?  (2026-10-03)  Offline.

Same galaxies in both catalogs (rows match).  Uncorrected windowed m1 linearised
into per-row contributions (each arm: dg_i = [q_i + ghat r_i] / sum(-r), first
component), so  gap ~ sum_i (d_i^varobs - d_i^sn8r).  Then
  1. outliers: top-k share of the summed gap, tails, 20-block jackknife;
  2. edges: the gap per bin of flux / size, binned on the sn8r g0 moments (one
     variable for both catalogs, independent of either arm's noise).
Usage: python dev/gap_locate.py [A.npz A_TAG B.npz B_TAG]   (gap = B - A)
"""
import os, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias import bias, window_mask

D, F, G = "../bfd_cnf_imsims/data/", (3000, 20000), 0.02
S = (2.2, float(os.environ.get("SIZE_HI", 3.2)))   # upper size edge of the window
RUNS = {"A": ("pqr/cv6_s2_sn8r_nopool.npz", "sn8r"), "B": ("pqr/cv6_s2.npz", "varobs_sn8")}
if len(sys.argv) == 5:
    RUNS = {"A": tuple(sys.argv[1:3]), "B": tuple(sys.argv[3:5])}


def load(pqr, tag):   # as dev/condition_split.py: per-target values onto catalog rows
    d = np.load(pqr)
    cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_{tag}.fits", columns=["moments", "badcenter"])
           for a in ("g1p02", "g1m02", "g0")}
    n = len(cat["g0"])
    keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
    pos = {tuple(r): i for i, r in enumerate(np.asarray(cat["g1p02"]["moments"], np.float64))}
    row = np.array([pos.get(tuple(r), -1) for r in d["obs_plus"]])
    assert (row >= 0).all()
    out = {k: np.zeros((n,) + d[k].shape[1:]) for k in ("plus_q", "plus_r", "minus_q", "minus_r")}
    for k in out:
        out[k][row] = d[k]
    integ = np.zeros(n, bool); integ[row] = True
    out["sp"] = keep & integ & window_mask(np.asarray(cat["g1p02"]["moments"], np.float64), S, F)
    out["sm"] = keep & integ & window_mask(np.asarray(cat["g1m02"]["moments"], np.float64), S, F)
    out["size"] = np.stack([cat[a]["moments"][:, 1] / np.maximum(cat[a]["moments"][:, 0], 1) for a in ("g1p02", "g1m02")])
    return out


def contrib(x, ref):
    """Per-row contribution to m1 about A's per-arm ghat: sum_i (q_i + ghat_A r_i) / sum(-r)
    = ghat_x - ghat_A exactly (1-component), so B's minus A's sums to the gap."""
    c = np.zeros(len(x["sp"]))
    for arm, s, sgn in (("plus", x["sp"], 1), ("minus", x["sm"], -1)):
        q, r = x[arm + "_q"][:, 0] * s, x[arm + "_r"][:, 0, 0] * s
        c += sgn * (q + ref[arm] * r) / -r.sum() / (2 * G)
    return c


def ghat1(x, arm, s):
    return (x[arm + "_q"][:, 0] * s).sum() / -(x[arm + "_r"][:, 0, 0] * s).sum()


R = {k: load(*v) for k, v in RUNS.items()}
mA, mB = (bias(R[k]["plus_q"], R[k]["plus_r"], R[k]["minus_q"], R[k]["minus_r"], G,
               sel=(R[k]["sp"], R[k]["sm"]))[0] for k in "AB")
ref = {"plus": ghat1(R["A"], "plus", R["A"]["sp"]), "minus": ghat1(R["A"], "minus", R["A"]["sm"])}
d = contrib(R["B"], ref) - contrib(R["A"], ref)
print(f"m1 uncorr A {mA:+.5f}  B {mB:+.5f}  gap {mB - mA:+.5f}   linearised sum {d.sum():+.5f}")
nz = d != 0
a = np.abs(d[nz]); o = np.argsort(-a)
print(f"rows contributing {nz.sum()}   sd {d[nz].std():.2e}   kurtosis "
      f"{((d[nz] - d[nz].mean())**4).mean() / d[nz].var()**2:.1f}")
for k in (10, 100, 1000):
    print(f"  top {k:>5d} |d| rows: sum {d[nz][o[:k]].sum():+.5f}   gap without them {d.sum() - d[nz][o[:k]].sum():+.5f}")
blk = np.array_split(np.random.default_rng(0).permutation(len(d)), 20)
jk = np.array([d.sum() - d[b].sum() * 1.0 for b in blk]) * 20 / 19
print(f"  20-block jackknife: gap {d.sum():+.5f} +/- {np.sqrt(19 / 20 * ((jk - jk.mean())**2).sum()):.5f}"
      f"   block sums min/max {min(d[b].sum() for b in blk):+.5f}/{max(d[b].sum() for b in blk):+.5f}")
print(f"  rows in only one catalog's window: {(R['A']['sp'] ^ R['B']['sp']).sum()} plus, "
      f"{(R['A']['sm'] ^ R['B']['sm']).sum()} minus;  their gap share "
      f"{d[(R['A']['sp'] ^ R['B']['sp']) | (R['A']['sm'] ^ R['B']['sm'])].sum():+.5f}")

m0 = np.asarray(fitsio.read(D + "targets_bdg2_g0_176k_sn8r.fits", columns=["moments"])["moments"], np.float64)
var = {"flux Mf": m0[:, 0], "size Mr/Mf": m0[:, 1] / np.maximum(m0[:, 0], 1)}
rng = np.random.default_rng(1)
for name, v in var.items():
    edges = np.percentile(v[R["A"]["sp"] | R["B"]["sp"]], np.linspace(0, 100, 9))
    edges[0], edges[-1] = -np.inf, np.inf
    print(f"\n{name} octile (sn8r g0)    gap share   +/-")
    for b in range(8):
        i = (v > edges[b]) & (v <= edges[b + 1])
        bs = [d[i][rng.integers(0, i.sum(), i.sum())].sum() for _ in range(200)]
        print(f"  {edges[b]:>9.3f} - {edges[b + 1]:<9.3f} {d[i].sum():+.5f}  {np.std(bs):.5f}")

# window-membership flips (in one catalog's window, not the other's) vs rows in both
flip = (R["A"]["sp"] ^ R["B"]["sp"]) | (R["A"]["sm"] ^ R["B"]["sm"])
both = (R["A"]["sp"] | R["A"]["sm"] | R["B"]["sp"] | R["B"]["sm"]) & ~flip
print("\nsubset           rows    gap share   +/-")
for name, i in (("in both windows", both), ("window flips", flip),
                ("  only in B (varobs)", flip & (R["B"]["sp"] | R["B"]["sm"]) & ~(R["A"]["sp"] | R["A"]["sm"])),
                ("  only in A (sn8r)", flip & (R["A"]["sp"] | R["A"]["sm"]) & ~(R["B"]["sp"] | R["B"]["sm"]))):
    bs = [d[i][rng.integers(0, i.sum(), i.sum())].sum() for _ in range(200)]
    print(f"  {name:<22s} {i.sum():>6d}  {d[i].sum():+.5f}  {np.std(bs):.5f}")

# the band just inside the upper edge: rows whose measured size (either catalog, either arm)
# lies in (BAND, S_hi] -- what the window at edge BAND loses
if "BAND" in os.environ:
    band = (np.any([(x["size"] > float(os.environ["BAND"])) & (x["size"] <= S[1]) for x in R.values()], axis=(0, 1))
            & (R["A"]["sp"] | R["A"]["sm"] | R["B"]["sp"] | R["B"]["sm"]))
    db = d[band]; ob = np.argsort(-np.abs(db))
    bs = [db[rng.integers(0, len(db), len(db))].sum() for _ in range(500)]
    print(f"\nband size ({os.environ['BAND']}, {S[1]}]: {band.sum()} rows   gap share {db.sum():+.5f} +/- {np.std(bs):.5f}"
          f"   rest {d[~band].sum():+.5f}")
    for k in (1, 3, 10, 30, 100):
        print(f"  top {k:>3d} |d| band rows: sum {db[ob[:k]].sum():+.5f}   band without them {db.sum() - db[ob[:k]].sum():+.5f}")
    print(f"  band rows d>0: {(db > 0).sum()}  d<0: {(db < 0).sum()}   flips among them {flip[band].sum()} share {d[band & flip].sum():+.5f}")
