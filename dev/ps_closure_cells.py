"""P_s closure per window: each run's model P_s (from its bias.py log) against the target
catalog's own tapered in-window fraction n_in / NPOP over the same --n-targets slice and the
same badcenter drop as bias.py.  (2026-10-14)  numpy only.
Usage: python dev/ps_closure_cells.py LOG SIZE_LO SIZE_HI FLUX_LO FLUX_HI [LOG ...]"""
import re, sys, os
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bias as B

D, N = "../bfd_cnf_imsims/data/", 300000
B.WINDOW_TAPER = (0.05, 150.0)
cat = {a: fitsio.read(D + f"targets_bdg2_{a}_1m_desv.fits", columns=["moments", "badcenter"], rows=range(N))
       for a in ("g1p02", "g1m02", "g0")}
keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
hdr = fitsio.read_header(D + "targets_bdg2_g1p02_1m_desv.fits", ext=1)
npop = hdr["NPOP"] * N / hdr["NAXIS2"]
args = sys.argv[1:]
print(f"{'run':44s} {'P_s model':>10s} {'P_obs':>9s} {'gap':>10s} {'sigma':>6s}  rel")
for i in range(0, len(args), 5):
    log, s, f = args[i], tuple(map(float, args[i + 1:i + 3])), tuple(map(float, args[i + 3:i + 5]))
    ps = float(re.search(r"P_s = ([-+.\de]+)", open(log).read()).group(1))
    p = np.mean([np.asarray(B.window_mask(np.asarray(cat[a]["moments"], np.float64)[keep], s, f), float).sum() / npop
                 for a in ("g1p02", "g1m02")])
    err = np.sqrt(p * (1 - p) / npop)
    print(f"{os.path.basename(log)[:44]:44s} {ps:10.5f} {p:9.5f} {ps - p:+10.5f} {(ps - p) / err:+6.1f}  {(ps - p) / p:+.2%}")
