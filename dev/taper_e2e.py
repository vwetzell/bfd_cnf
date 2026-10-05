"""Offline end-to-end check of bias.py's TAPERED target path.  (2026-10-03)  CPU only.

Per-target Q, R are window-independent, so feeding the saved full run (pqr/cv6_s2.npz,
varobs) through bias.py's own window_mask -> n_out_full -> bias()/bootstrap() exactly as
main() does reproduces what a full `--window-taper` run would print, minus re-integration.
Gate: hard must equal the logged run (bias_cv6_s2.log, m1 corrected -0.00042), tapered
must equal dev/taper_gap.py (-0.00148).
Usage: JAX_PLATFORMS=cpu python dev/taper_e2e.py
"""
import os, sys
import numpy as np, fitsio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import bias as B


def sel_terms(log):   # as dev/taper_gap.py (importing it would run it)
    t = open(log).read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


D, S, F, G = "../bfd_cnf_imsims/data/", (2.2, 3.2), (3000, 20000), 0.02
d = np.load("pqr/cv6_s2.npz")
qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
cat = {a: fitsio.read(D + f"targets_bdg2_{a}_176k_varobs_sn8.fits", columns=["moments", "badcenter"])
       for a in ("g1p02", "g1m02", "g0")}
keep = ~np.any([cat[a]["badcenter"].astype(bool) for a in cat], axis=0)
npop = fitsio.read_header(D + "targets_bdg2_g1p02_176k_varobs_sn8.fits", ext=1)["NPOP"]

for kind, taper, log in (("hard", None, "logs/cv6/bias_cv6_s2.log"),
                         ("taper", (0.05, 150.0), "logs/cv6/taper_sel_varobs_sn8.log")):
    B.WINDOW_TAPER = taper
    ps, qs, rs = sel_terms(log)
    n_out = {k: B.n_out_from_npop(npop, round(float(B.window_mask(
        np.asarray(cat[a]["moments"], np.float64)[keep], S, F).sum())))
             for k, a in (("plus", "g1p02"), ("minus", "g1m02"))}
    sp, sm = B.window_mask(d["obs_plus"], S, F), B.window_mask(d["obs_minus"], S, F)
    ns = ((n_out["plus"], ps, qs, rs), (n_out["minus"], ps, qs, rs))
    mu = B.bias(qp, rp, qm, rm, G, sel=(sp, sm))
    mc = B.bias(qp, rp, qm, rm, G, sel=(sp, sm), ns=ns)
    eu = B.bootstrap(qp, rp, qm, rm, G, 200, 0, sel=(sp, sm))
    ec = B.bootstrap(qp, rp, qm, rm, G, 200, 0, sel=(sp, sm), ns=ns, weights=d["prefilter_w"])
    print(f"{kind:>5s}  N_ns {n_out['plus']}/{n_out['minus']}  "
          f"uncorr m1 = {mu[0]:+.5f} +/- {eu[0]:.5f}   corrected m1 = {mc[0]:+.5f} +/- {ec[0]:.5f}   "
          f"c1 = {mc[1]:+.2e}  c2 = {mc[2]:+.2e}")
