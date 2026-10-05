"""Window-shift test of the selection correction (2026-09-28).

The sims' true m1 is zero for ANY window, so if R_s is wrong by a fraction eps
the corrected m1 moves with the window, in proportion to the correction's size
(corrected - uncorrected).  Per-target Q, R are reused from a --save-pqr run
(the window only re-selects rows and changes N_ns); only P_s, Q_s, R_s are
recomputed, by bias.py on a 2000-target slice (its m1 is ignored; its
selection terms depend only on the flow and the window).  Same --seed, so the
selection draws are common across windows and the differences are paired.

Usage: python dev/window_shift.py run      (GPU, serial bias.py calls)
       python dev/window_shift.py combine  (numpy; after `run` finishes)
"""
import os, re, subprocess, sys
import numpy as np

# TAG=varobs_sn8|sn8r picks the catalog; FLOW/PQR/LOGS/WIN override (2026-10-03, cv6 varobs-vs-sn8r gap).
TAG = os.environ.get("TAG", "varobs_sn8")
POP = f"bulgedisc_g2n_176k_{TAG}"
FLOW = os.environ.get("FLOW", "flows/cv2/centroid_s0.eqx")
PQR = os.environ.get("PQR", "pqr/cv2_s0.npz")   # the per-target Q, R to re-window
# JAC=full|quad sets bias.py --window-jacobian (the paper's |J(M)| weight in the
# selection probability) and writes logs/wshift_<JAC>; unset is the plain
# pre-2026-09-28 window ("none") into logs/wshift.
JAC = os.environ.get("JAC", "none")
LOGS = os.environ.get("LOGS", "logs/wshift" + (f"_{JAC}" if JAC != "none" else ""))
D = "../bfd_cnf_imsims/data/"
ARMS = {k: f"targets_bdg2_{a}_176k_{TAG}" for k, a in (("plus", "g1p02"), ("minus", "g1m02"), ("zero", "g0"))}
# Must sit inside the saved run's pre-filter pad: size (1.76, 3.84), flux (2400, 24000).
WINDOWS = {"base": ((2.2, 3.2), (3000, 20000)),
           "flux4k": ((2.2, 3.2), (4000, 20000)),
           "flux6k": ((2.2, 3.2), (6000, 20000)),
           "size30": ((2.2, 3.0), (3000, 20000)),
           "size28": ((2.2, 2.8), (3000, 20000)),
           "size24lo": ((2.4, 3.2), (3000, 20000)),
           "size35": ((2.2, 3.5), (3000, 20000)),
           "size31": ((2.2, 3.1), (3000, 20000)),
           "size33": ((2.2, 3.3), (3000, 20000)),
           "size34": ((2.2, 3.4), (3000, 20000))}
if "WIN" in os.environ:   # e.g. WIN="base flux6k size28 size35"
    WINDOWS = {k: WINDOWS[k] for k in os.environ["WIN"].split()}
BF = ("--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm --support "
      "--floor-eps 0 --g 0.02 --window-fd 0.005 --n-targets 2000").split() + ["--window-jacobian", JAC]


def run():
    os.makedirs(LOGS, exist_ok=True)
    for tag, (s, f) in WINDOWS.items():
        log = f"{LOGS}/{tag}.log"
        if os.path.exists(log) and "R_s =" in open(log).read():
            continue
        cmd = ["python", "-u", "bias.py", "--pop", POP, "--flow", FLOW, *BF,
               "--window-size", *map(str, s), "--window-flux", *map(str, f)]
        with open(log, "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(tag, "done", flush=True)


def sel_terms(tag):
    t = open(f"{LOGS}/{tag}.log").read()
    ps = float(re.search(r"P_s = ([-+.\de]+)", t).group(1))
    qs = np.array(re.search(r"Q_s = \(([-+.\de]+), ([-+.\de]+)\)", t).groups(), float)
    r = np.array(re.search(r"R_s = \[\[([-+.\de]+), ([-+.\de]+)\], \[([-+.\de]+), ([-+.\de]+)\]\]", t).groups(), float)
    return ps, qs, r.reshape(2, 2)


def combine(nboot=200):
    import fitsio
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bias import bias, window_mask   # numpy-only functions; no device work
    d = np.load(PQR)
    qp, rp, qm, rm = d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"]
    op, om = d["obs_plus"], d["obs_minus"]
    # N_ns over the FULL population, as bias.py counts it: NPOP minus the
    # in-window count among the rows kept after the badcenter drop.
    full = {k: fitsio.read(D + v + ".fits", columns=["moments", "badcenter"]) for k, v in ARMS.items()}
    keep = ~np.any([full[k]["badcenter"].astype(bool) for k in full], axis=0)
    npop = fitsio.read_header(D + ARMS["plus"] + ".fits", ext=1)["NPOP"]
    rng = np.random.default_rng(0)
    idxs = [rng.integers(0, len(qp), len(qp)) for _ in range(nboot)]
    res = {}
    for tag, (s, f) in WINDOWS.items():
        ps, qs, rs = sel_terms(tag)
        sp, sm = window_mask(op, s, f), window_mask(om, s, f)
        nin_p = window_mask(np.asarray(full["plus"]["moments"], float)[keep], s, f).sum()
        nin_m = window_mask(np.asarray(full["minus"]["moments"], float)[keep], s, f).sum()
        # the saved rows must hold every in-window galaxy (less the few
        # sane_targets drops), or the window has left the pre-filter pad
        assert 0 <= nin_p - sp.sum() < 50 and 0 <= nin_m - sm.sum() < 50, (tag, sp.sum(), nin_p)

        def est(i=slice(None), corr=True):
            # replicate N_ns = NPOP - replicate in-window count
            ns = lambda sel, nin: (npop - sel[i].sum(), ps, qs, rs) if corr else None
            return bias(qp[i], rp[i], qm[i], rm[i], 0.02, sel=(sp[i], sm[i]),
                        ns=(ns(sp, nin_p), ns(sm, nin_m)))[0]
        res[tag] = dict(c=est(), u=est(corr=False), rs=rs[0, 0], ps=ps, n=int(sp.sum()),
                        cb=np.array([est(i) for i in idxs]), ub=np.array([est(i, False) for i in idxs]))
        print(f"{tag:>9s} done", flush=True)
    b = res["base"]
    print(f"\n{'window':>9s} {'N_in':>7s} {'P_s':>7s} {'R_s11':>8s} {'m1 uncorr':>10s} {'m1 corr':>18s} "
          f"{'corr':>8s} {'d(m1 corr) vs base':>22s} {'d(corr) vs base':>16s}")
    for tag, r in res.items():
        dc = r["cb"] - b["cb"]
        dcorr = (r["cb"] - r["ub"]) - (b["cb"] - b["ub"])
        print(f"{tag:>9s} {r['n']:>7d} {r['ps']:>7.4f} {r['rs']:>+8.4f} {r['u']:>+10.5f} "
              f"{r['c']:>+10.5f}+/-{r['cb'].std():.5f} {r['c'] - r['u']:>+8.4f} "
              f"{r['c'] - b['c']:>+12.5f}+/-{dc.std():.5f} {(r['c'] - r['u']) - (b['c'] - b['u']):>+10.4f}"
              f"+/-{dcorr.std():.4f}")
    # m1_corr = m_true + eps * corr  =>  slope of corrected m1 on the correction, paired
    tags = [t for t in res if t != "base"]
    x = np.array([(res[t]["c"] - res[t]["u"]) - (b["c"] - b["u"]) for t in tags])
    y = np.array([res[t]["c"] - b["c"] for t in tags])
    slope = lambda x, y: (x @ y) / (x @ x)
    sb = [slope(np.array([(res[t]["cb"][k] - res[t]["ub"][k]) - (b["cb"][k] - b["ub"][k]) for t in tags]),
                np.array([res[t]["cb"][k] - b["cb"][k] for t in tags])) for k in range(nboot)]
    print(f"\nd(m1 corr) = eps * d(correction):  eps = {slope(x, y):+.3f} +/- {np.std(sb):.3f}"
          f"   (eps = -0.33 would make the base +0.008 all selection)")


if __name__ == "__main__":
    {"run": run, "combine": combine}[sys.argv[1]]()
