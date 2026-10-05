"""Calibrate bulgedisc's size median and ellipticity width (`sim.sample_population`'s
size_logmedian, sigma_e) so its MEASURED moments match the g2v4n gauss2_fwd prior:
bisect sigma_e on median |e|, then size_logmedian on median Mr/Mf, alternating.
Renders at the prior's noise 0.093, as in dev/render_g2v4n.sh.
Usage: python dev/bulgedisc_calib.py [n_per_render]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "bfd_cnf_imsims"))
import fitsio
import numpy as np


def stats(m):
    Mf, Mr, M1, M2, Mc = m.T[:5]
    return dict(log10Mf=np.log10(Mf), MrMf=Mr / Mf, McMr=Mc / Mr, e=np.hypot(M1, M2) / Mr)


def render(sim, logmed, se, n, seed):
    base = sim.POPULATIONS["bulgedisc"]
    sim.POPULATIONS["bulgedisc"] = (lambda k, rng: sim.sample_population(k, rng, logmed, se), base[1])
    rows, _ = sim.make_catalog(n, seed=seed, noise_sigma=0.093, add_noise=True, nproc=24)
    sim.POPULATIONS["bulgedisc"] = base
    return stats(rows["moments"])


def show(lab, s):
    print(f"{lab:34s}" + "  ".join(f"{k} {np.median(v):.4f}/{np.std(v):.4f}" for k, v in s.items()), flush=True)


if __name__ == "__main__":
    from imsims import sim
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    T = stats(fitsio.read("../bfd_cnf_imsims/data/moments_gauss2_fwd_g2v4n.fits", rows=range(100000))["moments"])
    show("target gauss2_fwd g2v4n (med/sd)", T)
    tm = {k: np.median(v) for k, v in T.items()}
    logmed, se = sim.SIZE_LOGMEDIAN, sim.ELLIP_SIGMA_WIDE
    for it in range(2):
        lo, hi = 0.1, se            # median |e| rises with sigma_e
        for _ in range(7):
            se = 0.5 * (lo + hi)
            if np.median(render(sim, logmed, se, n, 0)["e"]) > tm["e"]: hi = se
            else: lo = se
        se = 0.5 * (lo + hi)
        lo, hi = logmed - 0.3, logmed + 0.3   # Mr/Mf falls as size grows
        for _ in range(7):
            logmed = 0.5 * (lo + hi)
            if np.median(render(sim, logmed, se, n, 0)["MrMf"]) > tm["MrMf"]: lo = logmed
            else: hi = logmed
        logmed = 0.5 * (lo + hi)
        print(f"round {it}: sigma_e {se:.5f}  size_logmedian {logmed:.5f}", flush=True)
    show(f"held-out seed 1, n={n}", render(sim, logmed, se, n, 1))
