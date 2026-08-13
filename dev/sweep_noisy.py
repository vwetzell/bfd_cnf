"""Sweep training knobs against the NOISY bias, paired, at a cost that allows a sweep.

Every previous sweep scored `dev/check_noiseless_shear_recovery.py`, and
[[convolution-reaches-past-the-cut]] showed that metric does not track the noisy
bias: `--score-weight 3` improves it 4x and moves the noisy ramp by ~1 sigma.
So the 23-run capacity null, and the 5x shear-LR effect, are both statements
about a quantity we no longer care about.  This scores the thing we do.

**The shared `--proposal-flow` is GONE** ([[shared-proposal-flow-is-a-trap]]):
at alpha < 1 the weights carry `P_eval / q_proposal`, and one arm put 73% of the
ensemble `sum|R11|` on a single target and returned m1 = -0.84.  Every arm is
now self-proposed, so the arms are UNPAIRED at +/-0.002 each.  That is enough
here: the question is whether `--score-weight` moves m1 from +0.025 towards 0
on the measured cut, a 12-sigma target, and whether it flattens a 0.031 ramp
whose amplitude carries ~0.006.  Buying back the pairing costs ~2 h/arm
(alpha = 1 needs ~16x the draws for the same ESS) and is not worth it.

`ref` still reuses `flows/shear_bounded.eqx` / `flows/centroid_deep_bounded.eqx`
and their existing PQR -- that run WAS self-proposed (its `--proposal-flow`
equalled its `--flow`), so it is comparable and costs nothing.

**Draws were NOT reduced, and must not be.**  The first version of this ran at
2048 draws to save time; the `ref` arm then returned m1 +0.0162 against the
full-precision +0.0249 and a ramp amplitude of -0.0622 against +0.0314 --
sign-flipped and twice the size.  Median ESS at 2048 is ~100, where the paper's
own 1/ESS bias (sec. 2.5) is ~1%, which is exactly the m1 discrepancy; and ESS
starvation is resolution-dependent ([[ess-starvation-is-the-resolution-edge]]),
so it corrupts the AMPLITUDE preferentially.  The draw count is not negotiable.

    python dev/sweep_noisy.py                 # every arm
    python dev/sweep_noisy.py ref sw10        # named arms only
"""
import json
import subprocess
import sys
import time

import fitsio
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
from dev.check_bias_flatness import noise_free       # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SP = ("/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/"
      "19d04b68-9c48-44c7-a1da-e37c41b2f767/scratchpad")
REF = ("/tmp/claude-1000/-home-vwetzell-gitrepos-bfd-cnf/"
       "9f45a3ac-5204-4774-9992-dff1492404c7/scratchpad/bounded_with.npz")
OUT = "dev/sweep_noisy.jsonl"
BULK = "flows/bulk_bounded.eqx"
SAMPLES, CHUNK, G, NBOOT = 8192, 4096, 0.02, 300

# (bulk --score-weight, extra shear args).  The shear layer's own score term is
# now measured out -- 1/3/10 are a true null and the term is saturated -- so
# what is left is the BULK, the only stage still trained on likelihood alone.
# `bulk.score_match` supervises d log p / d logit(Mr/Mf), the resolution
# direction the ramp lives in.  Its scale is ~-1000 against an NLL of ~41, so
# the weights are small; at 1e-2 the term goes -969 -> -1221 for 0.43 nats.
ARMS = {
    "ref":  None,          # reuse what is on disk; see `train_and_measure`
    "smb3": ("1e-3", []),
    "smb2": ("1e-2", []),
    # `ref` reuses a bulk trained in an earlier session, so a score-matched arm
    # differs from it by the objective AND by whatever the retrain changes.
    # These two rebuild the identical chain at weight 0: the first is the fair
    # control, the pair of them is the run-to-run floor -- which has never been
    # measured on the noisy metric, and [[varq-correction-implemented-but-null]]
    # says `shear.train` does not reproduce.
    "ctl0a": ("0", []),
    "ctl0b": ("0", []),
}


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def train_and_measure(name, arm):
    """Train bulk (if the arm changes it) -> shear -> centroid -> noisy PQR."""
    if arm is None:
        return REF                           # already run at exactly these settings
    bulk_w, extra = arm
    bk = BULK if bulk_w is None else f"flows/bulk_{name}.eqx"
    sh, ce = f"flows/shear_{name}.eqx", f"flows/centroid_deep_{name}.eqx"
    npz = f"{SP}/{name}.npz"
    if bulk_w is not None:
        run(["python", "bulk.py", "train", "--data", f"{DATA}/moments.fits",
             "--flow", bk, "--score-weight", bulk_w])
    run(["python", "shear.py", "train", "--data", f"{DATA}/moments.fits",
         "--flow", sh, "--init", bk] + extra)
    run(["python", "centroid.py", "train", "--copies",
         f"{DATA}/copies_bulgedisc_deep.fits", "--flow", ce, "--init", sh])
    run(["python", "bias.py", "--flow", ce, "--pop", "bulgedisc_deep",
         "--n-targets", "20000", "--samples", str(SAMPLES), "--chunk", str(CHUNK),
         "--alpha", "0.5", "--save-pqr", npz])     # self-proposed; see the docstring
    return npz


def selections():
    """(noisy cut, noise-free Mr/Mf) on the rows the npz files carry."""
    cat = bias.CATALOGS["bulgedisc_deep"]
    rows = {k: fitsio.read(f"{DATA}/{v}.fits")[:20000] for k, v in cat.items()}
    keep = np.ones(len(rows["zero"]), bool)
    for v in rows.values():
        keep &= ~v["badcenter"]
    z = rows["zero"][keep]
    M0 = np.asarray(z["moments"], np.float64)
    sel = M0[:, 0] > 2800
    return sel & (M0[:, 1] / M0[:, 0] < 3.5), noise_free(z, M0[:, 1] / M0[:, 0], sel), sel


def score(npz, cut, rhat, sel, rng):
    """m1 on the measured cut, and the ramp amplitude across noise-free quartiles.

    Errors are the arm's OWN galaxy bootstrap -- unpaired, because the arms no
    longer share draws.  The amplitude resamples the base selection ONCE per
    draw and re-bins it, so the two quartiles' correlation is kept rather than
    added in quadrature.
    """
    d = np.load(npz)
    get = lambda: (d["plus_q"], d["plus_r"], d["minus_q"], d["minus_r"])
    m1 = lambda s: bias.bias(*get(), G, s)[0]
    base = sel & (rhat < 3.5)
    e = np.quantile(rhat[base], np.linspace(0, 1, 5))
    e[0], e[-1] = -np.inf, np.inf
    inq = [base & (rhat >= e[k]) & (rhat < e[k + 1]) for k in range(4)]
    q = [m1(s) for s in inq]

    icut, ibase = np.flatnonzero(cut), np.flatnonzero(base)
    b_cut = [m1(rng.choice(icut, len(icut))) for _ in range(NBOOT)]
    b_amp = []
    for _ in range(NBOOT):
        i = rng.choice(ibase, len(ibase))
        b_amp.append(m1(i[inq[3][i]]) - m1(i[inq[0][i]]))
    return {"m1_cut": float(m1(cut)), "m1_cut_err": float(np.std(b_cut)),
            "quartiles": [float(v) for v in q],
            "amplitude": float(q[-1] - q[0]),
            "amplitude_err": float(np.std(b_amp)),
            "m1_nf": float(m1(base))}


def main():
    want = sys.argv[1:] or list(ARMS)
    if any(w not in ARMS for w in want):
        raise SystemExit(f"unknown arm; have {list(ARMS)}")
    cut, rhat, sel = selections()
    print(f"measured cut Mf>2800 & Mr/Mf<3.5: N={cut.sum()}; "
          f"{SAMPLES} draws, self-proposed\n")

    rng = np.random.default_rng(0)
    res = {}
    for name in want:
        t0 = time.time()
        print(f"{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        npz = train_and_measure(name, ARMS[name])
        res[name] = {"name": name, "npz": npz, "minutes": (time.time() - t0) / 60,
                     **score(npz, cut, rhat, sel, rng)}
        with open(OUT, "a") as f:
            f.write(json.dumps(res[name]) + "\n")
        print(f"  {name}: m1(cut) {res[name]['m1_cut']:+.4f}  "
              f"amplitude {res[name]['amplitude']:.4f}  "
              f"[{res[name]['minutes']:.1f} min]", flush=True)

    print(f"\n{'arm':>6s} {'m1(cut)':>19s} {'amplitude':>19s}   quartiles")
    for name, r in res.items():
        print(f"{name:>6s} {r['m1_cut']:>+10.4f} +/- {r['m1_cut_err']:.4f} "
              f"{r['amplitude']:>+10.4f} +/- {r['amplitude_err']:.4f}   "
              + " ".join(f"{v:+.4f}" for v in r["quartiles"]))


if __name__ == "__main__":
    main()
