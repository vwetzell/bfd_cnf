"""Assert that a rendered chain is internally consistent, before anything trains
on it.

This exists because of the 2026-09-02 audit: the last month produced at least
five bugs whose whole shape was "an artifact was silently not what the code
assumed", each of which invalidated numbers that had already been acted on.
Every check below is one of those bugs turned into an assertion.  It is cheap
(seconds) and it runs from `rebuild_v3.sh`.

What it does NOT check is anything about the flows -- run it after `render` and
`copies`, before `train`.

Run: `python dev/check_provenance.py [tag [prior_tag [size [pop]]]]`
     (defaults: tag "v3", size "200k", pop "bulgedisc")
"""

import sys

import fitsio
import numpy as np

D = "../bfd_cnf_imsims/data"
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def col(path, name, ext=1):
    return np.asarray(fitsio.read(path, ext=ext)[name], dtype=np.float64)


def pop(path):
    return fitsio.read(path, ext="POPULATION")


def same_pop(a, b, n=None):
    n = n or min(len(a), len(b))
    return all(np.array_equal(np.asarray(a[c][:n], float),
                              np.asarray(b[c][:n], float))
               for c in a.dtype.names)


def main(tag="v3", prior_tag=None, size="200k", pop_kind="bulgedisc"):
    # The prior tag may differ from the targets': the elliptical-PSF configs
    # (`dev/render_psfe.sh`) re-render only the TARGETS, because BFD's moments
    # are PSF-corrected and a re-rendered prior would reproduce the circular
    # one bit for bit.  Everything below then still applies, and the PSF check
    # is the one that pins the difference to where it belongs.
    prior_tag = prior_tag or tag
    prior = f"{D}/moments_{pop_kind}_{prior_tag}.fits"
    copies = f"{D}/copies_{pop_kind}_{prior_tag}.fits"
    arms = {k: f"{D}/targets_{tag}_{k}_{size}.fits"
            for k in ("g0", "g1p02", "g1m02")}

    print("headers")
    hdrs = {p: fitsio.read_header(p, ext=1)
            for p in [prior, copies, *arms.values()]}
    sig = {p: h["NOISESIG"] for p, h in hdrs.items()}
    check("one noise_sigma across every artifact",
          len(set(sig.values())) == 1, f"{sorted(set(sig.values()))}")
    check("target arms carry g1 = 0, +0.02, -0.02",
          sorted(hdrs[arms[k]]["G1"] for k in arms) == [-0.02, 0.0, 0.02])
    check("targets have image noise, prior does not",
          all(hdrs[arms[k]]["IMGNOISE"] for k in arms)
          and not hdrs[prior]["IMGNOISE"])
    # The three ARMS must share a PSF, or +g and -g are different experiments
    # and the antithetic difference measures the PSF instead of the shear.
    # The PRIOR is allowed to differ and is only reported: BFD's moments are
    # PSF-corrected, so an elliptical-PSF prior reproduces the circular one bit
    # for bit (gate G0) and `dev/render_psfe.sh` deliberately does not re-render
    # it.  (`copies` has no --psf-e CLI, so it is circular by construction.)
    e = lambda h: (h.get("PSFE1", 0.0), h.get("PSFE2", 0.0))
    check("the three target arms share one PSF ellipticity",
          len({e(hdrs[arms[k]]) for k in arms}) == 1,
          f"targets e_psf = {e(hdrs[arms['g0']])}, prior {e(hdrs[prior])}")

    print("\nSigma_X -- the centroid layer trains on the copies' value and is "
          "evaluated at the targets'")
    sx_copies = col(copies, "cov_odd", ext="GALAXIES")[0]
    sx_target = col(arms["g0"], "cov_odd")[0]
    check("copies Sigma_X == targets Sigma_X",
          np.array_equal(sx_copies, sx_target),
          f"{sx_copies[0]:.4f} vs {sx_target[0]:.4f}")
    # `bias.load_cov` asserts this itself, but a heteroscedastic catalog would
    # only show up there after the run had already started.
    cov = col(arms["g0"], "cov")
    check("C_M is constant across targets", bool(np.allclose(cov, cov[0])))

    print("\npairing -- +g and -g must be the SAME galaxies, row for row")
    p0 = pop(arms["g0"])
    for k in ("g1p02", "g1m02"):
        check(f"{k} population == g0 population", same_pop(p0, pop(arms[k])))

    print("\nprior -- the copy sum and the flow must share one prior, galaxy "
          "for galaxy")
    check("copies GALAXIES moments == prior moments",
          np.array_equal(col(copies, "moments", ext="GALAXIES"),
                         col(prior, "moments")))
    check("copies POPULATION == prior POPULATION",
          same_pop(pop(prior), pop(copies)))
    for c in ("dm_dg", "d2m_dg2"):
        check(f"copies GALAXIES {c} == prior {c}",
              np.array_equal(col(copies, c, ext="GALAXIES"), col(prior, c)))

    print("\nindependence -- prior and targets must be different draws")
    check("targets are NOT the prior's galaxies",
          not same_pop(pop(prior), p0, n=min(len(pop(prior)), len(p0))))

    print("\nchart -- the prior must lie inside the flow's chart, or build_flow "
          "raises several layers from the cause")
    sys.path.insert(0, ".")
    from models.bijections import POINT_SOURCE, POINT_SOURCE_MC  # noqa: E402
    m = col(prior, "moments")
    check("no prior row at or above the Mr/Mf ceiling",
          bool((m[:, 1] < POINT_SOURCE * m[:, 0]).all()),
          f"max {np.max(m[:, 1] / m[:, 0]):.4f} vs {POINT_SOURCE:.4f}")
    check("no prior row at or above the Mc/Mr ceiling",
          bool((m[:, 4] < POINT_SOURCE_MC * m[:, 1]).all()),
          f"max {np.max(m[:, 4] / m[:, 1]):.4f} vs {POINT_SOURCE_MC:.4f}")

    print(f"\n{len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
