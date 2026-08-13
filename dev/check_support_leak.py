"""Does the flow put mass outside the population's own support?

`dev/check_edge_distance.py` shows the size-axis m1 failure sits at each
population's OWN support edge, not at the shared chart ceiling.  gauss2 says
why that edge is hard: `sim.sample_population_gauss2` keeps a draw only if the
Newton solve converges into a physical box (sigma, rho, |e| ranges), so the
realised density is P0 TRUNCATED -- a cliff, and ~70% of P0's mass is on the
wrong side of it.  A smooth autoregressive flow cannot put a cliff there; it
must smear across, which costs density just inside the edge and puts mass on
the empty side.

Direct test: sample the trained flow at g = 0, run each sample through the
SAME acceptance criterion the population used, and count the failures.  A
population catalog is 100% accepting by construction, so anything above the
solver's own failure rate is leaked mass.

    python dev/check_support_leak.py
"""
import argparse
import sys

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../bfd_cnf_imsims")

import bulk                                          # noqa: E402
import shear as shear_top                            # noqa: E402


def accept(m):
    """The `sample_population_gauss2` keep-mask, per moment row.

    `imsims.analytic` turns on jax x64 at import, which would make
    `build_flow`'s leaves float64 and break deserialising a float32 flow -- so
    it is imported here, after the flow is loaded and sampled.
    """
    from imsims import analytic, sim

    theta, resid = sim._gauss2_solve(np.asarray(m, dtype=np.float64))
    flux, sigma, rho, e1, e2 = np.asarray(jax.vmap(analytic.unpack)(theta)).T
    return {
        "solve": np.asarray(resid) < sim.GAUSS2_RESID_TOL,
        "sigma": (sigma > sim.GAUSS2_SIGMA_RANGE[0]) & (sigma < sim.GAUSS2_SIGMA_RANGE[1]),
        "rho": (rho > sim.GAUSS2_RHO_RANGE[0]) & (rho < sim.GAUSS2_RHO_RANGE[1]),
        "e": np.hypot(e1, e2) < sim.GAUSS2_E_MAX,
    }, rho


def sim_ranges():
    from imsims import sim
    return sim.GAUSS2_RHO_RANGE


# Chart coordinates worth testing (`bulk.to_coords` order); slot 1 is the size
# axis we bin ON, so it is not also a test coordinate.
TEST_COORDS = {0: "log10 Mf", 2: "Mc/Mr", 3: "M1/Mr", 4: "M2/Mr"}
Q_LO, Q_HI = 0.0005, 0.9995        # -> a correct sample lands outside 0.10%


def empirical_leak(m_train, s, nbin=10, rng=None):
    """Conditional-support leak WITHOUT an analytic acceptance box.

    bulgedisc has no `keep` mask to replay, so the support has to come from the
    catalog itself.  Split the training moments in two: half A *defines* the
    support (a per-size-bin quantile range in each chart coordinate), half B is
    the control that says what out-rate a genuinely-on-support sample gives.
    Anything the flow scores above B is leaked mass, and B absorbs the
    estimator's own bias -- a finite-sample quantile range, a coordinate that
    is not really box-shaped -- because A and B are the same population.

    Reported per size bin, since that is the axis m1 collapses along.
    """
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(len(m_train))
    A, B = m_train[perm[::2]], m_train[perm[1::2]]
    tA, tB, tS = (bulk.to_coords(np.asarray(v, np.float64)) for v in (A, B, s))

    edges = np.quantile(tA[:, 1], np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    rows = []
    for k in range(nbin):
        inA, inB, inS = ((t[:, 1] >= edges[k]) & (t[:, 1] < edges[k + 1])
                         for t in (tA, tB, tS))
        if inA.sum() < 500 or inS.sum() < 200:
            continue
        outB = np.zeros(int(inB.sum()), bool)
        outS = np.zeros(int(inS.sum()), bool)
        per_coord = {}
        for c, name in TEST_COORDS.items():
            lo, hi = np.quantile(tA[inA, c], [Q_LO, Q_HI])
            bB = (tB[inB, c] < lo) | (tB[inB, c] > hi)
            bS = (tS[inS, c] < lo) | (tS[inS, c] > hi)
            per_coord[name] = (100 * bB.mean(), 100 * bS.mean())
            outB |= bB
            outS |= bS
        rows.append((0.5 * (max(edges[k], tA[:, 1].min())
                            + min(edges[k + 1], tA[:, 1].max())),
                     int(inS.sum()), 100 * outB.mean(), 100 * outS.mean(),
                     per_coord))
    return rows


K_NN = 10


def knn_leak(m_train, s, nbin=10, rng=None):
    """Conditional-support leak by k-NN distance -- the non-axis-aligned test.

    `empirical_leak`'s per-coordinate quantile box is provably too weak: on
    gauss2 it recovers +2.0% excess in the top size bin where the exact
    acceptance box says 15.2%, and it blames the spin-2 pair when the truth is
    mostly `rho`.  A box can only catch a leak that is aligned with a chart
    axis, and the physical support is not box-shaped.

    Distance to the k-th nearest training point does not care about
    orientation: a sample in a region the population never reaches is far from
    all of it, whichever way the support is tilted.  Same A/B design as
    `empirical_leak` -- half A is the reference cloud, half B calibrates the
    threshold so the estimator's own bias cancels.
    """
    from scipy.spatial import cKDTree

    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(len(m_train))
    A, B = m_train[perm[::2]], m_train[perm[1::2]]
    tA, tB, tS = (bulk.to_coords(np.asarray(v, np.float64)) for v in (A, B, s))

    # Whiten on A, so no coordinate dominates the metric by its units alone.
    mu, sd = tA.mean(0), tA.std(0)
    zA, zB, zS = ((t - mu) / sd for t in (tA, tB, tS))

    tree = cKDTree(zA)
    dB = tree.query(zB, k=K_NN, workers=-1)[0][:, -1]
    dS = tree.query(zS, k=K_NN, workers=-1)[0][:, -1]

    edges = np.quantile(tA[:, 1], np.linspace(0, 1, nbin + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    rows = []
    for k in range(nbin):
        inB = (tB[:, 1] >= edges[k]) & (tB[:, 1] < edges[k + 1])
        inS = (tS[:, 1] >= edges[k]) & (tS[:, 1] < edges[k + 1])
        if inB.sum() < 500 or inS.sum() < 200:
            continue
        # Threshold PER BIN: the cloud's own density varies along the size
        # axis, so a global cut would read sparsity as leakage.
        thr = np.quantile(dB[inB], 0.999)
        rows.append((0.5 * (max(edges[k], tA[:, 1].min())
                            + min(edges[k + 1], tA[:, 1].max())),
                     int(inS.sum()), 0.10, 100 * (dS[inS] > thr).mean(),
                     float(np.median(dS[inS]) / np.median(dB[inB]))))
    return rows


def print_knn(rows):
    print(f"\n  k-NN conditional-support leak (k={K_NN}, threshold = q999 of "
          f"the control's own k-NN distance, per size bin)")
    print(f"  {'z0 centre':>10s} {'n_flow':>7s} {'ctrl B':>8s} {'flow':>8s} "
          f"{'excess':>8s} {'med ratio':>10s}")
    for z, n, ob, os_, ratio in rows:
        print(f"  {z:+10.3f} {n:7d} {ob:7.2f}% {os_:7.2f}% {os_ - ob:+7.2f}% "
              f"{ratio:10.3f}")
    print("    (med ratio = median flow k-NN distance / median control's; "
          ">1 means the flow sits in sparser regions than the population does)")


def print_empirical(rows):
    print(f"\n  EMPIRICAL conditional-support leak "
          f"(support = train half A, q[{Q_LO}, {Q_HI}] per size bin)")
    print(f"  {'z0 centre':>10s} {'n_flow':>7s} {'ctrl B':>8s} {'flow':>8s} "
          f"{'excess':>8s}   " + "  ".join(f"{n:>13s}" for n in TEST_COORDS.values()))
    for z, n, ob, os_, per in rows:
        cells = "  ".join(f"{b:5.2f}/{f:6.2f}" for b, f in per.values())
        print(f"  {z:+10.3f} {n:7d} {ob:7.2f}% {os_:7.2f}% "
              f"{os_ - ob:+7.2f}%   {cells}")
    print("    (per-coordinate cells are ctrl/flow, in %)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", default="flows/shear_gauss2.eqx")
    p.add_argument("--train-data", default="../bfd_cnf_imsims/data/moments_gauss2.fits")
    p.add_argument("-n", type=int, default=100_000)
    p.add_argument("--nbin", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exact-box", action="store_true",
                   help="also replay gauss2's analytic acceptance box "
                        "(gauss2 flows only)")
    a = p.parse_args()

    m_train = shear_top.load(a.train_data)[0]
    m_train = m_train[:int(0.9 * len(m_train))]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True)
    flow = eqx.tree_deserialise_leaves(a.flow, flow)

    s = np.asarray(flow.sample(jr.key(a.seed), (a.n,), condition=np.zeros(2)),
                   dtype=np.float64)
    finite = np.isfinite(s).all(1)
    print(f"{a.flow}: {a.n} samples at g=0, {int((~finite).sum())} non-finite")
    s = s[finite]

    # Works on any population; on gauss2 it is cross-checked against the exact
    # box below, which is the point of running both.
    print_empirical(empirical_leak(m_train, s, nbin=a.nbin))
    print_knn(knn_leak(m_train, s, nbin=a.nbin))

    if not a.exact_box:
        return

    masks, rho = accept(s)
    ok = np.logical_and.reduce(list(masks.values()))
    print(f"\n  outside the population's support: {100 * (~ok).mean():.2f}%")
    for name, mk in masks.items():
        print(f"    fails {name:6s}: {100 * (~mk).mean():6.2f}%")

    # Which side of the concentration boundary it escapes across -- that is the
    # side a bound would have to close.
    lo, hi = sim_ranges()
    print(f"    rho < {lo}: {100 * (rho < lo).mean():.2f}%   "
          f"rho > {hi}: {100 * (rho > hi).mean():.2f}%   "
          f"(train rho range is the population's, by construction)")

    # Where the leaked mass sits on the size axis -- the edge hypothesis says it
    # piles up beyond the accepted population's own Mr/Mf edge.
    y = s[:, 1] / s[:, 0]
    y_train = m_train[:, 1] / m_train[:, 0]
    edge = np.quantile(y_train, 0.999)
    print(f"\n  train Mr/Mf: max {y_train.max():.4f}, q999 {edge:.4f}   "
          f"(chart ceiling {bulk.POINT_SOURCE:.4f})")
    print(f"  flow  Mr/Mf: max {y.max():.4f}, q999 {np.quantile(y, 0.999):.4f}")
    print(f"  flow mass above the train q999: {100 * (y > edge).mean():.2f}% "
          f"(population: 0.10% by definition)")
    print(f"  of the rejected samples, {100 * (y[~ok] > edge).mean():.1f}% "
          f"are above it")

    # A right marginal says nothing about the CONDITIONAL: the support in the
    # other four coordinates narrows toward the size edge, so the flow can put
    # the right amount of mass at high Mr/Mf and still put it in the wrong
    # place.  m1 collapses over exactly this range, so bin the same way.
    print("\n  rejection rate vs size (the m1 collapse runs over these bins):")
    q = np.linspace(0, 1, 11)
    ed = np.quantile(y, q)
    ed[-1] = np.inf
    for k in range(10):
        s_k = (y >= ed[k]) & (y < ed[k + 1])
        if s_k.sum() < 100:
            continue
        sub = {n: 100 * (~mk[s_k]).mean() for n, mk in masks.items()}
        print(f"    Mr/Mf {ed[k]:6.3f}-{min(ed[k+1], y.max()):6.3f}  "
              f"n={int(s_k.sum()):6d}  outside={100 * (~ok[s_k]).mean():6.2f}%  "
              + "  ".join(f"{n}={v:5.2f}" for n, v in sub.items()))


if __name__ == "__main__":
    main()
