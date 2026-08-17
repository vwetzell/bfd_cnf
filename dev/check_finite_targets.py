"""Is the centroid control's +0.006 the estimator's own finite-TARGET-count bias?

ghat = -(sum_i r_i)^-1 (sum_i q_i) is a RATIO of two finite sums, so it is a
biased estimator of g at O(1/N) even when q and r are individually unbiased --
the same 1/N_template effect the BFD paper measures in its Fig. 4, but on the
target side, where nobody has looked.

Part A: verify the law and the jackknife estimator on an exactly-solvable toy.
Part B: apply the identical estimator to the saved control PQR.

ANSWER (2026-08-17): NO, and decisively.  On the deep centroid control the
jackknife O(1/N) bias is |b/N| < 3e-6 against the +0.006 to explain, and the
subsample curve is FLAT over 32x in N (+0.00066 at N = 20000, +0.00132 at
N = 625, where a 1/N bias would have grown 32-fold).  Same on the shallow
control.  So the last purely statistical term other than the draw count is
dead, and the pruning argument in the write-up applies: in a control whose
prior is exact by construction, only O(g^2), O(1/N), O(1/S) and numerics can
bias m1 at all -- and the first, second and fourth are now all measured small.
"""
import numpy as np


def m1_of(qp, rp, qm, rm, g=0.02):
    gp = -np.linalg.solve(rp.sum(0), qp.sum(0))
    gm = -np.linalg.solve(rm.sum(0), qm.sum(0))
    return (gp[0] - gm[0]) / (2 * g) - 1


def jackknife_bias(qp, rp, qm, rm, g):
    """O(1/N) bias of m1, from leave-one-out sums (closed form, no loop)."""
    n = len(qp)
    full = m1_of(qp, rp, qm, rm, g)
    loo = []
    for q, r in ((qp, rp), (qm, rm)):
        Q, R = q.sum(0), r.sum(0)
        gl = -np.linalg.solve(R[None] - r, (Q[None] - q)[..., None])[..., 0]
        loo.append(gl)
    m_loo = (loo[0][:, 0] - loo[1][:, 0]) / (2 * g) - 1
    return full, (n - 1) * (m_loo.mean() - full)


def subsample_curve(qp, rp, qm, rm, g, ks, seed=0):
    """m1 averaged over K disjoint blocks. Bias b/N scales to K*b/N."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(qp))
    out = []
    for k in ks:
        blocks = np.array_split(idx, k)
        v = [m1_of(qp[b], rp[b], qm[b], rm[b], g) for b in blocks]
        out.append((k, float(np.mean(v)), float(np.std(v) / np.sqrt(k))))
    return out


# ---------------------------------------------------------------- Part A: toy
def toy(n, g=0.02, seed=0, heavy=0.0):
    """Exactly-solvable stand-in for one BFD target.

    A target's datum is x ~ N(mu(g), s_i^2) with mu(g) = a_i * g: 'shear'
    shifts the mean, and a_i is the per-galaxy response.  Then
        log P = -(x - a g)^2 / 2 s^2
        q = dlogP/dg|_0 = a x / s^2
        r = d2logP/dg2   = -a^2 / s^2
    so q is exactly unbiased for the score and -r is the exact per-target
    information.  ghat = -(sum r)^-1 sum q is then the ML estimator, and any
    m1 it shows is PURELY the ratio of two finite sums.

    `heavy` mixes in a lognormal spread of s_i, which is what a heavy R tail
    looks like: it inflates Var(r)/mean(r)^2 without changing the model.
    """
    rng = np.random.default_rng(seed)
    a = np.ones(n)
    s = np.exp(heavy * rng.standard_normal(n))
    q, r = [], []
    for sign in (+1, -1):
        x = sign * g * a + s * rng.standard_normal(n)
        q.append((a * x / s ** 2)[:, None])
        r.append((-a ** 2 / s ** 2)[:, None, None])
    return q[0], r[0], q[1], r[1]


def part_a():
    print("=== Part A: toy model, ghat = -(sum r)^-1 sum q ===")
    print("  bias should be b/N with b independent of N, and predicted by")
    print("  b = Var(j)/mean(j)^2 - Cov(q,j)/(mean(q) mean(j)),  j = -r\n")
    for heavy in (0.0, 0.6, 1.0):
        print(f"  spread of per-target information: heavy={heavy}")
        for n in (500, 2000, 8000):
            # average many independent realisations to see the bias
            reps = max(200, int(4e6 / n))
            vals = [m1_of(*toy(n, seed=k, heavy=heavy)) for k in range(reps)]
            m, e = np.mean(vals), np.std(vals) / np.sqrt(reps)
            # analytic coefficient from one big realisation
            qp, rp, qm, rm = toy(200000, seed=999, heavy=heavy)
            j = -rp[:, 0, 0]
            qq = qp[:, 0]
            b = j.var() / j.mean() ** 2 - np.cov(qq, j)[0, 1] / (qq.mean() * j.mean())
            print(f"    N={n:6d}  m1 = {m:+.5f} +/- {e:.5f}   "
                  f"N*m1 = {n*m:+7.3f}   predicted b = {b:+7.3f}")
        print()


# -------------------------------------------------------------- Part B: real
def part_b(path, g=0.02, label=""):
    d = np.load(path)
    keep = d["keep"] if "keep" in d.files else np.ones(len(d["plus_q"]), bool)
    qp, rp = d["plus_q"][keep], d["plus_r"][keep]
    qm, rm = d["minus_q"][keep], d["minus_r"][keep]
    n = len(qp)
    full, b = jackknife_bias(qp, rp, qm, rm, g)
    print(f"=== {label or path}  (N = {n}) ===")
    print(f"  m1 (full sample)          = {full:+.5f}")
    print(f"  jackknife O(1/N) bias     = {b/n:+.5f}   (b = {b:+.1f})")
    print(f"  m1 - that bias            = {full - b/n:+.5f}")
    print("  subsample check: mean m1 over K disjoint blocks")
    for k, m, e in subsample_curve(qp, rp, qm, rm, g, [1, 2, 4, 8, 16, 32]):
        print(f"    K={k:3d}  N/K={n//k:6d}  m1 = {m:+.5f} +/- {e:.5f}"
              f"   predicted {full - b/n + k*b/n:+.5f}")
    print()


if __name__ == "__main__":
    part_a()
    part_b("dev/pqr_control_deep_centroid_20k.npz", label="centroid control, DEEP")
    part_b("dev/pqr_control_centroid_20k.npz", label="centroid control, shallow")
