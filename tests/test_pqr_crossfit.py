"""`bias._merge_finish`'s jackknife removes the estimator's leading O(1/S) bias.

`pqr_streamed` forms `Q = B/A` and `R = C/A - (B/A)(B/A)^T` from Monte-Carlo
sums.  That carries two O(1/S) biases which partly CANCEL: the squared MC
estimate `(B/A)(B/A)^T` makes R too LARGE, while the self-normalised ratios pull
the other way and are the same order.  Fixing only the square leaves the other
uncancelled, which is not a reliable gain -- it can come out better or worse
depending on the configuration.  That is the whole reason the correction is a
jackknife over chunks rather than a U-statistic on the square alone.

The statistical tests run on a case with a closed form: latent `m ~ N(g, 1)`,
noise `M = m + N(0, s^2)`, so `P(M|g) = N(M; g, 1+s^2)` exactly and
`j = -d2 logP/dg2 = 1/(1+s^2)`, independent of M.  The kernel is its own
proposal (alpha = 1, log_wt = 0), so a chunk of draws `x = M + s z` gives
`A = sum p(x|0)`, `B = sum x p`, `C = sum (x^2-1) p`.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")

from bias import _merge_chunk, _merge_finish, _merge_init, _shares  # noqa: E402

S_NOISE = 0.8
J_TRUE = 1.0 / (1.0 + S_NOISE ** 2)


def _feed(chunks):
    n = len(chunks[0][1])
    st = _merge_init(n)
    for la, df, coa in chunks:
        st = _merge_chunk(st, la, df, coa, np.ones(n, bool))
    return st


def _rand_chunks(rng, n, k):
    return [(np.log(rng.uniform(0.5, 2.0, n)),
             rng.normal(size=(n, 2)),
             rng.normal(size=(n, 2, 2))) for _ in range(k)]


# --------------------------------------------------------------- the algebra
def test_plain_is_the_A_weighted_ratio():
    """With the jackknife off, Q and R are exactly B_tot/A_tot and
    C_tot/A_tot - (B/A)(B/A)^T."""
    rng = np.random.default_rng(0)
    ch = _rand_chunks(rng, 7, 5)
    st = _feed(ch)
    A = np.stack([np.exp(la) for la, _, _ in ch])
    w = A / A.sum(0)
    q, r, n_fb = _merge_finish(st, jackknife=False)
    assert n_fb == 0
    assert np.allclose(q, np.einsum("kn,kna->na", w,
                                    np.stack([d for _, d, _ in ch])))
    chat = np.einsum("kn,knab->nab", w, np.stack([c for _, _, c in ch]))
    assert np.allclose(r, chat - np.einsum("na,nb->nab", q, q))


def test_jackknife_matches_the_explicit_delete_one_formula():
    rng = np.random.default_rng(1)
    ch = _rand_chunks(rng, 6, 4)
    st = _feed(ch)
    A = np.stack([np.exp(la) for la, _, _ in ch])
    k = 4

    def full(idx):
        Ai = A[idx]
        w = Ai / Ai.sum(0)
        b = np.einsum("kn,kna->na", w, np.stack([ch[i][1] for i in idx]))
        c = np.einsum("kn,knab->nab", w, np.stack([ch[i][2] for i in idx]))
        return b, c - np.einsum("na,nb->nab", b, b)

    b_all, r_all = full(range(k))
    loo = [full([i for i in range(k) if i != e]) for e in range(k)]
    q_ref = k * b_all - (k - 1) * np.mean([x[0] for x in loo], axis=0)
    r_ref = k * r_all - (k - 1) * np.mean([x[1] for x in loo], axis=0)

    q, r, n_fb = _merge_finish(st, jackknife=True)
    assert n_fb == 0
    assert np.allclose(q, q_ref)
    assert np.allclose(r, r_ref)


def test_single_chunk_falls_back():
    rng = np.random.default_rng(2)
    st = _feed(_rand_chunks(rng, 5, 1))
    q, r, n_fb = _merge_finish(st, jackknife=True)
    q0, r0, _ = _merge_finish(st, jackknife=False)
    assert n_fb == 5
    assert np.allclose(r, r0) and np.allclose(q, q0)


def test_one_dominant_chunk_falls_back():
    """A chunk holding ~all the weight leaves 1 - a_e ~ 0, so keep the plain
    estimator for that target rather than divide by it."""
    rng = np.random.default_rng(3)
    ch = _rand_chunks(rng, 4, 3)
    ch[0][0][:] = 60.0          # chunk 0 dominates every target
    st = _feed(ch)
    _, r, n_fb = _merge_finish(st, jackknife=True)
    _, r0, _ = _merge_finish(st, jackknife=False)
    assert n_fb == 4
    assert np.allclose(r, r0)


def test_dead_chunks_are_dropped_not_counted():
    """A chunk whose draws all underflow (la = -inf) contributes nothing."""
    rng = np.random.default_rng(4)
    ch = _rand_chunks(rng, 5, 3)
    st = _feed(ch)
    dead = (np.full(5, -np.inf), np.zeros((5, 2)), np.zeros((5, 2, 2)))
    st2 = _merge_chunk(_feed(ch), *dead, np.zeros(5, bool))
    a1, ok1 = _shares(np.stack(st["la"]))
    a2, ok2 = _shares(np.stack(st2["la"]))
    assert ok1.all() and ok2.all()
    assert np.allclose(a1, a2[:3])
    assert np.allclose(a2[3], 0.0)


# ------------------------------------------------------------ the statistics
def _gauss_chunks(rng, M, k, per_chunk):
    out = []
    for _ in range(k):
        x = M[:, None] + S_NOISE * rng.standard_normal((len(M), per_chunk))
        lp = -0.5 * x ** 2
        mx = lp.max(1, keepdims=True)
        p = np.exp(lp - mx)
        A = p.sum(1)
        b = (x * p).sum(1) / A
        c = ((x ** 2 - 1.0) * p).sum(1) / A
        z = np.zeros_like(b)
        out.append((np.log(A) + mx[:, 0],
                    np.stack([b, z], axis=-1),
                    np.stack([np.stack([c, z], -1), np.zeros((len(M), 2))], axis=-1)))
    return out


@pytest.mark.parametrize("M0", [0.0, 1.5])
def test_jackknife_shrinks_the_R_bias(M0):
    rng = np.random.default_rng(11)
    reps, k, per_chunk = 40000, 8, 40
    st = _feed(_gauss_chunks(rng, np.full(reps, M0), k, per_chunk))
    _, r_p, _ = _merge_finish(st, jackknife=False)
    _, r_j, n_fb = _merge_finish(st, jackknife=True)

    bias_plain = -r_p[:, 0, 0].mean() - J_TRUE
    bias_jack = -r_j[:, 0, 0].mean() - J_TRUE
    assert n_fb < 0.02 * reps
    assert abs(bias_jack) < 0.5 * abs(bias_plain), (bias_jack, bias_plain)


def test_jackknife_beats_plain_at_every_draw_count():
    rng = np.random.default_rng(12)
    for per_chunk in (20, 80):
        st = _feed(_gauss_chunks(rng, np.full(40000, 1.0), 8, per_chunk))
        _, r_p, _ = _merge_finish(st, jackknife=False)
        _, r_j, _ = _merge_finish(st, jackknife=True)
        bp = -r_p[:, 0, 0].mean() - J_TRUE
        bj = -r_j[:, 0, 0].mean() - J_TRUE
        assert abs(bj) < abs(bp), (per_chunk, bj, bp)


def test_the_two_O_1_over_S_biases_are_opposite_and_comparable():
    """Pins the reason the correction is a jackknife and not a U-statistic on
    (B/A)^2 alone.

    At M = 0 the true q is 0, so `E[(B/A)^2]` is pure Monte-Carlo variance and
    the cross-fit kills it.  But `C/A` carries its own O(1/S) ratio bias of the
    OPPOSITE sign and comparable size, so removing only the square leaves an
    uncancelled term just as big -- sometimes better than the plain estimator,
    sometimes worse, depending on the configuration.  That is why the shipped
    correction debiases the estimator as a whole.
    """
    rng = np.random.default_rng(13)
    reps, k, per_chunk = 40000, 8, 48
    st = _feed(_gauss_chunks(rng, np.zeros(reps), k, per_chunk))
    a, _ = _shares(np.stack(st["la"]))
    b = np.stack(st["b"])[:, :, 0]
    c = np.stack(st["c"])[:, :, 0, 0]
    bhat = (a * b).sum(0)
    chat = (a * c).sum(0)

    s2 = (a ** 2).sum(0)
    from_square = (bhat ** 2).mean()
    from_ratio = (-chat).mean() - J_TRUE
    qq_cross = ((bhat ** 2 - (a ** 2 * b ** 2).sum(0)) / (1.0 - s2)).mean()

    # the square is biased; the cross-fit removes essentially all of it
    assert from_square > 10 * abs(qq_cross), (from_square, qq_cross)
    # the ratio bias is of the OPPOSITE sign and the SAME order -- so the
    # cancellation is real and fixing one term alone is not a reliable gain
    assert from_ratio < 0.0 < from_square, (from_ratio, from_square)
    assert 0.25 < abs(from_ratio) / from_square < 4.0, (from_ratio, from_square)
