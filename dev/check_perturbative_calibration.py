"""check_perturbative_calibration.py
====================================
Calibration check for a ShearPerturbative-trained prior flow: pushes real
(standardised) galaxy moments through the flow's data->base map at a FIXED
condition and checks the result is a standard normal (mean~0, cov~I) --
first at g=0, then at a fixed nonzero g.

The held-out sample is drawn proportional to the SAME nda*detj batch-sampling
proposal make_nll_loss trains on (data["weights"]; use_nda_weight=True is the
config default), not a uniform draw over template rows -- the trained density
corresponds to that reweighted population, not the raw uniform row
distribution, so evaluating against a uniform draw would show a spurious
"miscalibration" that's really just a population mismatch in the CHECK, not a
defect in the flow.

For the nonzero-g case, the held-out moments are first physically SHEARED to
that g using the exact per-template analytic derivatives (models.flows.shear,
the same 2nd-order Taylor construction make_nll_loss uses to build its own
training targets -- see that module), THEN standardised and decoded under
condition=g. This is the honest test: "is P(M|g) a properly calibrated
density for data that's actually at shear g", not "what does the g=0 data
look like if force-decoded under the wrong condition".

This is the ordinary normalizing-flow sanity check (the flow's inverse
direction, per flowjax's Transformed: "we take ... the inverse bijection for
use in density evaluation" -- i.e. exactly what log_prob uses internally),
just done on physical data instead of asking for a single log_prob number, so
we can see the actual moment/covariance of the pushed-forward points instead
of one scalar.

Usage:
    cd bfd_cnf
    BFD_TRAIN_FITS=<same fits used to train> PYTHONPATH=. python dev/check_perturbative_calibration.py \\
        --prior flows/prior_flow_perturbative_noshift_noiseless.eqx \\
        --g 0.02 0.01
"""
from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from paramax import non_trainable
from flowjax.distributions import MultivariateNormal

from bfd_cnf.config import TRAIN_FITS_PATH
from bfd_cnf.data import load_training_dataset
from bfd_cnf.models.bijections import load_stats
from bfd_cnf.models.flows import build_flows, shear


def _report(name: str, z: np.ndarray) -> None:
    n, d = z.shape
    mean = z.mean(0)
    cov = np.cov(z.T)
    mean_err = 1.0 / np.sqrt(n)  # expected std of a mean-0 component's sample mean
    print(f"\n-- {name} (n={n}) --")
    print(f"  mean: {np.array2string(mean, precision=4)}  "
          f"(expect ~0, sampling noise ~{mean_err:.4f})")
    print(f"  cov diag: {np.array2string(np.diag(cov), precision=4)}  (expect ~1)")
    off = cov[np.triu_indices(d, k=1)]
    print(f"  cov off-diag max|.|: {np.max(np.abs(off)):.4f}  (expect ~0)")
    # per-dim z-score of the sample mean and (var-1), assuming ~unit variance
    mean_z = mean / mean_err
    var_err = np.sqrt(2.0 / n)
    var_z = (np.diag(cov) - 1.0) / var_err
    print(f"  mean z-scores:     {np.array2string(mean_z, precision=2)}")
    print(f"  (var-1) z-scores:  {np.array2string(var_z, precision=2)}")
    bad = np.any(np.abs(mean_z) > 5) or np.any(np.abs(var_z) > 5)
    print(f"  -> {'FAIL (>5 sigma)' if bad else 'OK (within 5 sigma)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="Path to the trained prior .eqx checkpoint.")
    ap.add_argument("--g", type=float, nargs=2, default=[0.02, 0.01],
                     help="Fixed nonzero (g1, g2) to test in addition to g=0.")
    ap.add_argument("--n", type=int, default=200_000, help="Number of held-out moments to check.")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--shear-nn-width", type=int, default=32,
                     help="Must match the checkpoint's --shear-nn-width at train time.")
    ap.add_argument("--shear-nn-depth", type=int, default=4,
                     help="Must match the checkpoint's --shear-nn-depth at train time.")
    args = ap.parse_args()

    print(f"Training set (for a held-out-style sample): {TRAIN_FITS_PATH}")
    raw2standard = load_stats(args.prior)
    data = load_training_dataset(key=jr.key(args.seed), subsample=1)
    moments_jnp = data["moments_jnp"]
    dm_dg_jnp = data["dm_dg_jnp"]
    d2m_dg2_jnp = data["d2m_dg2_jnp"]
    weights = np.asarray(data["weights"])  # nda*detj batch-sampling proposal, same as training
    N = moments_jnp.shape[0]

    rng = np.random.default_rng(args.seed + 1)
    p = weights / weights.sum()
    idx = rng.choice(N, size=min(args.n, N), replace=True, p=p)  # matches make_nll_loss's own draw
    m = jnp.asarray(np.asarray(moments_jnp)[idx])          # (n, 4) [Mf,Mr,M1,M2]
    dg = jnp.asarray(np.asarray(dm_dg_jnp)[idx])[:, :4]     # (n, 4, 2)
    d2g = jnp.asarray(np.asarray(d2m_dg2_jnp)[idx])[:, :4]  # (n, 4, 2, 2)

    def standardize_at(g1, g2):
        g_arr = jnp.array([[[g1, g2]]])  # (1, 1, 2), matches shear()'s expected shape
        m_g = shear(m, g_arr, dg, d2g)[:, 0, :]  # (n, 4), exact same Taylor construction as training
        x_std, _ = jax.vmap(raw2standard.transform_and_log_det)(m_g)
        return np.asarray(x_std)

    x_std = standardize_at(0.0, 0.0)

    prior_flow, _q = build_flows(
        jr.key(0), latent_dim=4, cond_dim=16, raw2standard=raw2standard,
        sigmax_layer_kind="none", shear_layer_kind="perturbative",
        prior_last_nn_width=args.shear_nn_width, prior_last_nn_depth=args.shear_nn_depth,
    )
    prior_flow = eqx.tree_deserialise_leaves(args.prior, prior_flow)

    def to_base(x, cond):
        z, _ = prior_flow.bijection.inverse_and_log_det(x, condition=cond)
        return z

    cond0 = jnp.zeros(5)
    z0 = np.asarray(jax.vmap(lambda x: to_base(x, cond0))(jnp.asarray(x_std)))
    _report("g = (0, 0)", z0)

    g1, g2 = args.g
    x_std_g = standardize_at(g1, g2)  # data actually sheared to g, not g=0 data reused
    condg = jnp.array([g1, g2, 0.0, 0.0, 0.0])
    zg = np.asarray(jax.vmap(lambda x: to_base(x, condg))(jnp.asarray(x_std_g)))
    _report(f"g = ({g1}, {g2})", zg)


if __name__ == "__main__":
    main()
