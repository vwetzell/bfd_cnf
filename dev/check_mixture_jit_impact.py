"""Does the jit hoist's float32 drift reach Q and R, or does it sit in the tail?

`check_mixture_jit_equivalence.py` compares log-weights draw by draw, which is
the wrong yardstick on its own: `log_wt = log_k - log_q`, and where a draw is
far out `log_q` is dominated by a `log_f` of order -1e4, so a float32 relative
difference of 1e-5 between the jitted and eager flow shows up as ~0.1 nats --
on a draw whose weight is e^-1e4 and which therefore contributes nothing.

So this asks the question that matters: after the logsumexp, do Q and R move?
Those are what enter the eq. (45)-(46) sums.
"""

import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import fitsio
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, ".")

import bias                                          # noqa: E402
import bulk                                          # noqa: E402
import shear                                         # noqa: E402
from dev.check_mixture_jit_equivalence import mixture_draws_reference  # noqa: E402

DATA = "../bfd_cnf_imsims/data"
SAMPLES, SEED = 8192, 12345


def main():
    m_train = shear.load(f"{DATA}/moments.fits")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves("flows/centroid_deep.eqx", flow)

    rows = fitsio.read(f"{DATA}/targets_deep_g0_200k.fits")[:32]
    m = np.asarray(rows["moments"], dtype=np.float64)
    sigma_x = np.asarray(rows["cov_odd"], dtype=np.float64)
    cov = bias.load_cov(f"{DATA}/targets_deep_g0_200k.fits")

    d_ref, w_ref = mixture_draws_reference(flow, m, cov, SAMPLES, 0.5, SEED,
                                           sigma_x=sigma_x)
    d_new, w_new = bias.mixture_draws(flow, m, cov, SAMPLES, 0.5, SEED,
                                      sigma_x=sigma_x)
    # Not bitwise: jitting fuses the sampling direction's float32 arithmetic
    # differently from eager op-by-op execution, so the draws agree to f32
    # rounding rather than exactly.  That is a reordering of the same
    # computation on the same RNG stream, not a different proposal.
    rel = np.abs(d_ref - d_new).max() / np.abs(d_ref).max()
    print(f"  relative |d draws| = {rel:.3e}  "
          f"(bitwise equal: {np.array_equal(d_ref, d_new)})")
    assert rel < 1e-5, rel

    # Where does the log-weight difference live?  Rank each draw by how far its
    # weight is below its target's best draw: within ~30 nats it can matter,
    # below that e^-30 makes it noise.
    lead = w_ref - w_ref.max(axis=1, keepdims=True)
    diff = np.abs(w_ref - w_new)
    for cut in (0.0, 5.0, 15.0, 30.0, 100.0, np.inf):
        sel = lead >= -cut if np.isfinite(cut) else np.ones_like(lead, bool)
        print(f"  draws within {cut:>5} nats of the best: {sel.sum():8d}"
              f"   max |dlog_wt| = {diff[sel].max():.3e}")

    # The number that actually matters: Q and R after the logsumexp.
    flow_g, layer = bias.split_centroid(flow)
    out = {}
    for name, (d, w) in (("ref", (d_ref, w_ref)), ("new", (d_new, w_new))):
        dd, ld = bias.centroid_transform(layer, d, sigma_x, to_host=False)
        ww = jnp.asarray(w, dtype=jnp.float32) + ld
        mz = bias.centroid_transform(layer, m, sigma_x, to_host=False)[0]
        # batch=2: the Hessian is forward-over-reverse, so targets x draws sets
        # the footprint and 32 x 8192 at once asks for 13 GB.
        out[name] = bias.pqr(flow_g, np.asarray(mz), np.asarray(dd),
                             np.asarray(ww), batch=2, sigma_x=sigma_x)

    (q_r, r_r), (q_n, r_n) = out["ref"], out["new"]
    dq = np.abs(q_r - q_n).max() / np.abs(q_r).max()
    dr = np.abs(r_r - r_n).max() / np.abs(r_r).max()
    print(f"\n  relative |dQ| = {dq:.3e}\n  relative |dR| = {dr:.3e}")
    # pqr_streamed's own documented chunk-merge error is 3.7e-3 on Q and 4.2e-3
    # on R, so anything well under that is lost in machinery already accepted.
    assert dq < 1e-3 and dr < 1e-3, (dq, dr)
    print("\nok -- the drift stays in the tail and does not reach Q or R")


if __name__ == "__main__":
    main()
