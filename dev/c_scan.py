"""Does the copy NLL actually have a minimum in the k^4 spin-2 coefficient?

`centroid.py train` wandered for 16000 steps without the NLL trending (77-133,
no slope) and left the layer at a spin-2 response ratio of -4.4.  Two
possibilities: the loop is mistuned, or the likelihood cannot see this
coefficient over the noise of the ~1% off-chart / poison tail.  This scans a
CONSTANT c on a large fixed batch and reports, side by side:

  * the copy NLL, on the same masked objective `centroid.train` optimises;
  * the spin-2 response ratio, the moment-matching target `check` prints.

If the NLL minimum sits where the response ratio is 1, the loop just needs a
smaller step.  If it is flat or somewhere else, NLL is the wrong objective for
this coefficient and it has to be fitted against the copy moments directly.

Usage: python dev/c_scan.py [flow] [copies]
"""
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bulk                                              # noqa: E402
import centroid as C                                     # noqa: E402
from models.bijections import in_domain, safe_point       # noqa: E402
from models.centroid import _C_MAX, dm_dsigma             # noqa: E402

FLOW = sys.argv[1] if len(sys.argv) > 1 else "flows/centroid_v3_s2init.eqx"
COPIES = (sys.argv[2] if len(sys.argv) > 2 else
          "../bfd_cnf_imsims/data/copies_bulgedisc_v3.fits")
N_NLL = 65536      # copies for the NLL, ~64x a training batch
N_RESP = 20000     # galaxies for the response ratio


def with_c(flow, c2=1.0, c4=1.0):
    """The flow with both bracket coefficients pinned to constants.

    Zeroing the head WEIGHT as well as setting its bias is what makes `c` a
    constant.  Zeroing only the bias silently leaves whatever the head weight
    has learned in place, so the scan's own x axis is wrong.
    """
    d = np.array([c2, c4]) - 1.0
    assert np.all(np.abs(d) < _C_MAX), (c2, c4)
    b = d / np.sqrt(1.0 - (d / _C_MAX) ** 2)
    head = lambda f: C._centroid_layer(f).coeff.layers[-1]
    flow = eqx.tree_at(lambda f: head(f).weight, flow,
                       jnp.zeros_like(head(flow).weight))
    return eqx.tree_at(lambda f: head(f).bias, flow,
                       jnp.asarray(b, dtype=jnp.float32))


def main():
    copies, galaxies = C.load_copies(COPIES)
    sigma_x = galaxies["cov_odd"][0]
    m_train = np.asarray(galaxies["moments"], dtype=np.float64)
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW, flow)

    # One fixed batch of copies, drawn exactly as `train` draws them, at g = 0
    # so the frozen shear layer contributes nothing but a constant.
    sampler = C.CopySampler(copies, sigma_x)
    x, _, _, _ = sampler.draw(np.random.default_rng(0), N_NLL)
    x = jnp.asarray(x, dtype=jnp.float32)
    cond = C.condition(sigma_x, N_NLL)
    ok0 = in_domain(x)
    x = jnp.where(ok0[:, None], x, safe_point(x))
    print(f"{N_NLL} copies, {float(1 - ok0.mean()):.2%} off-chart before the layer")

    @eqx.filter_jit
    def nll(model):
        lp = model.log_prob(x, condition=cond)
        ok = jnp.isfinite(lp) & (lp > -1e4) & ok0
        return (-jnp.sum(jnp.where(ok, lp, 0.0)) / jnp.maximum(jnp.sum(ok), 1),
                1.0 - jnp.mean(ok))

    target, keep = C.weighted_copy_mean(copies, galaxies, sigma_x)
    m0 = galaxies["moments"][keep][:N_RESP]
    target = target[:N_RESP]
    e = lambda mm: (mm[:, 2] + 1j * mm[:, 3]) / mm[:, 1]
    e0 = e(m0)
    resp = lambda mm: ((e(mm) * np.conj(e0)).real.sum()
                       / (e0 * np.conj(e0)).real.sum() - 1.0)
    rc = resp(target)

    def row(c2, c4):
        f = with_c(flow, c2, c4)
        v, masked = nll(f)
        pred = np.asarray(jax.vmap(dm_dsigma, in_axes=(None, 0, None, None))(
            C._centroid_layer(f), jnp.asarray(m0, dtype=jnp.float32),
            jnp.asarray(sigma_x, dtype=jnp.float32), C._chart(f)))
        print(f"{c2:7.2f} {c4:7.2f} {float(v):12.4f} {float(masked):8.2%} "
              f"{np.mean(pred[:, 1] / m0[:, 1]):11.4e} "
              f"{resp(m0 + pred) / rc:11.4f}")
        return resp(m0 + pred) / rc

    hdr = (f"\n{'c_spin2':>7s} {'c_spin4':>7s} {'NLL':>12s} {'masked':>8s} "
           f"{'dMr/Mr':>11s} {'resp ratio':>11s}")
    print(hdr)
    for c in (-1.0, -0.5, 0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0):
        row(c, 1.0)
    print(hdr)
    a = [row(1.0, c4) for c4 in (-1.9, -1.0, 0.0, 1.0, 2.0, 3.0)]
    b = [row(c2, 1.0) for c2 in (0.9, 1.1)]
    lev4 = (a[-1] - a[0]) / 4.9
    lev2 = (b[1] - b[0]) / 0.2
    print(f"\nresponse-ratio leverage per unit coefficient: c_spin2 {lev2:+.4f}"
          f"   c_spin4 {lev4:+.4f}   ratio {abs(lev2 / lev4):.1f}x")
    print(f"\ncatalog response {rc:+.4e}, catalog dMr/Mr "
          f"{np.mean((target[:, 1] - m0[:, 1]) / m0[:, 1]):+.4e}")


if __name__ == "__main__":
    main()
