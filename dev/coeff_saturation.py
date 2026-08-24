"""How much of the population pins each shear coefficient against _COEFF_MAX?

`models.shear._Coeffs` bounds its fourteen outputs by x/sqrt(1+(x/C)^2).  A
coefficient sitting at the bound is one the network could not represent, so the
fraction of the population above ~0.9C is a direct read on which term of the
response is capacity-limited -- the diagnosis HANDOFF.md's 2026-08-22 section
reaches for `rho` with.

    python -m dev.coeff_saturation --flow flows/shear_gauss2.eqx \
        --data ../bfd_cnf_imsims/data/gauss2_g0_1M.fits
"""
import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import bulk
import shear as shear_mod
from models.shear import _COEFF_MAX, _invariants

NAMES = ([f"{c}_{m}" for m in ("Mf", "Mr", "Mc") for c in ("a", "b", "c")]
         + ["A", "B", "mu", "nu", "rho"])


def saturation(flow, M, n=200000):
    """Per-coefficient fraction of galaxies above 0.9 * _COEFF_MAX, and |median|."""
    chart, layer = shear_mod._chart(flow), shear_mod._shear_layer(flow)
    z = jax.vmap(chart.transform)(jnp.asarray(M[:n]))

    @jax.vmap
    def coeffs(z_i):
        f, a, b, q, _ = _invariants(z_i)
        return layer.coeffs(f, a, b, q)

    c = np.asarray(coeffs(z))
    return c, (np.abs(c) > 0.9 * _COEFF_MAX).mean(axis=0), np.median(np.abs(c), axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", default="flows/shear_gauss2.eqx")
    p.add_argument("--data", default="../bfd_cnf_imsims/data/gauss2_g0_1M.fits")
    p.add_argument("--n", type=int, default=200000)
    a = p.parse_args()

    M = shear_mod.load(a.data)[0]
    flow = eqx.tree_deserialise_leaves(
        a.flow, bulk.build_flow(jr.key(0), M[:int(0.9 * len(M))], shear=True))
    c, frac, med = saturation(flow, M, a.n)

    print(f"{a.flow}: {min(a.n, len(M))} galaxies, _COEFF_MAX = {_COEFF_MAX}")
    print(f"{'coeff':>6s}{'|median|':>12s}{'frac at bound':>16s}")
    for name, m, f in zip(NAMES, med, frac):
        print(f"{name:>6s}{m:12.3f}{f:16.1%}")

    # Same split HANDOFF.md used: saturation vs proximity to the Mr/Mf ceiling.
    r = np.asarray(M[:len(c), 1] / M[:len(c), 0])
    edges = np.quantile(r, np.linspace(0, 1, 6))
    hot = [i for i, f in enumerate(frac) if f > 0.2]
    if hot:
        print("\nfraction at bound by Mr/Mf quintile")
        print(f"{'coeff':>6s}" + "".join(f"{lo:6.2f}-{hi:<5.2f}"
                                         for lo, hi in zip(edges[:-1], edges[1:])))
        for i in hot:
            bins = [( np.abs(c[(r >= lo) & (r < hi), i]) > 0.9 * _COEFF_MAX).mean()
                    for lo, hi in zip(edges[:-1], edges[1:])]
            print(f"{NAMES[i]:>6s}" + "".join(f"{b:11.1%}" for b in bins))


if __name__ == "__main__":
    main()
