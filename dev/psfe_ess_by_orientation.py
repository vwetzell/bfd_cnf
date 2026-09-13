"""Does the alpha=0.5 mixture-draws convolution's ESS degrade differently near
the Mr/Mf size ceiling depending on PSF orientation?

kernel_draws uses the FULL (anisotropic) C_M Cholesky factor, so the
noise-kernel component of the mixture proposal is itself elongated along the
PSF's own axis under an elliptical PSF. If that elongation interacts with the
(orientation-independent) Mr/Mf support ceiling, ESS near the edge should
differ between e1-configs, e2-configs, and psfe00 -- a candidate mechanism for
why the c-leak concentrates exactly where the m1 residual already does
(psfe-leak-localizes-to-size-edge memory).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import equinox as eqx
import fitsio
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

import bias as B
import bulk
import shear

FLOW_PATH = "flows/centroid_g2v3d_sigmaxblock_mc_lr3e4_s0.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
SAMPLES = 8192
ALPHA = 0.5
SEED = 7
N_PER_BIN = 400

CONFIGS = {"psfe00": (0.0, 0.0), "psfe1p05": (0.05, 0.0),
           "psfe2p05": (0.0, 0.05), "psfe1m05": (-0.05, 0.0)}
BINS = {"edge  (2.9-3.2)": (2.9, 3.2), "interior (2.2-2.6)": (2.2, 2.6)}


def build_flow():
    pop = "gauss2_v3d_psfe00"
    m_train = shear.load(f"{DATA_DIR}/{B.TRAIN_DATA[pop]}")[0]
    flow = bulk.build_flow(jr.key(0), m_train, shear=True, centroid=True)
    flow = eqx.tree_deserialise_leaves(FLOW_PATH, flow)
    return bulk.SupportedFlow(flow, eps=0.0, support=True)


def main():
    flow = build_flow()
    print(f"{'config':>10s} {'bin':>18s} {'n':>5s} {'ESS median':>11s} {'ESS p10':>8s} {'frac<10':>8s}")
    for tag in CONFIGS:
        pop = f"gauss2_v3d_{tag}"
        targets = f"{DATA_DIR}/{B.CATALOGS[pop]['zero']}.fits"
        d = fitsio.read(targets)
        m = np.asarray(d["moments"], dtype=np.float64)
        cov = B.load_cov(targets)
        sigma_x = np.asarray(d["cov_odd"], dtype=np.float64)
        r = m[:, 1] / m[:, 0]

        for bname, (lo, hi) in BINS.items():
            idx = np.flatnonzero((r > lo) & (r < hi))
            rng = np.random.default_rng(0)
            if len(idx) > N_PER_BIN:
                idx = rng.choice(idx, N_PER_BIN, replace=False)
            m_i = m[idx]
            sx_i = sigma_x[idx]

            draws, log_wt = B.mixture_draws(flow, m_i, cov, SAMPLES, ALPHA, SEED,
                                            sigma_x=sx_i)
            e = np.asarray(B.ess(flow, m_i, draws, log_wt, batch=64, sigma_x=sx_i))
            print(f"{tag:>10s} {bname:>18s} {len(idx):5d} {np.median(e):11.1f} "
                  f"{np.percentile(e,10):8.1f} {np.mean(e<10):8.3f}", flush=True)


if __name__ == "__main__":
    main()
