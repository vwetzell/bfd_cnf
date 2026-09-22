"""Pool all 9 psfe configs into ONE joint slope fit for dc1/d(psf_e1) and
dc2/d(psf_e2), using a single shared bootstrap resample across all 10 files
(psfe00 + the 9 perturbed configs) per draw.

Fitting each config's diff-from-baseline separately (as dev/psfe_paired_diff.py
does one pair at a time) throws away information: the 9 differences are NOT
independent draws -- they all share the same psfe00 baseline and, to the
extent their own galaxies overlap psfe00's, correlated per-galaxy noise. A
naive weighted least squares across the 9 independent point estimates
(ignoring that shared correlation) would UNDERSTATE the aggregate error.
The correct fix is a single joint bootstrap: resample the same set of common
galaxies once per draw, compute ALL 10 configs' bias() on that one resample,
form the 9 diffs, and fit the slope -- then the spread of FITTED SLOPES
across replicates is the properly correlated error, with no independence
assumption needed anywhere.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

import bias as B

SIZE, FLUX = (2.2, 3.2), (1500.0, 20000.0)
G = 0.02
N_BOOT = 400
FLOW_PATH = "flows/centroid_g2v3d_full2_jac.eqx"
DATA_DIR = "../bfd_cnf_imsims/data"
SEED = 0            # bias.py's --seed default; unused for any psfe config
N_TARGETS = 20000   # matches dev/bias_psfe_g2v3d.sh's --n-targets
WINDOW_DRAWS = 1 << 24

_terms_cache = {}


def compute_selection_terms(tag, flow_path=FLOW_PATH):
    """(ps, qs, rs) exactly as bias.py's own `main()` computes them for
    `--window-terms score --window-draws 16777216 --support --floor-eps 0
    --window-size 2.2 3.2 --window-flux 1500 20000 --flow <flow_path>
    --pop gauss2_v3d_<tag>` (see dev/bias_psfe_g2v3d.sh) -- there is no
    terms-cache file for this flow/window, so this reproduces the relevant
    slice of `main()` inline rather than reading a stale one.
    """
    if tag in _terms_cache:
        return _terms_cache[tag]

    pop = f"gauss2_v3d_{tag}"
    path = lambda v: f"{DATA_DIR}/{v}.fits"
    cat = B.CATALOGS[pop]

    img_noise = bool(
        B.fitsio.read_header(path(cat["zero"]), ext=1).get("IMGNOISE", False))
    use_centroid = img_noise  # no --centroid override in bias_psfe_g2v3d.sh

    train_data = f"{DATA_DIR}/{B.TRAIN_DATA[pop]}"
    m_train_full = B.shear.load(train_data)[0]
    slice90 = lambda arr: arr[:int(0.9 * len(arr))]
    m_train = m_train_full if use_centroid else slice90(m_train_full)

    # This function may run more than once per process (one call per tag),
    # and jax_enable_x64 is process-global -- force it off here so a flow
    # built/deserialised on a LATER call still matches its on-disk float32
    # checkpoint, exactly as it would on bias.py's first (and only) call.
    B.jax.config.update("jax_enable_x64", False)
    flow = B.bulk.build_flow(B.jr.key(SEED), m_train, shear=True,
                             centroid=use_centroid, flux_sas=None)
    flow = B.eqx.tree_deserialise_leaves(flow_path, flow)
    flow = B.bulk.SupportedFlow(flow, eps=0.0, broad_std=4.0, support=True)

    n = slice(0, N_TARGETS)
    rows = {k: B.fitsio.read(path(v))[n] for k, v in cat.items()}
    keep = np.ones(len(rows["zero"]), dtype=bool)
    if img_noise:
        for v in rows.values():
            keep &= ~v["badcenter"]
        rows = {k: v[keep] for k, v in rows.items()}
    sigma_x_all = np.asarray(rows["zero"]["cov_odd"], dtype=np.float64)

    cov = B.load_cov(path(cat["zero"]))
    density = B.build_support_density(np.asarray(m_train_full))

    # Same float64 promotion `main()` does before any --window-terms branch,
    # and BEFORE building sx1 -- otherwise jnp.asarray(..., dtype=float64)
    # silently truncates back to float32 with x64 still off.
    B.jax.config.update("jax_enable_x64", True)
    sx1 = (B.jnp.asarray(sigma_x_all[0], dtype=B.jnp.float64)
           if use_centroid else None)
    flow = B.jax.tree_util.tree_map(
        lambda x: (x.astype(B.jnp.float64) if B.eqx.is_inexact_array(x) else x),
        flow)

    zero2 = B.jnp.zeros(2)
    zs = flow.base_dist.sample(B.jr.key(SEED + 31),
                               (WINDOW_DRAWS,)).astype(B.jnp.float64)
    tr = B.eqx.filter_jit(B.jax.vmap(
        lambda z1: flow.bijection.transform(z1, B.condition(zero2, sx1))))
    m_draw = B.jnp.concatenate(
        [tr(zs[i:i + 16384]) for i in range(0, len(zs), 16384)])

    ps, qs, rs, qs_err = B.selection_terms_score(
        flow, m_draw, cov, SIZE, FLUX, sigma_x=sx1, density=density)

    result = (float(ps), qs, rs)
    _terms_cache[tag] = result
    return result

CONFIGS = {
    "psfe1p02": (0.02, 0.0), "psfe2p02": (0.0, 0.02), "psfe1m02": (-0.02, 0.0),
    "psfe1p05": (0.05, 0.0), "psfe2p05": (0.0, 0.05), "psfe1m05": (-0.05, 0.0),
    "psfe1p10": (0.10, 0.0), "psfe2p10": (0.0, 0.10), "psfe1m10": (-0.10, 0.0),
}


def load(tag):
    d = np.load(f"pqr/g2v3d_full2_jac_{tag}.npz")
    return d, compute_selection_terms(tag)


def pairwise_match(base_moments, other_moments):
    """base_idx -> other_idx, 1:1 nearest-neighbor, same rule as
    dev/psfe_paired_diff.py."""
    tree = cKDTree(base_moments)
    dist, j = tree.query(other_moments)
    self_dist, _ = cKDTree(other_moments).query(other_moments, k=2)
    scale = np.median(self_dist[:, 1])
    ok = dist < 0.5 * scale
    uniq, counts = np.unique(j[ok], return_counts=True)
    keep_base = set(uniq[counts == 1])
    return {int(j[k]): k for k in np.flatnonzero(ok) if int(j[k]) in keep_base}


def main():
    base, base_terms = load("psfe00")
    others = {tag: load(tag) for tag in CONFIGS}
    n_base = len(base["moments"])

    # Each config keeps its OWN pairwise match (~98-99% of psfe00's rows),
    # rather than forcing one index set shared by all 10 files at once --
    # that simultaneous-intersection approach only kept 2301/19937 rows
    # (~11.5%), because a ~1% pairwise miss rate compounds across 9
    # independent chances to miss. Pairwise keeps ~19700-19900 each.
    matches = {tag: pairwise_match(base["moments"], d["moments"])
               for tag, (d, _) in others.items()}
    print("pairwise match counts:", {t: len(m) for t, m in matches.items()})

    def obs_and_arrays(d, idx):
        sel = (B.window_mask(d["obs_plus"][idx], SIZE, FLUX),
               B.window_mask(d["obs_minus"][idx], SIZE, FLUX))
        n_out = ((~sel[0]).sum(), (~sel[1]).sum())
        return (d["plus_q"][idx], d["plus_r"][idx], d["minus_q"][idx], d["minus_r"][idx]), sel, n_out

    def biasf(arrs, sel, ns):
        return B.bias(*arrs, G, sel=sel, ns=ns)

    def wls_slope(xs, ys):
        xs, ys = np.asarray(xs), np.asarray(ys)
        return np.sum(xs * ys) / np.sum(xs * xs)

    x_c1 = {t: CONFIGS[t][0] for t in CONFIGS}
    x_c2 = {t: CONFIGS[t][1] for t in CONFIGS}

    def diffs_for(base_idx_full):
        """base_idx_full: an array of psfe00 row indices (may repeat, as in
        a bootstrap resample). For each config, keep only the entries that
        config's own pairwise map actually covers, and return (dc1, dc2).
        The baseline's own bias() is recomputed PER CONFIG on exactly the
        rows that config covers, so numerator and denominator of each diff
        use the same galaxy set (a config-specific baseline subset, not one
        shared cut) -- consistent with how psfe_paired_diff.py does it."""
        out = {}
        for tag, (d, terms) in others.items():
            m = matches[tag]
            base_ok = np.array([i for i in base_idx_full if i in m])
            other_ok = np.array([m[i] for i in base_ok])
            b_arrs, b_sel, b_nout = obs_and_arrays(base, base_ok)
            b_ns = ((float(b_nout[0]), *base_terms), (float(b_nout[1]), *base_terms))
            arrs, sel, nout = obs_and_arrays(d, other_ok)
            ns = ((float(nout[0]), *terms), (float(nout[1]), *terms))
            b_base_i = biasf(b_arrs, b_sel, b_ns)
            b = biasf(arrs, sel, ns)
            out[tag] = (b[1] - b_base_i[1], b[2] - b_base_i[2])
        return out

    diffs0 = diffs_for(np.arange(n_base))
    y_c1_0 = [diffs0[t][0] for t in CONFIGS]
    y_c2_0 = [diffs0[t][1] for t in CONFIGS]
    slope_c1_0 = wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                           [y for t, y in zip(CONFIGS, y_c1_0) if x_c1[t] != 0])
    slope_c2_0 = wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                           [y for t, y in zip(CONFIGS, y_c2_0) if x_c2[t] != 0])
    print(f"\npoint estimate: dc1/d(psf_e1) = {slope_c1_0:+.5f}   "
          f"dc2/d(psf_e2) = {slope_c2_0:+.5f}")

    rng = np.random.default_rng(0)
    slope_c1_boot, slope_c2_boot = [], []
    for _ in range(N_BOOT):
        idx = rng.choice(n_base, n_base)
        diffs = diffs_for(idx)
        y_c1 = [diffs[t][0] for t in CONFIGS]
        y_c2 = [diffs[t][1] for t in CONFIGS]
        slope_c1_boot.append(wls_slope([x_c1[t] for t in CONFIGS if x_c1[t] != 0],
                                       [y for t, y in zip(CONFIGS, y_c1) if x_c1[t] != 0]))
        slope_c2_boot.append(wls_slope([x_c2[t] for t in CONFIGS if x_c2[t] != 0],
                                       [y for t, y in zip(CONFIGS, y_c2) if x_c2[t] != 0]))

    sd_c1 = np.std(slope_c1_boot)
    sd_c2 = np.std(slope_c2_boot)
    print(f"joint-bootstrap error: dc1/d(psf_e1) = {slope_c1_0:+.5f} +/- {sd_c1:.5f} "
          f"({slope_c1_0/sd_c1:+.2f} sigma)")
    print(f"joint-bootstrap error: dc2/d(psf_e2) = {slope_c2_0:+.5f} +/- {sd_c2:.5f} "
          f"({slope_c2_0/sd_c2:+.2f} sigma)")


if __name__ == "__main__":
    main()
