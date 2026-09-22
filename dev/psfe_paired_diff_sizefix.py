"""Paired difference of the windowed-corrected (m1, c1, c2) between two psfe
saved-PQR runs, exploiting that they share the same seed-1 galaxies and pixel
noise (the render's own antithetic pairing) -- most of the galaxy-sample
scatter that dominates each run's own bootstrap error is common-mode and
cancels in the difference.

`bias.py --compare` does exactly this, but requires the two files to have the
IDENTICAL number of rows in the SAME order, which our psfe grid does not
have: each config's own recentring-convergence and |Q|/|R|-outlier drops
removed a slightly different handful of targets (order-preserving, but not
count-preserving -- see bias.py's `keep`/`sane` masks). So rows are realigned
here by nearest-neighbor match on the saved `moments` field (the g=0 arm's
raw moments) -- a per-galaxy fingerprint that differs between two configs
only through the tiny PSF-ellipticity difference in the render, since the
ADDED PIXEL NOISE is seeded by (seed, row index) alone and so is bit-identical
across every psfe config.

Usage: python dev/psfe_paired_diff.py <tag_a> <tag_b>
  e.g. python dev/psfe_paired_diff.py psfe00 psfe1p02
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree

import bias as B

SIZE, FLUX = (2.2, 3.2), (1500.0, 20000.0)
G = 0.02
FLOW_PATH = "flows/centroid_g2v3d_sizefix.eqx"
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


def load(tag):
    d = np.load(f"pqr/g2v3d_sizefix_{tag}.npz")
    return d, compute_selection_terms(tag)


def align(a, b):
    """Nearest-neighbor match on the 5D `moments` fingerprint; keeps only
    pairs whose match distance is far below the typical inter-galaxy gap, so
    a collision (two different galaxies matched together) can't sneak in."""
    tree = cKDTree(a["moments"])
    dist, j = tree.query(b["moments"])
    # typical nearest-neighbor gap AMONG b's own rows, as the collision scale
    self_dist, _ = cKDTree(b["moments"]).query(b["moments"], k=2)
    scale = np.median(self_dist[:, 1])
    ok = dist < 0.5 * scale
    i = j[ok]
    bidx = np.flatnonzero(ok)
    # a nearest-neighbor match can be many-to-one; keep only 1:1 pairs
    uniq, counts = np.unique(i, return_counts=True)
    keep_i = set(uniq[counts == 1])
    sel = np.array([k for k, ii in enumerate(i) if ii in keep_i])
    return i[sel], bidx[sel]


def stats(d, ns):
    obs_p, obs_m = d["obs_plus"], d["obs_minus"]
    sel_p, sel_m = B.window_mask(obs_p, SIZE, FLUX), B.window_mask(obs_m, SIZE, FLUX)
    n_out = ((~sel_p).sum(), (~sel_m).sum())
    return sel_p, sel_m, n_out


def main():
    tag_a, tag_b = sys.argv[1], sys.argv[2]
    a, terms_a = load(tag_a)
    b, terms_b = load(tag_b)

    ia, ib = align(a, b)
    print(f"{tag_a} vs {tag_b}: {len(a['moments'])} / {len(b['moments'])} rows, "
          f"{len(ia)} matched 1:1")

    sel_pa_full, sel_ma_full, n_out_a = stats(a, terms_a)
    sel_pb_full, sel_mb_full, n_out_b = stats(b, terms_b)

    def arrs(d, idx):
        return d["plus_q"][idx], d["plus_r"][idx], d["minus_q"][idx], d["minus_r"][idx]

    qp_a, rp_a, qm_a, rm_a = arrs(a, ia)
    qp_b, rp_b, qm_b, rm_b = arrs(b, ib)
    sel_a = (B.window_mask(a["obs_plus"][ia], SIZE, FLUX),
             B.window_mask(a["obs_minus"][ia], SIZE, FLUX))
    sel_b = (B.window_mask(b["obs_plus"][ib], SIZE, FLUX),
             B.window_mask(b["obs_minus"][ib], SIZE, FLUX))
    ns_a = ((float(n_out_a[0]), *terms_a), (float(n_out_a[1]), *terms_a))
    ns_b = ((float(n_out_b[0]), *terms_b), (float(n_out_b[1]), *terms_b))

    def biasf(qp, rp, qm, rm, sel, ns):
        return B.bias(qp, rp, qm, rm, G, sel=sel, ns=ns)

    ba = biasf(qp_a, rp_a, qm_a, rm_a, sel_a, ns_a)
    bb = biasf(qp_b, rp_b, qm_b, rm_b, sel_b, ns_b)

    rng = np.random.default_rng(0)
    n_boot = 400
    n_common = len(ia)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n_common, n_common)
        va = biasf(qp_a[idx], rp_a[idx], qm_a[idx], rm_a[idx],
                   (sel_a[0][idx], sel_a[1][idx]), ns_a)
        vb = biasf(qp_b[idx], rp_b[idx], qm_b[idx], rm_b[idx],
                   (sel_b[0][idx], sel_b[1][idx]), ns_b)
        diffs.append(np.subtract(vb, va))
    sd = np.std(np.array(diffs), axis=0)

    names = ["m1", "c1", "c2"]
    print(f"\n{'':>6s} {tag_a:>12s} {tag_b:>12s} {'diff (paired)':>22s}")
    for k, name in enumerate(names):
        print(f"  {name:>4s} {ba[k]:>+12.5f} {bb[k]:>+12.5f} "
              f"{bb[k]-ba[k]:>+10.5f} +/- {sd[k]:.5f}  "
              f"({(bb[k]-ba[k])/sd[k]:+.2f} sigma)")


if __name__ == "__main__":
    main()
