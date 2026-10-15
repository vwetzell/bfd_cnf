"""Did the trained shear layer learn the sheared-template-noise term?  Prediction: layer dm/dg(T) - D0(T)
= beta * t(T), t = -G_g grad log p_bulk(T) (Tweedie), beta = 1 if the layer fits E[D_T|T] exactly.
Control: the noisy supervision itself, dD = D_T - D0, must give beta = 1 (checks G and the flow's score).
After --noise-g the layer's beta should drop to ~0.  Uses the bdg2n twin -- a DIAGNOSTIC only, never a fix.
usage: python dev/flow_tweedie.py SHEAR_FLOW [TRAIN_DATA]   ("val" = rows 450k+ of the 500k catalog)"""
import sys
sys.path.insert(0, "/home/vwetzell/gitrepos/bfd_cnf")
import fitsio, numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
import bulk, shear
from models.shear import dm_dg
DD = "/home/vwetzell/gitrepos/bfd_cnf_imsims/data/moments_bulgedisc_g2_%s.fits"
FLOW = sys.argv[1] if len(sys.argv) > 1 else "flows/desm/shear_s2.eqx"
TRAIN = sys.argv[2] if len(sys.argv) > 2 else DD % "bdg2d"
cols = ["moments", "dm_dg", "badcenter"]
a, b = fitsio.read(DD % "bdg2d", 1, columns=cols), fitsio.read(DD % "bdg2n", 1, columns=cols)
sa, sb = fitsio.read(DD % "bdg2d", "OBS")["noise_sigma"], fitsio.read(DD % "bdg2n", "OBS")["noise_sigma"]
M = a["moments"].astype(np.float64)
phys = bulk.physical(M)                      # shear.load's row filter -> same train/val split
k = sa / (sa - sb)
dM = k[:, None] * (M - b["moments"])
dD = k[:, None, None] * (a["dm_dg"] - b["dm_dg"])
DT = a["dm_dg"].astype(np.float64); D0 = DT - dD
use = phys & ~a["badcenter"] & ~b["badcenter"]
# G per unit template noise variance (mean template), scaled to each row's own sigma_t^2
s2n = sa ** 2 / np.mean(sa[use] ** 2)
G = np.einsum("ngi,nj->gij", dD[use], dM[use]) / use.sum()

Mp = M[phys]; n_train = int(0.9 * len(Mp))
Mtr = shear.load(TRAIN)[0]
flow = bulk.build_flow(jr.key(2), Mtr[:int(0.9 * len(Mtr))], shear=True)
flow = eqx.tree_deserialise_leaves(FLOW, flow)
layer, chart = shear._shear_layer(flow), shear._chart(flow)
f = eqx.filter_jit(jax.vmap(dm_dg, in_axes=(None, 0, None)))
q = np.concatenate([np.asarray(f(layer, jnp.asarray(Mp[i:i + 8192]), chart)[0]) for i in range(0, len(Mp), 8192)])
sc = np.asarray(shear.bulk_score(flow, jnp.asarray(Mp), batch=8192))
Q = np.full_like(DT, np.nan); Q[phys] = q
S = np.full_like(M, np.nan); S[phys] = sc
t = -s2n[:, None, None] * np.einsum("gij,nj->ngi", G, S)          # (n, 2, 5) Tweedie prediction
val = np.zeros(len(M), bool); val[np.flatnonzero(phys)[n_train:]] = True
f_, r_ = M[:, 0], M[:, 1]
win = (f_ > 2200) & (f_ < 20000) & (r_ / f_ > 2.2) & (r_ / f_ < 3.5)
print(f"layer vs D_T: rms (layer - D_T)/rms D_T, g1 rows f r 1 2 c:",
      (np.sqrt(np.nanmean((Q - DT)[use, 0] ** 2, 0)) / DT[use, 0].std(0)).round(3))
names = "fr12c"
beta = {}
for lbl, sel in (("all, train", use & ~val), ("all, val", use & val), ("window, train", use & ~val & win), ("window, val", use & val & win)):
    print(f"--- {lbl}  N={sel.sum()}   beta (+/- se) for dm_x/dg1;  [control dD | layer]")
    for x in (0, 1, 2, 4):
        tt = t[sel, 0, x]
        out = []
        for y in (dD[sel, 0, x], (Q - D0)[sel, 0, x]):
            bt = np.sum(y * tt) / np.sum(tt * tt)
            se = np.sqrt(np.sum((y - bt * tt) ** 2) / np.sum(tt * tt) / (len(tt) - 1))
            out.append((bt, se))
        beta[(lbl, x)] = out
        print(f"  M{names[x]}:  control {out[0][0]:+.3f} +/- {out[0][1]:.3f}   layer {out[1][0]:+.3f} +/- {out[1][1]:.3f}"
              f"   corr(t, layer-D0) {np.corrcoef(tt, (Q - D0)[sel, 0, x])[0, 1]:+.3f}")
