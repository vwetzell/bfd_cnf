import sys; sys.path.insert(0, ".")
import equinox as eqx, jax, jax.random as jr, numpy as np
import bias as B, bulk, shear
m = np.asarray(shear.load(f"../bfd_cnf_imsims/data/{B.TRAIN_DATA['gauss2_v4n']}")[0])[:2000]
flow = eqx.tree_deserialise_leaves("flows/centroid_g2v4n.eqx", bulk.build_flow(jr.key(0), m, shear=True, centroid=True))
b = flow.bijection; print(type(b).__name__, type(getattr(b, "bijection", None)).__name__)
inner = b.bijection
for i, l in enumerate(inner.bijections):
    print(i, type(l).__name__, getattr(l, "cond_shape", None), sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(l, eqx.is_inexact_array))))
