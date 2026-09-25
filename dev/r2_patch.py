"""DIAGNOSTIC ONLY -- not a fix (see memory: diagnostic-scalings-are-not-fixes).

Rescales the shear layer's SECOND-ORDER spin-2 response so its sheared path's
d2X/dg2 matches the exact one, using the per-|e| slopes measured by
dev/shear_response_by_e2.py (flow/exact 0.849 -> 0.967).  If this moves the
bias, the follow-up is making the flow LEARN its second-order terms, never
keeping this patch.

The sheared path is X(g) = z - Q g + 1/2 g.M2.g with M2 = (C + C^T) - R and
C_iab = sum_j dQ_ia/dz_j Q_jb (`ShearResponse.shear`), so the patch sets
M2'[3:5] = M2[3:5] / s(|e|) by R' = (C + C^T) - M2', leaving Q, the spin-0
rows and everything else untouched.  `unshear` is rebuilt as z + Q g +
1/2 g.R'.g (exactly `response`'s form), so the log-det, `shear` and the
estimator all see the patched tensors.
"""
import jax
import jax.numpy as jnp

from models import shear as S
from models.bijections import spin2_unbound

E_CENTRES = jnp.array([0.025, 0.065, 0.095, 0.13, 0.175, 0.25])
SLOPES = jnp.array([0.849, 0.892, 0.919, 0.945, 0.969, 0.967])   # logs/shear_response_by_e2.log


def apply():
    orig_rt = S.ShearResponse.response_tensors

    def response_tensors(self, x):
        Q, R = orig_rt(self, x)
        dQ = jax.jacfwd(lambda zz: orig_rt(self, zz)[0])(x)
        C = jnp.einsum("iaj,jb->iab", dQ, Q)
        CC = C + jnp.swapaxes(C, 1, 2)
        M2 = CC - R
        es = S.unwrap(self.e_scale)
        e1, e2 = spin2_unbound(x[3] * es, x[4] * es)
        s = jnp.interp(jnp.hypot(e1, e2), E_CENTRES, SLOPES)
        M2 = M2.at[3:].set(M2[3:] / s)
        return Q, CC - M2

    def unshear(self, x, condition):
        g = condition[:2]
        Q, R = response_tensors(self, x)
        return x + Q @ g + 0.5 * jnp.einsum("iab,a,b->i", R, g, g)

    S.ShearResponse.response_tensors = response_tensors
    S.ShearResponse.unshear = unshear
    print("r2_patch: DIAGNOSTIC second-order spin-2 rescale ACTIVE")
