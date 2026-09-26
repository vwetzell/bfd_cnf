"""Regenerate `models/bijections.py`'s `_PHI_CHEB` (run only if the weight changes).

Fits 4 L''(tau(r)) = Phi(r) - r^2 on [_PHI_LO, POINT_SOURCE] by Chebyshev, using the
old Newton inversion as ground truth.  The chart itself must stay analytic -- no
solves -- so the solve lives here, offline."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

C = np.array([0.349792, 0.487396, 0.150208, 0.012604]); KMAX = 1.07635 * np.pi / 0.65
w = lambda k: sum(c * np.cos(n * np.pi * k / KMAX) for n, c in enumerate(C))
mom = lambda tau, p: quad(lambda k: k * w(k) * (k * k / 2) ** p * np.exp(-k * k * tau / 2), 0, KMAX, epsabs=0, epsrel=1e-13)[0]
def r_and_4lpp(tau):
    a0, a1, a2 = (mom(tau, p) for p in range(3))
    return 2 * a1 / a0, 4 * (a2 / a0 - (a1 / a0) ** 2)
PS = r_and_4lpp(1e-10)[0]
LO, DEG = 0.2, 16
x = np.polynomial.chebyshev.chebpts2(200) * (PS - LO) / 2 + (PS + LO) / 2
tau = [brentq(lambda t: r_and_4lpp(t)[0] - r, 1e-10, 1e3, xtol=1e-14) for r in x]
y = np.array([r_and_4lpp(t)[1] for t in tau])
c = np.polynomial.chebyshev.Chebyshev.fit(x, y, DEG, domain=[LO, PS])
print("POINT_SOURCE", PS, "max fit err", np.abs(c(x) - y).max())
print("_PHI_CHEB = np.array([" + ", ".join(f"{v:.17g}" for v in c.coef) + "])")
