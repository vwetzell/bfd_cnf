"""Corner plot: bulgedisc_g2 (bdg2n) prior vs gauss2_fwd (g2v4n) prior, measured moments.
Usage: python dev/corner_bdg2_vs_g2.py [out.png]"""
import sys
import corner, fitsio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "../bfd_cnf_imsims/data"
N = 200000
LAB = [r"$\log_{10} M_f$", r"$M_r/M_f$", r"$M_c/M_r$", r"$M_1/M_r$", r"$M_2/M_r$"]


def axes(path):
    Mf, Mr, M1, M2, Mc = fitsio.read(path, rows=range(N))["moments"].T[:5]
    return np.c_[np.log10(Mf), Mr / Mf, Mc / Mr, M1 / Mr, M2 / Mr]


g2 = axes(f"{D}/moments_gauss2_fwd_g2v4n.fits")
bd = axes(f"{D}/moments_bulgedisc_g2_bdg2n.fits")
rng = [tuple(np.percentile(np.r_[g2[:, i], bd[:, i]], [0.5, 99.5])) for i in range(5)]
kw = dict(labels=LAB, range=rng, bins=50, plot_datapoints=False, fill_contours=False,
          levels=(0.39, 0.86, 0.99), plot_density=False, smooth=1.0)
fig = corner.corner(g2, color="C0", hist_kwargs=dict(density=True, color="C0"), **kw)
corner.corner(bd, fig=fig, color="C3", hist_kwargs=dict(density=True, color="C3"), **kw)
fig.legend(handles=[plt.Line2D([], [], color="C0", label=f"gauss2_fwd (g2v4n prior)"),
                    plt.Line2D([], [], color="C3", label=f"bulgedisc_g2 (bdg2n prior)")],
           loc="upper right", fontsize=14, frameon=False)
fig.suptitle(f"measured moments, noise 0.093, {N} each; contours 1/2/3 sigma", fontsize=12)
out = sys.argv[1] if len(sys.argv) > 1 else "dev/corner_bdg2_vs_g2.png"
fig.savefig(out, dpi=110)
print("wrote", out)
