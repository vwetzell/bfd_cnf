"""Reproduce weight_functions.pdf's main+inset layout, comparing the updated
KBlackmanHarris window to KSigmaWeight: draw w(k) on a postage stamp, IFFT to
real space, sort pixels by radius, plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bfd.weightfunction import KBlackmanHarris, KSigmaWeight

N = 100
DX = 0.2  # arcsec/pixel
SIGMA = 0.65

INSET_XLIM = (1.5, 7.5)
INSET_YLIM = (3e-5, 3e-1)

kx = 2 * np.pi * np.fft.fftfreq(N, d=DX)
KX, KY = np.meshgrid(kx, kx)
coord = (np.arange(N) - N // 2) * DX
X, Y = np.meshgrid(coord, coord)
r_full = np.hypot(X, Y).ravel()


def radial_weight(weight_obj):
    weight_obj.set_k(KX, KY)
    real_space = np.fft.fftshift(np.fft.ifft2(weight_obj.w)).real
    real_space /= real_space.max()
    order = np.argsort(r_full)
    return r_full[order], real_space.ravel()[order]


def plot_signed(axis, r, v, color):
    """Plot v vs r, dashing+fading the runs where v is negative."""
    sign = np.sign(v)
    breaks = np.flatnonzero(np.diff(sign) != 0) + 1
    for seg_r, seg_v in zip(np.split(r, breaks), np.split(v, breaks)):
        if len(seg_r) == 0:
            continue
        if seg_v[0] >= 0:
            axis.plot(seg_r, seg_v, color=color)
        else:
            axis.plot(seg_r, np.abs(seg_v), color=color, linestyle='--',
                      alpha=0.4)


curves = {
    'Blackman-Harris': (KBlackmanHarris(weightSigma=SIGMA), 'tab:blue'),
    'Sigma Weight': (KSigmaWeight(weightSigma=SIGMA), 'tab:green'),
}

fig, ax = plt.subplots(figsize=(6, 5))
insets = {}
for i, (name, (wobj, color)) in enumerate(curves.items()):
    r, v = radial_weight(wobj)
    plot_signed(ax, r, v, color)

    axins = ax.inset_axes([0.45, 0.72 - 0.32 * i, 0.5, 0.24])
    mask = (r > INSET_XLIM[0]) & (r < INSET_XLIM[1])
    plot_signed(axins, r[mask], v[mask], color)
    axins.set_yscale('log')
    axins.set_xlim(*INSET_XLIM)
    axins.set_ylim(*INSET_YLIM)
    axins.set_title(name, fontsize=9)
    insets[name] = axins

ax.set_xlim(0, 5)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('Radius (arcsec)')
handles = [Line2D([], [], color=color, label=name)
           for name, (_, color) in curves.items()]
ax.legend(handles=handles, loc='lower right')

fig.tight_layout()
fig.savefig('dev/weight_function_bh.png', dpi=150)
print('wrote dev/weight_function_bh.png')
