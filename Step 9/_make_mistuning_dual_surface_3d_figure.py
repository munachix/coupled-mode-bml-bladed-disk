# -*- coding: utf-8 -*-
"""
Section 3.4.1's two mistuning relationships on one pair of 3D axes.

Replaces the two separate 2D scatters (the magnitude-versus-amplitude null
result and the direction-versus-frequency-shift relationship) with a single
figure showing both over the identical (magnitude, direction) footprint.

WHAT CHANGED IN THE RESULT, and it is not cosmetic. Building this figure meant
fitting a plane to each quantity over both mistuning coordinates rather than
correlating each against one coordinate, and that exposes an error of emphasis
in the old pair of figures. Both quantities are predicted by the signed
participation-weighted stiffness shift to machine precision:

    corr(direction, peak amplitude)   = +1.0000
    corr(direction, resonance shift)  = +1.0000
    corr(magnitude, peak amplitude)   = +0.101
    corr(magnitude, resonance shift)  = +0.102

So amplitude is not unpredictable, which is what a reader takes away from a flat
scatter against magnitude alone. It is predicted exactly, by the same variable
that predicts the frequency shift; the difference between the two is the size of
the effect, not the quality of the relationship. Direction moves the resonance
by 2.49 Hz per percent of stiffness shift, which over the ensemble's 6.8% range
is 17.1 Hz, about 4.5 times this mode's own half-power bandwidth at the peak
(2*zeta*f_peak = 3.80 Hz), so a blade that sat on resonance no longer does. The
same variable moves the peak amplitude by 0.00073 mm per percent, which over the
same range is 0.0050 mm, or 1.6% of the ensemble mean, and changes nothing.

The two panels are drawn to show exactly that: the same plane, fitted the same
way, nearly horizontal in (a) and steeply tilted in (b).

AXIS NOTE, disclosed in the caption because it is a choice. Panel (a)'s vertical
axis deliberately spans a +/-5% window about the ensemble mean rather than
autoscaling to the data. Autoscaling would magnify a 1.6% spread to fill the
panel and make a negligible dependence look like a strong one, which is the
opposite of the result. Panel (b) autoscales, since its spread is the result.

Output: figures/step9/step9_fig9h_mistuning_dual_surface_3d.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style  # noqa: E402

FIGS = os.path.join(ROOT, "figures", "step9")
OUT = os.path.join(HERE, "output")

ZETA = 0.002              # uniform modal damping, Table 1
AMP_WINDOW = 0.05         # panel (a) half-window, as a fraction of the mean

d = np.load(os.path.join(OUT, "mistuning_nonlinearity_extended.npz"))
mag = d["max_abs_df"]            # worst-blade |df/f|, %
shift = d["shift_m"]             # signed mode-0 participation-weighted shift, %
amp = d["peak_amp"]              # true nonlinear peak amplitude, mm
dfreq = d["peak_freq_centered"]  # peak frequency about the ensemble mean, Hz
pfreq = d["peak_freq"]
n = len(mag)

amp_mean = float(amp.mean())
bandwidth = 2.0 * ZETA * float(pfreq.mean())     # half-power width at the peak
G = np.column_stack([mag, shift, np.ones(n)])


def plane(z):
    """Least-squares plane over both mistuning coordinates, plus its R^2."""
    c, *_ = np.linalg.lstsq(G, z, rcond=None)
    pred = G @ c
    r2 = 1.0 - ((z - pred) ** 2).sum() / ((z - z.mean()) ** 2).sum()
    return c, float(r2)


c_amp, r2_amp = plane(amp)
c_frq, r2_frq = plane(dfreq)

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

INK = plot_style.INK
MUTED = plot_style.INK_SECONDARY

gx, gy = np.meshgrid(np.linspace(mag.min(), mag.max(), 40),
                     np.linspace(shift.min(), shift.max(), 40))


def panel(ax, z, coef, cmap, zlabel, zlim, title, note):
    gz = coef[0] * gx + coef[1] * gy + coef[2]
    ax.plot_surface(gx, gy, gz, cmap=cmap, alpha=0.80, linewidth=0,
                    antialiased=True, rstride=1, cstride=1, zorder=1)
    ax.scatter(mag, shift, z, s=13, color=INK, alpha=0.55, depthshade=False,
               linewidth=0, zorder=4)

    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 1.0))
        pane._axinfo["grid"]["color"] = (0.85, 0.87, 0.89, 1.0)
        pane._axinfo["grid"]["linewidth"] = 0.7
    ax.set_xlabel(r"Worst-blade mistuning" "\n" r"max $|\delta f/f|$  [%]",
                  labelpad=16, fontsize=14)
    ax.set_ylabel("Participation-weighted\nstiffness shift  [%]",
                  labelpad=16, fontsize=14)
    ax.set_zlabel(zlabel, labelpad=16, fontsize=14)
    ax.set_zlim(*zlim)
    ax.tick_params(labelsize=11.5)
    ax.view_init(elev=19, azim=-58)
    ax.set_title(title, fontsize=16, color=INK, fontweight="bold", pad=-16)
    ax.text2D(0.5, -0.02, note, transform=ax.transAxes, fontsize=13,
              color=MUTED, ha="center", va="top", linespacing=1.5)


fig = plt.figure(figsize=(14.6, 7.0))

ax1 = fig.add_subplot(121, projection="3d")
panel(ax1, amp, c_amp, "Blues",
      "Peak nonlinear amplitude  [mm]",
      (amp_mean * (1 - AMP_WINDOW), amp_mean * (1 + AMP_WINDOW)),
      "(a)  peak amplitude",
      f"plane slope {c_amp[1]:+.5f} mm per % of direction\n"
      f"total spread {100*(amp.max()-amp.min())/amp_mean:.1f}% of the "
      f"{amp_mean:.3f} mm mean")

ax2 = fig.add_subplot(122, projection="3d")
pad = 0.08 * (dfreq.max() - dfreq.min())
panel(ax2, dfreq, c_frq, "Oranges",
      "Resonance-peak shift  [Hz]",
      (dfreq.min() - pad, dfreq.max() + pad),
      "(b)  resonance-peak shift",
      f"plane slope {c_frq[1]:+.2f} Hz per % of direction\n"
      f"total spread {dfreq.max()-dfreq.min():.1f} Hz, "
      f"{(dfreq.max()-dfreq.min())/bandwidth:.1f} half-power bandwidths")

plot_style.figure_title(
    fig,
    "Mistuning direction sets both, but only one of them by enough to matter",
    f"{n} realizations at fixed forcing, each traced to its own fold; the same "
    f"least-squares plane over the same footprint fits each quantity to "
    f"R² = {r2_amp:.4f} and {r2_frq:.4f}",
    x=0.012, y_title=1.045, y_subtitle=1.002)
fig.subplots_adjust(top=0.94, bottom=0.09, left=0.00, right=0.99, wspace=0.06)

plot_style.savefig_pub(fig, FIGS, "step9_fig9h_mistuning_dual_surface_3d")

print("=== Section 3.4.1 dual-surface figure ===", flush=True)
print(f"  n = {n}", flush=True)
print(f"  corr(magnitude, amplitude)      = "
      f"{np.corrcoef(mag, amp)[0,1]:+.4f}", flush=True)
print(f"  corr(direction, amplitude)      = "
      f"{np.corrcoef(shift, amp)[0,1]:+.4f}", flush=True)
print(f"  corr(magnitude, resonance shift)= "
      f"{np.corrcoef(mag, dfreq)[0,1]:+.4f}", flush=True)
print(f"  corr(direction, resonance shift)= "
      f"{np.corrcoef(shift, dfreq)[0,1]:+.4f}", flush=True)
print(f"  amplitude plane : {c_amp[1]:+.5f} mm/%, R2 = {r2_amp:.4f}, "
      f"spread {100*(amp.max()-amp.min())/amp_mean:.2f}% of mean", flush=True)
print(f"  frequency plane : {c_frq[1]:+.4f} Hz/%, R2 = {r2_frq:.4f}, "
      f"spread {dfreq.max()-dfreq.min():.2f} Hz", flush=True)
print(f"  half-power bandwidth at the peak (2*zeta*f) = {bandwidth:.3f} Hz "
      f"-> spread is {(dfreq.max()-dfreq.min())/bandwidth:.2f} bandwidths",
      flush=True)
print("Saved step9_fig9h_mistuning_dual_surface_3d.png", flush=True)
