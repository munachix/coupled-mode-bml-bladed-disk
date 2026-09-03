# -*- coding: utf-8 -*-
"""
Four 3D candidate designs for Figure 24 (Section 3.4.1).

Figure 24's job is a null result: across 200 realizations traced to their own
saddle-node fold, worst-blade geometric mistuning magnitude does not set the
peak nonlinear response amplitude (r = +0.101). As a flat 2D scatter that reads
as an empty plot -- there is nothing for the eye to follow, which is exactly
the complaint. A 3D view fixes that by putting the null result next to the
non-null one on the same axes: the ensemble also carries a signed
participation-weighted stiffness shift (mistuning *direction*, Figure 25) and
the resulting resonance-peak frequency, and direction predicts frequency almost
exactly (r = 1.0000). Showing both on one set of axes makes the flatness
legible as a contrast rather than as an absence.

All four candidates use the same real ensemble, Step 9's
`mistuning_nonlinearity_extended.npz`, with no refitting or resampling.

Outputs (candidates, for selection -- only the chosen one goes in the paper):
    figures/step9/_cand24_A_amplitude_plane.png
    figures/step9/_cand24_B_frequency_plane_amp_color.png
    figures/step9/_cand24_C_stems_to_mean.png
    figures/step9/_cand24_D_dual_surface.png
    figures/step9/_cand24_CONTACT_SHEET.png
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

d = np.load(os.path.join(OUT, "mistuning_nonlinearity_extended.npz"))
mag = d["max_abs_df"]          # worst-blade |df/f|, %
shift = d["shift_m"]           # signed mode-0 participation-weighted shift, %
amp = d["peak_amp"]            # true nonlinear peak amplitude, mm
dfreq = d["peak_freq_centered"]  # peak frequency about ensemble mean, Hz
c_amp = float(d["corr_amp"])
c_frq = float(d["corr_freq"])
n = len(mag)
amp_mean = float(amp.mean())
amp_spread_pct = 100.0 * (amp.max() - amp.min()) / amp_mean

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

INK = plot_style.INK
MUTED = plot_style.INK_SECONDARY


def dress(ax, xl, yl, zl):
    """Common 3D axis treatment: pale panes, hairline grid, Times labels."""
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 1.0))
        pane._axinfo["grid"]["color"] = (0.85, 0.87, 0.89, 1.0)
        pane._axinfo["grid"]["linewidth"] = 0.7
    ax.set_xlabel(xl, labelpad=12)
    ax.set_ylabel(yl, labelpad=12)
    ax.set_zlabel(zl, labelpad=14)
    ax.tick_params(labelsize=12)


def title(fig, t, sub):
    plot_style.figure_title(fig, t, sub, x=0.02, y_title=0.99, y_subtitle=0.945)


# ---------------------------------------------------------------- candidate A
# Amplitude over the whole mistuning plane, against a flat reference plane at
# the ensemble mean. The point: the cloud hugs the plane no matter where in the
# (magnitude, direction) plane a realization sits.
def candidate_A():
    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    gx, gy = np.meshgrid(np.linspace(mag.min(), mag.max(), 2),
                         np.linspace(shift.min(), shift.max(), 2))
    ax.plot_surface(gx, gy, np.full_like(gx, amp_mean), color=plot_style.FADE,
                    alpha=0.30, edgecolor="none", zorder=1)

    ax.scatter(mag, shift, amp, s=26, c=plot_style.C_1B, alpha=0.75,
               depthshade=False, linewidth=0, zorder=3)

    ax.set_zlim(amp_mean - 0.012, amp_mean + 0.012)
    dress(ax, r"Worst-blade mistuning, max $|\delta f/f|$  [%]",
          "Participation-weighted\nstiffness shift  [%]",
          "Peak nonlinear amplitude  [mm]")
    ax.view_init(elev=22, azim=-58)
    ax.text2D(0.015, 0.90, f"grey plane: ensemble mean {amp_mean:.3f} mm\n"
                           f"total spread {amp_spread_pct:.1f}% of that mean",
              transform=ax.transAxes, fontsize=12.5, color=MUTED,
              ha="left", va="top")
    title(fig, "Amplitude is flat over the whole mistuning plane",
          f"{n} realizations, each traced to its own fold; neither mistuning "
          f"magnitude nor direction moves the peak (r = {c_amp:+.3f} against magnitude)")
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.00, right=0.92)
    plot_style.savefig_pub(fig, FIGS, "_cand24_A_amplitude_plane")
    plt.close(fig)


# ---------------------------------------------------------------- candidate B
# Frequency as the height, amplitude as the color. Says both of Section 3.4.1's
# results at once: the surface tilts along direction only, while the color is
# uniform everywhere -- amplitude responds to nothing.
def candidate_B():
    fig = plt.figure(figsize=(9.9, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(amp.min(), amp.max())
    sc = ax.scatter(mag, shift, dfreq, s=34, c=amp, cmap="viridis", norm=norm,
                    alpha=0.92, depthshade=False, linewidth=0)

    # least-squares plane through (mag, shift) -> dfreq, to show the tilt is
    # entirely along the direction axis.
    A = np.column_stack([mag, shift, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, dfreq, rcond=None)
    gx, gy = np.meshgrid(np.linspace(mag.min(), mag.max(), 8),
                         np.linspace(shift.min(), shift.max(), 8))
    gz = coef[0] * gx + coef[1] * gy + coef[2]
    ax.plot_surface(gx, gy, gz, color=plot_style.FADE, alpha=0.22,
                    edgecolor=plot_style.FADE, linewidth=0.4)

    dress(ax, r"Worst-blade mistuning, max $|\delta f/f|$  [%]",
          "Participation-weighted\nstiffness shift  [%]",
          "Resonance-peak shift  [Hz]")
    ax.view_init(elev=20, azim=-62)
    cb = fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.62, aspect=18)
    cb.set_label("Peak nonlinear amplitude  [mm]", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    ax.text2D(0.015, 0.92,
              f"fitted plane: {coef[1]:+.2f} Hz per % of direction,\n"
              f"{coef[0]:+.3f} Hz per % of magnitude",
              transform=ax.transAxes, fontsize=12.5, color=MUTED,
              ha="left", va="top")
    title(fig, "Direction sets the resonance shift, magnitude sets nothing",
          f"{n} realizations; the plane tilts only along direction "
          f"(r = {c_frq:+.4f}) while amplitude, in color, stays uniform "
          f"({amp_spread_pct:.1f}% total spread)")
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.00, right=0.97)
    plot_style.savefig_pub(fig, FIGS, "_cand24_B_frequency_plane_amp_color")
    plt.close(fig)


# ---------------------------------------------------------------- candidate C
# Same axes as A, but every realization gets a stem down to the mean plane, so
# the residuals are drawn rather than inferred. Reads as a bed of very short
# pins -- the flatness becomes a measured quantity.
def candidate_C():
    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    for xi, yi, zi in zip(mag, shift, amp):
        ax.plot([xi, xi], [yi, yi], [amp_mean, zi],
                color=plot_style.C_1B, lw=0.8, alpha=0.45, zorder=2)
    ax.scatter(mag, shift, amp, s=22, c=plot_style.C_1B, alpha=0.95,
               depthshade=False, linewidth=0, zorder=3)

    gx, gy = np.meshgrid(np.linspace(mag.min(), mag.max(), 2),
                         np.linspace(shift.min(), shift.max(), 2))
    ax.plot_surface(gx, gy, np.full_like(gx, amp_mean), color=plot_style.INK,
                    alpha=0.10, edgecolor="none", zorder=1)

    ax.set_zlim(amp_mean - 0.008, amp_mean + 0.008)
    dress(ax, r"Worst-blade mistuning, max $|\delta f/f|$  [%]",
          "Participation-weighted\nstiffness shift  [%]",
          "Peak nonlinear amplitude  [mm]")
    ax.view_init(elev=26, azim=-52)
    title(fig, "Every realization lands on the same peak amplitude",
          f"{n} realizations, stem length is the deviation from the ensemble "
          f"mean of {amp_mean:.3f} mm; total spread {amp_spread_pct:.1f}%")
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.00, right=0.92)
    plot_style.savefig_pub(fig, FIGS, "_cand24_C_stems_to_mean")
    plt.close(fig)


# ---------------------------------------------------------------- candidate D
# Two 3D panels over the identical (magnitude, direction) footprint: amplitude
# on the left, frequency on the right, both drawn as an interpolated surface.
# The strongest side-by-side statement -- a flat sheet next to a tilted one.
def candidate_D():
    from matplotlib.tri import Triangulation

    tri = Triangulation(mag, shift)
    fig = plt.figure(figsize=(13.4, 6.2))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_trisurf(tri, amp, cmap="Blues", edgecolor="none", alpha=0.95,
                     linewidth=0)
    ax1.set_zlim(amp_mean - 0.012, amp_mean + 0.012)
    dress(ax1, r"max $|\delta f/f|$  [%]", "stiffness shift  [%]",
          "Peak amplitude  [mm]")
    ax1.view_init(elev=20, azim=-58)
    ax1.zaxis.labelpad = 10
    ax1.set_title(f"(a)  amplitude: flat, r = {c_amp:+.3f}",
                  fontsize=15, color=INK, pad=2)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_trisurf(tri, dfreq, cmap="Oranges", edgecolor="none", alpha=0.95,
                     linewidth=0)
    dress(ax2, r"max $|\delta f/f|$  [%]", "stiffness shift  [%]",
          "Resonance shift  [Hz]")
    ax2.view_init(elev=20, azim=-58)
    ax2.zaxis.labelpad = 10
    ax2.set_title(f"(b)  frequency: a plane, r = {c_frq:+.4f}",
                  fontsize=15, color=INK, pad=2)

    title(fig, "Same mistuning ensemble, two different answers",
          f"{n} realizations over the identical (magnitude, direction) "
          f"footprint: amplitude is a flat sheet, resonance shift is a tilted plane")
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.01, right=0.99, wspace=0.10)
    plot_style.savefig_pub(fig, FIGS, "_cand24_D_dual_surface")
    plt.close(fig)


def contact_sheet():
    """One page showing all four candidates side by side for selection."""
    import matplotlib.image as mpimg

    names = [("A", "_cand24_A_amplitude_plane"),
             ("B", "_cand24_B_frequency_plane_amp_color"),
             ("C", "_cand24_C_stems_to_mean"),
             ("D", "_cand24_D_dual_surface")]
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 12.4))
    for (letter, fn), ax in zip(names, axes.ravel()):
        ax.imshow(mpimg.imread(os.path.join(FIGS, fn + ".png")))
        ax.set_title(f"Candidate {letter}", fontsize=20, color=INK,
                     fontweight="bold", pad=6)
        ax.axis("off")
        ax.grid(False)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99,
                        hspace=0.08, wspace=0.03)
    plot_style.savefig_pub(fig, FIGS, "_cand24_CONTACT_SHEET")
    plt.close(fig)


if __name__ == "__main__":
    print("=== Figure 24 3D candidates ===", flush=True)
    print(f"  n = {n}", flush=True)
    print(f"  |df/f|max  : [{mag.min():.2f}, {mag.max():.2f}] %", flush=True)
    print(f"  shift      : [{shift.min():+.2f}, {shift.max():+.2f}] %", flush=True)
    print(f"  peak amp   : [{amp.min():.4f}, {amp.max():.4f}] mm "
          f"(mean {amp_mean:.4f}, spread {amp_spread_pct:.1f}%)", flush=True)
    print(f"  peak dfreq : [{dfreq.min():+.2f}, {dfreq.max():+.2f}] Hz", flush=True)
    print(f"  corr(mag, amp)   = {c_amp:+.4f}", flush=True)
    print(f"  corr(shift, freq) = {c_frq:+.4f}", flush=True)
    candidate_A()
    candidate_B()
    candidate_C()
    candidate_D()
    contact_sheet()
    print("Saved 4 candidates + contact sheet to figures/step9/", flush=True)
