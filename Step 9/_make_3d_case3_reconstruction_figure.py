# -*- coding: utf-8 -*-
"""
Redesigns the Case 1 (mistuned, nonlinear, formerly "Case 3") progressive-
reconstruction figure. v3 (2026-08-30, explicit user request: neither a
bar chart nor 3D -- "use something else"): a horizontal convergence
("lollipop") chart -- modeling stage on the y-axis (ordinal, so long
labels read horizontally instead of needing rotation), predicted
displacement on the x-axis, each stage a colored dot on a thin
connecting line, against a shaded band marking the real ANSYS reference
value and its uncertainty. Color still encodes percent error against
that reference (the same real, meaningful quantity as the previous
heat-colored bars), so nothing about the underlying data changes -- only
the chart type and the stage labels, which now drop the parenthetical
detail entirely per explicit request (e.g. "all 70 modes (linear)" ->
"all 70 modes").

Real values, same source as before (Section 3.5.1's own text):
  2-of-70 basis modes             : 0.556 mm
  all 70 modes                    : 1.717 mm
  all 70 + real coupling          : 1.681 mm
  all 70 + mode 2 bridged         : 1.214 mm
  BML surrogate                 : 1.145 mm
  Real ANSYS                      : 1.222 +/- 0.019 mm
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step9")

STAGES = [
    "2 of 70 basis modes",
    "all 70 modes",
    "all 70 + real coupling",
    "all 70 + mode 2 bridged",
    "BML surrogate",
    "Full-order FEM",
]
VALUES = [0.556, 1.717, 1.681, 1.214, 1.145, 1.222]
ERR = [0, 0, 0, 0, 0, 0.019]


def main():
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    n = len(VALUES)
    y = np.arange(n)[::-1]  # first stage at top, ANSYS at bottom
    real_ansys = VALUES[-1]
    pct_err = [abs(v - real_ansys) / real_ansys * 100 for v in VALUES]
    cmap = cm.get_cmap("inferno_r")
    norm = mcolors.Normalize(vmin=0, vmax=max(pct_err[:-1]))
    colors = [cmap(norm(e)) for e in pct_err]

    band_lo, band_hi = real_ansys - ERR[-1], real_ansys + ERR[-1]
    ax.axvspan(band_lo, band_hi, color=plot_style.BLUE, alpha=0.12, zorder=0)
    ax.axvline(real_ansys, color=plot_style.BLUE, ls='--', lw=1.4, alpha=0.7, zorder=1,
               label=f'full-order FEM reference ({real_ansys:.3f} mm)')

    ax.hlines(y, 0, VALUES, color=plot_style.INK_MUTED, lw=1.6, zorder=2)
    ax.scatter(VALUES, y, s=220, color=colors, edgecolor=plot_style.INK, linewidth=1.2, zorder=3)

    for yi, v, pe, is_ref in zip(y, VALUES, pct_err, [False] * (n - 1) + [True]):
        label = f"{v:.3f} mm" if is_ref else f"{v:.3f} mm  ({pe:.1f}%)"
        ax.annotate(label, (v, yi), xytext=(12, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=13, color=plot_style.INK)

    ax.set_yticks(y)
    ax.set_yticklabels(STAGES, fontsize=13)
    ax.set_xlim(0, 2.55)
    ax.set_xlabel('Displacement at node 1171, UZ  [mm]')
    ax.set_ylim(-0.7, n - 0.3)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('% error vs. the FEM reference', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.legend(loc='lower right', frameon=True, facecolor=plot_style.SURFACE,
              edgecolor=plot_style.GRID_HAIRLINE, framealpha=0.92, borderpad=0.6,
              fontsize=12, handlelength=1.6)
    # Title/subtitle removed from the image itself -- the docx caption
    # carries this text instead.
    fig.tight_layout()
    paths = plot_style.savefig_pub(fig, FIGS_DIR, "step9_fig11a_case1_3d_reconstruction")
    print("Saved:", paths)


if __name__ == "__main__":
    main()
