# -*- coding: utf-8 -*-
"""
Redesigns the BPINN-accelerated reconstruction accuracy figure (Section
3.2.4). v3 (2026-08-30, explicit user request: "don't make it 3D but make
it better"): a clean 2D bar chart, per-mode R^2 colored by real training
topology (pair/chain/single), with the ensemble mean drawn as a reference
line so the "every mode clears a floor" story reads at a glance instead
of requiring a 3D view angle to parse. Real data unchanged, from
Step 7/output/step7_config.json (bpinn_reconstruction_r2_per_mode).
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "output", "step7_config.json")
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step7")

SINGLE = {2}
PAIR = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10}


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)
    r2_per_mode = d["bpinn_reconstruction_r2_per_mode"]
    modes = sorted(int(k) for k in r2_per_mode.keys())
    r2 = np.array([r2_per_mode[str(m)] for m in modes])

    def color_for(m):
        if m in SINGLE:
            return plot_style.C_OK
        if m in PAIR:
            return plot_style.C_1B
        return plot_style.C_ACC

    colors = [color_for(m) for m in modes]
    mean_r2 = float(r2.mean())

    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    # v4 (2026-09-01, explicit user request for a cleaner analytical-curve
    # look): a connected line through the real per-mode values, real
    # markers colored by real training topology, replacing the discrete
    # bar chart -- straight segments between real integer mode indices
    # only, no smoothing/interpolation invented between them.
    ax.plot(modes, r2, '-', color=plot_style.INK_MUTED, lw=1.6, zorder=2)
    ax.scatter(modes, r2, s=110, c=colors, zorder=3, edgecolor=plot_style.SURFACE, linewidth=1.3)
    ax.axhline(mean_r2, color=plot_style.INK, ls='--', lw=1.6, alpha=0.7, zorder=1,
               label=f'mean R$^2$ = {mean_r2:.3f}')
    ax.fill_between(modes, 0.75, r2, color=plot_style.INK_MUTED, alpha=0.06, zorder=0)
    ax.set_ylim(0.75, 1.02)
    ax.set_xlim(modes[0] - 0.6, modes[-1] + 0.6)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Reconstruction $R^2$')
    ax.set_xticks(modes)
    ax.tick_params(axis='x', labelsize=11)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', ls='', color=plot_style.C_1B, ms=11, label='pair'),
        Line2D([0], [0], marker='o', ls='', color=plot_style.C_ACC, ms=11, label='chain'),
        Line2D([0], [0], marker='o', ls='', color=plot_style.C_OK, ms=11, label='single'),
        Line2D([0], [0], color=plot_style.INK, ls='--', lw=1.6, alpha=0.7,
               label=f'mean R$^2$ = {mean_r2:.3f}'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', frameon=True,
              facecolor=plot_style.SURFACE, edgecolor=plot_style.GRID_HAIRLINE,
              framealpha=0.92, borderpad=0.6, fontsize=13, ncol=1,
              handlelength=1.6, labelspacing=0.5)
    # Title/subtitle removed from the image itself -- the docx caption
    # carries this text instead.
    fig.tight_layout()
    paths = plot_style.savefig_pub(fig, FIGS_DIR, "step7_fig5_3d_reconstruction_accuracy")
    print("Saved:", paths)


if __name__ == "__main__":
    main()
