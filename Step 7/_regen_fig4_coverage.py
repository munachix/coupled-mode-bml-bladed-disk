# -*- coding: utf-8 -*-
"""Test-case regeneration: Fig 12 (step7_fig4_coverage_calibration) at the
legend fontsize now matching axes.labelsize exactly (both 16), per
explicit user request to verify the approach on this one figure first."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step7")
NB = 24

d = np.load(os.path.join(OUT, "coverage_check.npz"))
coverage = {"levels": d["levels"], "empirical_coverage": d["empirical_coverage"], "n_trials": int(d["n_trials"])}

plot_style.apply_style()
fig, ax = plt.subplots(figsize=(6.5, 5.8))
ax.plot(coverage['levels'], coverage['empirical_coverage'], 'o-', color=plot_style.ORANGE,
        lw=2.0, ms=6, mec=plot_style.SURFACE, mew=1.0, label='inversion calibration')
ax.plot([0, 1], [0, 1], color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect calibration')
ax.set_xlabel('Nominal credible level')
ax.set_ylabel(f"Empirical coverage ({coverage['n_trials']} held-out trials x {NB} blades)")
plot_style.two_tier_title(ax, 'Posterior credible-interval calibration')
plot_style.legend_inside(ax, loc='upper left', bbox_to_anchor=(0.0, 0.92), fontsize=17)
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step7_fig4_coverage_calibration')
print("Saved step7_fig4_coverage_calibration.png")
