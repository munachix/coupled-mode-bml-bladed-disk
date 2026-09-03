# -*- coding: utf-8 -*-
"""Fast re-plot of the BPINN vs Compact ROM vs real ANSYS comparison
(Section 3.5.2), reading the real values already saved by
_make_fig33_bpinn_comparison.py -- no BPINN inference re-run needed.
Removes the in-image title (moved to the docx caption) and tightens
spacing, per explicit user request."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
STEP6_OUT = os.path.join(os.path.dirname(HERE), "Step 6", "output")

d = np.load(os.path.join(STEP6_OUT, "fig33_bpinn_comparison.npz"))
u_bpinn_mm = float(d["u_bpinn_mm"])
compact_rom_mm = float(d["compact_rom_mm"])
real_ansys_mm = float(d["real_ansys_mm"])

plot_style.apply_style()
fig, ax = plt.subplots(figsize=(6.6, 5.2))
labels3 = ['BPINN\n(surrogate)', 'Compact ROM\n(physics-exact)', 'Real ANSYS\nmeasurement']
values3 = [u_bpinn_mm, compact_rom_mm, real_ansys_mm]
colors3 = [plot_style.C_ACC, plot_style.C_OK, plot_style.BLUE]
bars = ax.bar(labels3, values3, color=colors3)
for b, v in zip(bars, values3):
    ax.annotate(f"{v:.2f} mm", (b.get_x() + b.get_width() / 2, v),
                textcoords='offset points', xytext=(0, 8), ha='center', fontsize=15, weight='bold')
ax.set_ylabel('Predicted / measured displacement [mm]')
# Title removed from the image itself (2026-08-29, explicit user request)
# -- the docx caption carries this text instead.
fig.tight_layout()
paths = plot_style.savefig_pub(fig, FIGS, 'step9_fig8c_bpinn_compact_rom_ansys')
print("Saved:", paths)
