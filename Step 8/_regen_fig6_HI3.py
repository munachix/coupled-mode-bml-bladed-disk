# -*- coding: utf-8 -*-
"""
Enriches the HI3 trajectory figure (Section 3.3.1) to match the visual
richness of the HI2 figure ("fix Figure 16 so it can be like 15"), per
explicit user request. HI3 has no calibrated detection threshold of its
own (only HI2 does), so rather than fabricate one, this adds the one
piece of real, already-available context that legitimately transfers
across indicators: the detection-time marker (the time step where HI2
first crosses its calibrated threshold, from calibration.npz -- the same
real event HI3 is being read alongside), plus bigger fonts throughout,
matching the rest of this pass.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step8")

traj = np.load(os.path.join(OUT, "damage_trajectory.npz"))
calib = np.load(os.path.join(OUT, "calibration.npz"))

t = traj["t_norm"]
HI3 = traj["HI3"]
HI2 = traj["HI2"]
threshold = float(calib["threshold"])
above = np.where(HI2 > threshold)[0]
detect_idx = int(above[0]) if len(above) else -1

plot_style.apply_style()
fig, ax = plt.subplots(figsize=(8.3, 5.6))
ax.plot(t, HI3, color=plot_style.ORANGE, lw=2.4, marker='o', ms=7,
        mec=plot_style.SURFACE, mew=1.0, label='HI3')
if detect_idx >= 0:
    ax.axvline(t[detect_idx], color=plot_style.AQUA, ls=':', lw=1.8, label='detection time')
ax.axhline(HI3[0], color=plot_style.INK_MUTED, ls='--', lw=1.4, label='trajectory start')
ax.set_xlabel('Normalized time / operating cycle')
ax.set_ylabel('HI3 [mm]\nRMS BML amplitude deviation from healthy baseline')
plot_style.two_tier_title(ax, 'HI3 monitoring dashboard', 'BML-driven forced-response amplitude deviation')
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
fig.subplots_adjust(left=0.19, top=0.84, right=0.97)
out_path = os.path.join(FIGS, 'step8_fig6_HI3_bpinn_amplitude.png')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.15, facecolor=fig.get_facecolor())
print("Saved step8_fig6_HI3_bpinn_amplitude.png")
