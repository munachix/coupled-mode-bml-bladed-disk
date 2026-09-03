# -*- coding: utf-8 -*-
"""
Case 2 (formerly Case 4) reconstruction figure. v2 (2026-08-30, explicit
user request: "Fig 23 and 24 are one fig just make it a and b so it's
just figure 23" -- reverses the earlier split-into-two-standalones pass):
one stacked image, frequency agreement on top (a), MAC mode-shape
agreement below (b), each panel's letter baked into the PNG via
loc="left" titles, matching this project's own established convention
for multi-panel figures (the waveform gallery). Real data unchanged,
from Step 9/output/case4_comparison.npz.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")

d = np.load(os.path.join(OUT, "case4_comparison.npz"))
ft, fi = d["freqs_true"], d["freqs_inferred"]
mac_diag = d["mac_diag"]
freq_err_pct = d["freq_err_pct"]

plot_style.apply_style()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 10.4))

lo, hi = min(ft.min(), fi.min()), max(ft.max(), fi.max())
pad = (hi - lo) * 0.06
ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '-', color=plot_style.INK, lw=1.4, alpha=0.5)
ax1.scatter(ft, fi, s=64, color=plot_style.BLUE, edgecolors=plot_style.SURFACE, linewidths=1.0, zorder=4)
ax1.set_xlabel('Reconstructed-from-TRUE frequency  [Hz]')
ax1.set_ylabel('Reconstructed-from-INFERRED frequency  [Hz]')
ax1.set_title(f"(a) Frequency agreement -- mean err {freq_err_pct.mean():.3f}%, "
              f"max {freq_err_pct.max():.3f}%", loc="left", fontsize=15, fontweight="bold",
              color=plot_style.INK, pad=8)

idx_arr = np.arange(len(ft))
ax2.bar(idx_arr, mac_diag, color=plot_style.ORANGE, width=0.65)
ax2.axhline(1.0, ls=(0, (4, 2)), color=plot_style.INK, lw=1.2, alpha=0.4)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel('Mode index')
ax2.set_ylabel('MAC (true vs. inferred reconstruction)')
ax2.set_title(f"(b) Mode-shape agreement -- min={np.nanmin(mac_diag):.3f}, "
              f"mean={np.nanmean(mac_diag):.3f}", loc="left", fontsize=15, fontweight="bold",
              color=plot_style.INK, pad=8)

fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step9_fig4_case2_reconstruction')
print("Saved step9_fig4_case2_reconstruction.png")
