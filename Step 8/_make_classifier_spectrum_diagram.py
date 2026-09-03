# -*- coding: utf-8 -*-
"""
Redesigns Fig. 24 (classifier population screen) as a horizontal
severity-spectrum diagram rather than a line plot (2026-09-01, explicit
user request): same real numbers as the line-plot version (real
crossover between 45% and 60% severity, real calibrated threshold), same
color scheme (green = tuned, red = mistuned), but presented as a filled
spectrum bar with zone shading and callouts rather than axes and a
scatter/line. Reads straight from the same real saved sweep
(Step 8/output/health_monitoring_sweep.json) used by the original figure
-- no new computation, only a new presentation of the same real result.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "output", "health_monitoring_sweep.json")
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step8")

with open(JSON_PATH) as f:
    d = json.load(f)
sweep_b = d["sweep_b"]
sev = np.array([s["severity_max"] for s in sweep_b]) * 100.0
verdicts = [s["classifier_final_verdict"] for s in sweep_b]

# Real crossover band: the last real "tuned" severity and the first real
# "MISTUNED" severity actually observed in the sweep (45% and 60%).
sev_tuned = sev[[not v for v in verdicts]]
sev_mistuned = sev[verdicts]
lo_thresh = float(sev_tuned.max())   # 45
hi_thresh = float(sev_mistuned.min())  # 60
sev_max = float(sev.max())  # 90

plot_style.apply_style()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11.5, 3.6))

BAR_Y0, BAR_Y1 = 0.30, 0.70
ax.axhspan(BAR_Y0, BAR_Y1, xmin=0, xmax=lo_thresh / sev_max, color=plot_style.C_OK, alpha=0.30, zorder=1)
ax.axhspan(BAR_Y0, BAR_Y1, xmin=lo_thresh / sev_max, xmax=hi_thresh / sev_max, color=plot_style.C_ACC, alpha=0.28, zorder=1)
ax.axhspan(BAR_Y0, BAR_Y1, xmin=hi_thresh / sev_max, xmax=1.0, color=plot_style.C_WARN, alpha=0.30, zorder=1)

# spectrum bar outline
ax.plot([0, sev_max], [BAR_Y0, BAR_Y0], color=plot_style.INK, lw=1.6, zorder=3)
ax.plot([0, sev_max], [BAR_Y1, BAR_Y1], color=plot_style.INK, lw=1.6, zorder=3)
ax.plot([0, 0], [BAR_Y0, BAR_Y1], color=plot_style.INK, lw=1.6, zorder=3)
ax.plot([sev_max, sev_max], [BAR_Y0, BAR_Y1], color=plot_style.INK, lw=1.6, zorder=3)

# threshold lines
for x, lbl in [(lo_thresh, f"{lo_thresh:.0f}%"), (hi_thresh, f"{hi_thresh:.0f}%")]:
    ax.plot([x, x], [BAR_Y0 - 0.06, BAR_Y1 + 0.06], color=plot_style.INK, ls="--", lw=2.0, zorder=4)
    ax.text(x, BAR_Y1 + 0.10, lbl, ha="center", va="bottom", fontsize=15, fontweight="bold", color=plot_style.INK)

# real severity sweep points across the bar (lower half of the band)
MARKER_Y = BAR_Y0 + 0.11
for s, v in zip(sev, verdicts):
    c = plot_style.C_WARN if v else plot_style.C_OK
    ax.scatter([s], [MARKER_Y], s=90, color=c, edgecolor=plot_style.SURFACE,
               linewidth=1.2, zorder=5)

# zone labels (upper half of the band, clear of the markers below)
LABEL_Y = BAR_Y1 - 0.10
ax.text(lo_thresh / 2, LABEL_Y, "TUNED", ha="center", va="center",
        fontsize=16, fontweight="bold", color=plot_style.INK, zorder=6)
ax.text((lo_thresh + hi_thresh) / 2, LABEL_Y, "CROSSOVER", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=plot_style.INK, zorder=6, rotation=0)
ax.text((hi_thresh + sev_max) / 2, LABEL_Y, "MISTUNED", ha="center", va="center",
        fontsize=16, fontweight="bold", color=plot_style.INK, zorder=6)

# callouts, placed well clear of the x-axis tick labels below
ax.annotate("score stays near\npopulation mean", xy=(lo_thresh * 0.35, BAR_Y0), xytext=(lo_thresh * 0.35, -0.20),
            ha="center", va="top", fontsize=12.5, color=plot_style.INK_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=plot_style.INK_MUTED, lw=1.2))
ax.annotate("crosses calibrated\nthreshold", xy=(hi_thresh, BAR_Y0), xytext=(hi_thresh + 12, -0.20),
            ha="center", va="top", fontsize=12.5, color=plot_style.INK_SECONDARY,
            arrowprops=dict(arrowstyle="-", color=plot_style.INK_MUTED, lw=1.2))

# blade 22 icon + sweep arrow, above the bar
ax.annotate("", xy=(sev_max, BAR_Y1 + 0.28), xytext=(0, BAR_Y1 + 0.28),
            arrowprops=dict(arrowstyle="-|>", color=plot_style.INK_SECONDARY, lw=2.0))
ax.text(0, BAR_Y1 + 0.34, "blade 22, severity swept", ha="left", va="bottom",
        fontsize=13, color=plot_style.INK_SECONDARY, style="italic")
ax.scatter([0], [BAR_Y1 + 0.28], s=260, color=plot_style.SURFACE, edgecolor=plot_style.INK, linewidth=1.6, zorder=7)
ax.text(0, BAR_Y1 + 0.28, "22", ha="center", va="center", fontsize=11, fontweight="bold", color=plot_style.INK, zorder=8)

# x tick labels along the bottom (severity %), no y axis at all
for xt in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    ax.text(xt, BAR_Y0 - 0.42, f"{xt}", ha="center", va="top", fontsize=12, color=plot_style.INK_SECONDARY)
ax.text(sev_max / 2, BAR_Y0 - 0.58, "Injected severity (max extra fractional-frequency loss, %)",
        ha="center", va="top", fontsize=14, color=plot_style.INK)

ax.set_xlim(-2, sev_max + 2)
ax.set_ylim(-0.68, BAR_Y1 + 0.55)
ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
plot_style.savefig_pub(fig, FIGS_DIR, "step8_fig9b_classifier_severity_sweep")
print(f"Saved. thresholds: {lo_thresh:.0f}% - {hi_thresh:.0f}%")
