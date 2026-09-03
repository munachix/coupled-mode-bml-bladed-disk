# -*- coding: utf-8 -*-
"""
Health-indicator logic-flow diagram (Section 2.8).

Required by the project's figure brief as one of the four diagrams, and the
only one authored directly rather than handed to an image generator. It shows
the single point the paper actually argues: every health-monitoring output --
all three indicators, the severity classifier and the sparse localization --
is reached from the network trained once for forced-response prediction, with
no second model and no retraining anywhere on the path.

Output: figures/step8/step8_fig10_health_indicator_flow.png
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style  # noqa: E402

FIGS = os.path.join(ROOT, "figures", "step8")

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = plot_style.INK
MUTED = plot_style.INK_SECONDARY
C_NET = plot_style.C_1B      # the trained network -- the reused asset
C_IND = plot_style.C_OK      # the three health indicators
C_OUT = plot_style.C_ACC     # the two decisions
C_DAT = plot_style.INK       # observed data

fig, ax = plt.subplots(figsize=(9.4, 7.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

BOXES = {}


def box(key, x, y, w, h, title, sub=None, color=INK, fill="#ffffff", lw=1.6,
        title_size=15.5, sub_size=12.5, ls="solid"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.6",
        linewidth=lw, edgecolor=color, facecolor=fill, linestyle=ls, zorder=2))
    if sub:
        ax.text(x, y + h * 0.17, title, ha="center", va="center", fontsize=title_size,
                color=INK, fontweight="bold", zorder=3)
        ax.text(x, y - h * 0.25, sub, ha="center", va="center", fontsize=sub_size,
                color=MUTED, zorder=3)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=title_size,
                color=INK, fontweight="bold", zorder=3)
    BOXES[key] = (x, y, w, h)


def arrow(a, b, color=INK, ls="solid", lw=1.6, shrink=3.0,
          a_side="bottom", b_side="top", rad=0.0):
    ax_, ay, aw, ah = BOXES[a]
    bx, by, bw, bh = BOXES[b]
    pa = {"bottom": (ax_, ay - ah / 2), "top": (ax_, ay + ah / 2),
          "right": (ax_ + aw / 2, ay), "left": (ax_ - aw / 2, ay)}[a_side]
    pb = {"bottom": (bx, by - bh / 2), "top": (bx, by + bh / 2),
          "right": (bx + bw / 2, by), "left": (bx - bw / 2, by)}[b_side]
    ax.add_patch(FancyArrowPatch(
        pa, pb, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
        color=color, linestyle=ls, zorder=1,
        shrinkA=shrink, shrinkB=shrink,
        connectionstyle=f"arc3,rad={rad}"))


# ------------------------------------------------------------------ layout
box("data", 50, 94, 40, 8, "Response observations  D",
    "measured or simulated, possibly damaged", color=C_DAT)

box("net", 17, 76, 30, 10, "Trained coupled-mode BML",
    "Section 2.6 — trained once,\nfor forced-response prediction",
    color=C_NET, fill="#eef4f8", lw=2.2)

box("inf", 61, 76, 34, 10, "Bayesian mistuning inference",
    "Section 2.7, Eq. (25)", color=INK)

box("post", 61, 58, 34, 8, "Posterior mean state  μ$_{post}$",
    "inferred, not measured", color=INK)

box("hi1", 16, 40, 27, 9.5, "HI1", "peak 1B frequency\ndeviation, Eq. (28)", color=C_IND)
box("hi2", 50, 40, 27, 9.5, "HI2", "Mahalanobis distance in the\nlatent subspace, Eq. (29)", color=C_IND)
box("hi3", 84, 40, 27, 9.5, "HI3", "BML amplitude deviation\nfrom baseline, Eq. (27)", color=C_IND)

box("track", 30, 23, 36, 9.5, "Detection and tracking",
    "threshold crossing and trajectory,\nSections 3.3.1-3.3.2", color=C_OUT)
box("cls", 78, 23, 34, 9.5, "Severity classifier",
    "Section 2.9, Eq. (30)\n→ tuned / mistuned verdict", color=C_OUT)
box("loc", 50, 6, 44, 9.5, "Support-search localization",
    "Section 2.10, Eq. (31)\n→ damaged blade index", color=C_OUT)

# ------------------------------------------------------------------ arrows
arrow("data", "inf", b_side="top")
arrow("net", "inf", a_side="right", b_side="left", color=C_NET, lw=2.0)
arrow("inf", "post")
for k in ("hi1", "hi2", "hi3"):
    arrow("post", k, rad=0.0 if k == "hi2" else (0.12 if k == "hi3" else -0.12))
for k in ("hi1", "hi2", "hi3"):
    arrow(k, "track", rad=0.0 if k == "hi1" else (0.14 if k == "hi3" else 0.07))
# The classifier is not a function of the three indicators. It is HI3's own
# computation with the reference moved from the unit's own baseline to the ideal
# tuned disk, so it hangs off HI3 rather than off a feature vector.
arrow("hi3", "cls", color=C_OUT, lw=1.8)
ax.text(68.5, 32.0, "same computation,\nzero-mistuning reference", ha="right",
        va="center", fontsize=11, color=MUTED, style="italic", zorder=3)
# Section 2.10 frames localization as the step after a unit has been flagged.
arrow("track", "loc", rad=-0.10)
arrow("cls", "loc", rad=0.10)

# HI3 is the one indicator evaluated through the trained network itself. That is
# already carried by its own label and by the highlighted network box, so it
# gets no extra arrow: a connector from the far-left network box to the
# far-right indicator would have to cross the whole diagram.
ax.text(6, 49.6, "no retraining anywhere below this line",
        ha="left", va="center", fontsize=12.5, color=C_NET, style="italic", zorder=3)
ax.plot([4, 96], [47.4, 47.4], color=C_NET, lw=1.0, ls=(0, (2, 3)), zorder=0)

plot_style.figure_title(
    fig,
    "Health-indicator logic flow",
    "how one network trained for forced-response prediction reaches every "
    "health-monitoring output",
    x=0.01, y_title=1.005, y_subtitle=0.968)
fig.subplots_adjust(top=0.90, bottom=0.01, left=0.01, right=0.99)

plot_style.savefig_pub(fig, FIGS, "step8_fig10_health_indicator_flow")
print("Saved step8_fig10_health_indicator_flow.png")
