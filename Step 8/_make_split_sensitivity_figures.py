# -*- coding: utf-8 -*-
"""
Regenerates the health-monitoring sensitivity sweep as two separate,
standalone figures (real data from Step 8/output/health_monitoring_sweep.json)
instead of the old step8_fig9 side-by-side composite, per
PAPER_INSTRUCTIONS.md's "no forced multi-panel composites" rule.

Fig A: localization residual heatmap across 8 independent damaged-blade
       scenarios (Sweep A, fixed 15% severity).
Fig B: classifier population-screen score vs. injected severity (Sweep B,
       fixed blade 22, severity swept 5%-30%).
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "output", "health_monitoring_sweep.json")
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step8")


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)

    plot_style.apply_style()

    # ---- Fig A: localization across 8 fault locations ----
    sweep_a = d["sweep_a"]
    NB = len(sweep_a[0]["localization_residuals"])
    n_scen = len(sweep_a)
    mat = np.array([s["localization_residuals"] for s in sweep_a])
    labels = [f"blade {s['damaged_blade']} damaged" for s in sweep_a]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(mat, aspect="auto", cmap=plot_style.SEQ_CMAP,
                    norm=LogNorm(vmin=mat.min(), vmax=mat.max()))
    for row, s in enumerate(sweep_a):
        col = s["localized_blade"]
        ax.scatter([col], [row], s=220, facecolors="none",
                   edgecolors=plot_style.INK, linewidths=1.8, zorder=5)
    ax.set_yticks(range(n_scen))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(0, NB, 2))
    ax.set_xlabel("Candidate blade index")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Support-search residual")
    n_exact = sum(1 for s in sweep_a if s["ring_distance"] == 0)
    plot_style.two_tier_title(
        ax, "Localization across independent fault locations",
        f"Sweep A, 15% severity, {n_exact}/{n_scen} exact -- true blade (ringed) is the row minimum every time"
    )
    fig.tight_layout()
    paths_a = plot_style.savefig_pub(fig, FIGS_DIR, "step8_fig9a_localization_sweep")
    print("Saved:", paths_a)

    # ---- Fig B: classifier population screen vs. severity ----
    sweep_b = d["sweep_b"]
    sev = np.array([s["severity_max"] for s in sweep_b]) * 100.0
    score_um = np.array([s["classifier_final_score_mm"] for s in sweep_b]) * 1000.0
    threshold_um = d["classifier_calib"]["threshold"] * 1000.0
    mean_um = d["classifier_calib"]["mean"] * 1000.0
    verdicts = [s["classifier_final_verdict"] for s in sweep_b]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhline(threshold_um, color=plot_style.C_ACC, lw=1.8, ls="--",
               label="calibrated threshold")
    ax.axhline(mean_um, color=plot_style.INK_MUTED, lw=1.2, ls=":",
               label="population mean")
    colors = [plot_style.C_WARN if v else plot_style.C_OK for v in verdicts]
    ax.plot(sev, score_um, color=plot_style.INK_SECONDARY, lw=1.4, zorder=3)
    ax.scatter(sev, score_um, s=70, color=colors, zorder=4,
               edgecolor=plot_style.SURFACE, linewidth=1.0)
    ax.scatter([], [], s=70, color=plot_style.C_OK, label="verdict: tuned")
    ax.scatter([], [], s=70, color=plot_style.C_WARN, label="verdict: MISTUNED")
    ax.set_xlabel("Injected severity (max extra fractional-frequency loss, %)")
    ax.set_ylabel("Classifier score, distance from perfect tuning (µm)")
    ax.set_ylim(0, max(threshold_um, score_um.max()) * 1.08)
    plot_style.two_tier_title(
        ax, "Classifier population screen (not localization)",
        "blade 22, severity swept 5%-90% -- score stays near the population "
        "mean at manufacturing-realistic severities, crosses threshold between 45% and 60%"
    )
    plot_style.legend_inside(ax, loc='upper left', fontsize=13)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.14)
    out_path = os.path.join(FIGS_DIR, "step8_fig9b_classifier_severity_sweep.png")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.15, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved:", [out_path])


if __name__ == "__main__":
    main()
