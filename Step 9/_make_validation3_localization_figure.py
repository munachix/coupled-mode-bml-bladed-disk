# -*- coding: utf-8 -*-
"""
Builds the real single-blade damage localization figure for Validation 3
(Section 3.6 of the paper), from the real ANSYS-injected damage result
already saved in Step 9/output/validation3_real_ansys_health_id.json
(blade 5, -4.5% severity).

v2 (2026-08-29, "makeover" per explicit user request -- the original flat
bar chart was unclear about which blade each model actually picks): redone
as a polar ring plot, matching this project's own established convention
for per-blade quantities (Step 3's mistuning-realization polar figures) --
the blisk's 24 blades really do sit on a physical ring, so a ring plot is
the natural (not merely decorative) way to show this.
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "output", "validation3_real_ansys_health_id.json")
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step9")


def main():
    with open(JSON_PATH) as f:
        d = json.load(f)

    NB = len(d["severities_all_blades_coupled"])
    true_blade = d["damaged_blade_true"]
    sev_diag = np.abs(np.array(d["severities_all_blades_diagonal"])) * 100.0
    sev_coup = np.abs(np.array(d["severities_all_blades_coupled"])) * 100.0
    loc_diag = d["localized_blade_diagonal"]
    loc_coup = d["localized_blade_coupled"]

    theta = np.linspace(0, 2 * np.pi, NB, endpoint=False)
    theta_c = np.append(theta, theta[0])
    sev_diag_c = np.append(sev_diag, sev_diag[0])
    sev_coup_c = np.append(sev_coup, sev_coup[0])

    plot_style.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw={"projection": "polar"})

    ax.plot(theta_c, sev_diag_c, color=plot_style.C_WARN, lw=2.0, marker="o", ms=5,
            alpha=0.85, label="diagonal-only model")
    ax.plot(theta_c, sev_coup_c, color=plot_style.C_1B, lw=2.0, marker="o", ms=5,
            alpha=0.9, label="coupled model")

    ax.scatter([theta[loc_diag]], [sev_diag[loc_diag]], s=260, facecolors="none",
               edgecolors=plot_style.C_WARN, linewidths=2.4, zorder=6,
               label=f"diagonal-only picks blade {loc_diag} (wrong)")
    ax.scatter([theta[loc_coup]], [sev_coup[loc_coup]], s=260, facecolors="none",
               edgecolors=plot_style.C_1B, linewidths=2.4, zorder=6,
               label=f"coupled picks blade {loc_coup} (correct)")
    ax.scatter([theta[true_blade]], [0], s=0)  # keep true-blade angle referenced for the radial line
    ax.plot([theta[true_blade], theta[true_blade]], [0, max(sev_diag.max(), sev_coup.max()) * 1.08],
            color=plot_style.INK, lw=2.0, ls="--", zorder=5,
            label=f"true damaged blade ({true_blade})")

    ax.set_xticks(theta)
    ax.set_xticklabels([str(b) for b in range(NB)], fontsize=15)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=14)
    # Title/subtitle removed from the image itself (2026-08-29, explicit
    # user request) -- the docx caption carries this text instead.
    # Horizontal legend below the plot (2026-08-30, explicit user request for
    # this figure specifically, reverting the inside-stacked placement used
    # elsewhere): the 5 entries read better as 2 rows below the ring than
    # stacked over any one quadrant of it.
    plot_style.legend_below(ax, ncol=2, y=-0.08)
    fig.tight_layout()
    paths = plot_style.savefig_pub(fig, FIGS_DIR, "step9_fig16_validation3_localization")
    print("Saved:", paths)


if __name__ == "__main__":
    main()
