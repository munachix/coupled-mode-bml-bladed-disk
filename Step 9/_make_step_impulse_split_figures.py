# -*- coding: utf-8 -*-
"""
Split Section 3.5.3's step/impulse figure into two standalone figures
(2026-09-02 audit): the v2 figure conjoined the step and impulse panels
side by side in one wide image, which the project's own figure rules
forbid ("do not force two plots side-by-side into one wide image; split
them and stack the two images vertically instead"). The v2 composite also
pushed each panel's legend into a strip below the axes, where the
"real ANSYS" marker entry read as a bare label, and its panel (a) tag
collided with the y-axis label.

This script re-plots, without re-solving anything, from the arrays the v2
run already saved to Step 9/output/step_impulse_ansys_verified.npz, so
the plotted curves are byte-identical to the validated ones. Each figure
carries its own two-tier title, its own inside-axes legend, and its own
measured error statistics.

Outputs:
    figures/step9/step9_fig28a_step_response.png
    figures/step9/step9_fig28b_impulse_response.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style  # noqa: E402

FIGS = os.path.join(ROOT, "figures", "step9")
OUT = os.path.join(HERE, "output")

d = np.load(os.path.join(OUT, "step_impulse_ansys_verified.npz"))

t_step = d["t_step"]
u_step_ansys = d["u_step_ansys"]
u_step_70 = d["u_step_70mode"]
u_step_m0 = d["u_step_mode0"]
t_imp = d["t_imp"]
u_imp_ansys = d["u_imp_ansys"]
u_imp_70 = d["u_imp_70mode"]
u_imp_m0 = d["u_imp_mode0"]

peak_step = float(np.abs(u_step_ansys).max())
peak_imp = float(np.abs(u_imp_ansys).max())
rmse_70 = float(d["rmse_70"])
rmse_70_imp = float(d["rmse_70_imp"])
corr_70 = float(d["corr_70"])
corr_70_imp = float(d["corr_70_imp"])

# mode-0-only share of the real peak, and the 70-mode peak error, exactly as
# Section 3.5.3 reports them
share_m0_step = float(np.abs(u_step_m0).max() / peak_step)
share_m0_imp = float(np.abs(u_imp_m0).max() / peak_imp)
peak_err_70_step = float((np.abs(u_step_70).max() - peak_step) / peak_step)
peak_err_70_imp = float((np.abs(u_imp_70).max() - peak_imp) / peak_imp)

print("=== Section 3.5.3 split step/impulse figures ===", flush=True)
print(f"  STEP    : real peak {peak_step:.3f} mm | 70-mode RMSE {rmse_70:.4f} mm "
      f"({100 * rmse_70 / peak_step:.1f}% of peak), corr {corr_70:.3f}, "
      f"peak err {100 * peak_err_70_step:+.1f}% | mode-0-only captures "
      f"{100 * share_m0_step:.1f}% of peak ({np.abs(u_step_m0).max():.3f} mm)", flush=True)
print(f"  IMPULSE : real peak {peak_imp:.3f} mm | 70-mode RMSE {rmse_70_imp:.4f} mm "
      f"({100 * rmse_70_imp / peak_imp:.1f}% of peak), corr {corr_70_imp:.3f}, "
      f"peak err {100 * peak_err_70_imp:+.1f}% | mode-0-only captures "
      f"{100 * share_m0_imp:.1f}% of peak ({np.abs(u_imp_m0).max():.3f} mm)", flush=True)

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402


def make(t, u_ansys, u_70, title, subtitle, name, zero_line, legend_loc):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    if zero_line:
        ax.axhline(0, color=plot_style.INK_MUTED, lw=0.8, zorder=1)
    # These transients are sampled far too densely to read as discrete points:
    # white-filled markers drawn over the model curve punched holes in it rather
    # than reading as measurements, and at this sample rate they could not track
    # the impulse case's oscillation at all. Both traces are therefore drawn as
    # lines -- the real ANSYS solution as the solid dark reference, the model as
    # a coloured overlay -- which is how the agreement is actually judged here.
    ax.plot(t * 1000, u_ansys, color=plot_style.INK, lw=2.4, zorder=2,
            label="full-order finite-element solution")
    ax.plot(t * 1000, u_70, color=plot_style.C_1B, lw=1.6, ls=(0, (5, 2)), zorder=3,
            label="70-mode reduced-order model")
    ax.set_xlabel("Time  [ms]")
    ax.set_ylabel("U$_Z$ at node 1171  [mm]")
    # Headroom so the inside legend never sits on top of the traces.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.42 * (hi - lo))
    plot_style.two_tier_title(ax, title, subtitle)
    plot_style.legend_inside(ax, loc=legend_loc, fontsize=13.5,
                             markerscale=1.1, handlelength=2.2)
    plot_style.savefig_pub(fig, FIGS, name)
    print(f"Saved {name}.png", flush=True)


make(t_step, u_step_ansys, u_step_70,
     "Step response against the full-order solution",
     f"70-mode reduced-order model: RMSE {100 * rmse_70 / peak_step:.1f}% of peak, "
     f"correlation {corr_70:.3f}, peak within {abs(100 * peak_err_70_step):.1f}%",
     "step9_fig28a_step_response", zero_line=False, legend_loc="upper right")

make(t_imp, u_imp_ansys, u_imp_70,
     "Impulse response against the full-order solution",
     f"70-mode reduced-order model: RMSE {100 * rmse_70_imp / peak_imp:.1f}% of peak, "
     f"correlation {corr_70_imp:.3f}, peak within {abs(100 * peak_err_70_imp):.1f}%",
     "step9_fig28b_impulse_response", zero_line=True, legend_loc="upper right")

print("Done.", flush=True)
