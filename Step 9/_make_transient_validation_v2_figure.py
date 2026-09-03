# -*- coding: utf-8 -*-
"""Figure for the redesigned Section 3.5.3 (v2): three real, independent
time-domain views -- (a) the real ANSYS transient's own build-up to a
converged limit cycle, (b) an independent from-rest time integration of
the same real coupled nonlinear ODEs (Compact ROM, no fitting to the
ANSYS data at all) building up to its own converged limit cycle, and (c)
their converged tails overlaid (cycle-aligned for display only -- both
periodic signals' own absolute time origins are arbitrary and physically
incomparable across two separate simulations, so aligning by cycle phase
is the standard way to display two independently-converged periodic
waveforms side by side) alongside the compositional BML's own steady-
state amplitude prediction as a reference band, since the BML does not
integrate in time and has no waveform of its own to overlay."""
import os
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

d = np.load(os.path.join(OUT, "transient_validation_v2.npz"))
t_eval, u_pred = d["t_eval"], d["u_pred"]
t_fem, u_fem = d["t_fem"], d["u_fem"]
t_tail_pred, u_tail_pred = d["t_tail_pred"], d["u_tail_pred"]
t_tail_fem, u_tail_fem = d["t_tail_fem"], d["u_tail_fem"]
pred_tail_amp = float(d["pred_tail_amp"])
fem_tail_amp = float(d["fem_tail_amp"])
amp_err_pct = float(d["amp_err_pct"])

bml_amp = 1.1449  # Section 3.5.1's own validated compositional BML prediction at this exact point
bml_err_pct = abs(bml_amp - fem_tail_amp) / fem_tail_amp * 100.0

# cycle-align the two independently-converged tails for display (phase
# shift only, chosen so each tail's own first peak sits at the same
# relative position -- does not alter either signal's amplitude or shape)
t_tail_pred_disp = t_tail_pred - t_tail_pred[np.argmax(u_tail_pred)]
t_tail_fem_disp = t_tail_fem - t_tail_fem[np.argmax(u_tail_fem)]

plot_style.apply_style()
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})

ax = axes[0]
ax.plot(t_fem, u_fem, color=plot_style.BLUE, lw=0.9, alpha=0.85)
ax.axhspan(-fem_tail_amp, fem_tail_amp, color=plot_style.BLUE, alpha=0.08)
ax.set_xlabel("Time  [s]")
ax.set_ylabel("U$_Z$ at node 1171  [mm]")
ax.text(0.03, 0.96, "(a) Real ANSYS transient", transform=ax.transAxes, fontsize=14,
        fontweight="bold", color=plot_style.BLUE, va="top",
        bbox=dict(facecolor=plot_style.SURFACE, edgecolor="none", alpha=0.85, pad=2))

ax = axes[1]
ax.plot(t_eval, u_pred, color=plot_style.C_OK, lw=0.7, alpha=0.85)
ax.axhspan(-pred_tail_amp, pred_tail_amp, color=plot_style.C_OK, alpha=0.08)
ax.set_xlabel("Time  [s]")
ax.set_yticklabels([])
ax.text(0.03, 0.96, "(b) Time-integrated Compact ROM\n(from rest, no fitting)",
        transform=ax.transAxes, fontsize=13, fontweight="bold", color=plot_style.C_OK, va="top",
        bbox=dict(facecolor=plot_style.SURFACE, edgecolor="none", alpha=0.85, pad=2))

ax = axes[2]
ax.plot(t_tail_fem_disp, u_tail_fem, "o-", color=plot_style.BLUE, lw=1.6, ms=4,
        mec=plot_style.SURFACE, mew=0.6, label=f"Real ANSYS ({fem_tail_amp:.3f} mm)")
ax.plot(t_tail_pred_disp, u_tail_pred, "s--", color=plot_style.C_OK, lw=1.6, ms=3.5,
        mec=plot_style.SURFACE, mew=0.6, label=f"Compact ROM ({pred_tail_amp:.3f} mm, {amp_err_pct:.1f}% err.)")
ax.axhline(bml_amp, color=plot_style.VIOLET, ls=":", lw=2.0,
           label=f"BML steady-state ({bml_amp:.3f} mm, {bml_err_pct:.1f}% err.)")
ax.axhline(-bml_amp, color=plot_style.VIOLET, ls=":", lw=2.0)
ax.set_xlabel("Time (cycle-aligned)  [s]")
ax.text(0.03, 0.96, "(c) Converged tails, overlaid", transform=ax.transAxes, fontsize=14,
        fontweight="bold", color=plot_style.INK, va="top",
        bbox=dict(facecolor=plot_style.SURFACE, edgecolor="none", alpha=0.85, pad=2))
plot_style.legend_inside(ax, loc="lower center", fontsize=10.5)

plot_style.figure_title(fig, "Real transient validation: node 1171, w = 1.0",
                         "real ANSYS FEM vs. an independent, from-rest time-integrated physics model "
                         "vs. the BML's own steady-state prediction -- no fitting anywhere",
                         y_title=1.04, y_subtitle=0.985)
fig.subplots_adjust(top=0.80, wspace=0.12)
plot_style.savefig_pub(fig, FIGS, "step9_fig25_transient_validation_v2")
print("Saved step9_fig25_transient_validation_v2.png")
print(f"BML error: {bml_err_pct:.2f}%  |  Compact ROM error: {amp_err_pct:.2f}%")
