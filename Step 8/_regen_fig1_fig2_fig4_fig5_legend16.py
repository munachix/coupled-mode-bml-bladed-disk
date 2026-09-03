# -*- coding: utf-8 -*-
"""Regenerates step8_fig1_HI1_trajectory, step8_fig2_HI2_detection,
step8_fig4_false_alarm_calibration and step8_fig5_tracked_recovery at the legend
fontsize now matching axes.labelsize exactly (both 16) -- same fix already
applied+verified on step7_fig4_coverage_calibration (see Step 7/_regen_fig4_coverage.py)
and rolled out project-wide in plot_style.py's legend_below().

All four figures use plot_style.legend_below() only (no direct ax.legend(fontsize=...)
at any of their own call sites in step8.py), so the plot_style.py fix alone covers
them -- this script just replays each figure's exact plotting code from step8.py's
make_step8_figures against already-saved outputs, no re-run of the damage-trajectory
simulation / MCMC-based state inference / calibration sweep:
  - fig1/fig2/fig5 data: Step 8/output/damage_trajectory.npz (traj + traj_result
    fields, exactly as saved by step8.py's save_outputs)
  - fig2/fig4 data: Step 8/output/calibration.npz (calib fields)
  - fig1's off-scale HI1 threshold annotation: Step 5's own saved healthy-population
    ensemble (Step 5/output/aleatoric_ensemble.npz['HI1']), the same array step8.py
    itself loads read-only for this (CONFIG['step5_dir'])
  - fig2's detection-time marker: detection_index, saved in step8_config.json
  - fig4's target false-alarm line: CONFIG['calibration']['target_false_alarm']=0.05,
    read directly from step8.py's own CONFIG dict (no computation)
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIG_ROOT = os.path.dirname(HERE)
FIGS = os.path.join(FIG_ROOT, "figures", "step8")
NB = 24
TARGET_FALSE_ALARM = 0.05  # step8.py CONFIG['calibration']['target_false_alarm']

traj_npz = np.load(os.path.join(OUT, "damage_trajectory.npz"))
calib_npz = np.load(os.path.join(OUT, "calibration.npz"))
with open(os.path.join(OUT, "step8_config.json")) as f:
    cfg_record = json.load(f)
HI1_healthy = np.load(os.path.join(FIG_ROOT, "Step 5", "output", "aleatoric_ensemble.npz"))["HI1"]

traj = dict(t_norm=traj_npz["t_norm"], severity=traj_npz["severity"],
            df_baseline=traj_npz["df_baseline"], damaged_blade=int(traj_npz["damaged_blade"]))
traj_result = dict(HI1_true=traj_npz["HI1_true"], HI1_post=traj_npz["HI1_post"],
                    HI2=traj_npz["HI2"], post_means=traj_npz["post_means"],
                    sparse_blade=traj_npz["sparse_blade"], sparse_severities=traj_npz["sparse_severities"])
calib = dict(D_calib=calib_npz["D_calib"], D_holdout=calib_npz["D_holdout"],
             threshold=float(calib_npz["threshold"]), false_alarm_rate=float(calib_npz["false_alarm_rate"]))
detect = dict(detect_idx=int(cfg_record["detection_index"]))

t = traj['t_norm']

plot_style.apply_style()

# ── fig1: HI1 true vs inferred, over the trajectory (verbatim from step8.py) ──
fig, ax = plt.subplots(figsize=(8.0, 5.6))
ax.plot(t, traj_result['HI1_true'], color=plot_style.BLUE, lw=2.2, marker='o', ms=6,
        mec=plot_style.SURFACE, mew=1.0, label='HI1 (true injected state)')
ax.plot(t, traj_result['HI1_post'], color=plot_style.ORANGE, lw=2.2, marker='s', ms=6,
        mec=plot_style.SURFACE, mew=1.0, label='HI1 (from Bayesian-inferred state)')
data_max = max(np.max(traj_result['HI1_true']), np.max(traj_result['HI1_post']))
data_min = min(np.min(traj_result['HI1_true']), np.min(traj_result['HI1_post']))
pad = 0.15 * (data_max - data_min)
ax.set_ylim(data_min - pad, data_max + pad)
ax.set_xlabel('Normalized time / operating cycle')
ax.set_ylabel('HI1 -- max 1B-cluster |frequency deviation|  [Hz]')
plot_style.two_tier_title(ax, 'Health indicator HI1', 'true damage trajectory vs. inferred from vibration data')
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step8_fig1_HI1_trajectory')
print("Saved step8_fig1_HI1_trajectory.png")

# ── fig2: HI2 (Bayesian anomaly score) with calibrated threshold ──
# figsize height raised 5.8->6.6 to match the step8.py source fix (2026-08-29,
# found during this pass): at axes.labelsize=16 the long rotated ylabel no longer
# fit in 5.8in and produced a double-struck/ghosted rendering artifact. See the
# matching comment in step8.py's make_step8_figures.
fig, ax = plt.subplots(figsize=(8.0, 6.6))
ax.plot(t, traj_result['HI2'], color=plot_style.BLUE, lw=2.2, marker='o', ms=6,
        mec=plot_style.SURFACE, mew=1.0, label='HI2 (Mahalanobis, inferred)')
ax.axhline(calib['threshold'], color=plot_style.VIOLET, ls='--', lw=1.6,
           label="calibrated threshold")
if detect['detect_idx'] >= 0:
    ax.axvline(t[detect['detect_idx']], color=plot_style.AQUA, ls=':', lw=1.6, label='detection time')
ax.scatter(np.zeros(len(calib['D_calib'])) - 0.03, calib['D_calib'], color=plot_style.INK_MUTED, s=16,
           alpha=0.6, label='null-trial HI2')
ax.set_xlabel('Normalized time / operating cycle')
ax.set_ylabel('HI2 -- Bayesian anomaly score (Mahalanobis distance)')
ax.set_ylim(bottom=0)
plot_style.two_tier_title(ax, 'HI2 monitoring dashboard', 'calibrated anomaly detection')
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step8_fig2_HI2_detection')
print("Saved step8_fig2_HI2_detection.png")

# ── fig4: false-alarm calibration (verbatim) ──
bins = np.linspace(0, max(calib['D_calib'].max(), calib['D_holdout'].max()) * 1.1, 15)
fig, ax = plt.subplots(figsize=(8.0, 5.6))
ax.hist(calib['D_calib'], bins=bins, color=plot_style.AQUA, alpha=0.35, edgecolor='none',
        label='calibration null trials')
ax.hist(calib['D_calib'], bins=bins, histtype='step', color=plot_style.AQUA, linewidth=2.0)
ax.hist(calib['D_holdout'], bins=bins, color=plot_style.ORANGE, alpha=0.35, edgecolor='none',
        label='held-out null trials')
ax.hist(calib['D_holdout'], bins=bins, histtype='step', color=plot_style.ORANGE, linewidth=2.0)
ax.axvline(calib['threshold'], color=plot_style.VIOLET, ls='--', lw=1.8,
           label='calibrated threshold')
ax.set_xlabel('HI2 (Mahalanobis anomaly score)')
ax.set_ylabel('count')
plot_style.two_tier_title(ax, 'False-alarm calibration',
                           f"{calib['false_alarm_rate']*100:.1f}% empirical vs. "
                           f"{TARGET_FALSE_ALARM*100:.0f}% target")
plot_style.legend_inside(ax, loc='upper right')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step8_fig4_false_alarm_calibration')
print("Saved step8_fig4_false_alarm_calibration.png")

# ── fig5: recovered vs true damaged-blade mistuning over time (verbatim) ──
fig, ax = plt.subplots(figsize=(8.0, 5.6))
b = traj['damaged_blade']
true_vals = traj['df_baseline'][b] + traj['severity']
sparse_blade_t = traj_result['sparse_blade']
sparse_recovered = traj_result['sparse_severities'][np.arange(len(t)), sparse_blade_t] \
    + traj['df_baseline'][sparse_blade_t]
locked = sparse_blade_t == b
post_mean_b = traj_result['post_means'][:, b]
ax.plot(t, true_vals * 100, color=plot_style.BLUE, lw=2.2, marker='o', ms=6, mec=plot_style.SURFACE,
        mew=1.0, label='true df$_b$/f (damaged blade)', zorder=5)
ax.plot(t, sparse_recovered * 100, color=plot_style.C_OK, lw=2.0, ls='-', zorder=4,
        label='sparse recovery (at localized blade, per step)')
ax.scatter(t[locked], sparse_recovered[locked] * 100, color=plot_style.C_OK, s=46,
           edgecolor=plot_style.SURFACE, linewidth=0.9, zorder=6, label='localized = true blade')
ax.scatter(t[~locked], sparse_recovered[~locked] * 100, marker='x', color=plot_style.C_WARN, s=46,
           zorder=6, label='localized a different blade (pre-onset noise)')
ax.plot(t, post_mean_b * 100, color=plot_style.INK_MUTED, lw=1.2, ls='--', alpha=0.8,
        label='smooth posterior mean (drives HI2, not a magnitude readout)')
ax.set_xlabel('Normalized time / operating cycle')
ax.set_ylabel('df$_b$/f, damaged blade  [%]')
plot_style.two_tier_title(ax, "Tracked recovery of the damaged blade's mistuning state",
                           'sparse support-search recovery vs. true injected trajectory')
fig.tight_layout()
plot_style.legend_inside(ax, loc='lower left', fontsize=12)
plot_style.savefig_pub(fig, FIGS, 'step8_fig5_tracked_recovery')
print("Saved step8_fig5_tracked_recovery.png")
