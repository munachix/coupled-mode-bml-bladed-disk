# -*- coding: utf-8 -*-
"""
Real classifier-vs-severity trajectory figure (Section 3.4.2), giving the
tuned/mistuned severity classifier its own dedicated result -- previously
only cited as a single correlation number (r = -0.980) in the abstract,
with no figure anywhere in Section 3. Replays step8.py's own validated
check (validate_classifier_on_trajectory, around line 1064) standalone:
evaluates the classifier on the Bayesian-INFERRED state at every step of
the real damage trajectory (Step 8/output/damage_trajectory.npz's own
saved post_means), exactly as the live pipeline does -- no retraining,
no recomputation of the trajectory itself.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import step8 as s8
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step8")

print("=== Classifier-vs-real-damage-trajectory check (standalone replay) ===", flush=True)

traj_npz = np.load(os.path.join(OUT, "damage_trajectory.npz"))
t = traj_npz["t_norm"]
severity = traj_npz["severity"]
post_means = traj_npz["post_means"]

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
mclf = s8.calibrate_mistuning_classifier(inp, df_all, models, pairs, chain)

torch.manual_seed(s8.CONFIG['random_seed'] + 61_000)
T = len(t)
scores = np.zeros(T)
sigmas = np.zeros(T)
verdicts = np.zeros(T, dtype=bool)
for i in range(T):
    result = s8.classify_mistuning(post_means[i], inp, models, pairs, chain, mclf)
    scores[i] = result['score']
    sigmas[i] = result['sigma']
    verdicts[i] = result['is_mistuned']

# Correlation is scale-invariant, so corr(severity, score) == corr(severity,
# sigma) exactly (sigma is just an affine rescaling of score); computed on
# sigma to match the number already cited in the abstract (-0.980) and
# step8.py's own docstring history. The PLOT below uses raw score (um),
# not sigma, since score is what's directly comparable to the threshold
# line (mclf['threshold'] and result['score'] are the same raw units;
# sigma is a separate standardized quantity and is not comparable to the
# threshold without first converting it back).
corr = float(np.corrcoef(severity, sigmas)[0, 1])
threshold_um = mclf['threshold'] * 1000.0
score_um = scores * 1000.0
print(f"  corr(severity, classifier sigma) = {corr:.4f}", flush=True)
print(f"  t=0 verdict: {'MISTUNED' if verdicts[0] else 'tuned'}", flush=True)
print(f"  final verdict: {'MISTUNED' if verdicts[-1] else 'tuned'}, score={score_um[-1]:.2f} um, "
      f"threshold={threshold_um:.2f} um", flush=True)
first_trip = np.where(verdicts)[0]
print(f"  first MISTUNED at step: {int(first_trip[0]) if len(first_trip) else 'never'}", flush=True)

np.savez(os.path.join(OUT, "classifier_trajectory_check.npz"),
         t=t, severity=severity, scores=scores, sigmas=sigmas, verdicts=verdicts, corr=corr,
         threshold_um=threshold_um)

# ---- figure ----
plot_style.apply_style()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.0, 5.6))
colors = [plot_style.C_WARN if v else plot_style.C_OK for v in verdicts]
ax.plot(t, score_um, color=plot_style.INK_SECONDARY, lw=1.6, zorder=3)
ax.scatter(t, score_um, s=60, color=colors, zorder=4, edgecolor=plot_style.SURFACE, linewidth=0.9)
ax.axhline(threshold_um, color=plot_style.C_ACC, lw=1.6, ls='--', label='calibrated threshold')
ax.scatter([], [], s=60, color=plot_style.C_OK, label='verdict: tuned')
ax.scatter([], [], s=60, color=plot_style.C_WARN, label='verdict: MISTUNED')
ax.set_ylim(0, max(score_um.max(), threshold_um) * 1.1)
ax.set_xlabel('Normalized time / operating cycle')
ax.set_ylabel('Classifier score, distance from perfect tuning ($\\mu$m)')
plot_style.two_tier_title(ax, 'Classifier vs. real damage trajectory',
                           f'inferred state, same trajectory as HI1-HI3 -- corr(severity, score) = {corr:.3f}')
fig.tight_layout()
plot_style.legend_inside(ax, loc='upper left')
plot_style.savefig_pub(fig, FIGS, 'step8_fig9c_classifier_trajectory')
print("Saved step8_fig9c_classifier_trajectory.png", flush=True)
