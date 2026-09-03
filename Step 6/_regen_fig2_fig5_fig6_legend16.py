# -*- coding: utf-8 -*-
"""Regenerates step6_fig2_predicted_vs_true, step6_fig5_calibration and
step6_fig6_resonance_frequency at the legend fontsize now matching axes.labelsize
exactly (both 16) -- same fix already applied+verified on
step7_fig4_coverage_calibration and rolled out project-wide in plot_style.py's
legend_below(). All three figures use plot_style.legend_below() only (no direct
ax.legend(fontsize=...) at their own call sites in step6.py), so the plot_style.py
fix alone covers them.

fig2 and fig5 need only the ALREADY-SAVED held-out predictions
(Step 6/output/bpinn_predictions.npz + step6_config.json's test_r2/test_rmse) --
no recomputation at all, exact same saved numbers.

fig6 additionally needs, per test sample, a dense predict_mc() sweep over w to find
the network's own reconstructed resonance peak -- this was never saved (only the
sparse observed points were). Reconstructing it requires the trained model (loaded
from bpinn_state.pt, not retrained) plus the exact held-out test set, which is NOT
saved as an array but IS fully deterministic: step6.py seeds rng=default_rng(42) once,
draws one permutation, then calls build_dataset(inp, train_idx, rng) followed by
build_dataset(inp, test_idx, rng) using that SAME rng instance in that SAME order
(step6.py's own __main__ block). Replaying that identical call sequence here
reproduces the identical test_rows (verified below against the saved w/true_amp/
sample_idx arrays) and the identical Feat_mean/Feat_std normalization the model was
trained with -- all of it closed-form/deterministic, not a network re-run. Only the
new dense-grid predict_mc() calls do any model inference, and that's cheap forward-
pass evaluation (no training) on a network that already exists on disk.
Timed: load_inputs + build_dataset for all 700 train+test samples takes ~2s total
(see Step 6/_time_test_build_dataset.py), not an 'expensive multi-stage' re-run.
"""
import os, sys, time
import numpy as np
import torch
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step6 as s6
import plot_style

FIG_ROOT = os.path.dirname(HERE)
FIGS = os.path.join(FIG_ROOT, "figures", "step6")
OUT = s6.OUT

t0 = time.time()

# ── fig2 / fig5: reload the already-saved held-out predictions verbatim ──
pred = np.load(os.path.join(OUT, "bpinn_predictions.npz"))
import json
with open(os.path.join(OUT, "step6_config.json")) as f:
    cfg_record = json.load(f)
val = dict(w=pred['w'], true_amp=pred['true_amp'], amp_mean=pred['amp_mean'],
           amp_std=pred['amp_std'], sample_idx=pred['sample_idx'],
           r2=cfg_record['test_r2'], rmse=cfg_record['test_rmse'])

plot_style.apply_style()

# ── fig2: predicted vs true amplitude (verbatim from step6.py make_step6_figures) ──
fig, ax = plt.subplots(figsize=(6.5, 6.3))
ax.errorbar(val['true_amp'], val['amp_mean'], yerr=2 * val['amp_std'], fmt='o', ms=5,
            color=plot_style.ORANGE, ecolor=plot_style.ORANGE, alpha=0.5, elinewidth=0.9,
            capsize=0, mec=plot_style.SURFACE, mew=0.9)
lims = [0, max(val['true_amp'].max(), val['amp_mean'].max()) * 1.05]
ax.plot(lims, lims, color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect prediction')
ax.set_xlabel('True amplitude  [mm]')
ax.set_ylabel('BML-predicted amplitude  [mm]  (+/-2 sigma)')
ax.set_xlim(val['true_amp'].min(), val['true_amp'].max())
ax.set_ylim((val['amp_mean'] - 2 * val['amp_std']).min(), (val['amp_mean'] + 2 * val['amp_std']).max())
plot_style.two_tier_title(ax, 'Nonlinear response amplitude',
                           f"BML vs. exact HBM continuation -- R2={val['r2']:.3f}, RMSE={val['rmse']:.4f} mm")
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step6_fig2_predicted_vs_true')
print("Saved step6_fig2_predicted_vs_true.png")

# ── fig5: calibration (reliability) curve (verbatim) ──
fig, ax = plt.subplots(figsize=(6.5, 5.6))
nominal_levels = np.linspace(0.1, 0.99, 15)
from scipy.stats import norm as _norm
empirical = []
z = np.abs(val['true_amp'] - val['amp_mean']) / (val['amp_std'] + 1e-8)
for p in nominal_levels:
    k = _norm.ppf(0.5 + p / 2)
    empirical.append(float(np.mean(z <= k)))
ax.plot(nominal_levels, empirical, 'o-', color=plot_style.ORANGE, lw=2.0, ms=6,
        mec=plot_style.SURFACE, mew=1.0, label='BML calibration')
ax.plot([0, 1], [0, 1], color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect calibration')
ax.set_xlabel('Nominal credible level')
ax.set_ylabel('Empirical coverage (held-out test set)')
empirical_arr = np.array(empirical)
ax.set_xlim(nominal_levels.min(), nominal_levels.max())
ax.set_ylim(empirical_arr.min(), empirical_arr.max())
plot_style.two_tier_title(ax, 'Predictive-uncertainty calibration', 'reliability diagram')
plot_style.legend_inside(ax, loc='lower right', fontsize=12)
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step6_fig5_calibration')
print("Saved step6_fig5_calibration.png")

# ── fig6: reconstruct the exact held-out test set + reload the trained model ──
inp = s6.load_inputs()
rng = np.random.default_rng(s6.CONFIG['random_seed'])
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:s6.CONFIG['n_train_samples']]
test_idx = perm[s6.CONFIG['n_train_samples']:s6.CONFIG['n_train_samples'] + s6.CONFIG['n_test_samples']]
train_rows = s6.build_dataset(inp, train_idx, rng)
test_rows = s6.build_dataset(inp, test_idx, rng)
print(f"  Rebuilt train/test datasets in {time.time() - t0:.2f}s")

labeled = [r for r in test_rows if r['alpha'] is not None]
w_check = np.array([r['w'] for r in labeled])
true_amp_check = np.array([r['amplitude'] for r in labeled])
sidx_check = np.array([r['sample_idx'] for r in labeled])
assert np.allclose(w_check, val['w']), "reconstructed test set w does not match saved predictions -- ordering/seed mismatch"
assert np.allclose(true_amp_check, val['true_amp']), "reconstructed test set true_amp does not match saved predictions"
assert np.array_equal(sidx_check, val['sample_idx']), "reconstructed test set sample_idx does not match saved predictions"
print("  Verified: reconstructed held-out test set matches bpinn_predictions.npz exactly (w, true_amp, sample_idx)")

Feat_train = torch.tensor(np.stack([r['features'] for r in train_rows]), dtype=torch.float32)
Feat_mean, Feat_std = Feat_train.mean(0), Feat_train.std(0)

feat_arr_test = np.stack([r['features'] for r in labeled])
val['features'] = feat_arr_test  # now matches make_step6_figures' val dict exactly

in_dim = 1 + 2 * len(s6.CONFIG['network']['fourier_w_freqs']) + Feat_train.shape[1]
model = s6.BPINN(in_dim, s6.CONFIG['network']['hidden_sizes'], 2, s6.CONFIG['network']['prior_sigma'])
model.load_state_dict(torch.load(os.path.join(OUT, 'bpinn_state.pt'), map_location='cpu'))
model.eval()
norm_stats = (Feat_mean, Feat_std)

torch.manual_seed(0)  # fixed seed for this script's own MC draws (predict_mc has no internal seed)

# ── fig6 body, verbatim from step6.py's make_step6_figures ──
fig, ax = plt.subplots(figsize=(6.5, 6.3))
uniq_samples = np.unique(val['sample_idx'])
w_true_res, w_pred_res = [], []
for sidx in uniq_samples:
    mask = val['sample_idx'] == sidx
    w_true_res.append(val['w'][mask][np.argmax(val['true_amp'][mask])])
    w_grid = np.linspace(val['w'][mask].min(), val['w'][mask].max(), 200)
    feat_s = np.tile(val['features'][mask][0], (len(w_grid), 1))
    amp_grid, _, _, _ = s6.predict_mc(model, w_grid, feat_s, *norm_stats, s6.CONFIG['training']['n_mc_eval'])
    w_pred_res.append(w_grid[np.argmax(amp_grid)])
w_true_res, w_pred_res = np.array(w_true_res), np.array(w_pred_res)
res_err = np.abs(w_true_res - w_pred_res)
rmse_res = float(np.sqrt(np.mean(res_err ** 2)))
max_res = float(res_err.max())
resolved = res_err < 0.02
frac_resolved = float(resolved.mean())
print(f"  Resonance location: {frac_resolved * 100:.0f}% resolved to <2% (RMSE={rmse_res:.4f}, max={max_res:.4f})")
colors_res = [plot_style.C_OK if r else plot_style.C_WARN for r in resolved]
ax.scatter(w_true_res, w_pred_res, s=30, color=colors_res, alpha=0.75,
           edgecolor=plot_style.SURFACE, linewidth=0.8)
ax.plot([], [], 'o', color=plot_style.C_OK, label='resolved to <2%')
ax.plot([], [], 'o', color=plot_style.C_WARN, label='peak-smoothing limited (>2%)')
lims = [w_true_res.min() * 0.98, w_true_res.max() * 1.02]
ax.plot(lims, lims, color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect prediction')
ax.set_xlabel('True resonance location, $w=\\Omega/\\omega_0$')
ax.set_ylabel('BML-identified resonance location, $w$')
plot_style.two_tier_title(ax, 'Resonance frequency identification',
                           f'{len(uniq_samples)} held-out samples -- {frac_resolved * 100:.0f}% resolved to <2%')
plot_style.legend_inside(ax, loc='lower right', fontsize=12)
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step6_fig6_resonance_frequency')
print("Saved step6_fig6_resonance_frequency.png")
print(f"  Total elapsed: {time.time() - t0:.2f}s")
