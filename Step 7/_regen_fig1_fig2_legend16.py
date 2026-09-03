# -*- coding: utf-8 -*-
"""Regenerates step7_fig1_recovery_scatter and step7_fig2_identifiability_vs_recovery
at the legend fontsize now matching axes.labelsize exactly (both 16), following the
same fix already applied+verified on step7_fig4_coverage_calibration (see
_regen_fig4_coverage.py) and rolled out project-wide in plot_style.py's legend_below().

Both figures use plot_style.legend_below() (no direct ax.legend(fontsize=...) call at
either figure's own call site in step7.py), so the plot_style.py fix alone covers them
-- this script just replays the exact plotting code from step7.py's make_step7_figures
(fig1 + fig2 blocks only) against already-saved outputs, no MCMC re-run:
  - fig1 needs primary['df_true'/'post_mean'/'post_lo'/'post_hi'] -- df_true is in
    synthetic_observation.npz, the rest are in mcmc_posterior.npz. Exact saved values,
    no recomputation.
  - fig2 needs svd_analysis['shrinkage'/'err'/'top'], which step7.py computes from
    primary['chains_z'] (the POOLED MCMC chain reprojected into z-space) -- chains_z
    itself isn't saved, but it's a pure linear reprojection of the saved 'pooled'
    physical-df-space chain via df_to_z(df, prior) = prior['V'].T @ (df - prior['mu0'])
    (step7.py line ~396), and mu0/V/lam are all saved in mcmc_posterior.npz. Redone
    here as plain matrix algebra on the saved chain -- not a re-run of MCMC itself.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step7")

post = np.load(os.path.join(OUT, "mcmc_posterior.npz"))
so = np.load(os.path.join(OUT, "synthetic_observation.npz"))

df_true = so["df_true"]
post_mean = post["post_mean"]
post_lo = post["post_lo"]
post_hi = post["post_hi"]
mu0 = post["mu0"]
V = post["V"]
lam = post["lam"]
K = int(post["K"])
pooled = post["pooled"]  # (n_chains*n_samples, 24), physical df space

primary = dict(df_true=df_true, post_mean=post_mean, post_lo=post_lo, post_hi=post_hi)
prior = dict(mu0=mu0, V=V, lam=lam, K=K)


def df_to_z(df, prior):
    d = df - prior['mu0']
    return d @ prior['V'] if d.ndim > 1 else prior['V'].T @ d


# ── replay of latent_shrinkage_analysis (step7.py ~811-848), against the
#    saved 'pooled' chain reprojected into z-space rather than a fresh run ──
z_true = df_to_z(df_true, prior)
pooled_z = df_to_z(pooled, prior)          # (N, K)
z_post_mean = pooled_z.mean(axis=0)
z_post_std = pooled_z.std(axis=0)
prior_std = np.sqrt(prior['lam'])

shrinkage = 1.0 - z_post_std / prior_std
err = np.abs(z_true - z_post_mean) / prior_std
top = shrinkage >= np.median(shrinkage)
svd_analysis = dict(shrinkage=shrinkage, err=err, top=top)

plot_style.apply_style()

# ── fig1: recovered vs true df_b/f, all 24 blades, 95% CI (verbatim from step7.py) ──
fig, ax = plt.subplots(figsize=(6.8, 6.5))
ax.errorbar(primary['df_true'] * 100, primary['post_mean'] * 100,
            yerr=[(primary['post_mean'] - primary['post_lo']) * 100,
                  (primary['post_hi'] - primary['post_mean']) * 100],
            fmt='o', ms=6, color=plot_style.BLUE, ecolor=plot_style.BLUE, alpha=0.7,
            elinewidth=1.1, capsize=2, mec=plot_style.SURFACE, mew=1.0)
lims = [min(primary['df_true'].min(), primary['post_mean'].min()) * 100 * 1.15,
        max(primary['df_true'].max(), primary['post_mean'].max()) * 100 * 1.15]
ax.plot(lims, lims, color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect recovery')
ax.set_xlabel('True per-blade mistuning  df$_b$/f  [%]')
ax.set_ylabel('Posterior mean  (error bars: 95% credible interval)  [%]')
corr = float(np.corrcoef(primary['post_mean'], primary['df_true'])[0, 1])
plot_style.two_tier_title(ax, 'Per-blade mistuning identification', f'24 blades -- corr={corr:.3f}')
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step7_fig1_recovery_scatter')
print("Saved step7_fig1_recovery_scatter.png  (corr=%.3f)" % corr)

# ── fig2: recovery error vs. posterior shrinkage (verbatim from step7.py) ──
fig, ax = plt.subplots(figsize=(7.5, 5.6))
colors = [plot_style.AQUA if t else plot_style.ORANGE for t in svd_analysis['top']]
ax.scatter(svd_analysis['shrinkage'], svd_analysis['err'], c=colors, s=70,
           edgecolor=plot_style.SURFACE, linewidth=1.0)
for k in range(len(svd_analysis['shrinkage'])):
    ax.annotate(str(k), (svd_analysis['shrinkage'][k], svd_analysis['err'][k]),
                fontsize=8, color=plot_style.INK_SECONDARY, xytext=(5, 4), textcoords='offset points')
ax.set_xlabel('Posterior shrinkage  (1 - posterior std / prior std)')
ax.set_ylabel('Prior-normalized recovery error')
plot_style.two_tier_title(ax, 'Recovery accuracy tracks data informativeness',
                           'per latent direction, not blade identity')
ax.plot([], [], 'o', color=plot_style.AQUA, label='high shrinkage')
ax.plot([], [], 'o', color=plot_style.ORANGE, label='low shrinkage')
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step7_fig2_identifiability_vs_recovery')
print("Saved step7_fig2_identifiability_vs_recovery.png")
