# -*- coding: utf-8 -*-
"""
Regenerates the MCMC trace-plot panel alone (Section 3.2.2), dropping the
R-hat bar-chart panel per explicit user request ("keep the A part and
remove the B part"). Reconstructed from real saved data without
re-running MCMC: mcmc_posterior.npz's flat 'pooled' array is
chains_df.reshape(-1, NB) in the original run (step7.py line ~770), so
reshaping it back to (n_chains, n_samples, NB) recovers the exact
per-chain trace used by the original figure.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step7")

N_CHAINS = 4
N_SAMPLES = 4000
NB = 24


def main():
    post = np.load(os.path.join(OUT, "mcmc_posterior.npz"))
    so = np.load(os.path.join(OUT, "synthetic_observation.npz"))

    pooled = post["pooled"]
    assert pooled.shape == (N_CHAINS * N_SAMPLES, NB), pooled.shape
    chains = pooled.reshape(N_CHAINS, N_SAMPLES, NB)
    df_true = so["df_true"]
    show_blade = int(np.argmax(np.abs(df_true)))

    # 2026-08-31 REDESIGN (explicit user/supervisor request -- "nobody can
    # understand it"): the real problem was never the trace data itself,
    # it was that all 4 chains were drawn overlaid on one axes in four
    # nearly-identical shades of blue (a sequential colormap sampled for a
    # categorical purpose), producing an indistinguishable tangle. Split
    # into a 2x2 small-multiples grid, the standard way MCMC trace plots
    # are actually presented in the literature specifically because an
    # overlaid single-axes trace of >2 chains reads as noise: each chain
    # gets its own panel and its own distinct categorical color, sharing
    # one y-axis so amplitudes are still directly comparable, with the
    # true value drawn as the same reference line in every panel.
    plot_style.apply_style()
    chain_colors = [plot_style.BLUE, plot_style.ORANGE, plot_style.VIOLET, plot_style.C_OK]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex=True, sharey=True)
    y_true = df_true[show_blade]
    for c in range(N_CHAINS):
        ax = axes.flat[c]
        ax.plot(chains[c, :800, show_blade], color=chain_colors[c], lw=1.1)
        ax.axhline(y_true, color=plot_style.INK_PRIMARY, ls="--", lw=1.6)
        ax.text(0.02, 0.95, f"Chain {c + 1}", transform=ax.transAxes, fontsize=15,
                fontweight="bold", color=chain_colors[c], va="top", ha="left")
        if c >= 2:
            ax.set_xlabel("Post-warmup iteration")
        if c % 2 == 0:
            ax.set_ylabel(f"df$_b$/f, blade {show_blade}")
    fig.legend([plt.Line2D([], [], color=plot_style.INK_PRIMARY, ls="--", lw=1.6)],
               ["true value"], loc="upper right", frameon=False, fontsize=14,
               bbox_to_anchor=(0.99, 0.98))
    plot_style.figure_title(fig, "MCMC trace, most-mistuned blade",
                             f"blade {show_blade}, 4 independent chains shown separately -- all four mix "
                             f"around the true value, the visual signature of convergence",
                             y_title=1.03, y_subtitle=0.985)
    fig.subplots_adjust(top=0.85, hspace=0.18, wspace=0.06)
    paths = plot_style.savefig_pub(fig, FIGS_DIR, "step7_fig3a_mcmc_trace")
    print("Saved:", paths)


if __name__ == "__main__":
    main()
