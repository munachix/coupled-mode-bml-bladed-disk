# -*- coding: utf-8 -*-
"""
Regenerates the mistuning-vs-nonlinearity relationship as two separate
standalone figures (real data from
Step 9/output/mistuning_nonlinearity_relationship.npz) instead of the old
step9_fig9 side-by-side composite, per PAPER_INSTRUCTIONS.md's "no forced
multi-panel composites" rule.

Fig A: worst-blade geometric mistuning magnitude vs. nonlinear peak
       forced-response amplitude (essentially uncorrelated).
Fig B: mode-0 participation-weighted stiffness shift (direction) vs.
       centered resonance-peak shift (perfectly, linearly related).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(HERE, "output", "mistuning_nonlinearity_relationship.npz")
FIGS_DIR = os.path.join(os.path.dirname(HERE), "figures", "step9")


def main():
    d = np.load(NPZ_PATH)
    max_abs_df = d["max_abs_df"]  # already in percent
    shift_m = d["shift_m"]  # already in percent
    peak_amp = d["peak_amp"]
    res_freq_centered = d["res_freq_centered"]

    corr_amp = float(np.corrcoef(max_abs_df, peak_amp)[0, 1])
    corr_shift = float(np.corrcoef(shift_m, res_freq_centered)[0, 1])

    plot_style.apply_style()

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.scatter(max_abs_df, peak_amp, s=26, color=plot_style.C_1B, alpha=0.65,
               edgecolor=plot_style.SURFACE, linewidth=0.4, zorder=3)
    # Least-squares fit line + 95% confidence band across the full x-range
    # (2026-08-31, explicit request to make the plot use its own empty
    # region rather than leave it blank): with corr=-0.032 the fit is
    # visually near-flat, which is itself the honest finding -- the line
    # does not manufacture a trend, it makes the absence of one explicit
    # and gives the eye something to check across the whole axis width
    # instead of only the scatter's own denser left-hand region.
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(max_abs_df, peak_amp)
    x_line = np.linspace(max_abs_df.min(), max_abs_df.max(), 100)
    y_line = slope * x_line + intercept
    resid = peak_amp - (slope * max_abs_df + intercept)
    resid_std = resid.std(ddof=2)
    n = len(max_abs_df)
    x_mean = max_abs_df.mean()
    sxx = np.sum((max_abs_df - x_mean) ** 2)
    se_line = resid_std * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / sxx)
    tval = stats.t.ppf(0.975, n - 2)
    ax.plot(x_line, y_line, color=plot_style.C_WARN, lw=2.0, zorder=4,
            label=f"linear fit (p = {p_value:.2f}, not significant)")
    ax.fill_between(x_line, y_line - tval * se_line, y_line + tval * se_line,
                     color=plot_style.C_WARN, alpha=0.14, zorder=2, label="95% CI")
    ax.set_xlabel(r"Worst-blade geometric mistuning  max|df$_b$/f|  [%]")
    ax.set_ylabel("Nonlinear peak forced-response amplitude  [mm]")
    plot_style.two_tier_title(
        ax, "Mistuning magnitude vs. nonlinear response amplitude",
        f"mode 0, fixed forcing (linear peak target=1.0), 200 real Step-3 samples -- corr = {corr_amp:.3f}"
    )
    plot_style.legend_inside(ax, loc='lower right', fontsize=12)
    fig.tight_layout()
    paths_a = plot_style.savefig_pub(fig, FIGS_DIR, "step9_fig9a_mistuning_magnitude_vs_amplitude")
    print("Saved:", paths_a)

    fig, ax = plt.subplots(figsize=(7.9, 5.0))
    ax.axhline(0, color=plot_style.INK_MUTED, lw=0.9, ls="--")
    ax.axvline(0, color=plot_style.INK_MUTED, lw=0.9, ls="--")
    ax.scatter(shift_m, res_freq_centered, s=26, color=plot_style.C_HF, alpha=0.75,
               edgecolor=plot_style.SURFACE, linewidth=0.4)
    ax.set_xlabel("Mode-0 participation-weighted stiffness shift  [%]")
    ax.set_ylabel("Resonance-peak shift vs. ensemble mean  [Hz]")
    plot_style.two_tier_title(
        ax, "Mistuning direction vs. resonance shift",
        f"mode 0, same 200 samples, mistuning-attributable part only -- corr = {corr_shift:.4f}"
    )
    fig.tight_layout()
    paths_b = plot_style.savefig_pub(fig, FIGS_DIR, "step9_fig9b_mistuning_direction_vs_shift")
    print("Saved:", paths_b)


if __name__ == "__main__":
    main()
