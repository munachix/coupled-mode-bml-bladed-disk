# Figure Captions — PCE Project (Steps 3–8)

> **Accuracy note (2026-09-02).** During the manuscript audit every number quoted here was re-checked against the Step outputs. Two were stale and have been corrected in place (Figure 5.4's GP R² and Figure 6.2's amplitude R²). Treat this file as a caption draft: the manuscript's own numbers are the audited ones, and `reproducibility/verify_paper_numbers.py` is what checks them.

Ready-to-paste captions for each figure, matched to the PNG/PDF files in `figures/step{N}/`. Each figure is saved as both a 300 dpi PNG (preview) and a vector PDF (print-ready). Figures carry no in-image caption text by design (journal convention) — the two-tier title on each plot is a short internal label, not the caption; use the text below in the manuscript.

---

## Step 3 — Geometric Mistuning Parameterization

**As of 2026-08-27, Step 3 models `d_tip` (blade-tip geometric deviation) only** — the other four candidate variables (d_length, d_thickness, d_le_te, d_twist_deg) were dropped from `CONFIG['tolerances']`, not merely unused; see `Step 3/step3.py`'s own tolerances comment for the rationale. Each figure below is now a single standalone plot, not an a–e panel set.

**Figure 3.1** (`step3_fig1_correlation_decay`). Spatial correlation of per-blade `d_tip` geometric mistuning as a function of circular blade distance (correlation length L = 2 blade-spacings).

**Figure 3.2** (`step3_fig2a_histogram_d_tip`). Realized marginal distribution of `d_tip`, pooled across 1000 Monte Carlo samples × 24 blades, compared against its target Gaussian N(0, σ). Realized mean (μ) and standard deviation (σ) are annotated.

**Figure 3.3** (`step3_fig3a_realization_d_tip`, standalone polar figure). One example `d_tip` mistuning realization (sample #0), shown as radial deviation from the nominal (dashed) baseline at each blade's true angular position around the 24-blade ring.

**Figure 3.4** (`step3_fig4_kl_spectrum`). Karhunen–Loève eigenvalue (variance) spectrum for `d_tip`. All 24 modes are retained — no truncation.

**Figure 3.5** (`step3_fig5a_covariance_d_tip`, standalone 2D heatmap). Theoretical circulant spatial-covariance matrix (24×24 blades) for `d_tip`, plotted as a flat heatmap — the circulant (constant-diagonal-band) structure reads directly as diagonal bands.

---

## Step 4 — Nonlinear Reduced-Order Model

**Figure 4.1** (`step4_fig1_mistuning_pattern`). Per-blade fractional-frequency mistuning pattern $\delta f/f$ (one Monte Carlo realization, sample #0), mapped from geometric deviations via the cantilever-beam sensitivity model.

**Figure 4.2** (`step4_fig2_freq_spectrum_spread`) — **Resonance frequency.** Nominal (tuned) vs. mistuned resonance frequencies across the 24-mode 1B cluster, shown for 15 mistuning realizations against the nominal (dashed) spectrum.

**Figure 4.3** (`step4_fig3_mistuning_magnitude_spread`). Distribution of the worst-blade (max $|\delta f/f|$) mistuning magnitude across the 40 validated Monte Carlo samples.

**Figure 4.4** (`step4_fig4_k3_diagonal`). Diagonal geometric-nonlinear (Duffing) hardening coefficient per secondary mode, log-scale bar chart (values span ~3.5 orders of magnitude across modes), color-coded by measured vs. extrapolated. All 70 of 70 secondary modes now have a real per-mode measurement; none remain an extrapolated placeholder.

**Figure 4.5** (`step4_fig5_duffing_backbone`) — **Nonlinear response amplitude & resonance frequency.** Forced-response amplitude vs. forcing frequency for mode 2 (one of the project's confirmed genuinely-isolated SDOF modes), traced via pseudo-arc-length continuation at three increasing forcing levels (F/F$_{ref}$ = 0.3, 0.7, 1.0; distinct categorical colors per level). The dotted skeleton curve is the undamped backbone; filled markers denote the fold (saddle-node) points bounding the unstable branch — all three levels shown here genuinely fold.

---

## Step 5 — Uncertainty Quantification Framework

**Figure 5.1** (`step5_fig1_hi1_distribution`). Aleatoric distribution of the health indicator HI1 (max 1B-cluster frequency deviation) across the full 1000-sample manufacturing-variability ensemble.

**Figure 5.2** (`step5_fig2_variance_decomposition`). Decomposition of total HI1 variance into aleatoric (manufacturing-variability) and epistemic (sensitivity-model-form) components via the law of total variance.

**Figure 5.3** (`step5_fig3_epistemic_fan`). Empirical CDF of HI1 under 25 perturbed sensitivity-coefficient draws (±25% placeholder epistemic prior), shown against the nominal-coefficient CDF.

**Figure 5.4** (`step5_fig4_gp_predicted_vs_true`). Gaussian-process surrogate predictions vs. true HI1 on a held-out test set (R² = 0.919, RMSE 0.79 Hz; verified against `Step 5/output/gp_surrogate.npz`, 2026-09-02 — the previously stated 0.876 was stale). Error bars show ±2σ GP predictive uncertainty.

**Figure 5.5** (`step5_fig5_gp_1d_slice`). GP surrogate mean surface vs. raw ensemble, shown as a 3D surface over the two most informative reduced features (remaining features held at their mean), with the raw Monte Carlo ensemble scattered around it.

---

## Step 6 — Physics-Informed Bayesian Neural Network

**Figure 6.1** (`step6_fig1_training_curves`). BPINN training loss components (data, physics-residual, boundary-condition, KL-divergence) vs. epoch, log scale.

**Figure 6.2** (`step6_fig2_predicted_vs_true`) — **Nonlinear response amplitude.** BML-predicted vs. exact (Step 4 continuation) forced-response amplitude on 100 held-out mistuning samples (R² = 0.998, RMSE 0.0018 mm; verified against `Step 6/output/multimode_bpinn_summary.json`, 2026-09-02 — the previously stated 0.91 was stale). Error bars show ±2σ predictive uncertainty.

**Figure 6.3** (`step6_fig3_residual_distribution`). Distribution of absolute prediction error between the BPINN mean and the exact continuation amplitude on the held-out test set.

**Figure 6.4a–d** (`step6_fig4{a-d}_example_curve_sample{N}`, one standalone figure per sample). BPINN-reconstructed forced-response curve (mean ± 2σ) against the exact continuation solution, for 4 representative held-out mistuning samples.

**Figure 6.5** (`step6_fig5_calibration`). Predictive-uncertainty calibration (reliability diagram): empirical coverage vs. nominal credible level on the held-out test set.

**Figure 6.6** (`step6_fig6_resonance_frequency`) — **Resonance frequency.** BPINN-identified resonance location (peak of its own reconstructed response curve) vs. the true continuation-curve peak, across 100 held-out mistuning samples. Accuracy is bimodal by design of the underlying network, not a plotting artifact: roughly half of samples resolve to within 2% of the true location (green), while the remainder show the same order of error as the peak-smoothing/underestimation already disclosed for the amplitude prediction itself (Section 4, Step 6's debugging journey, in `PROJECT_STATUS.md`) — investigated directly (ruled out: search-grid extrapolation, sparse-vs-exact ground-truth mismatch, search-window width) before being reported as a real network limitation rather than tuned away.

> **Note on "stress state":** the roadmap's Phase 7 output list (nonlinear response amplitude / resonance frequency / stress state) is only partially covered by design, not by oversight. No FEM stress recovery exists anywhere in this project — Step 1's PyMAPDL extraction was never run (no ANSYS access in this environment), and none of Steps 2–8 compute or store nodal/element stress. Adding a "stress state" figure here would require fabricating data this project does not have. See `PROJECT_STATUS.md` for the full disclosure and what real extraction would require.

---

## Step 7 — Bayesian Mistuning Identification

**Figure 7.1** (`step7_fig1_recovery_scatter`). Posterior mean per-blade mistuning identification vs. true values, 24 blades, with 95% credible intervals (corr = 0.92).

**Figure 7.2** (`step7_fig2_identifiability_vs_recovery`). Recovery accuracy vs. posterior shrinkage (1 − posterior std / prior std) across the 9 identifiable latent directions — demonstrates that the inversion's own uncertainty estimates track its actual accuracy.

**Figure 7.3** (`step7_fig3_mcmc_diagnostics`). MCMC convergence diagnostics: trace plot for the most-mistuned blade (left, 4 chains) and Gelman–Rubin R-hat by latent direction (right).

**Figure 7.4** (`step7_fig4_coverage_calibration`). Posterior credible-interval calibration: empirical coverage vs. nominal credible level across 20 independent held-out trials.

**Figure 7.5** (`step7_fig5_bpinn_reconstruction`) — **BPINN-accelerated reconstruction.** Mode-0 forced-response curve reconstructed from the *inferred* (not exact) mistuning state via the Step 6 BPINN, against the exact continuation solution for the true state (R² = 0.85).

---

## Step 8 — Health Monitoring Framework

**Figure 8.1** (`step8_fig1_HI1_trajectory`). Health indicator HI1 over a 20-step synthetic damage trajectory: true injected value vs. value computed from the Bayesian-inferred state, against the Step 5 healthy-population 99th-percentile reference.

**Figure 8.2** (`step8_fig2_HI2_detection`). HI2 (Mahalanobis anomaly score, computed in the identifiable latent subspace) over the trajectory, with the empirically calibrated detection threshold and null-trial (no-damage) reference scatter.

**Figure 8.3** (`step8_fig3_localization`). Per-blade recovered mistuning deviation at the final (most-damaged) time step; red bar marks the true damaged blade.

**Figure 8.4** (`step8_fig4_false_alarm_calibration`). False-alarm calibration: HI2 distributions for the calibration null-trial set vs. a separate held-out null-trial set, against the calibrated detection threshold.

**Figure 8.5** (`step8_fig5_tracked_recovery`). Tracked recovery of the damaged blade's mistuning state (posterior mean ± 2σ) vs. the true injected trajectory, over all 20 time steps.

---

## Step 2 (addendum) — Campbell diagram

`figures/step2/step2_fig_campbell_1B.png`, generated by `Step 2/_make_campbell_diagram_figure.py`.
Manuscript Figure 24 (Section 3.4.2).

> **Fig. 24.** Campbell diagram of the 1B cluster: rotating-frame frequencies with centrifugal
> stress stiffening, spin softening and Coriolis on the 70-mode secondary basis, against
> engine-order excitation lines. Circles mark the 13 crossings of the cluster below 20,000 rpm.
> The diamond marks the one speed with an independent prestressed full-order check (7,200 rpm,
> 1.24% mean and 1.44% maximum error over the 24 modes). Every other result in this paper is
> computed at the left-hand edge, at rest.

Source matrices are `rotating_secondary_bundle.npz` (G_sec, Kcs_sec, Ksigma_sec) and the M_sec/K_sec
pair, all in the ROM output directory; the validation numbers come from
`step2_rotating_validation.json`. The Section 3.4.1 fold is deliberately **not** overlaid: it was
traced at rest, and the stiffening that lifts the cluster lifts the fold with it, so a
frequency-axis comparison would mislead. The margin is reported as a ratio in the text instead.


---

## Step 9 (addendum) — mistuning relationships, combined 3D figure

`figures/step9/step9_fig9h_mistuning_dual_surface_3d.png`, generated by
`Step 9/_make_mistuning_dual_surface_3d_figure.py`. Manuscript Figure 23 (Section 3.4.1). Replaces
the two separate 2D scatters `step9_fig9f_*` and `step9_fig9g_*`, which are superseded and no longer
used by the manuscript.

> **Fig. 23.** Mistuning magnitude and direction against the two quantities they could set, mode 0,
> 200 realizations at fixed forcing, each traced to its own fold. Both panels share the same
> footprint and the same fitted least-squares plane: (a) peak nonlinear amplitude, plane slope
> 0.00073 mm per percent of stiffness shift, total spread 1.6% of the 0.308 mm mean; (b)
> resonance-peak shift about the ensemble mean, plane slope 2.49 Hz per percent, total spread
> 17.1 Hz or about 4.5 half-power bandwidths. Each plane fits to R² = 1.0000. Panel (a)'s vertical
> axis spans a five percent window about the mean rather than the data range, so that a negligible
> dependence is not magnified into an apparent one.

**Result note.** Fitting over both mistuning coordinates rather than correlating each quantity
against one showed that mistuning *direction* predicts the peak amplitude just as exactly
(r = +1.0000) as it predicts the resonance shift. The old pair of figures implied amplitude was
unpredictable; it is not, it is simply insensitive. The distinction is slope, not fit quality.

---

## Step 9 (addendum) — real damage-injection campaign

`figures/step9/step9_fig17_damage_sweep_summary.png`, generated by
`Step 9/_make_damage_sweep_summary_figure.py`. Manuscript Figure 32 (Section 3.6). Source data:
`Step 9/output/validation3_damage_sweep.json` (28 full-order injections run by
`Step 9/_validation3_damage_sweep.py` over two rounds) plus the two original cases.

> **Fig. 32.** All thirty real finite-element damage injections. (a) The full ring at a fixed -4.5%
> severity: ring distance from the true blade against blade index, with shaded columns marking the
> blades the coupled model misses. Failures are periodic in blade index with period 12, 23 of 24
> blades agreeing with their residue class against 17 of 24 for the next-best period. (b) Severity
> ladders on one blade from a failing class (12) and one from a succeeding class (14). (c) The blade
> each model picks against the true damaged blade, all thirty cases pooled; the diagonal-only model
> answers blade 21 in 27 of them regardless of the true location. Overall the coupled model is exact
> in 12 of 30 and within tolerance in 15, the diagonal-only model in 1 and 5.

**Result note.** Round 1 (ten cases) gave a rate with no mechanism. Round 2 measured the whole ring
and turned it into structure: localizability is periodic in blade index with period 12, so blades
2-7 and 14-19 are localized and 8-13 and 20-1 are not. Severity modulates the margin but position
decides whether localization is possible at all. The diagonal-only model's single "exact" hit is
blade 21, which is the answer it returns in 27 of 30 cases regardless of the truth.
