# Reproducibility package

Everything a reviewer needs to check the claims in *Coupled-Mode Bayesian Machine Learning for Nonlinear Forced Response Prediction and Health Monitoring of Mistuned Bladed Disks*.

The package is built around one principle: **no number in the paper should have to be taken on trust.** Every headline figure quoted in the text is recomputed here from the stored result files and compared against what the manuscript prints.

---

## Quick start

```bash
python reproducibility/verify_paper_numbers.py
```

This reads the saved Step 6-9 outputs, recomputes all 54 headline quantities, and prints a `PASS`/`FAIL` line for each against the value printed in the paper. It exits `0` only if every check passes. It needs `numpy` and nothing else: no ANSYS licence, no GPU, no retraining. It runs in a couple of seconds.

Current status: **54 of 54 checks pass.**

---

## What is where

| Path | Contents |
|---|---|
| `reproducibility/verify_paper_numbers.py` | The executable check described above. |
| `reproducibility/FIGURE_MANIFEST.md` | Every figure in the manuscript, resolved to its source image and the script that writes it. |
| `Step 2/` … `Step 9/` | The pipeline itself, in run order. Each `Step N/output/` holds that stage's saved results. |
| `figures/stepN/` | Every plotted figure, as written by the scripts in `Step N/`. |
| `diagrams/` | Prompts for the two generated diagrams, plus `rendered/` holding the images actually embedded as Figures 1 to 3. |
| `Coupled-Mode_BPINN_References.enw` / `.md` | The 54-entry reference list, as EndNote export and as Markdown. |
| `plot_style.py` | The shared figure style. Every figure in the paper is drawn through it. |

## Regenerating a figure

`FIGURE_MANIFEST.md` names the script for each figure number. Scripts are run from the project root, for example:

```bash
python "Step 8/_make_health_indicator_flow_figure.py"
```

Scripts prefixed `_make_` only re-plot from saved results, so they are cheap and safe to rerun. The pipeline scripts (`step2.py` … `step9.py`) re-solve or retrain and are correspondingly expensive.

## What cannot be rerun from this package alone

Stated plainly, because a reviewer should not discover it by hitting an error:

- **The real ANSYS solves.** The full-order finite-element runs (181,473 DOF harmonic, transient, step and impulse, and the geometry-perturbed damage injections) were produced in ANSYS against model files on a separate drive, and need an ANSYS licence to repeat. What is included is every result they produced: `Step 9/output/step_impulse_ansys_verified.npz`, `validation3_real_ansys_health_id.json`, the Case reconstruction archives, and the harmonic reference points. Every ANSYS-derived number the paper quotes is checked against those files by `verify_paper_numbers.py`.
- **Network training.** Retraining the per-mode and per-pair networks takes roughly five to six minutes each on CPU (`elapsed` is recorded in `Step 6/output/multimode_bpinn_summary.json`). The trained weights are included as `.pt` files, so predictions can be reproduced without retraining.
- **Two figures whose stats need the trained model.** Figures 9 and 10 (resonance identification, calibration) re-run Monte Carlo forward passes; their reported statistics are baked into the figure images themselves, which is how they were verified during the audit.

## Provenance note

The manuscript was audited against these files in September 2026. Two rounding errors were found and corrected in the paper at that time (the HI3 trajectory start value, and the classifier's severity correlation), along with a number of larger discrepancies; `verify_paper_numbers.py` is the artefact that caught the rounding pair and now guards against their reappearance.
