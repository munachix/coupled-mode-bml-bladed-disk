# Coupled-Mode Bayesian Machine Learning for Mistuned Bladed Disks

Reproducibility material for *Coupled-Mode Bayesian Machine Learning for Nonlinear
Forced Response Prediction and Health Monitoring of Mistuned Bladed Disks*.

The paper's claim is that one Bayesian surrogate, trained once for nonlinear
forced-response prediction, can be reused without retraining for health
monitoring. This repository holds the saved outputs behind every number and
figure in it, together with the scripts that produce them.

## Verifying the paper's numbers

```bash
python reproducibility/verify_paper_numbers.py
```

This recomputes every headline quantity in the manuscript directly from the
saved outputs and checks it against the value printed in the paper. It is
self-contained: the four reduced-order matrices it needs are in `rom_data/`, and
it should report **129 of 129 checks passed**. Set `ROM_DATA_DIR` to point it
somewhere else if you have the full extraction.

A failure is meaningful. Each check names the section, the quantity, the
recomputed value and the value the paper prints, so a mismatch localizes
immediately.

## Layout

| path | contents |
|---|---|
| `reproducibility/` | the verification script, the figure manifest, notes |
| `Step 2/` … `Step 9/` | the pipeline scripts, and each step's saved `output/` |
| `figures/` | every figure in the paper, as generated |
| `rom_data/` | the reduced matrices the verification script needs |

`reproducibility/FIGURE_MANIFEST.md` resolves each numbered figure in the paper
to the image file it was built from and the script that writes that image. The
mapping was produced by hashing the images embedded in the manuscript, so it
reflects what is actually in the document.

## What is not here

The full-order finite-element model (181,473 degrees of freedom), its extracted
stiffness and mass matrices, and the mode shapes are too large to distribute and
were produced with a commercial finite-element package. The full-order solutions
used as validation references are included as saved results, so every comparison
in the paper can be checked; regenerating them from the mesh cannot be done from
this repository alone.

The thirty real damage-injection cases of Section 3.6 are in
`Step 9/output/validation3_damage_sweep.json`, one record per case with the ring
distance, rank and margin for both forward models. The periodicity reported in
that section is re-derived from those records by the verification script rather
than asserted, so it is recomputed on every run.
