# -*- coding: utf-8 -*-
"""One-off: regenerate step9_fig8a_k3_real_vs_rom and step9_fig8b_bpinn_accuracy_per_mode
after converting their legend_below() calls to legend_inside(), per the legend-position
correction. Imports step9.py as a module (skips its __main__ block entirely) and calls
make_multimode_bpinn_ansys_figure() directly -- reads cached npz outputs only, no re-run."""
import step9 as s9
s9.make_multimode_bpinn_ansys_figure()
