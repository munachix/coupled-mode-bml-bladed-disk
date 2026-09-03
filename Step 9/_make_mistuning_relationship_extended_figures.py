# -*- coding: utf-8 -*-
"""
Rebuild the two mistuning-vs-nonlinearity scatter figures (Section 3.4.1) from
the extended-band ensemble.

The originals were computed with the continuation truncated at w = 1.6, where
this mode's response is still climbing, so "peak amplitude" was the value at the
edge of the band rather than the resonance peak. Reading a band edge instead of
a peak compressed the amplitude spread and left the reported correlation
describing the truncation as much as the physics. The rerun
(`_mistuning_nonlinearity_extended_band.py`) traces every realization to its
actual saddle-node fold, so both scatters now use true peak quantities.

Outputs:
    figures/step9/step9_fig9f_mistuning_magnitude_vs_amplitude.png
    figures/step9/step9_fig9g_mistuning_direction_vs_shift.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style  # noqa: E402

FIGS = os.path.join(ROOT, "figures", "step9")
OUT = os.path.join(HERE, "output")

d = np.load(os.path.join(OUT, "mistuning_nonlinearity_extended.npz"))
mag = d["max_abs_df"]
shift = d["shift_m"]
amp = d["peak_amp"]
dfreq = d["peak_freq_centered"]
pfreq = d["peak_freq"]
c_amp = float(d["corr_amp"])
c_frq = float(d["corr_freq"])
n = len(mag)

print("=== Extended-band mistuning relationship figures ===", flush=True)
print(f"  n = {n}", flush=True)
print(f"  worst-blade |df/f| : [{mag.min():.2f}, {mag.max():.2f}] %", flush=True)
print(f"  true peak amplitude: [{amp.min():.4f}, {amp.max():.4f}] mm "
      f"(spread {100*(amp.max()-amp.min())/amp.mean():.1f}% of mean)", flush=True)
print(f"  true peak frequency: [{pfreq.min():.1f}, {pfreq.max():.1f}] Hz", flush=True)
print(f"  corr(magnitude, amplitude) = {c_amp:+.4f}", flush=True)
print(f"  corr(direction, freq shift) = {c_frq:+.4f}", flush=True)

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

# ---- magnitude vs peak amplitude: the null result -------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.8))
ax.scatter(mag, amp, s=30, color=plot_style.C_1B, alpha=0.62,
           edgecolor=plot_style.SURFACE, linewidth=0.7, zorder=3)
lo, hi = amp.mean() - 0.02, amp.mean() + 0.02
ax.set_ylim(lo, hi)
ax.axhline(amp.mean(), color=plot_style.INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(mag.min(), amp.mean(), f"ensemble mean {amp.mean():.3f} mm  ",
        ha="left", va="bottom", fontsize=13, color=plot_style.INK_SECONDARY)
ax.set_xlabel(r"Worst-blade geometric mistuning, max $|\delta f/f|$  [%]")
ax.set_ylabel("Peak nonlinear amplitude  [mm]")
plot_style.two_tier_title(
    ax, "Mistuning magnitude does not set nonlinear amplitude",
    f"{n} realizations at fixed forcing, each traced to its own fold; "
    f"r = {c_amp:+.3f}")
plot_style.savefig_pub(fig, FIGS, "step9_fig9f_mistuning_magnitude_vs_amplitude")
print("Saved step9_fig9f_mistuning_magnitude_vs_amplitude.png", flush=True)

# ---- direction vs frequency shift: the near-exact relationship ------------
fig, ax = plt.subplots(figsize=(8.6, 5.8))
ax.scatter(shift, dfreq, s=30, color=plot_style.C_HF, alpha=0.62,
           edgecolor=plot_style.SURFACE, linewidth=0.7, zorder=3)
xs = np.linspace(shift.min(), shift.max(), 2)
sl, ic = np.polyfit(shift, dfreq, 1)
ax.plot(xs, sl * xs + ic, color=plot_style.INK, lw=1.4, ls=(0, (5, 3)), zorder=2,
        label=f"least squares, slope {sl:.2f} Hz per %")
ax.set_xlabel("Mode-0 participation-weighted stiffness shift  [%]")
ax.set_ylabel("Resonance-peak shift about the ensemble mean  [Hz]")
plot_style.two_tier_title(
    ax, "Mistuning direction sets the resonance shift",
    f"same {n} realizations, true nonlinear peak location; r = {c_frq:+.4f}")
plot_style.legend_inside(ax, loc="upper left", fontsize=13.5)
plot_style.savefig_pub(fig, FIGS, "step9_fig9g_mistuning_direction_vs_shift")
print("Saved step9_fig9g_mistuning_direction_vs_shift.png", flush=True)
