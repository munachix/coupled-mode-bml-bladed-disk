# -*- coding: utf-8 -*-
"""
Nonlinear FRF family showing the hardening fold (Section 3.4.1).

Replaces the previous pair of figures (forcing-level family and
mistuning-level family). Both were traced over w = Omega/omega_0 in [0.7, 1.6],
which for this mode's identified cubic stiffness stops well below the response
peak: every curve was still climbing at the right-hand edge, so the figures read
as monotonic ramps and showed neither the peak nor the fold that defines a
hardening Duffing resonance.

The continuation is unchanged; only the band it is traced over and the
arc-length step are. Extending w_stop_hi to 6.0 and halving ds to 0.004 lets the
solver turn both folds cleanly at every forcing level, so each curve now shows
the full S: hardening up the backbone, the saddle-node fold at the top, the
unstable middle branch, and the drop back to the low-amplitude branch.

The 1B cluster band is shaded, because the honest reading of this figure is that
the fold is a real property of the identified nonlinearity but sits above the
band this blisk actually operates in.

Output: figures/step9/step9_fig9e_frf_forcing_family_fold.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project")
import step4 as s4  # noqa: E402
import step6 as s6  # noqa: E402
import plot_style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

inp = s6.load_inputs()
K0 = float(inp["K_sec"][0, 0])
M0 = float(inp["M_sec"][0, 0])
C0 = float(inp["C_sec"][0, 0])
K3_0 = float(inp["K3_sec_diag"][0])
omega0 = np.sqrt(K0 / M0)
f0 = omega0 / (2 * np.pi)

cfg = s4.CONFIG["continuation"]
cfg["w_stop_hi"] = 6.0
cfg["ds"] = 0.004
cfg["n_steps"] = 25000

TARGET_PEAKS = [0.3, 0.6, 1.0, 1.4]
print("=== Nonlinear FRF family with folds (Section 3.4.1) ===", flush=True)
print(f"  mode 0: f0 = {f0:.2f} Hz, K3 = {K3_0:.4e}", flush=True)

curves = []
for tp in TARGET_PEAKS:
    r = s4.duffing_forced_response_continuation(omega0, M0, C0, K0, K3_0, 1.0, tp)
    hz = r["Omega"] / (2 * np.pi)
    amp = r["amplitude"]
    folds = r["fold_indices"]
    curves.append((tp, hz, amp, r["stable"], folds))
    up = hz[folds[0]] if len(folds) else np.nan
    dn = hz[folds[1]] if len(folds) > 1 else np.nan
    print(f"  F/F0={tp:<4}: {len(hz):5d} pts, {r['n_folds']} folds, "
          f"peak {amp.max():.4f} mm at {hz[amp.argmax()]:.1f} Hz, "
          f"upper fold {up:.1f} Hz, lower fold {dn:.1f} Hz", flush=True)

np.savez(os.path.join(OUT, "frf_forcing_family_fold.npz"),
         target_peaks=np.array(TARGET_PEAKS),
         **{f"hz_{i}": c[1] for i, c in enumerate(curves)},
         **{f"amp_{i}": c[2] for i, c in enumerate(curves)},
         **{f"stable_{i}": c[3] for i, c in enumerate(curves)})

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(9.2, 6.2))

# the band the blisk actually operates in
ax.axvspan(250, 500, color=plot_style.C_1B, alpha=0.07, zorder=0)
ax.text(375, 0.395, "1B cluster band", ha="center", va="top", fontsize=13,
        color=plot_style.INK_SECONDARY, style="italic")

# The four levels share one backbone, so a sequential ramp made them read as a
# single curve. Categorical colours keep each forcing level separable where the
# curves actually differ, which is at their folds.
pick = plot_style.CATEGORICAL[:4]
for (tp, hz, amp, stable, folds), col in zip(curves, pick):
    s = stable.copy()
    seg = np.split(np.arange(len(hz)), np.where(np.diff(s.astype(int)) != 0)[0] + 1)
    for idx in seg:
        if len(idx) < 2:
            continue
        style = "-" if s[idx[0]] else (0, (4, 3))
        ax.plot(hz[idx], amp[idx], ls=style, color=col, lw=2.0,
                zorder=3 if s[idx[0]] else 2)
    ax.plot([], [], color=col, lw=2.0, label=f"F/F$_0$ = {tp:g}")
    if len(folds):
        ax.plot(hz[folds[:2]], amp[folds[:2]], "o", color=col, ms=7,
                mec=plot_style.INK, mew=1.2, zorder=5)

ax.plot([], [], "o", color=plot_style.SURFACE, mec=plot_style.INK, mew=1.2, ms=7,
        label="saddle-node fold")
ax.plot([], [], ls=(0, (4, 3)), color=plot_style.INK_SECONDARY, lw=2.0,
        label="unstable branch")

ax.set_xlim(200, 1250)
ax.set_ylim(0, 0.42)
ax.set_xlabel("Frequency  [Hz]")
ax.set_ylabel("Response amplitude  [mm]")
plot_style.two_tier_title(
    ax, "Nonlinear forced response: hardening fold",
    "mode 0, tuned baseline, four real forcing levels traced by arc-length continuation")
plot_style.legend_inside(ax, loc="center right", fontsize=13.0)
plot_style.savefig_pub(fig, FIGS, "step9_fig9e_frf_forcing_family_fold")
print("Saved step9_fig9e_frf_forcing_family_fold.png", flush=True)
