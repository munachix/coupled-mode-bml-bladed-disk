# -*- coding: utf-8 -*-
"""
Real nonlinear FRF figures for Section 3.4.1 (2026-09-01, explicit user/
supervisor request: replace the scalar-summary scatter plots with actual
FRF curves showing clearly nonlinear -- jump-phenomenon, hardening --
behavior). Both figures use the SAME real, already-validated pseudo-arc-
length continuation solver (Step 4's duffing_forced_response_continuation)
that generated the exact-solution training targets used throughout
Section 3.1, for real mode 0 (K, M, C, K3 all from the project's own
extracted secondary-modal data) -- no fabricated curves, no illustrative
placeholders.

Fig A: forcing-level family at the TUNED baseline -- sweeps target_peak
       (the forcing level, in linear-response units) across several real
       levels for the tuned system, showing the real jump phenomenon
       emerge and strengthen as forcing increases.
Fig B: mistuning-level family at FIXED, strong forcing -- sweeps real
       mode-0 stiffness shift using three actual samples (low/mid/high
       |shift_m|) drawn from the SAME 200-realization ensemble behind the
       old scatter plots (Step 9/output/mistuning_nonlinearity_relationship
       .npz's own real shift_m array), showing how real mistuning moves
       the nonlinear backbone rather than only a peak-amplitude scalar.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step4 as s4
import step6 as s6
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

print("=== Real nonlinear FRF family figures (Section 3.4.1) ===", flush=True)

inp = s6.load_inputs()
K0 = float(inp['K_sec'][0, 0])
M0 = float(inp['M_sec'][0, 0])
C0 = float(inp['C_sec'][0, 0])
K3_0 = float(inp['K3_sec_diag'][0])
omega0 = np.sqrt(K0 / M0)
q_ref = 1.0
print(f"  mode 0: K={K0:.4e}, M={M0:.4f}, C={C0:.4e}, K3={K3_0:.4e}, f0={omega0/(2*np.pi):.2f} Hz", flush=True)

# ---- Fig A: forcing-level family, tuned baseline ----
TARGET_PEAKS = [0.3, 0.6, 1.0, 1.4]
curves_a = []
for tp in TARGET_PEAKS:
    r = s4.duffing_forced_response_continuation(omega0, M0, C0, K0, K3_0, q_ref, tp)
    curves_a.append(r)
    n_folds = r['n_folds']
    print(f"  target_peak={tp}: {len(r['Omega'])} points, {n_folds} fold(s) "
          f"({'jump phenomenon present' if n_folds >= 2 else 'no jump at this level'})", flush=True)

plot_style.apply_style()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 5.8))
colors_a = [plot_style.C_1B, plot_style.C_OK, plot_style.ORANGE, plot_style.C_WARN]
for tp, r, c in zip(TARGET_PEAKS, curves_a, colors_a):
    freq = r['Omega'] / (2 * np.pi)
    stable = r['stable']
    ax.plot(freq[stable], r['amplitude'][stable], color=c, lw=2.2, label=f"F/F$_0$ = {tp:g}")
    if (~stable).any():
        ax.plot(freq[~stable], r['amplitude'][~stable], color=c, lw=1.2, ls=(0, (2, 1.5)), alpha=0.6)
ax.set_xlabel("Frequency  [Hz]")
ax.set_ylabel("Response amplitude  [mm]")
plot_style.two_tier_title(ax, "Real nonlinear FRF family: forcing level",
                           "mode 0, tuned baseline -- dashed = unstable branch")
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step9_fig9c_frf_forcing_family')
print("Saved step9_fig9c_frf_forcing_family.png", flush=True)

# ---- Fig B: mistuning-level family, fixed strong forcing ----
d = np.load(os.path.join(OUT, "mistuning_nonlinearity_relationship.npz"))
shift_m = d["shift_m"]
order = np.argsort(np.abs(shift_m))
idx_low = order[len(order) // 6]
idx_mid = order[len(order) // 2]
idx_high = order[-1]
picks = [("low |shift|", idx_low), ("mid |shift|", idx_mid), ("high |shift|", idx_high)]
TP_FIXED = 1.2
print(f"  Fixed forcing target_peak={TP_FIXED} for the mistuning-level family", flush=True)

curves_b = []
for label, idx in picks:
    sm = float(shift_m[idx])
    K_shifted = K0 * (1.0 + sm / 100.0)
    r = s4.duffing_forced_response_continuation(omega0, M0, C0, K_shifted, K3_0, q_ref, TP_FIXED)
    curves_b.append((label, sm, r))
    print(f"  {label}: shift_m={sm:.3f}%, {r['n_folds']} fold(s)", flush=True)

fig, ax = plt.subplots(figsize=(8.6, 5.8))
colors_b = [plot_style.C_1B, plot_style.VIOLET, plot_style.C_WARN]
for (label, sm, r), c in zip(curves_b, colors_b):
    freq = r['Omega'] / (2 * np.pi)
    stable = r['stable']
    ax.plot(freq[stable], r['amplitude'][stable], color=c, lw=2.2,
             label=f"{label} (shift = {sm:+.2f}%)")
    if (~stable).any():
        ax.plot(freq[~stable], r['amplitude'][~stable], color=c, lw=1.2, ls=(0, (2, 1.5)), alpha=0.6)
ax.set_xlabel("Frequency  [Hz]")
ax.set_ylabel("Response amplitude  [mm]")
plot_style.two_tier_title(ax, "Real nonlinear FRF family: mistuning level",
                           f"mode 0, fixed strong forcing (F/F$_0$={TP_FIXED:g}) -- real samples from the 200-realization ensemble")
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step9_fig9d_frf_mistuning_family')
print("Saved step9_fig9d_frf_mistuning_family.png", flush=True)

np.savez(os.path.join(OUT, "nonlinear_frf_family.npz"),
         target_peaks=np.array(TARGET_PEAKS),
         n_folds_a=np.array([r['n_folds'] for r in curves_a]),
         shift_m_picks=np.array([sm for _, sm, _ in curves_b]),
         n_folds_b=np.array([r['n_folds'] for _, _, r in curves_b]),
         tp_fixed=TP_FIXED)
print("Saved nonlinear_frf_family.npz")
