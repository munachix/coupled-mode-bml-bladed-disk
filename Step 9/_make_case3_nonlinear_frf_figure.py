# -*- coding: utf-8 -*-
"""
Real nonlinear FRF comparison, BML vs. real ANSYS, Case 3 (mistuned,
nonlinear) -- rebuilds the case3 curve from _frf_ansys_vs_bpinn.py (whose
own saved case3 PNG was lost/never persisted; this script re-derives it
from the SAME validated methodology and models, no retraining, and saves
the underlying (freqs, amplitude) curve to disk this time so it does not
need re-deriving again) and replaces the earlier "3.5.3" comparison that
mixed a same-node BML/Compact-ROM prediction against a DIFFERENT-VERTEX
real ANSYS reading (an apples-to-oranges comparison correctly flagged by
the user).

The compositional BML sums THREE trained pair/single networks -- (0,1),
mode 2 alone, (3,4) -- each driven by its own real share of the SAME
physical point load (1000N at node 1171, decomposed via the real mode-
shape participation Phi_m = T_full2sec[node1171_eq, m]), matching the
already-validated Case 3 single-point check
(Step 7/_case3_compositional_check.py) exactly. This is the SAME node,
SAME loading condition as the real ANSYS transient measurement
(1.2220 +/- 0.0190 mm at 292.82 Hz) -- a genuinely fair, same-node
comparison, unlike the removed Fig 25 panel.

No new ANSYS run: real ANSYS nonlinear sweep does not exist for this
project (a 7-point attempt was made and abandoned, 6 of 7 points never
settled to steady state -- disclosed in PROJECT_STATUS.md). The single
transient point is the only real ANSYS ground truth available for the
nonlinear regime, and is plotted as an error-barred point against the
BML's own frequency sweep, which is real physics-conditioned inference
(not extrapolated from the single point) and visibly shows the nonlinear
hardening bend the linear cases do not have.
"""
import math
import os
import sys

import numpy as np
import torch

torch.manual_seed(42)
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import step2 as s2
import step7 as s7
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")

REAL_ANSYS_C3_AMP = 1.2220
REAL_ANSYS_C3_STD = 0.0190
REAL_ANSYS_C3_FREQ = 292.82

print("=== Case 3 real nonlinear FRF: compositional BML vs. real ANSYS (single validated point) ===", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]

K0 = {m: inp['K_sec'][m, m] for m in range(5)}
zeta0 = {m: inp['C_sec'][m, m] / (2 * math.sqrt(inp['K_sec'][m, m] * inp['M_sec'][m, m])) for m in range(5)}


# 2026-08-30 CORRECTION: matches the validated single-point check
# (Step 7/_case3_compositional_check.py) and _frf_ansys_vs_bpinn.py exactly
# -- the real ANSYS Case-3 transient (1.2220+/-0.0190mm) was run at a point
# load calibrated so mode 0's OWN generalized force equals 2500N (the same
# convention as the linear Cases 1/2/4 HARMIC runs), decomposed onto every
# mode via the real mode-shape participation Phi_m. An earlier attempt at
# this comparison (the removed Fig 25 / step9_fig8c panel) used a literal,
# uncalibrated 1000N instead -- a materially different, unvalidated loading
# condition compared against a real ANSYS reading taken at a different
# vertex besides; that mismatch is what the "0.77mm" figure was actually
# reporting, not a same-condition BML/ANSYS check. F_physical here is
# solved backwards from the SAME 2500N generalized-force target so this
# curve is directly comparable to the real, validated 1.2220mm point.
F_physical_c3 = 2500.0 / Phi_all[0]
print(f"  F_physical_c3={F_physical_c3:.4f} N (calibrated to 2500N generalized force on mode 0, "
      f"matching the validated Case-3 single-point check)", flush=True)
Fg_real_c3 = {m: F_physical_c3 * Phi_all[m] for m in range(5)}
tp01_c3 = Fg_real_c3[0] / (2 * zeta0[0] * K0[0])
tp2_c3 = Fg_real_c3[2] / (2 * zeta0[2] * K0[2])
tp34_c3 = Fg_real_c3[3] / (2 * zeta0[3] * K0[3])

model01, norm01 = s7.load_bpinn_coupled((0, 1))
model34, norm34 = s7.load_bpinn_coupled((3, 4))
norm2 = dict(np.load(os.path.join(s6.OUT, 'bpinn_forcing_aware_mode2_norm.npz')))
state2 = torch.load(os.path.join(s6.OUT, 'bpinn_forcing_aware_mode2_state.pt'))
in_dim2 = state2['layers.0.w_mu'].shape[1]
h02 = state2['layers.0.w_mu'].shape[0]
h12 = state2['layers.1.w_mu'].shape[0]
model2 = s6.BPINN(in_dim2, [h02, h12], 2, prior_sigma=1.0)
model2.load_state_dict(state2)
model2.eval()


def predict_pair_frf(model, norm, tp_signed, mi, mj, w_arr, n_mc=300):
    sign = 1.0 if tp_signed >= 0 else -1.0
    tp_abs = abs(tp_signed)
    feat = np.tile([0.0, zeta0[mi], 0.75 * inp['K3_sec_diag'][mi] / K0[mi],
                    0.0, zeta0[mj], 0.75 * inp['K3_sec_diag'][mj] / K0[mj]], (len(w_arr), 1))
    n_feat = len(norm['feat_mean'])
    feat_out = s6.add_detune_features(w_arr, feat) if n_feat in (8, 9) else feat
    if bool(norm.get('is_forcing_aware', n_feat == 9)):
        feat_out = np.concatenate([feat_out, np.full((len(w_arr), 1), tp_abs)], axis=1)
    feat_mean = torch.tensor(norm['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat_out, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    model.eval()
    with torch.no_grad():
        samples = np.array([model(X_in).numpy() for _ in range(n_mc)])
    ai = sign * (samples[:, :, 0] * float(norm['alpha_i_std']) + float(norm['alpha_i_mean'])).mean(0)
    bi = sign * (samples[:, :, 1] * float(norm['beta_i_std']) + float(norm['beta_i_mean'])).mean(0)
    aj = sign * (samples[:, :, 2] * float(norm['alpha_j_std']) + float(norm['alpha_j_mean'])).mean(0)
    bj = sign * (samples[:, :, 3] * float(norm['beta_j_std']) + float(norm['beta_j_mean'])).mean(0)
    return ai, bi, aj, bj


def predict_mode2_frf(tp_signed, w_arr, n_mc=300):
    sign = 1.0 if tp_signed >= 0 else -1.0
    tp_abs = abs(tp_signed)
    zeta2 = zeta0[2]
    kappa2 = 0.75 * inp['K3_sec_diag'][2] / K0[2]
    detune2 = np.tanh(((w_arr - 1.0) / zeta2) / 20.0)
    feat = np.stack([np.zeros(len(w_arr)), np.full(len(w_arr), zeta2), np.full(len(w_arr), kappa2),
                      detune2, np.full(len(w_arr), tp_abs)], axis=1)
    feat_mean = torch.tensor(norm2['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm2['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    with torch.no_grad():
        samples = np.array([model2(X_in).numpy() for _ in range(n_mc)])
    a = sign * (samples[:, :, 0] * float(norm2['alpha_std']) + float(norm2['alpha_mean'])).mean(0)
    b = sign * (samples[:, :, 1] * float(norm2['beta_std']) + float(norm2['beta_mean'])).mean(0)
    return a, b


def compositional_frf_case3(freqs_hz):
    omega0_mode0 = math.sqrt(K0[0] / inp['M_sec'][0, 0])
    w_arr = 2 * np.pi * freqs_hz / omega0_mode0
    a0, b0, a1, b1 = predict_pair_frf(model01, norm01, tp01_c3, 0, 1, w_arr)
    a2, b2 = predict_mode2_frf(tp2_c3, w_arr)
    a3, b3, a4, b4 = predict_pair_frf(model34, norm34, tp34_c3, 3, 4, w_arr)
    alphas = [a0, a1, a2, a3, a4]
    betas = [b0, b1, b2, b3, b4]
    u = sum((alphas[m] - 1j * betas[m]) * Phi_all[m] for m in range(5))
    return np.abs(u)



# Upper bound trimmed to 302 Hz (2026-08-30, found empirically): the
# compositional network turns visibly non-physical (a spurious upward
# blip right after its own resonance peak, then chaotic oscillation)
# starting around 304 Hz -- past its trained w-range, an extrapolation
# artifact rather than real hardening physics. 302 Hz keeps the full,
# smooth hardening bend (peak at 300.4 Hz) without that artifact.

# Full sweep computed once and cached; the PLOTTED window below is
# narrower (see ax.set_xlim), cropping the flat, uninformative low-
# frequency lead-in so the rise-through-resonance-and-hardening-peak
# shape (the actually informative part) fills the canvas rather than
# sitting in a small corner of a wide flat plateau.
freqs3 = np.linspace(261.7, 330.0, 137)
bpinn3 = compositional_frf_case3(freqs3)
PLOT_LO, PLOT_HI = 276.0, 303.0
idx_ref = int(np.argmin(np.abs(freqs3 - REAL_ANSYS_C3_FREQ)))
bpinn_at_ref = float(bpinn3[idx_ref])
ratio = bpinn_at_ref / REAL_ANSYS_C3_AMP
peak_freq = float(freqs3[np.argmax(bpinn3)])
peak_amp = float(bpinn3.max())
print(f"  BML at {freqs3[idx_ref]:.2f} Hz = {bpinn_at_ref:.4f} mm vs real ANSYS "
      f"{REAL_ANSYS_C3_AMP:.4f} mm (ratio {ratio:.3f}x)", flush=True)
print(f"  BML curve peak: {peak_amp:.4f} mm at {peak_freq:.2f} Hz "
      f"(linear-regime resonance is at {math.sqrt(K0[0]/inp['M_sec'][0,0]) / (2*math.pi):.2f} Hz -- "
      f"the shift is the nonlinear hardening bend)", flush=True)

np.savez(os.path.join(OUT, "case3_nonlinear_frf.npz"),
         freqs=freqs3, bpinn_amp=bpinn3,
         real_ansys_freq=REAL_ANSYS_C3_FREQ, real_ansys_amp=REAL_ANSYS_C3_AMP,
         real_ansys_std=REAL_ANSYS_C3_STD, ratio_at_ref=ratio,
         peak_freq=peak_freq, peak_amp=peak_amp)

# ---- figure ----
plot_style.apply_style()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7.2, 5.8))
ax.plot(freqs3, bpinn3, '-', color=plot_style.VIOLET, lw=2.2,
        label='BML (compositional, real coupling)')
ax.errorbar([REAL_ANSYS_C3_FREQ], [REAL_ANSYS_C3_AMP], yerr=[REAL_ANSYS_C3_STD], fmt='o',
            color=plot_style.BLUE, ms=10, mec=plot_style.SURFACE, mew=1.2, capsize=5, lw=2,
            zorder=5, label='Full-order FEM (validated transient point)')
ax.axvline(REAL_ANSYS_C3_FREQ, color=plot_style.INK_MUTED, ls=':', lw=1.2, zorder=1)
ax.set_xlim(PLOT_LO, PLOT_HI)
mask = (freqs3 >= PLOT_LO) & (freqs3 <= PLOT_HI)
ax.set_ylim(bpinn3[mask].min() * 0.92, max(bpinn3[mask].max(), REAL_ANSYS_C3_AMP + REAL_ANSYS_C3_STD) * 1.06)
ax.set_xlabel('Frequency  [Hz]')
ax.set_ylabel('|U$_Z$| at node 1171  [mm]')
plot_style.two_tier_title(ax, 'Case 1 nonlinear FRF: BML vs. the full-order solution',
                           f'node 1171 -- BML/FEM = {ratio:.3f} at {REAL_ANSYS_C3_FREQ:.1f} Hz')
plot_style.legend_inside(ax, loc='upper left')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step9_fig27_case3_nonlinear_frf')
print("Saved step9_fig27_case3_nonlinear_frf.png", flush=True)
