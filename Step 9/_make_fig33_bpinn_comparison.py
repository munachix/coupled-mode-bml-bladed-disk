# -*- coding: utf-8 -*-
"""
Adds a real BPINN-predicted point to the "effect of real cross-mode
coupling" comparison (Section 3.5.2 / step9_fig8c), per explicit user
request ("should be BPINN, Compact ROM and ANSYS"). Reuses the EXACT same
validated compositional-reconstruction methodology already used and
published for the Case 3 check (Step 7/_case3_compositional_check.py:
pair-(0,1) forcing-aware BPINN network, driven by its own share of the
real point force via the real mode-shape participation Phi, summed
phase-coherently), adapted to THIS panel's own real loading condition
(1000 N point force at node 1171, w=1.0 i.e. 292.82 Hz, modes 0+1 only --
matching the existing "coupled = real modes 0+1 coupling" panel exactly,
not the 5-mode Case-3 scenario).

Real reference values already established and printed by
step9.py's make_multimode_bpinn_ansys_figure() (hardcoded there from an
earlier real run, unchanged here):
  diagonal-only ROM : 12.84 mm  (mode 0 alone, no coupling)
  coupled ROM       : 1.26 mm   (real modes 0+1 coupling, physics-exact)
  real ANSYS        : 1.04 mm   (different vertex measurement)
"""
import sys, os, math
import numpy as np
import torch
torch.manual_seed(42)
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import step2 as s2
import step7 as s7
import plot_style

COMPACT_ROM_MM = 1.26
REAL_ANSYS_MM = 1.04
F_PHYSICAL_N = 1000.0

print("=== Fig 33: BPINN prediction for the real 1000N@292.82Hz, node 1171, modes 0+1 scenario ===", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]
print(f"  Phi[0]={Phi_all[0]:.6e}, Phi[1]={Phi_all[1]:.6e}", flush=True)

zeta0 = {m: inp['C_sec'][m, m] / (2 * math.sqrt(inp['K_sec'][m, m] * inp['M_sec'][m, m])) for m in (0, 1)}
K0 = {m: inp['K_sec'][m, m] for m in (0, 1)}
# Same real-force-decomposition convention as the validated Case-3 compositional
# check: each mode's own generalized force share is F_physical * Phi_m.
Fg_real = {m: F_PHYSICAL_N * Phi_all[m] for m in (0, 1)}
for m in (0, 1):
    print(f"  mode {m}: Fg_real={Fg_real[m]:.4f} N, zeta0={zeta0[m]:.5f}, K0={K0[m]:.4e}", flush=True)


def predict_pair_signed(model, norm, tp_signed, feat_pair, w_arr):
    """Verbatim from the validated Step 7/_case3_compositional_check.py
    (same sign-symmetry handling, same feature construction)."""
    sign = 1.0 if tp_signed >= 0 else -1.0
    tp_abs = abs(tp_signed)
    n_feat = len(norm['feat_mean'])
    use_detune = n_feat in (8, 9)
    feat_out = s6.add_detune_features(w_arr, feat_pair) if use_detune else feat_pair
    is_fa = bool(norm.get('is_forcing_aware', n_feat == 9))
    if is_fa:
        feat_out = np.concatenate([feat_out, np.full((1, 1), tp_abs)], axis=1)
    feat_mean = torch.tensor(norm['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat_out, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    model.eval()
    with torch.no_grad():
        samples = np.array([model(X_in).numpy() for _ in range(300)])
    ai = sign * float((samples[:, 0, 0] * float(norm['alpha_i_std']) + float(norm['alpha_i_mean'])).mean())
    bi = sign * float((samples[:, 0, 1] * float(norm['beta_i_std']) + float(norm['beta_i_mean'])).mean())
    aj = sign * float((samples[:, 0, 2] * float(norm['alpha_j_std']) + float(norm['alpha_j_mean'])).mean())
    bj = sign * float((samples[:, 0, 3] * float(norm['beta_j_std']) + float(norm['beta_j_mean'])).mean())
    return ai, bi, aj, bj


model01, norm01 = s7.load_bpinn_coupled((0, 1))
tp01 = Fg_real[0] / (2 * zeta0[0] * K0[0])
feat01 = np.array([[0.0, zeta0[0], 0.75 * inp['K3_sec_diag'][0] / K0[0],
                     0.0, zeta0[1], 0.75 * inp['K3_sec_diag'][1] / K0[1]]])
w_arr = np.array([1.0])
a0, b0, a1, b1 = predict_pair_signed(model01, norm01, tp01, feat01, w_arr)
print(f"  pair(0,1): tp01={tp01:.5f}, mode0=(a={a0:.5e},b={b0:.5e}), mode1=(a={a1:.5e},b={b1:.5e})", flush=True)

u_complex = (a0 - 1j * b0) * Phi_all[0] + (a1 - 1j * b1) * Phi_all[1]
u_bpinn_mm = abs(u_complex)

print(f"\n{'='*70}")
print(f"BPINN prediction, node 1171, 1000N @ 292.82Hz, modes 0+1 coupled: {u_bpinn_mm:.4f} mm")
print(f"Compact ROM (physics-exact, real coupling): {COMPACT_ROM_MM:.4f} mm")
print(f"Real ANSYS measurement: {REAL_ANSYS_MM:.4f} mm")
print(f"{'='*70}")

np.savez(os.path.join(s6.OUT, 'fig33_bpinn_comparison.npz'),
          u_bpinn_mm=u_bpinn_mm, compact_rom_mm=COMPACT_ROM_MM, real_ansys_mm=REAL_ANSYS_MM,
          alpha0=a0, beta0=b0, alpha1=a1, beta1=b1)

# ---- regenerate the figure itself ----
plot_style.apply_style()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.5, 5.6))
labels3 = ['BPINN\n(surrogate)', 'Compact ROM\n(physics-exact)', 'Real ANSYS\nmeasurement']
values3 = [u_bpinn_mm, COMPACT_ROM_MM, REAL_ANSYS_MM]
colors3 = [plot_style.C_ACC, plot_style.C_OK, plot_style.BLUE]
bars = ax.bar(labels3, values3, color=colors3)
for b, v in zip(bars, values3):
    ax.annotate(f"{v:.2f} mm", (b.get_x() + b.get_width() / 2, v),
                textcoords='offset points', xytext=(0, 8), ha='center', fontsize=13, weight='bold')
ax.set_ylabel('Predicted / measured displacement [mm]')
plot_style.two_tier_title(ax, 'BPINN vs. Compact ROM vs. real ANSYS: 1000N @ 292.82Hz, node 1171',
                           'both models use real modes 0+1 coupling; ANSYS = different vertex measurement')
fig.tight_layout()
figs = os.path.join(r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\figures\step9')
plot_style.savefig_pub(fig, figs, 'step9_fig8c_bpinn_compact_rom_ansys')
print("Saved step9_fig8c_bpinn_compact_rom_ansys.png")
