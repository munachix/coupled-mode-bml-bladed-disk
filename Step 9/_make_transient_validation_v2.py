# -*- coding: utf-8 -*-
"""
Redo of the Section 3.5.3 transient validation (v2, 2026-08-31, explicit
user request: "lacks the real validation I was looking for... redo it").

v1 compared the real ANSYS transient's own converged tail against the
compositional B-PINN's steady-state (alpha,beta) prediction, phase-ALIGNED
to the data via least-squares fit -- a legitimate but narrower check ("does
the network's predicted amplitude, once phase-matched to the data, trace
the same waveform"), and one that could read as fitting rather than an
independent prediction.

v2 instead performs a genuine forward TIME INTEGRATION of the same real,
already-published coupled nonlinear ODEs used throughout this paper's own
Compact ROM (Step 4's duffing_forced_response_coupled, the same function
driving the Section 3.3.4 waveform gallery and validated directly against
real ANSYS in Step 4's own docstring): real K/M/C for modes 0,1,2,3,4, the
real measured cross-mode cubic coupling coefficients for pairs (0,1) and
(3,4), and mode 2's own real cubic coefficient for its (trivial, isolated)
single-mode equation, each driven by its own real share of the SAME
physical point load used throughout Section 3.5 (F_physical = 2500N /
Phi[0], the calibration matching this project's real ANSYS harmonic-run
convention). Integrated from rest (q=0, v=0) under the real force for long
enough to reach a converged limit cycle, with NO fitting of any kind: the
three integrations share one time base and one forcing-phase convention
(cos(Omega*t)) by construction, so the resulting node-1171 waveform
(summed via the real mode-shape participation Phi_m) is an independent
forward prediction, not a fit to the real ANSYS data it is compared
against. Its converged tail is compared directly against the real ANSYS
transient's own converged tail (the same real run used in v1).
"""
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step4 as s4
import step6 as s6
import step2 as s2
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

REAL_ANSYS_AMP = 1.2220
REAL_ANSYS_STD = 0.0190
REAL_ANSYS_FREQ = 292.82

print("=== Section 3.5.3 v2: real time-integrated coupled-mode nonlinear ODEs vs. real ANSYS transient ===", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]

K = {m: inp['K_sec'][m, m] for m in range(5)}
M = {m: inp['M_sec'][m, m] for m in range(5)}
C = {m: inp['C_sec'][m, m] for m in range(5)}
K3_diag = {m: inp['K3_sec_diag'][m] for m in range(5)}

F_physical = 2500.0 / Phi_all[0]
Fg = {m: F_physical * Phi_all[m] for m in range(5)}
omega0_mode0 = math.sqrt(K[0] / M[0])
Omega = omega0_mode0  # w = 1.0, the real validated ANSYS transient point

print(f"  F_physical={F_physical:.4f} N (same real-force calibration as Section 3.5.1-3.5.2)", flush=True)
for m in range(5):
    print(f"  mode {m}: Fg={Fg[m]:.4f} N, K={K[m]:.4e}, M={M[m]:.6f}, C={C[m]:.6e}", flush=True)

n_cycles = 300
steps_per_cycle = 40
T_final = n_cycles * 2 * np.pi / Omega
t_eval = np.linspace(0, T_final, n_cycles * steps_per_cycle)

# ---- pair (0,1): real coupled 2-mode time integration (Step 4's own,
# already-published, already-validated-against-real-ANSYS function) ----
cc01 = s4.CONFIG["nonlinear"]["cross_coupling"][(0, 1)]
r01 = s4.duffing_forced_response_coupled(
    (0, 1), (K[0], K[1]), (M[0], M[1]), (C[0], C[1]),
    cc01["coef0"], cc01["coef1"], (Fg[0], Fg[1]), Omega,
    n_cycles=n_cycles, steps_per_cycle=steps_per_cycle)
q0_t = np.interp(t_eval, r01["t"], r01["q_i"])
q1_t = np.interp(t_eval, r01["t"], r01["q_j"])
print("  pair(0,1) integrated.", flush=True)

# ---- pair (3,4): same, real coupling coefficients for this pair ----
cc34 = s4.CONFIG["nonlinear"]["cross_coupling"][(3, 4)]
r34 = s4.duffing_forced_response_coupled(
    (3, 4), (K[3], K[4]), (M[3], M[4]), (C[3], C[4]),
    cc34["coef0"], cc34["coef1"], (Fg[3], Fg[4]), Omega,
    n_cycles=n_cycles, steps_per_cycle=steps_per_cycle)
q3_t = np.interp(t_eval, r34["t"], r34["q_i"])
q4_t = np.interp(t_eval, r34["t"], r34["q_j"])
print("  pair(3,4) integrated.", flush=True)

# ---- mode 2: isolated single-mode Duffing oscillator (no measured
# cross-mode coupling pathway from mode 0/1/3/4, Section 3.5.1's own
# isolation check), real K3, direct time integration from rest ----
K3_2 = K3_diag[2]


def rhs_mode2(t, y):
    q, v = y
    Fdrive = Fg[2] * np.cos(Omega * t)
    a = (Fdrive - C[2] * v - K[2] * q - K3_2 * q ** 3) / M[2]
    return [v, a]


sol2 = solve_ivp(rhs_mode2, [0, T_final], [0.0, 0.0], t_eval=t_eval, method="RK45",
                  rtol=1e-8, atol=1e-10, max_step=(2 * np.pi / Omega) / steps_per_cycle)
q2_t = sol2.y[0]
print("  mode 2 (isolated) integrated.", flush=True)

# ---- sum via real mode-shape participation ----
u_pred = q0_t * Phi_all[0] + q1_t * Phi_all[1] + q2_t * Phi_all[2] + q3_t * Phi_all[3] + q4_t * Phi_all[4]

# ---- converged tail: last 4 forcing periods ----
period = 2 * np.pi / Omega
tail_mask = t_eval >= (t_eval[-1] - 4 * period)
u_tail_pred = u_pred[tail_mask]
t_tail_pred = t_eval[tail_mask]
pred_tail_amp = (u_tail_pred.max() - u_tail_pred.min()) / 2.0
print(f"  Predicted converged amplitude (from-rest time integration): {pred_tail_amp:.4f} mm", flush=True)

# ---- real ANSYS transient (same real run as v1) ----
d = np.load(r'F:\ANSYS PCE\ROM_data_sensitivity\case3_transient\transient_point_warmstart_w1.000.npz')
t_fem = d['t']
u_fem = d['u']
tail_mask_fem = t_fem >= (t_fem[-1] - 4 * period)
t_tail_fem = t_fem[tail_mask_fem]
u_tail_fem = u_fem[tail_mask_fem]
fem_tail_amp = (u_tail_fem.max() - u_tail_fem.min()) / 2.0
fem_tail_mean = u_tail_fem.mean()

amp_err_pct = abs(pred_tail_amp - fem_tail_amp) / fem_tail_amp * 100.0
print(f"  Real ANSYS converged amplitude (same real run as Section 3.5.1): {fem_tail_amp:.4f} mm")
print(f"  Amplitude error (independent time integration vs. real ANSYS): {amp_err_pct:.2f}%")

np.savez(os.path.join(OUT, "transient_validation_v2.npz"),
         t_eval=t_eval, u_pred=u_pred, t_tail_pred=t_tail_pred, u_tail_pred=u_tail_pred,
         t_fem=t_fem, u_fem=u_fem, t_tail_fem=t_tail_fem, u_tail_fem=u_tail_fem,
         pred_tail_amp=pred_tail_amp, fem_tail_amp=fem_tail_amp, amp_err_pct=amp_err_pct,
         real_ansys_amp=REAL_ANSYS_AMP, real_ansys_std=REAL_ANSYS_STD)
print("Saved transient_validation_v2.npz")
