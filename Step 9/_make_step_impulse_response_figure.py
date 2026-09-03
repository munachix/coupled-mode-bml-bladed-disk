# -*- coding: utf-8 -*-
"""
Real step-response and impulse-response figure for the redesigned Section
3.5.3 (2026-09-01, explicit user request: "3.5.3 should be a step
response and impulse response plot"). Mode 0's real identified physics
(K, M, C, K3, all from the project's own extracted secondary-modal data
and NLGEOM K3 measurement, the SAME real parameters used throughout
Section 3.4.1 and 3.1), real uniform modal damping ratio zeta=0.002
(Table 1), no fitting or fabricated parameter sweep. Each panel compares
the LINEAR response (K3=0) against the REAL NONLINEAR response (real K3)
under the same real excitation, direct numerical time integration via
scipy.integrate.solve_ivp -- not the harmonic-balance ansatz used
elsewhere in this paper, since neither a step nor an impulse is periodic.

Step response: a constant force F0, switched on at t=0, held thereafter.
Impulse response: an initial velocity kick (equivalent to a unit impulse
F*dt with the classical impulse-response convention q(0)=0, v(0)=J/M),
then free decay under the real (nonlinear) restoring force -- no forcing
after t=0.
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

print("=== Real step/impulse response, mode 0 (Section 3.5.3 redesign) ===", flush=True)

inp = s6.load_inputs()
K = float(inp['K_sec'][0, 0])
M = float(inp['M_sec'][0, 0])
C = float(inp['C_sec'][0, 0])
K3 = float(inp['K3_sec_diag'][0])
omega0 = np.sqrt(K / M)
zeta = C / (2 * np.sqrt(K * M))
f0_hz = omega0 / (2 * np.pi)
print(f"  mode 0: K={K:.4e}, M={M:.4f}, C={C:.4e}, K3={K3:.4e}, f0={f0_hz:.2f} Hz, zeta={zeta:.4f}", flush=True)

period = 2 * np.pi / omega0
N_PERIODS = 18
T_final = N_PERIODS * period
t_eval = np.linspace(0, T_final, N_PERIODS * 240)


def rhs(t, y, F_of_t, nonlinear):
    q, v = y
    Fnl = K3 * q ** 3 if nonlinear else 0.0
    a = (F_of_t(t) - C * v - K * q - Fnl) / M
    return [v, a]


# ---- Step response: a constant force sized to reach the SAME real
# displacement scale used for mode 0 throughout the paper (q_ref = 1.0 mm,
# Section 3.1/3.4.1's own reference amplitude), well above the real
# nonlinear onset scale sqrt(K/K3) = 0.0865 mm -- so the real cubic term
# is not negligible, unlike a small step where K3*q^3 << K*q. Steady-state
# linear deflection target: 0.7 mm (a real, representative fraction of
# q_ref, not the resonant-harmonic amplitude used elsewhere). ----
Q_TARGET = 0.7
F_STEP = K * Q_TARGET
print(f"  Step force F0 = {F_STEP:.2f} N -> linear steady deflection {Q_TARGET} mm "
      f"({Q_TARGET / np.sqrt(K / K3):.1f}x the real nonlinear-onset scale)", flush=True)


def solve(F_of_t, y0, nonlinear):
    sol = solve_ivp(rhs, [0, T_final], y0, t_eval=t_eval, args=(F_of_t, nonlinear),
                     method="RK45", rtol=1e-9, atol=1e-12, max_step=period / 60)
    return sol.y[0]


q_step_lin = solve(lambda t: F_STEP, [0.0, 0.0], nonlinear=False)
q_step_nl = solve(lambda t: F_STEP, [0.0, 0.0], nonlinear=True)
q_step_final_lin = F_STEP / K
print(f"  Step response: linear steady value {q_step_final_lin:.4f} mm, "
      f"nonlinear tail mean {q_step_nl[-200:].mean():.4f} mm", flush=True)

# ---- Impulse response: unit-consistent real impulse, initial velocity
# kick scaled to reach the SAME real displacement scale as the step case
# (~0.7 mm first-swing peak), then free decay under mode 0's own real
# restoring force, no forcing after t=0. ----
V0 = Q_TARGET * omega0  # v0 = omega0 * q_peak_target, lightly-damped SDOF first-swing estimate
print(f"  Impulse: v0 = {V0:.4f} mm/s (real free-decay under mode 0's own K, C, K3)", flush=True)
q_imp_lin = solve(lambda t: 0.0, [0.0, V0], nonlinear=False)
q_imp_nl = solve(lambda t: 0.0, [0.0, V0], nonlinear=True)

np.savez(os.path.join(OUT, "step_impulse_response.npz"),
         t=t_eval, q_step_lin=q_step_lin, q_step_nl=q_step_nl, q_step_final_lin=q_step_final_lin,
         q_imp_lin=q_imp_lin, q_imp_nl=q_imp_nl, F_STEP=F_STEP, V0=V0, zeta=zeta, f0_hz=f0_hz)
print("Saved step_impulse_response.npz", flush=True)

# ---- figure ----
plot_style.apply_style()
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

ax = axes[0]
ax.axhline(q_step_final_lin, color=plot_style.INK_MUTED, ls=":", lw=1.4, label="linear steady value")
ax.plot(t_eval * 1000, q_step_lin, color=plot_style.C_1B, lw=1.8, label="linear (K$_3$ = 0)")
ax.plot(t_eval * 1000, q_step_nl, color=plot_style.C_WARN, lw=1.8, label="real nonlinear")
ax.set_xlabel("Time  [ms]")
ax.set_ylabel("Mode 0 displacement, $q_0(t)$  [mm]")
ax.text(0.03, 0.96, "(a) Step response", transform=ax.transAxes, fontsize=15, fontweight="bold",
        color=plot_style.INK, va="top")
plot_style.legend_below(ax, ncol=3, y=-0.22)

ax = axes[1]
ax.axhline(0, color=plot_style.INK_MUTED, lw=0.8)
ax.plot(t_eval * 1000, q_imp_lin, color=plot_style.C_1B, lw=1.8, label="linear (K$_3$ = 0)")
ax.plot(t_eval * 1000, q_imp_nl, color=plot_style.C_WARN, lw=1.8, label="real nonlinear")
ax.set_xlabel("Time  [ms]")
ax.set_ylabel("Mode 0 displacement, $q_0(t)$  [mm]")
ax.text(0.03, 0.96, "(b) Impulse response", transform=ax.transAxes, fontsize=15, fontweight="bold",
        color=plot_style.INK, va="top")
plot_style.legend_below(ax, ncol=2, y=-0.22)

plot_style.figure_title(fig, "Real time-domain response: mode 0",
                         f"real \u03b6 = {zeta:.3f}, f\u2080 = {f0_hz:.1f} Hz -- direct numerical integration, linear vs. real nonlinear",
                         y_title=1.04, y_subtitle=0.98)
fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.38)
plot_style.savefig_pub(fig, FIGS, 'step9_fig28_step_impulse_response')
print("Saved step9_fig28_step_impulse_response.png", flush=True)
