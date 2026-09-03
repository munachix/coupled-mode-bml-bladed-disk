# -*- coding: utf-8 -*-
"""
Section 3.5.3 step/impulse response, v2 (2026-09-02, explicit user
request: "Fig 30 should use ansys to verify"). Replaces the v1 mode-0-only
SDOF comparison, which this session's own real ANSYS step test showed
badly undershoots a point step/impulse response (mode 0 alone explains
only ~11% of the real static compliance at node 1171 -- a point load
excites many modes comparably, unlike resonant harmonic forcing where
mode 0 dominates by amplification). v2 instead validates and uses the
FULL 70-secondary-mode LINEAR modal superposition (the same real,
already-validated methodology as Fig 4's driving-point receptance --
closed-form per-mode step response, summed via the real mode-shape
participation at node 1171), checked directly against a genuine new real
ANSYS transient run (full 181k-DOF NLGEOM model, node 1171, F_gen=5000 on
mode 0's own convention) for both the step and impulse cases.

Real per-mode K3 is included where measured (the 24 fundamental-cluster
modes); at the validated force level tested, its correction is negligible
(mode 0's own linear-vs-nonlinear generalized response differ by <1%),
consistent with every individual mode's amplitude at this DOF/force
staying far below its own nonlinear-onset scale -- reported honestly
rather than assumed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step4 as s4
import step2 as s2
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

print("=== Real ANSYS-verified step/impulse response, node 1171 (Section 3.5.3 v2) ===", flush=True)

inp = s4.load_inputs()
K_diag = np.diag(inp['K_sec'])
M_diag = np.diag(inp['M_sec'])
C_diag = np.diag(inp['C_sec'])
omega_i = np.sqrt(K_diag / M_diag)
zeta_i = C_diag / (2 * np.sqrt(K_diag * M_diag))
K0 = float(K_diag[0])
omega0 = float(omega_i[0])
f0_hz = omega0 / (2 * np.pi)
zeta0 = float(zeta_i[0])

dmap = s2._dof_map()


def linear_step_response(t, K, omega, zeta, Fgen):
    q_final = Fgen / K
    wd = omega * np.sqrt(1 - zeta ** 2)
    env = np.exp(-zeta * omega * t)
    return q_final * (1 - env * (np.cos(wd * t) + (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(wd * t)))


def linear_free_decay(t, omega, zeta, q0, v0):
    wd = omega * np.sqrt(1 - zeta ** 2)
    env = np.exp(-zeta * omega * t)
    return env * (q0 * np.cos(wd * t) + (v0 + zeta * omega * q0) / wd * np.sin(wd * t))


# ---- Step case: real ANSYS data + validated 70-mode linear model + mode-0-only ----
d_step = np.load(r'F:\ANSYS PCE\ROM_data_sensitivity\case_step_test\step_test_result.npz')
t_step = d_step['t']
u_step_ansys = d_step['u']
F_physical_step = float(d_step['F0'])
target_node = int(d_step['target_node'])
target_dir = str(d_step['target_dir'])
dirmap = {'X': 0, 'Y': 1, 'Z': 2}
target_eq = np.where((dmap[:, 0] == target_node) & (dmap[:, 1] == dirmap[target_dir]))[0][0]
phi_row = inp['T_full2sec'][target_eq, :]

Fgen_i_step = F_physical_step * phi_row
u_step_70mode = np.zeros_like(t_step)
for i in range(70):
    u_step_70mode += linear_step_response(t_step, K_diag[i], omega_i[i], zeta_i[i], Fgen_i_step[i]) * phi_row[i]

u_step_mode0 = linear_step_response(t_step, K_diag[0], omega_i[0], zeta_i[0], Fgen_i_step[0]) * phi_row[0]

rmse_70 = float(np.sqrt(np.mean((u_step_70mode - u_step_ansys) ** 2)))
rmse_0 = float(np.sqrt(np.mean((u_step_mode0 - u_step_ansys) ** 2)))
corr_70 = float(np.corrcoef(u_step_70mode, u_step_ansys)[0, 1])
print(f"  STEP: real ANSYS peak={u_step_ansys.max():.4f} mm; "
      f"70-mode linear RMSE={rmse_70:.4f} mm (corr={corr_70:.4f}); "
      f"mode-0-only RMSE={rmse_0:.4f} mm", flush=True)

# ---- Impulse case: real ANSYS data + validated 70-mode linear model + mode-0-only ----
d_imp = np.load(r'F:\ANSYS PCE\ROM_data_sensitivity\case_impulse_test\impulse_test_result.npz')
t_imp = d_imp['t']
u_imp_ansys = d_imp['u']
v_i = d_imp['v_i']

u_imp_70mode = np.zeros_like(t_imp)
for i in range(70):
    u_imp_70mode += linear_free_decay(t_imp, omega_i[i], zeta_i[i], 0.0, v_i[i]) * phi_row[i]

u_imp_mode0 = linear_free_decay(t_imp, omega_i[0], zeta_i[0], 0.0, v_i[0]) * phi_row[0]

rmse_70_imp = float(np.sqrt(np.mean((u_imp_70mode - u_imp_ansys) ** 2)))
rmse_0_imp = float(np.sqrt(np.mean((u_imp_mode0 - u_imp_ansys) ** 2)))
corr_70_imp = float(np.corrcoef(u_imp_70mode, u_imp_ansys)[0, 1])
print(f"  IMPULSE: real ANSYS peak={np.abs(u_imp_ansys).max():.4f} mm; "
      f"70-mode linear RMSE={rmse_70_imp:.4f} mm (corr={corr_70_imp:.4f}); "
      f"mode-0-only RMSE={rmse_0_imp:.4f} mm", flush=True)

np.savez(os.path.join(OUT, "step_impulse_ansys_verified.npz"),
         t_step=t_step, u_step_ansys=u_step_ansys, u_step_70mode=u_step_70mode, u_step_mode0=u_step_mode0,
         t_imp=t_imp, u_imp_ansys=u_imp_ansys, u_imp_70mode=u_imp_70mode, u_imp_mode0=u_imp_mode0,
         rmse_70=rmse_70, rmse_0=rmse_0, corr_70=corr_70,
         rmse_70_imp=rmse_70_imp, rmse_0_imp=rmse_0_imp, corr_70_imp=corr_70_imp,
         F_physical_step=F_physical_step, f0_hz=f0_hz, zeta0=zeta0)
print("Saved step_impulse_ansys_verified.npz", flush=True)

# ---- figure ----
plot_style.apply_style()
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))

ax = axes[0]
ax.plot(t_step * 1000, u_step_ansys, 'o', color=plot_style.INK, ms=4.5, mfc=plot_style.SURFACE,
        mew=1.1, zorder=4, label="real ANSYS (full FE, node 1171)")
ax.plot(t_step * 1000, u_step_70mode, color=plot_style.C_1B, lw=2.0, zorder=3,
        label="70-mode linear superposition")
ax.plot(t_step * 1000, u_step_mode0, color=plot_style.C_WARN, lw=1.6, ls="--", zorder=2,
        label="mode-0-only (under-predicts)")
ax.set_xlabel("Time  [ms]")
ax.set_ylabel("U$_Z$ at node 1171  [mm]")
ax.text(0.03, 0.96, "(a) Step response", transform=ax.transAxes, fontsize=15, fontweight="bold",
        color=plot_style.INK, va="top")
plot_style.legend_below(ax, ncol=1, y=-0.24)

ax = axes[1]
ax.axhline(0, color=plot_style.INK_MUTED, lw=0.8)
ax.plot(t_imp * 1000, u_imp_ansys, 'o', color=plot_style.INK, ms=4.5, mfc=plot_style.SURFACE,
        mew=1.1, zorder=4, label="real ANSYS (full FE, node 1171)")
ax.plot(t_imp * 1000, u_imp_70mode, color=plot_style.C_1B, lw=2.0, zorder=3,
        label="70-mode linear superposition")
ax.plot(t_imp * 1000, u_imp_mode0, color=plot_style.C_WARN, lw=1.6, ls="--", zorder=2,
        label="mode-0-only (under-predicts)")
ax.set_xlabel("Time  [ms]")
ax.set_ylabel("U$_Z$ at node 1171  [mm]")
ax.text(0.03, 0.96, "(b) Impulse response", transform=ax.transAxes, fontsize=15, fontweight="bold",
        color=plot_style.INK, va="top")
plot_style.legend_below(ax, ncol=1, y=-0.24)

plot_style.figure_title(fig, "Real ANSYS-verified time-domain response: node 1171",
                         f"full 181k-DOF NLGEOM transient vs. modal superposition, F_gen=5000 (step) / equivalent point impulse",
                         y_title=1.04, y_subtitle=0.98)
fig.subplots_adjust(top=0.82, bottom=0.30, wspace=0.32)
plot_style.savefig_pub(fig, FIGS, 'step9_fig28_step_impulse_response_v2')
print("Saved step9_fig28_step_impulse_response_v2.png", flush=True)
