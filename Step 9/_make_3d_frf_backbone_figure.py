# -*- coding: utf-8 -*-
"""
Redesigns the mode-2 nonlinear FRF backbone validation figure (Section
3.5.5) as a 3D waterfall plot, per explicit user request ("too small... is
there another way in 3D or heatwave"). The original overlaid all 5 forcing
levels' True and BPINN curves on one 2D axis, making them hard to tell
apart; here each forcing level gets its own depth slot along a Y axis, so
the 5 True/BPINN curve pairs are visually separated.

Re-derives the exact same real curves as
_validation1_nonlinear_frf_backbone.py (same solver, same trained BPINN
checkpoint, same real ANSYS-motivated disclosure note) -- this script is
self-contained and computationally light (a 5-level continuation sweep
plus a BPINN forward pass, no retraining), so it is safe to re-run rather
than needing cached output.
"""
import sys, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import step4 as s4
import plot_style

torch.manual_seed(42)
MODE = 2
OUT6 = s6.OUT
FORCE_LEVELS = [0.3, 0.5, 0.7, 1.0, 1.5]
COLORS = [plot_style.C_1B, plot_style.C_OK, plot_style.C_HF, plot_style.C_WARN, plot_style.C_ACC]

inp = s6.load_inputs()
s6.CONFIG['mode_index'] = MODE
K = inp['K_sec'][MODE, MODE]; M = inp['M_sec'][MODE, MODE]; C = inp['C_sec'][MODE, MODE]
K3 = inp['K3_sec_diag'][MODE]
q_ref = 1.0
omega0 = math.sqrt(K / M)
zeta = C / (2 * math.sqrt(K * M))
kappa = 0.75 * K3 * q_ref ** 2 / K

norm2 = dict(np.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_norm.npz')))
state2 = torch.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_state.pt'))
in_dim2 = state2['layers.0.w_mu'].shape[1]
h0 = state2['layers.0.w_mu'].shape[0]; h1 = state2['layers.1.w_mu'].shape[0]
model2 = s6.BPINN(in_dim2, [h0, h1], 2, prior_sigma=1.0)
model2.load_state_dict(state2)
model2.eval()


def bpinn_amp(w_arr, tp):
    detune = np.tanh(((w_arr - math.sqrt(1.0)) / zeta) / 20.0)
    feat = np.stack([np.zeros(len(w_arr)), np.full(len(w_arr), zeta), np.full(len(w_arr), kappa),
                      detune, np.full(len(w_arr), tp)], axis=1)
    feat_mean = torch.tensor(norm2['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm2['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    with torch.no_grad():
        pred = model2.forward_mean(X_in).numpy()
    a = pred[:, 0] * float(norm2['alpha_std']) + float(norm2['alpha_mean'])
    b = pred[:, 1] * float(norm2['beta_std']) + float(norm2['beta_mean'])
    return np.hypot(a, b)


s4.CONFIG['continuation']['w_stop_hi'] = 3.0
s4.CONFIG['continuation']['n_steps'] = 4000
W_BPINN_MAX = 1.6
W_START = s4.CONFIG['continuation']['w_start']

plot_style.apply_style()
fig = plt.figure(figsize=(11.5, 9.0))
ax = fig.add_subplot(111, projection="3d")

for depth, (tp, color) in enumerate(zip(FORCE_LEVELS, COLORS)):
    cont = s4.duffing_forced_response_continuation(omega0, M, C, K, K3, q_ref, tp)
    w_curve = cont['Omega'] / omega0
    f_curve = w_curve * omega0 / (2 * math.pi)
    amp_curve = cont['amplitude'].copy()
    stable_mask = cont['stable']
    stable_plot = np.where(stable_mask, amp_curve, np.nan)

    ax.plot(f_curve, np.full_like(f_curve, depth), stable_plot, color=color, lw=2.2,
            label=f"tp={tp}")

    w_local = np.linspace(W_START, W_BPINN_MAX, 250)
    amp_bpinn = bpinn_amp(w_local, tp)
    f_bpinn = w_local * omega0 / (2 * math.pi)
    ax.plot(f_bpinn, np.full_like(f_bpinn, depth), amp_bpinn, color=color, lw=1.8,
            ls="--", alpha=0.9)

ax.set_xlabel("Frequency  [Hz]", labelpad=16, fontsize=15)
ax.set_ylabel("Forcing level (tp)", labelpad=18, fontsize=15)
ax.set_zlabel("Amplitude  [mm]", labelpad=12, fontsize=15)
ax.set_yticks(range(len(FORCE_LEVELS)))
ax.set_yticklabels([str(tp) for tp in FORCE_LEVELS], fontsize=12)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="z", labelsize=12)
ax.set_box_aspect((2.2, 1.0, 0.9))
ax.view_init(elev=22, azim=-62)

from matplotlib.lines import Line2D
legend_elems = [Line2D([0], [0], color=plot_style.INK, lw=2.2, label="True (continuation)"),
                Line2D([0], [0], color=plot_style.INK, lw=1.8, ls="--", label="BPINN (trained range only)")]
fig.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(0.98, 0.9), frameon=False, fontsize=13)

fig.text(0.03, 0.95, "Nonlinear FRF backbone: mode 2, real physics vs. BPINN", fontsize=18,
          fontweight="bold", color=plot_style.INK)
fig.text(0.03, 0.91, "5 forcing levels, each on its own depth slot -- solid = continuation solver, "
                     "dashed = BPINN (trained range only)", fontsize=13, color=plot_style.INK_SECONDARY)
fig.subplots_adjust(left=0.02, right=0.96, top=0.86, bottom=0.06)

figs = os.path.join(r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\figures\step9')
os.makedirs(figs, exist_ok=True)
plot_style.savefig_pub(fig, figs, 'step9_fig13_3d_frf_backbone')
print("Saved step9_fig13_3d_frf_backbone.png")
