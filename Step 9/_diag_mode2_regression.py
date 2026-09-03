import sys, os, math
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import step4 as s4

inp = s6.load_inputs()
MODE = 2
s6.CONFIG['mode_index'] = MODE
print("s6.CONFIG['target_peak_frac_qref'] =", s6.CONFIG['target_peak_frac_qref'])

K = inp['K_sec'][MODE, MODE]; M = inp['M_sec'][MODE, MODE]; C = inp['C_sec'][MODE, MODE]
K3 = inp['K3_sec_diag'][MODE]
q_ref = 1.0
omega0 = math.sqrt(K / M)
zeta = C / (2 * math.sqrt(K * M))
kappa = 0.75 * K3 * q_ref ** 2 / K
tp = s6.CONFIG['target_peak_frac_qref']
print(f"mode2 nominal: f0={omega0/2/math.pi:.3f} zeta={zeta:.4f} kappa={kappa:.4f}")

cont = s4.duffing_forced_response_continuation(omega0, M, C, K, K3, q_ref, tp)
w_stable = cont['Omega'][cont['stable']] / omega0
amp_true = cont['amplitude'][cont['stable']]
print(f"n stable points: {len(w_stable)}, w range ({w_stable.min():.3f},{w_stable.max():.3f}), amp range ({amp_true.min():.4f},{amp_true.max():.4f})")

OUT6 = s6.OUT
norm2 = dict(np.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_norm.npz')))
state2 = torch.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_state.pt'))
in_dim2 = state2['layers.0.w_mu'].shape[1]
h0 = state2['layers.0.w_mu'].shape[0]; h1 = state2['layers.1.w_mu'].shape[0]
model2 = s6.BPINN(in_dim2, [h0, h1], 2, prior_sigma=1.0)
model2.load_state_dict(state2)
model2.eval()
print(f"in_dim={in_dim2} hidden=[{h0},{h1}] feat_mean len={len(norm2['feat_mean'])}")

# feat_arr for shift=0 (nominal/tuned) -- matches p_true (df_true near 0 for this diagnostic)
feat_arr = np.tile(np.array([0.0, zeta, kappa]), (len(w_stable), 1))

amp_mean, amp_std, _, _ = s6.predict_mc(model2, w_stable, feat_arr, torch.tensor(norm2['feat_mean'],dtype=torch.float32),
                                          torch.tensor(norm2['feat_std'],dtype=torch.float32), n_mc=30,
                                          is_forcing_aware=True, target_peak=tp)
r2 = 1 - np.sum((amp_true-amp_mean)**2)/np.sum((amp_true-amp_true.mean())**2)
print(f"\npredict_mc (new path) R^2 vs true continuation curve: {r2:.4f}")
print("amp_mean sample:", amp_mean[:5])
print("amp_true sample:", amp_true[:5])
