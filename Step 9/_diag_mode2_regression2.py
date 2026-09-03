import sys, os, math
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step6 as s6
import step4 as s4

OUT6 = s6.OUT
norm2 = dict(np.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_norm.npz')))
state2 = torch.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_state.pt'))
in_dim2 = state2['layers.0.w_mu'].shape[1]
h0 = state2['layers.0.w_mu'].shape[0]; h1 = state2['layers.1.w_mu'].shape[0]
model2 = s6.BPINN(in_dim2, [h0, h1], 2, prior_sigma=1.0)
model2.load_state_dict(state2)
model2.eval()

zeta = 0.0020; kappa = 71.2727
w_arr = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
feat_arr = np.tile(np.array([0.0, zeta, kappa]), (len(w_arr), 1))
tp = 0.8
detune = np.tanh(((w_arr - np.sqrt(1.0)) / zeta) / 20.0)
print("detune:", detune)
feat_full = np.concatenate([feat_arr, detune[:,None], np.full((len(w_arr),1), tp)], axis=1)
print("feat_full:\n", feat_full)
feat_mean = torch.tensor(norm2['feat_mean'], dtype=torch.float32)
feat_std = torch.tensor(norm2['feat_std'], dtype=torch.float32)
Feat_n = (torch.tensor(feat_full, dtype=torch.float32) - feat_mean) / feat_std
print("Feat_n:\n", Feat_n)
W_t = torch.tensor(w_arr, dtype=torch.float32)
fenc = s6.fourier_encode_w(W_t)
print("fourier_encode_w shape:", fenc.shape, "sample row0:", fenc[0][:6])
X_in = torch.cat([fenc, Feat_n], dim=1)
print("X_in shape:", X_in.shape)
with torch.no_grad():
    pred_mean = model2.forward_mean(X_in)
print("forward_mean raw output (normalized alpha,beta):\n", pred_mean)
a = pred_mean[:,0]*float(norm2['alpha_std'])+float(norm2['alpha_mean'])
b = pred_mean[:,1]*float(norm2['beta_std'])+float(norm2['beta_mean'])
print("amp (deterministic):", np.hypot(a.numpy(), b.numpy()))
