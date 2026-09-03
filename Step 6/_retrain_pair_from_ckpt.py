"""Fast retrain-only script: reloads a pair's cached dataset checkpoint
(from _train_pair_bpinn.py's first run) and retrains with a different
PHYSICS_WEIGHT, skipping the ~5-6 min ODE data-generation phase -- for
quickly calibrating the physics-loss weight (2026-08-13), same pattern as
the original modes-0-1 retrain script.

Usage: python _retrain_pair_from_ckpt.py <mode_i> <mode_j> <physics_weight>
"""
import sys, os, time, math
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
import step6 as s6
import step4 as s4

MODE_I, MODE_J = int(sys.argv[1]), int(sys.argv[2])
PHYSICS_WEIGHT = float(sys.argv[3])
TARGET_PEAK = 1.0
OUT = s6.OUT
TAG = f'{MODE_I}{MODE_J}'

t_start = time.time()
inp = s6.load_inputs()
cc = s4.CONFIG['nonlinear']['cross_coupling'][(MODE_I, MODE_J)]
Ki0 = inp['K_sec'][MODE_I, MODE_I]; Kj0 = inp['K_sec'][MODE_J, MODE_J]
M_i = inp['M_sec'][MODE_I, MODE_I]; M_j = inp['M_sec'][MODE_J, MODE_J]
C_i = inp['C_sec'][MODE_I, MODE_I]; C_j = inp['C_sec'][MODE_J, MODE_J]
omega0_i = math.sqrt(Ki0 / M_i)
zeta_i0 = C_i / (2 * math.sqrt(Ki0 * M_i)); zeta_j0 = C_j / (2 * math.sqrt(Kj0 * M_j))
Fg_i = TARGET_PEAK * 2 * zeta_i0 * Ki0; Fg_j = TARGET_PEAK * 2 * zeta_j0 * Kj0
F_scale = max(abs(Fg_i), abs(Fg_j))

ckpt = np.load(os.path.join(OUT, f'_coupled_dataset_checkpoint_{TAG}.npz'))
W_t = torch.tensor(ckpt['train_w'], dtype=torch.float32)
Feat_t = torch.tensor(ckpt['train_feat'], dtype=torch.float32)
K_i_t = torch.tensor(ckpt['train_K_i'], dtype=torch.float32)
K_j_t = torch.tensor(ckpt['train_K_j'], dtype=torch.float32)
Omega_t = W_t * omega0_i
Ai_raw = ckpt['train_alpha_i']; Bi_raw = ckpt['train_beta_i']
Aj_raw = ckpt['train_alpha_j']; Bj_raw = ckpt['train_beta_j']
Ai_mean, Ai_std = Ai_raw.mean(), Ai_raw.std()
Bi_mean, Bi_std = Bi_raw.mean(), Bi_raw.std()
Aj_mean, Aj_std = Aj_raw.mean(), Aj_raw.std()
Bj_mean, Bj_std = Bj_raw.mean(), Bj_raw.std()
Ai_t = torch.tensor((Ai_raw - Ai_mean) / Ai_std, dtype=torch.float32)
Bi_t = torch.tensor((Bi_raw - Bi_mean) / Bi_std, dtype=torch.float32)
Aj_t = torch.tensor((Aj_raw - Aj_mean) / Aj_std, dtype=torch.float32)
Bj_t = torch.tensor((Bj_raw - Bj_mean) / Bj_std, dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)
coef0_t = [float(x) for x in cc['coef0']]; coef1_t = [float(x) for x in cc['coef1']]

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [32, 32], 4, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 6000
KL_BETA = 0.001
n_data = float(len(Ai_raw))
print(f"Retraining pair ({MODE_I},{MODE_J}) with PHYSICS_WEIGHT={PHYSICS_WEIGHT:.2e}...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    ai_p, bi_p, aj_p, bj_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    data_loss = ((ai_p - Ai_t) ** 2 + (bi_p - Bi_t) ** 2 + (aj_p - Aj_t) ** 2 + (bj_p - Bj_t) ** 2).mean()
    ai_phys = ai_p * Ai_std + Ai_mean; bi_phys = bi_p * Bi_std + Bi_mean
    aj_phys = aj_p * Aj_std + Aj_mean; bj_phys = bj_p * Bj_std + Bj_mean
    Ra_i, Rb_i, Ra_j, Rb_j = s4.coupled_hbm_residual(
        (K_i_t, K_j_t), (M_i, M_j), (C_i, C_j), coef0_t, coef1_t, (Fg_i, Fg_j), Omega_t,
        ai_phys, bi_phys, aj_phys, bj_phys)
    physics_loss = ((Ra_i / F_scale) ** 2 + (Rb_i / F_scale) ** 2 + (Ra_j / F_scale) ** 2 + (Rb_j / F_scale) ** 2).mean()
    kl = model.total_kl() / n_data
    anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
    loss = data_loss + anneal * PHYSICS_WEIGHT * physics_loss + anneal * KL_BETA * kl
    loss.backward()
    opt.step()
    if epoch % 2000 == 0:
        print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  physics={physics_loss.item():.6f}  "
              f"weighted_phys={(anneal*PHYSICS_WEIGHT*physics_loss).item():.6f}  kl={kl.item():.5f}", flush=True)

Alpha_i_test_raw = ckpt['test_alpha_i']; Beta_i_test_raw = ckpt['test_beta_i']
Alpha_j_test_raw = ckpt['test_alpha_j']; Beta_j_test_raw = ckpt['test_beta_j']
Amp_i_test_raw = ckpt['test_amp_i']; Amp_j_test_raw = ckpt['test_amp_j']
W_test = torch.tensor(ckpt['test_w'], dtype=torch.float32)
Feat_test = torch.tensor(ckpt['test_feat'], dtype=torch.float32)
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)

model.eval()
preds = []
with torch.no_grad():
    for _ in range(30):
        preds.append(model(X_test).numpy())
preds = np.array(preds)
alpha_i_mean = preds[:, :, 0].mean(0) * Ai_std + Ai_mean
beta_i_mean = preds[:, :, 1].mean(0) * Bi_std + Bi_mean
alpha_j_mean = preds[:, :, 2].mean(0) * Aj_std + Aj_mean
beta_j_mean = preds[:, :, 3].mean(0) * Bj_std + Bj_mean
amp_i_pred = np.hypot(alpha_i_mean, beta_i_mean); amp_j_pred = np.hypot(alpha_j_mean, beta_j_mean)

def r2(true, pred):
    return 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)

print(f"\nWEIGHT={PHYSICS_WEIGHT:.2e}: amp R^2: mode_i={r2(Amp_i_test_raw, amp_i_pred):.4f} "
      f"mode_j={r2(Amp_j_test_raw, amp_j_pred):.4f}  |  "
      f"phase R^2: a_i={r2(Alpha_i_test_raw, alpha_i_mean):.4f} b_i={r2(Beta_i_test_raw, beta_i_mean):.4f} "
      f"a_j={r2(Alpha_j_test_raw, alpha_j_mean):.4f} b_j={r2(Beta_j_test_raw, beta_j_mean):.4f}", flush=True)
print(f"TIME: {time.time()-t_start:.0f}s", flush=True)
