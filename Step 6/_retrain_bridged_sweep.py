"""Cheap retrain sweep for the bridged 0-1-2-3-4 BPINN (2026-08-23) --
reuses the already-generated dataset (_bridged_dataset_checkpoint.npz, 2.75
hours of real ODE solves, cached) so every config here is training-only
(minutes, not hours). Sweeps network capacity, epoch count, and physics
weight to find what actually closes the gap between the first attempt's
weak R^2 (mean amplitude 0.46, mode 1 as low as 0.11) and the pair BPINNs'
usual 0.83-0.99 range.
"""
import sys, os, time, math, itertools
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 5')
import step6 as s6
import step4 as s4
import step2 as s2
import step5 as s5

CHAIN = [0, 1, 2, 3, 4]
N_CHAIN = len(CHAIN)
TARGET_NODE, TARGET_DIR = 1171, 'Z'
OUT = s6.OUT

d = np.load(os.path.join(OUT, '_bridged_dataset_checkpoint.npz'))
train_w = d['train_w']; train_feat6 = d['train_feat']; train_K = d['train_K_arr']
train_alpha = d['train_alpha']; train_beta = d['train_beta']
test_w = d['test_w']; test_feat6 = d['test_feat']
test_alpha = d['test_alpha']; test_beta = d['test_beta']; test_amp = d['test_amp']
print(f"Loaded cached dataset: {len(train_w)} train, {len(test_w)} test points", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == TARGET_NODE) &
                      (dmap[:, 1] == {'X': 0, 'Y': 1, 'Z': 2}[TARGET_DIR]))[0][0]
Phi_all = T_full2sec[target_eq, :]
Phi_chain = Phi_all[CHAIN]
K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])
omega0_ref = math.sqrt(K0_arr[0] / M_arr[0])
F_physical = 2500.0 / Phi_all[0]
Fg_arr = F_physical * Phi_chain
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']


def add_detune(w, feat6):
    # feat6: (n, N_CHAIN, 3) = [shift, zeta, kappa] per mode
    n = feat6.shape[0]
    detune = np.zeros((n, N_CHAIN))
    for k in range(N_CHAIN):
        shift_k = feat6[:, k, 0]; zeta_k = feat6[:, k, 1]
        detune[:, k] = np.tanh(((w - np.sqrt(1.0 + shift_k)) / zeta_k) / 20.0)
    return detune


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


train_feat6_r = train_feat6.reshape(-1, N_CHAIN, 3)
test_feat6_r = test_feat6.reshape(-1, N_CHAIN, 3)
train_detune = add_detune(train_w, train_feat6_r)
test_detune = add_detune(test_w, test_feat6_r)
Feat_aug_train = np.concatenate([train_feat6, train_detune], axis=1)
Feat_aug_test = np.concatenate([test_feat6, test_detune], axis=1)


def run_config(hidden, epochs, physics_weight, lr, seed=42):
    Feat_mean, Feat_std = Feat_aug_train.mean(0), Feat_aug_train.std(0)
    Feat_n = (Feat_aug_train - Feat_mean) / Feat_std
    W_t = torch.tensor(train_w, dtype=torch.float32)
    Feat_n_t = torch.tensor(Feat_n, dtype=torch.float32)
    X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n_t], dim=1)
    Omega_t = W_t * omega0_ref
    K_t = torch.tensor(train_K, dtype=torch.float32)
    M_t = torch.tensor(M_arr, dtype=torch.float32)
    C_t = torch.tensor(C_arr, dtype=torch.float32)
    Fg_t = torch.tensor(Fg_arr, dtype=torch.float32)

    Alpha_mean, Alpha_std = train_alpha.mean(0), train_alpha.std(0)
    Beta_mean, Beta_std = train_beta.mean(0), train_beta.std(0)
    Alpha_t = torch.tensor((train_alpha - Alpha_mean) / Alpha_std, dtype=torch.float32)
    Beta_t = torch.tensor((train_beta - Beta_mean) / Beta_std, dtype=torch.float32)
    Alpha_mean_t = torch.tensor(Alpha_mean, dtype=torch.float32); Alpha_std_t = torch.tensor(Alpha_std, dtype=torch.float32)
    Beta_mean_t = torch.tensor(Beta_mean, dtype=torch.float32); Beta_std_t = torch.tensor(Beta_std, dtype=torch.float32)
    F_scale = float(np.max(np.abs(Fg_arr)))

    torch.manual_seed(seed)
    model = s6.BPINN(X_in.shape[1], hidden, 2 * N_CHAIN, prior_sigma=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    KL_BETA = 0.001
    n_data = float(len(train_w))
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X_in)
        alpha_p, beta_p = pred[:, :N_CHAIN], pred[:, N_CHAIN:]
        data_loss = ((alpha_p - Alpha_t) ** 2 + (beta_p - Beta_t) ** 2).mean()
        alpha_phys = alpha_p * Alpha_std_t + Alpha_mean_t
        beta_phys = beta_p * Beta_std_t + Beta_mean_t
        R_alpha_list, R_beta_list = s4.chain_hbm_residual(
            CHAIN, [K_t[:, k] for k in range(N_CHAIN)], list(M_t), list(C_t), pair_coefs,
            list(Fg_t), Omega_t,
            [alpha_phys[:, k] for k in range(N_CHAIN)], [beta_phys[:, k] for k in range(N_CHAIN)])
        physics_loss = sum(((ra / F_scale) ** 2).mean() + ((rb / F_scale) ** 2).mean()
                            for ra, rb in zip(R_alpha_list, R_beta_list)) / N_CHAIN
        kl = model.total_kl() / n_data
        anneal = min(1.0, epoch / max(1, epochs * 0.3))
        loss = data_loss + anneal * physics_weight * physics_loss + anneal * KL_BETA * kl
        loss.backward()
        opt.step()
        sched.step()

    # validate
    Feat_test_n = (Feat_aug_test - Feat_mean) / Feat_std
    W_test_t = torch.tensor(test_w, dtype=torch.float32)
    X_test = torch.cat([s6.fourier_encode_w(W_test_t), torch.tensor(Feat_test_n, dtype=torch.float32)], dim=1)
    model.eval()
    with torch.no_grad():
        preds = np.array([model(X_test).numpy() for _ in range(30)])
    alpha_pred = preds[:, :, :N_CHAIN].mean(0) * Alpha_std + Alpha_mean
    beta_pred = preds[:, :, N_CHAIN:].mean(0) * Beta_std + Beta_mean
    amp_pred = np.hypot(alpha_pred, beta_pred)
    r2_amp = [r2(test_amp[:, k], amp_pred[:, k]) for k in range(N_CHAIN)]
    r2_alpha = [r2(test_alpha[:, k], alpha_pred[:, k]) for k in range(N_CHAIN)]
    r2_beta = [r2(test_beta[:, k], beta_pred[:, k]) for k in range(N_CHAIN)]

    # the real check: w=1.0, zero mistuning
    feat0 = np.zeros(3 * N_CHAIN)
    for k, m in enumerate(CHAIN):
        zeta_k = C_arr[k] / (2 * math.sqrt(K0_arr[k] * M_arr[k]))
        kappa_k = 0.75 * inp['K3_sec_diag'][m] / K0_arr[k]
        feat0[3 * k:3 * k + 3] = [0.0, zeta_k, kappa_k]
    detune0 = add_detune(np.array([1.0]), feat0.reshape(1, N_CHAIN, 3))[0]
    feat0_aug = np.concatenate([feat0, detune0])
    feat0_n = (feat0_aug - Feat_mean) / Feat_std
    X_check = torch.cat([s6.fourier_encode_w(torch.tensor([1.0], dtype=torch.float32)),
                          torch.tensor(feat0_n[None, :], dtype=torch.float32)], dim=1)
    with torch.no_grad():
        preds_check = np.array([model(X_check).numpy() for _ in range(100)])
    alpha_c = preds_check[:, 0, :N_CHAIN] * Alpha_std + Alpha_mean
    beta_c = preds_check[:, 0, N_CHAIN:] * Beta_std + Beta_mean
    u_complex = np.sum((alpha_c - 1j * beta_c) * Phi_chain[None, :], axis=1)
    bpinn_amp = float(np.abs(u_complex).mean())

    elapsed = time.time() - t0
    return dict(r2_amp=r2_amp, r2_alpha=r2_alpha, r2_beta=r2_beta,
                mean_r2_amp=np.mean(r2_amp), bpinn_amp=bpinn_amp, elapsed=elapsed,
                model=model, Feat_mean=Feat_mean, Feat_std=Feat_std,
                Alpha_mean=Alpha_mean, Alpha_std=Alpha_std, Beta_mean=Beta_mean, Beta_std=Beta_std)


configs = [
    ('baseline [64,64] 8k ep, pw=0.05, lr=1e-3', [64, 64], 8000, 0.05, 1e-3),
    ('bigger [128,128] 15k ep, pw=0.01, lr=1e-3', [128, 128], 15000, 0.01, 1e-3),
    ('bigger [128,128] 20k ep, pw=0.005, lr=5e-4', [128, 128], 20000, 0.005, 5e-4),
    ('deep [96,96,96] 15k ep, pw=0.01, lr=1e-3', [96, 96, 96], 15000, 0.01, 1e-3),
]

results = {}
for label, hidden, epochs, pw, lr in configs:
    print(f"\n=== {label} ===", flush=True)
    r = run_config(hidden, epochs, pw, lr)
    results[label] = r
    print(f"  R2_amp per mode: {[round(x,3) for x in r['r2_amp']]}  mean={r['mean_r2_amp']:.4f}", flush=True)
    print(f"  R2_alpha: {[round(x,3) for x in r['r2_alpha']]}", flush=True)
    print(f"  R2_beta:  {[round(x,3) for x in r['r2_beta']]}", flush=True)
    print(f"  BPINN@w=1,tuned: {r['bpinn_amp']:.4f} mm vs real ANSYS 1.2220mm  ratio={1.2220/r['bpinn_amp']:.4f}x", flush=True)
    print(f"  elapsed={r['elapsed']:.0f}s", flush=True)

best_label = max(results, key=lambda k: results[k]['mean_r2_amp'])
print(f"\n{'='*70}\nBEST CONFIG: {best_label}  (mean R2_amp={results[best_label]['mean_r2_amp']:.4f})\n{'='*70}", flush=True)

best = results[best_label]
torch.save(best['model'].state_dict(), os.path.join(OUT, 'bpinn_bridged01234_state.pt'))
np.savez(os.path.join(OUT, 'bpinn_bridged01234_norm.npz'),
         feat_mean=best['Feat_mean'], feat_std=best['Feat_std'],
         alpha_mean=best['Alpha_mean'], alpha_std=best['Alpha_std'],
         beta_mean=best['Beta_mean'], beta_std=best['Beta_std'],
         f_gen=Fg_arr, chain_modes=np.array(CHAIN), Phi_chain=Phi_chain,
         r2_per_mode=np.array(best['r2_amp']), r2_alpha_per_mode=np.array(best['r2_alpha']),
         r2_beta_per_mode=np.array(best['r2_beta']),
         bpinn_amp_mean=best['bpinn_amp'], real_ansys_amp=1.2220, real_ansys_std=0.0190,
         target_node=TARGET_NODE, target_dir=TARGET_DIR, best_config=best_label)
print(f"Saved best model as the production bpinn_bridged01234_state.pt / _norm.npz", flush=True)
print("DONE", flush=True)
