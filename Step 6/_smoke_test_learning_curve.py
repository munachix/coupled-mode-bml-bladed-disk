"""SMOKE TEST (2026-08-23, explicit user request: verify the fix direction
before committing to a multi-hour data-generation campaign). Reuses the
ALREADY-cached bridged dataset (no new ODE solves) and trains on 33%/66%/
100% subsets with the best config found so far ([96,96,96], reduced epochs
for speed) to see whether mode 1's R^2 (the specific bottleneck: 0.09-0.24
across every capacity/epoch config tried) is trending UP with more data
(-> more data is the right lever, worth the multi-hour campaign) or FLAT
(-> capacity/architecture-limited, more data of the same kind won't help,
need a different fix instead).
"""
import sys, os, time, math
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
OUT = s6.OUT

d = np.load(os.path.join(OUT, '_bridged_dataset_checkpoint.npz'))
train_w_all = d['train_w']; train_feat6_all = d['train_feat']; train_K_all = d['train_K_arr']
train_alpha_all = d['train_alpha']; train_beta_all = d['train_beta']
test_w = d['test_w']; test_feat6 = d['test_feat']
test_alpha = d['test_alpha']; test_beta = d['test_beta']; test_amp = d['test_amp']
n_total = len(train_w_all)
print(f"Total cached training points: {n_total}, test points: {len(test_w)}", flush=True)

inp = s6.load_inputs()
K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])
omega0_ref = math.sqrt(K0_arr[0] / M_arr[0])
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']
F_physical = 2500.0 / (np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')[
    np.where((s2._dof_map()[:, 0] == 1171) & (s2._dof_map()[:, 1] == 2))[0][0], 0])
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_chain = T_full2sec[target_eq, CHAIN]
Fg_arr = F_physical * Phi_chain


def add_detune(w, feat6):
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


test_feat6_r = test_feat6.reshape(-1, N_CHAIN, 3)
test_detune = add_detune(test_w, test_feat6_r)
Feat_aug_test = np.concatenate([test_feat6, test_detune], axis=1)


def run_fraction(frac, epochs=6000, hidden=(96, 96, 96), pw=0.01, lr=1e-3, seed=42):
    n_use = int(n_total * frac)
    rng = np.random.default_rng(123)
    idx = rng.choice(n_total, size=n_use, replace=False)
    w_use = train_w_all[idx]; feat6_use = train_feat6_all[idx]; K_use = train_K_all[idx]
    alpha_use = train_alpha_all[idx]; beta_use = train_beta_all[idx]
    feat6_r = feat6_use.reshape(-1, N_CHAIN, 3)
    detune_use = add_detune(w_use, feat6_r)
    Feat_aug = np.concatenate([feat6_use, detune_use], axis=1)
    Feat_mean, Feat_std = Feat_aug.mean(0), Feat_aug.std(0)
    Feat_n = (Feat_aug - Feat_mean) / Feat_std
    W_t = torch.tensor(w_use, dtype=torch.float32)
    X_in = torch.cat([s6.fourier_encode_w(W_t), torch.tensor(Feat_n, dtype=torch.float32)], dim=1)
    Omega_t = W_t * omega0_ref
    K_t = torch.tensor(K_use, dtype=torch.float32)
    M_t = torch.tensor(M_arr, dtype=torch.float32); C_t = torch.tensor(C_arr, dtype=torch.float32)
    Fg_t = torch.tensor(Fg_arr, dtype=torch.float32)
    F_scale = float(np.max(np.abs(Fg_arr)))

    Alpha_mean, Alpha_std = alpha_use.mean(0), alpha_use.std(0)
    Beta_mean, Beta_std = beta_use.mean(0), beta_use.std(0)
    Alpha_t = torch.tensor((alpha_use - Alpha_mean) / Alpha_std, dtype=torch.float32)
    Beta_t = torch.tensor((beta_use - Beta_mean) / Beta_std, dtype=torch.float32)
    Alpha_mean_t = torch.tensor(Alpha_mean, dtype=torch.float32); Alpha_std_t = torch.tensor(Alpha_std, dtype=torch.float32)
    Beta_mean_t = torch.tensor(Beta_mean, dtype=torch.float32); Beta_std_t = torch.tensor(Beta_std, dtype=torch.float32)

    torch.manual_seed(seed)
    model = s6.BPINN(X_in.shape[1], list(hidden), 2 * N_CHAIN, prior_sigma=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    KL_BETA = 0.001
    n_data = float(n_use)
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
        loss = data_loss + anneal * pw * physics_loss + anneal * KL_BETA * kl
        loss.backward()
        opt.step()
        sched.step()

    Feat_test_n = (Feat_aug_test - Feat_mean) / Feat_std
    W_test_t = torch.tensor(test_w, dtype=torch.float32)
    X_test = torch.cat([s6.fourier_encode_w(W_test_t), torch.tensor(Feat_test_n, dtype=torch.float32)], dim=1)
    model.eval()
    with torch.no_grad():
        preds = np.array([model(X_test).numpy() for _ in range(20)])
    alpha_pred = preds[:, :, :N_CHAIN].mean(0) * Alpha_std + Alpha_mean
    beta_pred = preds[:, :, N_CHAIN:].mean(0) * Beta_std + Beta_mean
    amp_pred = np.hypot(alpha_pred, beta_pred)
    r2_amp = [r2(test_amp[:, k], amp_pred[:, k]) for k in range(N_CHAIN)]
    elapsed = time.time() - t0
    return dict(n_use=n_use, r2_amp=r2_amp, mean_r2=np.mean(r2_amp), elapsed=elapsed)


print("\n=== LEARNING CURVE: does R2 (esp. mode 1) improve with more (of the SAME kind of) data? ===", flush=True)
for frac in [0.33, 0.66, 1.0]:
    r = run_fraction(frac)
    print(f"  frac={frac:.2f} (n={r['n_use']}): R2_amp per mode = {[round(x,3) for x in r['r2_amp']]}  "
          f"mean={r['mean_r2']:.4f}  mode1={r['r2_amp'][1]:.4f}  elapsed={r['elapsed']:.0f}s", flush=True)
print("DONE", flush=True)
