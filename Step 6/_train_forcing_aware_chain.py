"""FORCING-AWARE 13-mode chain (11-23) BPINN (2026-08-24) -- same forcing-
as-input fix applied to the 5 pair networks (R^2 0.92-0.97) and to the
bridged 0-1-2-3-4 chain, now applied to the real HF-band chain topology.
Previously this chain was trained at a single fixed TARGET_PEAK=0.1
(see _train_chain_bpinn.py); this generalizes it across force levels the
same way, with target_peak as an explicit input feature.

Chain solve cost is per-mode-count (13 modes here vs 5 in the bridged
system), so W_GRID is kept at the original proven 10-point set (not the
widened per-pair grid) to keep total cost tractable -- the low-forcing
resolution trade-off found for pair (0,1) is a real, disclosed limitation
here too if it reproduces.
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

CHAIN = s4.MODE_GROUPS['chain']   # [11,...,23]
N_CHAIN = len(CHAIN)
OUT = s6.OUT
FORCE_LEVELS = [0.02, 0.05, 0.1, 0.2, 0.4]   # spans below/above the original single-level TARGET_PEAK=0.1
N_TRAIN_PER_LEVEL = 20
N_TEST_PER_LEVEL = 5
W_GRID = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6])
N_CYCLES = 200
STEPS_PER_CYCLE = 20
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 1e-2

t_start = time.time()
print(f"=== Forcing-aware chain BPINN, modes {CHAIN} ===", flush=True)
print(f"  Force levels (target_peak): {FORCE_LEVELS}", flush=True)
print(f"  W_GRID: {len(W_GRID)} points, [{W_GRID.min():.2f}, {W_GRID.max():.2f}]", flush=True)

inp = s6.load_inputs()
K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])
omega0_ref = math.sqrt(K0_arr.mean() / M_arr.mean())
zeta0_arr = C_arr / (2 * np.sqrt(K0_arr * M_arr))
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']
print(f"  chain freqs: [{inp['freqs_sec'][CHAIN].min():.2f}, {inp['freqs_sec'][CHAIN].max():.2f}] Hz", flush=True)


def sample_features(sample_idx):
    row = {v: inp['theta'][v][sample_idx] for v in s6.VAR_NAMES}
    df = s5.compute_delta_f_vectorized({k: v[None, :] for k, v in row.items()},
                                        inp['sens'], inp['L_ref'], inp['t_ref'])[0]
    scale = (1.0 + df) ** 2 - 1.0
    K_arr = np.zeros(N_CHAIN)
    feat = np.zeros(3 * N_CHAIN)
    for k, m in enumerate(CHAIN):
        shift_m = float(scale @ inp['P'][:, m])
        K_m = K0_arr[k] * (1.0 + shift_m)
        zeta_m = C_arr[k] / (2 * math.sqrt(K_m * M_arr[k]))
        kappa_m = 0.75 * inp['K3_sec_diag'][m] / K_m
        K_arr[k] = K_m
        feat[3 * k:3 * k + 3] = [shift_m, zeta_m, kappa_m]
    return dict(K_arr=K_arr, feat=feat)


def build_dataset(sample_indices, levels):
    rows = []
    n_rejected = 0
    n_total = 0
    for k, i in enumerate(sample_indices):
        p = sample_features(i)
        for tp in levels:
            Fg_arr = tp * 2 * zeta0_arr * K0_arr
            for w in W_GRID:
                n_total += 1
                Omega = w * omega0_ref
                r = s4.duffing_forced_response_chain(CHAIN, p['K_arr'], M_arr, C_arr, pair_coefs,
                                                      Fg_arr, Omega, n_cycles=N_CYCLES,
                                                      steps_per_cycle=STEPS_PER_CYCLE)
                ok = np.all(np.isfinite(r['amp'])) and np.all(np.abs(r['amp']) < DIVERGE_BOUND)
                if not ok:
                    n_rejected += 1
                    continue
                rows.append(dict(w=w, feat=p['feat'], K_arr=p['K_arr'], tp=tp,
                                  alpha=r['alpha'], beta=r['beta'], amp=r['amp']))
        if k % 5 == 0:
            print(f"  sample {k}/{len(sample_indices)}, elapsed={time.time()-t_start:.0f}s, "
                  f"rejected so far={n_rejected}/{n_total}", flush=True)
    print(f"  Total rejected: {n_rejected} of {n_total}", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN_PER_LEVEL]
test_idx = perm[N_TRAIN_PER_LEVEL:N_TRAIN_PER_LEVEL + N_TEST_PER_LEVEL]

print(f"\nGenerating training data ({N_TRAIN_PER_LEVEL} x {len(FORCE_LEVELS)} levels x "
      f"{len(W_GRID)} w-pts = {N_TRAIN_PER_LEVEL*len(FORCE_LEVELS)*len(W_GRID)} chain solves)...", flush=True)
train_rows = build_dataset(train_idx, FORCE_LEVELS)
print("\nGenerating test data...", flush=True)
test_rows = build_dataset(test_idx, FORCE_LEVELS)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

if len(train_rows) < 200 or len(test_rows) < 30:
    print("ABORT: too few surviving points", flush=True)
    sys.exit(1)

# ---- Train ----
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Omega_t = W_t * omega0_ref
K_arr_t = torch.tensor(np.stack([r['K_arr'] for r in train_rows]), dtype=torch.float32)
M_t = torch.tensor(M_arr, dtype=torch.float32); C_t = torch.tensor(C_arr, dtype=torch.float32)
TP_t = torch.tensor([r['tp'] for r in train_rows], dtype=torch.float32)
Fg_t = torch.stack([TP_t * 2 * float(zeta0_arr[k]) * float(K0_arr[k]) for k in range(N_CHAIN)], dim=1)  # (n,13)

Alpha_raw = np.stack([r['alpha'] for r in train_rows]); Beta_raw = np.stack([r['beta'] for r in train_rows])
Alpha_mean, Alpha_std = Alpha_raw.mean(0), Alpha_raw.std(0)
Beta_mean, Beta_std = Beta_raw.mean(0), Beta_raw.std(0)
Alpha_t = torch.tensor((Alpha_raw - Alpha_mean) / Alpha_std, dtype=torch.float32)
Beta_t = torch.tensor((Beta_raw - Beta_mean) / Beta_std, dtype=torch.float32)
Alpha_mean_t = torch.tensor(Alpha_mean, dtype=torch.float32); Alpha_std_t = torch.tensor(Alpha_std, dtype=torch.float32)
Beta_mean_t = torch.tensor(Beta_mean, dtype=torch.float32); Beta_std_t = torch.tensor(Beta_std, dtype=torch.float32)

feat3 = Feat_t.reshape(-1, N_CHAIN, 3)
detune = torch.tanh(((W_t[:, None] - torch.sqrt(1.0 + feat3[:, :, 0])) / feat3[:, :, 1]) / 20.0)  # (n,13)
Feat_aug = torch.cat([Feat_t, detune, TP_t[:, None]], dim=1)   # 3*13 + 13 + 1 = 53

Feat_mean, Feat_std = Feat_aug.mean(0), Feat_aug.std(0)
Feat_n = (Feat_aug - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)
F_scale = float(Fg_t.abs().max())

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [96, 96], 2 * N_CHAIN, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 10000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=5e-5)
KL_BETA = 0.001
n_data = float(len(train_rows))
print("\nTraining...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    alpha_p, beta_p = pred[:, :N_CHAIN], pred[:, N_CHAIN:]
    data_loss = ((alpha_p - Alpha_t) ** 2 + (beta_p - Beta_t) ** 2).mean()
    alpha_phys = alpha_p * Alpha_std_t + Alpha_mean_t
    beta_phys = beta_p * Beta_std_t + Beta_mean_t
    R_alpha_list, R_beta_list = s4.chain_hbm_residual(
        CHAIN, [K_arr_t[:, k] for k in range(N_CHAIN)], list(M_t), list(C_t), pair_coefs,
        [Fg_t[:, k] for k in range(N_CHAIN)], Omega_t,
        [alpha_phys[:, k] for k in range(N_CHAIN)], [beta_phys[:, k] for k in range(N_CHAIN)])
    physics_loss = sum(((ra / F_scale) ** 2).mean() + ((rb / F_scale) ** 2).mean()
                        for ra, rb in zip(R_alpha_list, R_beta_list)) / N_CHAIN
    kl = model.total_kl() / n_data
    anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
    loss = data_loss + anneal * PHYSICS_WEIGHT * physics_loss + anneal * KL_BETA * kl
    loss.backward()
    opt.step()
    sched.step()
    if epoch % 2000 == 0:
        print(f"  epoch {epoch:6d}  data={data_loss.item():.6f}  physics={physics_loss.item():.6f}  "
              f"kl={kl.item():.5f}", flush=True)

# ---- Validate ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
TP_test = torch.tensor([r['tp'] for r in test_rows], dtype=torch.float32)
Amp_test_raw = np.stack([r['amp'] for r in test_rows])
feat3_test = Feat_test.reshape(-1, N_CHAIN, 3)
detune_test = torch.tanh(((W_test[:, None] - torch.sqrt(1.0 + feat3_test[:, :, 0])) / feat3_test[:, :, 1]) / 20.0)
Feat_test_aug = torch.cat([Feat_test, detune_test, TP_test[:, None]], dim=1)
Feat_test_n = (Feat_test_aug - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)
model.eval()
with torch.no_grad():
    preds = np.array([model(X_test).numpy() for _ in range(30)])
alpha_pred = preds[:, :, :N_CHAIN].mean(0) * Alpha_std + Alpha_mean
beta_pred = preds[:, :, N_CHAIN:].mean(0) * Beta_std + Beta_mean
amp_pred = np.hypot(alpha_pred, beta_pred)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_per_mode = [r2(Amp_test_raw[:, k], amp_pred[:, k]) for k in range(N_CHAIN)]
print(f"\nOverall test R^2 per chain mode (amplitude, all force levels pooled): "
      f"{dict(zip(CHAIN, [round(x,4) for x in r2_per_mode]))}", flush=True)
print(f"Mean amplitude R^2: {np.mean(r2_per_mode):.4f}", flush=True)
test_tp = np.array([r['tp'] for r in test_rows])
for tp in FORCE_LEVELS:
    mask = np.isclose(test_tp, tp)
    if mask.sum() > 2:
        r2_lvl = [r2(Amp_test_raw[mask, k], amp_pred[mask, k]) for k in range(N_CHAIN)]
        print(f"  target_peak={tp:.3f}: mean R^2={np.mean(r2_lvl):.4f}  (n={mask.sum()})", flush=True)

fp_model = os.path.join(OUT, 'bpinn_forcing_aware_chain_state.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, 'bpinn_forcing_aware_chain_norm.npz')
np.savez(fp_norm, feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
          alpha_mean=Alpha_mean, alpha_std=Alpha_std, beta_mean=Beta_mean, beta_std=Beta_std,
          chain_modes=np.array(CHAIN), r2_per_mode=np.array(r2_per_mode),
          is_forcing_aware=True, default_target_peak=0.1, force_levels=np.array(FORCE_LEVELS))
print(f"Saved: {fp_model}")
print(f"Saved: {fp_norm}")
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
