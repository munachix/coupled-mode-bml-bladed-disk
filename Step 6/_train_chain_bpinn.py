"""
13-mode chain BPINN training (2026-08-13), UPGRADED same night to phase-
resolved output + a real physics-residual loss (step4.chain_hbm_residual
-- the same closed-form coupled harmonic-balance derivation used for the
pairs, summed over each chain mode's neighbor pairs; verified against the
pair case and against direct time-domain Fourier projection before being
trusted here).

Input: 13 modes x (shift_m, zeta_m, kappa_m) = 39 features + Fourier(w).
Output: 26 values (alpha_m, beta_m per chain mode), predicted JOINTLY
(shared network) so the network can exploit real correlations between
neighboring modes' responses. Amplitude is a DERIVED quantity,
amp = sqrt(alpha^2+beta^2), not the network's own output.
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

CHAIN = s4.MODE_GROUPS['chain']   # [11,12,...,23]
N_CHAIN = len(CHAIN)
TARGET_PEAK = 0.1   # calibrated 2026-08-13 (unchanged): largest value that stayed stable across the
                     # chain's own frequency band (see original docstring history in git/PROJECT_STATUS)
N_CYCLES = 200
STEPS_PER_CYCLE = 20
W_GRID = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6])
N_TRAIN = 150
N_TEST = 30
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 1e-2   # same value calibrated on pair (3,4) -- see _train_pair_bpinn.py's comment
OUT = s6.OUT

t_start = time.time()
print(f"=== Chain BPINN training (phase-resolved + physics loss): modes {CHAIN} ===", flush=True)

inp = s6.load_inputs()
K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])
omega0_ref = math.sqrt(K0_arr.mean() / M_arr.mean())
zeta0_arr = C_arr / (2 * np.sqrt(K0_arr * M_arr))
Fg_arr = TARGET_PEAK * 2 * zeta0_arr * K0_arr
F_scale = float(np.max(np.abs(Fg_arr)))
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']
print(f"  chain freqs: [{inp['freqs_sec'][CHAIN].min():.2f}, {inp['freqs_sec'][CHAIN].max():.2f}] Hz, "
      f"omega0_ref={omega0_ref/(2*np.pi):.2f} Hz", flush=True)
print(f"  Fg range: [{Fg_arr.min():.3e}, {Fg_arr.max():.3e}]", flush=True)


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


def build_dataset(sample_indices):
    rows = []
    n_rejected = 0
    for k, i in enumerate(sample_indices):
        p = sample_features(i)
        for w in W_GRID:
            Omega = w * omega0_ref
            r = s4.duffing_forced_response_chain(CHAIN, p['K_arr'], M_arr, C_arr, pair_coefs,
                                                  Fg_arr, Omega, n_cycles=N_CYCLES,
                                                  steps_per_cycle=STEPS_PER_CYCLE)
            ok = np.all(np.isfinite(r['amp'])) and np.all(np.abs(r['amp']) < DIVERGE_BOUND)
            if not ok:
                n_rejected += 1
                continue
            rows.append(dict(w=w, feat=p['feat'], K_arr=p['K_arr'],
                              alpha=r['alpha'], beta=r['beta'], amp=r['amp']))
        if k % 20 == 0:
            print(f"  sample {k}/{len(sample_indices)}, elapsed={time.time()-t_start:.0f}s, "
                  f"rejected so far={n_rejected}", flush=True)
    print(f"  Total rejected: {n_rejected} of {len(sample_indices)*len(W_GRID)}", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN]
test_idx = perm[N_TRAIN:N_TRAIN + N_TEST]

print(f"Generating training data ({N_TRAIN} x {len(W_GRID)} = {N_TRAIN*len(W_GRID)} chain solves)...", flush=True)
train_rows = build_dataset(train_idx)
print(f"Generating test data ({N_TEST} x {len(W_GRID)})...", flush=True)
test_rows = build_dataset(test_idx)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

if len(train_rows) < 50 or len(test_rows) < 10:
    print(f"ABORT: too few surviving points (train={len(train_rows)}, test={len(test_rows)})", flush=True)
    sys.exit(1)

ckpt_path = os.path.join(OUT, '_chain_dataset_checkpoint.npz')
np.savez(ckpt_path,
          train_w=np.array([r['w'] for r in train_rows]),
          train_feat=np.stack([r['feat'] for r in train_rows]),
          train_K_arr=np.stack([r['K_arr'] for r in train_rows]),
          train_alpha=np.stack([r['alpha'] for r in train_rows]),
          train_beta=np.stack([r['beta'] for r in train_rows]),
          test_w=np.array([r['w'] for r in test_rows]),
          test_feat=np.stack([r['feat'] for r in test_rows]),
          test_alpha=np.stack([r['alpha'] for r in test_rows]),
          test_beta=np.stack([r['beta'] for r in test_rows]),
          test_amp=np.stack([r['amp'] for r in test_rows]))
print(f"Checkpointed: {ckpt_path}", flush=True)

# ---- Train ----
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Omega_t = W_t * omega0_ref
K_arr_t = torch.tensor(np.stack([r['K_arr'] for r in train_rows]), dtype=torch.float32)   # (n,13)
M_t = torch.tensor(M_arr, dtype=torch.float32)
C_t = torch.tensor(C_arr, dtype=torch.float32)
Fg_t = torch.tensor(Fg_arr, dtype=torch.float32)

Alpha_raw = np.stack([r['alpha'] for r in train_rows])   # (n, 13)
Beta_raw = np.stack([r['beta'] for r in train_rows])
Alpha_mean = Alpha_raw.mean(0); Alpha_std = Alpha_raw.std(0)
Beta_mean = Beta_raw.mean(0); Beta_std = Beta_raw.std(0)
Alpha_t = torch.tensor((Alpha_raw - Alpha_mean) / Alpha_std, dtype=torch.float32)
Beta_t = torch.tensor((Beta_raw - Beta_mean) / Beta_std, dtype=torch.float32)
Alpha_mean_t = torch.tensor(Alpha_mean, dtype=torch.float32); Alpha_std_t = torch.tensor(Alpha_std, dtype=torch.float32)
Beta_mean_t = torch.tensor(Beta_mean, dtype=torch.float32); Beta_std_t = torch.tensor(Beta_std, dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [64, 64], 2 * N_CHAIN, prior_sigma=1.0)   # (alpha_1..13, beta_1..13)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 8000
KL_BETA = 0.001
n_data = float(len(train_rows))
print("Training...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    alpha_p, beta_p = pred[:, :N_CHAIN], pred[:, N_CHAIN:]
    data_loss = ((alpha_p - Alpha_t) ** 2 + (beta_p - Beta_t) ** 2).mean()

    alpha_phys = alpha_p * Alpha_std_t + Alpha_mean_t   # (n,13)
    beta_phys = beta_p * Beta_std_t + Beta_mean_t
    # chain_hbm_residual expects per-mode lists/arrays; vectorize over the
    # batch by transposing to (13, n) so each mode's residual is computed
    # for the WHOLE batch at once (still pure elementwise ops, autograd-safe).
    R_alpha_list, R_beta_list = s4.chain_hbm_residual(
        CHAIN, [K_arr_t[:, k] for k in range(N_CHAIN)], list(M_t), list(C_t), pair_coefs,
        list(Fg_t), Omega_t,
        [alpha_phys[:, k] for k in range(N_CHAIN)], [beta_phys[:, k] for k in range(N_CHAIN)])
    physics_loss = sum(((ra / F_scale) ** 2).mean() + ((rb / F_scale) ** 2).mean()
                        for ra, rb in zip(R_alpha_list, R_beta_list)) / N_CHAIN

    kl = model.total_kl() / n_data
    anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
    loss = data_loss + anneal * PHYSICS_WEIGHT * physics_loss + anneal * KL_BETA * kl
    loss.backward()
    opt.step()
    if epoch % 2000 == 0:
        print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  physics={physics_loss.item():.6f}  "
              f"kl={kl.item():.5f}", flush=True)

# ---- Validate ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
Alpha_test_raw = np.stack([r['alpha'] for r in test_rows])
Beta_test_raw = np.stack([r['beta'] for r in test_rows])
Amp_test_raw = np.stack([r['amp'] for r in test_rows])
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)

model.eval()
n_mc = 30
preds = []
with torch.no_grad():
    for _ in range(n_mc):
        preds.append(model(X_test).numpy())
preds = np.array(preds)   # (n_mc, n_test, 26)
alpha_mean_pred = preds[:, :, :N_CHAIN].mean(0) * Alpha_std + Alpha_mean
beta_mean_pred = preds[:, :, N_CHAIN:].mean(0) * Beta_std + Beta_mean
amp_mean_pred = np.hypot(alpha_mean_pred, beta_mean_pred)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_per_mode = [r2(Amp_test_raw[:, k], amp_mean_pred[:, k]) for k in range(N_CHAIN)]
r2_alpha_per_mode = [r2(Alpha_test_raw[:, k], alpha_mean_pred[:, k]) for k in range(N_CHAIN)]
r2_beta_per_mode = [r2(Beta_test_raw[:, k], beta_mean_pred[:, k]) for k in range(N_CHAIN)]
print(f"\nTest R^2 per chain mode (amplitude): {dict(zip(CHAIN, [round(r,4) for r in r2_per_mode]))}", flush=True)
print(f"Mean amplitude R^2: {np.mean(r2_per_mode):.4f}", flush=True)
print(f"Mean alpha R^2: {np.mean(r2_alpha_per_mode):.4f}  Mean beta R^2: {np.mean(r2_beta_per_mode):.4f}", flush=True)

fp_model = os.path.join(OUT, 'bpinn_chain_state.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, 'bpinn_chain_norm.npz')
np.savez(fp_norm, feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
         alpha_mean=Alpha_mean, alpha_std=Alpha_std, beta_mean=Beta_mean, beta_std=Beta_std,
         f_gen=Fg_arr, target_peak=TARGET_PEAK, chain_modes=np.array(CHAIN),
         r2_per_mode=np.array(r2_per_mode), r2_alpha_per_mode=np.array(r2_alpha_per_mode),
         r2_beta_per_mode=np.array(r2_beta_per_mode), physics_weight=PHYSICS_WEIGHT)
print(f"Saved: {fp_model}", flush=True)
print(f"Saved: {fp_norm}", flush=True)
print(f"TOTAL TIME: {time.time()-t_start:.0f}s", flush=True)
print("DONE", flush=True)
