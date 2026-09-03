"""BPINN over the bridged 0-1-2-3-4 chain (2026-08-23), trained with the REAL
Case 3 point-force decomposition -- not the idealized symmetric TARGET_PEAK
convention _train_chain_bpinn.py/_train_pair_bpinn.py use for general
training-data coverage. This is the network that must actually be checked
against real ANSYS, not the exact ODE solver (step9.py's
run_case3_full_multimode_dynamic used the solver directly -- a real,
disclosed gap: it validated the ROM's physics, not the trained surrogate).

Force: Fg_m = F_physical * Phi_m(node 1171 UZ), F_physical = 2500/Phi_0 =
88.0742N -- the SAME real point force used in the real ANSYS dynamic
transient and in step9.py's full-multimode check, decomposed properly per
mode (NOT equal across modes -- Fg_0=2500, Fg_1=-157.5, and whatever modes
2/3/4 actually get from their own real Phi at that point).

Chain modes 0-1-2-3-4, real cross-coupling for all 4 adjacent pairs
((0,1),(1,2),(2,3),(3,4)) -- all real ANSYS measurements, no placeholders.
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
TARGET_NODE, TARGET_DIR = 1171, 'Z'
REAL_FORCE_SCALE = 2500.0
REAL_ANSYS_AMP_MM = 1.2220
REAL_ANSYS_STD_MM = 0.0190

N_CYCLES = 200
STEPS_PER_CYCLE = 20
# Same densified near-resonance grid proven for pairs (0,1)/(5,6)/(7,8) --
# this system's resonance is in the same 292.8Hz region.
W_GRID = np.array([0.9, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14,
                    1.16, 1.18, 1.20, 1.22, 1.25, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6])
N_TRAIN = 150
N_TEST = 20
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 0.05   # the tuned value found this session for pairs needing real beta accuracy
OUT = s6.OUT

t_start = time.time()
print(f"=== Bridged chain BPINN training (real point-force decomposition): modes {CHAIN} ===", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == TARGET_NODE) &
                      (dmap[:, 1] == {'X': 0, 'Y': 1, 'Z': 2}[TARGET_DIR]))[0][0]
Phi_all = T_full2sec[target_eq, :]
Phi_chain = Phi_all[CHAIN]
F_physical = REAL_FORCE_SCALE / Phi_all[0]

K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])
omega0_ref = math.sqrt(K0_arr[0] / M_arr[0])   # mode 0's own frequency -- the real drive frequency
Fg_arr = F_physical * Phi_chain    # REAL point-force decomposition, not symmetric TARGET_PEAK
F_scale = float(np.max(np.abs(Fg_arr)))
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']
print(f"  Target: node {TARGET_NODE} U{TARGET_DIR}, F_physical={F_physical:.4f} N", flush=True)
print(f"  Phi_chain: {Phi_chain}", flush=True)
print(f"  Fg_arr (real decomposition): {Fg_arr}", flush=True)
print(f"  chain freqs: {inp['freqs_sec'][CHAIN]} Hz, omega0_ref={omega0_ref/(2*np.pi):.3f} Hz", flush=True)


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

ckpt_path = os.path.join(OUT, '_bridged_dataset_checkpoint.npz')
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
K_arr_t = torch.tensor(np.stack([r['K_arr'] for r in train_rows]), dtype=torch.float32)
M_t = torch.tensor(M_arr, dtype=torch.float32)
C_t = torch.tensor(C_arr, dtype=torch.float32)
Fg_t = torch.tensor(Fg_arr, dtype=torch.float32)

Alpha_raw = np.stack([r['alpha'] for r in train_rows])
Beta_raw = np.stack([r['beta'] for r in train_rows])
Alpha_mean = Alpha_raw.mean(0); Alpha_std = Alpha_raw.std(0)
Beta_mean = Beta_raw.mean(0); Beta_std = Beta_raw.std(0)
Alpha_t = torch.tensor((Alpha_raw - Alpha_mean) / Alpha_std, dtype=torch.float32)
Beta_t = torch.tensor((Beta_raw - Beta_mean) / Beta_std, dtype=torch.float32)
Alpha_mean_t = torch.tensor(Alpha_mean, dtype=torch.float32); Alpha_std_t = torch.tensor(Alpha_std, dtype=torch.float32)
Beta_mean_t = torch.tensor(Beta_mean, dtype=torch.float32); Beta_std_t = torch.tensor(Beta_std, dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
w_np = W_t.numpy()
feat6_np = Feat_t.numpy().reshape(-1, N_CHAIN, 3)
# detuning feature per chain mode (2026-08-21 fix, generalized here to N modes):
detune = np.zeros((len(train_rows), N_CHAIN))
for k in range(N_CHAIN):
    shift_k = feat6_np[:, k, 0]; zeta_k = feat6_np[:, k, 1]
    detune[:, k] = np.tanh(((w_np - np.sqrt(1.0 + shift_k)) / zeta_k) / 20.0)
Feat_aug = np.concatenate([Feat_t.numpy(), detune], axis=1)
Feat_aug_mean = Feat_aug.mean(0); Feat_aug_std = Feat_aug.std(0)
Feat_aug_n = torch.tensor((Feat_aug - Feat_aug_mean) / Feat_aug_std, dtype=torch.float32)
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_aug_n], dim=1)

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [64, 64], 2 * N_CHAIN, prior_sigma=1.0)
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

    alpha_phys = alpha_p * Alpha_std_t + Alpha_mean_t
    beta_phys = beta_p * Beta_std_t + Beta_mean_t
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

# ---- Validate on held-out test set ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
Alpha_test_raw = np.stack([r['alpha'] for r in test_rows])
Beta_test_raw = np.stack([r['beta'] for r in test_rows])
Amp_test_raw = np.stack([r['amp'] for r in test_rows])
w_test_np = W_test.numpy()
feat6_test_np = Feat_test.numpy().reshape(-1, N_CHAIN, 3)
detune_test = np.zeros((len(test_rows), N_CHAIN))
for k in range(N_CHAIN):
    shift_k = feat6_test_np[:, k, 0]; zeta_k = feat6_test_np[:, k, 1]
    detune_test[:, k] = np.tanh(((w_test_np - np.sqrt(1.0 + shift_k)) / zeta_k) / 20.0)
Feat_aug_test = np.concatenate([Feat_test.numpy(), detune_test], axis=1)
Feat_aug_test_n = torch.tensor((Feat_aug_test - Feat_aug_mean) / Feat_aug_std, dtype=torch.float32)
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_aug_test_n], dim=1)

model.eval()
n_mc = 30
preds = []
with torch.no_grad():
    for _ in range(n_mc):
        preds.append(model(X_test).numpy())
preds = np.array(preds)
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
print(f"\nTest R^2 per mode (amplitude): {dict(zip(CHAIN, [round(r,4) for r in r2_per_mode]))}", flush=True)
print(f"Mean amplitude R^2: {np.mean(r2_per_mode):.4f}", flush=True)
print(f"Mean alpha R^2: {np.mean(r2_alpha_per_mode):.4f}  Mean beta R^2: {np.mean(r2_beta_per_mode):.4f}", flush=True)

# ---- THE REAL CHECK: BPINN at w=1.0, ZERO mistuning, vs real ANSYS ----
feat0 = np.zeros(3 * N_CHAIN)
for k, m in enumerate(CHAIN):
    zeta_k = C_arr[k] / (2 * math.sqrt(K0_arr[k] * M_arr[k]))
    kappa_k = 0.75 * inp['K3_sec_diag'][m] / K0_arr[k]
    feat0[3 * k:3 * k + 3] = [0.0, zeta_k, kappa_k]
w_check = 1.0
detune0 = np.zeros(N_CHAIN)
for k in range(N_CHAIN):
    detune0[k] = np.tanh(((w_check - 1.0) / feat0[3 * k + 1]) / 20.0)   # shift=0 -> sqrt(1+0)=1
feat0_aug = np.concatenate([feat0, detune0])
feat0_n = (feat0_aug - Feat_aug_mean) / Feat_aug_std
X_check = torch.cat([s6.fourier_encode_w(torch.tensor([w_check], dtype=torch.float32)),
                      torch.tensor(feat0_n[None, :], dtype=torch.float32)], dim=1)
with torch.no_grad():
    preds_check = np.array([model(X_check).numpy() for _ in range(100)])
alpha_c = preds_check[:, 0, :N_CHAIN] * Alpha_std + Alpha_mean
beta_c = preds_check[:, 0, N_CHAIN:] * Beta_std + Beta_mean
u_complex_samples = np.sum((alpha_c - 1j * beta_c) * Phi_chain[None, :], axis=1)
u_mag_samples = np.abs(u_complex_samples)
bpinn_amp_mean = float(u_mag_samples.mean())
bpinn_amp_std = float(u_mag_samples.std())

print(f"\n{'='*70}")
print(f"BRIDGED BPINN vs REAL ANSYS: node {TARGET_NODE} U{TARGET_DIR}, w=1.0, tuned baseline")
print(f"{'='*70}")
print(f"  BPINN (trained on bridged 0-1-2-3-4 system, real force): {bpinn_amp_mean:.4f} +/- {bpinn_amp_std:.4f} mm")
print(f"  Real ANSYS (100-cycle converged transient):              {REAL_ANSYS_AMP_MM:.4f} +/- {REAL_ANSYS_STD_MM:.4f} mm")
print(f"  Ratio (real ANSYS / BPINN): {REAL_ANSYS_AMP_MM/bpinn_amp_mean:.4f}x")
print(f"  (for reference) exact-solver full-multimode result was: 1.2135mm, ratio 1.007x")
print(f"{'='*70}")

fp_model = os.path.join(OUT, 'bpinn_bridged01234_state.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, 'bpinn_bridged01234_norm.npz')
np.savez(fp_norm, feat_mean=Feat_aug_mean, feat_std=Feat_aug_std,
         alpha_mean=Alpha_mean, alpha_std=Alpha_std, beta_mean=Beta_mean, beta_std=Beta_std,
         f_gen=Fg_arr, chain_modes=np.array(CHAIN), Phi_chain=Phi_chain,
         r2_per_mode=np.array(r2_per_mode), r2_alpha_per_mode=np.array(r2_alpha_per_mode),
         r2_beta_per_mode=np.array(r2_beta_per_mode), physics_weight=PHYSICS_WEIGHT,
         bpinn_amp_mean=bpinn_amp_mean, bpinn_amp_std=bpinn_amp_std,
         real_ansys_amp=REAL_ANSYS_AMP_MM, real_ansys_std=REAL_ANSYS_STD_MM,
         target_node=TARGET_NODE, target_dir=TARGET_DIR)
print(f"Saved: {fp_model}")
print(f"Saved: {fp_norm}")
print(f"TOTAL TIME: {time.time()-t_start:.0f}s", flush=True)
print("DONE", flush=True)
