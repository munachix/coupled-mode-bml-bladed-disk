"""FORCING-AWARE bridged 0-1-2-3-4 BPINN, ROUND 2 (2026-08-24).

Round 1 (dense 49-point W_GRID, 30 train samples, force multipliers down
to 0.05) plateaued at R^2~0.45 -- confirmed a REAL limitation (not a metric
artifact: true-amplitude std/mean ratios were 1.5-3.8x), but comparison
against the 13-mode chain (11-23) trained with the SAME forcing-aware
recipe and a coarse 10-point W_GRID (R^2=0.976) shows the dense
near-resonance grid is very likely the actual cause, not "5 modes is just
too hard": the chain has more outputs (26 vs 10) yet fits far better with
fewer samples once the grid stops demanding sharp per-point resonance
resolution. Round 1 also included multiplier=0.05, which produced a
catastrophic R^2=-8.6 for mode 1 alone -- a pathological low-forcing point
(same failure mode diagnosed for pair (0,1)'s lowest levels) that likely
distorted the shared fit for every other level too.

Round 2 fix: switch to the chain's proven coarse grid, drop the
pathological multiplier=0.05 level (narrow the force span to bracket the
real mult=1.0 validation point more tightly instead of spanning all the
way to near-zero), and reinvest the ~5x cheaper per-sample cost into
substantially more training samples.

Force levels: a MULTIPLIER on the real Case 3 point force (F_physical=
88.0742N at node 1171, decomposed per mode via real Phi_chain), so
multiplier=1.0 reproduces the exact real validation scenario.
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
REAL_ANSYS_AMP_MM = 1.2220
REAL_ANSYS_STD_MM = 0.0190
# Multiplier on the REAL Case 3 point force (2500N at mode-0 generalized
# coordinate) -- 1.0 IS the real validation scenario. Narrowed off 0.05
# (round 1's pathological level, R^2=-8.6 for mode 1) to bracket 1.0 tightly.
FORCE_MULTIPLIERS = [0.3, 0.6, 1.0, 1.3, 1.6]
N_TRAIN_PER_LEVEL = 100
N_TEST_PER_LEVEL = 20
# Round 1's dense near-resonance band (49 points) is the prime suspect for
# the R^2~0.45 plateau -- switching to the chain(11-23)'s proven coarse
# grid, which achieved R^2=0.976 on a HIGHER-dimensional (26-output) system.
W_GRID = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6])
N_CYCLES = 200
STEPS_PER_CYCLE = 20    # matches the proven-working earlier bridged script (_train_bridged_bpinn.py) --
                        # an earlier 171 here was an unjustified carryover and made each solve ~8.5x too slow
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 0.05

t_start = time.time()
print(f"=== Forcing-aware bridged BPINN, chain {CHAIN} ===", flush=True)
print(f"  Force multipliers (of real Case 3 force): {FORCE_MULTIPLIERS}", flush=True)
print(f"  W_GRID: {len(W_GRID)} points, [{W_GRID.min():.3f}, {W_GRID.max():.2f}]", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]
Phi_chain = Phi_all[CHAIN]
F_physical_real = 2500.0 / Phi_all[0]
K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])
omega0_ref = math.sqrt(K0_arr[0] / M_arr[0])
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']
print(f"  F_physical_real={F_physical_real:.4f} N, Phi_chain={Phi_chain}", flush=True)


def detune_all(w, shift_arr, zeta_arr):
    return np.tanh(((w[:, None] - np.sqrt(1.0 + shift_arr)) / zeta_arr) / 20.0)


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


def build_dataset(sample_indices, multipliers):
    rows = []
    n_rejected = 0
    n_total = 0
    for k, i in enumerate(sample_indices):
        p = sample_features(i)
        for mult in multipliers:
            Fg_arr = mult * F_physical_real * Phi_chain
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
                feat3 = p['feat'].reshape(N_CHAIN, 3)
                detune = detune_all(np.array([w]), feat3[:, 0], feat3[:, 1])[0]
                feat_aug = np.concatenate([p['feat'], detune, [mult]])   # 3*N + N + 1 = 4N+1
                rows.append(dict(w=w, feat=feat_aug, K_arr=p['K_arr'], mult=mult,
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

print(f"\nGenerating training data ({N_TRAIN_PER_LEVEL} x {len(FORCE_MULTIPLIERS)} levels x "
      f"{len(W_GRID)} w-pts = {N_TRAIN_PER_LEVEL*len(FORCE_MULTIPLIERS)*len(W_GRID)} solves)...", flush=True)
train_rows = build_dataset(train_idx, FORCE_MULTIPLIERS)
print("\nGenerating test data...", flush=True)
test_rows = build_dataset(test_idx, FORCE_MULTIPLIERS)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

if len(train_rows) < 200 or len(test_rows) < 30:
    print("ABORT: too few surviving points", flush=True)
    sys.exit(1)

np.savez(os.path.join(OUT, '_forcing_aware_bridged_checkpoint.npz'),
          train_w=np.array([r['w'] for r in train_rows]), train_feat=np.stack([r['feat'] for r in train_rows]),
          train_mult=np.array([r['mult'] for r in train_rows]),
          train_alpha=np.stack([r['alpha'] for r in train_rows]), train_beta=np.stack([r['beta'] for r in train_rows]),
          test_w=np.array([r['w'] for r in test_rows]), test_feat=np.stack([r['feat'] for r in test_rows]),
          test_mult=np.array([r['mult'] for r in test_rows]),
          test_alpha=np.stack([r['alpha'] for r in test_rows]), test_beta=np.stack([r['beta'] for r in test_rows]),
          test_amp=np.stack([r['amp'] for r in test_rows]))

# ---- Train (single joint network, all 5 modes) ----
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Omega_t = W_t * omega0_ref
K_t = torch.tensor(np.stack([r['K_arr'] for r in train_rows]), dtype=torch.float32)
M_t = torch.tensor(M_arr, dtype=torch.float32); C_t = torch.tensor(C_arr, dtype=torch.float32)
Fg_t = torch.tensor(np.stack([r['mult'] * F_physical_real * Phi_chain for r in train_rows]), dtype=torch.float32)

Alpha_raw = np.stack([r['alpha'] for r in train_rows]); Beta_raw = np.stack([r['beta'] for r in train_rows])
Alpha_mean, Alpha_std = Alpha_raw.mean(0), Alpha_raw.std(0)
Beta_mean, Beta_std = Beta_raw.mean(0), Beta_raw.std(0)
Alpha_t = torch.tensor((Alpha_raw - Alpha_mean) / Alpha_std, dtype=torch.float32)
Beta_t = torch.tensor((Beta_raw - Beta_mean) / Beta_std, dtype=torch.float32)
Alpha_mean_t = torch.tensor(Alpha_mean, dtype=torch.float32); Alpha_std_t = torch.tensor(Alpha_std, dtype=torch.float32)
Beta_mean_t = torch.tensor(Beta_mean, dtype=torch.float32); Beta_std_t = torch.tensor(Beta_std, dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)
F_scale = float(np.abs([r['mult'] * F_physical_real * Phi_chain for r in train_rows]).max())

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [96, 96], 2 * N_CHAIN, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 15000
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
        CHAIN, [K_t[:, k] for k in range(N_CHAIN)], list(M_t), list(C_t), pair_coefs,
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
    if epoch % 3000 == 0:
        print(f"  epoch {epoch:6d}  data={data_loss.item():.6f}  physics={physics_loss.item():.6f}  "
              f"kl={kl.item():.5f}", flush=True)

# ---- Validate ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
Amp_test_raw = np.stack([r['amp'] for r in test_rows])
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
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
print(f"\nOverall test R^2 per mode (amplitude, all force levels pooled): "
      f"{dict(zip(CHAIN, [round(x,4) for x in r2_per_mode]))}  mean={np.mean(r2_per_mode):.4f}", flush=True)
test_mult = np.array([r['mult'] for r in test_rows])
for mult in FORCE_MULTIPLIERS:
    mask = np.isclose(test_mult, mult)
    if mask.sum() > 2:
        r2_lvl = [r2(Amp_test_raw[mask, k], amp_pred[mask, k]) for k in range(N_CHAIN)]
        print(f"  mult={mult:.2f}: R^2 per mode={[round(x,3) for x in r2_lvl]}  mean={np.mean(r2_lvl):.4f}  (n={mask.sum()})", flush=True)

# ---- THE REAL CHECK: mult=1.0 (real Case 3 force), w=1.0, zero mistuning, vs real ANSYS ----
feat0 = np.zeros(3 * N_CHAIN)
for k, m in enumerate(CHAIN):
    zeta_k = C_arr[k] / (2 * math.sqrt(K0_arr[k] * M_arr[k]))
    kappa_k = 0.75 * inp['K3_sec_diag'][m] / K0_arr[k]
    feat0[3 * k:3 * k + 3] = [0.0, zeta_k, kappa_k]
feat0_3 = feat0.reshape(N_CHAIN, 3)
detune0 = detune_all(np.array([1.0]), feat0_3[:, 0], feat0_3[:, 1])[0]
feat0_aug = np.concatenate([feat0, detune0, [1.0]])[None, :]
Feat0_n = (torch.tensor(feat0_aug, dtype=torch.float32) - Feat_mean) / Feat_std
X0 = torch.cat([s6.fourier_encode_w(torch.tensor([1.0], dtype=torch.float32)), Feat0_n], dim=1)
with torch.no_grad():
    preds0 = np.array([model(X0).numpy() for _ in range(100)])
alpha_c0 = preds0[:, 0, :N_CHAIN] * Alpha_std + Alpha_mean
beta_c0 = preds0[:, 0, N_CHAIN:] * Beta_std + Beta_mean
u_complex = np.sum((alpha_c0 - 1j * beta_c0) * Phi_chain[None, :], axis=1)
bpinn_amp_mean = float(np.abs(u_complex).mean())
bpinn_amp_std = float(np.abs(u_complex).std())

print(f"\n{'='*70}")
print(f"FORCING-AWARE BRIDGED BPINN vs REAL ANSYS: node 1171 UZ, real Case 3 force, w=1.0, tuned")
print(f"{'='*70}")
print(f"  BPINN: {bpinn_amp_mean:.4f} +/- {bpinn_amp_std:.4f} mm")
print(f"  Real ANSYS (converged): {REAL_ANSYS_AMP_MM:.4f} +/- {REAL_ANSYS_STD_MM:.4f} mm")
print(f"  Ratio (real/BPINN): {REAL_ANSYS_AMP_MM/bpinn_amp_mean:.4f}x")
print(f"  (for reference) exact-solver full-70-mode result: 1.2135mm, ratio 1.007x")
print(f"{'='*70}")

for k in range(N_CHAIN):
    torch.save(model.state_dict(), os.path.join(OUT, 'bpinn_forcing_aware_bridged01234_state.pt'))
np.savez(os.path.join(OUT, 'bpinn_forcing_aware_bridged01234_norm.npz'),
          feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
          alpha_mean=Alpha_mean, alpha_std=Alpha_std, beta_mean=Beta_mean, beta_std=Beta_std,
          chain=np.array(CHAIN), Phi_chain=Phi_chain, r2_per_mode=np.array(r2_per_mode),
          bpinn_amp_mean=bpinn_amp_mean, bpinn_amp_std=bpinn_amp_std,
          real_ansys_amp=REAL_ANSYS_AMP_MM, real_ansys_std=REAL_ANSYS_STD_MM)
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
