"""FORCING-AWARE mode-2 BPINN (2026-08-24) -- mode 2 is the one mode in the
1B cluster confirmed genuinely isolated by the real frequency-gap scan
(Step 4 MODE_GROUPS), so a true SDOF Duffing model is physically justified
here (unlike modes 0,1,3,4 which needed real cross-mode coupling). The
original per-mode trainer (_multimode_bpinn.py -> step6.train_bpinn) used a
single fixed target_peak_frac_qref=0.8; this generalizes it the same way as
the pairs/chain: multiple force levels via Step 4's own validated
pseudo-arc-length continuation, target_peak as an explicit input feature.
"""
import sys, os, time, math
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
import step6 as s6
import step4 as s4

MODE = 2
OUT = s6.OUT
FORCE_LEVELS = [0.1, 0.3, 0.8, 1.5, 2.5]   # target_peak_frac_qref; 0.8 = original single-level value
N_TRAIN_PER_LEVEL = 120
N_TEST_PER_LEVEL = 20
N_POINTS_PER_CURVE = 15
N_EXTRA_COLLOC = 5

t_start = time.time()
print(f"=== Forcing-aware mode-{MODE} BPINN (isolated SDOF) ===", flush=True)
print(f"  Force levels (target_peak_frac_qref): {FORCE_LEVELS}", flush=True)

s6.CONFIG['mode_index'] = MODE
inp = s6.load_inputs()


def build_dataset_multilevel(sample_indices, levels, rng):
    rows = []
    n_skipped = 0
    for i in sample_indices:
        p = s6.per_sample_sdof_params(inp, i)
        for tp in levels:
            cont = s4.duffing_forced_response_continuation(
                2 * math.pi * math.sqrt(p['K'] / p['M']), p['M'], p['C'], p['K'], p['K3'],
                p['q_ref'], tp)
            w_curve = cont['Omega'] / (2 * math.pi * p['omega0'])
            stable_idx = np.where(cont['stable'])[0]
            n_pts = len(stable_idx)
            if n_pts == 0:
                n_skipped += 1
                continue
            pick = stable_idx[rng.choice(n_pts, size=min(N_POINTS_PER_CURVE, n_pts), replace=False)]
            f_fixed = p['zeta'] * 2 * tp
            for j in pick:
                w = float(w_curve[j])
                target_amp = float(cont['amplitude'][j])
                alpha, beta = s6._solve_ab_at_w(w, p['zeta'], p['kappa'], f_fixed, target_amp)
                rows.append(dict(w=w, features=p['features'], zeta=p['zeta'], kappa=p['kappa'],
                                  tp=tp, alpha=alpha, beta=beta, amplitude=target_amp))
            for w_extra in rng.uniform(0.85, 1.6, N_EXTRA_COLLOC):
                rows.append(dict(w=float(w_extra), features=p['features'], zeta=p['zeta'],
                                  kappa=p['kappa'], tp=tp, alpha=None, beta=None, amplitude=None))
    if n_skipped:
        print(f"  ({n_skipped} sample/level combos had no stable branch, skipped)", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN_PER_LEVEL]
test_idx = perm[N_TRAIN_PER_LEVEL:N_TRAIN_PER_LEVEL + N_TEST_PER_LEVEL]

print(f"\nGenerating training data ({N_TRAIN_PER_LEVEL} samples x {len(FORCE_LEVELS)} levels)...", flush=True)
train_rows = build_dataset_multilevel(train_idx, FORCE_LEVELS, rng)
print("Generating test data...", flush=True)
test_rows = build_dataset_multilevel(test_idx, FORCE_LEVELS, rng)
print(f"  train rows: {len(train_rows)} (labeled: {sum(1 for r in train_rows if r['alpha'] is not None)})", flush=True)
print(f"  test rows: {len(test_rows)} (labeled: {sum(1 for r in test_rows if r['alpha'] is not None)})", flush=True)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

# ---- Train ----
train_labeled = [r for r in train_rows if r['alpha'] is not None]
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Zeta_t = torch.tensor([r['zeta'] for r in train_rows], dtype=torch.float32)
TP_t = torch.tensor([r['tp'] for r in train_rows], dtype=torch.float32)
Feat3_t = torch.tensor(np.stack([r['features'] for r in train_rows]), dtype=torch.float32)  # (n,3): shift,zeta,kappa
detune = torch.tanh(((W_t - torch.sqrt(1.0 + Feat3_t[:, 0])) / Feat3_t[:, 1]) / 20.0)
Feat_aug = torch.cat([Feat3_t, detune[:, None], TP_t[:, None]], dim=1)   # 3+1+1=5

is_labeled = torch.tensor([r['alpha'] is not None for r in train_rows])
Alpha_raw = np.array([r['alpha'] if r['alpha'] is not None else 0.0 for r in train_rows])
Beta_raw = np.array([r['beta'] if r['beta'] is not None else 0.0 for r in train_rows])
lab_mask_np = is_labeled.numpy()
Alpha_mean, Alpha_std = Alpha_raw[lab_mask_np].mean(), Alpha_raw[lab_mask_np].std()
Beta_mean, Beta_std = Beta_raw[lab_mask_np].mean(), Beta_raw[lab_mask_np].std()
Alpha_t = torch.tensor((Alpha_raw - Alpha_mean) / Alpha_std, dtype=torch.float32)
Beta_t = torch.tensor((Beta_raw - Beta_mean) / Beta_std, dtype=torch.float32)

Feat_mean, Feat_std = Feat_aug.mean(0), Feat_aug.std(0)
Feat_n = (Feat_aug - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)
F_arr = Zeta_t * 2 * TP_t
F_scale = float(F_arr.abs().max())
Omega_t = W_t * math.sqrt(inp['K_sec'][MODE, MODE] / inp['M_sec'][MODE, MODE])
K0 = inp['K_sec'][MODE, MODE]; M0 = inp['M_sec'][MODE, MODE]; C0 = inp['C_sec'][MODE, MODE]
K_t = K0 * (1.0 + Feat3_t[:, 0])

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [64, 64], 2, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 8000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=5e-5)
KL_BETA = 0.001
PHYSICS_WEIGHT = 0.05
n_data = float(is_labeled.sum().item())
print("\nTraining...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    alpha_p, beta_p = pred[:, 0], pred[:, 1]
    data_loss = ((alpha_p[is_labeled] - Alpha_t[is_labeled]) ** 2 +
                 (beta_p[is_labeled] - Beta_t[is_labeled]) ** 2).mean()
    alpha_phys = alpha_p * Alpha_std + Alpha_mean
    beta_phys = beta_p * Beta_std + Beta_mean
    kappa_t = Feat3_t[:, 2]
    r2_amp = alpha_phys ** 2 + beta_phys ** 2
    R1 = (1 - W_t ** 2) * alpha_phys + 2 * Zeta_t * W_t * beta_phys + kappa_t * r2_amp * alpha_phys - F_arr
    R2 = (1 - W_t ** 2) * beta_phys - 2 * Zeta_t * W_t * alpha_phys + kappa_t * r2_amp * beta_phys
    physics_loss = ((R1 / F_scale) ** 2 + (R2 / F_scale) ** 2).mean()
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
test_labeled = [r for r in test_rows if r['alpha'] is not None]
W_test = torch.tensor([r['w'] for r in test_labeled], dtype=torch.float32)
Feat3_test = torch.tensor(np.stack([r['features'] for r in test_labeled]), dtype=torch.float32)
TP_test = torch.tensor([r['tp'] for r in test_labeled], dtype=torch.float32)
detune_test = torch.tanh(((W_test - torch.sqrt(1.0 + Feat3_test[:, 0])) / Feat3_test[:, 1]) / 20.0)
Feat_test_aug = torch.cat([Feat3_test, detune_test[:, None], TP_test[:, None]], dim=1)
Feat_test_n = (Feat_test_aug - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)
Amp_test_raw = np.array([r['amplitude'] for r in test_labeled])
model.eval()
with torch.no_grad():
    preds = np.array([model(X_test).numpy() for _ in range(30)])
alpha_pred = preds[:, :, 0].mean(0) * Alpha_std + Alpha_mean
beta_pred = preds[:, :, 1].mean(0) * Beta_std + Beta_mean
amp_pred = np.hypot(alpha_pred, beta_pred)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


overall_r2 = r2(Amp_test_raw, amp_pred)
print(f"\nOverall test R^2 (amplitude, all force levels pooled): {overall_r2:.4f}", flush=True)
test_tp = np.array([r['tp'] for r in test_labeled])
for tp in FORCE_LEVELS:
    mask = np.isclose(test_tp, tp)
    if mask.sum() > 2:
        print(f"  target_peak_frac_qref={tp:.2f}: R^2={r2(Amp_test_raw[mask], amp_pred[mask]):.4f}  (n={mask.sum()})", flush=True)

# CASE 1 CHECK: near-linear limit resonance frequency vs real ANSYS mode 2 own frequency
low_mask = np.isclose(test_tp, FORCE_LEVELS[0])
if low_mask.sum() > 3:
    idx_peak = np.argmax(amp_pred[low_mask])
    w_at_peak = W_test.numpy()[low_mask][idx_peak]
    f_pred = w_at_peak * inp['freqs_sec'][MODE]
    print(f"\nCASE 1 CHECK: near-linear (target_peak={FORCE_LEVELS[0]}) resonance ~ w={w_at_peak:.4f}, "
          f"f={f_pred:.3f} Hz vs mode-2 own freq {inp['freqs_sec'][MODE]:.3f} Hz "
          f"(self-consistency reference, not real ANSYS)", flush=True)

fp_model = os.path.join(OUT, 'bpinn_forcing_aware_mode2_state.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, 'bpinn_forcing_aware_mode2_norm.npz')
np.savez(fp_norm, feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
          alpha_mean=Alpha_mean, alpha_std=Alpha_std, beta_mean=Beta_mean, beta_std=Beta_std,
          mode=MODE, r2_overall=overall_r2, is_forcing_aware=True, default_target_peak=0.8,
          force_levels=np.array(FORCE_LEVELS))
print(f"Saved: {fp_model}")
print(f"Saved: {fp_norm}")
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
