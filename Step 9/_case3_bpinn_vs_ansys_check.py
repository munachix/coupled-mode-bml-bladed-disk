"""One-off diagnostic (2026-08-21, explicit user request): does the trained
coupled (0,1) BPINN -- not just the ROM's own exact solver -- agree with the
real, converged ANSYS dynamic measurement at Case 3's own forcing level?

This closes a real gap found by direct inspection of PROJECT_STATUS.md
Section 9r item 7: every existing "BPINN comparison" in this project checks
BPINN against the ROM's OWN exact continuation/ODE solver, never against
real ANSYS. Real ANSYS dynamic data DOES exist (Section 9r item 6: 1.222mm
+/- 0.019mm, 100-cycle converged transient, mode 0, w=1.0) but was only ever
compared to the ROM (0.551mm SDOF / 0.556mm coupled), not to BPINN.

The production (0,1) BPINN (bpinn_coupled_state_01.pt) CANNOT answer this
directly -- it was trained at target_peak_frac_qref=0.8 (Step 6's own
convention), but Case 3's real ANSYS run used force_scale=2500, which
converts (via F_gen/(K*q_ref)/(2*zeta), step9.py's own rom_predict_steady_
state formula) to target_peak_frac_qref=0.1846 -- a genuinely different
forcing level (confirmed, not assumed: this exact mismatch is disclosed in
Section 9r item 7). Querying the production network at the wrong forcing
would be dishonest (extrapolating outside its training distribution and
calling it validation). So: train a SEPARATE one-off coupled BPINN at
Case 3's own exact forcing, evaluate it at w=1.0, zero mistuning (Case 3's
tuned baseline), compare against both the ROM's own exact solver (sanity
check: BPINN should reproduce this closely, since that's the ground truth
it was trained on) and the real ANSYS number (the actual open question).

Saved to a DIFFERENT filename than the production model -- does not touch
bpinn_coupled_state_01.pt.
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

MODE_I, MODE_J = 0, 1
REAL_ANSYS_AMP_MM = 1.222   # Section 9r item 6, 100-cycle converged transient
REAL_ANSYS_STD_MM = 0.019
ROM_COUPLED_AMP_MM = 0.556  # already-documented exact-solver prediction, sanity target

# Case 3's own real forcing level, converted to the BPINN's own
# target_peak_frac_qref convention (F_gen=force_scale=2500 for mode 0,
# same formula step9.py's rom_predict_steady_state uses to invert):
TARGET_PEAK = 0.184640   # computed this session: F_gen/(K0*q_ref)/(2*zeta), K0=3.384967e6, zeta=0.002

N_CYCLES = 200
STEPS_PER_CYCLE = 20
W_GRID = np.array([0.9, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14,
                    1.16, 1.18, 1.20, 1.22, 1.25, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6])
N_TRAIN = 100
N_TEST = 15
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 0.05
OUT = s6.OUT
TAG = 'case3check_01'

t_start = time.time()
print(f"=== Case 3 real-ANSYS-vs-BPINN check: coupled (0,1), TARGET_PEAK={TARGET_PEAK:.6f} "
      f"(Case 3's own real forcing, NOT the production model's 1.0) ===", flush=True)

inp = s6.load_inputs()
cc = s4.CONFIG['nonlinear']['cross_coupling'][(MODE_I, MODE_J)]
Ki0 = inp['K_sec'][MODE_I, MODE_I]; Kj0 = inp['K_sec'][MODE_J, MODE_J]
M_i = inp['M_sec'][MODE_I, MODE_I]; M_j = inp['M_sec'][MODE_J, MODE_J]
C_i = inp['C_sec'][MODE_I, MODE_I]; C_j = inp['C_sec'][MODE_J, MODE_J]
omega0_i = math.sqrt(Ki0 / M_i)
zeta_i0 = C_i / (2 * math.sqrt(Ki0 * M_i))
zeta_j0 = C_j / (2 * math.sqrt(Kj0 * M_j))

Fg_i = TARGET_PEAK * 2 * zeta_i0 * Ki0
Fg_j = TARGET_PEAK * 2 * zeta_j0 * Kj0
F_scale = max(abs(Fg_i), abs(Fg_j))
print(f"  K_i={Ki0:.4e}, K_j={Kj0:.4e}, Fg_i={Fg_i:.4e}, Fg_j={Fg_j:.4e}", flush=True)
print(f"  (sanity check: Fg_i should equal Case 3's real force_scale=2500 -- Fg_i={Fg_i:.2f})", flush=True)


def sample_features(sample_idx):
    row = {v: inp['theta'][v][sample_idx] for v in s6.VAR_NAMES}
    df = s5.compute_delta_f_vectorized({k: v[None, :] for k, v in row.items()},
                                        inp['sens'], inp['L_ref'], inp['t_ref'])[0]
    scale = (1.0 + df) ** 2 - 1.0
    shift_i = float(scale @ inp['P'][:, MODE_I])
    shift_j = float(scale @ inp['P'][:, MODE_J])
    K_i = Ki0 * (1.0 + shift_i)
    K_j = Kj0 * (1.0 + shift_j)
    zeta_i = C_i / (2 * math.sqrt(K_i * M_i))
    zeta_j = C_j / (2 * math.sqrt(K_j * M_j))
    kappa_i = 0.75 * inp['K3_sec_diag'][MODE_I] / K_i
    kappa_j = 0.75 * inp['K3_sec_diag'][MODE_J] / K_j
    feat = np.array([shift_i, zeta_i, kappa_i, shift_j, zeta_j, kappa_j])
    return dict(K_i=K_i, K_j=K_j, feat=feat)


def build_dataset(sample_indices):
    rows = []
    n_rejected = 0
    for k, i in enumerate(sample_indices):
        p = sample_features(i)
        for w in W_GRID:
            Omega = w * omega0_i
            r = s4.duffing_forced_response_coupled(
                (MODE_I, MODE_J), (p['K_i'], p['K_j']), (M_i, M_j), (C_i, C_j),
                cc['coef0'], cc['coef1'], (Fg_i, Fg_j), Omega,
                n_cycles=N_CYCLES, steps_per_cycle=STEPS_PER_CYCLE)
            ok = (np.isfinite(r['amp_i']) and np.isfinite(r['amp_j'])
                  and abs(r['amp_i']) < DIVERGE_BOUND and abs(r['amp_j']) < DIVERGE_BOUND)
            if not ok:
                n_rejected += 1
                continue
            feat8 = s6.add_detune_features(w, p['feat'])
            rows.append(dict(w=w, feat=feat8, K_i=p['K_i'], K_j=p['K_j'],
                              alpha_i=r['alpha_i'], beta_i=r['beta_i'],
                              alpha_j=r['alpha_j'], beta_j=r['beta_j'],
                              amp_i=r['amp_i'], amp_j=r['amp_j']))
        if k % 20 == 0:
            print(f"  sample {k}/{len(sample_indices)}, elapsed={time.time()-t_start:.0f}s, "
                  f"rejected so far={n_rejected}", flush=True)
    print(f"  Total rejected: {n_rejected} of {len(sample_indices)*len(W_GRID)}", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN]
test_idx = perm[N_TRAIN:N_TRAIN + N_TEST]

print(f"Generating training data ({N_TRAIN} x {len(W_GRID)} = {N_TRAIN*len(W_GRID)} coupled solves)...", flush=True)
train_rows = build_dataset(train_idx)
print(f"Generating test data ({N_TEST} x {len(W_GRID)})...", flush=True)
test_rows = build_dataset(test_idx)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Omega_t = W_t * omega0_i
K_i_t = torch.tensor([r['K_i'] for r in train_rows], dtype=torch.float32)
K_j_t = torch.tensor([r['K_j'] for r in train_rows], dtype=torch.float32)

Ai_raw = np.array([r['alpha_i'] for r in train_rows]); Bi_raw = np.array([r['beta_i'] for r in train_rows])
Aj_raw = np.array([r['alpha_j'] for r in train_rows]); Bj_raw = np.array([r['beta_j'] for r in train_rows])
Ai_mean, Ai_std = Ai_raw.mean(), Ai_raw.std()
Bi_mean, Bi_std = Bi_raw.mean(), Bi_raw.std()
Aj_mean, Aj_std = Aj_raw.mean(), Aj_raw.std()
Bj_mean, Bj_std = Bj_raw.mean(), Bj_raw.std()
Ai_t = torch.tensor((Ai_raw - Ai_mean) / Ai_std, dtype=torch.float32)
Bi_t = torch.tensor((Bi_raw - Bi_mean) / Bi_std, dtype=torch.float32)
Aj_t = torch.tensor((Aj_raw - Aj_mean) / Aj_std, dtype=torch.float32)
Bj_t = torch.tensor((Bj_raw - Bj_mean) / Bj_std, dtype=torch.float32)

Amp_i_true_t = torch.tensor(np.array([r['amp_i'] for r in train_rows]), dtype=torch.float32)
Amp_j_true_t = torch.tensor(np.array([r['amp_j'] for r in train_rows]), dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)

coef0_t = [float(x) for x in cc['coef0']]
coef1_t = [float(x) for x in cc['coef1']]

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [48, 48], 4, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 8000
KL_BETA = 0.001
n_data = float(len(train_rows))
print("Training...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    ai_p, bi_p, aj_p, bj_p = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    data_loss = ((ai_p - Ai_t) ** 2 + (bi_p - Bi_t) ** 2
                 + (aj_p - Aj_t) ** 2 + (bj_p - Bj_t) ** 2).mean()
    ai_phys = ai_p * Ai_std + Ai_mean
    bi_phys = bi_p * Bi_std + Bi_mean
    aj_phys = aj_p * Aj_std + Aj_mean
    bj_phys = bj_p * Bj_std + Bj_mean
    Ra_i, Rb_i, Ra_j, Rb_j = s4.coupled_hbm_residual(
        (K_i_t, K_j_t), (M_i, M_j), (C_i, C_j), coef0_t, coef1_t, (Fg_i, Fg_j), Omega_t,
        ai_phys, bi_phys, aj_phys, bj_phys)
    physics_loss = ((Ra_i / F_scale) ** 2 + (Rb_i / F_scale) ** 2
                     + (Ra_j / F_scale) ** 2 + (Rb_j / F_scale) ** 2).mean()
    amp_i_pred = torch.sqrt(ai_phys ** 2 + bi_phys ** 2 + 1e-12)
    amp_j_pred = torch.sqrt(aj_phys ** 2 + bj_phys ** 2 + 1e-12)
    amp_loss = (((amp_i_pred - Amp_i_true_t) / Ai_std) ** 2
                + ((amp_j_pred - Amp_j_true_t) / Aj_std) ** 2).mean()
    kl = model.total_kl() / n_data
    anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
    loss = data_loss + amp_loss + anneal * PHYSICS_WEIGHT * physics_loss + anneal * KL_BETA * kl
    loss.backward()
    opt.step()
    if epoch % 2000 == 0:
        print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  amp={amp_loss.item():.6f}  "
              f"physics={physics_loss.item():.6f}  kl={kl.item():.5f}", flush=True)

# ---- Validate on held-out test set (same as production trainer) ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
Amp_i_test_raw = np.array([r['amp_i'] for r in test_rows])
Amp_j_test_raw = np.array([r['amp_j'] for r in test_rows])
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)

model.eval()
n_mc = 30
with torch.no_grad():
    preds_test = np.array([model(X_test).numpy() for _ in range(n_mc)])
amp_i_test_pred = np.hypot(preds_test[:, :, 0].mean(0) * Ai_std + Ai_mean,
                            preds_test[:, :, 1].mean(0) * Bi_std + Bi_mean)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_test = r2(Amp_i_test_raw, amp_i_test_pred)
print(f"\nHeld-out test R^2 (amplitude, mode 0, at Case 3's own forcing) = {r2_test:.4f}", flush=True)

# ---- THE ACTUAL CHECK: evaluate at w=1.0, ZERO mistuning (Case 3's tuned baseline) ----
feat0 = np.array([0.0, zeta_i0, 0.75 * inp['K3_sec_diag'][MODE_I] / Ki0,
                   0.0, zeta_j0, 0.75 * inp['K3_sec_diag'][MODE_J] / Kj0])
w_check = np.array([1.0])
feat0_8 = s6.add_detune_features(w_check, feat0[None, :])
Feat0_n = (torch.tensor(feat0_8, dtype=torch.float32) - Feat_mean) / Feat_std
X_check = torch.cat([s6.fourier_encode_w(torch.tensor(w_check, dtype=torch.float32)), Feat0_n], dim=1)
with torch.no_grad():
    preds_check = np.array([model(X_check).numpy() for _ in range(100)])
alpha_i_c = preds_check[:, 0, 0] * Ai_std + Ai_mean
beta_i_c = preds_check[:, 0, 1] * Bi_std + Bi_mean
amp_i_c_generalized = np.hypot(alpha_i_c, beta_i_c)   # GENERALIZED coordinate, mm-equivalent (q_ref=1mm) -- NOT yet physical displacement at any specific node
# CORRECTION (found on first run of this script): duffing_forced_response_coupled's
# amp_i is the raw generalized coordinate q_i, same as the continuation solver's own
# alpha*q_ref -- comparing it directly against ROM_COUPLED_AMP_MM (which IS the
# physical u0=q0*Phi at node 1171, per rom_predict_steady_state/run_case3_transient_point)
# is exactly the units bug this whole check exists to rule OUT in the original code.
# Must multiply by the SAME target-DOF mode-shape value (node 1171, UZ, Phi=28.3852,
# from step9.py's own case3_convergence_run.log) to get an apples-to-apples number.
PHI_TARGET_DOF = 28.3852   # node 1171, UZ, mode 0 -- s9._target_dof_for_mode's own value
amp_i_c = amp_i_c_generalized * PHI_TARGET_DOF   # NOW physical displacement, mm, at node 1171 UZ
bpinn_amp_mean = float(amp_i_c.mean())
bpinn_amp_std = float(amp_i_c.std())

print(f"\n{'='*70}")
print(f"CASE 3 CHECK: BPINN vs ROM vs REAL ANSYS, mode 0, w=1.0, tuned baseline")
print(f"{'='*70}")
print(f"  BPINN (trained at Case 3's real forcing): {bpinn_amp_mean:.4f} +/- {bpinn_amp_std:.4f} mm")
print(f"  ROM exact solver (coupled, documented):    {ROM_COUPLED_AMP_MM:.4f} mm")
print(f"  Real ANSYS (100-cycle converged transient): {REAL_ANSYS_AMP_MM:.4f} +/- {REAL_ANSYS_STD_MM:.4f} mm")
print(f"  BPINN vs ROM ratio:        {bpinn_amp_mean/ROM_COUPLED_AMP_MM:.4f}x  (should be near 1.0 -- sanity check)")
print(f"  BPINN vs real ANSYS ratio: {bpinn_amp_mean/REAL_ANSYS_AMP_MM:.4f}x  (the actual open question)")
print(f"{'='*70}")

fp = os.path.join(OUT, f'bpinn_coupled_state_{TAG}.pt')
torch.save(model.state_dict(), fp)
np.savez(os.path.join(OUT, f'bpinn_coupled_norm_{TAG}.npz'),
          feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
          alpha_i_mean=Ai_mean, alpha_i_std=Ai_std, beta_i_mean=Bi_mean, beta_i_std=Bi_std,
          alpha_j_mean=Aj_mean, alpha_j_std=Aj_std, beta_j_mean=Bj_mean, beta_j_std=Bj_std,
          target_peak=TARGET_PEAK, r2_test=r2_test,
          bpinn_amp_mean=bpinn_amp_mean, bpinn_amp_std=bpinn_amp_std,
          rom_amp=ROM_COUPLED_AMP_MM, real_ansys_amp=REAL_ANSYS_AMP_MM, real_ansys_std=REAL_ANSYS_STD_MM)
print(f"Saved (diagnostic, NOT the production model): {fp}")
print(f"TOTAL TIME: {time.time()-t_start:.0f}s")
print("DONE")
