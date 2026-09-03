"""INDEPENDENTLY-FORCED BPINN for a coupled pair (2026-08-24) -- fixes a
real architectural gap found while building the ANSYS-vs-BPINN FRF plots:
the original forcing-aware pair network (_train_forcing_aware_01.py) takes
ONE shared target_peak that drives BOTH modes proportionally via their own
zeta*K (Fg_i=tp*2*zeta_i*K_i, Fg_j=tp*2*zeta_j*K_j) -- it CANNOT represent
"mode i forced, mode j silent," which is exactly the real ANSYS/ROM
harmonic-FRF force convention (F_gen[0]=2500N, F_gen[1:]=0; neighbor modes
respond ONLY through real cross-mode coupling). Feeding that network a
single shared tp for this scenario silently implied mode j was ALSO being
driven hard (since modes 0,1 have nearly identical zeta*K, being
near-degenerate) -- producing a smeared, unphysical double-mode response
instead of the real sharp single resonance (confirmed directly: predicted
peak was ~2.7x too small and completely lacked the real ~5Hz-wide Q~250
resonance shape).

Fix: two INDEPENDENT forcing inputs (target_peak_i, target_peak_j), so the
network can represent the whole space of physically real scenarios --
single-mode-forced (Case 1/2/4's real FRF convention), point-load-shared
(Case 3's real convention), and everything between -- not just the
shared-tp diagonal it was restricted to before.
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

MODE_I, MODE_J = int(sys.argv[1]), int(sys.argv[2])
OUT = s6.OUT
CASE3_TARGET_PEAK = 0.184640
# (tp_i, tp_j) combinations -- covers single-mode-forced (real FRF
# convention, both orderings), the original shared-tp diagonal (real
# Case-3 point-load-like scenario, both low and mid), and 2 asymmetric
# combinations for general coverage/robustness.
TP_COMBOS = [
    (CASE3_TARGET_PEAK, 0.0),
    (0.0, CASE3_TARGET_PEAK),
    (0.02, 0.02),
    (CASE3_TARGET_PEAK, CASE3_TARGET_PEAK),
    (0.4, 0.4),
    (0.6, 0.15),
]
N_TRAIN_PER_LEVEL = int(os.environ.get('N_TRAIN_PER_LEVEL', 40))
N_TEST_PER_LEVEL = int(os.environ.get('N_TEST_PER_LEVEL', 8))
W_GRID = np.unique(np.concatenate([
    np.arange(0.970, 1.101, 0.004),
    np.arange(0.95, 1.61, 0.03),
    [1.7, 2.0, 2.3, 2.6],
]))
N_CYCLES = 200
STEPS_PER_CYCLE = 20
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 0.05

t_start = time.time()
print(f"=== Independently-forced BPINN, pair ({MODE_I},{MODE_J}) ===", flush=True)
print(f"  (tp_i, tp_j) combos: {TP_COMBOS}", flush=True)
print(f"  W_GRID: {len(W_GRID)} points, [{W_GRID.min():.2f}, {W_GRID.max():.2f}]", flush=True)

inp = s6.load_inputs()
cc = s4.CONFIG['nonlinear']['cross_coupling'][(MODE_I, MODE_J)]
Ki0 = inp['K_sec'][MODE_I, MODE_I]; Kj0 = inp['K_sec'][MODE_J, MODE_J]
M_i = inp['M_sec'][MODE_I, MODE_I]; M_j = inp['M_sec'][MODE_J, MODE_J]
C_i = inp['C_sec'][MODE_I, MODE_I]; C_j = inp['C_sec'][MODE_J, MODE_J]
omega0_i = math.sqrt(Ki0 / M_i)
zeta_i0 = C_i / (2 * math.sqrt(Ki0 * M_i))
zeta_j0 = C_j / (2 * math.sqrt(Kj0 * M_j))


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


def build_dataset(sample_indices, combos):
    rows = []
    n_rejected = 0
    n_total = 0
    for k, i in enumerate(sample_indices):
        p = sample_features(i)
        for tp_i, tp_j in combos:
            Fg_i = tp_i * 2 * zeta_i0 * Ki0
            Fg_j = tp_j * 2 * zeta_j0 * Kj0
            # 2026-08-24 FIX: single-mode-forced combos (one of tp_i/tp_j
            # exactly 0) need far more cycles to settle when the pair is
            # near-degenerate -- confirmed via direct convergence check for
            # pair (0,1) (0.047 Hz split): amp_i at n_cycles=200 was still
            # 7% off its converged value (200->1000 cycles: 0.01231->
            # 0.01146, flat beyond 1000). Forcing only one of two
            # near-identical-frequency coupled oscillators lets energy
            # slowly slosh into the unforced one on a timescale set by the
            # coupling, not the damping -- a genuinely slower transient than
            # the both-forced combos (which settle fast, already verified
            # at 200 cycles, R^2=0.87-0.90). This is NOT needed for
            # well-separated pairs (pair (3,4), 292Hz apart... actually
            # separated by more like several Hz, already got a clean 1.0%
            # FRF check at 200 cycles) -- but applying it here whenever one
            # side is silent is a safe, general rule, not pair-specific.
            n_cyc = 1500 if (tp_i == 0.0 or tp_j == 0.0) else N_CYCLES
            for w in W_GRID:
                n_total += 1
                Omega = w * omega0_i
                r = s4.duffing_forced_response_coupled(
                    (MODE_I, MODE_J), (p['K_i'], p['K_j']), (M_i, M_j), (C_i, C_j),
                    cc['coef0'], cc['coef1'], (Fg_i, Fg_j), Omega,
                    n_cycles=n_cyc, steps_per_cycle=STEPS_PER_CYCLE)
                ok = (np.isfinite(r['amp_i']) and np.isfinite(r['amp_j'])
                      and abs(r['amp_i']) < DIVERGE_BOUND and abs(r['amp_j']) < DIVERGE_BOUND)
                if not ok:
                    n_rejected += 1
                    continue
                feat8 = s6.add_detune_features(w, p['feat'])
                feat_aug = np.concatenate([feat8, [tp_i, tp_j]])   # BOTH forcing levels as explicit inputs
                rows.append(dict(w=w, feat=feat_aug, K_i=p['K_i'], K_j=p['K_j'], tp_i=tp_i, tp_j=tp_j,
                                  alpha_i=r['alpha_i'], beta_i=r['beta_i'],
                                  alpha_j=r['alpha_j'], beta_j=r['beta_j'],
                                  amp_i=r['amp_i'], amp_j=r['amp_j']))
        if k % 10 == 0:
            print(f"  sample {k}/{len(sample_indices)}, elapsed={time.time()-t_start:.0f}s, "
                  f"rejected so far={n_rejected}/{n_total}", flush=True)
    print(f"  Total rejected: {n_rejected} of {n_total}", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN_PER_LEVEL]
test_idx = perm[N_TRAIN_PER_LEVEL:N_TRAIN_PER_LEVEL + N_TEST_PER_LEVEL]

print(f"\nGenerating training data ({N_TRAIN_PER_LEVEL} samples x {len(TP_COMBOS)} combos x "
      f"{len(W_GRID)} w-points = {N_TRAIN_PER_LEVEL*len(TP_COMBOS)*len(W_GRID)} solves)...", flush=True)
train_rows = build_dataset(train_idx, TP_COMBOS)
print(f"\nGenerating test data...", flush=True)
test_rows = build_dataset(test_idx, TP_COMBOS)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

if len(train_rows) < 200 or len(test_rows) < 30:
    print("ABORT: too few surviving points", flush=True)
    sys.exit(1)

TAG = f'{MODE_I}{MODE_J}'
ckpt_path = os.path.join(OUT, f'_forcing_aware_indep_{TAG}_checkpoint.npz')
np.savez(ckpt_path,
          train_w=np.array([r['w'] for r in train_rows]), train_feat=np.stack([r['feat'] for r in train_rows]),
          train_tp_i=np.array([r['tp_i'] for r in train_rows]), train_tp_j=np.array([r['tp_j'] for r in train_rows]),
          train_alpha_i=np.array([r['alpha_i'] for r in train_rows]), train_beta_i=np.array([r['beta_i'] for r in train_rows]),
          train_alpha_j=np.array([r['alpha_j'] for r in train_rows]), train_beta_j=np.array([r['beta_j'] for r in train_rows]),
          test_w=np.array([r['w'] for r in test_rows]), test_feat=np.stack([r['feat'] for r in test_rows]),
          test_tp_i=np.array([r['tp_i'] for r in test_rows]), test_tp_j=np.array([r['tp_j'] for r in test_rows]),
          test_alpha_i=np.array([r['alpha_i'] for r in test_rows]), test_beta_i=np.array([r['beta_i'] for r in test_rows]),
          test_alpha_j=np.array([r['alpha_j'] for r in test_rows]), test_beta_j=np.array([r['beta_j'] for r in test_rows]),
          test_amp_i=np.array([r['amp_i'] for r in test_rows]), test_amp_j=np.array([r['amp_j'] for r in test_rows]))
print(f"Checkpointed: {ckpt_path}", flush=True)

# ---- Train ----
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Omega_t = W_t * omega0_i
K_i_t = torch.tensor([r['K_i'] for r in train_rows], dtype=torch.float32)
K_j_t = torch.tensor([r['K_j'] for r in train_rows], dtype=torch.float32)
Fg_i_t = torch.tensor([r['tp_i'] * 2 * zeta_i0 * Ki0 for r in train_rows], dtype=torch.float32)
Fg_j_t = torch.tensor([r['tp_j'] * 2 * zeta_j0 * Kj0 for r in train_rows], dtype=torch.float32)

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
F_scale = max(float(Fg_i_t.abs().max()), float(Fg_j_t.abs().max()))

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [96, 96], 4, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 12000
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=5e-5)
KL_BETA = 0.001
n_data = float(len(train_rows))
print("\nTraining...", flush=True)
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
        (K_i_t, K_j_t), (M_i, M_j), (C_i, C_j), coef0_t, coef1_t, (Fg_i_t, Fg_j_t), Omega_t,
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
    sched.step()
    if epoch % 3000 == 0:
        print(f"  epoch {epoch:6d}  data={data_loss.item():.6f}  amp={amp_loss.item():.6f}  "
              f"physics={physics_loss.item():.6f}  kl={kl.item():.5f}", flush=True)

# ---- Validate on held-out test set ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
Amp_i_test_raw = np.array([r['amp_i'] for r in test_rows])
Amp_j_test_raw = np.array([r['amp_j'] for r in test_rows])
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)
model.eval()
with torch.no_grad():
    preds = np.array([model(X_test).numpy() for _ in range(30)])
alpha_i_mean = preds[:, :, 0].mean(0) * Ai_std + Ai_mean
beta_i_mean = preds[:, :, 1].mean(0) * Bi_std + Bi_mean
alpha_j_mean = preds[:, :, 2].mean(0) * Aj_std + Aj_mean
beta_j_mean = preds[:, :, 3].mean(0) * Bj_std + Bj_mean
amp_i_pred = np.hypot(alpha_i_mean, beta_i_mean)
amp_j_pred = np.hypot(alpha_j_mean, beta_j_mean)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_i_overall = r2(Amp_i_test_raw, amp_i_pred)
r2_j_overall = r2(Amp_j_test_raw, amp_j_pred)
print(f"\nOverall test R^2 (amplitude, all combos pooled): mode {MODE_I}={r2_i_overall:.4f}, "
      f"mode {MODE_J}={r2_j_overall:.4f}", flush=True)
test_tpi = np.array([r['tp_i'] for r in test_rows])
test_tpj = np.array([r['tp_j'] for r in test_rows])
for tp_i, tp_j in TP_COMBOS:
    mask = np.isclose(test_tpi, tp_i) & np.isclose(test_tpj, tp_j)
    if mask.sum() > 2:
        r2_i_level = r2(Amp_i_test_raw[mask], amp_i_pred[mask])
        r2_j_level = r2(Amp_j_test_raw[mask], amp_j_pred[mask])
        print(f"  (tp_i={tp_i:.4f}, tp_j={tp_j:.4f}): R^2 mode_i={r2_i_level:.4f}, "
              f"R^2 mode_j={r2_j_level:.4f}  (n={mask.sum()})", flush=True)

# ---- REAL FRF CHECK: mode-i-only forced, tuned, sweep w ----
w_sweep = np.linspace(0.90, 1.15, 201)
feat6_c1 = np.array([0.0, zeta_i0, 0.75 * inp['K3_sec_diag'][MODE_I] / Ki0,
                      0.0, zeta_j0, 0.75 * inp['K3_sec_diag'][MODE_J] / Kj0])
feat_arr_c1 = np.tile(feat6_c1, (len(w_sweep), 1))
feat8_c1 = s6.add_detune_features(w_sweep, feat_arr_c1)
tp_i_c1 = np.full(len(w_sweep), CASE3_TARGET_PEAK)
tp_j_c1 = np.zeros(len(w_sweep))
feat_aug_c1 = np.concatenate([feat8_c1, tp_i_c1[:, None], tp_j_c1[:, None]], axis=1)
Feat_n_c1 = (torch.tensor(feat_aug_c1, dtype=torch.float32) - Feat_mean) / Feat_std
X_c1 = torch.cat([s6.fourier_encode_w(torch.tensor(w_sweep, dtype=torch.float32)), Feat_n_c1], dim=1)
with torch.no_grad():
    preds_c1 = np.array([model(X_c1).numpy() for _ in range(40)])
amp_i_c1 = np.hypot(preds_c1[:, :, 0].mean(0) * Ai_std + Ai_mean, preds_c1[:, :, 1].mean(0) * Bi_std + Bi_mean)
peak_idx = np.argmax(amp_i_c1)
peak_w = w_sweep[peak_idx]
peak_freq = peak_w * omega0_i / (2 * np.pi)
CASE1_REAL_FREQ_HZ = 292.818 if MODE_I == 0 else float(inp['freqs_sec'][MODE_I])
ref_label = "Real ANSYS Case 1 (tuned)" if MODE_I == 0 else "ROM tuned freqs_sec[MODE_I] (self-consistency ref)"
print(f"\n{'='*70}\nREAL FRF CHECK: mode-{MODE_I}-only forced (tp_i={CASE3_TARGET_PEAK}, tp_j=0), "
      f"matching real ANSYS/ROM harmonic force convention\n{'='*70}")
print(f"  Predicted resonance peak: w={peak_w:.4f}, f={peak_freq:.4f} Hz, amp={amp_i_c1.max():.5f}")
print(f"  {ref_label}: {CASE1_REAL_FREQ_HZ:.4f} Hz")
print(f"  Error: {abs(peak_freq-CASE1_REAL_FREQ_HZ):.4f} Hz ({100*abs(peak_freq-CASE1_REAL_FREQ_HZ)/CASE1_REAL_FREQ_HZ:.4f}%)")

torch.save(model.state_dict(), os.path.join(OUT, f'bpinn_forcing_aware_indep_{TAG}_state.pt'))
np.savez(os.path.join(OUT, f'bpinn_forcing_aware_indep_{TAG}_norm.npz'),
          feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
          alpha_i_mean=Ai_mean, alpha_i_std=Ai_std, beta_i_mean=Bi_mean, beta_i_std=Bi_std,
          alpha_j_mean=Aj_mean, alpha_j_std=Aj_std, beta_j_mean=Bj_mean, beta_j_std=Bj_std,
          r2_i_overall=r2_i_overall, r2_j_overall=r2_j_overall, is_indep_forcing=True,
          case1_peak_freq=peak_freq, case1_real_freq=CASE1_REAL_FREQ_HZ,
          tp_combos=np.array(TP_COMBOS))
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
