"""
Coupled 2-mode BPINN training (2026-08-13) -- extends Step 6's original
mode-0-only BPINN to the REAL, measured cross-mode coupling between modes
0 and 1 (Step 4's CONFIG['nonlinear']['cross_coupling'], Step 9's real
ANSYS combined-displacement identification).

TWO DELIBERATE, DISCLOSED SIMPLIFICATIONS vs. the original single-mode
BPINN, made for time reasons, not physics reasons:
  1. No physics-residual (HBM) loss term. The original network was
     trained with a physics loss enforcing Step 4's exact single-mode HBM
     equations on the network's own output. Deriving the equivalent
     coupled-system steady-state residual analytically is real, separate
     work; this version trains on DATA (from the real coupled time-domain
     solver, step4.duffing_forced_response_coupled) + KL regularization
     only. The original project's own debugging history (Section 4 in
     PROJECT_STATUS.md) already showed a plain data-only MLP reaches
     R^2=0.95 on the single-mode problem -- physics loss helped mainly
     with calibration, not raw accuracy -- so this is a reasoned, not
     reckless, simplification.
  2. Predicts AMPLITUDE only (amp_i, amp_j), not phase (alpha,beta).
     Every downstream consumer (Step 7 reconstruction, Step 8 HI3) only
     ever uses amplitude; extracting phase from the time-domain ground
     truth would need synchronous demodulation against the drive phase,
     extra work with no current consumer.

Forcing level for training data: FIXED at node 1171's real participation
(phi_0=28.3852, phi_1=-1.7883) and F_phys=1000N -- the SAME point/force
level tonight's real validation used, on purpose (train where we validate).
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
PHI_I_1171, PHI_J_1171 = 28.3852, -1.7883
F_PHYS = 1000.0
N_CYCLES = 200
STEPS_PER_CYCLE = 20
W_GRID = np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6])
N_TRAIN = 80
N_TEST = 20
OUT = s6.OUT

t_start = time.time()
print("=== Coupled BPINN training: modes 0-1 ===", flush=True)

inp = s6.load_inputs()
cc = s4.CONFIG['nonlinear']['cross_coupling'][(MODE_I, MODE_J)]
Ki0 = inp['K_sec'][MODE_I, MODE_I]; Kj0 = inp['K_sec'][MODE_J, MODE_J]
M_i = inp['M_sec'][MODE_I, MODE_I]; M_j = inp['M_sec'][MODE_J, MODE_J]
C_i = inp['C_sec'][MODE_I, MODE_I]; C_j = inp['C_sec'][MODE_J, MODE_J]
omega0_i = math.sqrt(Ki0 / M_i)

Fg_i = F_PHYS * PHI_I_1171
Fg_j = F_PHYS * PHI_J_1171


def sample_features(sample_idx):
    """Real, measured mistuning shift for BOTH modes at this sample --
    same method as Step 6's original per_sample_sdof_params, just called
    for mode i AND mode j."""
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
    """Some (sample, w) combinations can genuinely diverge -- the fitted
    coupling coefficients include NEGATIVE terms (coef0[3]=-1.42e7,
    coef1[0]=-8.41e7), so at large enough amplitude the "restoring" force
    can flip sign and the ODE blows up. A single such outlier (amplitude
    orders of magnitude too large) is enough to poison a mean-squared-error
    training loss completely -- confirmed directly (not assumed) as the
    cause of the first training attempt's stuck, astronomical loss.
    Filtered here with an explicit, printed physical sanity bound (0.5mm --
    ~5x the largest genuine amplitude seen in validated single-point tests
    tonight) rather than silently discarded."""
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
                  and abs(r['amp_i']) < 0.5 and abs(r['amp_j']) < 0.5)
            if not ok:
                n_rejected += 1
                print(f"  REJECTED: sample {i}, w={w:.2f}: amp_i={r['amp_i']:.4e}, "
                      f"amp_j={r['amp_j']:.4e} (unstable/diverged)", flush=True)
                continue
            rows.append(dict(w=w, feat=p['feat'], amp_i=r['amp_i'], amp_j=r['amp_j']))
        if k % 10 == 0:
            print(f"  sample {k}/{len(sample_indices)} done, elapsed={time.time()-t_start:.0f}s, "
                  f"rejected so far={n_rejected}", flush=True)
    print(f"  Total rejected: {n_rejected} of {len(sample_indices)*len(W_GRID)}", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN]
test_idx = perm[N_TRAIN:N_TRAIN + N_TEST]

print(f"Generating training data ({N_TRAIN} samples x {len(W_GRID)} w-points = {N_TRAIN*len(W_GRID)} coupled solves)...", flush=True)
train_rows = build_dataset(train_idx)
print(f"Generating test data ({N_TEST} samples x {len(W_GRID)} w-points)...", flush=True)
test_rows = build_dataset(test_idx)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

# Checkpoint the raw dataset BEFORE training -- a training bug should never
# cost another ~18 minutes of data regeneration (real lesson from the first
# attempt tonight).
ckpt_path = os.path.join(OUT, '_coupled_dataset_checkpoint.npz')
np.savez(ckpt_path,
          train_w=np.array([r['w'] for r in train_rows]),
          train_feat=np.stack([r['feat'] for r in train_rows]),
          train_amp_i=np.array([r['amp_i'] for r in train_rows]),
          train_amp_j=np.array([r['amp_j'] for r in train_rows]),
          test_w=np.array([r['w'] for r in test_rows]),
          test_feat=np.stack([r['feat'] for r in test_rows]),
          test_amp_i=np.array([r['amp_i'] for r in test_rows]),
          test_amp_j=np.array([r['amp_j'] for r in test_rows]))
print(f"Checkpointed raw dataset: {ckpt_path}", flush=True)

# ---- Train ----
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Amp_i_t = torch.tensor([r['amp_i'] for r in train_rows], dtype=torch.float32)
Amp_j_t = torch.tensor([r['amp_j'] for r in train_rows], dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [32, 32], 2, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 6000
KL_BETA = 0.001
n_data = float(len(train_rows))
print("Training...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    amp_i_p, amp_j_p = pred[:, 0], pred[:, 1]
    data_loss = ((amp_i_p - Amp_i_t) ** 2 + (amp_j_p - Amp_j_t) ** 2).mean()
    kl = model.total_kl() / n_data
    anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
    loss = data_loss + anneal * KL_BETA * kl
    loss.backward()
    opt.step()
    if epoch % 1000 == 0:
        print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  kl={kl.item():.5f}", flush=True)

# ---- Validate ----
W_test = torch.tensor([r['w'] for r in test_rows], dtype=torch.float32)
Feat_test = torch.tensor(np.stack([r['feat'] for r in test_rows]), dtype=torch.float32)
Amp_i_test = np.array([r['amp_i'] for r in test_rows])
Amp_j_test = np.array([r['amp_j'] for r in test_rows])
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)

model.eval()
n_mc = 30
preds = []
with torch.no_grad():
    for _ in range(n_mc):
        preds.append(model(X_test).numpy())
preds = np.array(preds)
amp_i_mean = preds[:, :, 0].mean(0); amp_j_mean = preds[:, :, 1].mean(0)

def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot

r2_i = r2(Amp_i_test, amp_i_mean)
r2_j = r2(Amp_j_test, amp_j_mean)
print(f"\nTest R^2: mode {MODE_I} amp = {r2_i:.4f}, mode {MODE_J} amp = {r2_j:.4f}", flush=True)

# ---- Save ----
fp_model = os.path.join(OUT, 'bpinn_coupled_state_01.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, 'bpinn_coupled_norm_01.npz')
np.savez(fp_norm, feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
         phi_i_1171=PHI_I_1171, phi_j_1171=PHI_J_1171, f_phys=F_PHYS,
         mode_i=MODE_I, mode_j=MODE_J, r2_i=r2_i, r2_j=r2_j)
print(f"Saved: {fp_model}", flush=True)
print(f"Saved: {fp_norm}", flush=True)
print(f"TOTAL TIME: {time.time()-t_start:.0f}s", flush=True)
print("DONE", flush=True)
