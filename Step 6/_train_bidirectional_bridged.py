"""Bidirectional/Gauss-Seidel chain BPINN for the bridged 0-1-2-3-4 system
(2026-08-23) -- the one-directional sequential version (left-neighbor only)
improved mean R^2 to 0.54-0.61 but left real accuracy on the table for
INTERIOR modes (1,2,3): the real physics (s4.duffing_forced_response_chain)
couples every interior mode to BOTH its left AND right neighbor
simultaneously, confirmed directly in that function's own implementation,
not assumed. Giving each interior sub-network only the left neighbor's
state was a real, fixable gap, not a fundamental limit.

Fix: interior modes (1,2,3) take BOTH neighbors' (alpha,beta) as explicit
input (8-dim: own 4 [shift,zeta,kappa,detune] + left alpha/beta + right
alpha/beta); boundary modes (0,4) keep their single-neighbor form (mode 0
has no left neighbor at all; mode 4 keeps its left-only design from the
sequential version, since it already reached R^2=0.90+ there). Trained
with full teacher forcing (TRUE neighbor values from the cached dataset --
well-defined regardless of ordering, since both neighbors' ground truth is
already known for every training row). At INFERENCE time, since the right
neighbor isn't known until after a first pass, resolve via Gauss-Seidel
iteration: forward cascade for an initial guess, then repeated sweeps
re-predicting each interior mode from its neighbors' current best
estimates until converged (typically 2-3 sweeps for a well-damped fixed
point, checked directly).
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
train_w = d['train_w']; train_feat6 = d['train_feat'].reshape(-1, N_CHAIN, 3)
train_alpha = d['train_alpha']; train_beta = d['train_beta']
test_w = d['test_w']; test_feat6 = d['test_feat'].reshape(-1, N_CHAIN, 3)
test_alpha = d['test_alpha']; test_beta = d['test_beta']; test_amp = d['test_amp']
print(f"Loaded cached dataset: {len(train_w)} train, {len(test_w)} test points", flush=True)

inp = s6.load_inputs()
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]
Phi_chain = Phi_all[CHAIN]
K0_arr = np.array([inp['K_sec'][m, m] for m in CHAIN])
M_arr = np.array([inp['M_sec'][m, m] for m in CHAIN])
C_arr = np.array([inp['C_sec'][m, m] for m in CHAIN])


def detune_k(w, shift_k, zeta_k):
    return np.tanh(((w - np.sqrt(1.0 + shift_k)) / zeta_k) / 20.0)


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


HIDDEN = [64, 64]
EPOCHS = 10000
LR = 1e-3
KL_BETA = 0.001

# k=0: boundary, no left neighbor. k=4: boundary, left neighbor only (mode 3).
# k=1,2,3: interior, BOTH neighbors.
NEIGHBORS = {0: [], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}

models = {}
norms = {}
t_start = time.time()

for k in range(N_CHAIN):
    m = CHAIN[k]
    nbrs = NEIGHBORS[k]
    print(f"\n=== Training sub-network for mode {m} (chain pos {k}, neighbors={nbrs}) ===", flush=True)
    shift_k = train_feat6[:, k, 0]; zeta_k = train_feat6[:, k, 1]; kappa_k = train_feat6[:, k, 2]
    detune = detune_k(train_w, shift_k, zeta_k)
    feat_own = np.column_stack([shift_k, zeta_k, kappa_k, detune])
    parts = [feat_own]
    for nb in nbrs:
        parts.append(train_alpha[:, nb:nb + 1])
        parts.append(train_beta[:, nb:nb + 1])
    X_raw = np.column_stack(parts) if len(parts) > 1 else feat_own

    X_mean, X_std = X_raw.mean(0), X_raw.std(0)
    Xn = (X_raw - X_mean) / X_std
    W_t = torch.tensor(train_w, dtype=torch.float32)
    X_in = torch.cat([s6.fourier_encode_w(W_t), torch.tensor(Xn, dtype=torch.float32)], dim=1)

    y_alpha = train_alpha[:, k]; y_beta = train_beta[:, k]
    Alpha_mean, Alpha_std = y_alpha.mean(), y_alpha.std()
    Beta_mean, Beta_std = y_beta.mean(), y_beta.std()
    Y_alpha_t = torch.tensor((y_alpha - Alpha_mean) / Alpha_std, dtype=torch.float32)
    Y_beta_t = torch.tensor((y_beta - Beta_mean) / Beta_std, dtype=torch.float32)

    torch.manual_seed(42 + k)
    model = s6.BPINN(X_in.shape[1], HIDDEN, 2, prior_sigma=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)
    n_data = float(len(train_w))

    for epoch in range(EPOCHS):
        opt.zero_grad()
        pred = model(X_in)
        alpha_p, beta_p = pred[:, 0], pred[:, 1]
        data_loss = ((alpha_p - Y_alpha_t) ** 2 + (beta_p - Y_beta_t) ** 2).mean()
        kl = model.total_kl() / n_data
        anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
        loss = data_loss + anneal * KL_BETA * kl
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 2500 == 0:
            print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  kl={kl.item():.5f}", flush=True)

    models[k] = model
    norms[k] = dict(X_mean=X_mean, X_std=X_std, Alpha_mean=Alpha_mean, Alpha_std=Alpha_std,
                     Beta_mean=Beta_mean, Beta_std=Beta_std, neighbors=nbrs)
    print(f"  mode {m} sub-network trained, elapsed={time.time()-t_start:.0f}s", flush=True)


def predict_mode(k, w, feat_own, alpha_est, beta_est, n_mc=20):
    """alpha_est/beta_est: (N_CHAIN, n_samples) current best estimates for
    ALL modes (used to pull whichever neighbors this mode needs)."""
    nrm = norms[k]
    nbrs = nrm['neighbors']
    parts = [feat_own]
    for nb in nbrs:
        parts.append(alpha_est[nb][:, None])
        parts.append(beta_est[nb][:, None])
    X_raw = np.column_stack(parts) if len(parts) > 1 else feat_own
    Xn = (X_raw - nrm['X_mean']) / nrm['X_std']
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w, dtype=torch.float32)),
                       torch.tensor(Xn, dtype=torch.float32)], dim=1)
    models[k].eval()
    with torch.no_grad():
        preds = np.array([models[k](X_in).numpy() for _ in range(n_mc)])
    a = preds[:, :, 0].mean(0) * nrm['Alpha_std'] + nrm['Alpha_mean']
    b = preds[:, :, 1].mean(0) * nrm['Beta_std'] + nrm['Beta_mean']
    return a, b


def gauss_seidel_predict(w, feat6, n_sweeps=3):
    """feat6: (n_samples, N_CHAIN, 3) = [shift,zeta,kappa] per mode.
    Returns (alpha, beta): each (N_CHAIN, n_samples)."""
    n = feat6.shape[0]
    feat_own_list = []
    for k in range(N_CHAIN):
        detune = detune_k(w, feat6[:, k, 0], feat6[:, k, 1])
        feat_own_list.append(np.column_stack([feat6[:, k, 0], feat6[:, k, 1], feat6[:, k, 2], detune]))

    alpha_est = [np.zeros(n) for _ in range(N_CHAIN)]
    beta_est = [np.zeros(n) for _ in range(N_CHAIN)]
    # sweep 0: pure left-to-right cascade for an initial guess (right
    # neighbor not yet available -> use 0 for it on this first pass only)
    for k in range(N_CHAIN):
        a, b = predict_mode(k, w, feat_own_list[k], alpha_est, beta_est)
        alpha_est[k], beta_est[k] = a, b
    # refinement sweeps: now every mode's current estimate is available for
    # its neighbors, both left and right
    for sweep in range(n_sweeps):
        for k in range(N_CHAIN):
            a, b = predict_mode(k, w, feat_own_list[k], alpha_est, beta_est)
            alpha_est[k], beta_est[k] = a, b
    return np.array(alpha_est), np.array(beta_est)


print(f"\n{'='*70}\nVALIDATION (Gauss-Seidel cascade, 3 refinement sweeps)\n{'='*70}", flush=True)
alpha_pred, beta_pred = gauss_seidel_predict(test_w, test_feat6, n_sweeps=3)
amp_pred = np.hypot(alpha_pred, beta_pred).T   # (n_test, N_CHAIN)
r2_gs = [r2(test_amp[:, k], amp_pred[:, k]) for k in range(N_CHAIN)]
print(f"R2_amp (Gauss-Seidel, 3 sweeps): {[round(x,3) for x in r2_gs]}  mean={np.mean(r2_gs):.4f}", flush=True)

# convergence check: does more sweeps change anything?
alpha_pred1, beta_pred1 = gauss_seidel_predict(test_w, test_feat6, n_sweeps=1)
amp_pred1 = np.hypot(alpha_pred1, beta_pred1).T
r2_gs1 = [r2(test_amp[:, k], amp_pred1[:, k]) for k in range(N_CHAIN)]
print(f"R2_amp (Gauss-Seidel, 1 sweep, convergence check): {[round(x,3) for x in r2_gs1]}  mean={np.mean(r2_gs1):.4f}", flush=True)

# ---- THE REAL CHECK: w=1.0, zero mistuning, vs real ANSYS ----
feat6_check = np.zeros((1, N_CHAIN, 3))
for k in range(N_CHAIN):
    m = CHAIN[k]
    zeta_k = C_arr[k] / (2 * math.sqrt(K0_arr[k] * M_arr[k]))
    kappa_k = 0.75 * inp['K3_sec_diag'][m] / K0_arr[k]
    feat6_check[0, k] = [0.0, zeta_k, kappa_k]
alpha_c, beta_c = gauss_seidel_predict(np.array([1.0]), feat6_check, n_sweeps=3)
alpha_c = alpha_c[:, 0]; beta_c = beta_c[:, 0]
u_complex = np.sum((alpha_c - 1j * beta_c) * Phi_chain)
bpinn_amp = float(abs(u_complex))
print(f"\n{'='*70}")
print(f"BIDIRECTIONAL GAUSS-SEIDEL BPINN vs REAL ANSYS: node 1171 UZ, w=1.0, tuned baseline")
print(f"{'='*70}")
print(f"  BPINN (bidirectional, converged): {bpinn_amp:.4f} mm")
print(f"  Real ANSYS (converged): 1.2220 +/- 0.0190 mm")
print(f"  Ratio (real/BPINN): {1.2220/bpinn_amp:.4f}x")
print(f"{'='*70}")

for k in range(N_CHAIN):
    torch.save(models[k].state_dict(), os.path.join(OUT, f'bpinn_bidir_mode{CHAIN[k]}_state.pt'))
np.savez(os.path.join(OUT, 'bpinn_bidir_bridged_norms.npz'),
         chain=np.array(CHAIN), r2_gs=np.array(r2_gs), r2_gs1sweep=np.array(r2_gs1),
         bpinn_amp=bpinn_amp, real_ansys_amp=1.2220, real_ansys_std=0.0190)
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
