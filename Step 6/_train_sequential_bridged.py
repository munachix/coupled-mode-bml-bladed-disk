"""Sequential chain-conditioned BPINN for the bridged 0-1-2-3-4 system
(2026-08-23) -- the joint 10-output network plateaued at mean R^2~0.53
(mode 1 stuck at 0.09-0.24) across 4 capacity/epoch configs, and a learning-
curve smoke test showed more training data does NOT help (R^2 flat/
declining with more data: 0.173->0.134->0.115 for mode 1). Root cause: mode
1 gets almost no direct force (Fg_1=-157N vs 1165-2500N for the other 4
modes) -- its response is almost entirely INDIRECT, through coupling with
mode 0. A single joint network has to implicitly "simulate" mode 0's actual
response internally to predict mode 1's coupling-driven force; that's a
much harder implicit computation than the pair BPINNs' proven, direct
2-mode joint prediction (which already achieves R^2 0.83-0.99).

Fix: decompose into 5 SEQUENTIAL sub-networks along the real chain
topology (0->1->2->3->4, matching the real measured adjacent coupling).
Each sub-network k>0 receives the ACTUAL VALUE of its left neighbor's
response (alpha_{k-1}, beta_{k-1}) as an explicit input, not just static
mistuning features -- directly giving the network the state information it
otherwise has to re-derive. Trained with teacher forcing (TRUE neighbor
values from the cached ODE dataset); at inference time run as a genuine
forward cascade using each net's own prediction for the next.
"""
import sys, os, time, math
import numpy as np
import torch
import torch.nn as nn
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
train_w = d['train_w']; train_feat6 = d['train_feat'].reshape(-1, N_CHAIN, 3); train_K = d['train_K_arr']
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


class SmallBPINN(nn.Module):
    """Same Bayesian-linear architecture as s6.BPINN, just instantiated
    per chain step with its own (smaller) input dimension."""
    def __init__(self, in_dim, hidden, out_dim, prior_sigma=1.0):
        super().__init__()
        self.net = s6.BPINN(in_dim, hidden, out_dim, prior_sigma=prior_sigma)

    def forward(self, x):
        return self.net(x)

    def total_kl(self):
        return self.net.total_kl()


HIDDEN = [64, 64]
EPOCHS = 10000
LR = 1e-3
KL_BETA = 0.001

models = []
norms = []
t_start = time.time()

for k in range(N_CHAIN):
    m = CHAIN[k]
    print(f"\n=== Training sub-network for mode {m} (chain position {k}) ===", flush=True)
    shift_k = train_feat6[:, k, 0]; zeta_k = train_feat6[:, k, 1]; kappa_k = train_feat6[:, k, 2]
    detune = detune_k(train_w, shift_k, zeta_k)
    feat_own = np.column_stack([shift_k, zeta_k, kappa_k, detune])   # (n,4)

    if k == 0:
        X_raw = feat_own
    else:
        # neighbor = chain position k-1's TRUE alpha/beta (teacher forcing)
        neighbor_alpha = train_alpha[:, k - 1]
        neighbor_beta = train_beta[:, k - 1]
        X_raw = np.column_stack([feat_own, neighbor_alpha, neighbor_beta])

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

    models.append(model)
    norms.append(dict(X_mean=X_mean, X_std=X_std, Alpha_mean=Alpha_mean, Alpha_std=Alpha_std,
                       Beta_mean=Beta_mean, Beta_std=Beta_std))
    print(f"  mode {m} sub-network trained, elapsed={time.time()-t_start:.0f}s", flush=True)

# ---- Validate: TWO ways --------------------------------------------------
# (A) Teacher-forced (each net given the TRUE neighbor value) -- isolates
#     whether each sub-network itself learned its own mapping well.
# (B) Real cascade (each net fed the PREVIOUS net's OWN prediction) -- the
#     actual inference-time behavior, where errors can compound down the
#     chain. This is the number that matters for real use.
print(f"\n{'='*70}\nVALIDATION\n{'='*70}", flush=True)

alpha_pred_tf = np.zeros((len(test_w), N_CHAIN))
beta_pred_tf = np.zeros((len(test_w), N_CHAIN))
alpha_pred_cascade = np.zeros((len(test_w), N_CHAIN))
beta_pred_cascade = np.zeros((len(test_w), N_CHAIN))

for k in range(N_CHAIN):
    m = CHAIN[k]
    model = models[k]; nrm = norms[k]
    shift_k = test_feat6[:, k, 0]; zeta_k = test_feat6[:, k, 1]; kappa_k = test_feat6[:, k, 2]
    detune = detune_k(test_w, shift_k, zeta_k)
    feat_own = np.column_stack([shift_k, zeta_k, kappa_k, detune])

    # (A) teacher-forced
    if k == 0:
        X_raw_tf = feat_own
    else:
        X_raw_tf = np.column_stack([feat_own, test_alpha[:, k - 1], test_beta[:, k - 1]])
    Xn_tf = (X_raw_tf - nrm['X_mean']) / nrm['X_std']
    X_in_tf = torch.cat([s6.fourier_encode_w(torch.tensor(test_w, dtype=torch.float32)),
                          torch.tensor(Xn_tf, dtype=torch.float32)], dim=1)
    model.eval()
    with torch.no_grad():
        preds = np.array([model(X_in_tf).numpy() for _ in range(20)])
    alpha_pred_tf[:, k] = preds[:, :, 0].mean(0) * nrm['Alpha_std'] + nrm['Alpha_mean']
    beta_pred_tf[:, k] = preds[:, :, 1].mean(0) * nrm['Beta_std'] + nrm['Beta_mean']

    # (B) real cascade
    if k == 0:
        X_raw_c = feat_own
    else:
        X_raw_c = np.column_stack([feat_own, alpha_pred_cascade[:, k - 1], beta_pred_cascade[:, k - 1]])
    Xn_c = (X_raw_c - nrm['X_mean']) / nrm['X_std']
    X_in_c = torch.cat([s6.fourier_encode_w(torch.tensor(test_w, dtype=torch.float32)),
                         torch.tensor(Xn_c, dtype=torch.float32)], dim=1)
    with torch.no_grad():
        preds_c = np.array([model(X_in_c).numpy() for _ in range(20)])
    alpha_pred_cascade[:, k] = preds_c[:, :, 0].mean(0) * nrm['Alpha_std'] + nrm['Alpha_mean']
    beta_pred_cascade[:, k] = preds_c[:, :, 1].mean(0) * nrm['Beta_std'] + nrm['Beta_mean']

amp_pred_tf = np.hypot(alpha_pred_tf, beta_pred_tf)
amp_pred_cascade = np.hypot(alpha_pred_cascade, beta_pred_cascade)
r2_tf = [r2(test_amp[:, k], amp_pred_tf[:, k]) for k in range(N_CHAIN)]
r2_cascade = [r2(test_amp[:, k], amp_pred_cascade[:, k]) for k in range(N_CHAIN)]

print(f"R2_amp (A: teacher-forced, isolates each sub-net): {[round(x,3) for x in r2_tf]}  mean={np.mean(r2_tf):.4f}", flush=True)
print(f"R2_amp (B: real cascade, actual inference behavior): {[round(x,3) for x in r2_cascade]}  mean={np.mean(r2_cascade):.4f}", flush=True)

# ---- THE REAL CHECK: cascade at w=1.0, zero mistuning, vs real ANSYS ----
feat0_own = np.zeros((1, 4))
w_check = np.array([1.0])
alpha_c0 = np.zeros(N_CHAIN); beta_c0 = np.zeros(N_CHAIN)
for k in range(N_CHAIN):
    m = CHAIN[k]
    zeta_k = C_arr[k] / (2 * math.sqrt(K0_arr[k] * M_arr[k]))
    kappa_k = 0.75 * inp['K3_sec_diag'][m] / K0_arr[k]
    detune0 = detune_k(w_check, np.array([0.0]), np.array([zeta_k]))
    feat_own0 = np.array([[0.0, zeta_k, kappa_k, detune0[0]]])
    if k == 0:
        X_raw0 = feat_own0
    else:
        X_raw0 = np.column_stack([feat_own0, [alpha_c0[k - 1]], [beta_c0[k - 1]]])
    nrm = norms[k]
    Xn0 = (X_raw0 - nrm['X_mean']) / nrm['X_std']
    X_in0 = torch.cat([s6.fourier_encode_w(torch.tensor(w_check, dtype=torch.float32)),
                        torch.tensor(Xn0, dtype=torch.float32)], dim=1)
    with torch.no_grad():
        preds0 = np.array([models[k](X_in0).numpy() for _ in range(100)])
    alpha_c0[k] = float(preds0[:, 0, 0].mean()) * nrm['Alpha_std'] + nrm['Alpha_mean']
    beta_c0[k] = float(preds0[:, 0, 1].mean()) * nrm['Beta_std'] + nrm['Beta_mean']

u_complex = np.sum((alpha_c0 - 1j * beta_c0) * Phi_chain)
bpinn_amp = float(abs(u_complex))
print(f"\n{'='*70}")
print(f"SEQUENTIAL CHAIN BPINN vs REAL ANSYS: node 1171 UZ, w=1.0, tuned baseline")
print(f"{'='*70}")
print(f"  BPINN (sequential chain cascade): {bpinn_amp:.4f} mm")
print(f"  Real ANSYS (converged): 1.2220 +/- 0.0190 mm")
print(f"  Ratio (real/BPINN): {1.2220/bpinn_amp:.4f}x")
print(f"{'='*70}")

for k in range(N_CHAIN):
    torch.save(models[k].state_dict(), os.path.join(OUT, f'bpinn_seq_mode{CHAIN[k]}_state.pt'))
np.savez(os.path.join(OUT, 'bpinn_seq_bridged_norms.npz'),
         chain=np.array(CHAIN), r2_tf=np.array(r2_tf), r2_cascade=np.array(r2_cascade),
         bpinn_amp=bpinn_amp, real_ansys_amp=1.2220, real_ansys_std=0.0190)
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
