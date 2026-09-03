"""Retrain-only pass, loading the CHECKPOINTED dataset from the first run
(no need to redo the ~742s of coupled-ODE data generation). Fixes a real
bug found in that first attempt: output targets (amp_i, amp_j) were never
normalized, and amp_j's scale is ~8x smaller than amp_i's -- the MSE loss
was dominated by amp_i, leaving amp_j barely fit (R^2=0.05). Normalizing
both outputs to zero-mean/unit-std before computing the loss (matching
what was already done for inputs) fixes the incentive imbalance."""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
import step6 as s6

OUT = s6.OUT
ckpt = np.load(os.path.join(OUT, '_coupled_dataset_checkpoint.npz'))

W_t = torch.tensor(ckpt['train_w'], dtype=torch.float32)
Feat_t = torch.tensor(ckpt['train_feat'], dtype=torch.float32)
Amp_i_raw = torch.tensor(ckpt['train_amp_i'], dtype=torch.float32)
Amp_j_raw = torch.tensor(ckpt['train_amp_j'], dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)

# NEW: normalize output targets too
Ai_mean, Ai_std = Amp_i_raw.mean(), Amp_i_raw.std()
Aj_mean, Aj_std = Amp_j_raw.mean(), Amp_j_raw.std()
Amp_i_n = (Amp_i_raw - Ai_mean) / Ai_std
Amp_j_n = (Amp_j_raw - Aj_mean) / Aj_std

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [32, 32], 2, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 8000
KL_BETA = 0.001
n_data = float(len(W_t))
t0 = time.time()
print("Retraining with normalized outputs...", flush=True)
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(X_in)
    amp_i_p, amp_j_p = pred[:, 0], pred[:, 1]
    data_loss = ((amp_i_p - Amp_i_n) ** 2 + (amp_j_p - Amp_j_n) ** 2).mean()
    kl = model.total_kl() / n_data
    anneal = min(1.0, epoch / max(1, EPOCHS * 0.3))
    loss = data_loss + anneal * KL_BETA * kl
    loss.backward()
    opt.step()
    if epoch % 1000 == 0:
        print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  kl={kl.item():.5f}", flush=True)

# ---- Validate on held-out test set ----
W_test = torch.tensor(ckpt['test_w'], dtype=torch.float32)
Feat_test = torch.tensor(ckpt['test_feat'], dtype=torch.float32)
Amp_i_test = ckpt['test_amp_i']
Amp_j_test = ckpt['test_amp_j']
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)

model.eval()
n_mc = 30
preds = []
with torch.no_grad():
    for _ in range(n_mc):
        preds.append(model(X_test).numpy())
preds = np.array(preds)
amp_i_pred = preds[:, :, 0].mean(0) * Ai_std.item() + Ai_mean.item()
amp_j_pred = preds[:, :, 1].mean(0) * Aj_std.item() + Aj_mean.item()
amp_i_std_pred = preds[:, :, 0].std(0) * Ai_std.item()
amp_j_std_pred = preds[:, :, 1].std(0) * Aj_std.item()

def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot

r2_i = r2(Amp_i_test, amp_i_pred)
r2_j = r2(Amp_j_test, amp_j_pred)
print(f"\nTest R^2: mode 0 amp = {r2_i:.4f}, mode 1 amp = {r2_j:.4f}", flush=True)
print(f"RMSE: mode 0 = {np.sqrt(np.mean((Amp_i_test-amp_i_pred)**2)):.5f}, "
      f"mode 1 = {np.sqrt(np.mean((Amp_j_test-amp_j_pred)**2)):.5f}", flush=True)

fp_model = os.path.join(OUT, 'bpinn_coupled_state_01.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, 'bpinn_coupled_norm_01.npz')
np.savez(fp_norm, feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
         amp_i_mean=Ai_mean.item(), amp_i_std=Ai_std.item(),
         amp_j_mean=Aj_mean.item(), amp_j_std=Aj_std.item(),
         phi_i_1171=28.3852, phi_j_1171=-1.7883, f_phys=1000.0,
         mode_i=0, mode_j=1, r2_i=r2_i, r2_j=r2_j)
print(f"Saved: {fp_model}", flush=True)
print(f"Saved: {fp_norm}", flush=True)
print(f"TOTAL TIME: {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
