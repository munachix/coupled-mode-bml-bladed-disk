"""
Amplitude-only fallback for pair (5,6) (2026-08-13). Three separate fixes
were tried to recover this pair's amplitude R^2 after the phase-resolved
architecture upgrade -- bigger network+more data, an explicit amplitude-
consistency loss term, and disabling the physics loss entirely -- all
plateaued at R^2~0.83-0.86/0.70-0.73, confirmed NOT physics-loss-driven
(near-identical with PHYSICS_WEIGHT=0). Root cause, measured directly: this
pair's quadrature (beta) component is extremely concentrated (99.7-99.9%
of its signal sits in the top 10% of points by magnitude -- close to a
delta function), a fundamentally harder regression target than every
other pair (72-96% concentration), independent of loss/capacity choices.

Rather than keep forcing a physics-informed architecture that demonstrably
doesn't fit this ONE pair's response character, this reverts pair (5,6) to
the ORIGINAL amplitude-only design (2 outputs, data loss only) that gave
it R^2=0.99/0.988 before tonight's phase-resolved upgrade -- reusing the
SAME cached dataset (amp_i/amp_j ground truth already computed from the
real coupled ODE solver), just predicting amplitude directly instead of
via phase. Disclosed, not hidden: this pair does NOT get the physics-
residual loss (4 of 5 pairs + the chain do); its own beta signal is simply
too information-poor to support it.
"""
import sys, os, time
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
import step6 as s6

MODE_I, MODE_J = 5, 6
TAG = f'{MODE_I}{MODE_J}'
OUT = s6.OUT

t0 = time.time()
ckpt_path = os.path.join(OUT, f'_coupled_dataset_checkpoint_{TAG}.npz')
d = np.load(ckpt_path)
print(f"Loaded cached dataset: {ckpt_path}", flush=True)

train_w = d['train_w']; train_feat = d['train_feat']
train_amp_i = np.hypot(d['train_alpha_i'], d['train_beta_i'])
train_amp_j = np.hypot(d['train_alpha_j'], d['train_beta_j'])
test_w = d['test_w']; test_feat = d['test_feat']
test_amp_i = d['test_amp_i']; test_amp_j = d['test_amp_j']
print(f"{len(train_w)} train / {len(test_w)} test points", flush=True)

W_t = torch.tensor(train_w, dtype=torch.float32)
Feat_t = torch.tensor(train_feat, dtype=torch.float32)
Amp_i_raw = train_amp_i; Amp_j_raw = train_amp_j
Ai_mean, Ai_std = Amp_i_raw.mean(), Amp_i_raw.std()
Aj_mean, Aj_std = Amp_j_raw.mean(), Amp_j_raw.std()
Amp_i_t = torch.tensor((Amp_i_raw - Ai_mean) / Ai_std, dtype=torch.float32)
Amp_j_t = torch.tensor((Amp_j_raw - Aj_mean) / Aj_std, dtype=torch.float32)

Feat_mean, Feat_std = Feat_t.mean(0), Feat_t.std(0)
Feat_n = (Feat_t - Feat_mean) / Feat_std
X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)

torch.manual_seed(42)
model = s6.BPINN(X_in.shape[1], [32, 32], 2, prior_sigma=1.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 6000
KL_BETA = 0.001
n_data = float(len(train_w))
print("Training (amplitude-only)...", flush=True)
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
    if epoch % 2000 == 0:
        print(f"  epoch {epoch:5d}  data={data_loss.item():.6f}  kl={kl.item():.5f}", flush=True)

W_test = torch.tensor(test_w, dtype=torch.float32)
Feat_test = torch.tensor(test_feat, dtype=torch.float32)
Feat_test_n = (Feat_test - Feat_mean) / Feat_std
X_test = torch.cat([s6.fourier_encode_w(W_test), Feat_test_n], dim=1)

model.eval()
n_mc = 30
preds = []
with torch.no_grad():
    for _ in range(n_mc):
        preds.append(model(X_test).numpy())
preds = np.array(preds)
amp_i_mean = preds[:, :, 0].mean(0) * Ai_std + Ai_mean
amp_j_mean = preds[:, :, 1].mean(0) * Aj_std + Aj_mean


def r2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot


r2_i = r2(test_amp_i, amp_i_mean)
r2_j = r2(test_amp_j, amp_j_mean)
print(f"\nTest R^2 (amplitude-only): mode {MODE_I}={r2_i:.4f}, mode {MODE_J}={r2_j:.4f}", flush=True)

fp_model = os.path.join(OUT, f'bpinn_ampOnly_state_{TAG}.pt')
torch.save(model.state_dict(), fp_model)
fp_norm = os.path.join(OUT, f'bpinn_ampOnly_norm_{TAG}.npz')
np.savez(fp_norm, feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
         amp_i_mean=Ai_mean, amp_i_std=Ai_std, amp_j_mean=Aj_mean, amp_j_std=Aj_std,
         mode_i=MODE_I, mode_j=MODE_J, r2_i=r2_i, r2_j=r2_j, architecture='amplitude_only')
print(f"Saved: {fp_model}\nSaved: {fp_norm}", flush=True)
print(f"TOTAL TIME: {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
