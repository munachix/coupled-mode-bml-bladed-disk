"""FORCING-AWARE BPINN for pair (0,1) (2026-08-23) -- the real architectural
fix identified this session: every BPINN trained so far has forcing level
BAKED IN (one fixed target_peak per trained network), so no existing
network can answer both a near-linear question (Case 1/2's natural
frequency) and a real-forcing nonlinear question (Case 3's amplitude) --
confirmed directly: sweeping the production (0,1) network (trained at
target_peak=0.8) found its resonance peak still climbing at w=1.10, because
at that substantial forcing level this strongly-hardening mode's resonance
has shifted well past the linear frequency. Comparing that to Case 1's
LINEAR 292.818 Hz was never going to work -- different physical regimes.

Fix: add forcing level (target_peak) as an EXPLICIT network input, and
generate training data across MULTIPLE forcing levels (not one), so a
SINGLE trained network spans the whole range: near-zero forcing recovers
the linear regime (Case 1/2), Case 3's own real forcing level is included
directly (target_peak=0.184640, computed from F_gen=2500 this session),
and everything in between/beyond.

Uses the SAME proven recipe already validated this session for pair (0,1):
densified w-grid, explicit saturated detuning feature, tuned physics
weight -- now with target_peak sampled per training row instead of fixed.

--- 2026-08-28 CONVFIX VARIANT ---
Post-t_ref-fix retraining left pair (0,1) at R^2=0.7248/0.7308, the weakest
of the 5 real coupled pairs (others 0.90-0.97). The amplitude-consistency
loss (amp_loss below, hypot(alpha,beta) penalized directly against true
amplitude) that fixed pair (7,8) previously is ALREADY present in this
script (both pairs are trained by the same generic file) -- so it's not a
missing ingredient here, confirmed by inspection, not assumed.

Root-caused instead by direct inspection of the (0,1) checkpoint
(_forcing_aware_01_checkpoint.npz): the test-set R^2 collapse is
concentrated almost entirely at target_peak=0.4 (R^2=0.4276 vs 0.90+ at
the neighboring levels 0.1846 and 0.8 -- no other pair shows a mid-range
collapse like this, only the well-documented low-forcing weakness at the
bottom of the range). Tracing one specific test trajectory found a single
point at w=1.014 with amp_i=0.345 sandwiched between smooth values of
~0.02-0.03 on one side and ~0.04 on the other, with SEVERAL adjacent
w-values on both sides silently rejected (diverged) -- i.e. exactly the
signature of a hardening-Duffing jump/bifurcation region where the
time-domain integration (solve_ivp over N_CYCLES=200 cycles, amplitude
read from the LAST 10% of the run) sometimes hasn't settled into either
stable branch yet within the simulated window, producing a transient
peak-to-peak reading far above any real steady-state amplitude. Quantified:
this ONE contaminated test point accounts for 58.7% of the total variance
(SS_tot) at target_peak=0.4 in the current test set -- a single bad label
dominating an R^2 computed over only 424 points. This is a data-quality/
settling-time artifact, not a genuine information ceiling (unlike pair
(5,6)'s confirmed real limit).

Fix tried here: a stationarity/convergence check computed from the ALREADY
RETURNED full time history (r['t'], r['q_i'], r['q_j']) -- zero extra
solves. Split the tail into two consecutive 10%-of-run windows and compare
their peak-to-peak amplitudes; reject the point (same bucket as the
existing DIVERGE_BOUND rejection) if either mode's amplitude estimate
moved by more than CONV_TOL between the two windows, i.e. it had not
reached a stationary periodic state by the end of the simulated window.
This is purely a training/test DATA quality gate -- no change to the loss
function (amp_loss stays as already implemented) and no change to the
physics model.
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
# Case 1's real-ANSYS-validated 292.818 Hz is specific to mode 0 (Step 2's
# own validated tuned frequency, cross-checked against real ANSYS). For
# pairs other than (0,1), there is no separate real-ANSYS Case-1 number to
# check against -- the low-forcing check below instead validates against
# THIS pair's own ROM-predicted tuned frequency (freqs_sec[MODE_I]), a real
# self-consistency check (does forcing-awareness correctly recover the
# known-correct linear limit), not a fresh real-ANSYS validation. Reported
# as such, not conflated with the mode-0-specific real measurement.
CASE3_TARGET_PEAK = 0.184640   # real Case 3 forcing (mode-0/node-1171 specific); reused as a generic mid-level anchor for other pairs
FORCE_LEVELS = [0.02, 0.05, CASE3_TARGET_PEAK, 0.4, 0.8]   # near-linear -> mid-level -> strongly nonlinear
N_TRAIN_PER_LEVEL = int(os.environ.get('N_TRAIN_PER_LEVEL', 40))
N_TEST_PER_LEVEL = int(os.environ.get('N_TEST_PER_LEVEL', 8))
# 2026-08-23 FIX: near-zero forcing showed weak amplitude R^2 (0.38-0.89
# across pairs) -- root-caused, not assumed: true amplitude at
# target_peak=0.02 spans ~1000x within the w-sweep (std > mean), because at
# near-zero forcing the resonance is genuinely razor-sharp (linear regime,
# Q~250 from zeta=0.002, true half-power bandwidth ~0.004 in w-units) --
# the 0.03 grid spacing was ~7x too coarse to resolve it. This is a REAL
# resolution problem (confirmed via the true-amplitude variance, not an
# R^2-metric artifact from small target variance -- SS_tot here is large,
# not small). Fix: extra-dense band right at w=1 (step 0.003, matching the
# real half-power bandwidth scale) layered on top of the existing wide
# grid (which stays needed for the higher-forcing hardened/shifted
# resonance).
#
# WIDENED 2026-08-23 (round 2): the first attempt used a NARROW dense band
# (0.985-1.015, 11 points) centered exactly on the pure-linear resonance --
# this fixed target_peak=0.02 and the Case 1 check (error 0.70%->0.0001%)
# but broke target_peak=0.05 (R^2 0.90->0.26), because even mild forcing
# shifts this strongly-hardening mode's resonance measurably off w=1.0, so
# the narrow band no longer covered where 0.05's actual peak sits, while
# the OLD coarser grid had coincidentally spread more broadly around it.
# Fix: widen the dense band to cover both the exact-linear point AND the
# mild-forcing-shifted region (0.97-1.10), same fine step, so no force
# level in the trained range is left with a poorly-centered sample cluster.
W_GRID = np.unique(np.concatenate([
    np.arange(0.970, 1.101, 0.004),          # extra-dense: covers linear AND mild-forcing-shifted resonance
    np.arange(0.95, 1.61, 0.03),              # original wide coverage for hardened/shifted resonance
    [1.7, 2.0, 2.3, 2.6],
]))
N_CYCLES = 200
STEPS_PER_CYCLE = 20
DIVERGE_BOUND = 0.5
PHYSICS_WEIGHT = 0.05
# 2026-08-28: stationarity gate -- reject points whose peak-to-peak
# amplitude is still changing by more than this fraction between the last
# two 10%-of-run windows (see module docstring: root-caused via a 0.345
# transient spike at w=1.014, tp=0.4, dominating 58.7% of that level's
# test-set SS_tot). Chosen generously (real jump artifacts found were
# >5x, not borderline) so genuine steady periodic points are not
# over-rejected.
CONV_TOL = 0.15


def is_converged(t_arr, q_t, tol=CONV_TOL):
    n = len(t_arr)
    tail = max(int(0.1 * n), 10)
    if n < 3 * tail:
        return True  # too short a run to check meaningfully; don't over-reject
    last = q_t[-tail:]
    prev = q_t[-2 * tail:-tail]
    amp_last = (last.max() - last.min()) / 2
    amp_prev = (prev.max() - prev.min()) / 2
    denom = max(amp_last, amp_prev, 1e-10)
    return abs(amp_last - amp_prev) / denom < tol

t_start = time.time()
print(f"=== Forcing-aware BPINN, pair ({MODE_I},{MODE_J}) ===", flush=True)
print(f"  Force levels (target_peak): {FORCE_LEVELS}", flush=True)
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


def build_dataset(sample_indices, force_levels):
    rows = []
    n_rejected = 0
    n_nonconverged = 0
    n_total = 0
    for k, i in enumerate(sample_indices):
        p = sample_features(i)
        for target_peak in force_levels:
            Fg_i = target_peak * 2 * zeta_i0 * Ki0
            Fg_j = target_peak * 2 * zeta_j0 * Kj0
            for w in W_GRID:
                n_total += 1
                Omega = w * omega0_i
                r = s4.duffing_forced_response_coupled(
                    (MODE_I, MODE_J), (p['K_i'], p['K_j']), (M_i, M_j), (C_i, C_j),
                    cc['coef0'], cc['coef1'], (Fg_i, Fg_j), Omega,
                    n_cycles=N_CYCLES, steps_per_cycle=STEPS_PER_CYCLE)
                ok = (np.isfinite(r['amp_i']) and np.isfinite(r['amp_j'])
                      and abs(r['amp_i']) < DIVERGE_BOUND and abs(r['amp_j']) < DIVERGE_BOUND)
                if ok:
                    conv_ok = (is_converged(r['t'], r['q_i']) and is_converged(r['t'], r['q_j']))
                    if not conv_ok:
                        ok = False
                        n_nonconverged += 1
                if not ok:
                    n_rejected += 1
                    continue
                feat8 = s6.add_detune_features(w, p['feat'])
                feat_aug = np.concatenate([feat8, [target_peak]])   # forcing level as explicit input
                rows.append(dict(w=w, feat=feat_aug, K_i=p['K_i'], K_j=p['K_j'], target_peak=target_peak,
                                  alpha_i=r['alpha_i'], beta_i=r['beta_i'],
                                  alpha_j=r['alpha_j'], beta_j=r['beta_j'],
                                  amp_i=r['amp_i'], amp_j=r['amp_j']))
        if k % 10 == 0:
            print(f"  sample {k}/{len(sample_indices)}, elapsed={time.time()-t_start:.0f}s, "
                  f"rejected so far={n_rejected}/{n_total} (of which non-converged={n_nonconverged})", flush=True)
    print(f"  Total rejected: {n_rejected} of {n_total} (of which non-converged/jump-transient={n_nonconverged})", flush=True)
    return rows


rng = np.random.default_rng(42)
perm = rng.permutation(inp['n_samples'])
train_idx = perm[:N_TRAIN_PER_LEVEL]
test_idx = perm[N_TRAIN_PER_LEVEL:N_TRAIN_PER_LEVEL + N_TEST_PER_LEVEL]

print(f"\nGenerating training data ({N_TRAIN_PER_LEVEL} samples x {len(FORCE_LEVELS)} force levels x "
      f"{len(W_GRID)} w-points = {N_TRAIN_PER_LEVEL*len(FORCE_LEVELS)*len(W_GRID)} solves)...", flush=True)
train_rows = build_dataset(train_idx, FORCE_LEVELS)
print(f"\nGenerating test data...", flush=True)
test_rows = build_dataset(test_idx, FORCE_LEVELS)
print(f"Data generation done in {time.time()-t_start:.0f}s", flush=True)

TAG = f'{MODE_I}{MODE_J}'
ckpt_path = os.path.join(OUT, f'_forcing_aware_{TAG}_checkpoint.npz')
np.savez(ckpt_path,
          train_w=np.array([r['w'] for r in train_rows]),
          train_feat=np.stack([r['feat'] for r in train_rows]),
          train_target_peak=np.array([r['target_peak'] for r in train_rows]),
          train_alpha_i=np.array([r['alpha_i'] for r in train_rows]), train_beta_i=np.array([r['beta_i'] for r in train_rows]),
          train_alpha_j=np.array([r['alpha_j'] for r in train_rows]), train_beta_j=np.array([r['beta_j'] for r in train_rows]),
          test_w=np.array([r['w'] for r in test_rows]),
          test_feat=np.stack([r['feat'] for r in test_rows]),
          test_target_peak=np.array([r['target_peak'] for r in test_rows]),
          test_alpha_i=np.array([r['alpha_i'] for r in test_rows]), test_beta_i=np.array([r['beta_i'] for r in test_rows]),
          test_alpha_j=np.array([r['alpha_j'] for r in test_rows]), test_beta_j=np.array([r['beta_j'] for r in test_rows]),
          test_amp_i=np.array([r['amp_i'] for r in test_rows]), test_amp_j=np.array([r['amp_j'] for r in test_rows]))
print(f"Checkpointed: {ckpt_path}", flush=True)

if len(train_rows) < 200 or len(test_rows) < 30:
    print("ABORT: too few surviving points", flush=True)
    sys.exit(1)

# ---- Train ----
W_t = torch.tensor([r['w'] for r in train_rows], dtype=torch.float32)
Feat_t = torch.tensor(np.stack([r['feat'] for r in train_rows]), dtype=torch.float32)
Omega_t = W_t * omega0_i
K_i_t = torch.tensor([r['K_i'] for r in train_rows], dtype=torch.float32)
K_j_t = torch.tensor([r['K_j'] for r in train_rows], dtype=torch.float32)
Fg_i_t = torch.tensor([r['target_peak'] * 2 * zeta_i0 * Ki0 for r in train_rows], dtype=torch.float32)
Fg_j_t = torch.tensor([r['target_peak'] * 2 * zeta_j0 * Kj0 for r in train_rows], dtype=torch.float32)

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
F_scale = max(np.abs([r['target_peak'] * 2 * zeta_i0 * Ki0 for r in train_rows]).max(),
              np.abs([r['target_peak'] * 2 * zeta_j0 * Kj0 for r in train_rows]).max())

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
print(f"\nOverall test R^2 (amplitude, all force levels pooled): mode 0={r2_i_overall:.4f}, mode 1={r2_j_overall:.4f}", flush=True)
test_tp = np.array([r['target_peak'] for r in test_rows])
for tp in FORCE_LEVELS:
    mask = np.isclose(test_tp, tp)
    if mask.sum() > 2:
        r2_i_level = r2(Amp_i_test_raw[mask], amp_i_pred[mask])
        print(f"  target_peak={tp:.4f}: R^2 mode0={r2_i_level:.4f}  (n={mask.sum()})", flush=True)

# ---- CASE 1 CHECK: near-zero forcing, sweep w, find resonance peak ----
w_sweep = np.linspace(0.95, 1.10, 151)
feat6_c1 = np.array([0.0, zeta_i0, 0.75 * inp['K3_sec_diag'][MODE_I] / Ki0,
                      0.0, zeta_j0, 0.75 * inp['K3_sec_diag'][MODE_J] / Kj0])
feat_arr_c1 = np.tile(feat6_c1, (len(w_sweep), 1))
feat8_c1 = s6.add_detune_features(w_sweep, feat_arr_c1)
tp_c1 = np.full(len(w_sweep), 0.02)   # near-linear forcing
feat_aug_c1 = np.concatenate([feat8_c1, tp_c1[:, None]], axis=1)
Feat_n_c1 = (torch.tensor(feat_aug_c1, dtype=torch.float32) - Feat_mean) / Feat_std
X_c1 = torch.cat([s6.fourier_encode_w(torch.tensor(w_sweep, dtype=torch.float32)), Feat_n_c1], dim=1)
with torch.no_grad():
    preds_c1 = np.array([model(X_c1).numpy() for _ in range(30)])
amp_i_c1 = np.hypot(preds_c1[:, :, 0].mean(0) * Ai_std + Ai_mean, preds_c1[:, :, 1].mean(0) * Bi_std + Bi_mean)
peak_idx = np.argmax(amp_i_c1)
peak_w = w_sweep[peak_idx]
peak_freq = peak_w * omega0_i / (2 * np.pi)
# Mode 0 has a real-ANSYS-validated Case 1 number (292.818 Hz, Step 2's own
# cross-checked value); every other mode's reference here is the ROM's own
# tuned freqs_sec[MODE_I] instead -- a self-consistency check (does
# forcing-awareness correctly recover the already-known-correct linear
# limit), not a fresh real-ANSYS measurement. Labeled accordingly below.
CASE1_REAL_FREQ_HZ = 292.818 if MODE_I == 0 else float(inp['freqs_sec'][MODE_I])
ref_label = "Real ANSYS Case 1 (tuned)" if MODE_I == 0 else "ROM tuned freqs_sec[MODE_I] (self-consistency ref, not real-ANSYS)"
print(f"\n{'='*70}\nCASE 1 CHECK: forcing-aware BPINN at near-zero forcing (target_peak=0.02)\n{'='*70}")
print(f"  Predicted resonance peak: w={peak_w:.4f}, f={peak_freq:.4f} Hz")
print(f"  {ref_label}: {CASE1_REAL_FREQ_HZ:.4f} Hz")
print(f"  Error: {abs(peak_freq-CASE1_REAL_FREQ_HZ):.4f} Hz ({100*abs(peak_freq-CASE1_REAL_FREQ_HZ)/CASE1_REAL_FREQ_HZ:.4f}%)")

# ---- CASE 3 CHECK: real forcing level, w=1.0, tuned ----
feat_arr_c3 = feat6_c1[None, :]
feat8_c3 = s6.add_detune_features(np.array([1.0]), feat_arr_c3)
feat_aug_c3 = np.concatenate([feat8_c3, [[CASE3_TARGET_PEAK]]], axis=1)
Feat_n_c3 = (torch.tensor(feat_aug_c3, dtype=torch.float32) - Feat_mean) / Feat_std
X_c3 = torch.cat([s6.fourier_encode_w(torch.tensor([1.0], dtype=torch.float32)), Feat_n_c3], dim=1)
with torch.no_grad():
    preds_c3 = np.array([model(X_c3).numpy() for _ in range(100)])
amp_i_c3 = np.hypot(preds_c3[:, 0, 0] * Ai_std + Ai_mean, preds_c3[:, 0, 1] * Bi_std + Bi_mean)
print(f"\nMID-LEVEL FORCING CHECK: forcing-aware BPINN at target_peak={CASE3_TARGET_PEAK:.4f}")
print(f"  Predicted amp (mode {MODE_I} generalized coord): {amp_i_c3.mean():.6f} +/- {amp_i_c3.std():.6f}")
print(f"  (mode-0-only, NOT the full 5-mode physical displacement -- partial check)")

torch.save(model.state_dict(), os.path.join(OUT, f'bpinn_forcing_aware_{TAG}_state.pt'))
np.savez(os.path.join(OUT, f'bpinn_forcing_aware_{TAG}_norm.npz'),
          feat_mean=Feat_mean.numpy(), feat_std=Feat_std.numpy(),
          alpha_i_mean=Ai_mean, alpha_i_std=Ai_std, beta_i_mean=Bi_mean, beta_i_std=Bi_std,
          alpha_j_mean=Aj_mean, alpha_j_std=Aj_std, beta_j_mean=Bj_mean, beta_j_std=Bj_std,
          r2_i_overall=r2_i_overall, r2_j_overall=r2_j_overall,
          case1_peak_freq=peak_freq, case1_real_freq=CASE1_REAL_FREQ_HZ, force_levels=np.array(FORCE_LEVELS))
print(f"\nTotal time: {time.time()-t_start:.0f}s")
print("DONE", flush=True)
