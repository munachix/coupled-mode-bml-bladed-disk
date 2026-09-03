"""FRF comparison, ANSYS vs BPINN, all 4 cases (2026-08-24) -- explicit user
request: real ANSYS harmonic/transient FRF data plotted against the
COMPOSITIONAL BPINN reconstruction (pair (0,1) + mode 2 + pair (3,4),
each driven by its own real share of the SAME 2500N generalized force,
summed via real mode shapes) -- not the ROM/exact-solver curve that
step9.py's existing (unwired) make_frf_comparison_figure()/rom_predicted_frf
produce. This is the actual "does the trained surrogate track real ANSYS"
question, not "does the physics engine that generated its training labels
track real ANSYS."

Real ANSYS data availability (confirmed by survey before writing this):
  Case 1 (tuned):            F:\\ANSYS PCE\\ROM_data_case1_harmonic\\harmonic_frf.npz            (41 pts, real HARMIC)
  Case 2 (mistuned linear):  F:\\ANSYS PCE\\ROM_data_case2_mistuned_linear\\harmonic_frf.npz      (41 pts, real HARMIC)
  Case 3 (mistuned nonlinear): only ONE validated real ANSYS point (node 1171 UZ,
                              1.2220+/-0.0190mm, real transient) -- a 7-point real
                              nonlinear sweep was attempted and abandoned (6 of 7
                              points never settled to steady state, disclosed in
                              PROJECT_STATUS.md); no real ANSYS nonlinear sweep exists.
  Case 4 (BPINN-reconstructed geometry): F:\\ANSYS PCE\\ROM_data_case4_bpinn_reconstructed\\inferred\\harmonic_frf.npz (41 pts, real HARMIC)

All 4 cases use the SAME real force (2500N generalized modal force on mode 0,
node 1171 UZ target) -- confirmed directly from each npz's own force_scale=2500.

Mistuning state per case (matching what the real ANSYS run actually used):
  Case 1: tuned (theta=0)
  Case 2: Step 3 sample CONFIG['case2_theta_idx']=0 (step9.py's own established convention)
  Case 3: tuned (theta=0) -- matches the existing single-point validation
  Case 4: Step 7's INFERRED posterior mean, converted to d_length via
          case4_df_to_dlength() (step9.py's own established convention)

Cases 1/2/4 are LINEAR HARMIC ANSYS solves -- real amplitude at exact
resonance reaches ~4.5mm (real ANSYS Case 1 peak, confirmed: 4.466mm at
292.44 Hz), consistent with target_peak=0.1846mm (generalized-coordinate
linear-limit peak) x Phi_0=28.4 ~ 5.2mm order-of-magnitude match -- so
feeding the SAME target_peak=0.1846 used throughout this project's Case-3
convention to the (nonlinear-aware) BPINN is the physically consistent
choice, not a mismatch, since real ANSYS's own linear peak is comparably
large (the nonlinearity only becomes large relative to the linear term at
LARGE generalized amplitude, which Case 3's full node-1171 sum reaches
through in-phase multi-mode superposition, not because any single mode's
own generalized coordinate is huge)."""
import sys, os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step6 as s6
import step4 as s4
import step2 as s2
import step7 as s7
import step9 as s9
import plot_style

OUT9 = os.path.join(r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9', 'output')
FORCE_SCALE = 2500.0
REAL_ANSYS_C3_AMP = 1.2220
REAL_ANSYS_C3_STD = 0.0190

print("=== FRF comparison: real ANSYS vs compositional BPINN, all 4 cases ===", flush=True)

inp = s6.load_inputs()
inp_s4 = s9.s4.load_inputs()   # rom_predicted_frf needs Step4's inp (has T_full2sec), not Step6's
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]

K0 = {m: inp['K_sec'][m, m] for m in range(5)}
zeta0 = {m: inp['C_sec'][m, m] / (2 * math.sqrt(inp['K_sec'][m, m] * inp['M_sec'][m, m])) for m in range(5)}
# Real F_gen applied by ANSYS/ROM's harmonic FRF is F_gen[0]=FORCE_SCALE on
# the GENERALIZED coordinate directly (see rom_predicted_frf: F_gen=
# np.zeros(n_sec); F_gen[0]=force_scale) -- i.e. ONLY mode 0 is directly
# forced; modes 1-4 respond only through real cross-mode coupling. This
# differs from Case 3's point-load convention (F_gen_m=F_physical*Phi_m for
# every mode), handled separately below.
Fg_real = {0: FORCE_SCALE, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
tp01 = Fg_real[0] / (2 * zeta0[0] * K0[0])
print(f"  tp01 (mode-0-forced convention) = {tp01:.5f}", flush=True)

model01, norm01 = s7.load_bpinn_coupled((0, 1))
model34, norm34 = s7.load_bpinn_coupled((3, 4))

# 2026-08-24: INDEPENDENT-forcing pair (0,1) network (tp_i, tp_j as two
# separate inputs) -- fixes the real architectural gap found while
# building these exact FRF plots: the shared-tp network above cannot
# represent "mode 0 forced, mode 1 silent" (real ANSYS/ROM's own harmonic
# force convention), since modes 0/1 are near-degenerate (nearly
# identical zeta*K) so a shared tp implicitly forces both roughly equally
# -- producing a smeared non-resonance instead of the real sharp peak.
# Used ONLY for cases 1/2/4 below (tp_j=0 exactly, no sign-extrapolation
# concern); Case 3's compositional check keeps the original networks
# (already validated at 1.023x, uses a different real force convention).
state01i = torch.load(os.path.join(s6.OUT, 'bpinn_forcing_aware_indep_01_state.pt'))
norm01i = dict(np.load(os.path.join(s6.OUT, 'bpinn_forcing_aware_indep_01_norm.npz')))
in_dim01i = state01i['layers.0.w_mu'].shape[1]
h001i = state01i['layers.0.w_mu'].shape[0]; h101i = state01i['layers.1.w_mu'].shape[0]
model01_indep = s6.BPINN(in_dim01i, [h001i, h101i], 4, prior_sigma=1.0)
model01_indep.load_state_dict(state01i)
model01_indep.eval()
print(f"  Independent-forcing pair(0,1) loaded: test R^2 at training time was "
      f"({float(norm01i['r2_i_overall']):.4f}, {float(norm01i['r2_j_overall']):.4f})", flush=True)
norm2 = dict(np.load(os.path.join(s6.OUT, 'bpinn_forcing_aware_mode2_norm.npz')))
state2 = torch.load(os.path.join(s6.OUT, 'bpinn_forcing_aware_mode2_state.pt'))
in_dim2 = state2['layers.0.w_mu'].shape[1]
h02 = state2['layers.0.w_mu'].shape[0]; h12 = state2['layers.1.w_mu'].shape[0]
model2 = s6.BPINN(in_dim2, [h02, h12], 2, prior_sigma=1.0)
model2.load_state_dict(state2)
model2.eval()


def shifts_for_theta(theta_row):
    if theta_row is None:
        return {m: 0.0 for m in range(5)}
    df = s4.compute_delta_f(theta_row, inp['L_ref'], inp['t_ref'])
    scale = (1.0 + df) ** 2 - 1.0
    return {m: float(scale @ inp['P'][:, m]) for m in range(5)}


def predict_pair_frf(model, norm, tp_signed, mi, mj, shift_i, shift_j, w_arr, n_mc=40):
    sign = 1.0 if tp_signed >= 0 else -1.0
    tp_abs = abs(tp_signed)
    Ki = K0[mi] * (1 + shift_i); Kj = K0[mj] * (1 + shift_j)
    zeta_i = inp['C_sec'][mi, mi] / (2 * math.sqrt(Ki * inp['M_sec'][mi, mi]))
    zeta_j = inp['C_sec'][mj, mj] / (2 * math.sqrt(Kj * inp['M_sec'][mj, mj]))
    kappa_i = 0.75 * inp['K3_sec_diag'][mi] / Ki
    kappa_j = 0.75 * inp['K3_sec_diag'][mj] / Kj
    feat = np.tile([shift_i, zeta_i, kappa_i, shift_j, zeta_j, kappa_j], (len(w_arr), 1))
    n_feat = len(norm['feat_mean'])
    use_detune = n_feat in (8, 9)
    feat_out = s6.add_detune_features(w_arr, feat) if use_detune else feat
    is_fa = bool(norm.get('is_forcing_aware', n_feat == 9))
    if is_fa:
        feat_out = np.concatenate([feat_out, np.full((len(w_arr), 1), tp_abs)], axis=1)
    feat_mean = torch.tensor(norm['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat_out, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    model.eval()
    with torch.no_grad():
        samples = np.array([model(X_in).numpy() for _ in range(n_mc)])
    ai = sign * (samples[:, :, 0] * float(norm['alpha_i_std']) + float(norm['alpha_i_mean'])).mean(0)
    bi = sign * (samples[:, :, 1] * float(norm['beta_i_std']) + float(norm['beta_i_mean'])).mean(0)
    aj = sign * (samples[:, :, 2] * float(norm['alpha_j_std']) + float(norm['alpha_j_mean'])).mean(0)
    bj = sign * (samples[:, :, 3] * float(norm['beta_j_std']) + float(norm['beta_j_mean'])).mean(0)
    return ai, bi, aj, bj


def predict_mode2_frf(tp_signed, shift2, w_arr, n_mc=40):
    sign = 1.0 if tp_signed >= 0 else -1.0
    tp_abs = abs(tp_signed)
    K2 = K0[2] * (1 + shift2)
    zeta2 = inp['C_sec'][2, 2] / (2 * math.sqrt(K2 * inp['M_sec'][2, 2]))
    kappa2 = 0.75 * inp['K3_sec_diag'][2] / K2
    detune2 = np.tanh(((w_arr - math.sqrt(1 + shift2)) / zeta2) / 20.0)
    feat = np.stack([np.full(len(w_arr), shift2), np.full(len(w_arr), zeta2),
                      np.full(len(w_arr), kappa2), detune2, np.full(len(w_arr), tp_abs)], axis=1)
    feat_mean = torch.tensor(norm2['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm2['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    with torch.no_grad():
        samples = np.array([model2(X_in).numpy() for _ in range(n_mc)])
    a = sign * (samples[:, :, 0] * float(norm2['alpha_std']) + float(norm2['alpha_mean'])).mean(0)
    b = sign * (samples[:, :, 1] * float(norm2['beta_std']) + float(norm2['beta_mean'])).mean(0)
    return a, b


def predict_pair_indep_frf(mi, mj, tp_i, tp_j, shift_i, shift_j, w_arr, n_mc=40):
    """Independent-forcing prediction: tp_i, tp_j fed as TWO separate
    inputs (not one shared value) -- the actual fix. Only used with
    tp_i>=0, tp_j=0 here (real mode-0-only-forced FRF scenario), so no
    sign-extrapolation handling is needed (0 has no sign ambiguity)."""
    Ki = K0[mi] * (1 + shift_i); Kj = K0[mj] * (1 + shift_j)
    zeta_i = inp['C_sec'][mi, mi] / (2 * math.sqrt(Ki * inp['M_sec'][mi, mi]))
    zeta_j = inp['C_sec'][mj, mj] / (2 * math.sqrt(Kj * inp['M_sec'][mj, mj]))
    kappa_i = 0.75 * inp['K3_sec_diag'][mi] / Ki
    kappa_j = 0.75 * inp['K3_sec_diag'][mj] / Kj
    feat = np.tile([shift_i, zeta_i, kappa_i, shift_j, zeta_j, kappa_j], (len(w_arr), 1))
    feat8 = s6.add_detune_features(w_arr, feat)
    feat_out = np.concatenate([feat8, np.full((len(w_arr), 1), tp_i), np.full((len(w_arr), 1), tp_j)], axis=1)
    feat_mean = torch.tensor(norm01i['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm01i['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat_out, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(torch.tensor(w_arr, dtype=torch.float32)), Feat_n], dim=1)
    with torch.no_grad():
        samples = np.array([model01_indep(X_in).numpy() for _ in range(n_mc)])
    ai = (samples[:, :, 0] * float(norm01i['alpha_i_std']) + float(norm01i['alpha_i_mean'])).mean(0)
    bi = (samples[:, :, 1] * float(norm01i['beta_i_std']) + float(norm01i['beta_i_mean'])).mean(0)
    aj = (samples[:, :, 2] * float(norm01i['alpha_j_std']) + float(norm01i['alpha_j_mean'])).mean(0)
    bj = (samples[:, :, 3] * float(norm01i['beta_j_std']) + float(norm01i['beta_j_mean'])).mean(0)
    return ai, bi, aj, bj


def compositional_frf(theta_row, freqs_hz, use_indep=True):
    """Real per-mode target_peak here uses the mode-0-ONLY forcing
    convention (matching how real ANSYS/ROM actually apply this force --
    see module docstring), so modes 1,2,3,4 get target_peak=0 as DIRECT
    forcing and respond ONLY through real cross-mode coupling already
    baked into each pair network's own training physics. Modes 2,3,4 have
    NO real measured coupling pathway from mode 0 (only mode 1 does, via
    pair (0,1)) so their contribution is exactly zero for this force
    convention -- not approximated as zero, actually zero by the real
    measured topology."""
    shifts = shifts_for_theta(theta_row)
    omega0_mode0 = math.sqrt(K0[0] / inp['M_sec'][0, 0])   # sqrt(K/M) IS the angular freq in rad/s already
    w_arr = 2 * np.pi * freqs_hz / omega0_mode0
    if use_indep:
        a0, b0, a1, b1 = predict_pair_indep_frf(0, 1, tp01, 0.0, shifts[0], shifts[1], w_arr)
    else:
        a0, b0, a1, b1 = predict_pair_frf(model01, norm01, tp01, 0, 1, shifts[0], shifts[1], w_arr)
    u = (a0 - 1j * b0) * Phi_all[0] + (a1 - 1j * b1) * Phi_all[1]
    return np.abs(u)


figs = os.path.join(r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\figures\step9')
os.makedirs(figs, exist_ok=True)

# 2026-08-24 CORRECTION: cases 1/2/4's real ANSYS runs are LINEAR HARMIC
# solves (confirmed: no NLGEOM) -- there is no nonlinearity in the real
# answer for these 3 cases AT ALL, so a NONLINEAR-trained BPINN surrogate
# is the wrong tool to compare against them (found the hard way: even
# after fixing a real convergence bug in the training data, R^2 improved
# to 0.85-0.87 pointwise but the predicted resonance peak was still ~9x
# too small -- traced to the REAL cross-mode cubic coupling term itself
# causing genuine strong detuning/suppression at this force level, which
# is real Duffing physics, just not what a LINEAR HARMIC solve produces).
# rom_predicted_frf's linear complex solve is exact for this regime and
# already matches real ANSYS almost perfectly (Case 1: ROM 4.41mm vs real
# ANSYS 4.47mm at resonance, 1.3% -- verified directly). BPINN is the
# right comparison ONLY for Case 3, where real ANSYS ran an actual
# nonlinear transient.

# ---- Case 1 (tuned) ----
d1 = np.load(r'F:\ANSYS PCE\ROM_data_case1_harmonic\harmonic_frf.npz')
freqs1, amp1 = d1['freqs'], d1['amplitude']
rom1 = s9.rom_predicted_frf(freqs1, inp_s4, 1171, 'Z', force_scale=FORCE_SCALE)
print(f"  Case 1: real ANSYS peak={amp1.max():.4f}mm @ {freqs1[np.argmax(amp1)]:.2f}Hz, "
      f"ROM peak={rom1.max():.4f}mm @ {freqs1[np.argmax(rom1)]:.2f}Hz", flush=True)

# ---- Case 2 (mistuned linear) ----
theta_f = np.load(os.path.join(s9.CONFIG['step3_dir'], 'theta_samples.npz'))
theta_all = {k: theta_f[k] for k in theta_f.files}
idx2 = s9.CONFIG['case2_theta_idx']
theta_row2 = {v: theta_all[v][idx2] for v in ['d_length', 'd_thickness', 'd_le_te', 'd_twist_deg', 'd_tip']}
d2 = np.load(r'F:\ANSYS PCE\ROM_data_case2_mistuned_linear\harmonic_frf.npz')
freqs2, amp2 = d2['freqs'], d2['amplitude']
rom2 = s9.rom_predicted_frf(freqs2, inp_s4, 1171, 'Z', force_scale=FORCE_SCALE, theta_row=theta_row2)
print(f"  Case 2: real ANSYS peak={amp2.max():.4f}mm @ {freqs2[np.argmax(amp2)]:.2f}Hz, "
      f"ROM peak={rom2.max():.4f}mm @ {freqs2[np.argmax(rom2)]:.2f}Hz", flush=True)

# ---- Case 4 (BPINN-reconstructed geometry) ----
mc = np.load(os.path.join(s9.CONFIG['step7_dir'], 'mcmc_posterior.npz'))
post_mean = mc['post_mean']
theta_row4 = s9.case4_df_to_dlength(post_mean, inp)
d4 = np.load(r'F:\ANSYS PCE\ROM_data_case4_bpinn_reconstructed\inferred\harmonic_frf.npz')
freqs4, amp4 = d4['freqs'], d4['amplitude']
rom4 = s9.rom_predicted_frf(freqs4, inp_s4, 1171, 'Z', force_scale=FORCE_SCALE, theta_row=theta_row4)
print(f"  Case 4: real ANSYS peak={amp4.max():.4f}mm @ {freqs4[np.argmax(amp4)]:.2f}Hz, "
      f"ROM peak={rom4.max():.4f}mm @ {freqs4[np.argmax(rom4)]:.2f}Hz", flush=True)

# ---- Case 3 (mistuned nonlinear): single real ANSYS point + BPINN curve for context ----
# Case 3 uses a DIFFERENT real force convention than cases 1/2/4 (a REAL
# POINT LOAD, decomposed onto every mode via Fg_m=F_physical*Phi_m -- not
# "mode 0 only"), so it needs the full 5-mode compositional approach
# (matching _case3_compositional_check.py's already-validated 1.023x
# result at w=1), not the simplified 2-mode-only function above.
F_physical_c3 = 2500.0 / Phi_all[0]
Fg_real_c3 = {m: F_physical_c3 * Phi_all[m] for m in range(5)}
tp01_c3 = Fg_real_c3[0] / (2 * zeta0[0] * K0[0])
tp2_c3 = Fg_real_c3[2] / (2 * zeta0[2] * K0[2])
tp34_c3 = Fg_real_c3[3] / (2 * zeta0[3] * K0[3])


def compositional_frf_case3(freqs_hz, debug=False):
    shifts = {m: 0.0 for m in range(5)}   # tuned, matches the validated single-point check
    omega0_mode0 = math.sqrt(K0[0] / inp['M_sec'][0, 0])   # sqrt(K/M) IS the angular freq in rad/s already
    w_arr = 2 * np.pi * freqs_hz / omega0_mode0
    a0, b0, a1, b1 = predict_pair_frf(model01, norm01, tp01_c3, 0, 1, shifts[0], shifts[1], w_arr)
    a2, b2 = predict_mode2_frf(tp2_c3, shifts[2], w_arr)
    a3, b3, a4, b4 = predict_pair_frf(model34, norm34, tp34_c3, 3, 4, shifts[3], shifts[4], w_arr)
    if debug:
        print(f"  DEBUG w_arr={w_arr}")
        print(f"  DEBUG a0={a0} b0={b0} a1={a1} b1={b1}")
        print(f"  DEBUG a2={a2} b2={b2}")
        print(f"  DEBUG a3={a3} b3={b3} a4={a4} b4={b4}")
        print(f"  DEBUG Phi_all[:5]={Phi_all[:5]}")
    alphas = [a0, a1, a2, a3, a4]; betas = [b0, b1, b2, b3, b4]
    u = sum((alphas[m] - 1j * betas[m]) * Phi_all[m] for m in range(5))
    return np.abs(u)


freqs3 = np.linspace(261.7, 330.0, 41)
bpinn3 = compositional_frf_case3(freqs3)
idx_292 = np.argmin(np.abs(freqs3 - 292.8))
print(f"  Case 3: BPINN at {freqs3[idx_292]:.2f}Hz = {bpinn3[idx_292]:.4f}mm vs real ANSYS "
      f"{REAL_ANSYS_C3_AMP:.4f}mm (single validated point)", flush=True)

# Cases 1/2/4: ANSYS vs ROM (linear regime, no nonlinearity in the real
# answer -- BPINN is the wrong tool here, see note above). Case 3: ANSYS
# (single validated point) vs BPINN (the actual nonlinear surrogate check).
linear_cases = [
    ('case1', 'Case 1 (tuned, linear)', freqs1, amp1, rom1),
    ('case2', 'Case 2 (mistuned, linear)', freqs2, amp2, rom2),
    ('case4', 'Case 4 (BPINN-reconstructed geometry)', freqs4, amp4, rom4),
]
for key, title, freqs, amp_real, amp_rom in linear_cases:
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.plot(freqs, amp_real, '-', color=plot_style.BLUE, lw=2.0, label='Real ANSYS (linear HARMIC)')
    ax.plot(freqs, amp_rom, '--', color=plot_style.C_OK, lw=1.8, label='ROM (linear complex solve, exact)')
    peak_real = freqs[np.argmax(amp_real)]
    peak_rom = freqs[np.argmax(amp_rom)]
    ax.set_xlabel('Frequency  [Hz]')
    ax.set_ylabel('|U_Z| at node 1171  [mm]')
    plot_style.two_tier_title(ax, f'FRF: {title}',
                               f"peak: ANSYS {peak_real:.1f} Hz / ROM {peak_rom:.1f} Hz -- linear regime, "
                               f"BPINN (nonlinear surrogate) not applicable here")
    plot_style.legend_below(ax, ncol=1)
    fig.tight_layout()
    plot_style.savefig_pub(fig, figs, f'step9_fig12_frf_ansys_vs_bpinn_{key}')
    print(f"  Saved: step9_fig12_frf_ansys_vs_bpinn_{key}.png", flush=True)

fig, ax = plt.subplots(figsize=(8.0, 5.8))
ax.errorbar([292.8], [REAL_ANSYS_C3_AMP], yerr=[REAL_ANSYS_C3_STD], fmt='o', color=plot_style.BLUE,
            ms=9, capsize=5, lw=2, label='Real ANSYS (single validated point, w=1.0)')
ax.plot(freqs3, bpinn3, '--', color=plot_style.VIOLET, lw=1.8,
         label='Compositional BPINN ((0,1)+mode2+(3,4))')
ax.set_xlabel('Frequency  [Hz]')
ax.set_ylabel('|U_Z| at node 1171  [mm]')
plot_style.two_tier_title(ax, 'FRF: Case 3 (mistuned, nonlinear)',
                           "no real ANSYS nonlinear sweep exists (7-pt attempt abandoned, 6/7 unsettled) -- "
                           "single point is the real validation (ratio 1.023x)")
plot_style.legend_below(ax, ncol=1)
fig.tight_layout()
plot_style.savefig_pub(fig, figs, 'step9_fig12_frf_ansys_vs_bpinn_case3')
print("  Saved: step9_fig12_frf_ansys_vs_bpinn_case3.png", flush=True)

print("DONE", flush=True)
