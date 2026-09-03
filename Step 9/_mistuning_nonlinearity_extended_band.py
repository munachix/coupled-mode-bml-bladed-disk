# -*- coding: utf-8 -*-
"""
Re-run the 200-realization mistuning-vs-nonlinearity ensemble over the extended
frequency band (Section 3.4.1).

Why this rerun was needed. The original ensemble traced each realization with
w_stop_hi = 1.6, then took `peak_amp = max(amplitude)` and the frequency at that
maximum. For this mode's identified cubic stiffness the response is still
climbing at w = 1.6, so that maximum was the value at the right-hand edge of the
band, not the resonance peak. Two consequences ran straight into the reported
results:

  * every realization returned a band-edge amplitude, so `peak_amp` spanned only
    0.1238-0.1303 mm and correlated with mistuning at -0.032 -- a null that was
    at least partly the truncation, not the physics;
  * `res_freq` carried a large shared offset (~170-190 Hz), which the original
    script noticed and centred away. That offset was the truncation edge, not a
    hardening offset.

Extending w_stop_hi to 6.0 with ds = 0.004 lets every realization reach its true
saddle-node fold, so `peak_amp` is the actual resonance peak and `res_freq` is
the actual nonlinear resonance location. Nothing else changes: same 200 samples,
same forcing level, same already-validated continuation solver.

Output: Step 9/output/mistuning_nonlinearity_extended.npz
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 5")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project")
import step4 as s4  # noqa: E402
import step5 as s5  # noqa: E402
import step6 as s6  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

N_SAMPLES = 200
TARGET_PEAK = 1.0

cfg = s4.CONFIG["continuation"]
cfg["w_stop_hi"] = 6.0
cfg["ds"] = 0.004
cfg["n_steps"] = 25000

inp = s6.load_inputs()
sample_idx = np.arange(min(N_SAMPLES, int(inp["n_samples"])))
f0_linear = inp["freqs_sec"][0]

n = len(sample_idx)
max_abs_df = np.zeros(n)
shift_m = np.zeros(n)
peak_amp = np.zeros(n)
peak_freq = np.zeros(n)
fold_freq = np.zeros(n)
n_folds = np.zeros(n, dtype=int)

print(f"=== Extended-band mistuning/nonlinearity ensemble ({n} samples) ===", flush=True)
print(f"  linear f0 = {f0_linear:.2f} Hz, w_stop_hi = {cfg['w_stop_hi']}, ds = {cfg['ds']}",
      flush=True)
t0 = time.time()

for i, idx in enumerate(sample_idx):
    row = {v: inp["theta"][v][idx] for v in s6.VAR_NAMES}
    df = s5.compute_delta_f_vectorized({k: v[None, :] for k, v in row.items()},
                                       inp["sens"], inp["L_ref"], inp["t_ref"])[0]
    max_abs_df[i] = np.max(np.abs(df)) * 100

    params = s6.per_sample_sdof_params(inp, idx)
    shift_m[i] = params["features"][0] * 100

    cont = s4.duffing_forced_response_continuation(
        params["omega0"], params["M"], params["C"], params["K"], params["K3"],
        params["q_ref"], TARGET_PEAK)
    amp = cont["amplitude"]
    hz = cont["Omega"] / (2 * np.pi)
    j = int(np.argmax(amp))
    peak_amp[i] = amp[j]
    peak_freq[i] = hz[j]
    folds = cont["fold_indices"]
    n_folds[i] = cont["n_folds"]
    fold_freq[i] = hz[folds[0]] if len(folds) else np.nan

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{n} done ({time.time()-t0:.0f}s)", flush=True)

peak_freq_centered = peak_freq - peak_freq.mean()

print(f"\n  elapsed {time.time()-t0:.0f}s", flush=True)
print(f"  realizations reaching a fold: {(n_folds >= 1).sum()}/{n}", flush=True)
print(f"  worst-blade |df/f|      : [{max_abs_df.min():.3f}, {max_abs_df.max():.3f}] %", flush=True)
print(f"  mode-0 stiffness shift  : [{shift_m.min():.3f}, {shift_m.max():.3f}] %", flush=True)
print(f"  TRUE peak amplitude     : [{peak_amp.min():.4f}, {peak_amp.max():.4f}] mm", flush=True)
print(f"  TRUE peak frequency     : [{peak_freq.min():.1f}, {peak_freq.max():.1f}] Hz", flush=True)
print(f"  centered peak-freq shift: [{peak_freq_centered.min():.3f}, "
      f"{peak_freq_centered.max():.3f}] Hz", flush=True)
c_amp = float(np.corrcoef(max_abs_df, peak_amp)[0, 1])
c_frq = float(np.corrcoef(shift_m, peak_freq_centered)[0, 1])
print(f"\n  corr(worst-blade |df/f|, TRUE peak amplitude) = {c_amp:+.4f}", flush=True)
print(f"  corr(mode-0 shift,       centered peak freq)  = {c_frq:+.4f}", flush=True)

np.savez(os.path.join(OUT, "mistuning_nonlinearity_extended.npz"),
         max_abs_df=max_abs_df, shift_m=shift_m, peak_amp=peak_amp,
         peak_freq=peak_freq, peak_freq_centered=peak_freq_centered,
         fold_freq=fold_freq, n_folds=n_folds,
         corr_amp=c_amp, corr_freq=c_frq, target_peak=TARGET_PEAK)
print("\nSaved mistuning_nonlinearity_extended.npz", flush=True)
