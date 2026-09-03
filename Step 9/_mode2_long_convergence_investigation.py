"""INVESTIGATION (2026-08-27): does mode 2's real ANSYS dynamic response at
w=1.20 actually converge, and if so, to what amplitude vs. the static-
NLGEOM-derived kappa's prediction?

Context: two backbone sweep attempts (8+6=14 points, ~20 hours total real
ANSYS compute) both failed to show clean convergence at n_cycles=30. The
w=1.20 point specifically showed the CLEANEST (lowest-noise) signal: a
steady, monotonic amplitude climb from 2.4mm to 28.9mm over 30 cycles,
still rising (decelerating growth rate) at cutoff -- ~20x the ROM's
kappa=100-based prediction of 1.41mm. Exponential-approach-model
extrapolation on the existing data was inconclusive (wildly different
fitted time constants at neighboring frequencies, some clearly
unreliable fits) -- not trustworthy enough to answer the question without
more real data.

This is a genuine, bounded, deliberate investigation (user-authorized
after the extrapolation attempt): rerun w=1.20 with n_cycles=100 (up from
30) and a slightly coarser steps_per_cycle=15 (down from 20, ~25% cheaper
per cycle) to make ~3.3x more simulated time affordable in similar wall-
clock budget. If the amplitude has clearly plateaued by cycle 100, that's
a real converged value to compare against the static K3 prediction. If
it's STILL climbing, that itself is informative (points toward zeta~0.002
light damping simply needing >>100 cycles regardless of nonlinearity,
i.e. a general settling-time problem, not a kappa-specific one)."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

MODE_INDEX = 2
FORCE_SCALE = 11078.136
W_POINT = 1.20
N_CYCLES = 60   # dialed back from 100 after a crash at ~8GB free C: -- ANSYS stages
                # scratch data on C: even though the main working dir is on F:,
                # and it accumulates fast; 60 cycles (2x the original 30) is a
                # more conservative disk/memory footprint for this retry
STEPS_PER_CYCLE = 15
OUT_DIR = r'F:\ANSYS PCE\ROM_data_sensitivity\mode2_backbone_sweep'
SUMMARY_PATH = os.path.join(OUT_DIR, 'mode2_long_convergence_w1.20.json')

t0 = time.time()
print(f"=== LONG-CYCLE INVESTIGATION: mode 2, w={W_POINT}, n_cycles={N_CYCLES}, "
      f"steps_per_cycle={STEPS_PER_CYCLE} ===", flush=True)
inp = s9.s4.load_inputs()
r = s9.run_case3_transient_point(mode_index=MODE_INDEX, w=W_POINT, force_scale=FORCE_SCALE,
                                   n_cycles=N_CYCLES, steps_per_cycle=STEPS_PER_CYCLE,
                                   inp=inp, warm_start=True)
t_arr, u = r['t'], r['u']
Omega = r['Omega']
T_period = 2 * np.pi / Omega
cyc_idx = ((t_arr - t_arr.min()) / T_period).astype(int)
amps = []
for c in range(int(cyc_idx.max()) + 1):
    mask = cyc_idx == c
    if mask.sum() > 1:
        amps.append((u[mask].max() - u[mask].min()) / 2)
amps = np.array(amps)

# Compare last-10-cycle mean/std to first-attempt-scale 30-cycle window,
# and separately to the very last cycles only (tightest convergence check).
last10_mean = float(amps[-10:].mean()); last10_std = float(amps[-10:].std())
last30_mean = float(amps[-30:].mean()) if len(amps) >= 30 else None
pred_amp = float(np.hypot(r['pred']['alpha'], r['pred']['beta']) * r['q_ref'] * r['phi_t'])

# Back out the EFFECTIVE kappa the observed converged (or still-drifting)
# amplitude would imply, via the same resonance backbone relation used
# throughout this project (1 - w^2 + kappa*a^2 = 0 near resonance,
# rearranged): kappa_eff = (w^2 - 1) / a^2 (only meaningful/positive if
# w is above the shifted resonance, i.e. genuinely on the hardening side).
K = inp['K_sec'][MODE_INDEX, MODE_INDEX]; M = inp['M_sec'][MODE_INDEX, MODE_INDEX]
omega0 = np.sqrt(K / M)
w_ratio = Omega / omega0
kappa_static = 0.75 * inp['K3_sec_diag'][MODE_INDEX] / K
a_gen = last10_mean / abs(r['phi_t'])   # convert physical mm back to generalized coordinate
kappa_eff = (w_ratio**2 - 1) / a_gen**2 if a_gen > 0 else float('nan')

print(f"\nAmplitude trajectory (every 5th cycle):")
for i in range(0, len(amps), 5):
    print(f"  cycle {i:3d}: {amps[i]:.3f} mm")
print(f"\nLast 10 cycles: mean={last10_mean:.3f} mm, std={last10_std:.3f} mm, "
      f"CV={last10_std/last10_mean:.3f}")
print(f"ROM/BPINN prediction (static-K3-derived kappa={kappa_static:.2f}): {pred_amp:.3f} mm")
print(f"Ratio (real ANSYS / static-K3 prediction): {last10_mean/abs(pred_amp):.2f}x")
print(f"Effective kappa implied by observed amplitude: {kappa_eff:.3f}  "
      f"(static-test kappa: {kappa_static:.3f}, ratio: {kappa_eff/kappa_static:.3f}x)")

trend_slope = float(np.polyfit(np.arange(10), amps[-10:], 1)[0])
print(f"\nSlope over last 10 cycles: {trend_slope:.4f} mm/cycle "
      f"({'still rising -- NOT converged' if abs(trend_slope) > 0.05 else 'flat -- looks converged'})")

results = dict(w=W_POINT, n_cycles=N_CYCLES, steps_per_cycle=STEPS_PER_CYCLE,
               all_cycle_amps=amps.tolist(), last10_mean=last10_mean, last10_std=last10_std,
               last30_mean=last30_mean, pred_amp_static_kappa=pred_amp,
               kappa_static=float(kappa_static), kappa_eff=float(kappa_eff),
               trend_slope_last10=trend_slope, elapsed_s=time.time() - t0)
with open(SUMMARY_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {SUMMARY_PATH}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
print("DONE", flush=True)
