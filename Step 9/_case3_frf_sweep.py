"""
Real multi-point Case 3 dynamic FRF sweep (2026-08-20, explicit user request: "make Case 3
perfect for a Q1 paper" -- a proper forced-response curve, not one point). Made feasible by the
same-day warm-start fix (build_case3_ic_inp): each point now converges in ~10-12 cycles
(~25-35 min) instead of needing ~200+ cold-start cycles (~6-8 hrs) per point.

Points concentrated tightly around resonance (w=0.95-1.05), not spread wide (e.g. 0.80-1.20):
a cheap ROM-only reconnaissance (rom_predict_steady_state at 19 points, no ANSYS) found the
peak is extremely sharp (amplitude ~0.2mm at w=0.95, ~3.4mm at w=1.0, back to ~0.2mm at w=1.05)
-- light damping (zeta~0.002) makes this a needle, not a broad hump. Points far from resonance
are also near-linear (K3 barely engages at small amplitude) and don't add new nonlinear-model
information beyond what Cases 1/2/4's linear FRF comparisons already validated. Concentrating
real ANSYS compute where the nonlinearity actually matters is the more informative use of a
limited real-data budget for a paper figure, not a wider-but-emptier sweep.
"""
import sys, os, time, json
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9
import numpy as np

W_POINTS = [0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05]
OUT_DIR = r'F:\ANSYS PCE\ROM_data_sensitivity\case3_frf_sweep'
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(OUT_DIR, 'frf_sweep_summary.json')

results = []
t_start = time.time()
for i, w in enumerate(W_POINTS):
    t0 = time.time()
    print(f"\n=== POINT {i+1}/{len(W_POINTS)}: w={w} ===", flush=True)
    r = s9.run_case3_transient_point(mode_index=0, w=w, force_scale=2500.0,
                                       n_cycles=12, steps_per_cycle=20, warm_start=True)
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
    # converged amplitude = mean of the second half of cycles (excludes any
    # residual initial-transient wobble in the first few cycles)
    half = len(amps) // 2
    amp_converged = float(amps[half:].mean())
    amp_std = float(amps[half:].std())
    pred_amp = float(np.hypot(r['pred']['alpha'], r['pred']['beta']) * r['q_ref'] * r['phi_t'])
    elapsed = time.time() - t0
    result = dict(w=w, freq_hz=w * 292.818, amp_ansys_mm=amp_converged, amp_ansys_std_mm=amp_std,
                  amp_rom_mm=pred_amp, all_cycle_amps=amps.tolist(), elapsed_s=elapsed)
    results.append(result)
    print(f"  w={w}: ANSYS converged amp={amp_converged:.3f}+/-{amp_std:.3f} mm, "
          f"ROM predicted={pred_amp:.3f} mm, diff={100*(amp_converged-pred_amp)/pred_amp:+.1f}%  "
          f"({elapsed/60:.1f} min)", flush=True)
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(dict(w_points=W_POINTS, results=results,
                        total_elapsed_s=time.time() - t_start), f, indent=2)
    print(f"  Progress saved: {SUMMARY_PATH}", flush=True)

print(f"\n=== SWEEP DONE in {(time.time()-t_start)/60:.1f} min ===", flush=True)
for r in results:
    print(f"  w={r['w']:.3f}: ANSYS={r['amp_ansys_mm']:.3f}mm, ROM={r['amp_rom_mm']:.3f}mm, "
          f"diff={100*(r['amp_ansys_mm']-r['amp_rom_mm'])/r['amp_rom_mm']:+.1f}%")
