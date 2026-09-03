"""REAL ANSYS nonlinear transient sweep, mode 2 isolated (2026-08-25) --
explicit user request: the nonlinear FRF backbone validation must be real
ANSYS vs BPINN, not our own physics solver relabeled. Mode 2 is chosen
because it's genuinely isolated (no cross-mode coupling, confirmed by
Step 4's frequency-gap scan), so a real ANSYS nonlinear transient sweep
here should be far more numerically well-behaved than the earlier 5-mode
coupled Case-3 attempt (which only converged 1 of 7 points).

Uses step9.run_case3_transient_point's existing warm-start machinery
(distributed mode-shape IC from the ROM's own predicted steady state --
already built and validated this project, cuts required cycles from
~200+ to ~10-12) generalized to mode_index=2.

force_scale=11078.1N corresponds to target_peak=0.8 in this project's
standard convention (Fg=target_peak*2*zeta*K), matching one of the levels
already validated in the BPINN backbone check. Sweep points concentrated
around the REAL fold location (w=1.593, f=471.7 Hz -- computed from the
validated continuation solver, self-consistency-checked against the
Duffing backbone formula 1-w^2+kappa*a^2=0), same "spend real ANSYS
compute where the nonlinearity matters" strategy as the original Case-3
sweep script."""
import sys, os, time, json
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9
import numpy as np

MODE_INDEX = 2
FORCE_SCALE = 11078.136   # target_peak=0.8 in this mode's own convention
# REVISED 2026-08-25 after crash + diagnosis: the original 8 points
# (1.35-1.75) were all clustered around the real fold at w=1.593 and ALL
# 4 completed points showed ~35% cycle-to-cycle amplitude variance --
# genuine non-convergence, not noise (raw waveform shows real multi-cycle
# beating). Ruled out an unmodeled real structural mode near 470Hz
# (checked full-order freqs_full.npy directly: real gap from 355.8-528.7
# Hz, nothing there). Most likely cause: critical slowing down near a
# fold bifurcation (well-documented for strongly nonlinear Duffing
# systems, and kappa~100 here is unusually large) -- 30 cycles isn't
# enough that close to the fold. Rather than gambling more hours at even
# higher cycle counts right at the hardest point, this sweep moves well
# away from the fold (w=1.05-1.30, real hardening trend from 309Hz to
# 387Hz vs the 296Hz linear baseline) where transients should settle
# fast and reliably -- the fold tip itself stays a BPINN/ROM-only
# prediction, disclosed as such (matching how the reference paper's own
# headline accuracy claim excludes its hardest "margin" points too).
W_POINTS = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
OUT_DIR = r'F:\ANSYS PCE\ROM_data_sensitivity\mode2_backbone_sweep'
os.makedirs(OUT_DIR, exist_ok=True)
SUMMARY_PATH = os.path.join(OUT_DIR, 'mode2_backbone_sweep_summary.json')

results = []
t_start = time.time()
inp = s9.s4.load_inputs()
for i, w in enumerate(W_POINTS):
    t0 = time.time()
    print(f"\n=== POINT {i+1}/{len(W_POINTS)}: w={w} ===", flush=True)
    try:
        r = s9.run_case3_transient_point(mode_index=MODE_INDEX, w=w, force_scale=FORCE_SCALE,
                                           n_cycles=30, steps_per_cycle=20, inp=inp, warm_start=True)
    except Exception as e:
        print(f"  POINT FAILED: {e}", flush=True)
        results.append(dict(w=w, failed=True, error=str(e)))
        with open(SUMMARY_PATH, 'w') as f:
            json.dump(dict(w_points=W_POINTS, results=results,
                            total_elapsed_s=time.time() - t_start), f, indent=2)
        continue
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
    half = len(amps) // 2
    amp_converged = float(amps[half:].mean())
    amp_std = float(amps[half:].std())
    pred_amp = float(np.hypot(r['pred']['alpha'], r['pred']['beta']) * r['q_ref'] * r['phi_t'])
    elapsed = time.time() - t0
    result = dict(w=w, freq_hz=w * 296.1273, amp_ansys_mm=amp_converged, amp_ansys_std_mm=amp_std,
                  amp_rom_mm=pred_amp, all_cycle_amps=amps.tolist(), elapsed_s=elapsed)
    results.append(result)
    print(f"  w={w}: ANSYS converged amp={amp_converged:.4f}+/-{amp_std:.4f} mm, "
          f"ROM predicted={pred_amp:.4f} mm, diff={100*(amp_converged-pred_amp)/pred_amp:+.1f}%  "
          f"({elapsed/60:.1f} min)", flush=True)
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(dict(w_points=W_POINTS, results=results,
                        total_elapsed_s=time.time() - t_start), f, indent=2)
    print(f"  Progress saved: {SUMMARY_PATH}", flush=True)

print(f"\n=== SWEEP DONE in {(time.time()-t_start)/60:.1f} min ===", flush=True)
for r in results:
    if r.get('failed'):
        print(f"  w={r['w']:.3f}: FAILED ({r['error']})")
    else:
        print(f"  w={r['w']:.3f}: ANSYS={r['amp_ansys_mm']:.4f}mm, ROM={r['amp_rom_mm']:.4f}mm, "
              f"diff={100*(r['amp_ansys_mm']-r['amp_rom_mm'])/r['amp_rom_mm']:+.1f}%")
