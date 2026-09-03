import sys, time, json
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9
import _envelope_fit as ef
import numpy as np

PHI_T = 28.3852  # node 1171, mode 0 (from earlier runs)
OMEGA0 = 2 * np.pi * 292.82   # mode 0 linear natural frequency, rad/s (Step 4's own value)
ZETA = 0.002
W = 1.6
OMEGA = W * OMEGA0

RUNS = [
    ('190N', 190.80 * PHI_T),
    ('380N', 381.60 * PHI_T),
]

results = {}
OUTDIR = r'F:\ANSYS PCE\ROM_data_sensitivity\case3_transient'

for label, F_gen in RUNS:
    t0 = time.time()
    print(f"\n{'#'*70}\n# RUN {label}: w={W}, F_gen={F_gen:.2f}, 160 cycles\n{'#'*70}", flush=True)
    r = s9.run_case3_transient_point(mode_index=0, w=W, force_scale=F_gen,
                                      n_cycles=160, steps_per_cycle=15)
    elapsed = time.time() - t0
    t_arr, u_arr = r['t'], r['u']
    print(f"RUN {label} SOLVE DONE in {elapsed:.1f}s ({elapsed/3600:.2f}h) -- "
          f"{len(t_arr)} points, u range [{u_arr.min():.4f}, {u_arr.max():.4f}] mm", flush=True)

    np.savez(f'{OUTDIR}/overnight_{label}.npz', t=t_arr, u=u_arr, elapsed=elapsed)

    print(f"\n--- Analyzing RUN {label} ---", flush=True)
    check = ef.consistency_check(t_arr, u_arr, OMEGA, OMEGA0, ZETA, label=label)
    results[label] = dict(A_ss_mm=check['full']['A_ss'], r2=check['full']['r2'],
                           passed=bool(check['passed']),
                           rel_diff_half_full=check['rel_diff_half_full'],
                           rel_diff_3q_full=check['rel_diff_3q_full'],
                           elapsed_s=elapsed)

    with open(f'{OUTDIR}/overnight_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if not check['passed']:
        print(f"\n*** RUN {label} FAILED THE CONSISTENCY CHECK -- STOPPING, NOT RUNNING THE NEXT LEVEL ***", flush=True)
        print("Do not trust this result without manual review.", flush=True)
        sys.exit(1)

    print(f"RUN {label}: A_ss = {check['full']['A_ss']:.5f} mm (target-DOF, node 1171) -- "
          f"consistency check PASSED, proceeding.", flush=True)

print(f"\n{'#'*70}\n# BOTH RUNS COMPLETE\n{'#'*70}", flush=True)
a190 = results['190N']['A_ss_mm']
a380 = results['380N']['A_ss_mm']
ratio = a380 / a190
print(f"190N steady amplitude: {a190:.5f} mm", flush=True)
print(f"380N steady amplitude: {a380:.5f} mm", flush=True)
print(f"Ratio (380N/190N): {ratio:.4f}  (linear scaling would give ~2.0, ROM's saturation "
      f"prediction expects ~1.0)", flush=True)
with open(f'{OUTDIR}/overnight_results.json', 'w') as f:
    json.dump(dict(results, ratio_380_190=ratio), f, indent=2)
print("ALL DONE", flush=True)
