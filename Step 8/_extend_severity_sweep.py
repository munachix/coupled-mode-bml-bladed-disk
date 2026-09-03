"""
Extends Sweep B (classifier vs. injected severity, Fig. "classifier
population screen") beyond the original 5%-30% range so the figure
actually shows a real MISTUNED verdict, not only "tuned" points
(explicit user/supervisor request -- "everything can't be tuned if
we're talking about mistuning"). Sweep A (8-location localization sweep)
is loaded unchanged from the already-saved health_monitoring_sweep.json,
not re-run, since only Sweep B's severity range needs to change. Uses
Step 8's own validated machinery (build_damage_trajectory, run_trajectory,
the trained classifier) exactly as _health_monitoring_sweep.py did --
same fixed blade (same seed), same calibration -- just a wider real
severity range: 5% to 90%.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 8')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step8 as s8

OUT = s8.OUT
t_start = time.time()
print("=== Extending Sweep B (severity) to find a real MISTUNED crossover ===", flush=True)

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
calib = s8.calibrate_and_validate(inp, prior, df_all)
mclf = s8.calibrate_mistuning_classifier(inp, df_all, models, pairs, chain)
print(f"Calibrations done at {time.time()-t_start:.0f}s", flush=True)


def run_one_scenario(damaged_blade, severity_max, tag):
    t0 = time.time()
    traj = s8.build_damage_trajectory(df_all, damaged_blade=damaged_blade, severity_max=severity_max,
                                       record_checks=False)
    traj_result = s8.run_trajectory(inp, prior, traj, models, pairs, chain)
    detect = s8.detect_and_localize(traj, traj_result, calib, HI1_healthy, inp)
    mclf_val = s8.validate_mistuning_classifier(inp, traj, traj_result, mclf, models, pairs, chain)
    cyclic_dist = min(abs(detect['localized_blade'] - traj['damaged_blade']),
                       s8.NB - abs(detect['localized_blade'] - traj['damaged_blade']))
    final_sigma = float(mclf_val['sigmas'][-1])
    final_score = mclf['mean'] + final_sigma * mclf['std']
    result = dict(
        tag=tag, damaged_blade=damaged_blade, severity_max=severity_max,
        detect_idx=detect['detect_idx'], localized_blade=detect['localized_blade'],
        ring_distance=cyclic_dist,
        localization_residuals=[float(v) for v in detect['sparse_result']['residuals']],
        hi2_corr=float(np.corrcoef(traj['severity'], traj_result['HI2'])[0, 1]),
        hi3_corr=float(np.corrcoef(traj['severity'], traj_result['HI3'])[0, 1]),
        classifier_corr=mclf_val['corr'],
        classifier_final_verdict=bool(mclf_val['verdicts'][-1]),
        classifier_final_sigma=final_sigma,
        classifier_final_score_mm=final_score,
        elapsed=time.time() - t0)
    print(f"  [{tag}] blade={damaged_blade}, severity={severity_max:.2f}: "
          f"detect_idx={detect['detect_idx']}, score_mm={final_score:.5f}, "
          f"verdict={'MISTUNED' if result['classifier_final_verdict'] else 'tuned'}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return result


rng7 = np.random.default_rng(s8.CONFIG['damage']['damaged_blade_seed'])
fixed_blade = int(rng7.integers(0, s8.NB))
SEVERITIES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]
sweep_b = []
for sev in SEVERITIES:
    sweep_b.append(run_one_scenario(fixed_blade, sev, f'B-sev{sev:.2f}'))

n_detected_b = sum(1 for r in sweep_b if r['detect_idx'] >= 0)
n_classified_b = sum(1 for r in sweep_b if r['classifier_final_verdict'])
print(f"\nExtended Sweep B: detected {n_detected_b}/{len(sweep_b)}; "
      f"classifier flagged MISTUNED at final step for {n_classified_b}/{len(sweep_b)}", flush=True)

fp = os.path.join(OUT, 'health_monitoring_sweep.json')
with open(fp) as f:
    old = json.load(f)
old['sweep_b'] = sweep_b
old['aggregate']['n_detected_b'] = n_detected_b
old['aggregate']['n_classified_b'] = n_classified_b
old['aggregate']['n_scenarios_b'] = len(sweep_b)
with open(fp, 'w') as f:
    json.dump(old, f, indent=2)
print(f"Saved (sweep_a preserved unchanged): {fp}", flush=True)
print(f"Total elapsed: {time.time()-t_start:.0f}s", flush=True)
