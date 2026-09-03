"""
Diagnostic (not part of the pipeline): WHY does compute_mistuning_severity
(referenced against TRUE ZERO mistuning) correlate with true severity so
much worse (-0.877) than compute_HI3 (referenced against the unit's own
as-built baseline, -0.992), despite sharing the identical precision-
weighted aggregation code (compute_HI3 IS compute_mistuning_severity's
implementation -- only the reference dict differs)?

Two candidate confounds, isolated separately below:
  (1) REFERENCE POINT: zero-mistuning vs. the unit's own baseline df.
  (2) INPUT POINT: run_trajectory's HI3 call uses the Bayesian-INFERRED
      posterior mean (pm) at each step; validate_mistuning_classifier's
      call uses the TRUE injected df at each step (df_true_t). These are
      different quantities even before the reference point is considered.

We cross all 4 combinations (input in {true, posterior-mean} x reference
in {baseline, zero}) at every one of the 20 real trajectory steps (using
the ACTUAL saved posterior means from Step 8's own last run,
output/damage_trajectory.npz, to avoid re-running 20 expensive MCMC
inversions here), and also do a per-channel breakdown (chain vs pairs vs
single contribution fraction of total precision-weighted score) at every
step for the two reference points, to see whether chain-variance
instability specifically gets WORSE when referenced against zero.

We ALSO repeat the per-channel breakdown across many (30) real calibration
samples (same population calibrate_mistuning_classifier draws from),
referenced against zero (what the classifier actually does) vs referenced
against the trajectory's own baseline (a stand-in for "a nearby, not
maximally-distant reference"), to see if reference distance changes how
often the chain channel dominates.
"""
import sys, os, math
import numpy as np

_STEP8 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 8'
_STEP7 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7'
_STEP6 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6'
_STEP4 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4'
for p in (_STEP8, _STEP7, _STEP6, _STEP4):
    sys.path.insert(0, p)
import torch
import step8 as s8, step7 as s7, step6 as s6

EPS_STD = 1e-7


def per_channel_breakdown(df_point, ref_pred, inp, models, pairs, chain, w_eval=1.0):
    """Same math as compute_HI3/compute_mistuning_severity, but returns the
    full per-channel ledger instead of collapsing to one scalar."""
    rows = []
    weighted_dev_sq = 0.0
    weight_sum = 0.0

    for pair, (pair_model, pair_norm) in pairs.items():
        bp = ref_pred['pairs'][pair]
        p_cur = s7.coupled_features_from_df(df_point, pair, inp)
        omega0_cur = np.sqrt(p_cur['Ki'] / p_cur['Mi'])
        omega_ref = w_eval * bp['omega0']
        w_cur = omega_ref / omega0_cur
        ai_cur, si_cur, aj_cur, sj_cur = s7.predict_coupled_mc(
            pair_model, pair_norm, np.array([w_cur]), p_cur['feat'][None, :], n_mc=40)
        wi = 1.0 / (si_cur[0] ** 2 + bp['si'] ** 2 + EPS_STD)
        wj = 1.0 / (sj_cur[0] ** 2 + bp['sj'] ** 2 + EPS_STD)
        dev_i = float(ai_cur[0]) - bp['ai']; dev_j = float(aj_cur[0]) - bp['aj']
        ci = wi * dev_i ** 2; cj = wj * dev_j ** 2
        weighted_dev_sq += ci + cj
        weight_sum += wi + wj
        rows.append(dict(name=f'pair{pair}_i', weight=float(wi), dev=float(dev_i), contrib=float(ci),
                          std_cur=float(si_cur[0]), std_ref=float(bp['si'])))
        rows.append(dict(name=f'pair{pair}_j', weight=float(wj), dev=float(dev_j), contrib=float(cj),
                          std_cur=float(sj_cur[0]), std_ref=float(bp['sj'])))

    chain_model, chain_norm, chain_modes = chain
    bc = ref_pred['chain']
    p_cur_c = s7.chain_features_from_df(df_point, chain_modes, inp)
    omega0_cur_chain = math.sqrt((p_cur_c['K_arr'] / p_cur_c['M_arr']).mean())
    omega_ref_c = w_eval * bc['omega0']
    w_cur_c = omega_ref_c / omega0_cur_chain
    amp_cur_c, std_cur_c = s7.predict_chain_mc(chain_model, chain_norm, np.array([w_cur_c]),
                                                p_cur_c['feat'][None, :], n_mc=40)
    w_chain = 1.0 / (std_cur_c[0] ** 2 + bc['std'] ** 2 + EPS_STD)
    dev_chain = amp_cur_c[0] - bc['amp']
    contrib_chain = w_chain * dev_chain ** 2
    weighted_dev_sq += float(np.sum(contrib_chain))
    weight_sum += float(np.sum(w_chain))
    rows.append(dict(name='chain_total', weight=float(np.sum(w_chain)), dev=float(np.sqrt(np.mean(dev_chain**2))),
                      contrib=float(np.sum(contrib_chain)),
                      std_cur=float(np.mean(std_cur_c)), std_ref=float(np.mean(bc['std'])),
                      contrib_per_mode=[float(c) for c in contrib_chain],
                      std_cur_per_mode=[float(s) for s in std_cur_c[0]]))

    for m, (model, norm_stats) in models.items():
        feat_mean, feat_std, out_norm = norm_stats
        is_fa = len(feat_mean) == 5
        bm = ref_pred['single'][m]
        p_cur = s7.sdof_params_from_df(df_point, m, inp)
        omega0_cur_m = np.sqrt(p_cur['K'] / p_cur['M'])
        omega_ref_m = w_eval * bm['omega0']
        w_cur_m = omega_ref_m / omega0_cur_m
        amp_cur, std_cur, _, _ = s6.predict_mc(model, np.array([w_cur_m]), p_cur['features'][None, :],
                                                feat_mean, feat_std, n_mc=40,
                                                is_forcing_aware=is_fa, target_peak=0.8 if is_fa else None,
                                                out_norm=out_norm)
        w_m = 1.0 / (std_cur[0] ** 2 + bm['std'] ** 2 + EPS_STD)
        dev_m = float(amp_cur[0]) - bm['amp']
        cm = w_m * dev_m ** 2
        weighted_dev_sq += cm
        weight_sum += w_m
        rows.append(dict(name=f'single{m}', weight=float(w_m), dev=float(dev_m), contrib=float(cm),
                          std_cur=float(std_cur[0]), std_ref=float(bm['std'])))

    score = float(np.sqrt(weighted_dev_sq / weight_sum))
    total_contrib = sum(r['contrib'] for r in rows)
    for r in rows:
        r['frac'] = r['contrib'] / total_contrib if total_contrib > 0 else 0.0
    return score, rows


def summarize(rows, score):
    chain_frac = sum(r['frac'] for r in rows if r['name'] == 'chain_total')
    max_row = max(rows, key=lambda r: r['frac'])
    return chain_frac, max_row['name'], max_row['frac'], score


if __name__ == '__main__':
    torch.manual_seed(s8.CONFIG['random_seed'])
    print("Loading Step 8 inputs...", flush=True)
    inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()

    traj = s8.build_damage_trajectory(df_all, record_checks=False)
    d = np.load(os.path.join(s8.OUT, 'damage_trajectory.npz'))
    post_means = d['post_means']
    assert np.allclose(d['df_baseline'], traj['df_baseline']), "saved npz baseline mismatch with current CONFIG"
    assert int(d['damaged_blade']) == traj['damaged_blade']
    severity = traj['severity']
    T = len(severity)
    print(f"Loaded trajectory: baseline_idx={s8.CONFIG['unit_baseline_idx']}, damaged_blade={traj['damaged_blade']}, T={T}")

    print("\nPrecomputing reference predictions (baseline-referenced and zero-referenced)...", flush=True)
    ref_pred_baseline = s8.compute_baseline_predictions(traj['df_baseline'], inp, models, pairs, chain)
    ref_pred_zero = s8.compute_baseline_predictions(np.zeros(s8.NB), inp, models, pairs, chain)

    combos = {
        'true_vs_baseline (HI3-style input, HI3 reference)': [],
        'true_vs_zero (classifier AS-IMPLEMENTED)': [],
        'pm_vs_baseline (HI3 AS-IMPLEMENTED)': [],
        'pm_vs_zero (posterior mean, zero ref)': [],
    }
    chain_frac_true_zero = []
    chain_frac_true_base = []
    chain_frac_pm_zero = []
    chain_frac_pm_base = []

    print("\nRunning per-step per-channel breakdown across all 20 trajectory steps "
          "(4 combos x 20 steps = 80 full BPINN evaluations)...", flush=True)
    for t in range(T):
        df_true_t = traj['df_traj'][t]
        pm_t = post_means[t]

        s_tb, r_tb = per_channel_breakdown(df_true_t, ref_pred_baseline, inp, models, pairs, chain)
        s_tz, r_tz = per_channel_breakdown(df_true_t, ref_pred_zero, inp, models, pairs, chain)
        s_pb, r_pb = per_channel_breakdown(pm_t, ref_pred_baseline, inp, models, pairs, chain)
        s_pz, r_pz = per_channel_breakdown(pm_t, ref_pred_zero, inp, models, pairs, chain)

        combos['true_vs_baseline (HI3-style input, HI3 reference)'].append(s_tb)
        combos['true_vs_zero (classifier AS-IMPLEMENTED)'].append(s_tz)
        combos['pm_vs_baseline (HI3 AS-IMPLEMENTED)'].append(s_pb)
        combos['pm_vs_zero (posterior mean, zero ref)'].append(s_pz)

        cf_tb, mn_tb, mf_tb, _ = summarize(r_tb, s_tb)
        cf_tz, mn_tz, mf_tz, _ = summarize(r_tz, s_tz)
        cf_pb, mn_pb, mf_pb, _ = summarize(r_pb, s_pb)
        cf_pz, mn_pz, mf_pz, _ = summarize(r_pz, s_pz)
        chain_frac_true_base.append(cf_tb)
        chain_frac_true_zero.append(cf_tz)
        chain_frac_pm_base.append(cf_pb)
        chain_frac_pm_zero.append(cf_pz)

        print(f"  t={t:2d} sev={severity[t]*100:6.2f}%  "
              f"score[true,base]={s_tb:.6f} (chain_frac={cf_tb:.3f}, top={mn_tb}:{mf_tb:.3f})  "
              f"score[true,zero]={s_tz:.6f} (chain_frac={cf_tz:.3f}, top={mn_tz}:{mf_tz:.3f})  "
              f"score[pm,base]={s_pb:.6f} (chain_frac={cf_pb:.3f})  "
              f"score[pm,zero]={s_pz:.6f} (chain_frac={cf_pz:.3f})", flush=True)

    print("\n=== CORRELATIONS (severity vs each combo's score) ===")
    for name, vals in combos.items():
        c = float(np.corrcoef(severity, vals)[0, 1])
        print(f"  {name}: corr = {c:.4f}")

    print("\n=== MEAN CHAIN-CHANNEL FRACTION OF TOTAL WEIGHTED SCORE (across 20 steps) ===")
    print(f"  true input,  baseline ref: mean={np.mean(chain_frac_true_base):.3f}  max={np.max(chain_frac_true_base):.3f}")
    print(f"  true input,  zero ref:     mean={np.mean(chain_frac_true_zero):.3f}  max={np.max(chain_frac_true_zero):.3f}")
    print(f"  pm input,    baseline ref: mean={np.mean(chain_frac_pm_base):.3f}  max={np.max(chain_frac_pm_base):.3f}")
    print(f"  pm input,    zero ref:     mean={np.mean(chain_frac_pm_zero):.3f}  max={np.max(chain_frac_pm_zero):.3f}")

    # ---- Multi-sample calibration-population check ----
    print("\n=== CALIBRATION-POPULATION CHECK (30 real Step-3 samples, zero ref vs baseline ref) ===")
    rng = np.random.default_rng(12345)
    idxs = rng.choice(df_all.shape[0], size=30, replace=False)
    cf_zero_list, cf_base_list = [], []
    for i, idx in enumerate(idxs):
        df_i = df_all[idx]
        s_z, r_z = per_channel_breakdown(df_i, ref_pred_zero, inp, models, pairs, chain)
        s_b, r_b = per_channel_breakdown(df_i, ref_pred_baseline, inp, models, pairs, chain)
        cf_z, mn_z, mf_z, _ = summarize(r_z, s_z)
        cf_b, mn_b, mf_b, _ = summarize(r_b, s_b)
        cf_zero_list.append(cf_z); cf_base_list.append(cf_b)
        print(f"  sample #{idx}: score[zero_ref]={s_z:.6f} (chain_frac={cf_z:.3f}, top={mn_z}:{mf_z:.3f})   "
              f"score[baseline_ref]={s_b:.6f} (chain_frac={cf_b:.3f}, top={mn_b}:{mf_b:.3f})", flush=True)

    print(f"\n  Across 30 real samples: mean chain_frac (zero ref) = {np.mean(cf_zero_list):.3f}, "
          f"n with chain_frac>0.5: {sum(1 for c in cf_zero_list if c > 0.5)}/30")
    print(f"  Across 30 real samples: mean chain_frac (baseline ref) = {np.mean(cf_base_list):.3f}, "
          f"n with chain_frac>0.5: {sum(1 for c in cf_base_list if c > 0.5)}/30")

    print("\nDONE")
