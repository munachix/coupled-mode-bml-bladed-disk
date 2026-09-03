"""
Two more candidate fixes to measure honestly before picking one:
  (1) larger n_mc (more MC draws -> less noisy variance estimate) for the
      CURRENT-state evaluation, applied to the weak true_vs_zero case.
  (2) a variance floor expressed RELATIVE to each channel's own reference
      amplitude scale (not a fixed absolute mm^2), applied post-hoc to the
      cached weight/dev terms.
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


def all_terms_nmc(df_point, ref_pred, inp, models, pairs, chain, w_eval=1.0, n_mc=40):
    weights, dev_sqs, names, amps_ref = [], [], [], []

    for pair, (pair_model, pair_norm) in pairs.items():
        bp = ref_pred['pairs'][pair]
        p_cur = s7.coupled_features_from_df(df_point, pair, inp)
        omega0_cur = np.sqrt(p_cur['Ki'] / p_cur['Mi'])
        omega_ref = w_eval * bp['omega0']
        w_cur = omega_ref / omega0_cur
        ai_cur, si_cur, aj_cur, sj_cur = s7.predict_coupled_mc(
            pair_model, pair_norm, np.array([w_cur]), p_cur['feat'][None, :], n_mc=n_mc)
        wi = 1.0 / (si_cur[0] ** 2 + bp['si'] ** 2 + EPS_STD)
        wj = 1.0 / (sj_cur[0] ** 2 + bp['sj'] ** 2 + EPS_STD)
        weights += [float(wi), float(wj)]
        dev_sqs += [(float(ai_cur[0]) - bp['ai']) ** 2, (float(aj_cur[0]) - bp['aj']) ** 2]
        names += [f'pair{pair}_i', f'pair{pair}_j']
        amps_ref += [bp['ai'], bp['aj']]

    chain_model, chain_norm, chain_modes = chain
    bc = ref_pred['chain']
    p_cur_c = s7.chain_features_from_df(df_point, chain_modes, inp)
    omega0_cur_chain = math.sqrt((p_cur_c['K_arr'] / p_cur_c['M_arr']).mean())
    omega_ref_c = w_eval * bc['omega0']
    w_cur_c = omega_ref_c / omega0_cur_chain
    amp_cur_c, std_cur_c = s7.predict_chain_mc(chain_model, chain_norm, np.array([w_cur_c]),
                                                p_cur_c['feat'][None, :], n_mc=n_mc)
    w_chain = 1.0 / (std_cur_c[0] ** 2 + bc['std'] ** 2 + EPS_STD)
    dev_chain_sq = (amp_cur_c[0] - bc['amp']) ** 2
    for k, (wc, dc) in enumerate(zip(w_chain, dev_chain_sq)):
        weights.append(float(wc)); dev_sqs.append(float(dc)); names.append(f'chain_m{chain_modes[k]}')
        amps_ref.append(bc['amp'][k])

    for m, (model, norm_stats) in models.items():
        feat_mean, feat_std, out_norm = norm_stats
        is_fa = len(feat_mean) == 5
        bm = ref_pred['single'][m]
        p_cur = s7.sdof_params_from_df(df_point, m, inp)
        omega0_cur_m = np.sqrt(p_cur['K'] / p_cur['M'])
        omega_ref_m = w_eval * bm['omega0']
        w_cur_m = omega_ref_m / omega0_cur_m
        amp_cur, std_cur, _, _ = s6.predict_mc(model, np.array([w_cur_m]), p_cur['features'][None, :],
                                                feat_mean, feat_std, n_mc=n_mc,
                                                is_forcing_aware=is_fa, target_peak=0.8 if is_fa else None,
                                                out_norm=out_norm)
        w_m = 1.0 / (std_cur[0] ** 2 + bm['std'] ** 2 + EPS_STD)
        weights.append(float(w_m)); dev_sqs.append((float(amp_cur[0]) - bm['amp']) ** 2); names.append(f'single{m}')
        amps_ref.append(bm['amp'])

    return np.array(weights), np.array(dev_sqs), names, np.array(amps_ref)


def score(weights, dev_sqs):
    return float(np.sqrt(np.sum(weights * dev_sqs) / np.sum(weights)))


if __name__ == '__main__':
    inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
    traj = s8.build_damage_trajectory(df_all, record_checks=False)
    severity = traj['severity']
    T = len(severity)
    ref_pred_zero = s8.compute_baseline_predictions(np.zeros(s8.NB), inp, models, pairs, chain)

    print("=== (1) n_mc sweep on true_vs_zero (the weak, as-implemented classifier path) ===")
    for n_mc in [40, 100, 200]:
        torch.manual_seed(s8.CONFIG['random_seed'] + 99_000 + n_mc)
        scores = []
        for t in range(T):
            w, d_, names, amps_ref = all_terms_nmc(traj['df_traj'][t], ref_pred_zero, inp, models, pairs, chain,
                                                     n_mc=n_mc)
            scores.append(score(w, d_))
        c = float(np.corrcoef(severity, scores)[0, 1])
        print(f"  n_mc={n_mc:4d}: corr(severity, true_vs_zero score) = {c:.4f}   "
              f"scores[0]={scores[0]:.6f} scores[-1]={scores[-1]:.6f}")

    print("\n=== (2) relative variance floor sweep on true_vs_zero (post-hoc on n_mc=40 terms) ===")
    torch.manual_seed(s8.CONFIG['random_seed'] + 99_040)
    cached = []
    for t in range(T):
        w, d_, names, amps_ref = all_terms_nmc(traj['df_traj'][t], ref_pred_zero, inp, models, pairs, chain, n_mc=40)
        cached.append((w, d_, names, amps_ref))

    # weight = 1/(combined_std^2+eps) where combined_std^2 = 1/w_raw - EPS_STD (recover approx std^2)
    # apply a floor: combined_std^2_floored = max(combined_std^2, (floor_frac*amp_ref)^2)
    for floor_frac in [0.0, 0.01, 0.02, 0.05, 0.10]:
        scores = []
        for (w, d_, names, amps_ref) in cached:
            combined_var = 1.0 / w  # approx (EPS_STD negligible at this scale)
            floor_var = (floor_frac * np.maximum(amps_ref, 1e-9)) ** 2
            combined_var_floored = np.maximum(combined_var, floor_var)
            w_floored = 1.0 / combined_var_floored
            scores.append(score(w_floored, d_))
        c = float(np.corrcoef(severity, scores)[0, 1])
        print(f"  floor_frac={floor_frac:.2f}: corr = {c:.4f}  scores[0]={scores[0]:.6f} scores[-1]={scores[-1]:.6f}")

    print("\nDONE")
