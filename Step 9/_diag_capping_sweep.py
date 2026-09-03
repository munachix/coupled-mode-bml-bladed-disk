"""
Measure the effect of a robust median-based weight cap (task-suggested
approach #3: clip each channel's weight to a multiple of the median weight
across all 24 mode-channels in that evaluation) on:
  (a) corr(severity, classifier score) for true_vs_zero (the weak case, real
      classify_mistuning path AS CURRENTLY IMPLEMENTED before any input fix)
  (b) corr(severity, HI3) for pm_vs_baseline (HI3 AS-IMPLEMENTED -- must not
      degrade)
across several candidate cap multiples, before picking one.
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


def all_terms(df_point, ref_pred, inp, models, pairs, chain, w_eval=1.0):
    """Same math as compute_HI3, but returns the flat list of (weight, dev_sq)
    terms instead of collapsing to one scalar, so different capping schemes
    can be evaluated post-hoc without rerunning the (expensive) BPINN calls."""
    weights, dev_sqs, names = [], [], []

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
        weights += [float(wi), float(wj)]
        dev_sqs += [(float(ai_cur[0]) - bp['ai']) ** 2, (float(aj_cur[0]) - bp['aj']) ** 2]
        names += [f'pair{pair}_i', f'pair{pair}_j']

    chain_model, chain_norm, chain_modes = chain
    bc = ref_pred['chain']
    p_cur_c = s7.chain_features_from_df(df_point, chain_modes, inp)
    omega0_cur_chain = math.sqrt((p_cur_c['K_arr'] / p_cur_c['M_arr']).mean())
    omega_ref_c = w_eval * bc['omega0']
    w_cur_c = omega_ref_c / omega0_cur_chain
    amp_cur_c, std_cur_c = s7.predict_chain_mc(chain_model, chain_norm, np.array([w_cur_c]),
                                                p_cur_c['feat'][None, :], n_mc=40)
    w_chain = 1.0 / (std_cur_c[0] ** 2 + bc['std'] ** 2 + EPS_STD)
    dev_chain_sq = (amp_cur_c[0] - bc['amp']) ** 2
    for k, (wc, dc) in enumerate(zip(w_chain, dev_chain_sq)):
        weights.append(float(wc)); dev_sqs.append(float(dc)); names.append(f'chain_m{chain_modes[k]}')

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
        weights.append(float(w_m)); dev_sqs.append((float(amp_cur[0]) - bm['amp']) ** 2); names.append(f'single{m}')

    return np.array(weights), np.array(dev_sqs), names


def score_with_cap(weights, dev_sqs, cap_mult=None):
    if cap_mult is None:
        w = weights
    else:
        med = np.median(weights)
        cap = cap_mult * med if med > 0 else np.inf
        w = np.minimum(weights, cap)
    return float(np.sqrt(np.sum(w * dev_sqs) / np.sum(w)))


def chain_frac(weights, dev_sqs, names, cap_mult=None):
    if cap_mult is None:
        w = weights
    else:
        med = np.median(weights)
        cap = cap_mult * med if med > 0 else np.inf
        w = np.minimum(weights, cap)
    contrib = w * dev_sqs
    total = contrib.sum()
    if total <= 0:
        return 0.0
    is_chain = np.array([n.startswith('chain_m') for n in names])
    return float(contrib[is_chain].sum() / total)


if __name__ == '__main__':
    torch.manual_seed(s8.CONFIG['random_seed'])
    inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
    traj = s8.build_damage_trajectory(df_all, record_checks=False)
    d = np.load(os.path.join(s8.OUT, 'damage_trajectory.npz'))
    post_means = d['post_means']
    severity = traj['severity']
    T = len(severity)

    ref_pred_baseline = s8.compute_baseline_predictions(traj['df_baseline'], inp, models, pairs, chain)
    ref_pred_zero = s8.compute_baseline_predictions(np.zeros(s8.NB), inp, models, pairs, chain)

    print("Computing raw per-term weights/devs for all 20 steps x {true,pm} x {baseline,zero}...", flush=True)
    terms_true_zero, terms_true_base, terms_pm_zero, terms_pm_base = [], [], [], []
    for t in range(T):
        df_true_t = traj['df_traj'][t]
        pm_t = post_means[t]
        terms_true_zero.append(all_terms(df_true_t, ref_pred_zero, inp, models, pairs, chain))
        terms_true_base.append(all_terms(df_true_t, ref_pred_baseline, inp, models, pairs, chain))
        terms_pm_zero.append(all_terms(pm_t, ref_pred_zero, inp, models, pairs, chain))
        terms_pm_base.append(all_terms(pm_t, ref_pred_baseline, inp, models, pairs, chain))
        print(f"  t={t} done", flush=True)

    print("\n=== EFFECT OF CAP MULTIPLE ON CORRELATION AND CHAIN-DOMINANCE ===")
    print(f"{'cap_mult':>10s}  {'corr(true,zero)':>16s}  {'corr(pm,base)=HI3':>18s}  "
          f"{'meanChainFrac(true,zero)':>26s}  {'meanChainFrac(pm,base)':>24s}")
    for cap_mult in [None, 100, 50, 20, 10, 5, 3, 2]:
        scores_tz = [score_with_cap(w, d_, cap_mult) for (w, d_, n) in terms_true_zero]
        scores_pb = [score_with_cap(w, d_, cap_mult) for (w, d_, n) in terms_pm_base]
        cf_tz = [chain_frac(w, d_, n, cap_mult) for (w, d_, n) in terms_true_zero]
        cf_pb = [chain_frac(w, d_, n, cap_mult) for (w, d_, n) in terms_pm_base]
        c_tz = float(np.corrcoef(severity, scores_tz)[0, 1])
        c_pb = float(np.corrcoef(severity, scores_pb)[0, 1])
        label = 'none' if cap_mult is None else str(cap_mult)
        print(f"{label:>10s}  {c_tz:16.4f}  {c_pb:18.4f}  {np.mean(cf_tz):26.3f}  {np.mean(cf_pb):24.3f}")

    print("\nDONE")
