"""Quick scale-check: typical amplitude and epistemic-std magnitude for the
chain channel (13 modes) vs. the pair/single channels, across many real
calibration samples, to choose a physically-motivated variance floor for
the chain specifically (not just tune a constant blindly)."""
import sys, os
import numpy as np

_STEP8 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 8'
_STEP7 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7'
_STEP6 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6'
_STEP4 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4'
for p in (_STEP8, _STEP7, _STEP6, _STEP4):
    sys.path.insert(0, p)
import step7 as s7
import step8 as s8

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
zero_pred = s8.compute_baseline_predictions(np.zeros(s8.NB), inp, models, pairs, chain)
chain_model, chain_norm, chain_modes = chain
bc = zero_pred['chain']
print(f"Chain reference amp (df=0): mean={bc['amp'].mean():.6f} mm, "
      f"range=[{bc['amp'].min():.6f},{bc['amp'].max():.6f}]")
print(f"Chain reference std (df=0): mean={bc['std'].mean():.8f} mm, "
      f"range=[{bc['std'].min():.8f},{bc['std'].max():.8f}]")
print(f"  => reference std as fraction of reference amp: mean={np.mean(bc['std']/bc['amp']):.5f}, "
      f"min={np.min(bc['std']/bc['amp']):.6f}")

rng = np.random.default_rng(999)
idxs = rng.choice(df_all.shape[0], size=50, replace=False)
all_std_cur = []
all_amp_cur = []
for idx in idxs:
    df_i = df_all[idx]
    p_cur_c = s7.chain_features_from_df(df_i, chain_modes, inp)
    import math
    omega0_cur_c = math.sqrt((p_cur_c['K_arr'] / p_cur_c['M_arr']).mean())
    w_cur_c = bc['omega0'] / omega0_cur_c
    amp_cur_c, std_cur_c = s7.predict_chain_mc(chain_model, chain_norm, np.array([w_cur_c]),
                                                p_cur_c['feat'][None, :], n_mc=40)
    all_std_cur.append(std_cur_c[0])
    all_amp_cur.append(amp_cur_c[0])
all_std_cur = np.array(all_std_cur)   # (50, 13)
all_amp_cur = np.array(all_amp_cur)

print(f"\nAcross 50 real samples, chain per-mode CURRENT std: "
      f"mean={all_std_cur.mean():.8f}, median={np.median(all_std_cur):.8f}, "
      f"5th pct={np.percentile(all_std_cur,5):.8f}, min={all_std_cur.min():.8f}, max={all_std_cur.max():.8f}")
print(f"Across 50 real samples, chain per-mode CURRENT amp: "
      f"mean={all_amp_cur.mean():.6f}, median={np.median(all_amp_cur):.6f}, "
      f"min={all_amp_cur.min():.6f}, max={all_amp_cur.max():.6f}")
print(f"Ratio std/amp per (sample,mode): mean={np.mean(all_std_cur/np.maximum(all_amp_cur,1e-9)):.5f}, "
      f"median={np.median(all_std_cur/np.maximum(all_amp_cur,1e-9)):.5f}, "
      f"5th pct={np.percentile(all_std_cur/np.maximum(all_amp_cur,1e-9), 5):.6f}, "
      f"min={np.min(all_std_cur/np.maximum(all_amp_cur,1e-9)):.7f}")

# Compare to a pair channel and the single independent mode, same treatment
print("\n--- for comparison: pair (0,1) and single mode 2 ---")
for pair, (pair_model, pair_norm) in pairs.items():
    bp = zero_pred['pairs'][pair]
    stds_i, stds_j, amps_i = [], [], []
    for idx in idxs[:50]:
        df_i = df_all[idx]
        p_cur = s7.coupled_features_from_df(df_i, pair, inp)
        omega0_cur = np.sqrt(p_cur['Ki'] / p_cur['Mi'])
        w_cur = bp['omega0'] / omega0_cur
        ai_cur, si_cur, aj_cur, sj_cur = s7.predict_coupled_mc(
            pair_model, pair_norm, np.array([w_cur]), p_cur['feat'][None, :], n_mc=40)
        stds_i.append(si_cur[0]); stds_j.append(sj_cur[0]); amps_i.append(ai_cur[0])
    stds_i = np.array(stds_i); amps_i = np.array(amps_i)
    print(f"  pair {pair}: std_i median={np.median(stds_i):.8f}, amp_i median={np.median(amps_i):.6f}, "
          f"ratio median={np.median(stds_i/np.maximum(amps_i,1e-9)):.5f}, "
          f"ratio min={np.min(stds_i/np.maximum(amps_i,1e-9)):.7f}")

for m, (model, norm_stats) in models.items():
    feat_mean, feat_std, out_norm = norm_stats
    is_fa = len(feat_mean) == 5
    import step6 as s6
    bm = zero_pred['single'][m]
    stds_m, amps_m = [], []
    for idx in idxs[:50]:
        df_i = df_all[idx]
        p_cur = s7.sdof_params_from_df(df_i, m, inp)
        omega0_cur_m = np.sqrt(p_cur['K'] / p_cur['M'])
        w_cur_m = bm['omega0'] / omega0_cur_m
        amp_cur, std_cur, _, _ = s6.predict_mc(model, np.array([w_cur_m]), p_cur['features'][None, :],
                                                feat_mean, feat_std, n_mc=40,
                                                is_forcing_aware=is_fa, target_peak=0.8 if is_fa else None,
                                                out_norm=out_norm)
        stds_m.append(std_cur[0]); amps_m.append(amp_cur[0])
    stds_m = np.array(stds_m); amps_m = np.array(amps_m)
    print(f"  single mode {m}: std median={np.median(stds_m):.8f}, amp median={np.median(amps_m):.6f}, "
          f"ratio median={np.median(stds_m/np.maximum(amps_m,1e-9)):.5f}, "
          f"ratio min={np.min(stds_m/np.maximum(amps_m,1e-9)):.7f}")

print("\nDONE")
