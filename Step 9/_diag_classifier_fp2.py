import sys
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 8')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
import numpy as np, math
import step8 as s8, step7 as s7

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
zero_pred = s8.compute_baseline_predictions(np.zeros(s8.NB), inp, models, pairs, chain)
baseline_idx = s8.CONFIG['unit_baseline_idx']
df_point = df_all[baseline_idx]

print(f"Per-channel breakdown for sample #{baseline_idx}:")
for pair, (pair_model, pair_norm) in pairs.items():
    bp = zero_pred['pairs'][pair]
    p_cur = s7.coupled_features_from_df(df_point, pair, inp)
    omega0_cur = np.sqrt(p_cur['Ki'] / p_cur['Mi'])
    w_cur = bp['omega0'] / omega0_cur
    ai_cur, si_cur, aj_cur, sj_cur = s7.predict_coupled_mc(pair_model, pair_norm, np.array([w_cur]), p_cur['feat'][None,:], n_mc=40)
    dev_i = abs(float(ai_cur[0]) - bp['ai']); dev_j = abs(float(aj_cur[0]) - bp['aj'])
    print(f"  pair {pair}: dev_i={dev_i:.6f} (si_cur={si_cur[0]:.6f}, si_ref={bp['si']:.6f})  "
          f"dev_j={dev_j:.6f} (sj_cur={sj_cur[0]:.6f}, sj_ref={bp['sj']:.6f})  "
          f"weight_i={1/(si_cur[0]**2+bp['si']**2+1e-7):.1f}  weight_j={1/(sj_cur[0]**2+bp['sj']**2+1e-7):.1f}  "
          f"contrib_i={ (1/(si_cur[0]**2+bp['si']**2+1e-7))*dev_i**2:.6f}  contrib_j={(1/(sj_cur[0]**2+bp['sj']**2+1e-7))*dev_j**2:.6f}")

bc = zero_pred['chain']
p_cur_c = s7.chain_features_from_df(df_point, chain[2], inp)
omega0_cur_c = math.sqrt((p_cur_c['K_arr']/p_cur_c['M_arr']).mean())
w_cur_c = bc['omega0']/omega0_cur_c
amp_cur_c, std_cur_c = s7.predict_chain_mc(chain[0], chain[1], np.array([w_cur_c]), p_cur_c['feat'][None,:], n_mc=40)
dev_c = np.abs(amp_cur_c[0]-bc['amp'])
w_c = 1/(std_cur_c[0]**2+bc['std']**2+1e-7)
print(f"  chain: mean dev={dev_c.mean():.6f}  mean weight={w_c.mean():.1f}  total contrib={np.sum(w_c*dev_c**2):.6f}")
