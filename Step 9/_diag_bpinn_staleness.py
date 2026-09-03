import sys, os
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 5')
import step4 as s4, step5 as s5

inp = s4.load_inputs()
df_all = s5.compute_delta_f_vectorized(inp['theta'], s4.CONFIG['sensitivity'], inp['L_ref'], inp['t_ref'])
P = np.load(os.path.join(s4.OUT, 'nonlinear_rom.npz'))['participation']
scale = (1.0 + df_all) ** 2 - 1.0   # (1000, 24)

print("TODAY's actual shift_m distribution per mode (post-pivot, real current data):")
for m in [0,1,2,3,4,5,6,9,10]:
    shift_m = scale @ P[:, m]
    print(f"  mode {m}: shift_m mean={shift_m.mean():.6f} std={shift_m.std():.6f} range=({shift_m.min():.6f},{shift_m.max():.6f})")

print("\nWhat each forcing-aware network's OWN saved norm file says it was trained on (feat_mean[0]/feat_std[0] = shift):")
OUT6 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6\output'
for tag in ['01','34','56','78','910']:
    norm = dict(np.load(os.path.join(OUT6, f'bpinn_forcing_aware_{tag}_norm.npz')))
    print(f"  pair {tag}: shift feat_mean={norm['feat_mean'][0]:.6f} feat_std={norm['feat_std'][0]:.6f}")
norm2 = dict(np.load(os.path.join(OUT6, 'bpinn_forcing_aware_mode2_norm.npz')))
print(f"  mode2: shift feat_mean={norm2['feat_mean'][0]:.6f} feat_std={norm2['feat_std'][0]:.6f}")
normc = dict(np.load(os.path.join(OUT6, 'bpinn_forcing_aware_chain_norm.npz')))
print(f"  chain: feat_mean[:3]={normc['feat_mean'][:3]}")
