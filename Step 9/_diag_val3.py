import sys, json
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step4 as s4

with open(r"Step 9\output\validation3_real_ansys_health_id.json") as f:
    r = json.load(f)

freqs_full = np.array(r['freqs_full_matched'])
freqs_tuned = np.array(r['freqs_tuned'])
df_real_obs = (freqs_full - freqs_tuned) / freqs_tuned   # observed fractional freq shift per mode, real ANSYS
df_true_model = np.array(r['df_true'])  # model-predicted df (participation-weighted), all 24 modes

print("Observed real df/f (24 modes):")
print(np.array2string(df_real_obs, precision=5, suppress_small=True))
print("\nModel-predicted df/f (all 24 modes) from injected blade 10:")
print(np.array2string(df_true_model, precision=5, suppress_small=True))
print(f"\ncorr(observed, model-predicted) = {np.corrcoef(df_real_obs, df_true_model)[0,1]:.4f}")

inp = s4.load_inputs()
P = inp['theta']  # not P; need participation
nlrom = np.load(r"Step 4\output\nonlinear_rom.npz")
P = nlrom['participation']  # (24 blades, 70 modes)
P1b = P[:, :24]  # (24 blades, 24 modes) restricted to 1B cluster

print("\nCorrelation of observed real df pattern against EACH blade's participation column (candidate fit quality):")
corrs = []
for b in range(24):
    c = np.corrcoef(df_real_obs, P1b[b, :])[0,1]
    corrs.append(c)
corrs = np.array(corrs)
order = np.argsort(-np.abs(corrs))
for b in order[:8]:
    print(f"  blade {b:2d}: corr={corrs[b]:+.4f}")
print(f"\n  True blade (10) corr: {corrs[10]:+.4f}   rank among 24: {list(order).index(10)+1}")
print(f"  Localized blade (21) corr: {corrs[21]:+.4f}   rank among 24: {list(order).index(21)+1}")
