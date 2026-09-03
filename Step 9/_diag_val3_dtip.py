import sys, json
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step4 as s4

with open(r"Step 9\output\validation3_real_ansys_health_id.json") as f:
    r = json.load(f)
y_real = np.array(r['freqs_full_matched'])
freqs_tuned = np.array(r['freqs_tuned'])
df_real_obs = (y_real - freqs_tuned) / freqs_tuned
TRUE_BLADE = r['damaged_blade_true']

inp = s4.load_inputs()
nlrom = np.load(r"Step 4\output\nonlinear_rom.npz")
P = nlrom['participation']
P1b = P[:, :24]

print("Correlation of REAL observed df pattern against each blade's diagonal participation column:")
corrs = []
for b in range(24):
    c = np.corrcoef(df_real_obs, P1b[b, :])[0,1]
    corrs.append(c)
corrs = np.array(corrs)
order = np.argsort(-np.abs(corrs))
for b in order[:8]:
    print(f"  blade {b:2d}: corr={corrs[b]:+.4f}")
print(f"\n  True blade ({TRUE_BLADE}) corr: {corrs[TRUE_BLADE]:+.4f}   rank: {list(order).index(TRUE_BLADE)+1} of 24")
print(f"  Diagonal-localized blade (4) corr: {corrs[4]:+.4f}   rank: {list(order).index(4)+1}")
print(f"  Coupled-localized blade (2) corr:  {corrs[2]:+.4f}   rank: {list(order).index(2)+1}")

print(f"\nObserved df pattern (24 modes):\n{np.array2string(df_real_obs, precision=5, suppress_small=True)}")
print(f"\nP[{TRUE_BLADE},:24] (true blade's own participation):\n{np.array2string(P1b[TRUE_BLADE], precision=5, suppress_small=True)}")
