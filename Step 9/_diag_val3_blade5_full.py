import sys, json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import eigh
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step4 as s4

with open(r"Step 9\output\validation3_real_ansys_health_id.json") as f:
    r = json.load(f)
y_real = np.array(r['freqs_full_matched'])
TRUE_BLADE = r['damaged_blade_true']
NB = 24
inp = s4.load_inputs()
K_sec = inp['K_sec']; M_sec = inp['M_sec']

def freqs_coupled(df_vec):
    dK = s4.assemble_dK_sec_coupled(df_vec, inp, K_sec)
    w, _ = eigh(K_sec + dK, M_sec)
    return np.sqrt(np.clip(w[:24], 0, None)) / (2 * np.pi)

residuals = np.full(NB, np.inf)
for b in range(NB):
    def resid(s):
        df = np.zeros(NB); df[b] = s
        return float(np.sum((y_real - freqs_coupled(df)) ** 2))
    res = minimize_scalar(resid, bounds=(-0.5, 0.1), method='bounded', options={'xatol': 1e-5})
    residuals[b] = res.fun
order = np.argsort(residuals)
for b in order:
    marker = " <-- TRUE" if b == TRUE_BLADE else ""
    print(f"  blade {b:2d}: residual={residuals[b]:.5f}{marker}")
print(f"\nTrue blade rank: {list(order).index(TRUE_BLADE)+1} of 24")
print(f"Margin (2nd best - best): {residuals[order[1]]-residuals[order[0]]:.5f}")
print(f"Residual spread: min={residuals.min():.5f} max={residuals.max():.5f} ratio={residuals.max()/residuals.min():.2f}x")
