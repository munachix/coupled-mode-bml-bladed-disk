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
N1B = 24

def freqs_coupled(df_vec):
    dK = s4.assemble_dK_sec_coupled(df_vec, inp, K_sec)
    w, _ = eigh(K_sec + dK, M_sec)
    return np.sqrt(np.clip(w[:N1B], 0, None)) / (2 * np.pi)

print("=== Full 24-candidate coupled localization breakdown (d_tip case) ===")
residuals = np.full(NB, np.inf)
severities = np.zeros(NB)
for b in range(NB):
    def resid(s):
        df = np.zeros(NB); df[b] = s
        fp = freqs_coupled(df)
        return float(np.sum((y_real - fp) ** 2))
    res = minimize_scalar(resid, bounds=(-0.5, 0.1), method='bounded', options={'xatol': 1e-5})
    residuals[b] = res.fun
    severities[b] = res.x
    marker = " <-- TRUE" if b == TRUE_BLADE else ""
    print(f"  blade {b:2d}: severity={res.x:+.4f}  residual={res.fun:.5f}{marker}")

order = np.argsort(residuals)
print(f"\nRanking (best to worst): {list(order)}")
print(f"True blade {TRUE_BLADE} rank: {list(order).index(TRUE_BLADE)+1} of 24, residual={residuals[TRUE_BLADE]:.5f}")
print(f"Best blade {order[0]}: residual={residuals[order[0]]:.5f}, severity={severities[order[0]]:+.4f}")
print(f"True blade's severity if forced: {severities[TRUE_BLADE]:+.4f} (true injected: -0.03)")
