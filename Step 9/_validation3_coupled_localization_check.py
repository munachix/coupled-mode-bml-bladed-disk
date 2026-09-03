"""Diagnostic (2026-08-27, no new ANSYS needed -- reuses Validation 3's
already-saved real data): sparse_localize_blade() in step7.py fits each
candidate blade via forward_freqs_1b(), which uses the DIAGONAL-ONLY
mistuning shortcut (scale @ P[:, :N1B]) -- the SAME simplification
Section 8e/9d already found causes ~5x frequency error and MAC~0.4
against real ANSYS for the FORWARD (ROM-vs-ANSYS) problem, fixed there by
switching to assemble_dK_sec_coupled (the real FMM off-diagonal model).
That fix was never propagated into Step 7's inversion forward model.
This script tests directly whether swapping in the coupled model fixes
Validation 3's real localization miss (true blade 10, diagonal-only
localized to blade 21, ring-distance 11)."""
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

print("=== Coupled-model sparse localization check (real Validation 3 data) ===")
residuals = np.full(NB, np.inf)
best_sev = np.zeros(NB)
for b in range(NB):
    def resid(s):
        df = np.zeros(NB); df[b] = s
        fp = freqs_coupled(df)
        return float(np.sum((y_real - fp) ** 2))
    res = minimize_scalar(resid, bounds=(-0.5, 0.1), method='bounded',
                           options={'xatol': 1e-5})
    residuals[b] = res.fun
    best_sev[b] = res.x
    print(f"  blade {b:2d}: severity={res.x:+.4f}  residual={res.fun:.4f}")

order = np.argsort(residuals)
best_blade = int(order[0])
margin = residuals[order[1]] - residuals[order[0]]
ring_dist = min(abs(best_blade - TRUE_BLADE), NB - abs(best_blade - TRUE_BLADE))
print(f"\nBest-fit blade (coupled model): {best_blade}  (severity={best_sev[best_blade]:+.4f}, residual={residuals[best_blade]:.4f})")
print(f"True blade: {TRUE_BLADE}   ring-distance: {ring_dist}   margin: {margin:.4f}")
print(f"True blade's own rank: {list(order).index(TRUE_BLADE)+1} of {NB}  (residual={residuals[TRUE_BLADE]:.4f})")
print("CORRECT" if ring_dist <= 2 else "STILL INCORRECT")
