"""Generalization check (2026-08-27): twist_coeff_per_deg/le_te_coeff_per_frac
were measured at blade 0 only (Section 9a of PROJECT_STATUS.md), then
applied to all 24 blades assuming cyclic symmetry -- an assumption never
independently checked. This reruns the SAME single-blade sensitivity
measurement (Step 9's own sensitivity_calibrate, unchanged) at blade 12
(diametrically opposite blade 0 on the 24-blade ring) with the IDENTICAL
magnitudes used at blade 0, to see whether the fitted coefficient is
consistent."""
import sys
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

BLADE = 12
print("=== Multi-blade sensitivity generalization check (blade 12 vs blade 0) ===", flush=True)

print("\n--- d_twist_deg ---", flush=True)
res_twist = s9.sensitivity_calibrate('d_twist_deg', blade_idx=BLADE, magnitudes=(0.15, 0.35))

print("\n--- d_le_te ---", flush=True)
res_lete = s9.sensitivity_calibrate('d_le_te', blade_idx=BLADE, magnitudes=(0.05, 0.12))

import numpy as np
mean_twist = np.mean([f['coeff'] for f in res_twist])
mean_lete = np.mean([f['coeff'] for f in res_lete])
print("\n=== SUMMARY ===")
print(f"twist_coeff_per_deg: blade 0 = -1.276e-05 (documented) vs blade {BLADE} = {mean_twist:.4e}")
print(f"le_te_coeff_per_frac: blade 0 = -0.00877 (documented) vs blade {BLADE} = {mean_lete:.5f}")
print("DONE", flush=True)
