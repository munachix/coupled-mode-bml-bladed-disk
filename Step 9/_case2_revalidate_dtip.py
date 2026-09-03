"""Revalidates Case 2 (mistuned linear, real ANSYS) under the new d_tip-only
mistuning model (2026-08-27 scope change) -- this uses a REAL Step 3 sample
(smooth, spatially-correlated KL field, typical manufacturing-scale
magnitudes), NOT a sparse single-blade injected defect like Validation 3.
Checks whether the refit tip_coeff_per_frac + coupled dK_sec model still
gives good agreement for the pipeline's PRIMARY use case."""
import sys
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

print("=== Case 2 revalidation under d_tip-only mistuning ===")
s9.run_case2_extraction()
s9.case2_comparison()
print("DONE")
