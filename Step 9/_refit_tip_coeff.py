"""Refits tip_coeff_per_frac (2026-08-27) with the CORRECTED t_ref
(36.4267mm, was 52.0mm -- see Step 1's measure_blade_geometry fix). The
original fit (Section 8d, -0.92911) used the wrong, contaminated t_ref, so
the coefficient itself is now inconsistent with the corrected normalization
and must be refit, not just carried over."""
import sys
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

print("=== Refitting tip_coeff_per_frac with corrected t_ref ===")
fits = s9.sensitivity_calibrate('d_tip', blade_idx=0, magnitudes=(1.5, 3.0))
print("DONE")
