"""Revalidates Case 4 (BPINN-reconstructed geometry) under the new
d_tip-only mistuning model (2026-08-27/28 scope change). The code
(case4_df_to_dlength, kept name for compat, now does d_tip math) was
already updated, but the actual real ANSYS extraction had never been
rerun since 2026-08-09 -- stale, from before the entire pivot."""
import sys
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

print("=== Case 4 revalidation under d_tip-only mistuning ===")
s9.run_case4_extraction()
s9.case4_comparison()
print("DONE")
