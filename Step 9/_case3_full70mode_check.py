"""THE REAL FIX (2026-08-21): Case 3's dynamic validation compared real ANSYS
(which naturally includes ALL 70 modes' response to a real point force)
against a ROM/BPINN prediction using only 2 modes (0,1) -- a severe,
previously-unrecognized truncation, not a physics bug. Root-caused this
session:
  - Static force-driven check: real ANSYS static NLGEOM = 0.212mm; 2-mode
    ROM static = 0.021mm (10x too small); full 70-mode LINEAR static
    compliance = 0.196mm (8% of real, matches the earlier point-load
    compliance check). Confirms the 2-mode truncation is the problem, not
    nonlinearity or dynamics.
  - Dynamic estimate: nonlinear (0,1) resonant response (0.553mm) + linear
    dynamic FRF from modes 2-69 (properly phase-combined, 1.165mm) =
    1.717mm -- OVERSHOOTS real ANSYS (1.222mm) by ~40%, because the modes
    2-69 correction was treated as purely LINEAR, missing the real
    hardening nonlinearity that would suppress amplitude at this larger
    total motion.

This script runs the REAL fix: the full nonlinear coupled dynamic response
across ALL 70 modes at once, using the REAL point force decomposed onto
every mode (Fg_m = F_physical * Phi_m(node1171)) and the REAL measured
K3/cross-coupling data for every mode and every measured pair/chain (49
pairs total: 17 in the 1B cluster + 32 in the HF band, all measured this
session and the one before). No new ANSYS work -- this uses ONLY data
already on disk, reusing s4.duffing_forced_response_chain() directly: pass
ALL 70 modes in index order as `chain_modes` and the FULL real
cross_coupling dict -- the function's own adjacency filter
(abs(idx_of[mi]-idx_of[mj])==1) naturally recovers the correct topology,
since every real measured pair happens to be index-adjacent in a 0..69
ordering (checked, not assumed: (0,1),(3,4),...,(44,45) are all consecutive
pairs; the two chains are contiguous ranges by construction) -- no changes
needed to step4.py's own solver.

Cross-nonlinear-coupling between DIFFERENT groups (e.g. mode 1 and mode 3,
or between the 1B cluster and the HF band) was never measured and is
treated as exactly zero -- a disclosed limitation matching what was
actually identified, not an assumption papered over.
"""
import sys, time
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
import step4 as s4
import step2 as s2

t0 = time.time()
bundle = np.load(r'F:\ANSYS PCE\ROM_data\secondary_bundle.npz')
K_sec, M_sec, C_sec = bundle['K_sec'], bundle['M_sec'], bundle['C_sec']
K_diag = np.diag(K_sec); M_diag = np.diag(M_sec); C_diag = np.diag(C_sec)
T_full2sec = np.load(r'F:\ANSYS PCE\ROM_data\T_full2sec.npy')
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == 1171) & (dmap[:, 1] == 2))[0][0]
Phi_all = T_full2sec[target_eq, :]   # (70,) real participation of every mode at node 1171 UZ
Phi_0 = Phi_all[0]
F_physical = 2500.0 / Phi_0
omega0_mode0 = 2 * np.pi * 292.818
Omega = omega0_mode0

n_modes = 70
chain_modes = list(range(n_modes))
F_gen_arr = F_physical * Phi_all   # real point-force decomposition onto EVERY mode
pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']   # all 49 real measured pairs (17 1B + 32 HF)

print(f"F_physical = {F_physical:.4f} N, Omega = {Omega:.4f} rad/s ({Omega/2/np.pi:.3f} Hz)")
print(f"n_modes = {n_modes}, n_real_coupling_pairs = {len(pair_coefs)}")
print(f"F_gen_arr range: [{F_gen_arr.min():.4f}, {F_gen_arr.max():.4f}]  (mode 0 = {F_gen_arr[0]:.4f}, should be 2500)")

# HF modes reach up to ~1670Hz vs the 292.82Hz drive -- steps_per_cycle is
# defined relative to the DRIVING frequency in duffing_forced_response_chain,
# so it must be scaled up enough to still resolve the FASTEST mode's own
# oscillation, not just the drive's.
freq_ratio = 1670.91 / 292.818
steps_per_cycle = int(20 * freq_ratio * 1.5)   # safety margin beyond the minimum needed
print(f"HF/drive frequency ratio = {freq_ratio:.2f}x -> steps_per_cycle = {steps_per_cycle}")

n_cycles = 300
print(f"\nRunning full {n_modes}-mode coupled nonlinear ODE solve "
      f"({n_cycles} cycles, {steps_per_cycle} steps/cycle = {n_cycles*steps_per_cycle} total steps)...", flush=True)

r = s4.duffing_forced_response_chain(chain_modes, K_diag, M_diag, C_diag, pair_coefs,
                                       F_gen_arr, Omega, n_cycles=n_cycles, steps_per_cycle=steps_per_cycle)
print(f"Solve done in {time.time()-t0:.1f}s", flush=True)

amp = r['amp']       # (70,) settled amplitude per mode
alpha = r['alpha']; beta = r['beta']
print(f"\nAmplitude range across all 70 modes: [{amp.min():.6e}, {amp.max():.6e}]")
print(f"Finite check: {np.all(np.isfinite(amp))}")
print(f"Top 5 contributing modes by |amp*Phi|:")
contrib = np.abs(amp * Phi_all)
order = np.argsort(contrib)[::-1]
for m in order[:5]:
    print(f"  mode {m}: amp={amp[m]:.6e}, Phi={Phi_all[m]:.4f}, contribution={contrib[m]:.4f} mm")

# physical displacement at node 1171 = phase-correct sum over all modes
u_complex = np.sum((alpha - 1j * beta) * Phi_all)
u_total = abs(u_complex)

print(f"\n{'='*70}")
print(f"FULL 70-MODE NONLINEAR DYNAMIC SOLVE: RESULT")
print(f"{'='*70}")
print(f"  u(node 1171, UZ), full 70-mode nonlinear coupled solve: {u_total:.4f} mm")
print(f"  Real ANSYS (converged 100-cycle dynamic transient):     1.2220 mm +/- 0.0190 mm")
print(f"  Ratio (real/this estimate): {1.2220/u_total:.4f}x")
print(f"  --- for comparison ---")
print(f"  Original 2-mode-only nonlinear estimate: 0.556 mm  (ratio 2.20x)")
print(f"  Linear-superposition estimate (previous step): 1.717 mm  (ratio 0.71x)")
print(f"{'='*70}")

np.savez(r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9\output\case3_full70mode_check.npz',
          u_total=u_total, amp=amp, alpha=alpha, beta=beta, Phi_all=Phi_all,
          F_gen_arr=F_gen_arr, Omega=Omega, real_ansys=1.2220, real_ansys_std=0.0190)
print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
