import sys, time
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9
import numpy as np

F_gen_190N = 190.80 * 28.3852

t0 = time.time()
print("=== SPEED TEST: w=1.6, F0~190.8N, 20 cycles, lnsrch+pred+relaxed cnvtol+10 cores ===", flush=True)
r = s9.run_case3_transient_point(mode_index=0, w=1.6, force_scale=F_gen_190N,
                                  n_cycles=20, steps_per_cycle=15)
elapsed = time.time() - t0
print(f"ELAPSED: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
print(f"n points retrieved: {len(r['t'])}", flush=True)
print(f"t range: {r['t'].min():.5f} to {r['t'].max():.5f}", flush=True)
print(f"u range: {r['u'].min():.5f} to {r['u'].max():.5f}", flush=True)
np.savez(r'F:\ANSYS PCE\ROM_data_sensitivity\case3_transient\speedtest_20cyc.npz',
         t=r['t'], u=r['u'], elapsed=elapsed, phi_t=r['phi_t'], q_ref=r['q_ref'], Omega=r['Omega'])
print("DONE", flush=True)
