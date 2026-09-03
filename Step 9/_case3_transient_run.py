import sys, time
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9
import numpy as np

# phi_t at node 1171 (mode 0) = 28.3852 mm, so F_gen = F0_physical * phi_t
# target_peak=0.4 -> F0_physical=190.80N (matches the user's own Run 3, 190N)
F_gen_190N = 190.80 * 28.3852

t0 = time.time()
print(f"=== RUN A: w=1.6 (~468.5 Hz), F0~190.8N physical, 400 cycles ===", flush=True)
rA = s9.run_case3_transient_point(mode_index=0, w=1.6, force_scale=F_gen_190N,
                                   n_cycles=400, steps_per_cycle=15)
print(f"RUN A done in {time.time()-t0:.1f}s", flush=True)
np.savez(r'F:\ANSYS PCE\ROM_data_sensitivity\case3_transient\runA_190N.npz',
         t=rA['t'], u=rA['u'], phi_t=rA['phi_t'], q_ref=rA['q_ref'],
         Omega=rA['Omega'])
print("Saved runA_190N.npz", flush=True)
