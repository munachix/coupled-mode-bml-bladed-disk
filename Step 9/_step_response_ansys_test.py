# -*- coding: utf-8 -*-
"""
Methodology-validation test for a real ANSYS STEP-response transient
(2026-09-02, explicit user request: "Fig 30 should use ansys to verify").
This project's existing real transient pipeline (step9.run_case3_transient_
point) has only ever been exercised with smooth harmonic (cosine) forcing;
a sudden step onset is new territory for the solver's convergence settings,
so this is a small, cheap sanity run FIRST -- modest force level (close to
the already-proven ~2500-5412 generalized-force range used elsewhere in
this project's real transient campaign), short duration -- to confirm the
step-loading methodology converges and to measure real wall-clock cost
before committing to the full-scale run at the paper's actual q_ref=0.7 mm
target (which requires a far larger force, ~2.37e6 generalized, since a
STATIC/step force must overcome the full real stiffness K directly rather
than being amplified ~250x by resonance the way the harmonic runs are).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 1')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step1 as s1
import step2 as s2
import step4 as s4
import step9 as s9

MODE_INDEX = 0
F_GEN_TEST = 5000.0   # generalized force, close to this project's already-proven range
N_CYCLES = 8
STEPS_PER_CYCLE = 30

t0 = time.time()
print("=== ANSYS step-response methodology test (small-scale) ===", flush=True)

inp = s4.load_inputs()
target_node, target_dir = s9._target_dof_for_mode(inp, MODE_INDEX)
Phi_target = inp['T_full2sec'][:, MODE_INDEX]
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == target_node) & (dmap[:, 1] == {'X': 0, 'Y': 1, 'Z': 2}[target_dir]))[0][0]
phi_t = Phi_target[target_eq]
F0_physical = F_GEN_TEST / phi_t

K0 = float(inp['K_sec'][0, 0])
M0 = float(inp['M_sec'][0, 0])
omega0 = np.sqrt(K0 / M0)
period = 2 * np.pi / omega0
t_end = N_CYCLES * period
dt = period / STEPS_PER_CYCLE
n_time = N_CYCLES * STEPS_PER_CYCLE
t_arr = np.linspace(0, t_end, n_time + 1)

# Step profile: 0 for the first table point, then F0 held constant from the
# second point onward -- a fast ramp within one substep (dt), not a
# literal mathematical discontinuity, since ANSYS interpolates linearly
# between *DIM table entries regardless of KBC for this array-parameter
# load mechanism.
f_arr = np.full(n_time + 1, F0_physical)
f_arr[0] = 0.0

print(f"  Target DOF: node {target_node}, U{target_dir} (Phi={phi_t:.4f})")
print(f"  F0_physical = {F0_physical:.4f} N (generalized {F_GEN_TEST:.1f}), step onset over dt={dt*1000:.4f} ms")
print(f"  {N_CYCLES} cycles, {n_time} substeps, t_end={t_end*1000:.3f} ms", flush=True)

case_dir = os.path.join(r'F:\ANSYS PCE\ROM_data_sensitivity', 'case_step_test')
os.makedirs(case_dir, exist_ok=True)
tab_path = os.path.join(case_dir, 'force_table_step.inp')
lines = ['*DIM,ftab,TABLE,%d,1,1,TIME' % (n_time + 1)]
for i, (ti, fi) in enumerate(zip(t_arr, f_arr), start=1):
    lines.append(f'*SET,ftab({i},0),{ti:.8e}')
    lines.append(f'*SET,ftab({i},1),{fi:.8e}')
with open(tab_path, 'w') as f_:
    f_.write('\n'.join(lines) + '\n')
print(f"  Force table written -> {tab_path}", flush=True)

zeta = float(inp['C_sec'][0, 0]) / (2 * np.sqrt(K0 * M0))
betad = 2 * zeta / omega0

mapdl = None
try:
    mapdl = s1.launch_mapdl()
    mapdl = s1.setup_model(mapdl)
    print(f"  Model ready at t={time.time()-t0:.1f}s", flush=True)

    mapdl.prep7()
    mapdl.input(tab_path)

    mapdl.slashsolu()
    mapdl.antype('TRANS')
    mapdl.nlgeom('ON')
    mapdl.timint('ON')
    mapdl.betad(betad)
    mapdl.f(target_node, f'F{target_dir}', '%ftab%')
    mapdl.time(t_end)
    mapdl.deltim(dt, dt / 20, dt)
    mapdl.autots('ON')
    mapdl.nropt('FULL')
    mapdl.lnsrch('ON')
    mapdl.pred('ON')
    mapdl.cnvtol('U', '', 0.02, 2, '')
    mapdl.kbc(0)
    mapdl.outres('NSOL', 'ALL')
    print(f"  Solving... (start t={time.time()-t0:.1f}s)", flush=True)
    mapdl.solve()
    mapdl.finish()
    print(f"  Solve done at t={time.time()-t0:.1f}s", flush=True)

    mapdl.post1()
    n_sets = int(mapdl.get_value('ACTIVE', 0, 'SET', 'NSET'))
    print(f"  {n_sets} result sets stored -> polling via POST1", flush=True)
    t_list, u_list = [], []
    for i in range(1, n_sets + 1):
        mapdl.set(1, i)
        tv = float(mapdl.get_value('ACTIVE', 0, 'SET', 'TIME'))
        uv = mapdl.get_value('NODE', target_node, 'U', target_dir)
        t_list.append(tv)
        u_list.append(uv)
    mapdl.finish()
    t_real = np.array(t_list)
    u_real = np.array(u_list)
    print(f"  Retrieved {len(t_real)} time-history points", flush=True)
finally:
    if mapdl:
        try:
            mapdl.exit(force=True)
        except Exception:
            pass

out_path = os.path.join(case_dir, 'step_test_result.npz')
np.savez(out_path, t=t_real, u=u_real, F0=F0_physical, F_gen=F_GEN_TEST, phi_t=phi_t,
         target_node=target_node, target_dir=target_dir)
print(f"  Saved: {out_path}", flush=True)
print(f"=== DONE in {time.time()-t0:.1f}s total ===", flush=True)
