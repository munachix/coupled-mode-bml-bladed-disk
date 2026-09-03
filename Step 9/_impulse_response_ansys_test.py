# -*- coding: utf-8 -*-
"""
Real ANSYS impulse-response methodology test (2026-09-02), matching scale
to the just-completed real step-response test (F_gen=5000 generalized,
node 1171 UZ). No force after t=0 -- a distributed, physically smooth
initial velocity field (real point impulse J projected onto ALL 70
secondary modes via their own real participation at the target DOF,
v_i(0) = J*phi_i/M_i, mass-normalized so M_i=1), NOT a single-point
velocity kick: this project's own step9.build_case3_ic_inp docstring
records that a single-point velocity IC was tried once and diverged,
root-caused to the physically harsh "one point yanking on a stationary
structure" discontinuity -- a multi-mode distributed field avoids that by
construction, every DOF starts moving together consistent with the real
combined mode shape.
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

t0 = time.time()
print("=== ANSYS impulse-response methodology test (small-scale) ===", flush=True)

inp = s4.load_inputs()
MODE_INDEX = 0
target_node, target_dir = s9._target_dof_for_mode(inp, MODE_INDEX)
dirmap = {'X': 0, 'Y': 1, 'Z': 2}
dmap = s2._dof_map()
target_eq = np.where((dmap[:, 0] == target_node) & (dmap[:, 1] == dirmap[target_dir]))[0][0]
phi_row = inp['T_full2sec'][target_eq, :]

K0 = float(inp['K_sec'][0, 0])
M0 = float(inp['M_sec'][0, 0])
omega0 = np.sqrt(K0 / M0)
period = 2 * np.pi / omega0

# Same real physical force level as the just-completed step test
# (F_gen=5000 generalized on mode 0 -> F_physical=176.15 N), treated as
# applied for a short reference duration (a quarter of mode 0's own
# period, a physically reasonable "hammer tap" timescale) to define a
# real point impulse J = F*dt, then projected onto all 70 modes.
F_physical_ref = 5000.0 / phi_row[0]
dt_impulse = period / 4.0
J_physical = F_physical_ref * dt_impulse
v_i = J_physical * phi_row / (np.diag(inp['M_sec']))  # mass-normalized, M_i=1
print(f"  F_physical_ref={F_physical_ref:.4f} N, dt_impulse={dt_impulse*1000:.4f} ms, "
      f"J_physical={J_physical:.6f} N*s", flush=True)
print(f"  mode 0 v_i(0)={v_i[0]:.4f} mm/s (generalized)", flush=True)

N_CYCLES = 8
STEPS_PER_CYCLE = 30
t_end = N_CYCLES * period
dt = period / STEPS_PER_CYCLE
n_time = N_CYCLES * STEPS_PER_CYCLE

case_dir = os.path.join(r'F:\ANSYS PCE\ROM_data_sensitivity', 'case_impulse_test')
os.makedirs(case_dir, exist_ok=True)

# ---- multi-mode distributed velocity IC (real, all 70 modes) ----
mode_shapes_all = inp['T_full2sec']  # (n_full_eq, 70)
dmap_node = dmap[:, 0].astype(int)
dmap_dir = dmap[:, 1].astype(int)
labels = np.array(['UX', 'UY', 'UZ'])
v_field = mode_shapes_all @ v_i  # (n_full_eq,) real distributed physical velocity field
thresh = 1e-4 * np.abs(v_field).max()
mask = np.abs(v_field) > thresh
ic_path = os.path.join(case_dir, 'ic_field_impulse.inp')
lines = [f'IC,{n},{labels[d]},0.0,{vv:.8e}' for n, d, vv, m in
         zip(dmap_node, dmap_dir, v_field, mask) if m]
with open(ic_path, 'w') as f_:
    f_.write('\n'.join(lines) + '\n')
print(f"  Distributed multi-mode IC ({len(lines)} node/DOF commands) written -> {ic_path}", flush=True)

zeta0 = float(inp['C_sec'][0, 0]) / (2 * np.sqrt(K0 * M0))
betad = 2 * zeta0 / omega0

mapdl = None
try:
    mapdl = s1.launch_mapdl()
    mapdl = s1.setup_model(mapdl)
    print(f"  Model ready at t={time.time()-t0:.1f}s", flush=True)

    mapdl.slashsolu()
    mapdl.antype('TRANS')
    mapdl.nlgeom('ON')
    mapdl.timint('ON')
    mapdl.betad(betad)
    mapdl.input(ic_path)
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

out_path = os.path.join(case_dir, 'impulse_test_result.npz')
np.savez(out_path, t=t_real, u=u_real, J_physical=J_physical, v_i=v_i, phi_row=phi_row,
         target_node=target_node, target_dir=target_dir)
print(f"  Saved: {out_path}", flush=True)
print(f"=== DONE in {time.time()-t0:.1f}s total ===", flush=True)
