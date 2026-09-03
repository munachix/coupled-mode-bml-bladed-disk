"""Decisive diagnostic (2026-08-21, explicit user request): does the reduced
2-mode nonlinear model get the STATICS right under real force control, or is
the ~2.2x Case 3 dynamic gap already present with damping/dynamics removed?

Applies the REAL physical point force (88.07N, same magnitude used in the
real dynamic transient, F0_physical = force_scale/Phi_0 = 2500/28.3852) at
node 1171 UZ, NLGEOM ON, STATIC solve -- no prescribed mode-shape
displacement (unlike the existing K3 identification points), no dynamics,
no damping. Compares against the reduced 2-mode ROM's own static
equilibrium at the identical force (solved directly, not via the dynamic
ODE integrator -- a real nonlinear algebraic solve, K*q + F_nl(q) = F_gen).

This isolates whether the gap is a REDUCED-BASIS/static-nonlinearity issue
(would show up here too) or something genuinely DYNAMIC (would NOT show up
here, since this test has no dynamics at all).
"""
import sys, os, time
import numpy as np
from scipy.optimize import fsolve
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 1')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step1 as s1
import step2 as s2
import step4 as s4
import step9 as s9

OUT = s9.OUT
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
inp = s4.load_inputs()
target_node, target_dir = s9._target_dof_for_mode(inp, 0)
Phi_0 = float(inp['T_full2sec'][:, 0][
    np.where((s2._dof_map()[:, 0] == target_node) &
              (s2._dof_map()[:, 1] == {'X': 0, 'Y': 1, 'Z': 2}[target_dir]))[0][0]])
Phi_1 = float(inp['T_full2sec'][:, 1][
    np.where((s2._dof_map()[:, 0] == target_node) &
              (s2._dof_map()[:, 1] == {'X': 0, 'Y': 1, 'Z': 2}[target_dir]))[0][0]])
F0_physical = 2500.0 / Phi_0
print(f"Target DOF: node {target_node}, U{target_dir}, Phi_0={Phi_0:.4f}, Phi_1={Phi_1:.4f}")
print(f"Real physical point force: {F0_physical:.4f} N", flush=True)

# ---- Part A: real ANSYS static NLGEOM solve, real point force ----
mapdl = None
try:
    mapdl = s1.launch_mapdl()
    mapdl = s1.setup_model(mapdl)

    mapdl.slashsolu()
    mapdl.antype('STATIC')
    mapdl.nlgeom('ON')
    mapdl.nsubst(20, 100, 5)   # finer substepping than the identification points -- this is a single real point we need right
    mapdl.autots('ON')
    mapdl.outres('ALL', 'ALL')
    mapdl.f(target_node, f'F{target_dir}', F0_physical)
    mapdl.solve()
    mapdl.finish()

    mapdl.post1()
    mapdl.set('LAST')
    u_real_static = mapdl.get_value('NODE', target_node, 'U', target_dir)
    print(f"\nREAL ANSYS static NLGEOM displacement at node {target_node} U{target_dir}: {u_real_static:.6f} mm", flush=True)
finally:
    if mapdl:
        try:
            mapdl.exit(force=True)
        except Exception:
            pass

# ---- Part B: ROM's own static equilibrium at the identical force ----
K0 = inp['K_sec'][0, 0]; K1 = inp['K_sec'][1, 1]
cc = s4.CONFIG['nonlinear']['cross_coupling'][(0, 1)]
Fg_0 = F0_physical * Phi_0
Fg_1 = F0_physical * Phi_1
print(f"\nGeneralized forces (real point-force decomposition): Fg_0={Fg_0:.4f}, Fg_1={Fg_1:.4f}")


def static_residual(q):
    q0, q1 = q
    Fnl_0 = s4.coupled_nonlinear_force(cc['coef0'], q0, q1)
    Fnl_1 = s4.coupled_nonlinear_force(cc['coef1'], q0, q1)
    return [K0 * q0 + Fnl_0 - Fg_0, K1 * q1 + Fnl_1 - Fg_1]


q0_lin = Fg_0 / K0  # linear guess to seed the solve
q1_lin = Fg_1 / K1
q_sol = fsolve(static_residual, [q0_lin, q1_lin], full_output=False)
q0_rom, q1_rom = q_sol
u_rom_static = q0_rom * Phi_0 + q1_rom * Phi_1
print(f"ROM static solve: q0={q0_rom:.6f}, q1={q1_rom:.6f}")
print(f"ROM static-equilibrium displacement at node {target_node} U{target_dir}: {u_rom_static:.6f} mm")

# ---- Also: SDOF-only static equilibrium (mode 0 alone, ignore mode 1 entirely) ----
K3_0 = inp['K3_sec_diag'][0]


def sdof_residual(q0):
    return K0 * q0 + K3_0 * q0 ** 3 - Fg_0


q0_sdof = fsolve(sdof_residual, [q0_lin])[0]
u_sdof_static = q0_sdof * Phi_0
print(f"SDOF-only static-equilibrium displacement: {u_sdof_static:.6f} mm")

print(f"\n{'='*70}")
print("STATIC FORCE-DRIVEN CHECK: SUMMARY")
print(f"{'='*70}")
print(f"  REAL ANSYS static NLGEOM (real point force, no dynamics):  {u_real_static:.4f} mm")
print(f"  ROM static equilibrium, coupled 2-mode (same force):        {u_rom_static:.4f} mm")
print(f"  ROM static equilibrium, SDOF mode-0-only (same force):      {u_sdof_static:.4f} mm")
print(f"  Ratio (real static / ROM static, coupled):  {u_real_static/u_rom_static:.4f}x")
print(f"  Ratio (real static / ROM static, SDOF):     {u_real_static/u_sdof_static:.4f}x")
print(f"  (for comparison) real DYNAMIC / ROM DYNAMIC ratio was: 1.2220/0.5560 = {1.2220/0.5560:.4f}x")
print(f"{'='*70}")

np.savez(os.path.join(OUT, 'case3_static_force_check.npz'),
          target_node=target_node, target_dir=target_dir, Phi_0=Phi_0, Phi_1=Phi_1,
          F0_physical=F0_physical, u_real_static=u_real_static,
          u_rom_static=u_rom_static, u_sdof_static=u_sdof_static,
          q0_rom=q0_rom, q1_rom=q1_rom, q0_sdof=q0_sdof)
print(f"\nSaved: {os.path.join(OUT, 'case3_static_force_check.npz')}")
print(f"Total time: {time.time()-t0:.1f}s")
print("DONE")
