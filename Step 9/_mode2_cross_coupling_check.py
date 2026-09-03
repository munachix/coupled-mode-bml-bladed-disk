"""NONLINEAR CROSS-MODE COUPLING check for mode 2 (2026-08-27) -- both
leading hypotheses for the 190x real-vs-ROM dynamic amplitude gap (static
K3 extrapolation failure; SDOF branch-jump) have been checked and
falsified with real data. This checks the remaining plausible
explanation: mode 2 was established as ISOLATED via a LINEAR frequency-
gap scan (no nearby natural frequency) and its diagonal K3 is now
confirmed accurate to large amplitude -- but neither check ever tested
whether exciting mode 2 ALONE at LARGE amplitude produces a nonlinear
reaction force onto OTHER modes. Small-amplitude cross-coupling tests for
OTHER mode pairs never included mode 2 (it was never paired with anything,
precisely because it looked isolated) -- so real nonlinear energy
transfer from mode 2 into a neighbor, only significant at large
amplitude, has genuinely never been tested.

Method: same real ANSYS NLGEOM static solve as the K3 identification
(prescribe ONLY mode 2's shape at each amplitude), but this time project
the SAME reaction force field onto EVERY 1B-cluster mode (0-23), not just
mode 2 itself. A nonzero, amplitude-scaling projection onto another mode
is direct evidence of real nonlinear cross-mode coupling."""
import sys, os, time
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
import step9 as s9
import step2 as s2

MODE_INDEX = 2
CHECK_MODES = list(range(24))   # project onto every 1B-cluster mode
AMPLITUDES = [0.11, 0.5, 0.9, 1.3]   # smallest (known-good baseline) + 3 large points

t0 = time.time()
print(f"=== Cross-mode coupling check: mode {MODE_INDEX} alone, projected onto modes 0-23 ===", flush=True)
inp = s9.s4.load_inputs()
dmap = s2._dof_map()
node_arr = dmap[:, 0].astype(int)
dir_arr = dmap[:, 1].astype(int)

mapdl = None
results = []
try:
    mapdl = s9.s1.launch_mapdl()
    mapdl = s9.s1.setup_model(mapdl)
    for i, a in enumerate(AMPLITUDES):
        t1 = time.time()
        inp_path = os.path.join(s9.SENSITIVITY_CASE_DIR, 'case3', f'case3_disp_cc{i}.inp')
        os.makedirs(os.path.dirname(inp_path), exist_ok=True)
        n_written = s9.build_case3_displacement_inp(MODE_INDEX, a, inp_path, inp)
        mapdl.prep7()
        mapdl.input(inp_path)
        mapdl.slashsolu()
        mapdl.antype('STATIC')
        mapdl.nlgeom('ON')
        mapdl.nsubst(10, 50, 5)
        mapdl.autots('ON')
        mapdl.outres('ALL', 'ALL')
        mapdl.solve()
        mapdl.finish()
        mapdl.post1()
        mapdl.set('LAST')
        rf_x = mapdl.get_array(entity='NODE', item1='RF', it1num='FX')
        rf_y = mapdl.get_array(entity='NODE', item1='RF', it1num='FY')
        rf_z = mapdl.get_array(entity='NODE', item1='RF', it1num='FZ')
        nnum_rf = mapdl.mesh.nnum
        id2row = {int(n): i2 for i2, n in enumerate(nnum_rf)}
        rows = np.array([id2row.get(n, -1) for n in node_arr])
        valid = rows >= 0
        rf_stack = np.stack([rf_x, rf_y, rf_z], axis=1)
        rf_at_eq = np.zeros(dmap.shape[0])
        rf_at_eq[valid] = rf_stack[rows[valid], dir_arr[valid]]

        F_gen_all = {m: float(np.dot(rf_at_eq, inp['T_full2sec'][:, m])) for m in CHECK_MODES}
        F_self = F_gen_all[MODE_INDEX]
        print(f"\n  a={a}: F_gen[mode {MODE_INDEX}] (self) = {F_self:.6e}  ({time.time()-t1:.1f}s)", flush=True)
        others = sorted(((m, f) for m, f in F_gen_all.items() if m != MODE_INDEX),
                         key=lambda x: -abs(x[1]))
        print(f"    Top 5 other-mode projections (relative to self):", flush=True)
        for m, f in others[:5]:
            print(f"      mode {m}: F_gen={f:.6e}  ({100*f/F_self:+.3f}% of self)", flush=True)
        results.append(dict(a=a, F_gen_all=F_gen_all))
finally:
    if mapdl:
        try:
            mapdl.exit(force=True)
        except Exception:
            pass

out_path = os.path.join(s9.OUT, 'mode2_cross_coupling_check.npz')
np.savez(out_path, amplitudes=np.array(AMPLITUDES),
          F_gen_matrix=np.array([[r['F_gen_all'][m] for m in CHECK_MODES] for r in results]),
          check_modes=np.array(CHECK_MODES))
print(f"\nSaved: {out_path}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
print("DONE", flush=True)
