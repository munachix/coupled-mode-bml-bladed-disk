import sys, time
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 1')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 2')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
import step1 as s1
import step2 as s2
import step4 as s4
import numpy as np

WB_MECH = r'F:\ANSYS PCE\PCE_files\dp0\SYS-3\MECH'

t0 = time.time()
inp = s4.load_inputs()
T_full2sec = inp['T_full2sec']  # (181473, 70)
dmap = s2._dof_map()
node_arr = dmap[:, 0].astype(int)
dir_arr = dmap[:, 1].astype(int)  # 0=X,1=Y,2=Z

mapdl = None
try:
    mapdl = s1.launch_mapdl()
    mapdl.run(f"/CWD,'{WB_MECH}'")
    import os
    # SAFETY: the real ds.dat ends in a `solve` command (Workbench bundles the
    # ENTIRE model+solve into one file) -- inputting it whole would silently
    # re-run the full 160-cycle transient from scratch. Use the truncated,
    # solve-free copy instead (everything up to but NOT including `solve`).
    fp = os.path.join(WB_MECH, 'ds_nomodel_only.dat')
    mapdl.input(fp)
    print(f"Loaded mesh/model context from ds_nomodel_only.dat (solve command excluded)", flush=True)
    mapdl.finish()
    mapdl.post1()
    mapdl.run('FILE,file,rst')
    n_sets = int(mapdl.get_value('ACTIVE', 0, 'SET', 'NSET'))
    print(f"Result file has {n_sets} stored sets", flush=True)

    # Drastically reduced sample count (12, not 240) -- we only need to see
    # WHETHER other modes carry real energy, not fine time resolution.
    N_SAMPLES = 12
    set_indices = sorted(set(int(x) for x in np.linspace(1, n_sets, N_SAMPLES)))
    print(f"Sampling {len(set_indices)} sets: {set_indices}", flush=True)

    n_modes_check = 6
    q_hist = np.zeros((len(set_indices), n_modes_check))
    t_hist = np.zeros(len(set_indices))

    nnum = mapdl.mesh.nnum  # fetch ONCE, outside the loop -- this was the bug
    id2row = {int(n): r for r, n in enumerate(nnum)}
    rows = np.array([id2row.get(n, -1) for n in node_arr])
    valid = rows >= 0

    for k, i in enumerate(set_indices):
        t_k0 = time.time()
        mapdl.set(1, i)
        t_hist[k] = float(mapdl.get_value('ACTIVE', 0, 'SET', 'TIME'))
        ux = mapdl.post_processing.nodal_displacement('X')
        uy = mapdl.post_processing.nodal_displacement('Y')
        uz = mapdl.post_processing.nodal_displacement('Z')
        u_stack = np.stack([ux, uy, uz], axis=1)
        u_full = np.zeros(dmap.shape[0])
        u_full[valid] = u_stack[rows[valid], dir_arr[valid]]
        for m in range(n_modes_check):
            q_hist[k, m] = float(T_full2sec[:, m] @ u_full)
        print(f"  [{time.time()-t_k0:.1f}s] set {i}/{n_sets} (t={t_hist[k]:.4f}s): q0={q_hist[k,0]:.5f} "
              f"q1={q_hist[k,1]:.5f} q2={q_hist[k,2]:.5f} q3={q_hist[k,3]:.5f}", flush=True)
    mapdl.finish()
finally:
    if mapdl:
        try:
            mapdl.exit(force=True)
        except Exception:
            pass

elapsed = time.time() - t0
print(f"\nELAPSED: {elapsed:.1f}s", flush=True)

np.savez(r'F:\ANSYS PCE\ROM_data_sensitivity\case3_transient\multimode_energy_check.npz',
         t=t_hist, q=q_hist)

print("\n=== RMS amplitude per mode (proxy for how much energy each mode carries) ===", flush=True)
for m in range(n_modes_check):
    rms = np.sqrt(np.mean(q_hist[:, m]**2))
    peak = np.max(np.abs(q_hist[:, m]))
    print(f"Mode {m}: RMS q = {rms:.6f} mm, peak |q| = {peak:.6f} mm", flush=True)
print("DONE", flush=True)
