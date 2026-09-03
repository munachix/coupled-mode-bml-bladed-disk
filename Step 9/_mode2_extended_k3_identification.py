"""EXTENDED static NLGEOM K3 identification, mode 2 (2026-08-27) -- fixes
the real, verified root cause of the backbone-sweep failures: mode 2's
real static K3 measurement (case3_k3_identification_mode2.npz) only
tested generalized-coordinate amplitudes up to 0.11, but the real dynamic
response at w=1.20/target_peak=0.8 converges to amplitude ~1.08 -- a
~9.8x extrapolation beyond the calibrated range. The original fit was
excellent WITHIN its range (F_nl/a^3 constant to <0.05%), so this isn't a
bad measurement, it's a small-signal model being trusted 10x past where
it was ever validated.

This extends the same real ANSYS static NLGEOM method (run_case3_static_point,
already proven, no new methodology) to amplitudes spanning the original
range through and past the real dynamic operating point, so the fit (or
its breakdown) is characterized where the system actually operates."""
import sys, os, time
import numpy as np
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

MODE_INDEX = 2
# Original 4 points (0.02-0.11) kept for continuity, extended well past
# the real dynamic operating amplitude (~1.08) for margin.
AMPLITUDES = [0.02, 0.05, 0.08, 0.11, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1, 1.3]

t0 = time.time()
print(f"=== Extended static K3 identification, mode {MODE_INDEX} ===", flush=True)
print(f"  Amplitudes: {AMPLITUDES}", flush=True)

inp = s9.s4.load_inputs()
K_lin = inp['K_sec'][MODE_INDEX, MODE_INDEX]
print(f"  Linear K_sec[{MODE_INDEX},{MODE_INDEX}] = {K_lin:.6e}", flush=True)

mapdl = None
F_vals = []
try:
    mapdl = s9.s1.launch_mapdl()
    mapdl = s9.s1.setup_model(mapdl)
    for i, a in enumerate(AMPLITUDES):
        t1 = time.time()
        F = s9.run_case3_static_point(mapdl, MODE_INDEX, a, inp, tag=f'ext{i}')
        F_vals.append(F)
        print(f"  a={a:.3f}: F={F:.6e}  ({time.time()-t1:.1f}s)", flush=True)
finally:
    if mapdl:
        try:
            mapdl.exit(force=True)
        except Exception:
            pass

a_arr = np.array(AMPLITUDES, dtype=float)
F_arr = np.array(F_vals, dtype=float)
F_nl = F_arr - K_lin * a_arr

print(f"\n  a       F(a)            F_linear        F_nl=F-F_lin     F_nl/a^3")
for a, F, Fnl in zip(a_arr, F_arr, F_nl):
    print(f"  {a:6.3f}  {F:14.6e}  {K_lin*a:14.6e}  {Fnl:14.6e}  {Fnl/a**3 if a>0 else float('nan'):14.6e}", flush=True)

# Fit K3 two ways: (a) using ONLY the original small-amplitude points
# (should reproduce the original 3.29e8 fit almost exactly, a sanity
# check), and (b) using ALL points (reveals whether/how badly a single
# cubic term fails to fit the extended range).
mask_small = a_arr <= 0.11
K3_small = float(np.sum(a_arr[mask_small] ** 3 * F_nl[mask_small]) / np.sum(a_arr[mask_small] ** 6))
K3_all = float(np.sum(a_arr ** 3 * F_nl) / np.sum(a_arr ** 6))
print(f"\n  K3 fit (small-amplitude points only, a<=0.11): {K3_small:.6e}  "
      f"(original saved value: 3.28987e+08)")
print(f"  K3 fit (ALL points, 0.02-1.3): {K3_all:.6e}")
print(f"  Ratio (all-range fit / small-range fit): {K3_all/K3_small:.4f}")

out_path = os.path.join(s9.OUT, f'case3_k3_identification_mode{MODE_INDEX}_extended.npz')
np.savez(out_path, mode_index=MODE_INDEX, amplitudes=a_arr, F=F_arr, F_nl=F_nl,
          K3_small=K3_small, K3_all=K3_all, K_lin=K_lin)
print(f"\nSaved: {out_path}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
print("DONE", flush=True)
