import sys, time, json, os
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

t0 = time.time()
print(f"=== Phase A: real ANSYS static K3, mode {MODE} ===", flush=True)
result = s9.run_case3_identification(mode_index=MODE, amplitudes=(0.02, 0.05, 0.08, 0.11))
elapsed = time.time() - t0
print(f"ELAPSED mode {MODE}: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

# run_case3_identification already saves to OUT/case3_k3_identification.npz -- rename per-mode
src = os.path.join(s9.OUT, 'case3_k3_identification.npz')
dst = os.path.join(s9.OUT, f'case3_k3_identification_mode{MODE}.npz')
if os.path.exists(src):
    os.replace(src, dst)
    print(f"Saved: {dst}", flush=True)

print("DONE", flush=True)
