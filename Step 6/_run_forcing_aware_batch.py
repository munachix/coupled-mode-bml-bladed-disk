"""Batch runner for the forcing-aware BPINN, remaining 4 pairs (2026-08-23).
Pair (0,1) is already running as a separate process; this covers (3,4),
(5,6), (7,8), (9,10) sequentially so they don't contend with each other for
CPU (they will contend with the already-running (0,1) job to some degree,
accepted given the time pressure)."""
import subprocess, time

PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10)]
PY = r'C:\Users\Ronin\AppData\Local\Programs\Python\Python312\python.exe'
SCRIPT = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6\_train_forcing_aware_01.py'

for (i, j) in PAIRS:
    t0 = time.time()
    print(f"\n{'='*70}\n  FORCING-AWARE TRAINING: PAIR ({i},{j})\n{'='*70}", flush=True)
    result = subprocess.run([PY, SCRIPT, str(i), str(j)], capture_output=False)
    print(f"  Pair ({i},{j}) exit code {result.returncode}, elapsed {time.time()-t0:.0f}s", flush=True)
    if result.returncode != 0:
        print(f"  WARNING: pair ({i},{j}) failed, continuing to next pair", flush=True)

print("\nALL 4 REMAINING PAIRS DONE (forcing-aware)", flush=True)
