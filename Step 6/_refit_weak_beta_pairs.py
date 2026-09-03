"""Retrain the 3 pairs with poor beta (quadrature-phase) R^2 -- (0,1), (5,6),
(7,8) -- using _train_pair_bpinn.py's newly-densified w-grid (2026-08-21 fix,
see that file's own comment for the root-cause diagnosis). Runs sequentially,
each a fresh subprocess so torch/numpy state never leaks between pairs.
"""
import subprocess, sys, time

PAIRS = [(0, 1), (5, 6), (7, 8)]
PY = r'C:\Users\Ronin\AppData\Local\Programs\Python\Python312\python.exe'
SCRIPT = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6\_train_pair_bpinn.py'

for (i, j) in PAIRS:
    t0 = time.time()
    print(f"\n{'='*70}\n  RETRAINING PAIR ({i},{j}) WITH DENSIFIED W_GRID\n{'='*70}", flush=True)
    result = subprocess.run([PY, SCRIPT, str(i), str(j)], capture_output=False)
    print(f"  Pair ({i},{j}) exit code {result.returncode}, elapsed {time.time()-t0:.0f}s", flush=True)
    if result.returncode != 0:
        print(f"  WARNING: pair ({i},{j}) failed, continuing to next pair", flush=True)

print("\nALL WEAK-BETA PAIRS RETRAINED", flush=True)
