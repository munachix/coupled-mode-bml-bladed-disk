"""Targeted real ANSYS measurement (2026-08-21): the full 70-mode dynamic
solve found mode 2 is a huge, previously-uncoupled contributor at node 1171
(0.459mm, second only to mode 0's 0.569mm) -- but mode 2 has never had its
real cross-mode nonlinear coupling measured with any neighbor, since the
frequency-gap-scan (correctly) found it too far detuned to need coupling
for RESONANCE purposes. That criterion answers "does this mode need
resonant locking with a neighbor" -- it does NOT answer "does nonlinear
energy transfer between this mode and a nearby one matter at the
amplitudes this specific point actually reaches," which is a separate
question this dynamic-validation gap just answered: yes.

Measures (1,2) and (2,3) -- bridging mode 2 into the ALREADY-real (0,1) and
(3,4) pairs as a connected 0-1-2-3-4 chain (mode 1 is already tightly
locked to mode 0, so coupling mode 2 to mode 1 transitively covers mode
0 too, without needing a separate (0,2) measurement). Same proven method
used for all 49 real pairs already measured this project: combined-
displacement NLGEOM static solves, 7 amplitude-pairs, general cubic
polynomial fit, projected reaction force per mode.
"""
import sys, time
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

t0 = time.time()
for pair in [(1, 2), (2, 3)]:
    print(f"\n{'='*70}\n  MEASURING PAIR {pair}\n{'='*70}", flush=True)
    t1 = time.time()
    s9.run_case3_cross_identification(mode_pair=pair)
    print(f"  Pair {pair} done in {time.time()-t1:.1f}s", flush=True)

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("DONE")
