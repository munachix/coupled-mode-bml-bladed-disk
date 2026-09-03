"""FINAL (2nd attempt, after fixing a real detuning-feature scale bug) retrain
of all 5 pair BPINNs. PAIR-SPECIFIC architecture, not uniform:

- (0,1), (5,6), (7,8): USE_DETUNE=1 (saturated tanh(detune/20), see
  step6.add_detune_features), physics_weight=0.05 -- the combination that
  gave the best real, measured beta R^2 gain for these 3 pairs without the
  earlier scale bug (which made things WORSE, e.g. (3,4) beta 0.65/0.89 ->
  -0.09/-0.21) and without over-shooting physics_weight (0.1/0.3 measured
  WORSE beta than 0.05, confirmed non-monotonic on a real sweep).
- (3,4), (9,10): USE_DETUNE=0, physics_weight=0.01 (original default) --
  these two were already predicting beta well (0.65-0.99) BEFORE any of
  today's feature changes; the detuning feature measurably hurt (3,4) even
  after the saturation fix (-0.043/-0.120), so they keep the original
  6-feature architecture. Still re-run (not just restored from a stale
  checkpoint) because the W_GRID densification (a separate, unambiguously
  good fix, real amplitude gains everywhere it was tried) is now baked into
  this script by default and neither pair has been trained with it yet.

step7.py's load_bpinn_coupled()/predict_coupled_mc() were updated to be
self-describing (infer architecture from each checkpoint's own saved
in_dim/feat_mean length) specifically so this pair-specific mix works
without special-casing pair numbers in Step 7's own code.
"""
import subprocess, os, time

PAIRS_DETUNE = [(0, 1), (5, 6), (7, 8)]
PAIRS_ORIGINAL = [(3, 4), (9, 10)]
PY = r'C:\Users\Ronin\AppData\Local\Programs\Python\Python312\python.exe'
SCRIPT = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6\_train_pair_bpinn.py'

jobs = ([(p, True, 0.05) for p in PAIRS_DETUNE]
        + [(p, False, None) for p in PAIRS_ORIGINAL])

for (i, j), use_detune, pw in jobs:
    t0 = time.time()
    env = os.environ.copy()
    env['USE_DETUNE'] = '1' if use_detune else '0'
    if pw is not None:
        env['PHYSICS_WEIGHT_OVERRIDE'] = str(pw)
    else:
        env.pop('PHYSICS_WEIGHT_OVERRIDE', None)
    env.pop('REUSE_CKPT', None)   # every pair needs fresh data (grid/feature set changed)
    print(f"\n{'='*70}\n  FINAL RETRAIN PAIR ({i},{j}): USE_DETUNE={env['USE_DETUNE']}, "
          f"physics_weight={pw if pw is not None else '0.01 (default)'}\n{'='*70}", flush=True)
    result = subprocess.run([PY, SCRIPT, str(i), str(j)], env=env, capture_output=False)
    print(f"  Pair ({i},{j}) exit code {result.returncode}, elapsed {time.time()-t0:.0f}s", flush=True)
    if result.returncode != 0:
        print(f"  WARNING: pair ({i},{j}) failed", flush=True)

print("\nALL 5 PAIRS RETRAINED, FINAL ARCHITECTURE (PAIR-SPECIFIC)", flush=True)
