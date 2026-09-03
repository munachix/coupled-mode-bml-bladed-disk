"""Full retraining of every forcing-aware BPINN (5 pairs + chain + mode2)
on the CURRENT post-pivot mistuning distribution (2026-08-27 d_tip-only
scope change). The previous checkpoints (2026-08-24) were trained on the
old 5-variable theta distribution -- confirmed via a direct comparison
of today's real shift_m distribution against each network's own saved
feat_mean/feat_std: real, measurable (though not extreme) mismatch. Run
sequentially (not in parallel) to avoid CPU contention distorting any
one run's wall-clock estimate; each script reads theta_samples.npz fresh
so no code changes needed, just a rerun."""
import subprocess, time

PY = r'C:\Users\Ronin\AppData\Local\Programs\Python\Python312\python.exe'
S6 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6'

jobs = [
    (f'{S6}\_train_forcing_aware_01.py', ['0', '1']),
    (f'{S6}\_train_forcing_aware_01.py', ['3', '4']),
    (f'{S6}\_train_forcing_aware_01.py', ['5', '6']),
    (f'{S6}\_train_forcing_aware_01.py', ['7', '8']),
    (f'{S6}\_train_forcing_aware_01.py', ['9', '10']),
    (f'{S6}\_train_forcing_aware_chain.py', []),
    (f'{S6}\_train_forcing_aware_mode2.py', []),
]

t_start = time.time()
for script, args in jobs:
    t0 = time.time()
    name = script.replace('\\', '/').split('/')[-1]
    print(f"\n{'='*70}\n  RETRAINING: {name} {args}\n{'='*70}", flush=True)
    result = subprocess.run([PY, script] + args, capture_output=False)
    print(f"  {name} {args} exit code {result.returncode}, elapsed {time.time()-t0:.0f}s "
          f"(total so far {time.time()-t_start:.0f}s)", flush=True)
    if result.returncode != 0:
        print(f"  WARNING: {name} {args} failed, continuing to next job", flush=True)

print(f"\nALL RETRAINING DONE, total {time.time()-t_start:.0f}s", flush=True)
