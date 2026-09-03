# Trains 4 independent (uncoupled) per-mode BPINNs, modes 0-3, using step6.py's
# own SDOF training code with CONFIG['mode_index'] swapped per mode.
# STATUS (2026-08-21): only mode 2's output (bpinn_state_mode2.pt) is still
# real, load-bearing data -- mode 2 is the one 1B-cluster mode confirmed
# genuinely isolated by the real frequency-gap scan (Step 4's MODE_GROUPS).
# Modes 0, 1, and 3's independent outputs here are SUPERSEDED by the coupled
# pair BPINNs trained in _train_pair_bpinn.py (bpinn_coupled_state_01.pt,
# bpinn_coupled_state_34.pt) once modes 0-1 and 3-4 were found near-degenerate
# and requiring real cross-mode coupling (PROJECT_STATUS.md Section 9j/9k).
# Kept on disk for historical/comparison value, not re-run or deleted.
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
import step6 as s6

OUT = s6.OUT
MODES = [0, 1, 2, 3]
results = {}

for m in MODES:
    t0 = time.time()
    print(f"\n{'='*70}\n  MODE {m}: TRAINING PER-MODE BPINN\n{'='*70}", flush=True)
    s6.CONFIG['mode_index'] = m

    inp = s6.load_inputs()
    rng = np.random.default_rng(s6.CONFIG['random_seed'])
    perm = rng.permutation(inp['n_samples'])
    train_idx = perm[:s6.CONFIG['n_train_samples']]
    test_idx = perm[s6.CONFIG['n_train_samples']:s6.CONFIG['n_train_samples'] + s6.CONFIG['n_test_samples']]

    train_rows = s6.build_dataset(inp, train_idx, rng)
    test_rows = s6.build_dataset(inp, test_idx, rng)
    n_labeled_train = sum(1 for r in train_rows if r['alpha'] is not None)
    print(f"  Train: {n_labeled_train} labeled + {len(train_rows) - n_labeled_train} unlabeled; "
          f"Test: {sum(1 for r in test_rows if r['alpha'] is not None)} labeled", flush=True)

    model, history, norm_stats = s6.train_bpinn(train_rows, s6.CONFIG['n_train_samples'])
    val = s6.validate_bpinn(model, test_rows, norm_stats)

    fp_model = os.path.join(OUT, f'bpinn_state_mode{m}.pt')
    torch.save(model.state_dict(), fp_model)
    fp_pred = os.path.join(OUT, f'bpinn_predictions_mode{m}.npz')
    np.savez(fp_pred, w=val['w'], true_amp=val['true_amp'], amp_mean=val['amp_mean'],
             amp_std=val['amp_std'], sample_idx=val['sample_idx'],
             feat_mean=norm_stats[0], feat_std=norm_stats[1])

    elapsed = time.time() - t0
    print(f"  MODE {m} DONE: R2={val['r2']:.4f} RMSE={val['rmse']:.4f} "
          f"within_2sigma={val['within_2sigma']:.3f} residual_rms={val['residual_rms']:.4e} "
          f"elapsed={elapsed:.1f}s", flush=True)
    results[m] = dict(r2=val['r2'], rmse=val['rmse'], within_2sigma=val['within_2sigma'],
                       residual_rms=val['residual_rms'], elapsed=elapsed)

fp_summary = os.path.join(OUT, 'multimode_bpinn_summary.json')
with open(fp_summary, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved summary: {fp_summary}", flush=True)
print("ALL MODES DONE", flush=True)
