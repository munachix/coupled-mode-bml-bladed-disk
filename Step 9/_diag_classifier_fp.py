import sys
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 8')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
import numpy as np
import step8 as s8

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
zero_pred = s8.compute_baseline_predictions(np.zeros(s8.NB), inp, models, pairs, chain)
import torch
torch.manual_seed(s8.CONFIG['random_seed'] + 60_000)
rng = np.random.default_rng(s8.CONFIG['random_seed'] + 60_000)
idx = rng.choice(df_all.shape[0], size=400, replace=False)
scores = np.array([s8.compute_mistuning_severity(df_all[i], zero_pred, inp, models, pairs, chain) for i in idx])
threshold = float(np.percentile(scores, 95))
baseline_idx = s8.CONFIG['unit_baseline_idx']
base_score = s8.compute_mistuning_severity(df_all[baseline_idx], zero_pred, inp, models, pairs, chain)
pct = (scores < base_score).mean() * 100
print(f"Calibration: mean={scores.mean():.5f} std={scores.std():.5f} threshold(95th)={threshold:.5f}")
print(f"Baseline sample #{baseline_idx} score = {base_score:.5f}  (percentile among 400 calib samples: {pct:.1f}%)")
print(f"df_all[{baseline_idx}] stats: mean={df_all[baseline_idx].mean():.5f} std={df_all[baseline_idx].std():.5f} max|.|={np.abs(df_all[baseline_idx]).max():.5f}")
print(f"Population df_all stats: mean|.| per-sample max, distribution -- mean={np.abs(df_all).max(axis=1).mean():.5f}, std={np.abs(df_all).max(axis=1).std():.5f}")
print(f"Baseline sample's own max|df| rank: {(np.abs(df_all).max(axis=1) < np.abs(df_all[baseline_idx]).max()).mean()*100:.1f} percentile")

# Find a baseline sample near the MEDIAN of the classifier score distribution
# (not cherry-picked to minimize score -- picked for being TYPICAL, since the
# original arbitrary index (975) happens to trip a known, pre-existing
# chain-aggregate-noise weak point, Section 9o/9l of PROJECT_STATUS.md).
median_score = np.median(scores)
closest = idx[np.argmin(np.abs(scores - median_score))]
print(f"\nMedian calib score: {median_score:.5f}")
print(f"Sample closest to median: #{closest}, score={scores[np.argmin(np.abs(scores-median_score))]:.5f}")
