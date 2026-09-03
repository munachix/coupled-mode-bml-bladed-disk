"""
Follow-up diagnostic: using the REAL project code (step8.calibrate_mistuning_classifier
+ step8.classify_mistuning, unmodified), test whether feeding the classifier the
Bayesian-INFERRED posterior mean (traj_result['post_means'][t], what a real monitoring
system actually has) instead of the unobservable TRUE injected state
(traj['df_traj'][t], what validate_mistuning_classifier currently uses) closes most of
the gap to HI3's own -0.992 correlation, with the identical (unmodified) aggregation
math -- isolating the "which input" question from the "aggregation/weighting" question.
"""
import sys, os
import numpy as np
import torch

_STEP8 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 8'
_STEP7 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7'
_STEP6 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6'
_STEP4 = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4'
for p in (_STEP8, _STEP7, _STEP6, _STEP4):
    sys.path.insert(0, p)
import step8 as s8

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
traj = s8.build_damage_trajectory(df_all, record_checks=False)
d = np.load(os.path.join(s8.OUT, 'damage_trajectory.npz'))
post_means = d['post_means']
assert np.allclose(d['df_baseline'], traj['df_baseline'])

mclf = s8.calibrate_mistuning_classifier(inp, df_all, models, pairs, chain)

T = len(traj['t_norm'])
torch.manual_seed(s8.CONFIG['random_seed'] + 61_000)
sigmas_true = np.zeros(T)
for t in range(T):
    r = s8.classify_mistuning(traj['df_traj'][t], inp, models, pairs, chain, mclf)
    sigmas_true[t] = r['sigma']

torch.manual_seed(s8.CONFIG['random_seed'] + 61_000)
sigmas_pm = np.zeros(T)
for t in range(T):
    r = s8.classify_mistuning(post_means[t], inp, models, pairs, chain, mclf)
    sigmas_pm[t] = r['sigma']

corr_true = float(np.corrcoef(traj['severity'], sigmas_true)[0, 1])
corr_pm = float(np.corrcoef(traj['severity'], sigmas_pm)[0, 1])
print(f"\n=== REAL CODE, UNMODIFIED AGGREGATION ===")
print(f"corr(severity, classifier sigma) using TRUE df (validate_mistuning_classifier AS-IS)   = {corr_true:.4f}")
print(f"corr(severity, classifier sigma) using POSTERIOR-MEAN df (what HI2/HI3 actually use)    = {corr_pm:.4f}")
print(f"\nsigmas_true: {sigmas_true}")
print(f"sigmas_pm:   {sigmas_pm}")
