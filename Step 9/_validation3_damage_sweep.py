# -*- coding: utf-8 -*-
"""
VALIDATION 3 SWEEP: real finite-element damage injections beyond the first two.

WHY. Section 3.6's claim -- that the coupled forward model localizes real,
independently generated single-blade damage where a diagonal-only model does not
-- rested on two cases, one exact and one a miss. Two points cannot support a
success rate, and the paper's own explanation for the miss (Section 4.2, "the
two cases differ in defect type, length at blade 5 versus tip at blade 10") is
not true of the code that produced them: both were injected through `d_tip`,
because the pipeline moved to a d_tip-only mistuning model on 2026-08-27 and the
Section 4.2 sentence was never updated. The two cases actually differ in
SEVERITY, -4.5% against -3%, which is a signal-to-noise explanation rather than
a defect-type one. This sweep tests that directly.

DESIGN. Eight new cases, chosen so that each axis answers something:

  severity sweep, blade 12 held fixed:  -2%, -3%, -4.5%, -6%
      The informative axis. If the -3% miss is a signal-to-noise floor rather
      than a defect-type effect, success should be a monotone function of
      severity and should fail at the bottom of this range.

  position spread, severity held at -4.5%:  blades 0, 7, 14, 19
      The disk is cyclically symmetric, so the physics of damaging blade 0 and
      blade 12 is identical up to rotation. What is NOT symmetric is the rest of
      the pipeline: the MAC assignment against the ROM's own mode ordering, and
      the participation matrix's per-blade columns. This axis tests those.

Together with the existing blade 5 (-4.5%) and blade 10 (-3%) runs, that is ten
real cases.

EACH CASE is a genuine full-order extraction: the mesh is perturbed nodally, a
real modal solve runs on the 181,473-DOF model, K and M are re-extracted, and
the resulting frequencies are MAC-matched to the ROM's mode ordering by
Hungarian assignment before either localizer sees them. Nothing is reused from
the ROM except the mode ordering to match against.

RANKS ARE RECORDED, not just the top pick. Reporting only "correct/incorrect"
throws away the distinction between a near-miss and a confident error, which is
exactly the distinction Section 3.6 needs for the blade 10 case.

Results accumulate to output/validation3_damage_sweep.json after every case, so
an interrupted run keeps what it has.
"""
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

ROOT = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project'
for _p in ('Step 9', 'Step 8', 'Step 7', 'Step 2'):
    sys.path.insert(0, os.path.join(ROOT, _p))

import step9 as s9  # noqa: E402
import step8 as s8  # noqa: E402
import step7 as s7  # noqa: E402
import step2 as s2  # noqa: E402

NB = s8.NB
N1B = s9.N1B
OUT_JSON = os.path.join(s9.OUT, 'validation3_damage_sweep.json')
CASE_ROOT = r'F:\ANSYS PCE\ROM_data_damage_sweep'
KEEP_CASE_DIRS = False      # each case is ~120 MB; the results are what matter

# (blade, severity, axis label)
#
# ROUND 1 (eight cases) established the rate and ruled out the severity story in
# the direction expected: blade 12 localizes exactly at -2% and -3% and misses by
# six positions at -4.5% and -6%. It left two candidate explanations open, and
# they make different predictions, so round 2 is designed to separate them.
#
#   H1  the linear frequency-shift map inside the support-search forward model
#       loses accuracy as the perturbation grows. Predicts that a blade which
#       succeeds at -4.5% should also fail once driven hard enough, regardless of
#       which blade it is.
#
#   H2  aliasing between nodal-diameter harmonics. Round 1's misses sit at ring
#       distances of six and eight, which are 24/4 and 24/3 exactly. Predicts
#       that success depends on WHERE the blade sits, in a pattern periodic in
#       the blade index, and not on how hard it is driven.
#
# The discriminating experiment for H2 is the full ring at one fixed severity:
# if the failures are periodic in blade index, 24 of 24 at -4.5% will show it
# outright, and if they are scattered it is ruled out. Eighteen blades remain
# (0, 5, 7, 12, 14, 19 are already done at that severity).
#
# The discriminating experiment for H1 is to push a blade that SUCCEEDED at
# -4.5% (blade 14) further, to -6% and -8%. If H1 holds it should break down
# there too.
_FULL_RING = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13,
              15, 16, 17, 18, 20, 21, 22, 23]

CASES = [
    # --- round 1 -----------------------------------------------------------
    (12, -0.020, 'severity'),
    (12, -0.030, 'severity'),
    (12, -0.045, 'severity'),
    (12, -0.060, 'severity'),
    (0,  -0.045, 'position'),
    (7,  -0.045, 'position'),
    (14, -0.045, 'position'),
    (19, -0.045, 'position'),
    # --- round 2: the rest of the ring at a fixed severity, for H2 ----------
] + [(b, -0.045, 'ring') for b in _FULL_RING] + [
    # --- round 2: push a success case harder, for H1 ------------------------
    (14, -0.060, 'severity'),
    (14, -0.080, 'severity'),
]

print("=== Validation 3 sweep: real FE damage injections ===", flush=True)
print(f"  {len(CASES)} cases in the plan (already-run ones are skipped)",
      flush=True)

inp, prior, HI1_healthy, df_all, models, pairs, chain = s8.load_inputs()
_s4inp = s9.s4.load_inputs()
inp['T_full2sec'] = _s4inp['T_full2sec']
inp['blade_dofs'] = _s4inp['blade_dofs']

tip_coeff = s9.s4.CONFIG['sensitivity']['tip_coeff_per_frac']
freqs_tuned = inp['freqs_sec'][:N1B]
dmap = s2._dof_map()
node_arr = dmap[:, 0].astype(int)
dir_arr = dmap[:, 1].astype(int)

baseline_pred = s8.compute_baseline_predictions(np.zeros(NB), inp, models, pairs, chain)


def ring_distance(a, b):
    d = abs(int(a) - int(b))
    return int(min(d, NB - d))


def rank_of(residuals, blade):
    """1-based rank of `blade` when candidates are ordered by residual."""
    return int(np.argsort(residuals).tolist().index(int(blade)) + 1)


def match_real_frequencies(full, df_true):
    """MAC-match a real extraction's modes onto the ROM's own 1B ordering."""
    dK_sec = s9.s4.assemble_dK_sec_coupled(df_true, inp, inp['K_sec'])
    _, v_eig = eigh(inp['K_sec'] + dK_sec, inp['M_sec'])
    Phi_pred_eqorder = inp['T_full2sec'] @ v_eig[:, :N1B]

    nnum = full['node_ids']
    id2row = {int(n): i for i, n in enumerate(nnum)}
    rows = np.array([id2row.get(n, -1) for n in node_arr])
    valid = (rows >= 0) & (dir_arr >= 0) & (dir_arr < 3)
    Phi_nodal = np.zeros((len(nnum), 3, N1B))
    Phi_nodal[rows[valid], dir_arr[valid], :] = Phi_pred_eqorder[valid, :]
    Phi_at_full = Phi_nodal.reshape(-1, N1B)

    n_cand = full['freqs'].shape[0]
    full_flat = full['Phi'].reshape(-1, full['Phi'].shape[-1])[:, :n_cand]
    MAC = s2._mac_matrix(full_flat, Phi_at_full)
    fi, ri = linear_sum_assignment(-MAC)
    order = np.argsort(ri)
    fi, ri = fi[order], ri[order]
    return full['freqs'][fi], MAC[fi, ri]


def run_case(blade, severity, axis):
    t0 = time.time()
    case_dir = os.path.join(CASE_ROOT, f'b{blade:02d}_s{abs(severity)*1000:03.0f}')
    theta_row = {v: np.zeros(NB) for v in
                 ['d_length', 'd_thickness', 'd_le_te', 'd_twist_deg', 'd_tip']}
    dtip = severity * inp['t_ref'] / tip_coeff
    theta_row['d_tip'][blade] = dtip
    print(f"\n{'='*70}\n  CASE blade {blade}, severity {severity*100:+.1f}%  "
          f"(d_tip = {dtip:+.4f} mm)\n{'='*70}", flush=True)

    df_true = s9.s4.compute_delta_f(theta_row, inp['L_ref'], inp['t_ref'])
    s9.run_perturbed_extraction(theta_row, case_dir,
                                label=f'damage sweep blade {blade} @ {severity*100:.1f}%')
    full = s9.load_full_order_case(case_dir)
    if full is None:
        raise RuntimeError('extraction produced no loadable output')

    y_real, mac = match_real_frequencies(full, df_true)
    HI1 = float(np.max(np.abs(y_real - freqs_tuned)))

    df_baseline = np.zeros(NB)
    diag = s7.sparse_localize_blade(y_real, df_baseline, inp)
    coup = s7.sparse_localize_blade_coupled(y_real, df_baseline, inp)

    df_inf = df_baseline.copy()
    df_inf[coup['best_blade']] += coup['severities'][coup['best_blade']]
    HI2 = float(s8.compute_HI2(df_inf, prior))
    HI3 = float(s8.compute_HI3(df_inf, baseline_pred, inp, models, pairs, chain))

    rec = dict(
        blade=int(blade), severity=float(severity), axis=axis,
        d_tip_mm=float(dtip),
        diagonal_blade=int(diag['best_blade']),
        diagonal_ring=ring_distance(diag['best_blade'], blade),
        diagonal_rank=rank_of(diag['residuals'], blade),
        diagonal_margin=float(diag['margin']),
        coupled_blade=int(coup['best_blade']),
        coupled_ring=ring_distance(coup['best_blade'], blade),
        coupled_rank=rank_of(coup['residuals'], blade),
        coupled_margin=float(coup['margin']),
        coupled_severity_fit=float(coup['severities'][coup['best_blade']]),
        HI1=HI1, HI2=HI2, HI3=HI3,
        mac_min=float(mac.min()), mac_mean=float(mac.mean()),
        freqs_matched=y_real.tolist(),
        residuals_diagonal=np.asarray(diag['residuals']).tolist(),
        residuals_coupled=np.asarray(coup['residuals']).tolist(),
        wall_seconds=round(time.time() - t0, 1),
    )
    print(f"  diagonal -> blade {rec['diagonal_blade']:2d}  ring {rec['diagonal_ring']}  "
          f"true-blade rank {rec['diagonal_rank']}/24", flush=True)
    print(f"  coupled  -> blade {rec['coupled_blade']:2d}  ring {rec['coupled_ring']}  "
          f"true-blade rank {rec['coupled_rank']}/24  margin {rec['coupled_margin']:.4f}",
          flush=True)
    print(f"  MAC min {rec['mac_min']:.4f}  |  HI1 {HI1:.3f} Hz  |  "
          f"{rec['wall_seconds']:.0f} s", flush=True)

    if not KEEP_CASE_DIRS:
        shutil.rmtree(case_dir, ignore_errors=True)
    return rec


results = []
if os.path.exists(OUT_JSON):
    results = json.load(open(OUT_JSON))
    done = {(r['blade'], round(r['severity'], 4)) for r in results}
    print(f"  resuming: {len(results)} case(s) already recorded", flush=True)
else:
    done = set()

for blade, severity, axis in CASES:
    if (blade, round(severity, 4)) in done:
        print(f"  skipping blade {blade} @ {severity*100:.1f}% (already done)", flush=True)
        continue
    try:
        results.append(run_case(blade, severity, axis))
    except Exception:
        print(f"  CASE FAILED (blade {blade}, {severity*100:.1f}%):", flush=True)
        traceback.print_exc()
        results.append(dict(blade=int(blade), severity=float(severity), axis=axis,
                            failed=True, error=traceback.format_exc()[-800:]))
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

ok = [r for r in results if not r.get('failed')]
print(f"\n{'='*70}\n  SWEEP COMPLETE: {len(ok)}/{len(results)} cases produced results",
      flush=True)
if ok:
    dc = sum(r['diagonal_ring'] <= 2 for r in ok)
    cc = sum(r['coupled_ring'] <= 2 for r in ok)
    print(f"  diagonal correct: {dc}/{len(ok)}   coupled correct: {cc}/{len(ok)}",
          flush=True)
print(f"  saved -> {OUT_JSON}", flush=True)
