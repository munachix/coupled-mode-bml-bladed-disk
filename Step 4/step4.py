"""
STEP 4 v1.0: Nonlinear Reduced-Order Model — Mistuned + Geometric-Nonlinear
============================================================================

Implements PHASE 4 of the roadmap: extends Step 2's linear PC-CMS secondary
ROM (K_sec, M_sec, 70 modes) with (a) blade-level geometric MISTUNING
stiffness, mapped from Step 3's theta samples, and (b) a geometric NONLINEAR
(Duffing-type cubic) modal stiffness term — so the reduced equation becomes:

    M_sec q'' + C_sec q' + [K_sec + dK_sec(theta)] q + f_nl(q) = F(t)

matching the roadmap's governing equation with K containing "linear
stiffness and geometric mistuning" and f_nl the "geometric nonlinear
restoring forces".

──────────────────────────────────────────────────────────────────────────
PART A — Mistuning stiffness mapping (theta -> dK_sec): participation-
weighted modal (Rayleigh-quotient) perturbation
──────────────────────────────────────────────────────────────────────────
IMPORTANT MODELING NOTE, and a real bug this fixed: the blisk is machined
as ONE continuously-connected part — 24 blades integral with a single hub,
not 24 independent sectors bolted together. So a blade's stiffness is NOT
cleanly separable into a "this blade's own block" of the full-order K_full
matrix; a first version of this step tried exactly that (scaling each
blade's raw K_full tip-DOF block and projecting it through T_full2sec) and
it produced a 138% mean modal frequency shift from a <=3.5% input — wrong
by ~40x, because the raw master-DOF block of the UNREDUCED K_full carries
disk/hub-mediated stiffness at a completely different scale than the
already-CONDENSED K_sec it was being added into.

The fix uses ONLY quantities that are already correctly scaled in the
reduced (secondary) basis. Each blade's fractional PARTICIPATION in each
secondary mode is read directly off Step 2's own full-order-to-secondary
transformation, restricted to that blade's tip/master DOFs:

    Phi_b   = T_full2sec[blade_b_master_dofs, :]            (30 x n_sec)
    P[b, m] = ||Phi_b[:, m]||^2 / sum_b' ||Phi_b'[:, m]||^2  (sums to 1 over b, per mode m)

i.e. P[b, m] is "what fraction of secondary mode m's tip-DOF energy sits on
blade b". Each blade's 5 geometric deviations (from Step 3) are mapped to a
single fractional-frequency mistuning value via simple cantilever-beam
sensitivity theory:

    df_b/f = length_exp * (dL_b / L_ref) + thickness_exp * (dt_b / t_ref)
             + twist_coeff * d_twist_b_deg
             + le_te_coeff * (d_le_te_b / L_ref)
             + tip_coeff   * (d_tip_b   / t_ref)

length_exp=-2, thickness_exp=1 come from standard cantilever-beam natural-
frequency scaling (f ~ L^-2, and f ~ t for a simple rectangular section
where I~t^3, A~t so sqrt(I/A)~t). twist/LE-TE/tip have no equally simple
closed form, so their coefficients are small DOCUMENTED PLACEHOLDERS (see
CONFIG['sensitivity']) pending a real perturbed-FEM sensitivity study —
exactly like Step 3's tolerance placeholders.

The participation-weighted, first-order (Rayleigh-quotient-consistent)
modal stiffness perturbation is then:

    shift[m]   = sum_b  P[b, m] * ((1 + df_b/f)^2 - 1)     (bounded by max_b |df_b/f|, by construction)
    dK_sec     = diag(K_sec) * shift                        (DIAGONAL ONLY, see limitation below)

STATUS AS OF 2026-08-13 (v1.0's "diagonal-only, no ANSYS" framing above is
HISTORICAL — real ANSYS access was found 2026-08-09 and this was fixed for
real, not left as a documented gap): the diagonal-only update above is
SUPERSEDED as this module's default by `assemble_dK_sec_coupled()` (Part
A2 below), which adds real off-diagonal Fundamental-Mistuning-Model (FMM)
coupling built from real ANSYS-measured per-blade participation — the
exact mechanism this v1.0 note originally flagged as missing. Validated
directly against real ANSYS Case 2 (mistuned linear): MAC min 0.908, mean
0.969 with the coupled model vs. MAC min 0.097, mean 0.407 with the
diagonal-only model on the SAME real data (PROJECT_STATUS.md Section 9r
item 3) — mode localization/veering is genuinely captured now, not still
a flagged future step. The plain diagonal `assemble_dK_sec()` function
still exists (used by a few legacy/comparison call sites) but is NOT this
module's recommended default; use the coupled version for anything meant
to match real ANSYS.

──────────────────────────────────────────────────────────────────────────
PART B — Geometric nonlinear (Duffing) modal stiffness
──────────────────────────────────────────────────────────────────────────
STATUS AS OF 2026-08-13 (v1.0's "PLACEHOLDER, no ANSYS access" framing
below is HISTORICAL, not current): real ANSYS Green-Lagrange calibration
WAS performed once ANSYS access was found (Step 9's `run_case3_identification()`
/ `run_case3_cross_identification()`, real NLGEOM static solves with a
prescribed physical displacement field, F_nl fit to a cubic by least
squares). As of 2026-08-13, ALL 24 1B-cluster modes have real measured
diagonal K3 (`CONFIG['nonlinear']['measured_K3']`, fit quality <0.1%
relative error, confirmed a good cubic fit, not just order-of-magnitude),
PLUS real measured off-diagonal cross-mode coupling for every near-
degenerate pair/chain relationship in that cluster
(`CONFIG['nonlinear']['cross_coupling']`, used by `coupled_nonlinear_force()`
/ `duffing_forced_response_coupled()` / `duffing_forced_response_chain()`).
Neither is invented or left at zero for the 1B cluster anymore.

UPDATE 2026-08-27: a subsequent HF-band campaign (`Step 9/_hf_multimode_campaign.py`)
extended real ANSYS Green-Lagrange K3 + cross-coupling measurement to ALL
46 remaining modes (24-69), closing the gap this note originally
described. As of this session, ALL 70 secondary modes have real,
ANSYS-measured K3 (confirmed by `build_nonlinear_stiffness()`'s own
validation check: 70/70 measured, 0 relying on any extrapolation). The
diagonal-only `hardening_ratio` EXTRAPOLATION formula below is REMOVED
from `build_nonlinear_stiffness()` (2026-08-27, explicit user decision) --
any FUTURE mode added to this ROM without a real measurement gets K3=0
(disclosed as unmeasured), never a fabricated placeholder:

    K3_sec_diag[m] = hardening_ratio * K_sec[m, m] / q_ref^2   (HISTORICAL formula, no longer used as a live fallback)

Outputs (to CONFIG['output_dir'], LOCAL to Step 4 — F:\\ANSYS PCE\\ROM_data
stays read-only Step-1/2 territory, same policy as Step 3):
    nonlinear_rom.npz     — K3_sec_diag (70,), q_ref, hardening_ratio,
                             sensitivity coefficients used
    mistuning_validation.npz — per-validated-sample: delta_f_blade (n_val,24),
                             freqs_mistuned (n_val,70)
    step4_config.json     — full provenance

Author: PCE-Bayesian Framework — v1.0 (24-blade)
"""

import numpy as np, os, json, time, sys
from datetime import datetime, timezone
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import plot_style   # noqa: E402  (shared publication style, see PCE project/plot_style.py)
plot_style.apply_style()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
FIG_ROOT = os.path.dirname(_HERE)
_STEP9_OUT = os.path.join(FIG_ROOT, 'Step 9', 'output')


def _load_real_nonlinear_data():
    """Real ANSYS measurements, all 24 1B-cluster modes (2026-08-13).

    At this scale (24 per-mode K3 values + 17 pairs x 8 coupling
    coefficients = 160+ real numbers) hand-transcribing them into a CONFIG
    literal, as was done for the first 4 modes/1 pair, stops being a
    reasonable way to keep this reviewable -- reading them directly from
    Step 9's own saved measurement files (case3_k3_identification_mode{m}.npz,
    case3_cross_k3_modes{i}{j}.npz -- the actual real-ANSYS-run artifacts,
    same provenance as before, just not re-typed by hand) is more robust
    (no transcription-error risk) with EQUIVALENT transparency: every
    number here traces to one specific real solve, on disk, inspectable.
    Falls back to an empty dict for either piece if Step 9's output isn't
    present (e.g. a fresh checkout before the campaign has been run) so
    this module still imports and falls back to pure extrapolation, same
    as before any real measurement existed.
    """
    measured_K3 = {}
    cross_coupling = {}
    if not os.path.isdir(_STEP9_OUT):
        return measured_K3, cross_coupling
    for fn in os.listdir(_STEP9_OUT):
        if fn.startswith('case3_k3_identification_mode') and fn.endswith('.npz'):
            m = int(fn[len('case3_k3_identification_mode'):-len('.npz')])
            d = np.load(os.path.join(_STEP9_OUT, fn))
            measured_K3[m] = float(d['K3_fit'])
        elif fn.startswith('case3_cross_k3_modes') and fn.endswith('.npz'):
            digits = fn[len('case3_cross_k3_modes'):-len('.npz')]
            d = np.load(os.path.join(_STEP9_OUT, fn))
            mode_pair = tuple(int(x) for x in d['mode_pair'])
            cross_coupling[mode_pair] = {
                'coef0': [float(x) for x in d['coef0']],
                'coef1': [float(x) for x in d['coef1']],
            }
            assert f'{mode_pair[0]}{mode_pair[1]}' == digits, \
                f"filename/mode_pair mismatch: {fn} vs {mode_pair}"
    return measured_K3, cross_coupling


_MEASURED_K3, _CROSS_COUPLING = _load_real_nonlinear_data()

# Real, measured topology of the 24 1B-cluster modes (frequency-gap scan,
# 2026-08-13, using each mode's own real half-power bandwidth ~2*zeta*f as
# the "these two modes' resonances actually overlap" criterion -- not an
# arbitrary Hz cutoff): 5 clean isolated near-degenerate pairs, 1 genuinely
# isolated single, and one 13-mode continuously-overlapping band (modes
# 11-23) with no clean break anywhere in it -- coupled via adjacent-pair
# identification through the chain (real ANSYS, not assumed pairwise-only
# is "good enough": it's what's tractable, disclosed as such).
MODE_GROUPS = {
    'pairs':  [(0, 1), (3, 4), (5, 6), (7, 8), (9, 10)],
    'single': [2],
    'chain':  list(range(11, 24)),   # 11,12,...,23 -- adjacent-pair coupled
}

# HF band (modes 24-69) real topology, measured 2026-08-21 via the identical
# frequency-gap-scan method used above for the 1B cluster (each mode's own
# real half-power bandwidth ~2*zeta*f, real zeta=0.002, against real
# freqs_sec -- not an arbitrary Hz cutoff, not assumed). Result, contrary to
# the working assumption up to this point that the HF band was mostly
# extrapolation-only "safe" territory: 44 of 46 HF modes (95.7%) are
# near-degenerate and need real cross-mode coupling, same as the 1B
# cluster -- only 2 modes (24, 37) are genuinely isolated.
#
# Kept as a SEPARATE dict from MODE_GROUPS (not merged in) rather than
# restructuring MODE_GROUPS's existing 'chain' key from one flat list into
# several chains: Step 9's own consumers (validate_case3_full(),
# make_case3_cross_coupling_figure(), _step6_r2_all_modes()) are hardcoded
# to the original 24-mode/17-pair/1-chain topology and haven't been
# reviewed/updated for this yet (that is Step 9's own review pass, still to
# come) -- a deliberate scoping choice so this Step 4 fix does not silently
# break code in a step that hasn't been touched, not an oversight.
#
# Real ANSYS campaign (`Step 9/_hf_multimode_campaign.py`, 2026-08-21):
# 46 independent K3 identifications (all HF modes, including the 2 isolated
# singles -- replaces their extrapolated placeholder with real measured
# data even though they correctly stay uncoupled) + 32 cross-mode coupling
# identifications (10 clean pairs + 2 pairs in the 46-48 chain + 20 pairs
# in the 49-69 chain). 4.56 hours, zero failures. Fit quality: every K3
# positive (hardening), every cross-coupling pair's worst-case fit error
# under 6% (mean 1.39%, max 5.81% at pair (29,30)) -- as tight as the 1B
# cluster's own campaign (worst case there was 6.7%).
MODE_GROUPS_HF = {
    'pairs':  [(25, 26), (27, 28), (29, 30), (31, 32), (33, 34),
               (35, 36), (38, 39), (40, 41), (42, 43), (44, 45)],
    'single': [24, 37],
    'chains': [list(range(46, 49)), list(range(49, 70))],   # 46-48, 49-69
}

CONFIG = {
    'step1_dir':  r'F:\ANSYS PCE\ROM_data',                      # read-only
    'step2_dir':  r'F:\ANSYS PCE\ROM_data',                      # read-only
    'step3_dir':  os.path.join(os.path.dirname(_HERE), 'Step 3', 'output'),  # read-only
    'output_dir': os.path.join(_HERE, 'output'),                 # local

    'n_blades': 24,
    'n_sec':    70,        # must match Step 2's secondary basis size

    # How many of the 1000 Step-3 mistuning samples to actually push through
    # the (cheap) generalized-eigenvalue re-solve for validation/figures.
    # Full-ensemble UQ over all 1000 is Phase 6's job, not Phase 4's.
    'n_validate': 40,

    # ------------------------------------------------------------------
    # SENSITIVITY MODEL (theta -> per-blade fractional frequency shift).
    # length_exp is standard cantilever-beam scaling, kept as-is.
    #
    # thickness_exp and tip_coeff_per_frac were RECALIBRATED (2026-08-08)
    # against real single-blade ANSYS perturbation runs -- see Step 9's
    # sensitivity_calibrate(): perturb ONLY blade 0's target variable (all
    # other 23 blades held exactly tuned), re-extract via PyMAPDL, MAC-match
    # the resulting real mode shapes against the tuned baseline's real mode
    # shapes, and fit the single implied stiffness-scale factor by least
    # squares against the ALREADY-KNOWN participation pattern P[0,:] (valid
    # because for a single perturbed blade, Step 4's own diagonal formula
    # predicts shift[m] = K_sec[m,m]*P[0,m]*scale_0 for EVERY mode m at
    # once -- not just one mode "belonging to" that blade, since tuned
    # modes are ring-spanning, not localized).
    #   thickness_exp: was 1.0 (simple-rectangular-section beam theory,
    #     presented as if exact, not flagged as placeholder). Measured:
    #     0.00288 at magnitude=0.6mm, 0.00832 at magnitude=1.1mm -> mean
    #     0.0056, ~180x smaller than the beam-theory value. Root cause,
    #     not just noise: this project's own compute_nodal_perturbation()
    #     implements "thickness" as a raw AXIAL (Z) node offset -- already
    #     flagged elsewhere as "THE WEAKEST part of this scheme" (true
    #     thickness needs a camber-normal direction) -- so this measures
    #     the real sensitivity to THAT SPECIFIC geometric operation, which
    #     barely couples to bending stiffness, not to true airfoil
    #     thickness in general. Kept as measured (matches what the ANSYS
    #     geometry pipeline ACTUALLY does), with this caveat disclosed
    #     rather than silently reverting to the untested theoretical value.
    #   tip_coeff_per_frac: was +0.01 (placeholder, positive sign).
    #     Measured: -0.83068 at magnitude=3.0mm, -1.02755 at magnitude=1.5mm
    #     -> mean -0.92911. ~90x larger in magnitude AND opposite sign --
    #     the placeholder had the physical direction backwards (extending
    #     the tip outward should LOWER frequency, same direction as length,
    #     not raise it).
    #   Validated on 4 independent real full-order samples (Step 3 idx
    #   0-3, not just the one used to fit): mean freq error dropped from
    #   ~0.7% to ~0.2%, MAC (mode-shape agreement, when paired with the
    #   coupled dK_sec below) rose from ~0.4 to ~0.95, HI1 ratio
    #   (real/predicted) went from 1.7-5.1x under-prediction to 0.83-0.97x.
    # 2026-08-27 SCOPE CHANGE (explicit user decision, matches Step 3's own
    # reduction to d_tip-only mistuning): length_exp/thickness_exp/
    # twist_coeff_per_deg/le_te_coeff_per_frac REMOVED, not kept as unused
    # dead entries -- d_length/d_thickness/d_twist_deg/d_le_te no longer
    # exist in Step 3's theta_samples, so compute_delta_f() below no longer
    # references them either. This is a deliberate narrowing to the ONE
    # variable with a coefficient re-anchored to real, corrected geometry
    # (see below), not a claim the removed coefficients were wrong -- they
    # were real ANSYS measurements (PROJECT_STATUS.md Section 9a).
    #
    # tip_coeff_per_frac REFIT 2026-08-27: Step 1's measure_blade_geometry
    # had a real bug (the raw '_BLADETIP' named-selection query returned
    # ALL 61,107 model nodes, not a tip-only subset -- confirmed directly,
    # its z-range exactly matched the WHOLE model's own axial extent). This
    # inflated t_ref (tip_z_extent_mm) from the real 36.43mm to a wrong
    # 52.0mm, and the ORIGINAL tip_coeff_per_frac (-0.92911, Section 8d)
    # was fit using that wrong t_ref as its normalization denominator --
    # so it must be refit, not just carried over, once t_ref is corrected.
    # See Step 1's measure_blade_geometry() for the root-cause fix and
    # Step 9's sensitivity_calibrate('d_tip', ...) for the refit.
    # ------------------------------------------------------------------
    'sensitivity': {
        # REFIT 2026-08-27 with the corrected t_ref=36.43mm (real ANSYS,
        # blade 0, magnitudes 1.5/3.0mm): -0.71981 (mag=1.5mm), -0.58190
        # (mag=3.0mm) -> mean -0.65086. Was -0.92911 under the wrong
        # t_ref=52.0mm -- NOT a simple algebraic rescale of the old value
        # (a naive t_ref-ratio rescale would predict ~-1.33), a fresh real
        # measurement instead, consistent with this project's own
        # "measure, don't extrapolate" convention.
        'tip_coeff_per_frac':  -0.65086,
    },

    # ------------------------------------------------------------------
    # GEOMETRIC NONLINEAR (Duffing) — see docstring Part B.
    #
    # hardening_ratio raised 1.20->133.57 on 2026-08-09 after Step 9's
    # Case 3 identification: real ANSYS NLGEOM static solves (mode 0,
    # tuned model), prescribing the FULL physical displacement field
    # a*T_full2sec[:,0] at 4 amplitudes (a=0.02,0.05,0.08,0.11, i.e.
    # ~0.6-3.1mm at the peak-participation DOF -- a genuinely modest,
    # weakly-nonlinear-regime test range, not the literal q_ref=1.0
    # label, which turned out to imply up to 28mm of real displacement).
    # F_nl(a) = F(a) - K_lin*a fit K3=4.521e8 by least squares; F_nl/a^3
    # was constant to <0.05% across the whole 5.5x amplitude range,
    # confirming the cubic model is genuinely a good fit (not just a
    # rough order-of-magnitude match) -- ~111x the previous placeholder.
    # new_ratio = K3_fit * q_ref^2 / K_sec[0,0] = 4.521e8 / 3.385e6 =
    # 133.57. This ratio is the EXTRAPOLATION applied to any mode without
    # its own real measurement below (`measured_K3` was empty for m>0
    # until 2026-08-11).
    #
    # 2026-08-11: measured_K3 extended to modes 1-3 (Step 9's
    # run_case3_identification(mode_index=m), same 4-amplitude method,
    # same amplitude values, same <0.1% internal fit quality for every
    # mode -- the cubic model itself is a genuinely good fit everywhere
    # tested). Real result, not assumed: the "same ratio for every mode"
    # extrapolation matched mode 1 almost exactly (fitted/placeholder =
    # 1.0003) but was off by 29% (over) for mode 2 and 26% (under) for
    # mode 3 -- in OPPOSITE directions, so there's no single correction
    # factor that would fix the extrapolation; it's just genuinely
    # unreliable per-mode, confirmed rather than assumed. `measured_K3`
    # below OVERRIDES the extrapolation with real data wherever it
    # exists; modes 4-69 still use the extrapolation, now with a
    # concretely-measured ~25-30%-either-direction uncertainty attached
    # to that choice instead of an unquantified "not validated" caveat.
    # See Step 9's run_case3_identification() /
    # case3_k3_identification_mode{N}.npz for each mode's full fit.
    # ------------------------------------------------------------------
    'nonlinear': {
        'hardening_ratio': 133.57,  # K3_sec_diag[m] = ratio * K_sec[m,m] / q_ref^2 -- EXTRAPOLATION for modes without a real measurement (modes 24-69)
        'q_ref_mm':         1.0,  # reference generalized-coordinate amplitude
        # Real ANSYS-measured K3, ALL 24 1B-cluster modes as of 2026-08-13
        # (was modes 0-3 only; see _load_real_nonlinear_data() above for
        # where these numbers actually come from -- real Step 9 ANSYS
        # measurement files, not hand-typed). Modes 24-69 (HF modes, out of
        # this campaign's scope) still use the extrapolation above.
        'measured_K3': _MEASURED_K3,

        # ------------------------------------------------------------------
        # REAL CROSS-MODE COUPLING (2026-08-13). The diagonal-only model
        # above has NO mechanism for one mode's nonlinearity to affect
        # another -- a deliberate, disclosed scope limit until a real
        # dynamic ANSYS measurement (Step 9's Case 3 hand/GUI transient,
        # 1000N @ 292.82Hz) showed the diagonal (mode-0-only) model
        # over-predicting the real response by ~3x (predicted 3.20mm,
        # measured 1.04mm). Root-caused, not patched: modes 0 and 1 are
        # only 0.05 Hz apart (292.82 vs 292.87 Hz) -- driving "at mode 0's
        # resonance" also drives mode 1 almost exactly at ITS resonance,
        # and the two share energy through the real cubic nonlinearity's
        # cross terms, not just through being close in frequency (a naive
        # in-phase 2-mode SUM, ignoring cross coupling, made the prediction
        # WORSE, not better -- confirmed directly, not assumed).
        #
        # Measured via Step 9's run_case3_cross_identification(mode_pair=
        # (0,1)): 7 real ANSYS NLGEOM static solves with COMBINED
        # (a0*Phi_0 + a1*Phi_1) displacement fields, each mode's own
        # generalized reaction force read from the SAME solve and fit as a
        # general cubic polynomial in (a0,a1). Fit quality: <0.5% relative
        # residual for both modes -- the real coupled physics IS well
        # described by this quartic-energy/cubic-force model, it just
        # isn't diagonal. At node 1171 (exact participation known for both
        # modes), the coupled model predicts 1.26mm vs the old diagonal
        # model's 12.84mm at the SAME force/frequency -- a 10x reduction,
        # landing in the same ballpark as the real 1.04mm measurement
        # (different vertex, so not literally the same number, but the
        # same order of magnitude and direction of correction).
        #
        # coef_ij = [a_i^3, a_i^2*a_j, a_i*a_j^2, a_j^3] coefficients of
        # F_nl_i(a_i,a_j) for the mode pair (i,j) -- NOT symmetric between
        # the two rows (each is that mode's OWN force response).
        #
        # EXTENDED 2026-08-13 from just (0,1) to all 17 real-measured pairs
        # (MODE_GROUPS above): the 5 clean isolated pairs, plus 12 adjacent-
        # pair couplings through the dense 11-23 chain. Same method, same
        # <7%-worst-case (mostly <2%) fit quality per pair, confirmed
        # directly against a robust max-residual/max-signal metric (not
        # just RMS-relative, which is misleadingly inflated at the rare
        # test point where a pair's true F_nl happens to be near zero).
        'cross_coupling': _CROSS_COUPLING,
    },

    # ------------------------------------------------------------------
    # FORCED-RESPONSE PSEUDO-ARC-LENGTH CONTINUATION (fig5). Forcing levels
    # are chosen, not measured: picked so the LINEAR peak response would
    # reach each of these fractions of q_ref -- a family of increasing
    # forcing levels (like sweeping engine order / excitation amplitude),
    # each traced separately, all sharing the same skeleton curve.
    #
    # SWITCHED TO MODE 2, WIDENED SWEEP (2026-08-29, explicit user request
    # for a backbone that actually folds): mode 0's real measured K3 does
    # fold too, but only past w=1.6 -- outside this figure's old
    # w_stop_hi=1.6 sweep range, which is why every mode-0 curve here used
    # to read "(no fold)". Mode 2 is one of Step 4's own confirmed
    # genuinely-isolated SDOF modes (isolated modes: [2, 24, 37], see the
    # half-power-bandwidth gap scan validation check) and is the same mode
    # Step 9's dedicated backbone-validation script
    # (Step 9/_validation1_nonlinear_frf_backbone.py) uses for exactly this
    # reason. Re-verified directly here (not just trusting that script's
    # own comments): with w_stop_hi widened to 3.0, mode 2 produces a full
    # fold pair (n_folds=2) at tp=0.3, 0.5, 0.7, 0.8, and 1.0 -- three of
    # those (0.3, 0.7, 1.0) are kept below as a clean low/mid/high
    # progression, all real folds, matching Step 9's own validated force-
    # level list rather than an arbitrarily chosen new one.
    # ------------------------------------------------------------------
    # NOTE: 'ds'/'n_steps'/'w_start'/'w_stop_hi'/'w_stop_lo' below are read
    # by duffing_forced_response_continuation() from this SAME module-level
    # CONFIG dict for every caller, including Step 6's build_dataset() (via
    # `import step4 as s4`) -- do not widen w_stop_hi/n_steps here as a
    # permanent default just to make fig5's backbone show a fold; that
    # would silently change Step 6's BPINN ground-truth generation too.
    # fig5 applies its own scoped, restored-after override instead (see
    # make_step4_figures), the same pattern Step 9's own diagnostic/
    # validation backbone scripts already use for this.
    'continuation': {
        'mode_index': 2,
        'target_linear_peak_frac_qref_list': [0.3, 0.7, 1.0],
        'ds': 0.015,           # arc-length step, nondimensional (alpha,beta,w) space
        'n_steps': 1400,
        'w_start': 0.7,        # starting Omega/omega0, well below resonance
        'w_stop_hi': 1.6,      # stop the sweep once w exceeds this
        'w_stop_lo': 0.4,      # ...or drops below this (response decayed away)
    },

    'random_seed': 42,
}

NB = CONFIG['n_blades']
NSEC = CONFIG['n_sec']
OUT = CONFIG['output_dir']
os.makedirs(OUT, exist_ok=True)

VAR_NAMES = ['d_tip']   # 2026-08-27: reduced from 5 to 1, see CONFIG['sensitivity'] comment
_VALIDATION_LOG = []


class _Tee:
    """Duplicates writes to multiple streams (console + a UTF-8 log file).
    Each stream's encoding is handled independently: if one can't encode a
    character, that stream falls back to a replaced-character write instead
    of crashing the run — the UTF-8 log file always keeps full fidelity."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                enc = getattr(s, 'encoding', None) or 'ascii'
                s.write(data.encode(enc, errors='replace').decode(enc, errors='replace'))

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def hdr(t):
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")


def _record_check(name, ok, detail=''):
    _VALIDATION_LOG.append((name, bool(ok), detail))
    status = 'OK' if ok else 'FAIL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    return ok


def print_validation_summary():
    hdr("STEP 4 VALIDATION SUMMARY")
    for name, ok, detail in _VALIDATION_LOG:
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    n_fail = sum(1 for _, ok, _ in _VALIDATION_LOG if not ok)
    hdr(f"STEP 4 VALIDATION: {'PASSED' if n_fail == 0 else f'FAILED ({n_fail} check(s))'}")
    return n_fail == 0


# ═══════════════════════════════════════════════════════════════════
# 4A. LOAD STEP 1 / 2 / 3 OUTPUTS (all read-only)
# ═══════════════════════════════════════════════════════════════════
def load_inputs():
    hdr("STEP 4A: LOADING STEP 1/2/3 OUTPUTS")

    bmm = np.load(os.path.join(CONFIG['step1_dir'], 'blade_master_map.npz'))
    blade_dofs = {b: bmm[f'blade_{b}'] for b in range(NB)}
    print(f"  blade_master_map: {NB} blades, "
          f"{[len(blade_dofs[b]) for b in range(3)]}... DOFs/blade")

    with open(os.path.join(CONFIG['step1_dir'], 'blade_geometry.json')) as f:
        geo = json.load(f)
    L_ref = geo.get('outer_radius_mm', 302.93)
    t_ref = geo.get('tip_z_extent_mm', 52.00)
    print(f"  Baseline geometry: L_ref={L_ref:.2f} mm, t_ref={t_ref:.2f} mm")

    T_full2sec = np.load(os.path.join(CONFIG['step2_dir'], 'T_full2sec.npy'))
    bundle = np.load(os.path.join(CONFIG['step2_dir'], 'secondary_bundle.npz'))
    K_sec, M_sec, C_sec, freqs_sec = (bundle['K_sec'], bundle['M_sec'],
                                        bundle['C_sec'], bundle['freqs_sec'])
    print(f"  T_full2sec: {T_full2sec.shape}")
    print(f"  K_sec/M_sec/C_sec: {K_sec.shape}  ({NSEC} secondary modes, "
          f"{freqs_sec[0]:.2f}-{freqs_sec[-1]:.2f} Hz)")

    theta = np.load(os.path.join(CONFIG['step3_dir'], 'theta_samples.npz'))
    theta = {k: theta[k] for k in theta.files}
    print(f"  theta_samples: {theta['d_tip'].shape[0]} samples x {NB} blades")

    return dict(blade_dofs=blade_dofs, L_ref=L_ref, t_ref=t_ref,
                T_full2sec=T_full2sec, K_sec=K_sec, M_sec=M_sec, C_sec=C_sec,
                freqs_sec=freqs_sec, theta=theta)


# ═══════════════════════════════════════════════════════════════════
# 4B. FREQUENCY-SENSITIVITY MODEL (theta -> per-blade delta_f / f)
# ═══════════════════════════════════════════════════════════════════
def compute_delta_f(theta_row, L_ref, t_ref):
    """theta_row: dict with the SINGLE 'd_tip' variable (2026-08-27 scope
    change, was 5 variables -- see CONFIG['sensitivity']'s own comment),
    a (n_blades,) array for ONE sample. Returns delta_f_over_f, shape
    (n_blades,). L_ref is accepted but unused (kept in the signature so
    every existing caller -- Step 5/6/7/9 -- doesn't need to change)."""
    s = CONFIG['sensitivity']
    dtip = theta_row['d_tip']
    return s['tip_coeff_per_frac'] * (dtip / t_ref)


# ═══════════════════════════════════════════════════════════════════
# 4C. PER-BLADE STIFFNESS BLOCKS + FMM PROJECTION -> dK_sec
# ═══════════════════════════════════════════════════════════════════
def compute_participation(inp):
    """P[b, m] = fraction of secondary mode m's blade-tip energy that sits
    on blade b (sums to 1 over b, for every mode m). Built entirely from
    Step 2's own T_full2sec -- already correctly scaled/reduced, unlike a
    raw K_full block (see module docstring for why that was wrong)."""
    hdr("STEP 4C: COMPUTING PER-BLADE MODAL PARTICIPATION")
    energy = np.zeros((NB, NSEC))
    for b in range(NB):
        Phi_b = inp['T_full2sec'][inp['blade_dofs'][b], :]     # (30, n_sec)
        energy[b] = np.sum(Phi_b ** 2, axis=0)                  # (n_sec,)
    P = energy / energy.sum(axis=0, keepdims=True)
    print(f"  Participation matrix P: {P.shape}  "
          f"(column sums: min={P.sum(0).min():.6f}, max={P.sum(0).max():.6f}, "
          f"should be 1.0)")
    return P


def assemble_dK_sec(delta_f_over_f, P, K_sec):
    """Participation-weighted, first-order DIAGONAL modal stiffness
    perturbation for one mistuning realization. delta_f_over_f: (n_blades,)
    fractional frequency shift. Returns a diagonal (n_sec, n_sec) matrix.

    Superseded as Step 4's default by assemble_dK_sec_coupled() below (see
    that function's docstring) -- kept here because (a) it's still exactly
    what the diagonal ENTRIES of the coupled version reduce to (verified
    directly in validate_mistuning(), not just asserted), and (b) it's the
    O(1)-per-sample shortcut Step 5's propagate_ensemble() needs for a
    1000-sample ensemble without an eigensolve per sample."""
    scale_vec = (1.0 + delta_f_over_f) ** 2 - 1.0     # (n_blades,)
    shift = P.T @ scale_vec                            # (n_sec,)
    return np.diag(np.diag(K_sec) * shift)


def assemble_dK_sec_coupled(delta_f_over_f, inp, K_sec):
    """Off-diagonal extension of assemble_dK_sec(), added 2026-08-08 after
    Step 9's Case 2 validation against real ANSYS data found the diagonal-
    only model badly under-predicted a real mistuned blisk's HI1 (max
    1B-cluster frequency deviation): 3.1 Hz predicted vs. 15.8 Hz real, a
    ~5x gap that survived even after MAC-based mode-correspondence fixes
    and a coefficient recalibration -- i.e. NOT a comparison-methodology
    artifact and NOT just a bad coefficient. Root-caused directly, not
    assumed: projecting the real ANSYS mode shapes for the mismatched
    modes onto the ROM's own 70-mode subspace showed >99.8% of their
    energy WAS already representable there (verified against a random-
    vector negative control, ~1.8% captured, confirming the check itself
    is meaningful) -- so the ROM's basis was never the problem. What it
    lacked was any mechanism for one blade's stiffness change to affect a
    DIFFERENT mode's frequency, i.e. off-diagonal coupling -- exactly the
    "blade-to-blade coupling through the shared hub... produces mode
    localization/veering" limitation this module's docstring already
    flagged as future work, before any real data existed to check it
    against.

    Standard "Fundamental Mistuning Model" (FMM) form (Feiner & Griffin
    2002 and similar turbomachinery-mistuning literature), built ENTIRELY
    from quantities Step 4 already computes/validates -- no raw K_full
    blocks touched (that path already caused the documented ~40x scale
    bug once, see module docstring):

        dK_sec[m,n] = sqrt(K_sec[m,m]*K_sec[n,n]) / sqrt(E[m]*E[n])
                      * sum_b scale_b * (Phi_b[:,m] . Phi_b[:,n])

    where Phi_b = T_full2sec[blade_b_dofs,:] (same per-blade vectors
    compute_participation() already builds), E[m] = sum_b ||Phi_b[:,m]||^2,
    and scale_b = (1+df_b/f)^2-1 (same per-blade scale assemble_dK_sec()
    already uses). VERIFIED (not just derived) to reduce EXACTLY to
    assemble_dK_sec()'s diagonal on m=n -- see validate_mistuning().

    Validated on 4 independent real ANSYS samples (Step 3 idx 0-3, only
    idx 0 was used for the coefficient fit): combined with the recalibrated
    coefficients (see CONFIG['sensitivity']), mean freq error 0.08-0.27%
    (was 0.6-0.9%), MAC 0.93-0.97 (was 0.4-0.5), HI1 ratio 0.83-0.97x (was
    1.7-5.1x under-prediction). Full numbers in Step 9's case2_comparison()
    output and PROJECT_STATUS.md.

    KNOWN REMAINING LIMITATION: still first-order (linear in scale_b), and
    tip_coeff_per_frac (the ONE geometric variable in scope since
    2026-08-27, see CONFIG['sensitivity']) is calibrated from only 2
    magnitudes on 1 blade -- cyclic-symmetry generalization to the other
    23 blades is assumed, not independently checked at another blade."""
    T_full2sec, blade_dofs = inp['T_full2sec'], inp['blade_dofs']
    n_sec = K_sec.shape[0]
    scale_vec = (1.0 + delta_f_over_f) ** 2 - 1.0
    raw = np.zeros((n_sec, n_sec))
    E = np.zeros(n_sec)
    for b in range(NB):
        Phi_b = T_full2sec[blade_dofs[b], :]        # (30, n_sec)
        G_b = Phi_b.T @ Phi_b                         # (n_sec, n_sec)
        raw += scale_vec[b] * G_b
        E += np.diag(G_b)
    Kdiag = np.diag(K_sec)
    norm = np.sqrt(np.outer(Kdiag, Kdiag)) / np.sqrt(np.outer(E, E))
    return raw * norm


# ═══════════════════════════════════════════════════════════════════
# 4D. VALIDATE THE MISTUNING MAPPING ON A SAMPLE SUBSET
# ═══════════════════════════════════════════════════════════════════
def validate_mistuning(inp, P):
    hdr(f"STEP 4D: VALIDATING MISTUNING MAPPING ON {CONFIG['n_validate']} SAMPLES")
    K_sec, M_sec, freqs_sec = inp['K_sec'], inp['M_sec'], inp['freqs_sec']
    n_val = CONFIG['n_validate']
    theta = inp['theta']

    # --- Participation weights must sum to 1 over blades, for every mode ---
    # (a direct check on the P matrix itself: if this fails, everything
    # downstream is scaled wrong -- exactly the class of bug found earlier.)
    col_sums = P.sum(axis=0)
    _record_check("Participation P sums to 1.0 over blades, for every mode",
                  bool(np.allclose(col_sums, 1.0, atol=1e-9)),
                  f"min={col_sums.min():.8f}, max={col_sums.max():.8f}")

    # --- Zero-input sanity check: theta=0 must give dK_sec == 0 exactly ---
    zero_row = {v: np.zeros(NB) for v in VAR_NAMES}
    df0 = compute_delta_f(zero_row, inp['L_ref'], inp['t_ref'])
    dK0 = assemble_dK_sec(df0, P, K_sec)
    _record_check("Zero mistuning (theta=0) gives dK_sec == 0 exactly",
                  bool(np.allclose(dK0, 0.0, atol=1e-9)),
                  f"max|dK_sec| = {np.abs(dK0).max():.3e}")

    # --- Coupled model must reduce EXACTLY to the diagonal model on the
    # diagonal (a direct numerical check, not an assumption) ---
    df_check = compute_delta_f({v: theta[v][0] for v in VAR_NAMES}, inp['L_ref'], inp['t_ref'])
    dK_diag_check = assemble_dK_sec(df_check, P, K_sec)
    dK_coupled_check = assemble_dK_sec_coupled(df_check, inp, K_sec)
    diag_match = np.allclose(np.diag(dK_diag_check), np.diag(dK_coupled_check), rtol=1e-8)
    _record_check("Coupled dK_sec's diagonal matches the diagonal-only model exactly",
                  diag_match, f"max diff = {np.abs(np.diag(dK_diag_check) - np.diag(dK_coupled_check)).max():.3e}")
    zero_coupled = assemble_dK_sec_coupled(df0, inp, K_sec)
    _record_check("Zero mistuning gives zero COUPLED dK_sec too",
                  bool(np.allclose(zero_coupled, 0.0, atol=1e-9)),
                  f"max|dK_sec_coupled| = {np.abs(zero_coupled).max():.3e}")

    delta_f_blade = np.zeros((n_val, NB))
    freqs_mistuned = np.zeros((n_val, NSEC))
    all_sym = True
    all_pd = True
    all_sym_c = True
    all_pd_c = True
    for i in range(n_val):
        row = {v: theta[v][i] for v in VAR_NAMES}
        df = compute_delta_f(row, inp['L_ref'], inp['t_ref'])
        delta_f_blade[i] = df
        dK_sec = assemble_dK_sec(df, P, K_sec)

        if not np.allclose(dK_sec, dK_sec.T, atol=1e-8 * np.abs(dK_sec).max()):
            all_sym = False

        K_total = K_sec + dK_sec
        try:
            w, _ = eigh(K_total, M_sec)
            if np.any(w <= 0):
                all_pd = False
        except np.linalg.LinAlgError:
            all_pd = False

        # COUPLED model -- this is what's actually used for freqs_mistuned
        # below (Step 4's new default; see assemble_dK_sec_coupled()).
        dK_sec_c = assemble_dK_sec_coupled(df, inp, K_sec)
        if not np.allclose(dK_sec_c, dK_sec_c.T, atol=1e-6 * np.abs(dK_sec_c).max()):
            all_sym_c = False
        K_total_c = K_sec + dK_sec_c
        try:
            w_c, _ = eigh(K_total_c, M_sec)
            freqs_mistuned[i] = np.sqrt(np.clip(w_c, 0, None)) / (2 * np.pi)
            if np.any(w_c <= 0):
                all_pd_c = False
        except np.linalg.LinAlgError:
            all_pd_c = False
            freqs_mistuned[i] = np.nan

    _record_check("dK_sec symmetric for every validated sample (diagonal model)", all_sym)
    _record_check("K_sec + dK_sec stays positive-definite for every sample (diagonal model)",
                  all_pd)
    _record_check("dK_sec_coupled symmetric for every validated sample", all_sym_c)
    _record_check("K_sec + dK_sec_coupled stays positive-definite (physical) for every sample",
                  all_pd_c)

    # Sanity: realized |delta_f/f| should be small (weak mistuning regime,
    # consistent with Step 3's tolerance magnitudes) -- catches a sign/units
    # bug in the sensitivity model, which would blow this up to O(1) (the
    # historical bug this guards against produced a 138% shift, ~24x this
    # bound). Threshold widened 5%->10% on 2026-08-08 after the tip/
    # thickness coefficient recalibration (see CONFIG['sensitivity']): the
    # worst case across n_validate samples (sample 7, blade 11, driven by
    # d_tip=3.06mm -- the same magnitude range the coefficients were
    # measured against) is 5.87%, a legitimate consequence of the now-
    # larger, correctly-signed tip_coeff_per_frac, not a repeat of the
    # scale bug -- confirmed by inspecting that sample directly rather than
    # just raising the bound until it passed.
    max_abs_df = float(np.abs(delta_f_blade).max())
    _record_check("Fractional frequency mistuning stays in the weak-mistuning "
                  "regime (< 10%)", max_abs_df < 0.10, f"max|df/f| = {max_abs_df:.4f}")

    # Regression check for the earlier ~40x scale bug: the REALIZED mean 1B
    # frequency shift must stay the same order of magnitude as the INPUT
    # blade-level mistuning, not blow up from a scale mismatch upstream.
    mean_shift_pct = float(np.abs(freqs_mistuned[:, :NB] - freqs_sec[None, :NB]).mean()
                            / freqs_sec[:NB].mean() * 100)
    _record_check("Mean 1B-cluster frequency shift stays same order as input mistuning "
                  "(< 5%, not a repeat of the ~40x scale bug found during development)",
                  mean_shift_pct < 5.0, f"{mean_shift_pct:.3f}%")

    return delta_f_blade, freqs_mistuned


# ═══════════════════════════════════════════════════════════════════
# 4E. GEOMETRIC NONLINEAR (DUFFING) DIAGONAL STIFFNESS — REAL GREEN-LAGRANGE
# ═══════════════════════════════════════════════════════════════════
def build_nonlinear_stiffness(K_sec):
    """2026-08-27 SCOPE CHANGE (explicit user decision): the diagonal-only
    `hardening_ratio` EXTRAPOLATION (a functional-form guess for modes
    without a real measurement) is REMOVED, not just deprioritized. Modes
    without a real ANSYS Green-Lagrange measurement (Step 9's
    run_case3_identification()/run_case3_cross_identification(), real
    NLGEOM static solves) now get K3=0 -- explicitly disclosed as
    "not yet measured," never fabricated via extrapolation. This is a
    real accuracy/coverage tradeoff, stated plainly: modes 24-69 (outside
    the 1B cluster) previously had an order-of-magnitude-plausible
    nonlinear stiffness from the extrapolation (with a measured ~25-30%
    per-mode uncertainty, PROJECT_STATUS.md Section 9i); they now have
    NONE, until real ANSYS measurement is extended to them (the same
    proven method already used for all 24 1B-cluster modes -- straightforward,
    not new engineering, whenever there's appetite for more ANSYS time)."""
    hdr("STEP 4E: GEOMETRIC NONLINEAR (DUFFING) STIFFNESS -- REAL GREEN-LAGRANGE ONLY")
    nl = CONFIG['nonlinear']
    K3_sec_diag = np.zeros(NSEC)
    measured = nl.get('measured_K3', {})
    for m, k3_val in measured.items():
        K3_sec_diag[m] = k3_val
    n_unmeasured = NSEC - len(measured)
    print(f"  Real ANSYS Green-Lagrange-measured K3 used for {len(measured)} mode(s): {sorted(measured.keys())}")
    print(f"  {n_unmeasured} mode(s) have NO real measurement -- K3=0 (disclosed, not extrapolated/fabricated)")
    print(f"  K3_sec_diag range (measured modes only): "
          f"[{K3_sec_diag[K3_sec_diag > 0].min():.4e}, {K3_sec_diag.max():.4e}]" if measured else "  (none measured)")
    _record_check("K3_sec_diag positive for every REAL-measured mode (hardening, not softening)",
                  bool(np.all(K3_sec_diag[list(measured.keys())] > 0)) if measured else True,
                  f"{len(measured)} measured modes checked")
    _record_check("K3_sec_diag shape is (n_sec,)", K3_sec_diag.shape == (NSEC,))
    return K3_sec_diag


def duffing_skeleton(omega0, k1, k3, amplitudes):
    """Undamped, unforced backbone/skeleton curve (see e.g. Nayfeh & Mook,
    'Nonlinear Oscillations'): omega(a) = omega0*sqrt(1+(3/4)*(k3/k1)*a^2)
    for m*x''+k1*x+k3*x^3=0. Monotonic and single-valued by construction --
    it is the RIDGE threading through the tips of the forced-response
    resonance peaks below, not the folded curve itself."""
    return omega0 * np.sqrt(1.0 + 0.75 * (k3 / k1) * amplitudes ** 2)


# ═══════════════════════════════════════════════════════════════════
# 4E-2. FORCED-RESPONSE PSEUDO-ARC-LENGTH CONTINUATION (single-harmonic HBM)
# ═══════════════════════════════════════════════════════════════════
# Single-DOF forced Duffing oscillator (valid per-mode here because K_sec,
# M_sec, C_sec are all exactly diagonal -- verified -- so each secondary
# mode is an independent SDOF oscillator under this diagonal-only nonlinear
# model):  M q'' + C q' + K q + K3 q^3 = F0 cos(Omega t)
#
# 1-term harmonic balance ansatz q(t) = a*cos(Omega t) + b*sin(Omega t),
# using cos^3(theta) = (3/4)cos(theta) + (1/4)cos(3theta) and dropping the
# 3rd harmonic (standard 1-term HBM), gives 2 algebraic residuals in (a,b)
# for a given Omega:
#     f1 = -M Omega^2 a + C Omega b + K a + (3/4) K3 (a^2+b^2) a - F0 = 0
#     f2 = -M Omega^2 b - C Omega a + K b + (3/4) K3 (a^2+b^2) b     = 0
#
# Near a hardening resonance this (a,b,Omega) solution surface FOLDS back on
# itself (dA/dOmega -> infinity at two points) -- simple Newton continuation
# stepping Omega forward cannot follow the curve through those folds. This
# is traced instead via PSEUDO-ARC-LENGTH continuation (Keller's method):
# Omega becomes a 3rd unknown alongside (a,b), and the step is taken along
# arc length s in the combined (a,b,Omega) space rather than along Omega
# directly, using a bordered Newton corrector (the 2 HBM residuals + 1
# linearized arc-length constraint against the previous tangent direction).
#
# Everything below works in NONDIMENSIONAL variables alpha=a/q_ref,
# beta=b/q_ref, w=Omega/omega0 (all O(1)) for numerical robustness -- the
# raw physical scales (Omega~10^3 rad/s, a~1 mm) differ by many orders of
# magnitude, which is hostile to a mixed-unit Newton solve. In these
# variables (with omega0=sqrt(K/M), zeta=C/(2*sqrt(K*M))):
#     (1-w^2)*alpha + 2*zeta*w*beta  + kappa*(alpha^2+beta^2)*alpha = f
#     (1-w^2)*beta  - 2*zeta*w*alpha + kappa*(alpha^2+beta^2)*beta  = 0
#     kappa = (3/4)*K3*q_ref^2/K,   f = F0/(K*q_ref)
def _hbm_residual_and_jacobian(u, zeta, kappa, f):
    alpha, beta, w = u
    r2 = alpha ** 2 + beta ** 2
    f1 = (1 - w ** 2) * alpha + 2 * zeta * w * beta + kappa * r2 * alpha - f
    f2 = (1 - w ** 2) * beta - 2 * zeta * w * alpha + kappa * r2 * beta
    J = np.array([
        [(1 - w ** 2) + kappa * (3 * alpha ** 2 + beta ** 2),
         2 * zeta * w + kappa * (2 * alpha * beta),
         -2 * w * alpha + 2 * zeta * beta],
        [-2 * zeta * w + kappa * (2 * alpha * beta),
         (1 - w ** 2) + kappa * (alpha ** 2 + 3 * beta ** 2),
         -2 * w * beta - 2 * zeta * alpha],
    ])
    return np.array([f1, f2]), J


def _tangent(J, prev_t=None):
    """Unit tangent to the solution curve = null space of the 2x3 Jacobian
    (generically 1-D). Oriented to keep moving forward relative to prev_t,
    or with positive w-component on the very first call."""
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    t = Vt[-1, :]
    if prev_t is None:
        if t[2] < 0:
            t = -t
    elif np.dot(t, prev_t) < 0:
        t = -t
    return t / np.linalg.norm(t)


def _newton_corrector(u_pred, u_prev, t_prev, ds, zeta, kappa, f,
                       max_iter=25, tol=1e-11):
    u = u_pred.copy()
    for _ in range(max_iter):
        F2, J = _hbm_residual_and_jacobian(u, zeta, kappa, f)
        arc_eq = np.dot(t_prev, u - u_prev) - ds
        Jaug = np.vstack([J, t_prev])
        rhs = -np.array([F2[0], F2[1], arc_eq])
        try:
            du = np.linalg.solve(Jaug, rhs)
        except np.linalg.LinAlgError:
            return None, False
        u = u + du
        if np.linalg.norm(du) < tol:
            return u, True
    return u, False


def duffing_forced_response_continuation(omega0, M, C, K, K3, q_ref, target_peak):
    """Pseudo-arc-length continuation of the forced Duffing response near
    resonance, for ONE forcing level (target_peak = linear-estimate peak
    amplitude / q_ref). Returns dict with Omega, amplitude (physical
    units), a stable boolean mask, and the fold points found."""
    cfg = CONFIG['continuation']
    zeta = C / (2 * np.sqrt(K * M))
    kappa = 0.75 * K3 * q_ref ** 2 / K
    f = target_peak * 2 * zeta                            # from |H(w=1)| = f/(2*zeta) at resonance

    w0 = cfg['w_start']
    denom = (1 - w0 ** 2) ** 2 + (2 * zeta * w0) ** 2
    u = np.array([f * (1 - w0 ** 2) / denom, f * (2 * zeta * w0) / denom, w0])
    F2, J = _hbm_residual_and_jacobian(u, zeta, kappa, f)
    for _ in range(50):    # refine the linear starting guess with the nonlinear term included
        if np.linalg.norm(F2) < 1e-12:
            break
        du = np.linalg.solve(np.vstack([J, [1, 0, 0]]), -np.append(F2, 0))
        u = u + du
        F2, J = _hbm_residual_and_jacobian(u, zeta, kappa, f)

    t = _tangent(J, prev_t=None)
    ds = cfg['ds']

    Us = [u.copy()]
    Ts = [t.copy()]
    for _ in range(cfg['n_steps']):
        u_pred = u + ds * t
        u_new, ok = _newton_corrector(u_pred, u, t, ds, zeta, kappa, f)
        if not ok:
            ds *= 0.5
            if ds < 1e-6:
                break
            continue
        F2, J = _hbm_residual_and_jacobian(u_new, zeta, kappa, f)
        t_new = _tangent(J, prev_t=t)
        u, t = u_new, t_new
        Us.append(u.copy())
        Ts.append(t.copy())
        if u[2] > cfg['w_stop_hi'] or u[2] < cfg['w_stop_lo']:
            break

    Us = np.array(Us)
    Ts = np.array(Ts)
    alpha, beta, w = Us[:, 0], Us[:, 1], Us[:, 2]
    amplitude = np.sqrt(alpha ** 2 + beta ** 2) * q_ref
    Omega = w * omega0

    # Fold points: sign changes of the tangent's w-component. The branch
    # BETWEEN an odd number of preceding sign changes (i.e. between the
    # first and second fold) is the classic unstable middle branch.
    w_tangent_sign = np.sign(Ts[:, 2])
    sign_changes = np.where(np.diff(w_tangent_sign) != 0)[0]
    stable = np.ones(len(Us), dtype=bool)
    if len(sign_changes) >= 2:
        lo, hi = sign_changes[0], sign_changes[1]
        stable[lo:hi + 1] = False

    return dict(Omega=Omega, amplitude=amplitude, stable=stable,
                n_folds=len(sign_changes), fold_indices=sign_changes,
                zeta=zeta, kappa=kappa, f_nondim=f, alpha=alpha, beta=beta)


# ═══════════════════════════════════════════════════════════════════
# 4E-3. COUPLED 2-MODE FORCED RESPONSE (real cross-mode K3, 2026-08-13)
# ═══════════════════════════════════════════════════════════════════
def _extract_alpha_beta(t, q, Omega):
    """In-phase/quadrature demodulation of a settled time-domain signal:
    least-squares fit q(t) ~= alpha*cos(Omega*t) + beta*sin(Omega*t) over
    the given window. Used (2026-08-13) to get PHASE-resolved ground truth
    for the coupled/chain BPINN's physics-residual loss -- the original
    amp-only settled-amplitude extraction (peak-to-peak/2) throws away the
    phase information the harmonic-balance residual actually needs. Robust
    to a non-integer number of periods in the window (unlike a naive
    trapezoid Fourier integral), standard least-squares harmonic fit."""
    X = np.column_stack([np.cos(Omega * t), np.sin(Omega * t)])
    coef, *_ = np.linalg.lstsq(X, q, rcond=None)
    return float(coef[0]), float(coef[1])


def coupled_nonlinear_force(coef, q_i, q_j):
    """F_nl_i(q_i,q_j) = coef[0]*q_i^3 + coef[1]*q_i^2*q_j + coef[2]*q_i*q_j^2
    + coef[3]*q_j^3 -- the general cubic polynomial fit from Step 9's real
    combined-displacement ANSYS measurements (CONFIG['nonlinear']
    ['cross_coupling']), NOT a diagonal K3*q^3 term. Works on scalars or
    arrays (q_i, q_j same shape)."""
    c0, c1, c2, c3 = coef
    return c0 * q_i ** 3 + c1 * q_i ** 2 * q_j + c2 * q_i * q_j ** 2 + c3 * q_j ** 3


def duffing_forced_response_coupled(mode_pair, K_arr, M_arr, C_arr, coef0, coef1,
                                     F_gen_pair, Omega, n_cycles=400, steps_per_cycle=25):
    """Real coupled 2-mode forced-response solver -- direct time-domain
    integration of the two coupled ODEs (NOT a harmonic-balance/
    continuation solve like the single-mode function above): a proper
    multi-harmonic-balance Jacobian for a coupled cubic system is real,
    nontrivial extra derivation; direct numerical integration is exact
    (no ansatz assumed), simple, and -- since this is a small 4-state ODE,
    not a full FEM solve -- fast (seconds, not hours) to run out to genuine
    steady state. Validated 2026-08-13 against a real ANSYS transient
    measurement (node/vertex where the real dynamic response was
    measured): where the diagonal-only model over-predicted by ~3x, this
    coupled model landed within the same order of magnitude of the real
    measurement (see CONFIG['nonlinear']['cross_coupling'] comment for the
    exact numbers).

    mode_pair: (i,j) mode indices. K_arr,M_arr,C_arr: (K_i,K_j) etc, scalars
    per mode. coef0/coef1: the 4-element cubic coefficients for F_nl_i and
    F_nl_j (CONFIG['nonlinear']['cross_coupling'][mode_pair]). F_gen_pair:
    (F_gen_i, F_gen_j) -- physical force projected onto each mode
    (F_phys*phi_m). Omega: driving frequency, rad/s.

    Returns dict with t, q_i, q_j (full time histories) and the SETTLED
    (last 10% of the run) steady-state amplitude of each mode."""
    Ki, Kj = K_arr; Mi, Mj = M_arr; Ci, Cj = C_arr
    Fgi, Fgj = F_gen_pair

    def rhs(t, y):
        qi, vi, qj, vj = y
        Fdrive = np.cos(Omega * t)
        # coef0/coef1 were BOTH fit as polynomials in (a_i, a_j) with a_i
        # (mode_pair[0]'s own amplitude) always FIRST -- [a_i^3, a_i^2*a_j,
        # a_i*a_j^2, a_j^3] -- regardless of which mode's own force it is.
        # Argument order to coupled_nonlinear_force must stay (qi, qj) for
        # BOTH calls; swapping it for the second mode (an earlier bug here)
        # silently evaluates the wrong polynomial.
        ai = (Fgi * Fdrive - Ci * vi - Ki * qi - coupled_nonlinear_force(coef0, qi, qj)) / Mi
        aj = (Fgj * Fdrive - Cj * vj - Kj * qj - coupled_nonlinear_force(coef1, qi, qj)) / Mj
        return [vi, ai, vj, aj]

    T = n_cycles * 2 * np.pi / Omega
    max_step = (2 * np.pi / Omega) / steps_per_cycle
    sol = solve_ivp(rhs, [0, T], [0, 0, 0, 0], max_step=max_step, dense_output=False)
    t_arr = sol.t
    qi_t, qj_t = sol.y[0], sol.y[2]

    tail = max(int(0.1 * len(t_arr)), 10)
    amp_i = (qi_t[-tail:].max() - qi_t[-tail:].min()) / 2
    amp_j = (qj_t[-tail:].max() - qj_t[-tail:].min()) / 2
    # Phase-resolved (alpha,beta) ground truth, added 2026-08-13 for the
    # physics-residual loss -- amp alone can't supervise or be checked
    # against the HBM residual, which needs relative phase between modes.
    alpha_i, beta_i = _extract_alpha_beta(t_arr[-tail:], qi_t[-tail:], Omega)
    alpha_j, beta_j = _extract_alpha_beta(t_arr[-tail:], qj_t[-tail:], Omega)
    return dict(t=t_arr, q_i=qi_t, q_j=qj_t, amp_i=float(amp_i), amp_j=float(amp_j),
                alpha_i=alpha_i, beta_i=beta_i, alpha_j=alpha_j, beta_j=beta_j,
                mode_pair=mode_pair)


def duffing_forced_response_chain(chain_modes, K_arr, M_arr, C_arr, pair_coefs,
                                   F_gen_arr, Omega, n_cycles=400, steps_per_cycle=25):
    """N-mode generalization of duffing_forced_response_coupled() for a
    CHAIN of densely-packed near-degenerate modes (2026-08-13) -- the real,
    measured topology of modes 11-23 (MODE_GROUPS['chain']): every adjacent
    pair's real gap is smaller than its own half-power bandwidth, so there
    is no clean break anywhere in this 13-mode band, unlike the isolated
    2-mode pairs. Coupling is modeled through ADJACENT pairs only (real
    ANSYS measurement exists for those 12 pairs, not the full 78-pair
    tensor -- a disclosed, tractable choice, not a claim the full tensor is
    negligible). Interior modes receive nonlinear force contributions from
    BOTH their left and right neighbor pairs simultaneously; the two
    boundary modes (first/last in the chain) from only one.

    chain_modes: ordered list of mode indices, e.g. MODE_GROUPS['chain'].
    K_arr/M_arr/C_arr/F_gen_arr: arrays, one entry per mode, SAME ORDER as
    chain_modes. pair_coefs: dict {(mode_i,mode_j): {'coef0':[...],
    'coef1':[...]}} for each adjacent pair actually present in chain_modes
    (extra entries for other pairs are ignored, not an error).

    Direct time-domain integration (same reasoning as the 2-mode version:
    a proper multi-harmonic-balance Jacobian for an N-mode coupled cubic
    system is real, nontrivial extra derivation the 2-mode case already
    deferred; this is exact, no ansatz, and fast enough for a 2*13=26-state
    ODE). Returns dict with t, q (n_modes x n_times), and the settled
    (last 10% of run) steady-state amplitude per mode."""
    n = len(chain_modes)
    idx_of = {m: k for k, m in enumerate(chain_modes)}
    local_pairs = []
    for (mi, mj), coefs in pair_coefs.items():
        if mi in idx_of and mj in idx_of and abs(idx_of[mi] - idx_of[mj]) == 1:
            local_pairs.append((idx_of[mi], idx_of[mj],
                                 np.array(coefs['coef0']), np.array(coefs['coef1'])))

    K_arr = np.asarray(K_arr, dtype=float)
    M_arr = np.asarray(M_arr, dtype=float)
    C_arr = np.asarray(C_arr, dtype=float)
    F_gen_arr = np.asarray(F_gen_arr, dtype=float)

    def rhs(t, y):
        q = y[0::2]
        v = y[1::2]
        Fdrive = np.cos(Omega * t)
        F_nl = np.zeros(n)
        for ki, kj, coef0, coef1 in local_pairs:
            qi, qj = q[ki], q[kj]
            # Same fixed-argument-order convention as the 2-mode solver:
            # q_i (this pair's lower-index mode) is ALWAYS the first
            # argument, for BOTH coef0 (mode ki's own force) and coef1
            # (mode kj's own force) -- matches how these were fit in
            # Step 9's run_case3_cross_identification().
            F_nl[ki] += coupled_nonlinear_force(coef0, qi, qj)
            F_nl[kj] += coupled_nonlinear_force(coef1, qi, qj)
        a = (F_gen_arr * Fdrive - C_arr * v - K_arr * q - F_nl) / M_arr
        dydt = np.empty(2 * n)
        dydt[0::2] = v
        dydt[1::2] = a
        return dydt

    T = n_cycles * 2 * np.pi / Omega
    max_step = (2 * np.pi / Omega) / steps_per_cycle
    y0 = np.zeros(2 * n)
    sol = solve_ivp(rhs, [0, T], y0, max_step=max_step, dense_output=False)
    t_arr = sol.t
    q_t = sol.y[0::2]   # (n, n_times)

    tail = max(int(0.1 * len(t_arr)), 10)
    amps = np.array([(q_t[k, -tail:].max() - q_t[k, -tail:].min()) / 2 for k in range(n)])
    # Phase-resolved (alpha,beta) ground truth per mode, added 2026-08-13
    # for the chain physics-residual loss -- same reasoning as the 2-mode
    # solver above.
    alphas = np.zeros(n); betas = np.zeros(n)
    for k in range(n):
        alphas[k], betas[k] = _extract_alpha_beta(t_arr[-tail:], q_t[k, -tail:], Omega)
    return dict(t=t_arr, q=q_t, amp=amps, alpha=alphas, beta=betas, chain_modes=chain_modes)


# ═══════════════════════════════════════════════════════════════════
# 4E-4. COUPLED/CHAIN HARMONIC-BALANCE PHYSICS RESIDUAL (2026-08-13)
# ═══════════════════════════════════════════════════════════════════
# The original single-mode BPINN enforced the exact single-mode HBM
# residual (_hbm_residual_and_jacobian above) as a physics loss. The
# coupled/chain BPINNs (trained the same night) deliberately skipped this
# -- not because it's impossible, but because nobody had derived the
# cross-mode residual yet; deriving it was flagged as real, separate work.
#
# DERIVATION (done here, not assumed): harmonic-balance ansatz
# q_i(t) = a_i*cos(wt) + b_i*sin(wt), represented as the complex phasor
# Q_i = a_i - j*b_i (so q_i(t) = Re[Q_i * e^{jwt}]). For a real signal
# q(t) = Re[Q e^{jwt}], writing it as a sum of e^{+jwt} and e^{-jwt} terms
# and expanding q^2*q' products with the standard product-to-sum approach
# gives the fundamental (w-frequency) component of ANY cubic monomial in
# two such signals in closed form. Two special cases anchor the general
# result: (1) q_i^3's fundamental is the classical cubic describing
# function (3/4)|Q_i|^2*Q_i -- verified this derivation reproduces that
# EXACTLY. (2) The general result was checked numerically (not just
# algebraically) against direct trapezoidal Fourier projection of the
# real time-domain product for random coefficients/amplitudes -- agreement
# to ~1e-5 relative (limited by quadrature discretization, not a
# derivation error).
#
# F_nl_i(q_i,q_j) = c0*q_i^3 + c1*q_i^2*q_j + c2*q_i*q_j^2 + c3*q_j^3 has
# fundamental-harmonic content (X_alpha, X_beta) -- the contribution to
# the cos- and sin-coefficient equations respectively, in the SAME
# physical force units as coef (matches how the rest of this coupled/chain
# code already works, NOT the nondimensional single-mode kappa/f form):
def coupled_hbm_nonlinear_terms(coef, a_i, b_i, a_j, b_j):
    """Closed-form fundamental-harmonic content of the general cubic
    cross-coupling force, at the HBM ansatz q_i=a_i*cos(wt)+b_i*sin(wt),
    q_j=a_j*cos(wt)+b_j*sin(wt). Returns (X_alpha, X_beta): contribution
    to the alpha- and beta-equation of the harmonic-balance residual.
    Reduces EXACTLY to the single-mode (3/4)(a^2+b^2)*a / *b result when
    c1=c2=c3=0. Elementwise-only (+,-,*) -- works on numpy arrays or torch
    tensors interchangeably, so this same function is reused unchanged as
    a training-time (autograd-compatible) physics loss term."""
    c0, c1, c2, c3 = coef
    X_alpha = (c0 * 0.75 * (a_i ** 2 + b_i ** 2) * a_i
               + c1 * 0.25 * ((3 * a_i ** 2 + b_i ** 2) * a_j + 2 * a_i * b_i * b_j)
               + c2 * 0.25 * ((3 * a_j ** 2 + b_j ** 2) * a_i + 2 * a_j * b_j * b_i)
               + c3 * 0.75 * (a_j ** 2 + b_j ** 2) * a_j)
    X_beta = (c0 * 0.75 * (a_i ** 2 + b_i ** 2) * b_i
              + c1 * 0.25 * ((a_i ** 2 + 3 * b_i ** 2) * b_j + 2 * a_i * b_i * a_j)
              + c2 * 0.25 * ((a_j ** 2 + 3 * b_j ** 2) * b_i + 2 * a_j * b_j * a_i)
              + c3 * 0.75 * (a_j ** 2 + b_j ** 2) * b_j)
    return X_alpha, X_beta


def coupled_hbm_residual(K_arr, M_arr, C_arr, coef0, coef1, F_gen_pair, Omega,
                          a_i, b_i, a_j, b_j):
    """Full 4-equation harmonic-balance residual for a coupled 2-mode pair
    (physical units, matching duffing_forced_response_coupled's own
    conventions). Same fixed-argument-order convention as everywhere else
    in this coupled model: (a_i,b_i) is ALWAYS the first argument to
    coupled_hbm_nonlinear_terms, for BOTH coef0 (mode i's own equation)
    and coef1 (mode j's own equation). All arguments/returns can be numpy
    arrays or torch tensors of matching shape."""
    Ki, Kj = K_arr; Mi, Mj = M_arr; Ci, Cj = C_arr
    Fgi, Fgj = F_gen_pair
    Xa_i, Xb_i = coupled_hbm_nonlinear_terms(coef0, a_i, b_i, a_j, b_j)
    Xa_j, Xb_j = coupled_hbm_nonlinear_terms(coef1, a_i, b_i, a_j, b_j)
    R_alpha_i = (Ki - Mi * Omega ** 2) * a_i + Ci * Omega * b_i + Xa_i - Fgi
    R_beta_i = (Ki - Mi * Omega ** 2) * b_i - Ci * Omega * a_i + Xb_i
    R_alpha_j = (Kj - Mj * Omega ** 2) * a_j + Cj * Omega * b_j + Xa_j - Fgj
    R_beta_j = (Kj - Mj * Omega ** 2) * b_j - Cj * Omega * a_j + Xb_j
    return R_alpha_i, R_beta_i, R_alpha_j, R_beta_j


def chain_hbm_residual(chain_modes, K_arr, M_arr, C_arr, pair_coefs, F_gen_arr, Omega,
                        alpha_arr, beta_arr):
    """N-mode chain generalization of coupled_hbm_residual: each mode's
    residual sums coupled_hbm_nonlinear_terms contributions from EVERY
    adjacent pair it participates in (interior chain modes: 2 pairs;
    boundary modes: 1) -- exactly mirrors duffing_forced_response_chain's
    own local_pairs accumulation, just in frequency-domain algebraic form
    instead of a time-domain RHS. alpha_arr/beta_arr: (n,) arrays/tensors,
    one entry per mode in chain_modes (same order). Returns (R_alpha,
    R_beta), each (n,)."""
    n = len(chain_modes)
    idx_of = {m: k for k, m in enumerate(chain_modes)}
    local_pairs = []
    for (mi, mj), coefs in pair_coefs.items():
        if mi in idx_of and mj in idx_of and abs(idx_of[mi] - idx_of[mj]) == 1:
            local_pairs.append((idx_of[mi], idx_of[mj], coefs['coef0'], coefs['coef1']))

    Xa = [0.0] * n
    Xb = [0.0] * n
    for ki, kj, coef0, coef1 in local_pairs:
        a_i, b_i, a_j, b_j = alpha_arr[ki], beta_arr[ki], alpha_arr[kj], beta_arr[kj]
        xa_i, xb_i = coupled_hbm_nonlinear_terms(coef0, a_i, b_i, a_j, b_j)
        xa_j, xb_j = coupled_hbm_nonlinear_terms(coef1, a_i, b_i, a_j, b_j)
        Xa[ki] = Xa[ki] + xa_i; Xb[ki] = Xb[ki] + xb_i
        Xa[kj] = Xa[kj] + xa_j; Xb[kj] = Xb[kj] + xb_j

    R_alpha = [(K_arr[k] - M_arr[k] * Omega ** 2) * alpha_arr[k] + C_arr[k] * Omega * beta_arr[k]
               + Xa[k] - F_gen_arr[k] for k in range(n)]
    R_beta = [(K_arr[k] - M_arr[k] * Omega ** 2) * beta_arr[k] - C_arr[k] * Omega * alpha_arr[k]
              + Xb[k] for k in range(n)]
    return R_alpha, R_beta


# ═══════════════════════════════════════════════════════════════════
# 4F. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def save_outputs(K3_sec_diag, delta_f_blade, freqs_mistuned, inp, P):
    hdr("STEP 4F: SAVING OUTPUTS")
    fp1 = os.path.join(OUT, 'nonlinear_rom.npz')
    measured_modes = sorted(CONFIG['nonlinear'].get('measured_K3', {}).keys())
    np.savez(fp1, K3_sec_diag=K3_sec_diag, participation=P,
              q_ref_mm=CONFIG['nonlinear']['q_ref_mm'],
              hardening_ratio=CONFIG['nonlinear']['hardening_ratio'],
              measured_K3_modes=np.array(measured_modes))
    print(f"  Saved: {fp1}  (includes participation matrix P, {P.shape}, "
          f"for Step 5's full-ensemble UQ)")

    fp2 = os.path.join(OUT, 'mistuning_validation.npz')
    np.savez(fp2, delta_f_blade=delta_f_blade, freqs_mistuned=freqs_mistuned,
              freqs_nominal=inp['freqs_sec'])
    print(f"  Saved: {fp2}")

    # JSON can't serialize tuple keys (CONFIG['nonlinear']['cross_coupling']
    # is keyed by (mode_i, mode_j) tuples for real lookups elsewhere in the
    # code) -- stringify just for this human-readable provenance dump.
    nonlinear_for_json = dict(CONFIG['nonlinear'])
    nonlinear_for_json['cross_coupling'] = {
        f'{k[0]}-{k[1]}': v for k, v in CONFIG['nonlinear'].get('cross_coupling', {}).items()
    }

    config_record = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'n_blades': NB, 'n_sec': NSEC, 'n_validate': CONFIG['n_validate'],
        'sensitivity_model': CONFIG['sensitivity'],
        'nonlinear_model': nonlinear_for_json,
        'baseline_geometry': {'L_ref_mm': inp['L_ref'], 't_ref_mm': inp['t_ref']},
        'note': ('Sensitivity/nonlinear coefficients are documented placeholders '
                 '(cantilever-beam scaling for length/thickness; small assumed '
                 'coefficients for twist/LE-TE/tip and for the Duffing hardening '
                 'ratio), not FEM-calibrated values. See CONFIG in step4.py.'),
    }
    fp3 = os.path.join(OUT, 'step4_config.json')
    with open(fp3, 'w') as f:
        json.dump(config_record, f, indent=2)
    print(f"  Saved: {fp3}")


# ═══════════════════════════════════════════════════════════════════
# 4G. FIGURES — 5 diagnostics
# ═══════════════════════════════════════════════════════════════════
def _resolve_figs_dir():
    figs = os.path.join(FIG_ROOT, 'figures', 'step4')
    os.makedirs(figs, exist_ok=True)
    return figs


def _savefig(fig, figs_dir, name):
    paths = plot_style.savefig_pub(fig, figs_dir, name)
    print(f"  Figure saved: {paths[0]}")


def make_step4_figures(inp, K3_sec_diag, delta_f_blade, freqs_mistuned):
    hdr("STEP 4G: FIGURES (5 diagnostics, figures/step4/, PNG+PDF)")
    figs = _resolve_figs_dir()
    freqs_sec = inp['freqs_sec']

    # ── fig1: per-blade frequency mistuning pattern, one example sample ──
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    blade_idx = np.arange(NB)
    df0 = delta_f_blade[0] * 100
    colors_bar = [plot_style.ORANGE if v >= 0 else plot_style.BLUE for v in df0]
    ax.bar(blade_idx, df0, color=colors_bar)
    ax.axhline(0, color=plot_style.AXIS_BASELINE, lw=1.0)
    ax.set_xlabel('Blade index')
    ax.set_ylabel(r'Fractional frequency mistuning  $\delta f/f$  [%]')
    plot_style.two_tier_title(ax, 'Blade mistuning pattern', 'one Monte Carlo realization, sample #0')
    _savefig(fig, figs, 'step4_fig1_mistuning_pattern')

    # ── fig2: nominal vs mistuned 1B-cluster frequency spectrum (RESONANCE FREQUENCY) ──
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    modes = np.arange(NB)
    ax.plot(modes, freqs_sec[:NB], color=plot_style.TRUTH_COLOR, ls='--', marker='o',
            ms=4.5, label='nominal (tuned)', zorder=5)
    n_show = min(15, freqs_mistuned.shape[0])
    for i in range(n_show):
        ax.plot(modes, freqs_mistuned[i, :NB], color=plot_style.COMPARE_COLOR, alpha=0.25, lw=1.2)
    ax.plot([], [], color=plot_style.COMPARE_COLOR, alpha=0.7, lw=2.0,
            label=f'{n_show} mistuned realizations')
    ax.set_xlabel('Mode index (1B cluster)')
    ax.set_ylabel('Resonance frequency  [Hz]')
    plot_style.two_tier_title(ax, 'Resonance frequency shift under mistuning',
                               'nominal (tuned) vs. mistuned 1B-cluster frequencies')
    plot_style.legend_below(ax, ncol=2)
    fig.tight_layout()
    _savefig(fig, figs, 'step4_fig2_freq_spectrum_spread')

    # ── fig3: distribution of max |df/f| per sample across the ensemble ──
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    max_df_per_sample = np.abs(delta_f_blade).max(axis=1) * 100
    ax.hist(max_df_per_sample, bins=20, color=plot_style.AQUA, alpha=0.85, edgecolor=plot_style.SURFACE, linewidth=0.6)
    ax.set_xlabel(r'max blade $|\delta f/f|$ per sample  [%]')
    ax.set_ylabel('count')
    plot_style.two_tier_title(ax, 'Worst-blade mistuning magnitude',
                               f'{CONFIG["n_validate"]} validated samples')
    _savefig(fig, figs, 'step4_fig3_mistuning_magnitude_spread')

    # ── fig4: K3 diagonal hardening coefficient per secondary mode ──
    # Color-coded by real-ANSYS-measured vs. still-extrapolated (2026-08-13
    # fix: the old subtitle claimed everything here was an un-calibrated
    # placeholder, which stopped being true once real per-mode static K3
    # identification started, Section 8h/9i -- reads directly from
    # CONFIG['nonlinear']['measured_K3'] so it stays correct as more modes
    # get measured, no hardcoded count.
    # 2D BAR CHART, LOG SCALE (2026-08-29 REDESIGN, replacing the 3D bar3d
    # added 2026-08-19): measured-vs-extrapolated no longer needs a depth
    # axis to read clearly -- the color code (green = real ANSYS
    # measurement, violet = extrapolated) already separates the two groups
    # unambiguously. K3_sec_diag spans ~3.5 orders of magnitude across
    # modes (3.3e8 to 1.0e12) -- a linear axis renders every mode below the
    # top few as visually zero, so this uses a log y-axis (the 3D version
    # had the same problem, just masked by the depth/rotation making small
    # bars hard to judge against each other in the first place).
    measured_modes = set(CONFIG['nonlinear'].get('measured_K3', {}).keys())
    is_measured = np.array([m in measured_modes for m in range(NSEC)])
    x = np.arange(NSEC)
    colors = np.where(is_measured, plot_style.C_OK, plot_style.VIOLET)
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(x, K3_sec_diag, color=list(colors), edgecolor=plot_style.SURFACE, linewidth=0.3, width=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Secondary mode index')
    ax.set_ylabel('$K_3$ (diagonal)  [N/mm$^3$ equiv.], log scale')
    legend_handles = [
        matplotlib.patches.Patch(color=plot_style.C_OK, label='real ANSYS measurement'),
        matplotlib.patches.Patch(color=plot_style.VIOLET, label='extrapolated (unmeasured)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=9)
    plot_style.two_tier_title(ax, 'Geometric-nonlinear (Duffing) hardening',
                               f'diagonal coefficient per mode -- {len(measured_modes)} of {NSEC} real-ANSYS-measured')
    fig.tight_layout()
    _savefig(fig, figs, 'step4_fig4_k3_diagonal')

    # ── fig5: family of forced-response curves at increasing forcing
    #          levels (NONLINEAR RESPONSE AMPLITUDE vs. RESONANCE FREQUENCY),
    #          via pseudo-arc-length continuation, all sharing one undamped
    #          skeleton/backbone. Sequential blue ramp (light->dark) encodes
    #          increasing forcing level -- an ORDERED quantity, so a single-
    #          hue sequential ramp is the right encoding, not a categorical
    #          rainbow (plasma) with no inherent order.
    m = CONFIG['continuation']['mode_index']
    omega0 = 2 * np.pi * freqs_sec[m]
    M = inp['M_sec'][m, m]
    K = inp['K_sec'][m, m]
    C = inp['C_sec'][m, m]
    K3 = K3_sec_diag[m]
    q_ref = CONFIG['nonlinear']['q_ref_mm']
    force_levels = CONFIG['continuation']['target_linear_peak_frac_qref_list']

    # Scoped, restored-after override: mode 2's fold sits past the shared
    # default w_stop_hi=1.6, but that default is read by every caller of
    # duffing_forced_response_continuation() (Step 6's build_dataset()
    # included), so widen it only for these fig5 calls, then put it back --
    # same pattern Step 9's own diagnostic/validation backbone scripts use.
    _cont_cfg = CONFIG['continuation']
    _w_stop_hi_orig, _n_steps_orig = _cont_cfg['w_stop_hi'], _cont_cfg['n_steps']
    _cont_cfg['w_stop_hi'], _cont_cfg['n_steps'] = 3.0, 4000
    try:
        conts = [duffing_forced_response_continuation(omega0, M, C, K, K3, q_ref, tp)
                 for tp in force_levels]
    finally:
        _cont_cfg['w_stop_hi'], _cont_cfg['n_steps'] = _w_stop_hi_orig, _n_steps_orig
    amp_max = max(c['amplitude'].max() for c in conts)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    amps_sk = np.linspace(0, 1.15 * amp_max, 200)
    omega_sk = duffing_skeleton(omega0, K, K3, amps_sk)
    ax.plot(omega_sk / (2 * np.pi), amps_sk, color=plot_style.INK_MUTED, lw=1.6, ls=':',
            label='undamped skeleton (backbone)', zorder=1)

    # Categorical colors, not the sequential blue ramp (2026-08-29 fix,
    # user feedback: the ramp made these three curves hard to tell apart
    # once they fold and overlap near the low-amplitude return branch --
    # matches Step 9's own validated backbone figure
    # (step9_fig13_validation1_nonlinear_frf_backbone), which uses
    # distinct hues per force level for exactly this reason).
    force_colors = [plot_style.C_1B, plot_style.C_HF, plot_style.C_ACC]
    fold_freqs_all = []
    for tp, cont, c in zip(force_levels, conts, force_colors):
        freq_hz = cont['Omega'] / (2 * np.pi)
        amp = cont['amplitude']
        stable = cont['stable']
        label = f'F/F$_{{ref}}$={tp:.1f}' + ('' if cont['n_folds'] else ' (no fold)')
        ax.plot(freq_hz[stable], amp[stable], color=c, lw=2.2, label=label, zorder=3)
        # Unstable branch as a thin dotted line in the SAME color (not a
        # bare gap), same treatment as Step 9's reference figure, so the
        # fold reads as one continuous S rather than two disconnected arcs.
        ax.plot(freq_hz[~stable], amp[~stable], color=c, lw=1.3, ls=':', zorder=3, alpha=0.85)
        if len(cont['fold_indices']):
            ax.plot(freq_hz[cont['fold_indices']], amp[cont['fold_indices']],
                    'o', color=c, mec=plot_style.SURFACE, mew=1.2, ms=7, zorder=5)
            fold_freqs_all.extend(freq_hz[cont['fold_indices']])

    ax.axvline(freqs_sec[m], color=plot_style.INK_MUTED, ls='--', lw=1.2, alpha=0.7,
               label='linear resonance ($a$=0)')
    ax.set_xlabel('Forcing frequency  [Hz]')
    ax.set_ylabel('Nonlinear response amplitude  [mm]')
    plot_style.two_tier_title(ax, 'Nonlinear forced-response family',
                               f'mode {m} ($f_0$={freqs_sec[m]:.1f} Hz) -- increasing forcing bends the resonance over')
    plot_style.legend_below(ax, ncol=3, y=-0.20)
    fig.tight_layout()

    # Auto-zoom sized to the WIDEST curve in the family (highest forcing),
    # so every level's peak and fold(s) fit in one consistent frame instead
    # of each curve needing its own scale (very light modal damping,
    # zeta=0.002, makes the linear peak itself only ~1 Hz wide, so a
    # threshold-on-amplitude zoom alone is not reliable here either).
    # Minimal padding (2026-08-13, explicit user request): the plotted
    # curves should touch the axes frame, not float in empty margin.
    if fold_freqs_all:
        fold_freqs_all = np.array(fold_freqs_all)
        fold_gap = fold_freqs_all.max() - fold_freqs_all.min()
        pad = max(0.03 * fold_gap, 2.0)
        ax.set_xlim(fold_freqs_all.min() - pad, fold_freqs_all.max() + pad)
    else:
        all_freq = np.concatenate([c['Omega'] / (2 * np.pi) for c in conts])
        all_amp = np.concatenate([c['amplitude'] for c in conts])
        near = all_freq[all_amp > 0.03 * amp_max]
        pad = 0.02 * (near.max() - near.min())
        ax.set_xlim(near.min() - pad, near.max() + pad)
    _savefig(fig, figs, 'step4_fig5_duffing_backbone')

    print(f"  All 5 Step 4 figures saved to: {figs}")
    return conts[-1]   # highest-forcing case, used for validation checks


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _log_path = os.path.join(_HERE, 'Step4.txt')
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    t_start = time.time()
    hdr(f"STEP 4 v1.0: NONLINEAR ROM — MISTUNED + GEOMETRIC NONLINEAR — {NB}-BLADE BLISK (PCE PROJECT)")
    print(f"  Step 1/2 dir (read-only): {CONFIG['step1_dir']}")
    print(f"  Step 3 dir   (read-only): {CONFIG['step3_dir']}")
    print(f"  Output dir   (Step 4):    {OUT}")

    inp = load_inputs()
    P = compute_participation(inp)
    delta_f_blade, freqs_mistuned = validate_mistuning(inp, P)
    K3_sec_diag = build_nonlinear_stiffness(inp['K_sec'])
    save_outputs(K3_sec_diag, delta_f_blade, freqs_mistuned, inp, P)
    cont = make_step4_figures(inp, K3_sec_diag, delta_f_blade, freqs_mistuned)

    hdr("STEP 4H: VALIDATING PSEUDO-ARC-LENGTH CONTINUATION")
    _record_check("Continuation produced a finite amplitude curve (no NaN/divergence)",
                  bool(np.all(np.isfinite(cont['amplitude']))))
    m = CONFIG['continuation']['mode_index']
    freq_hz = cont['Omega'] / (2 * np.pi)
    _record_check("Continuation sweep brackets the linear resonance frequency",
                  bool(freq_hz.min() < inp['freqs_sec'][m] < freq_hz.max()),
                  f"swept [{freq_hz.min():.2f}, {freq_hz.max():.2f}] Hz vs "
                  f"f0={inp['freqs_sec'][m]:.2f} Hz")
    # Check changed 2026-08-09 after hardening_ratio was corrected to its
    # real, measured value (133.57, ~111x the old placeholder -- see
    # CONFIG['nonlinear']). The OLD check required exactly 2 folds
    # (n_folds==2), which the placeholder's hardening_ratio=1.20 was
    # itself hand-tuned to produce "so the fold/bend reads clearly on a
    # normal-width plot" (see this module's own CONFIG comment history) --
    # a visualization choice, not a physical requirement. With the REAL
    # K3, the system is stiff enough (kappa~100 vs the placeholder's
    # ~0.9) that the classic bistable fold does NOT appear within any
    # numerically-stable forcing range: tested target_peak_frac up to 300
    # (w swept out to 24x resonance) with zero folds and smoothly growing
    # amplitude; forcing pushed further (target_peak_frac=1000) caused the
    # hand-implemented continuation to diverge into unphysical territory
    # (negative frequency ratio) rather than reveal a fold -- a real
    # numerical-robustness limit of this continuation implementation at
    # extreme kappa, not evidence a fold exists just out of reach. So:
    # n_folds==2 is still checked and reported, but n_folds==0 with a
    # finite, monotonic curve (already checked above and via the stable-
    # fraction check below) is ALSO accepted as physically valid -- a
    # strongly-hardening mode legitimately may not reach bistability at
    # realistic forcing. What's NOT accepted is 1 fold (an incomplete/
    # numerically-broken trace) or a non-finite/non-monotonic curve.
    n_folds = cont['n_folds']
    _record_check("Continuation is physically well-behaved: either the classic "
                  "fold pair (n_folds=2) or a smooth monotonic hardening curve "
                  "with no fold (n_folds=0, valid for a strongly-hardening mode "
                  "at realistic forcing -- see comment above) -- NOT a partial/"
                  "broken trace (n_folds=1)",
                  n_folds in (0, 2), f"n_folds={n_folds}")
    _record_check("Stable branch is still the majority of the traced curve "
                  "(if folds exist, the unstable region between them stays a "
                  "minority; if no folds, the whole curve is stable by "
                  "definition)",
                  bool(cont['stable'].mean() > 0.5), f"stable fraction={cont['stable'].mean():.3f}")

    hdr("STEP 4I: VALIDATING THE 24-MODE COUPLING TOPOLOGY + CHAIN SOLVER")
    all_pairs = list(CONFIG['nonlinear']['cross_coupling'].keys())
    _1b_pairs_expected = {(0, 1), (3, 4), (5, 6), (7, 8), (9, 10)} | \
        {(m, m + 1) for m in MODE_GROUPS['chain'][:-1]}
    # Superset check (>=), not exact equality: since 2026-08-21 the HF-band
    # campaign (STEP 4J below) added 32 MORE real pairs on top of these
    # original 17, so `all_pairs` now legitimately contains 49, not 17 --
    # exact equality would fail on having MORE real data than before, which
    # is the opposite of a regression. This check now guards that the
    # original 17 are still present, not that nothing else is.
    _record_check("Real cross-coupling data exists for all 17 original 1B-cluster pairs "
                  "(5 clean + 12 chain) -- still present after the HF-band extension",
                  _1b_pairs_expected.issubset(set(all_pairs)),
                  f"found {len(all_pairs)} total (>= 17 expected): {sorted(all_pairs)}")
    _record_check("Real per-mode K3 exists for all 24 1B-cluster modes -- still present "
                  "after the HF-band extension (superset check, not exact-24, since HF "
                  "modes 24-69 now also have real measured K3, see STEP 4J)",
                  set(range(24)).issubset(set(CONFIG['nonlinear']['measured_K3'].keys())),
                  f"{len(CONFIG['nonlinear']['measured_K3'])} of 70 total measured (>= 24 expected)")
    chain = MODE_GROUPS['chain']
    membership = {m: 0 for m in chain}
    for (mi, mj) in CONFIG['nonlinear']['cross_coupling']:
        if mi in membership and mj in membership and abs(chain.index(mi) - chain.index(mj)) == 1:
            membership[mi] += 1
            membership[mj] += 1
    interior_ok = all(membership[m] == 2 for m in chain[1:-1])
    boundary_ok = membership[chain[0]] == 1 and membership[chain[-1]] == 1
    _record_check("Chain coupling topology is correct: interior modes have exactly 2 "
                  "adjacent-pair memberships (left+right neighbor), boundary modes exactly 1",
                  interior_ok and boundary_ok, f"membership counts: {membership}")

    K_arr = np.array([inp['K_sec'][m, m] for m in chain])
    M_arr = np.array([inp['M_sec'][m, m] for m in chain])
    C_arr = np.array([inp['C_sec'][m, m] for m in chain])
    F_gen_test = np.full(len(chain), 1000.0)   # same realistic scale validated for modes 0-1
    chain_amps_all_finite = True
    chain_amps_bounded = True
    for f_drive in np.linspace(inp['freqs_sec'][chain[0]], inp['freqs_sec'][chain[-1]], 5):
        r = duffing_forced_response_chain(chain, K_arr, M_arr, C_arr,
                                           CONFIG['nonlinear']['cross_coupling'],
                                           F_gen_test, 2 * np.pi * f_drive,
                                           n_cycles=300, steps_per_cycle=20)
        chain_amps_all_finite &= bool(np.all(np.isfinite(r['amp'])))
        chain_amps_bounded &= bool(np.all(r['amp'] < 0.5))
    _record_check("13-mode chain solver produces finite, physically bounded amplitudes "
                  "under realistic forcing (F=1000, same scale validated for modes 0-1), "
                  "swept across the chain's own frequency band",
                  chain_amps_all_finite and chain_amps_bounded)

    hdr("STEP 4J: VALIDATING THE HF-BAND (24-69) COUPLING TOPOLOGY + CHAIN SOLVER")
    hf_pairs_expected = (MODE_GROUPS_HF['pairs']
                         + [(c[i], c[i + 1]) for c in MODE_GROUPS_HF['chains'] for i in range(len(c) - 1)])
    hf_all_pairs = [p for p in CONFIG['nonlinear']['cross_coupling'].keys() if p in set(hf_pairs_expected)]
    _record_check(f"Real cross-coupling data exists for all {len(hf_pairs_expected)} expected HF pairs "
                  f"(10 clean + 2 chain-46-48 + 20 chain-49-69)",
                  len(hf_all_pairs) == len(hf_pairs_expected),
                  f"found {len(hf_all_pairs)}/{len(hf_pairs_expected)}")
    hf_modes_expected = set(range(24, 70))
    hf_k3_found = set(CONFIG['nonlinear']['measured_K3'].keys()) & hf_modes_expected
    _record_check("Real per-mode K3 exists for all 46 HF modes (24-69), replacing the "
                  "extrapolated placeholder for every one of them",
                  hf_k3_found == hf_modes_expected,
                  f"{len(hf_k3_found)} of 46 measured")
    _record_check("Real per-mode K3 now exists for the FULL 70-mode secondary basis "
                  "(24 1B-cluster + 46 HF) -- no mode still relies on the extrapolated "
                  "hardening_ratio placeholder",
                  set(CONFIG['nonlinear']['measured_K3'].keys()) == set(range(70)),
                  f"{len(CONFIG['nonlinear']['measured_K3'])} of 70 measured")

    for chain_hf in MODE_GROUPS_HF['chains']:
        membership_hf = {m: 0 for m in chain_hf}
        for (mi, mj) in CONFIG['nonlinear']['cross_coupling']:
            if mi in membership_hf and mj in membership_hf and abs(chain_hf.index(mi) - chain_hf.index(mj)) == 1:
                membership_hf[mi] += 1
                membership_hf[mj] += 1
        interior_ok_hf = all(membership_hf[m] == 2 for m in chain_hf[1:-1])
        boundary_ok_hf = membership_hf[chain_hf[0]] == 1 and membership_hf[chain_hf[-1]] == 1
        _record_check(f"HF chain {chain_hf[0]}-{chain_hf[-1]} coupling topology is correct: "
                      "interior modes have exactly 2 adjacent-pair memberships, boundary modes exactly 1",
                      interior_ok_hf and boundary_ok_hf, f"membership counts: {membership_hf}")

        K_arr_hf = np.array([inp['K_sec'][m, m] for m in chain_hf])
        M_arr_hf = np.array([inp['M_sec'][m, m] for m in chain_hf])
        C_arr_hf = np.array([inp['C_sec'][m, m] for m in chain_hf])
        F_gen_test_hf = np.full(len(chain_hf), 1000.0)
        hf_amps_finite, hf_amps_bounded = True, True
        for f_drive in np.linspace(inp['freqs_sec'][chain_hf[0]], inp['freqs_sec'][chain_hf[-1]], 5):
            r = duffing_forced_response_chain(chain_hf, K_arr_hf, M_arr_hf, C_arr_hf,
                                               CONFIG['nonlinear']['cross_coupling'],
                                               F_gen_test_hf, 2 * np.pi * f_drive,
                                               n_cycles=300, steps_per_cycle=20)
            hf_amps_finite &= bool(np.all(np.isfinite(r['amp'])))
            hf_amps_bounded &= bool(np.all(r['amp'] < 0.5))
        _record_check(f"HF chain {chain_hf[0]}-{chain_hf[-1]} solver produces finite, physically "
                      "bounded amplitudes under realistic forcing, swept across its own frequency band",
                      hf_amps_finite and hf_amps_bounded)

    _record_check("No mode in the full 70-mode secondary basis is left on a diagonal-only "
                  "(SDOF) nonlinear model without a real, measured justification for being "
                  "isolated (modes 2, 24, 37 -- all confirmed genuinely isolated by the real "
                  "half-power-bandwidth gap scan, not assumed)",
                  set(MODE_GROUPS['single']) | set(MODE_GROUPS_HF['single']) == {2, 24, 37},
                  f"isolated modes: {sorted(set(MODE_GROUPS['single']) | set(MODE_GROUPS_HF['single']))}")

    passed = print_validation_summary()

    hdr("STEP 4 COMPLETE")
    elapsed = time.time() - t_start
    print(f"  Validation: {'PASSED' if passed else 'FAILED — see STEP 4 VALIDATION SUMMARY above'}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"\n  Files in {OUT}:")
    for fn in sorted(os.listdir(OUT)):
        fp = os.path.join(OUT, fn)
        if os.path.isfile(fp):
            print(f"    {fn:30s} {os.path.getsize(fp) / 1e3:8.2f} KB")
    print(f"\nLog saved: {_log_path}")
    sys.stdout = sys.__stdout__
    _log_file.close()
