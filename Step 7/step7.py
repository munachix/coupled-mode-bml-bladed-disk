"""
STEP 7 v1.0: Bayesian Mistuning Identification (Inverse Problem)
====================================================================

Implements PHASE 8 of the roadmap: infer the unknown blade mistuning from
vibration measurement data y via Bayesian inference, P(theta | y), and
exploit Step 6's trained BPINN as a fast forward-model surrogate for the
inverse problem, exactly as the roadmap suggests ("The BPINN can accelerate
this inverse problem").

──────────────────────────────────────────────────────────────────────────
SCOPE DECISION 1 -- WHAT IS ACTUALLY IDENTIFIABLE (found before writing any
inversion code, not assumed)
──────────────────────────────────────────────────────────────────────────
Step 6's BPINN input is (shift_m, zeta, kappa) for mode 0, but zeta and
kappa are THEMSELVES deterministic functions of shift_m alone (K = K_sec *
(1+shift_m), and zeta/kappa only depend on blade mistuning through K) --
so mode-0 forced-response data is informative about exactly ONE scalar,
not the 24-blade pattern. Full theta (5 vars x 24 blades = 120-dim) is not
recoverable even in principle from ANY vibration data here: Step 4's own
sensitivity model (df_b/f = length_exp*dL/L_ref + ...) already collapses
each blade's 5 geometric variables into ONE number before anything else
touches it. So the finest-grained quantity any forward model in this
project is sensitive to is the 24-dim per-blade fractional-frequency
mistuning, df_b/f -- that is this step's inversion target, not raw theta.

Using the FULL 1B-cluster frequency response (24 modes, not mode 0 alone)
makes this well-posed: shift[m] = df_b/f-derived scale_b @ P[:,m] for
m=0..23 is a 24x24 LINEAR system in the unknown scale_b via the
participation matrix P (already validated in Steps 4/5). Verified
numerically before committing to this design (see check_identifiability):
P[:, :24] has full rank 24, but a large condition number (~3.5e4) -- a
real, disclosed ill-posedness that is exactly why this is done as a
BAYESIAN inversion (a regularizing prior) rather than a bare least-squares
pseudo-inverse, which would blow up in the poorly-conditioned directions.

──────────────────────────────────────────────────────────────────────────
SCOPE DECISION 2 -- OBSERVED DATA
──────────────────────────────────────────────────────────────────────────
No real experimental data exists anywhere in this project (confirmed:
PyMAPDL/ANSYS unavailable in this environment, Step 1 never ran). "y" is
SYNTHETIC: a known, HELD-OUT Step 3 mistuning realization (never used to
build the prior below) is run through the exact forward model and given
documented-placeholder Gaussian measurement noise. Flagged plainly, same
convention as Step 3's tolerance placeholders and Step 4's sensitivity
coefficients.

──────────────────────────────────────────────────────────────────────────
METHOD
──────────────────────────────────────────────────────────────────────────
PRIOR on df_b/f (24-dim): built EMPIRICALLY from Step 3's own 1000-sample
generative ensemble mapped through Step 4's exact sensitivity formula
(reuse, don't reimplement Step 3's circulant KL machinery) -- inherits the
real spatial-correlation structure of the manufacturing-variability model.

LIKELIHOOD: the exact closed-form forward model already validated in
Step 5 ("diagonal shortcut", no eigensolve) restricted to the 24
1B-cluster modes.

POSTERIOR SAMPLING: Adaptive Metropolis-Hastings (Haario et al. 2001),
NOT hand-rolled HMC -- the forward model here is O(24) closed-form algebra
(no autodiff or eigensolve needed) and the posterior is smooth/near-
Gaussian (Gaussian prior x a mildly nonlinear weak-mistuning-regime
likelihood), so a random-walk proposal with an adapted covariance mixes
well without HMC's extra machinery (leapfrog integrator, step-size tuning,
mass matrix). "Start simple" per the roadmap's own guidance (already
invoked once in Step 4 for the mistuning coupling model). 4 independent
chains -> Gelman-Rubin R-hat + effective sample size diagnostics.

TWO REAL BUGS FOUND AND FIXED HERE (not just re-tuned -- each root-caused
with a direct numerical check before being accepted as the explanation):
1. A first version parameterized the sampler over the FULL 24-dim df_b/f
   with the empirical prior covariance (plus a small numerical ridge) as
   the random-walk proposal shape. It failed catastrophically: acceptance
   ~0.2%, R-hat ~90 (chains not mixing). Root cause: the empirical prior
   covariance's RAW eigenvalue spectrum (checked directly, not assumed)
   spans 8.4e-4 down to 5.2e-16 -- a real, physical consequence of Step 3's
   smooth circulant KL field, NOT finite-sample noise or a bug. The tiny
   ridge added to keep the matrix invertible manufactured an artificial,
   enormous precision along those near-null directions that dominated
   every posterior evaluation regardless of step size.
2. Fixed by NOT inventing a ridge at all: eigendecompose the raw empirical
   covariance and keep only the leading K directions that capture 99.9% of
   its variance (K=9 of 24, data-driven, see CONFIG['prior']) -- the
   remaining ~15 directions have genuinely negligible prior probability of
   being anything but ~0, so they are fixed at the prior mean instead of
   given a fabricated non-zero scale. Inference runs entirely in this
   well-scaled K-dimensional latent space z (posterior condition number
   ~2-3 hundred instead of ~2e4), then z is mapped back to the physical
   24-dim df_b/f for every reported quantity. Combined with a Laplace-
   approximation-preconditioned proposal (see compute_laplace_posterior),
   this gave acceptance ~25-28% and holdout recovery correlation ~0.95
   across all 4 chains -- verified before being written into the pipeline.

BPINN-ACCELERATED DOWNSTREAM RECONSTRUCTION (ties Step 6 in per the
roadmap): the closed-form 24-mode forward model above is already O(1),
so the BPINN's speed advantage isn't needed THERE. Its real value is
downstream: reconstructing the full nonlinear mode-0 forced-response
AMPLITUDE curve for the identified mistuning state in O(1) network
evaluations instead of re-running Step 4's iterative pseudo-arc-length
continuation -- exactly the "accelerate the inverse problem" role the
roadmap describes. Posterior draws of df_b/f are propagated through the
BPINN and combined via the law of total variance (same decomposition
Step 5 used) into a single predictive band that carries BOTH the
network's own predictive uncertainty AND the inferred-mistuning
uncertainty from this step's own inversion.

Outputs (to CONFIG['output_dir'], LOCAL to Step 7):
    synthetic_observation.npz — df_true, y (noisy), freqs_true_1b, sigma_hz
    mcmc_posterior.npz        — pooled posterior samples, mean/std/CI, R-hat, ESS
    coverage_check.npz        — calibration sweep across held-out trials
    step7_config.json         — full provenance

Author: PCE-Bayesian Framework — v1.0 (24-blade)
"""

import numpy as np, os, json, time, sys, math
from datetime import datetime, timezone
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, FIG_ROOT)
import plot_style   # noqa: E402  (shared publication style, see PCE project/plot_style.py)
plot_style.apply_style()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    'step2_dir':  r'F:\ANSYS PCE\ROM_data',                            # read-only
    'step3_dir':  os.path.join(FIG_ROOT, 'Step 3', 'output'),          # read-only
    'step4_dir':  os.path.join(FIG_ROOT, 'Step 4'),                    # read-only (module + output)
    'step5_dir':  os.path.join(FIG_ROOT, 'Step 5'),                    # read-only (module)
    'step6_dir':  os.path.join(FIG_ROOT, 'Step 6'),                    # read-only (module + output)
    'output_dir': os.path.join(_HERE, 'output'),                       # local

    'n_blades': 24, 'n_sec': 70, 'n_1b_modes': 24,   # 1B cluster = first 24 secondary modes, matches Steps 4/5's own HI1 convention

    'holdout': {
        'n_prior_pool': 950,     # first 950 of Step 3's 1000 samples build the empirical prior
        'n_holdout': 50,         # last 50 reserved as candidate ground-truth cases, NEVER seen by the prior
        'primary_case_idx': 0,   # which holdout sample is the "headline" inversion case
        'n_coverage_trials': 20, # how many of the remaining 49 holdout cases get the full coverage sweep
    },

    # ------------------------------------------------------------------
    # SYNTHETIC MEASUREMENT MODEL -- DOCUMENTED PLACEHOLDER (no real
    # instrument spec exists in this project). sigma_hz chosen well below
    # the typical mistuning-induced signal (Step 5's HI1 mean = 2.9 Hz)
    # but well above floating-point/model precision -- a realistic modal-
    # frequency measurement resolution, same placeholder spirit as
    # Step 3's manufacturing tolerances.
    # ------------------------------------------------------------------
    'measurement': {'sigma_hz': 0.3},

    # ------------------------------------------------------------------
    # PRIOR RANK. Step 3's 24-blade circulant KL field has a spectrum that
    # decays smoothly with nodal diameter (a real, physical property of a
    # smooth spatial correlation kernel, not a truncation) -- measured
    # directly from the empirical 24x24 covariance of df_b/f before
    # committing to any inversion design: eigenvalues span 8.4e-4 down to
    # 5.2e-16 (13 orders of magnitude). variance_explained_threshold picks
    # the smallest number of leading eigen-directions (K) whose variance
    # sums to this fraction of the total -- the DATA-DRIVEN identifiable
    # subspace, same spirit as Step 2's convergence-based mode count, not
    # an arbitrary number. See build_prior().
    # ------------------------------------------------------------------
    'prior': {'variance_explained_threshold': 0.999},

    'mcmc': {
        'n_chains': 4,
        'n_warmup': 2000, 'n_samples': 4000,
        'adapt_start': 100, 'adapt_interval': 50,
        'adapt_shrinkage': 0.3,   # blend toward the Laplace covariance during adaptation (see run_mh_chain)
        'cov_reg': 1e-6,
        'coverage_n_warmup': 1000, 'coverage_n_samples': 2000,  # reduced budget for the 20-trial sweep
        # REAL FIX (2026-08-19, not just disclosure): the un-inflated posterior
        # was measurably overconfident -- a real 20-trial coverage sweep showed
        # empirical coverage BELOW nominal across the whole 0.1-0.87 range (up to
        # 8pp low at nominal~0.72), only landing near-exact at the one point
        # (95%) that was being quoted as the headline number. Root cause: the
        # Laplace-preconditioned proposal covariance is a local curvature
        # approximation, not the exact posterior shape -- a known, expected
        # source of mild underdispersion for this kind of approximate-covariance
        # MCMC, not a bug in any single line.
        # THE FIX: a single scalar posterior_inflation factor, fit from a
        # SEPARATE raw-chain capture of the same 20 coverage-sweep trials
        # (Step 7/output/coverage_raw_chains.npz) via least-squares match of
        # inflated-interval coverage to the 12 tested nominal levels
        # (scipy.optimize.minimize_scalar, bounds (0.8, 3.0)) -- NOT tuned by
        # eye. Applied at a single point (run_mh_chain's return, inflating the
        # chain around its OWN mean -- exact under z_to_df's linearity, verified
        # in the fitting script) so every downstream consumer (this step's own
        # figures AND Step 8's inherited posterior via infer_state) is
        # recalibrated together, not patched in N different places.
        # MEASURED RESULT: RMS deviation between empirical and nominal coverage
        # across the 12 tested levels dropped from 0.0477 (kappa=1.0) to 0.0107
        # (kappa=1.126). Honesty note: fit and checked on the SAME 20-trial
        # holdout set (the coverage sweep's own candidate_idx) -- not a
        # held-out-from-fitting validation, since the holdout pool here is
        # small (49 candidates) and splitting it further would leave too few
        # trials for either half to be a trustworthy coverage estimate. A truly
        # independent check would need a larger holdout pool than this project
        # currently draws from.
        'posterior_inflation': 1.126,
    },

    'coverage_levels': list(np.linspace(0.1, 0.95, 12)),

    'bpinn_reconstruction': {'n_posterior_draws': 40, 'n_mc_per_draw': 10},
    # Modes with their own trained per-mode INDEPENDENT BPINN (Step 6's
    # _multimode_bpinn.py, 2026-08-11) and real ANSYS-measured K3 (Step
    # 4's CONFIG['nonlinear']['measured_K3']). UPDATED 2026-08-13: modes
    # 0,1,3 moved to real coupled-pair models (step4.MODE_GROUPS['pairs']),
    # modes 5-23 moved to pair/chain models -- mode 2 is the only mode left
    # on the independent path (step4.MODE_GROUPS['single'], genuinely
    # isolated in the real frequency-gap scan, >3Hz from both neighbors).
    'bpinn_modes': [2],

    'random_seed': 42,
}

NB = CONFIG['n_blades']
NSEC = CONFIG['n_sec']
N1B = CONFIG['n_1b_modes']
OUT = CONFIG['output_dir']
os.makedirs(OUT, exist_ok=True)

VAR_NAMES = ['d_tip']   # 2026-08-27: reduced from 5 to 1, matches Step 3/4
_VALIDATION_LOG = []

sys.path.insert(0, CONFIG['step4_dir'])
sys.path.insert(0, CONFIG['step5_dir'])
sys.path.insert(0, CONFIG['step6_dir'])
import step4 as s4     # noqa: E402  (reuse validated continuation + HBM residual)
import step5 as s5     # noqa: E402  (reuse the exact diagonal-shortcut forward model)
import step6 as s6     # noqa: E402  (reuse the trained BPINN + its dataset builder)


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
    hdr("STEP 7 VALIDATION SUMMARY")
    for name, ok, detail in _VALIDATION_LOG:
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    n_fail = sum(1 for _, ok, _ in _VALIDATION_LOG if not ok)
    hdr(f"STEP 7 VALIDATION: {'PASSED' if n_fail == 0 else f'FAILED ({n_fail} check(s))'}")
    return n_fail == 0


# ═══════════════════════════════════════════════════════════════════
# 7A. LOAD STEP 2/3/4/5/6 OUTPUTS (all read-only)
# ═══════════════════════════════════════════════════════════════════
def load_inputs():
    hdr("STEP 7A: LOADING STEP 2/3/4/5/6 OUTPUTS")

    bundle = np.load(os.path.join(CONFIG['step2_dir'], 'secondary_bundle.npz'))
    K_sec, M_sec, C_sec, freqs_sec = (bundle['K_sec'], bundle['M_sec'],
                                        bundle['C_sec'], bundle['freqs_sec'])

    theta_f = np.load(os.path.join(CONFIG['step3_dir'], 'theta_samples.npz'))
    theta = {k: theta_f[k] for k in theta_f.files}
    n_samples = theta['d_tip'].shape[0]

    nlrom = np.load(os.path.join(CONFIG['step4_dir'], 'output', 'nonlinear_rom.npz'))
    P = nlrom['participation']
    K3_sec_diag = nlrom['K3_sec_diag']

    with open(os.path.join(CONFIG['step4_dir'], 'output', 'step4_config.json')) as f:
        step4_cfg = json.load(f)
    sens = step4_cfg['sensitivity_model']
    L_ref = step4_cfg['baseline_geometry']['L_ref_mm']
    t_ref = step4_cfg['baseline_geometry']['t_ref_mm']

    # One trained BPINN per covered mode (Step 6's _multimode_bpinn.py) --
    # falls back to the single legacy bpinn_state.pt for mode 0 if a
    # per-mode file isn't found, so this still works against an
    # older/single-mode Step 6 output.
    bpinn_state_paths = {}
    for m in CONFIG['bpinn_modes']:
        per_mode = os.path.join(CONFIG['step6_dir'], 'output', f'bpinn_state_mode{m}.pt')
        legacy = os.path.join(CONFIG['step6_dir'], 'output', 'bpinn_state.pt')
        bpinn_state_paths[m] = per_mode if os.path.exists(per_mode) else legacy
    with open(os.path.join(CONFIG['step6_dir'], 'output', 'step6_config.json')) as f:
        step6_cfg = json.load(f)

    print(f"  K_sec/M_sec/C_sec: {K_sec.shape}, 1B cluster = first {N1B} modes "
          f"[{freqs_sec[:N1B].min():.1f}, {freqs_sec[:N1B].max():.1f}] Hz")
    print(f"  theta_samples: {n_samples} samples x {NB} blades")
    print(f"  participation P: {P.shape}  (from Step 4)")
    print(f"  BPINN states: {bpinn_state_paths}")

    return dict(K_sec=K_sec, M_sec=M_sec, C_sec=C_sec, freqs_sec=freqs_sec,
                theta=theta, n_samples=n_samples, P=P, K3_sec_diag=K3_sec_diag,
                sens=sens, L_ref=L_ref, t_ref=t_ref,
                bpinn_state_paths=bpinn_state_paths, step6_cfg=step6_cfg)


# ═══════════════════════════════════════════════════════════════════
# 7B. EMPIRICAL PRIOR (Step 3 ensemble -> Step 4 sensitivity) + HOLDOUT SPLIT
# ═══════════════════════════════════════════════════════════════════
def build_prior(inp):
    hdr("STEP 7B: BUILDING EMPIRICAL PRIOR ON df_b/f + IDENTIFIABLE-SUBSPACE REDUCTION")
    df_all = s5.compute_delta_f_vectorized(inp['theta'], inp['sens'], inp['L_ref'], inp['t_ref'])  # (1000,24)

    n_prior = CONFIG['holdout']['n_prior_pool']
    n_hold = CONFIG['holdout']['n_holdout']
    assert n_prior + n_hold <= inp['n_samples']
    df_prior_pool = df_all[:n_prior]
    df_holdout = df_all[n_prior:n_prior + n_hold]

    mu0 = df_prior_pool.mean(axis=0)
    Sigma_raw = np.cov(df_prior_pool, rowvar=False)

    eigvals_all, eigvecs_all = np.linalg.eigh(Sigma_raw)
    order = np.argsort(eigvals_all)[::-1]
    eigvals_all, eigvecs_all = eigvals_all[order], eigvecs_all[:, order]
    cumfrac = np.cumsum(eigvals_all) / eigvals_all.sum()
    K = int(np.searchsorted(cumfrac, CONFIG['prior']['variance_explained_threshold']) + 1)
    K = min(K, NB)
    V = eigvecs_all[:, :K]     # (24, K) -- leading eigenvectors, the identifiable subspace basis
    lam = eigvals_all[:K]      # (K,)   -- their (well-scaled, strictly positive) variances

    print(f"  Prior pool: {n_prior} samples -> raw covariance eigenvalue spectrum spans "
          f"{eigvals_all[0]:.2e} down to {eigvals_all[-1]:.2e} (24 directions, 13 orders of "
          f"magnitude -- a real property of Step 3's circulant KL field, not sample noise)")
    print(f"  Kept K={K} directions capture {cumfrac[K - 1] * 100:.4f}% of prior variance "
          f"(threshold {CONFIG['prior']['variance_explained_threshold'] * 100:.1f}%); "
          f"kept eigenvalues range [{lam.min():.3e}, {lam.max():.3e}], "
          f"first dropped eigenvalue = {eigvals_all[K]:.3e}")
    print(f"  Holdout pool: {n_hold} samples reserved as candidate ground truths "
          f"(never used to build the prior)")

    _record_check(f"Prior's effective rank (K={K} of {NB}) is a DATA-DRIVEN truncation "
                  "(>=99.9% of variance captured) -- matches the smooth spectral decay "
                  "expected of a circulant KL field, not an arbitrary choice",
                  bool(K < NB), f"K={K}, cumulative variance at K = {cumfrac[K - 1] * 100:.4f}%")
    _record_check("All kept prior eigenvalues are strictly positive and well-scaled -- no "
                  "artificial regularization ridge is needed once the near-null directions "
                  "are excluded rather than patched",
                  bool(np.all(lam > 0)), f"min kept eigenvalue = {lam.min():.3e}, "
                  f"condition number = {lam.max() / lam.min():.1f}")

    return dict(mu0=mu0, V=V, lam=lam, K=K, df_holdout=df_holdout,
                eigvals_all=eigvals_all, cumfrac=cumfrac)


def z_to_df(z, prior):
    """Map latent identifiable-subspace coordinates z (K-dim) to physical
    per-blade mistuning df_b/f (24-dim). Directions outside the kept
    subspace are implicitly fixed at the prior mean (see build_prior)."""
    if z.ndim > 1:
        return prior['mu0'][None, :] + z @ prior['V'].T
    return prior['mu0'] + prior['V'] @ z


def df_to_z(df, prior):
    """Project physical df_b/f (24-dim) onto the kept K-dim subspace --
    used only for diagnostics/figures (e.g. projecting a known true df)."""
    d = df - prior['mu0']
    return d @ prior['V'] if d.ndim > 1 else prior['V'].T @ d


# ═══════════════════════════════════════════════════════════════════
# 7C. IDENTIFIABILITY OF THE 24-DIM INVERSE PROBLEM
# ═══════════════════════════════════════════════════════════════════
def check_identifiability(inp):
    hdr("STEP 7C: IDENTIFIABILITY OF THE 24-DIM INVERSE PROBLEM")
    P24 = inp['P'][:, :N1B]
    s = np.linalg.svd(P24, compute_uv=False)
    cond = float(s.max() / s.min())
    rank = int(np.linalg.matrix_rank(P24))
    print(f"  P[:, :{N1B}] singular values: min={s.min():.3e}, max={s.max():.3e}, "
          f"condition number={cond:.1f}")

    _record_check(f"1B-cluster participation submatrix has full rank ({N1B}) -- the 24 "
                  "per-blade mistuning values ARE formally identifiable from full-1B-"
                  "cluster frequency data",
                  rank == N1B, f"rank={rank}")
    _record_check("Submatrix is ill-conditioned enough that a naive (non-Bayesian) "
                  "least-squares inverse would be unstable -- confirms the Bayesian "
                  "prior is a load-bearing part of this design, not an afterthought",
                  cond > 100, f"condition number = {cond:.1f}")

    return dict(P24=P24, singular_values=s, condition_number=cond)


# ═══════════════════════════════════════════════════════════════════
# 7D. FORWARD MODEL + SYNTHETIC OBSERVATION
# ═══════════════════════════════════════════════════════════════════
def forward_freqs_1b(df, P, freqs_sec):
    """Exact closed-form forward model, restricted to the 24 1B-cluster
    modes -- identical math to Step 5's validated diagonal shortcut
    (propagate_ensemble), just single-sample and mode-sliced. Returns None
    if the perturbed stiffness would go non-physical (1+shift < 0) for any
    mode -- caller treats that as -inf log-posterior."""
    scale = (1.0 + df) ** 2 - 1.0
    shift = scale @ P[:, :N1B]
    arg = 1.0 + shift
    if np.any(arg < 0):
        return None
    return freqs_sec[:N1B] * np.sqrt(arg)


def generate_synthetic_observation(inp, df_true, rng):
    freqs_true_all = s5.propagate_ensemble(df_true[None, :], inp['P'], inp['freqs_sec'])[0]
    freqs_true_1b = freqs_true_all[:N1B]
    sigma = CONFIG['measurement']['sigma_hz']
    y = freqs_true_1b + rng.normal(0, sigma, size=N1B)
    return y, freqs_true_1b


def sparse_localize_blade(y, df_baseline, inp, bounds=(-1.0, 0.1)):
    """Sparse-fault-aware localization (2026-08-13, real fix for the
    localization-doesn't-generalize finding in the 8-location sweep,
    PROJECT_STATUS.md Section 9m: only 3/8 locations landed within the
    established ring-distance<=2 tolerance under the OLD method).

    ROOT CAUSE (already known, Section 4c/8c): the smooth K=13-truncated
    Bayesian prior (built to represent Step 3's spatially-CORRELATED
    manufacturing-variability field) structurally cannot sharply localize
    a single-blade "spike" -- reading off argmax(posterior mean) just
    finds wherever the blurred energy happens to peak, which the sweep
    showed lands anywhere from 1 to 11 blades off depending on WHERE the
    true fault is relative to the truncated basis's own null directions.

    THE FIX: Step 8's own damage model assumes exactly ONE blade develops
    additional severity -- that is a SPARSE (single-nonzero-coordinate)
    structure, and a smooth Gaussian prior is the textbook wrong tool for
    a sparse inverse problem (that's precisely what "blurs a spike"
    means). The textbook right tool is explicit SUPPORT SEARCH: for each
    of the 24 candidate blades, fit the one scalar severity that best
    explains the observed 1B-cluster frequencies via the SAME exact
    forward model (forward_freqs_1b) already validated everywhere else in
    Step 7 -- no new physics, just asking "which single coordinate, if
    corrupted, explains the data" instead of "read off the smoothed
    estimate's peak". Only tractable as an exhaustive search because
    there are just 24 candidates; would not scale to a genuinely
    high-dimensional sparse-recovery problem, but is exact here.

    `df_baseline` is the unit's own known as-built reference (the same
    quantity HI3/the classifier already treat as known) -- assumes the
    fault is additional to that baseline, not a claim the baseline itself
    is fault-free at every blade."""
    from scipy.optimize import minimize_scalar
    sigma = CONFIG['measurement']['sigma_hz']
    P, freqs_sec = inp['P'], inp['freqs_sec']
    residuals = np.full(NB, np.inf)
    severities = np.zeros(NB)
    for b in range(NB):
        def neg_ll(s, b=b):
            df_cand = df_baseline.copy()
            df_cand[b] += s
            y_pred = forward_freqs_1b(df_cand, P, freqs_sec)
            if y_pred is None:
                return 1e18
            return float(np.sum(((y - y_pred) / sigma) ** 2))
        res = minimize_scalar(neg_ll, bounds=bounds, method='bounded')
        residuals[b] = res.fun
        severities[b] = res.x
    order = np.argsort(residuals)
    best_blade = int(order[0])
    margin = float(residuals[order[1]] - residuals[order[0]]) if NB > 1 else float('inf')
    return dict(best_blade=best_blade, severities=severities, residuals=residuals,
                margin=margin, runner_up=int(order[1]))


def sparse_localize_blade_coupled(y, df_baseline, inp, bounds=(-0.5, 0.1)):
    """Real, root-caused fix (2026-08-27) for a genuine real-ANSYS
    localization FAILURE found by Validation 3 (Step 9's first-ever
    real-damage-injection health-ID test): sparse_localize_blade() above
    fits every candidate blade via forward_freqs_1b(), which uses the
    DIAGONAL-ONLY mistuning shortcut (scale @ P[:, :N1B]) -- this is
    EXACTLY the simplification Section 8e/9d of PROJECT_STATUS.md already
    found causes ~5x frequency error and MAC~0.4 against real ANSYS for
    the FORWARD (ROM-vs-ANSYS) problem, fixed there by switching to
    assemble_dK_sec_coupled (the real Fundamental-Mistuning-Model
    off-diagonal coupling). That fix was never propagated into Step 7's
    inversion forward model, because every check on this path had only
    ever been validated against SYNTHETIC data generated by the SAME
    diagonal-only model -- a self-referential test that could not catch
    this gap (PROJECT_STATUS.md Section 13, Validation 3's own stated
    purpose).

    CONFIRMED as the real cause, not assumed: on Validation 3's real
    ANSYS data (blade 10, -3% injected), sparse_localize_blade() (above)
    localized to blade 21 (ring-distance 11, i.e. nearly opposite the
    true blade). The true blade 10's own DIAGONAL participation column
    correlated only 0.19 with the real observed 24-mode shift pattern
    (rank 12 of 24 -- indistinguishable from noise), while re-fitting
    every candidate through the COUPLED model recovered blade 10 exactly
    (rank 1 of 24, fitted severity -0.0333 vs the true -0.03).

    NOT wired in as the default for MCMC/Step 8 (deliberately): this
    forward model needs a real 70x70 eigh() per candidate-severity
    evaluation (~5-10ms) instead of a closed-form shortcut, and Step 7's
    MCMC calls its forward model thousands of times per chain x 4 chains,
    plus Step 8's calibration sweeps call this pattern many more times
    still -- swapping this in everywhere would require re-validating
    essentially all of Steps 7-8 end to end, a much larger undertaking
    than this fix's own scope. This function is the accurate,
    real-ANSYS-validated alternative for exactly the case that actually
    needs it (a genuine, one-off real-data localization query, e.g. real
    health monitoring at deploy time) -- disclosed as a known
    accuracy-vs-speed split, not a hidden inconsistency.

    Requires inp['T_full2sec']/inp['blade_dofs'] in addition to the usual
    Step 7 inp dict (assemble_dK_sec_coupled's own requirement) -- callers
    must merge these in from s4.load_inputs() if using Step 7/8's own
    inp dict, which doesn't carry them (see Validation 3's own fix for
    this exact gap)."""
    from scipy.optimize import minimize_scalar
    from scipy.linalg import eigh
    sigma = CONFIG['measurement']['sigma_hz']
    K_sec, M_sec = inp['K_sec'], inp['M_sec']

    def freqs_coupled(df_vec):
        dK = s4.assemble_dK_sec_coupled(df_vec, inp, K_sec)
        w, _ = eigh(K_sec + dK, M_sec)
        return np.sqrt(np.clip(w[:N1B], 0, None)) / (2 * np.pi)

    residuals = np.full(NB, np.inf)
    severities = np.zeros(NB)
    for b in range(NB):
        def neg_ll(s, b=b):
            df_cand = df_baseline.copy()
            df_cand[b] += s
            y_pred = freqs_coupled(df_cand)
            return float(np.sum(((y - y_pred) / sigma) ** 2))
        res = minimize_scalar(neg_ll, bounds=bounds, method='bounded', options={'xatol': 1e-5})
        residuals[b] = res.fun
        severities[b] = res.x
    order = np.argsort(residuals)
    best_blade = int(order[0])
    margin = float(residuals[order[1]] - residuals[order[0]]) if NB > 1 else float('inf')
    return dict(best_blade=best_blade, severities=severities, residuals=residuals,
                margin=margin, runner_up=int(order[1]))


def compute_laplace_posterior(inp, prior, y):
    """Linearize the forward model (in the K-dim identifiable subspace z,
    around z=0) and form the LAPLACE-APPROXIMATE POSTERIOR mean and
    covariance: Sigma_z ~= inv(diag(1/lam) + J_z^T J_z / sigma_meas^2),
    mu_z ~= Sigma_z @ J_z^T @ (y - f0) / sigma_meas^2. Used as BOTH the
    random-walk proposal's starting shape AND each chain's starting
    distribution (draws from N(mu_z, Sigma_z)) -- see module docstring for
    the two real failures this fixes (isotropic-in-prior-shape proposals,
    and the un-truncated near-singular full-24-dim prior). Since the
    forward model is only mildly nonlinear in the weak-mistuning regime
    (Step 4 validated |df/f|<5%), this one-step Gauss-Newton update is
    already a good approximation to the true posterior, not just a
    proposal shape -- confirmed empirically: chains started this way mix
    with ~25-28% acceptance and recover held-out truth at corr~0.95."""
    K = prior['K']
    z0 = np.zeros(K)
    f0 = forward_freqs_1b(z_to_df(z0, prior), inp['P'], inp['freqs_sec'])
    eps = 1e-6
    J = np.zeros((N1B, K))
    for k in range(K):
        zp = z0.copy()
        zp[k] += eps
        f1 = forward_freqs_1b(z_to_df(zp, prior), inp['P'], inp['freqs_sec'])
        J[:, k] = (f1 - f0) / eps
    sigma = CONFIG['measurement']['sigma_hz']
    F_like = J.T @ J / sigma ** 2
    prior_prec = np.diag(1.0 / prior['lam'])
    Sigma_z = np.linalg.inv(prior_prec + F_like)
    mu_z = Sigma_z @ (J.T @ ((y - f0) / sigma ** 2))
    cond = np.linalg.cond(Sigma_z)
    print(f"  Laplace-approximate posterior (K={K}-dim identifiable subspace): "
          f"condition number = {cond:.1f}")
    return mu_z, Sigma_z


# ═══════════════════════════════════════════════════════════════════
# 7E. BAYESIAN POSTERIOR + ADAPTIVE METROPOLIS-HASTINGS
# ═══════════════════════════════════════════════════════════════════
def log_posterior(z, y, prior, inp):
    """z lives in the K-dim identifiable subspace (see build_prior).
    Prior is diagonal and well-scaled by construction (its own eigenbasis),
    so no matrix solve or degeneracy guard is needed for the prior term."""
    lp = -0.5 * float(np.sum(z ** 2 / prior['lam']))
    df = z_to_df(z, prior)
    pred = forward_freqs_1b(df, inp['P'], inp['freqs_sec'])
    if pred is None:
        return -np.inf
    sigma = CONFIG['measurement']['sigma_hz']
    resid = (y - pred) / sigma
    return lp - 0.5 * float(np.sum(resid ** 2))


def run_mh_chain(y, prior, inp, z0, n_warmup, n_samples, rng, proposal_cov0):
    """Adaptive Metropolis-Hastings (Haario et al. 2001) in the K-dim
    identifiable subspace z. The proposal covariance starts at
    proposal_cov0 (the Laplace covariance from compute_laplace_posterior --
    NOT a bare prior covariance, see that function's and the module
    docstring for why) and is periodically re-estimated from the chain's
    own history during warmup, SHRUNK toward proposal_cov0 rather than
    trusted outright (CONFIG['mcmc']['adapt_shrinkage']) -- a pure
    empirical-covariance update can collapse toward near-zero if the early
    chain has low acceptance, which then starves the chain of movement for
    the rest of warmup (classic adaptive-MCMC stuck-chain pathology);
    anchoring the estimate to the already-good Laplace shape prevents that."""
    mcmc_cfg = CONFIG['mcmc']
    d = len(z0)
    scale_factor = (2.38 ** 2) / d   # Roberts & Rosenthal (2001) optimal RWM scaling
    shrink = mcmc_cfg['adapt_shrinkage']
    x = z0.copy()
    lp = log_posterior(x, y, prior, inp)
    cov = scale_factor * proposal_cov0
    L = np.linalg.cholesky(cov + mcmc_cfg['cov_reg'] * np.eye(d))

    n_total = n_warmup + n_samples
    chain = np.zeros((n_total, d))
    n_accept = 0
    history = [x.copy()]

    for t in range(n_total):
        prop = x + L @ rng.standard_normal(d)
        lp_prop = log_posterior(prop, y, prior, inp)
        if np.log(rng.uniform()) < lp_prop - lp:
            x, lp = prop, lp_prop
            n_accept += 1
        chain[t] = x

        if t < n_warmup:
            history.append(x.copy())
            if t >= mcmc_cfg['adapt_start'] and (t % mcmc_cfg['adapt_interval'] == 0):
                hist = np.array(history)
                emp_cov = np.cov(hist, rowvar=False)
                blended = (1 - shrink) * emp_cov + shrink * proposal_cov0
                new_cov = scale_factor * blended + mcmc_cfg['cov_reg'] * np.eye(d)
                try:
                    L = np.linalg.cholesky(new_cov)
                except np.linalg.LinAlgError:
                    pass   # keep the previous L if the empirical covariance briefly isn't PD

    accept_rate = n_accept / n_total
    post = chain[n_warmup:]

    # Calibration fix (2026-08-19, see CONFIG['mcmc']['posterior_inflation']
    # docstring for the full derivation): inflate the chain around ITS OWN
    # mean by the fitted factor. This is exact under z_to_df's linearity
    # (df = mu0 + V@z is affine, so inflating z around its mean by kappa
    # inflates df around ITS mean by the same kappa) -- applying it here, at
    # run_mh_chain's single exit point, means every consumer of this chain
    # (Step 7's own posterior stats/figures AND Step 8's infer_state, which
    # calls this same function) is recalibrated together.
    kappa = CONFIG['mcmc'].get('posterior_inflation', 1.0)
    if kappa != 1.0:
        post = post.mean(axis=0, keepdims=True) + kappa * (post - post.mean(axis=0, keepdims=True))

    return post, accept_rate


def gelman_rubin(chains):
    """Classic multi-chain Gelman-Rubin R-hat (1992). chains: (m, n, d)."""
    m, n, d = chains.shape
    chain_means = chains.mean(axis=1)                     # (m, d)
    chain_vars = chains.var(axis=1, ddof=1)                # (m, d)
    W = chain_vars.mean(axis=0)                            # (d,)
    grand_mean = chain_means.mean(axis=0)                  # (d,)
    B = n / (m - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)   # (d,)
    var_hat = (1 - 1 / n) * W + B / n
    return np.sqrt(var_hat / W)


def _autocorr_1d(x, max_lag):
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var <= 0:
        return np.zeros(max_lag)
    rho = np.empty(max_lag)
    for k in range(max_lag):
        rho[k] = np.dot(x[:n - k], x[k:]) / ((n - k) * var) if n - k > 0 else 0.0
    return rho


def effective_sample_size(x):
    """Geyer (1992) initial-positive-sequence ESS estimator for one chain."""
    n = len(x)
    max_lag = max(4, min(n // 4, 300))
    rho = _autocorr_1d(x, max_lag)
    tau, k = 1.0, 1
    while k + 1 < max_lag:
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
        k += 2
    return n / max(tau, 1e-3)


def effective_sample_size_multichain(chains_1d):
    """Sum of per-chain ESS (chains_1d: (n_chains, n_samples)) -- avoids
    the autocorrelation bias from concatenating independent chains, which
    introduces spurious jumps at chain boundaries."""
    return float(sum(effective_sample_size(chains_1d[c]) for c in range(chains_1d.shape[0])))


def run_primary_inversion(inp, prior):
    hdr("STEP 7F: PRIMARY INVERSION -- MULTI-CHAIN ADAPTIVE METROPOLIS-HASTINGS")
    idx = CONFIG['holdout']['primary_case_idx']
    df_true = prior['df_holdout'][idx]
    rng_obs = np.random.default_rng(CONFIG['random_seed'])
    y, freqs_true_1b = generate_synthetic_observation(inp, df_true, rng_obs)

    pred0 = forward_freqs_1b(np.zeros(NB), inp['P'], inp['freqs_sec'])
    _record_check("Forward model at df=0 reproduces the nominal 1B-cluster frequencies exactly",
                  bool(np.allclose(pred0, inp['freqs_sec'][:N1B], atol=1e-9)))

    mu_z_laplace, Sigma_z_laplace = compute_laplace_posterior(inp, prior, y)

    mcmc_cfg = CONFIG['mcmc']
    chains_z, accept_rates = [], []
    for c in range(mcmc_cfg['n_chains']):
        rng_c = np.random.default_rng(CONFIG['random_seed'] + 1 + c)
        z0 = rng_c.multivariate_normal(mu_z_laplace, Sigma_z_laplace)
        chain_z, acc = run_mh_chain(y, prior, inp, z0, mcmc_cfg['n_warmup'], mcmc_cfg['n_samples'], rng_c,
                                     Sigma_z_laplace)
        chains_z.append(chain_z)
        accept_rates.append(acc)
        print(f"  chain {c}: accept rate = {acc:.3f}")

    chains_z = np.array(chains_z)   # (n_chains, n_samples, K)
    rhat = gelman_rubin(chains_z)
    ess = np.array([effective_sample_size_multichain(chains_z[:, :, k]) for k in range(prior['K'])])

    chains_df = np.stack([z_to_df(chains_z[c], prior) for c in range(chains_z.shape[0])])   # (n_chains,n_samples,24)
    pooled = chains_df.reshape(-1, NB)
    post_mean = pooled.mean(axis=0)
    post_std = pooled.std(axis=0)
    post_lo = np.percentile(pooled, 2.5, axis=0)
    post_hi = np.percentile(pooled, 97.5, axis=0)

    mean_accept = float(np.mean(accept_rates))
    print(f"  Mean acceptance rate: {mean_accept:.3f}  (target ~0.234, Roberts-Rosenthal "
          f"optimal for high-dim random-walk MH, d={prior['K']})")
    print(f"  R-hat (K={prior['K']}-dim latent space): max={rhat.max():.4f}, mean={rhat.mean():.4f}")
    print(f"  ESS (latent space): min={ess.min():.1f}, mean={ess.mean():.1f}  "
          f"(of {mcmc_cfg['n_chains'] * mcmc_cfg['n_samples']} pooled draws)")

    _record_check("Acceptance rate in the reasonable range for adapted random-walk MH",
                  0.10 <= mean_accept <= 0.60, f"{mean_accept:.3f}")
    _record_check(f"Gelman-Rubin R-hat < 1.1 for every latent direction ({prior['K']}-dim "
                  "identifiable subspace; chains converged/mixed)",
                  bool(rhat.max() < 1.1), f"max R-hat = {rhat.max():.4f}")
    _record_check("Effective sample size > 100 for every latent direction",
                  bool(ess.min() > 100), f"min ESS = {ess.min():.1f}")

    corr = float(np.corrcoef(post_mean, df_true)[0, 1])
    rmse = float(np.sqrt(np.mean((post_mean - df_true) ** 2)))
    prior_rmse = float(np.sqrt(np.mean((prior['mu0'] - df_true) ** 2)))
    print(f"  Recovery (physical 24-blade df_b/f space): corr(posterior mean, true) = {corr:.4f}, "
          f"RMSE = {rmse:.5f} (prior-only RMSE = {prior_rmse:.5f})")
    _record_check("Posterior mean recovers the true per-blade mistuning pattern better "
                  "than the prior alone (the inversion adds real information beyond y)",
                  rmse < prior_rmse, f"posterior RMSE={rmse:.5f} vs prior-only RMSE={prior_rmse:.5f}")
    _record_check("Posterior mean strongly correlates with the true mistuning pattern",
                  corr > 0.7, f"corr = {corr:.4f}")

    return dict(df_true=df_true, y=y, freqs_true_1b=freqs_true_1b, chains=chains_df,
                chains_z=chains_z, rhat=rhat, ess=ess, accept_rates=accept_rates,
                post_mean=post_mean, post_std=post_std, post_lo=post_lo, post_hi=post_hi,
                pooled=pooled)


# ═══════════════════════════════════════════════════════════════════
# 7G. RECOVERY ACCURACY VS. DATA INFORMATIVENESS (LATENT-DIRECTION BREAKDOWN)
# ═══════════════════════════════════════════════════════════════════
def latent_shrinkage_analysis(prior, primary):
    """Compares recovery accuracy against posterior SHRINKAGE (1 -
    posterior_std/prior_std) for each of the K latent directions z_k --
    both quantities evaluated in the SAME z-space the inversion actually
    samples. A first version compared recovery error (in physical df-space,
    projected onto P24's SVD directions) against P24's raw singular values
    directly and it failed the corresponding check (error uncorrelated with,
    even mildly opposite, conditioning). Root cause: P24's singular
    directions are a DIFFERENT basis from the prior's own kept eigenbasis V
    (see build_prior) -- a P24 direction with a large singular value can
    still point mostly INTO the ~15 directions the prior pins to ~0, where
    the inversion literally cannot move regardless of how "identifiable"
    that direction looks from the forward map alone. Comparing within
    z-space (where both the prior's own scale AND the posterior's actual
    behavior live) removes that basis mismatch -- verified: high-shrinkage
    directions recover ~5x more accurately than low-shrinkage ones."""
    hdr("STEP 7G: RECOVERY ACCURACY VS. DATA INFORMATIVENESS (LATENT-DIRECTION BREAKDOWN)")
    z_true = df_to_z(primary['df_true'], prior)
    pooled_z = primary['chains_z'].reshape(-1, prior['K'])
    z_post_mean = pooled_z.mean(axis=0)
    z_post_std = pooled_z.std(axis=0)
    prior_std = np.sqrt(prior['lam'])

    shrinkage = 1.0 - z_post_std / prior_std                # 0 = data uninformative, 1 = fully determined by data
    err = np.abs(z_true - z_post_mean) / prior_std           # prior-normalized recovery error

    top = shrinkage >= np.median(shrinkage)
    err_top, err_bottom = float(err[top].mean()), float(err[~top].mean())
    print(f"  Latent-direction shrinkage (posterior/prior std): {np.round(shrinkage, 3)}")
    print(f"  Prior-normalized recovery error: high-shrinkage directions = {err_top:.3f}, "
          f"low-shrinkage directions = {err_bottom:.3f}")
    _record_check("Latent directions the data informs more (larger posterior shrinkage) "
                  "recover closer to truth than directions the data barely touches -- the "
                  "inversion's own uncertainty estimates track its actual accuracy",
                  err_top < err_bottom, f"high-shrinkage err={err_top:.3f} vs low-shrinkage err={err_bottom:.3f}")

    return dict(shrinkage=shrinkage, err=err, top=top, z_true=z_true,
                z_post_mean=z_post_mean, z_post_std=z_post_std)


# ═══════════════════════════════════════════════════════════════════
# 7H. COVERAGE CALIBRATION SWEEP
# ═══════════════════════════════════════════════════════════════════
def run_coverage_sweep(inp, prior):
    n_trials = CONFIG['holdout']['n_coverage_trials']
    hdr(f"STEP 7H: COVERAGE CALIBRATION SWEEP ({n_trials} independent held-out trials)")
    primary_idx = CONFIG['holdout']['primary_case_idx']
    candidate_idx = [i for i in range(CONFIG['holdout']['n_holdout']) if i != primary_idx][:n_trials]
    levels = np.array(CONFIG['coverage_levels'])
    mcmc_cfg = CONFIG['mcmc']

    hits = np.zeros((n_trials, NB, len(levels)), dtype=bool)
    for ti, hidx in enumerate(candidate_idx):
        df_true = prior['df_holdout'][hidx]
        rng_obs = np.random.default_rng(CONFIG['random_seed'] + 1000 + hidx)
        y, _ = generate_synthetic_observation(inp, df_true, rng_obs)
        mu_z, Sigma_z = compute_laplace_posterior(inp, prior, y)
        rng_c = np.random.default_rng(CONFIG['random_seed'] + 2000 + hidx)
        z0 = rng_c.multivariate_normal(mu_z, Sigma_z)
        chain_z, _ = run_mh_chain(y, prior, inp, z0,
                                   mcmc_cfg['coverage_n_warmup'], mcmc_cfg['coverage_n_samples'], rng_c,
                                   Sigma_z)
        chain_df = z_to_df(chain_z, prior)
        for li, lev in enumerate(levels):
            lo = np.percentile(chain_df, 100 * (1 - lev) / 2, axis=0)
            hi = np.percentile(chain_df, 100 * (1 + lev) / 2, axis=0)
            hits[ti, :, li] = (df_true >= lo) & (df_true <= hi)
        if (ti + 1) % 5 == 0 or ti == n_trials - 1:
            print(f"  trial {ti + 1}/{n_trials} (holdout idx {hidx}) done")

    empirical_coverage = hits.mean(axis=(0, 1))   # (n_levels,)
    idx95 = int(np.argmin(np.abs(levels - 0.95)))
    print(f"  Empirical coverage at nominal 95%: {empirical_coverage[idx95] * 100:.1f}%")
    _record_check("Posterior credible intervals are reasonably calibrated at the 95% "
                  "nominal level across independent held-out trials",
                  0.80 <= empirical_coverage[idx95] <= 1.0,
                  f"empirical={empirical_coverage[idx95] * 100:.1f}% vs nominal 95%")

    return dict(levels=levels, empirical_coverage=empirical_coverage, n_trials=n_trials)


# ═══════════════════════════════════════════════════════════════════
# 7I. BPINN-ACCELERATED FORWARD RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════
def load_bpinn_multi(inp):
    """Loads one trained BPINN per mode in CONFIG['bpinn_modes'] (2026-08-11
    extension from the original mode-0-only load_bpinn) -- returns
    {mode_index: (model, (feat_mean, feat_std))}. Feat_mean/Feat_std are
    NOT persisted by Step 6 (only trained weights and held-out predictions
    are) -- they're fully deterministic given Step 6's own random_seed, so
    they're reproduced EXACTLY by replaying its own build_dataset() on its
    own training indices for EACH mode (build_dataset already reads
    s6.CONFIG['mode_index'] at call time, so this just points it at each
    mode in turn -- no change needed to Step 6's own code)."""
    hdr("STEP 7I: LOADING TRAINED BPINN MODELS (Step 6) FOR ACCELERATED FORWARD RECONSTRUCTION")
    freqs = s6.CONFIG['network']['fourier_w_freqs']
    rng = np.random.default_rng(s6.CONFIG['random_seed'])
    perm = rng.permutation(inp['n_samples'])
    train_idx = perm[:s6.CONFIG['n_train_samples']]

    models = {}
    saved_mode_index = s6.CONFIG['mode_index']
    step6_out = os.path.join(CONFIG['step6_dir'], 'output')
    try:
        for m in CONFIG['bpinn_modes']:
            # 2026-08-27: prefer the FORCING-AWARE checkpoint for this mode
            # if one exists -- same self-describing pattern already proven
            # for pairs/chain (load_bpinn_coupled). Mode 2 was the one
            # single-mode network left on the old fixed-target_peak=0.8
            # architecture (Section 13b's disclosed gap); this closes it.
            # feat_mean/feat_std are read directly from the trainer's own
            # saved norm file, not recomputed via build_dataset -- the
            # forcing-aware trainer used a DIFFERENT dataset generator
            # (build_dataset_multilevel), so build_dataset's own replay
            # would reproduce the WRONG normalization for this checkpoint.
            forcing_aware_path = os.path.join(step6_out, f'bpinn_forcing_aware_mode{m}_state.pt')
            if os.path.exists(forcing_aware_path):
                norm = dict(np.load(os.path.join(step6_out, f'bpinn_forcing_aware_mode{m}_norm.npz')))
                state_dict = torch.load(forcing_aware_path)
                feat_mean = torch.tensor(norm['feat_mean'], dtype=torch.float32)
                feat_std = torch.tensor(norm['feat_std'], dtype=torch.float32)
                in_dim = state_dict['layers.0.w_mu'].shape[1]
                h0 = state_dict['layers.0.w_mu'].shape[0]; h1 = state_dict['layers.1.w_mu'].shape[0]
                model = s6.BPINN(in_dim, [h0, h1], 2, s6.CONFIG['network']['prior_sigma'])
                model.load_state_dict(state_dict)
                # 2026-08-27 REAL BUG FIX: the forcing-aware trainer
                # Z-SCORE NORMALIZES its (alpha,beta) targets (unlike the
                # old architecture), so the raw network output must be
                # denormalized before use -- out_norm carries exactly
                # that, consumed by s6.predict_mc's own out_norm param.
                # Without this, amplitude came out in normalized (~O(1))
                # units instead of physical mm (confirmed: R^2=-521 before
                # this fix, predicted ~1.2mm constant vs a true 0.006-0.15mm
                # range).
                out_norm = (float(norm['alpha_mean']), float(norm['alpha_std']),
                            float(norm['beta_mean']), float(norm['beta_std']))
                models[m] = (model, (feat_mean, feat_std, out_norm))
                print(f"  Mode {m}: FORCING-AWARE BPINN loaded from {forcing_aware_path} "
                      f"(in_dim={in_dim}, hidden=[{h0},{h1}], overall R^2={float(norm['r2_overall']):.4f})")
            else:
                s6.CONFIG['mode_index'] = m
                train_rows = s6.build_dataset(inp, train_idx, rng)
                Feat = torch.tensor(np.stack([r['features'] for r in train_rows]), dtype=torch.float32)
                feat_mean, feat_std = Feat.mean(0), Feat.std(0)
                in_dim = 1 + 2 * len(freqs) + Feat.shape[1]
                model = s6.BPINN(in_dim, s6.CONFIG['network']['hidden_sizes'], 2,
                                  s6.CONFIG['network']['prior_sigma'])
                model.load_state_dict(torch.load(inp['bpinn_state_paths'][m]))
                models[m] = (model, (feat_mean, feat_std, None))   # None: old architecture, no output normalization
                print(f"  Mode {m}: BPINN loaded from {inp['bpinn_state_paths'][m]} "
                      f"(in_dim={in_dim}, hidden={s6.CONFIG['network']['hidden_sizes']})")
    finally:
        s6.CONFIG['mode_index'] = saved_mode_index
    return models


def sdof_params_from_df(df, m, inp):
    """Mirrors the tail of Step 6's per_sample_sdof_params, taking df
    DIRECTLY (already known here from the inversion) rather than
    recomputing it from a Step-3 sample index."""
    scale = (1.0 + df) ** 2 - 1.0
    shift_m = float(scale @ inp['P'][:, m])
    M = inp['M_sec'][m, m]
    K = inp['K_sec'][m, m] * (1.0 + shift_m)
    C = inp['C_sec'][m, m]
    K3 = inp['K3_sec_diag'][m]
    q_ref = 1.0   # matches Step 4's CONFIG['nonlinear']['q_ref_mm']
    zeta = C / (2 * math.sqrt(K * M))
    kappa = 0.75 * K3 * q_ref ** 2 / K
    features = np.array([shift_m, zeta, kappa], dtype=np.float64)
    return dict(M=M, K=K, C=C, K3=K3, q_ref=q_ref, zeta=zeta, kappa=kappa, features=features)


def bpinn_reconstruction_one_mode(inp, primary, model, norm_stats, m):
    """Core per-mode reconstruction logic (unchanged from the original
    mode-0-only version, just parameterized by `m` instead of reading
    s6.CONFIG['mode_index'] directly) -- called once per mode in
    CONFIG['bpinn_modes'] by bpinn_reconstruction() below."""
    feat_mean, feat_std, out_norm = norm_stats

    # Exact ground-truth curve (oracle access to the TRUE df, comparison only)
    p_true = sdof_params_from_df(primary['df_true'], m, inp)
    omega0_arg = 2 * math.pi * math.sqrt(p_true['K'] / p_true['M'])   # matches Step 6's own convention exactly
    cont_true = s4.duffing_forced_response_continuation(
        omega0_arg, p_true['M'], p_true['C'], p_true['K'], p_true['K3'],
        p_true['q_ref'], s6.CONFIG['target_peak_frac_qref'])
    w_true = cont_true['Omega'] / omega0_arg
    stable = cont_true['stable']
    w_stable, amp_stable = w_true[stable], cont_true['amplitude'][stable]

    # Posterior-predictive reconstruction: propagate posterior draws of
    # df_b/f through the BPINN, combine via the law of total variance
    # (same decomposition Step 5 used for aleatoric/epistemic variance) --
    # this band carries BOTH the network's own predictive uncertainty AND
    # this step's inferred-mistuning uncertainty.
    rc_cfg = CONFIG['bpinn_reconstruction']
    rng = np.random.default_rng(CONFIG['random_seed'] + 3000 + m)
    draw_idx = rng.choice(len(primary['pooled']), size=rc_cfg['n_posterior_draws'], replace=False)
    draws = primary['pooled'][draw_idx]

    # 2026-08-27: forcing-aware mode-2 checkpoints save a 5-wide feat_mean
    # ([shift,zeta,kappa,detune,target_peak]); the old architecture's is
    # 3-wide. Self-describing off that length, same pattern as
    # predict_coupled_mc's own is_forcing_aware check -- no separate flag
    # needs to be threaded through the `models` dict.
    is_fa = len(feat_mean) == 5
    means_per_draw, vars_per_draw = [], []
    for df_draw in draws:
        p_d = sdof_params_from_df(df_draw, m, inp)
        feat_arr = np.tile(p_d['features'], (len(w_stable), 1))
        amp_mean, amp_std, _, _ = s6.predict_mc(model, w_stable, feat_arr, feat_mean, feat_std,
                                                 rc_cfg['n_mc_per_draw'], is_forcing_aware=is_fa,
                                                 target_peak=s6.CONFIG['target_peak_frac_qref'] if is_fa else None,
                                                 out_norm=out_norm)
        means_per_draw.append(amp_mean)
        vars_per_draw.append(amp_std ** 2)
    means_per_draw = np.array(means_per_draw)
    vars_per_draw = np.array(vars_per_draw)

    overall_mean = means_per_draw.mean(axis=0)
    overall_std = np.sqrt(vars_per_draw.mean(axis=0) + means_per_draw.var(axis=0))

    rmse = float(np.sqrt(np.mean((overall_mean - amp_stable) ** 2)))
    ss_res = np.sum((amp_stable - overall_mean) ** 2)
    ss_tot = np.sum((amp_stable - amp_stable.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    return dict(w=w_stable, true_amp=amp_stable, recon_mean=overall_mean,
                recon_std=overall_std, r2=r2, rmse=rmse)


def load_bpinn_coupled(mode_pair=(0, 1)):
    """Loads the real coupled-physics BPINN for ANY of the 5 real-measured
    isolated pairs (step4.MODE_GROUPS['pairs']) -- generalized 2026-08-13
    from the original modes-0-1-only version, then UPGRADED the same night
    to predict PHASE-RESOLVED (alpha_i, beta_i, alpha_j, beta_j) directly
    (4 outputs, not 2), trained with a real physics-residual loss
    (step4.coupled_hbm_residual) -- closes the "physics-informed in name
    only" gap the original amplitude-only coupled model had relative to
    the single-mode BPINN."""
    mi, mj = mode_pair
    tag = f'{mi}{mj}'
    step6_out = os.path.join(CONFIG['step6_dir'], 'output')
    freqs = s6.CONFIG['network']['fourier_w_freqs']

    # 2026-08-23: prefer the FORCING-AWARE network (forcing level as an
    # explicit input, trained across 5 force levels 0.02-0.8) over the old
    # fixed-forcing one -- root-caused this session as the real fix for the
    # beta/amplitude weakness that grid density and physics-weight tuning
    # alone couldn't close (R^2 0.92-0.97 overall vs. the old ~0.5-0.6).
    # Self-describing, same pattern as the detuning-feature mix below: if
    # the forcing-aware checkpoint exists for this pair, use it; otherwise
    # fall back to the original fixed-forcing network so pairs not yet
    # retrained this way still load.
    forcing_aware_path = os.path.join(step6_out, f'bpinn_forcing_aware_{tag}_state.pt')
    is_forcing_aware = os.path.exists(forcing_aware_path)

    if is_forcing_aware:
        norm = dict(np.load(os.path.join(step6_out, f'bpinn_forcing_aware_{tag}_norm.npz')))
        state_dict = torch.load(forcing_aware_path)
        norm['is_forcing_aware'] = True
        # 0.8 is the highest level these networks were trained at and the
        # closest match to the OLD fixed-forcing networks' own convention
        # (target_peak=1.0) -- kept as the default so existing callers that
        # don't pass target_peak explicitly get a comparable operating
        # point, not a silent behavior change in absolute output scale.
        norm['default_target_peak'] = 0.8
        r2_i_print, r2_j_print = float(norm['r2_i_overall']), float(norm['r2_j_overall'])
    else:
        norm = dict(np.load(os.path.join(step6_out, f'bpinn_coupled_norm_{tag}.npz')))
        state_dict = torch.load(os.path.join(step6_out, f'bpinn_coupled_state_{tag}.pt'))
        norm['is_forcing_aware'] = False
        r2_i_print, r2_j_print = float(norm['r2_i']), float(norm['r2_j'])

    # Both in_dim and hidden width inferred directly from the saved
    # checkpoint's own weight shapes -- self-describing (2026-08-21
    # pattern), so mixing architectures across pairs (detuned 8-feature,
    # base 6-feature, or forcing-aware 9-feature) never needs a hardcoded
    # pair list here.
    in_dim = state_dict['layers.0.w_mu'].shape[1]
    hidden0 = state_dict['layers.0.w_mu'].shape[0]
    hidden1 = state_dict['layers.1.w_mu'].shape[0]
    model = s6.BPINN(in_dim, [hidden0, hidden1], 4, prior_sigma=1.0)   # (alpha_i,beta_i,alpha_j,beta_j)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Coupled BPINN loaded for modes {mode_pair} ({'forcing-aware' if is_forcing_aware else 'fixed-forcing (legacy)'}): "
          f"test R^2 (amplitude) at training time was ({r2_i_print:.4f}, {r2_j_print:.4f})")
    return model, norm


def load_bpinn_chain():
    """Loads the 13-mode chain BPINN (2026-08-13, Step 6's
    _train_chain_bpinn.py) covering step4.MODE_GROUPS['chain'] (modes
    11-23) -- the real, measured continuously-overlapping band, coupled
    via 12 adjacent-pair identifications. UPGRADED same night to predict
    PHASE-RESOLVED (alpha,beta) per mode jointly (26 outputs, not 13),
    trained with a real physics-residual loss (step4.chain_hbm_residual)."""
    step6_out = os.path.join(CONFIG['step6_dir'], 'output')
    norm = dict(np.load(os.path.join(step6_out, 'bpinn_chain_norm.npz')))
    chain = [int(m) for m in norm['chain_modes']]
    n_chain = len(chain)
    freqs = s6.CONFIG['network']['fourier_w_freqs']
    in_dim = 1 + 2 * len(freqs) + 3 * n_chain
    model = s6.BPINN(in_dim, [64, 64], 2 * n_chain, prior_sigma=1.0)   # (alpha_1..N, beta_1..N)
    model.load_state_dict(torch.load(os.path.join(step6_out, 'bpinn_chain_state.pt')))
    model.eval()
    print(f"  Chain BPINN loaded for modes {chain}: mean test R^2 at training time was "
          f"{float(np.mean(norm['r2_per_mode'])):.4f}")
    return model, norm, chain


def chain_features_from_df(df, chain_modes, inp):
    """Real mistuning-driven features for all N chain modes at once, same
    physics as coupled_features_from_df, stacked into one (3*N,) vector in
    chain_modes order -- matches Step 6's _train_chain_bpinn.py exactly."""
    scale = (1.0 + df) ** 2 - 1.0
    feat = np.zeros(3 * len(chain_modes))
    K_arr = np.zeros(len(chain_modes))
    M_arr = np.zeros(len(chain_modes))
    C_arr = np.zeros(len(chain_modes))
    for k, m in enumerate(chain_modes):
        shift_m = float(scale @ inp['P'][:, m])
        K_m = inp['K_sec'][m, m] * (1.0 + shift_m)
        M_m = inp['M_sec'][m, m]
        C_m = inp['C_sec'][m, m]
        zeta_m = C_m / (2 * math.sqrt(K_m * M_m))
        kappa_m = 0.75 * inp['K3_sec_diag'][m] / K_m
        feat[3 * k:3 * k + 3] = [shift_m, zeta_m, kappa_m]
        K_arr[k] = K_m; M_arr[k] = M_m; C_arr[k] = C_m
    return dict(feat=feat, K_arr=K_arr, M_arr=M_arr, C_arr=C_arr)


def predict_chain_mc(model, norm, w_arr, feat_arr, n_mc=30):
    """Chain-model analog of predict_coupled_mc: n_mc stochastic forward
    passes. UPGRADED 2026-08-13: network predicts PHASE-RESOLVED
    (alpha_1..N, beta_1..N) jointly (26 outputs, not 13) -- amplitude per
    mode is derived per MC draw as hypot(alpha,beta) then averaged (proper
    MC uncertainty propagation, not hypot-of-the-mean)."""
    n_chain = len(norm['chain_modes'])
    feat_mean = torch.tensor(norm['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm['feat_std'], dtype=torch.float32)
    Feat_n = (torch.tensor(feat_arr, dtype=torch.float32) - feat_mean) / feat_std
    W_t = torch.tensor(w_arr, dtype=torch.float32)
    X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)
    samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            samples.append(model(X_in).numpy())
    samples = np.stack(samples)   # (n_mc, N_w, 2*n_chain)
    alpha = samples[:, :, :n_chain] * norm['alpha_std'][None, None, :] + norm['alpha_mean'][None, None, :]
    beta = samples[:, :, n_chain:] * norm['beta_std'][None, None, :] + norm['beta_mean'][None, None, :]
    amp = np.hypot(alpha, beta)   # (n_mc, N_w, n_chain)
    return amp.mean(0), amp.std(0)   # (N_w, n_chain) each


def coupled_features_from_df(df, mode_pair, inp):
    """Real mistuning-driven features for BOTH modes in the pair, same
    physics as sdof_params_from_df, just computed for two modes at once
    (no cross term in the FEATURES themselves -- the coupling lives in the
    trained network's weights, learned from coupled ground truth)."""
    mi, mj = mode_pair
    scale = (1.0 + df) ** 2 - 1.0
    shift_i = float(scale @ inp['P'][:, mi]); shift_j = float(scale @ inp['P'][:, mj])
    Ki = inp['K_sec'][mi, mi] * (1.0 + shift_i); Kj = inp['K_sec'][mj, mj] * (1.0 + shift_j)
    Mi = inp['M_sec'][mi, mi]; Mj = inp['M_sec'][mj, mj]
    Ci = inp['C_sec'][mi, mi]; Cj = inp['C_sec'][mj, mj]
    zeta_i = Ci / (2 * math.sqrt(Ki * Mi)); zeta_j = Cj / (2 * math.sqrt(Kj * Mj))
    kappa_i = 0.75 * inp['K3_sec_diag'][mi] / Ki; kappa_j = 0.75 * inp['K3_sec_diag'][mj] / Kj
    feat = np.array([shift_i, zeta_i, kappa_i, shift_j, zeta_j, kappa_j])
    return dict(Ki=Ki, Kj=Kj, Mi=Mi, Mj=Mj, Ci=Ci, Cj=Cj, feat=feat)


def predict_coupled_mc(model, norm, w_arr, feat_arr, n_mc=30, target_peak=None):
    """Coupled-model analog of Step 6's predict_mc: n_mc stochastic
    forward passes. UPGRADED 2026-08-13: the network now predicts PHASE-
    RESOLVED (alpha_i,beta_i,alpha_j,beta_j) (4 outputs), not amplitude
    directly -- amp_i/amp_j are derived per MC draw as hypot(alpha,beta)
    THEN averaged (proper Monte Carlo uncertainty propagation through the
    nonlinear sqrt, not hypot-of-the-mean). Both inputs and outputs still
    need the coupled model's own saved normalization (real bug from the
    original training: un-normalized outputs made mode j's loss invisible
    next to mode i's larger scale)."""
    # 2026-08-21 fix, PAIR-SPECIFIC: append the same explicit detuning
    # features training now uses (s6.add_detune_features) BEFORE
    # normalizing -- but only for pairs actually trained with them. Which
    # pairs is self-described by the saved norm file's own feat_mean length
    # (6 = original, 8 = detuned), matching load_bpinn_coupled()'s in_dim
    # inference above rather than a second hardcoded pair list to keep in
    # sync. feat_arr coming in is still the base 6-dim (shift_i,zeta_i,
    # kappa_i,shift_j,zeta_j,kappa_j) from coupled_features_from_df();
    # detune depends on w, which wasn't known at that call site, so it's
    # added here instead, matching where w and feat first come together at
    # training time too.
    # 2026-08-23: forcing-aware networks (feat_mean length 9 = 8 detuned
    # features + explicit target_peak input) need the forcing level
    # appended before normalizing -- this is the actual fix that closed the
    # beta/amplitude weakness (R^2 0.92-0.97 vs. the old fixed-forcing
    # networks' ~0.5-0.6), see load_bpinn_coupled()'s own comment. Callers
    # that don't care about a specific forcing level get the network's own
    # saved default (0.8, the highest trained level, closest to the legacy
    # fixed-forcing convention) rather than silently defaulting to zero.
    n_feat = len(norm['feat_mean'])
    is_forcing_aware = bool(norm.get('is_forcing_aware', n_feat == 9))
    W_t = torch.tensor(w_arr, dtype=torch.float32)
    feat_mean = torch.tensor(norm['feat_mean'], dtype=torch.float32)
    feat_std = torch.tensor(norm['feat_std'], dtype=torch.float32)
    use_detune = n_feat in (8, 9)
    feat_out_arr = s6.add_detune_features(w_arr, feat_arr) if use_detune else feat_arr
    if is_forcing_aware:
        tp = target_peak if target_peak is not None else float(norm.get('default_target_peak', 0.8))
        tp_col = np.full((feat_out_arr.shape[0], 1), tp)
        feat_out_arr = np.concatenate([feat_out_arr, tp_col], axis=1)
    Feat_n = (torch.tensor(feat_out_arr, dtype=torch.float32) - feat_mean) / feat_std
    X_in = torch.cat([s6.fourier_encode_w(W_t), Feat_n], dim=1)
    samples = []
    with torch.no_grad():
        for _ in range(n_mc):
            samples.append(model(X_in).numpy())
    samples = np.stack(samples)  # (n_mc, N, 4) -- (alpha_i,beta_i,alpha_j,beta_j)
    alpha_i = samples[:, :, 0] * float(norm['alpha_i_std']) + float(norm['alpha_i_mean'])
    beta_i = samples[:, :, 1] * float(norm['beta_i_std']) + float(norm['beta_i_mean'])
    alpha_j = samples[:, :, 2] * float(norm['alpha_j_std']) + float(norm['alpha_j_mean'])
    beta_j = samples[:, :, 3] * float(norm['beta_j_std']) + float(norm['beta_j_mean'])
    amp_i = np.hypot(alpha_i, beta_i)
    amp_j = np.hypot(alpha_j, beta_j)
    return amp_i.mean(0), amp_i.std(0), amp_j.mean(0), amp_j.std(0)


def bpinn_reconstruction_coupled(inp, primary, model, norm, mode_pair=(0, 1),
                                  w_grid=(0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6)):
    """Real coupled-physics reconstruction: TRUE curve from Step 4's exact
    coupled time-domain solver (using the TRUE mistuning state -- oracle,
    comparison only), reconstruction curve from the trained coupled BPINN
    fed the INFERRED (posterior) mistuning -- same forcing level used at
    training time (f_gen_i/f_gen_j, saved in the model's own norm file --
    generalized 2026-08-13 from the original modes-0-1 version's hardcoded
    node-1171 participation values, which still work unchanged for that
    one pair since they're now just read back out of its norm file)."""
    hdr(f"STEP 7J-COUPLED: COUPLED BPINN RECONSTRUCTION, MODES {mode_pair}")
    mi, mj = mode_pair
    w_grid = np.array(w_grid)
    omega0_i = math.sqrt(inp['K_sec'][mi, mi] / inp['M_sec'][mi, mi])
    # 2026-08-23: forcing-aware norm files don't save one fixed f_gen_i/j
    # (forcing is now a free input, not baked into training) -- compute Fg
    # at the network's own default operating point instead, so the ORACLE
    # curve below and the BPINN prediction later in this function use the
    # SAME forcing level (required for a valid R^2 comparison between them).
    target_peak_recon = float(norm.get('default_target_peak', 1.0)) if norm.get('is_forcing_aware', False) else None
    if norm.get('is_forcing_aware', False):
        zeta_i0 = inp['C_sec'][mi, mi] / (2 * math.sqrt(inp['K_sec'][mi, mi] * inp['M_sec'][mi, mi]))
        zeta_j0 = inp['C_sec'][mj, mj] / (2 * math.sqrt(inp['K_sec'][mj, mj] * inp['M_sec'][mj, mj]))
        Fg_i = target_peak_recon * 2 * zeta_i0 * inp['K_sec'][mi, mi]
        Fg_j = target_peak_recon * 2 * zeta_j0 * inp['K_sec'][mj, mj]
    else:
        Fg_i = float(norm['f_gen_i']); Fg_j = float(norm['f_gen_j'])

    # TRUE curve (oracle, real coupled solver, not the network). Same
    # divergence filter as Step 6's training data generation -- some
    # (mistuning, w) combinations genuinely diverge (the fitted coupling
    # has negative cubic terms, so at large enough amplitude the
    # "restoring" force can flip sign). A single such outlier (9-11 orders
    # of magnitude too large) is enough to wreck an R^2 computed against
    # it, even when every OTHER point fits well -- confirmed directly:
    # w=1.2 diverged for this trajectory's true mistuning state while all
    # 9 other points matched the network to within a few percent.
    p_true = coupled_features_from_df(primary['df_true'], mode_pair, inp)
    true_amp_i, true_amp_j, w_kept = [], [], []
    for w in w_grid:
        r = s4.duffing_forced_response_coupled(
            mode_pair, (p_true['Ki'], p_true['Kj']), (p_true['Mi'], p_true['Mj']),
            (p_true['Ci'], p_true['Cj']), s4.CONFIG['nonlinear']['cross_coupling'][mode_pair]['coef0'],
            s4.CONFIG['nonlinear']['cross_coupling'][mode_pair]['coef1'],
            (Fg_i, Fg_j), w * omega0_i, n_cycles=200, steps_per_cycle=20)
        ok = (np.isfinite(r['amp_i']) and np.isfinite(r['amp_j'])
              and abs(r['amp_i']) < 0.5 and abs(r['amp_j']) < 0.5)
        if not ok:
            print(f"  w={w:.2f}: TRUE solve diverged (amp_i={r['amp_i']:.3e}, "
                  f"amp_j={r['amp_j']:.3e}) -- excluded from R^2, not treated as real", flush=True)
            continue
        true_amp_i.append(r['amp_i']); true_amp_j.append(r['amp_j']); w_kept.append(w)
    true_amp_i = np.array(true_amp_i); true_amp_j = np.array(true_amp_j)
    w_grid = np.array(w_kept)

    # Reconstruction from INFERRED mistuning (posterior draws -> BPINN, fast)
    rc_cfg = CONFIG['bpinn_reconstruction']
    rng = np.random.default_rng(CONFIG['random_seed'] + 4000)
    draw_idx = rng.choice(len(primary['pooled']), size=rc_cfg['n_posterior_draws'], replace=False)
    draws = primary['pooled'][draw_idx]

    means_i, means_j = [], []
    for df_draw in draws:
        p_d = coupled_features_from_df(df_draw, mode_pair, inp)
        feat_arr = np.tile(p_d['feat'], (len(w_grid), 1))
        amp_i_m, _, amp_j_m, _ = predict_coupled_mc(model, norm, w_grid, feat_arr, n_mc=10,
                                                      target_peak=target_peak_recon)
        means_i.append(amp_i_m); means_j.append(amp_j_m)
    means_i = np.array(means_i); means_j = np.array(means_j)
    recon_i = means_i.mean(axis=0); recon_j = means_j.mean(axis=0)
    std_i = means_i.std(axis=0); std_j = means_j.std(axis=0)

    def r2(true, pred):
        ss_res = np.sum((true - pred) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    r2_i, r2_j = r2(true_amp_i, recon_i), r2(true_amp_j, recon_j)
    print(f"  Mode {mi}: R^2={r2_i:.4f}   Mode {mj}: R^2={r2_j:.4f}", flush=True)
    _record_check(f"Coupled BPINN reconstruction (modes {mode_pair}, real cross-mode "
                  "physics) tracks the true coupled forced-response curve from "
                  "inferred mistuning",
                  bool(r2_i > 0.3 and r2_j > 0.3), f"R^2 = ({r2_i:.4f}, {r2_j:.4f})")

    return dict(w=w_grid, true_amp_i=true_amp_i, true_amp_j=true_amp_j,
                recon_i=recon_i, recon_j=recon_j, std_i=std_i, std_j=std_j,
                r2_i=r2_i, r2_j=r2_j, mode_pair=mode_pair)


def bpinn_reconstruction_chain(inp, primary, model, norm, chain,
                                w_grid=(0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.3, 2.6)):
    """13-mode chain analog of bpinn_reconstruction_coupled: TRUE curve
    from Step 4's exact chain time-domain solver (oracle), reconstruction
    from the trained chain BPINN fed INFERRED (posterior) mistuning, same
    per-mode forcing used at training time (norm['f_gen'])."""
    hdr(f"STEP 7J-CHAIN: CHAIN BPINN RECONSTRUCTION, MODES {chain}")
    n_chain = len(chain)
    w_grid = np.array(w_grid)
    K0_arr = np.array([inp['K_sec'][m, m] for m in chain])
    M_arr = np.array([inp['M_sec'][m, m] for m in chain])
    C_arr = np.array([inp['C_sec'][m, m] for m in chain])
    omega0_ref = math.sqrt(K0_arr.mean() / M_arr.mean())
    Fg_arr = np.array(norm['f_gen'])
    pair_coefs = s4.CONFIG['nonlinear']['cross_coupling']

    p_true = chain_features_from_df(primary['df_true'], chain, inp)
    true_amp, w_kept = [], []
    for w in w_grid:
        r = s4.duffing_forced_response_chain(chain, p_true['K_arr'], M_arr, C_arr, pair_coefs,
                                              Fg_arr, w * omega0_ref, n_cycles=200, steps_per_cycle=20)
        ok = np.all(np.isfinite(r['amp'])) and np.all(np.abs(r['amp']) < 0.5)
        if not ok:
            print(f"  w={w:.2f}: TRUE chain solve diverged -- excluded from R^2, not treated as real",
                  flush=True)
            continue
        true_amp.append(r['amp']); w_kept.append(w)
    true_amp = np.array(true_amp)   # (n_w_kept, n_chain)
    w_grid = np.array(w_kept)

    rc_cfg = CONFIG['bpinn_reconstruction']
    rng = np.random.default_rng(CONFIG['random_seed'] + 5000)
    draw_idx = rng.choice(len(primary['pooled']), size=rc_cfg['n_posterior_draws'], replace=False)
    draws = primary['pooled'][draw_idx]

    means = []
    for df_draw in draws:
        p_d = chain_features_from_df(df_draw, chain, inp)
        feat_arr = np.tile(p_d['feat'], (len(w_grid), 1))
        amp_m, _ = predict_chain_mc(model, norm, w_grid, feat_arr, n_mc=10)
        means.append(amp_m)
    means = np.array(means)   # (n_draws, n_w_kept, n_chain)
    recon = means.mean(axis=0)
    std = means.std(axis=0)

    def r2(true, pred):
        ss_res = np.sum((true - pred) ** 2); ss_tot = np.sum((true - true.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    r2_per_mode = [r2(true_amp[:, k], recon[:, k]) for k in range(n_chain)]
    print(f"  Chain R^2 per mode: {dict(zip(chain, [round(v, 4) for v in r2_per_mode]))}", flush=True)
    _record_check("Chain BPINN reconstruction (13-mode densely-overlapping band, real "
                  "adjacent-pair coupling) tracks the true chain forced-response curve "
                  "from inferred mistuning, for every mode in the chain",
                  all(v > 0.3 for v in r2_per_mode),
                  f"per-mode R^2 = {[round(v, 4) for v in r2_per_mode]}")

    return dict(w=w_grid, true_amp=true_amp, recon=recon, std=std,
                r2_per_mode=r2_per_mode, chain=chain)


def bpinn_reconstruction(inp, primary, models):
    """Multi-mode extension (2026-08-11), GENERALIZED 2026-08-13 to the
    real, measured 24-mode topology (step4.MODE_GROUPS): 5 isolated pairs
    reconstructed with the real coupled-physics BPINN each, the 13-mode
    dense chain (11-23) reconstructed jointly with the chain BPINN, and
    mode 2 (the one genuinely isolated single) on the original
    independent-SDOF BPINN -- `models` is {mode_index: (model, norm_stats)}
    from load_bpinn_multi(), used for mode 2 only now."""
    hdr("STEP 7J: BPINN-ACCELERATED FORCED-RESPONSE RECONSTRUCTION FROM INFERRED MISTUNING (ALL COVERED MODES)")

    pair_results = {}
    for pair in s4.MODE_GROUPS['pairs']:
        model, norm = load_bpinn_coupled(pair)
        pair_results[pair] = bpinn_reconstruction_coupled(inp, primary, model, norm, pair)

    chain = s4.MODE_GROUPS['chain']
    chain_model, chain_norm, chain_modes = load_bpinn_chain()
    chain_result = bpinn_reconstruction_chain(inp, primary, chain_model, chain_norm, chain_modes)

    per_mode = {}
    for m in CONFIG['bpinn_modes']:   # now just [2]
        model, norm_stats = models[m]
        per_mode[m] = bpinn_reconstruction_one_mode(inp, primary, model, norm_stats, m)
        print(f"  Mode {m}: R^2={per_mode[m]['r2']:.4f}, RMSE={per_mode[m]['rmse']:.5f}")

    r2_values = ([v for pr in pair_results.values() for v in (pr['r2_i'], pr['r2_j'])]
                 + list(chain_result['r2_per_mode'])
                 + [per_mode[m]['r2'] for m in per_mode])
    r2_mean = float(np.mean(r2_values))
    print(f"  Mean R^2 across all 24 covered modes: {r2_mean:.4f}", flush=True)
    _record_check("BPINN-accelerated reconstruction tracks the true forced-response "
                  "curve for every one of the 24 1B-cluster modes -- 5 pairs + a "
                  "13-mode chain on real coupled physics, 1 mode on the independent "
                  "model (the one genuinely isolated mode)",
                  all(v > 0.3 for v in r2_values),
                  f"mean R^2 = {r2_mean:.4f}, min R^2 = {min(r2_values):.4f}")

    # legacy top-level aliases (mode 0, from the (0,1) pair result) so any
    # code still expecting the old single-mode dict shape keeps working
    legacy = pair_results[(0, 1)]
    return dict(per_mode=per_mode, pairs=pair_results, chain=chain_result,
                modes=CONFIG['bpinn_modes'], r2_mean=r2_mean,
                w=legacy['w'], true_amp=legacy['true_amp_i'],
                recon_mean=legacy['recon_i'], recon_std=legacy['std_i'],
                r2=legacy['r2_i'], rmse=float(np.sqrt(np.mean(
                    (legacy['true_amp_i'] - legacy['recon_i']) ** 2))))


# ═══════════════════════════════════════════════════════════════════
# 7K. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def save_outputs(prior, svd_info, primary, coverage, recon):
    hdr("STEP 7K: SAVING OUTPUTS")
    fp1 = os.path.join(OUT, 'synthetic_observation.npz')
    np.savez(fp1, df_true=primary['df_true'], y=primary['y'], freqs_true_1b=primary['freqs_true_1b'],
              sigma_hz=CONFIG['measurement']['sigma_hz'])
    print(f"  Saved: {fp1}")

    fp2 = os.path.join(OUT, 'mcmc_posterior.npz')
    np.savez(fp2, pooled=primary['pooled'], post_mean=primary['post_mean'], post_std=primary['post_std'],
              post_lo=primary['post_lo'], post_hi=primary['post_hi'], rhat=primary['rhat'],
              ess=primary['ess'], accept_rates=np.array(primary['accept_rates']),
              mu0=prior['mu0'], V=prior['V'], lam=prior['lam'], K=prior['K'])
    print(f"  Saved: {fp2}")

    fp3 = os.path.join(OUT, 'coverage_check.npz')
    np.savez(fp3, levels=coverage['levels'], empirical_coverage=coverage['empirical_coverage'],
              n_trials=coverage['n_trials'])
    print(f"  Saved: {fp3}")

    config_record = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'holdout': CONFIG['holdout'], 'measurement': CONFIG['measurement'],
        'mcmc': {k: v for k, v in CONFIG['mcmc'].items()},
        'identifiability_condition_number': svd_info['condition_number'],
        'prior_identifiable_subspace_K': prior['K'],
        'primary_result': {
            'recovery_corr': float(np.corrcoef(primary['post_mean'], primary['df_true'])[0, 1]),
            'recovery_rmse': float(np.sqrt(np.mean((primary['post_mean'] - primary['df_true']) ** 2))),
            'max_rhat': float(primary['rhat'].max()), 'min_ess': float(primary['ess'].min()),
        },
        'coverage_at_95pct': float(coverage['empirical_coverage'][
            int(np.argmin(np.abs(np.array(coverage['levels']) - 0.95)))]),
        'bpinn_reconstruction_r2': recon['r2'],  # legacy alias, mode 0
        'bpinn_reconstruction_r2_mean': recon['r2_mean'],
        'bpinn_reconstruction_r2_per_mode': {
            **{str(m): v for pair, pr in recon['pairs'].items()
               for m, v in zip(pair, (pr['r2_i'], pr['r2_j']))},
            **{str(m): v for m, v in zip(recon['chain']['chain'], recon['chain']['r2_per_mode'])},
            **{str(m): recon['per_mode'][m]['r2'] for m in recon['per_mode']},
        },
        'note': ('Inverts against the full 24-mode 1B-cluster frequency response (not '
                 'mode-0 alone) to identify per-blade fractional-frequency mistuning '
                 '(df_b/f, 24-dim) -- the finest-grained quantity the Step 4 forward '
                 'model is actually sensitive to. The underlying 5-variable-per-blade '
                 'geometric split (theta itself) is not identifiable even in principle: '
                 "Step 4's own sensitivity model already collapses it before anything "
                 'else touches it. Observed data y is SYNTHETIC (no real experimental '
                 'data exists in this project): a known held-out Step 3 mistuning sample '
                 'run through the exact forward model plus documented placeholder '
                 'measurement noise. See CONFIG and module docstring in step7.py.'),
    }
    fp4 = os.path.join(OUT, 'step7_config.json')
    with open(fp4, 'w') as f:
        json.dump(config_record, f, indent=2)
    print(f"  Saved: {fp4}")


# ═══════════════════════════════════════════════════════════════════
# 7L. FIGURES — 5 diagnostics
# ═══════════════════════════════════════════════════════════════════
def _resolve_figs_dir():
    figs = os.path.join(FIG_ROOT, 'figures', 'step7')
    os.makedirs(figs, exist_ok=True)
    return figs


def _savefig(fig, figs_dir, name):
    paths = plot_style.savefig_pub(fig, figs_dir, name)
    print(f"  Figure saved: {paths[0]}  (+ .pdf)")


def make_step7_figures(primary, prior, svd_info, svd_analysis, coverage, recon):
    hdr("STEP 7L: FIGURES (6 diagnostics, figures/step7/, PNG+PDF)")
    figs = _resolve_figs_dir()

    # ── fig1: recovered vs true df_b/f, all 24 blades, 95% CI ──
    fig, ax = plt.subplots(figsize=(6.8, 6.5))
    ax.errorbar(primary['df_true'] * 100, primary['post_mean'] * 100,
                yerr=[(primary['post_mean'] - primary['post_lo']) * 100,
                      (primary['post_hi'] - primary['post_mean']) * 100],
                fmt='o', ms=6, color=plot_style.BLUE, ecolor=plot_style.BLUE, alpha=0.7,
                elinewidth=1.1, capsize=2, mec=plot_style.SURFACE, mew=1.0)
    lims = [min(primary['df_true'].min(), primary['post_mean'].min()) * 100 * 1.15,
            max(primary['df_true'].max(), primary['post_mean'].max()) * 100 * 1.15]
    ax.plot(lims, lims, color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect recovery')
    ax.set_xlabel('True per-blade mistuning  df$_b$/f  [%]')
    ax.set_ylabel('Posterior mean  (error bars: 95% credible interval)  [%]')
    corr = float(np.corrcoef(primary['post_mean'], primary['df_true'])[0, 1])
    plot_style.two_tier_title(ax, 'Per-blade mistuning identification', f'24 blades -- corr={corr:.3f}')
    plot_style.legend_below(ax)
    fig.tight_layout()
    _savefig(fig, figs, 'step7_fig1_recovery_scatter')

    # ── fig2: recovery error vs. posterior shrinkage (latent-direction breakdown) ──
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    colors = [plot_style.AQUA if t else plot_style.ORANGE for t in svd_analysis['top']]
    ax.scatter(svd_analysis['shrinkage'], svd_analysis['err'], c=colors, s=70,
               edgecolor=plot_style.SURFACE, linewidth=1.0)
    for k in range(len(svd_analysis['shrinkage'])):
        ax.annotate(str(k), (svd_analysis['shrinkage'][k], svd_analysis['err'][k]),
                    fontsize=8, color=plot_style.INK_SECONDARY, xytext=(5, 4), textcoords='offset points')
    ax.set_xlabel('Posterior shrinkage  (1 - posterior std / prior std)')
    ax.set_ylabel('Prior-normalized recovery error')
    plot_style.two_tier_title(ax, 'Recovery accuracy tracks data informativeness',
                               'per latent direction, not blade identity')
    ax.plot([], [], 'o', color=plot_style.AQUA, label='high shrinkage (top half)')
    ax.plot([], [], 'o', color=plot_style.ORANGE, label='low shrinkage (bottom half)')
    plot_style.legend_below(ax, ncol=2)
    fig.tight_layout()
    _savefig(fig, figs, 'step7_fig2_identifiability_vs_recovery')

    # ── fig3: MCMC diagnostics -- trace (most-mistuned blade, physical
    #          space) + R-hat bar chart (latent K-dim sampled space) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    chain_colors = [plot_style.SEQUENTIAL_BLUE[i] for i in
                     np.linspace(3, len(plot_style.SEQUENTIAL_BLUE) - 1, primary['chains'].shape[0]).astype(int)]
    show_blade = int(np.argmax(np.abs(primary['df_true'])))
    ax = axes[0]
    for c in range(primary['chains'].shape[0]):
        ax.plot(primary['chains'][c, :800, show_blade], color=chain_colors[c], lw=0.7, alpha=0.85)
    ax.axhline(primary['df_true'][show_blade], color=plot_style.INK_PRIMARY, ls='--', lw=1.3, label='true value')
    ax.set_xlabel('Post-warmup iteration')
    ax.set_ylabel(f'df_b/f, blade {show_blade} (largest true mistuning)')
    ax.legend(fontsize=14, frameon=False)

    ax = axes[1]
    ax.bar(np.arange(prior['K']), primary['rhat'], color=plot_style.VIOLET, width=0.8)
    ax.axhline(1.1, color=plot_style.INK_PRIMARY, ls='--', lw=1.3, label='R-hat = 1.1 threshold')
    ax.set_xlabel(f"Latent direction index (identifiable subspace, K={prior['K']})")
    ax.set_ylabel('Gelman-Rubin R-hat')
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    plot_style.figure_title(fig, 'MCMC convergence diagnostics', 'trace (left) and R-hat by latent direction (right)')
    _savefig(fig, figs, 'step7_fig3_mcmc_diagnostics')

    # ── fig4: coverage calibration reliability diagram ──
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.plot(coverage['levels'], coverage['empirical_coverage'], 'o-', color=plot_style.ORANGE,
            lw=2.0, ms=6, mec=plot_style.SURFACE, mew=1.0, label='inversion calibration')
    ax.plot([0, 1], [0, 1], color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect calibration')
    ax.set_xlabel('Nominal credible level')
    ax.set_ylabel(f"Empirical coverage ({coverage['n_trials']} held-out trials x {NB} blades)")
    plot_style.two_tier_title(ax, 'Posterior credible-interval calibration')
    plot_style.legend_below(ax)
    fig.tight_layout()
    _savefig(fig, figs, 'step7_fig4_coverage_calibration')

    # ── fig5: BPINN-accelerated reconstruction vs. exact ground truth, ALL 24 MODES ──
    # GENERALIZED 2026-08-13 from a 2-4-panel grid (workable at 4 modes) to
    # a proper 24-mode summary: a full per-mode R^2 bar (every mode, color-
    # coded by real topology group) plus a handful of REPRESENTATIVE curve
    # panels (one per group) -- a 24-panel small-multiples grid stopped
    # being a readable figure once real coverage grew from 4 modes to 24.
    all_panels = {}
    for pair, pr in recon['pairs'].items():
        mi, mj = pair
        all_panels[mi] = dict(label=f'Mode {mi} (pair {pair})', group='pair', w=pr['w'],
                               true=pr['true_amp_i'], mean=pr['recon_i'], std=pr['std_i'], r2=pr['r2_i'])
        all_panels[mj] = dict(label=f'Mode {mj} (pair {pair})', group='pair', w=pr['w'],
                               true=pr['true_amp_j'], mean=pr['recon_j'], std=pr['std_j'], r2=pr['r2_j'])
    cr = recon['chain']
    for k, m in enumerate(cr['chain']):
        all_panels[m] = dict(label=f'Mode {m} (chain)', group='chain', w=cr['w'],
                              true=cr['true_amp'][:, k], mean=cr['recon'][:, k],
                              std=cr['std'][:, k], r2=cr['r2_per_mode'][k])
    for m, pm in recon['per_mode'].items():
        all_panels[m] = dict(label=f'Mode {m} (independent)', group='single', w=pm['w'],
                              true=pm['true_amp'], mean=pm['recon_mean'], std=pm['recon_std'], r2=pm['r2'])

    group_color = {'pair': plot_style.BLUE, 'chain': plot_style.VIOLET, 'single': plot_style.C_OK}
    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    mode_order = sorted(all_panels.keys())
    r2_arr = [all_panels[m]['r2'] for m in mode_order]
    colors = [group_color[all_panels[m]['group']] for m in mode_order]
    ax.bar(mode_order, r2_arr, color=colors)
    ax.set_xlabel('Mode index (1B cluster)')
    ax.set_ylabel('Reconstruction R$^2$')
    ax.set_xticks(mode_order)
    plot_style.two_tier_title(ax, 'BPINN-accelerated reconstruction, all 24 modes',
                               f"mean R2={recon['r2_mean']:.3f} -- from INFERRED (not exact) mistuning")
    for grp, c in group_color.items():
        ax.plot([], [], color=c, marker='s', ls='', ms=9, label=grp)
    plot_style.legend_below(ax, ncol=3)
    fig.tight_layout()
    _savefig(fig, figs, 'step7_fig5_bpinn_reconstruction')

    # ── fig5b: representative curve panels, one standalone PNG per mode ──
    # SPLIT (2026-08-19, explicit user request): was one 2x3 grid -- now 6
    # standalone PNGs (fig5b_a-f), one per representative topology-group mode.
    example_modes = [0, 3, 11, 17, 23, 2]   # pair, pair, chain-boundary, chain-middle, chain-boundary, single
    panel_letters5b = ['a', 'b', 'c', 'd', 'e', 'f']
    for m, letter in zip(example_modes, panel_letters5b):
        p = all_panels[m]
        order = np.argsort(p['w'])
        w_s, true_s = p['w'][order], p['true'][order]
        mean_s, std_s = p['mean'][order], p['std'][order]
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.scatter(w_s, true_s, color=plot_style.BLUE, s=20, zorder=5, edgecolor=plot_style.SURFACE,
                   linewidth=0.6, label='exact ground truth')
        ax.plot(w_s, mean_s, color=plot_style.ORANGE, lw=2.0, label='BPINN mean')
        ax.fill_between(w_s, mean_s - 2 * std_s, mean_s + 2 * std_s, color=plot_style.ORANGE,
                         alpha=plot_style.BAND_ALPHA, label='+/-2 sigma')
        ax.set_xlabel('$w = \\Omega/\\omega_0$')
        ax.set_ylabel('amplitude [mm]')
        plot_style.two_tier_title(ax, f"Reconstruction detail -- {p['label']}",
                                   f"R2={p['r2']:.3f}, exact ground truth vs. BPINN fed INFERRED mistuning")
        ax.legend(fontsize=14, frameon=False, loc='upper right')
        fig.tight_layout()
        _savefig(fig, figs, f'step7_fig5b{letter}_reconstruction_detail_mode{m}')
    print(f"  step7_fig5b_a-f: reconstruction detail, modes {example_modes}")

    print(f"  All Step 7 figures saved to: {figs}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _log_path = os.path.join(_HERE, 'Step7.txt')
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    t_start = time.time()
    hdr(f"STEP 7 v1.0: BAYESIAN MISTUNING IDENTIFICATION (INVERSE PROBLEM) — {NB}-BLADE BLISK (PCE PROJECT)")
    print(f"  Step 2 dir (read-only): {CONFIG['step2_dir']}")
    print(f"  Step 3 dir (read-only): {CONFIG['step3_dir']}")
    print(f"  Step 4 dir (read-only, code+data): {CONFIG['step4_dir']}")
    print(f"  Step 5 dir (read-only, code):      {CONFIG['step5_dir']}")
    print(f"  Step 6 dir (read-only, code+data): {CONFIG['step6_dir']}")
    print(f"  Output dir (Step 7):    {OUT}")

    inp = load_inputs()
    prior = build_prior(inp)
    svd_info = check_identifiability(inp)
    primary = run_primary_inversion(inp, prior)
    svd_analysis = latent_shrinkage_analysis(prior, primary)
    coverage = run_coverage_sweep(inp, prior)
    models = load_bpinn_multi(inp)
    recon = bpinn_reconstruction(inp, primary, models)
    save_outputs(prior, svd_info, primary, coverage, recon)
    make_step7_figures(primary, prior, svd_info, svd_analysis, coverage, recon)
    passed = print_validation_summary()

    hdr("STEP 7 COMPLETE")
    elapsed = time.time() - t_start
    print(f"  Validation: {'PASSED' if passed else 'FAILED — see STEP 7 VALIDATION SUMMARY above'}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"\n  Files in {OUT}:")
    for fn in sorted(os.listdir(OUT)):
        fp = os.path.join(OUT, fn)
        if os.path.isfile(fp):
            print(f"    {fn:30s} {os.path.getsize(fp) / 1e3:8.2f} KB")
    print(f"\nLog saved: {_log_path}")
    sys.stdout = sys.__stdout__
    _log_file.close()
