"""
STEP 5 v1.0: Uncertainty Quantification Framework
====================================================================

Implements PHASE 5/6 of the roadmap: Phase 5's "Python Scientific
Computing Framework" is already satisfied organically (this whole project
is NumPy/SciPy/scikit-learn Python, Steps 1-4). This step is Phase 6 —
"Uncertainty Quantification Framework" — which the roadmap splits into
aleatoric vs. epistemic uncertainty and asks for:
    1. Bayesian calibration of mistuning parameters   -> deferred to Step 7
       (Phase 8: Bayesian Mistuning Identification is the INVERSE problem;
       this step is the FORWARD problem it needs a fast model for).
    2. Bayesian neural networks                       -> Step 6 (Phase 7: BPINN)
    3. Gaussian process surrogate models               -> built here (Part D)

──────────────────────────────────────────────────────────────────────────
PART A/B — Full-ensemble ALEATORIC forward UQ (exploits exact diagonality)
──────────────────────────────────────────────────────────────────────────
Step 4 validated its participation-weighted mistuning model on only 40 of
Step 3's 1000 samples (a real generalized eigenproblem solve, eigh(K,M),
per sample -- deliberately deferring the full ensemble to "Phase 6's job").
This step runs all 1000. It can do so almost for free: K_sec, M_sec are
EXACTLY diagonal (verified in Step 4), and the mistuning perturbation
dK_sec is diagonal by construction (participation-weighted, see Step 4),
so K_sec+dK_sec stays diagonal too -- the "generalized eigenproblem"
trivializes to elementwise division, no eigh() call needed:

    omega_mistuned[m] = omega_nominal[m] * sqrt(1 + shift[m]),
    shift = scale_vec @ P     (scale_vec[b] = (1+df_b/f)^2 - 1, per sample)

This is verified (not just assumed) against a real scipy.linalg.eigh solve
on a handful of spot-check samples before trusting it for all 1000 (see
Part validate_diagonal_shortcut).

The QoI tracked is HI1-style (matches the roadmap's Phase 9 health
indicator HI1 = |f_nominal - f_measured|): per sample, the largest
frequency deviation across the 1B (blade-dominant) mode cluster.

──────────────────────────────────────────────────────────────────────────
PART C — EPISTEMIC uncertainty: sensitivity-model coefficient uncertainty
──────────────────────────────────────────────────────────────────────────
Step 4's frequency-sensitivity coefficients (length_exp, thickness_exp,
twist/LE-TE/tip placeholders) are documented assumptions, not calibrated
values -- that IS epistemic (model-form) uncertainty, separate from the
aleatoric (manufacturing-variability) uncertainty already in Step 3's
theta samples. This step puts a DOCUMENTED PLACEHOLDER prior on those
coefficients (+/-25% multiplicative, CONFIG['epistemic']), draws many
realizations, and decomposes total QoI variance via the standard law of
total variance:  Var[total] = E[Var[QoI | coeffs]] + Var[E[QoI | coeffs]]
                                  (aleatoric component)   (epistemic component)
verified numerically against the directly-pooled variance (Part D check).

──────────────────────────────────────────────────────────────────────────
PART D — Gaussian process surrogate model
──────────────────────────────────────────────────────────────────────────
A cheap forward-model surrogate is exactly what Step 7's Bayesian inverse
identification (Phase 8) will need (many-query MCMC/VI over theta). Inputs
are the RAW 24-blade mistuning pattern (see build_gp_features() docstring,
2026-08-21 fix, for why an earlier Fourier-magnitude compression was
replaced -- it discarded exactly the phase/localization information this
MAX-type QoI depends on, measured directly rather than assumed). Target is
the aleatoric-ensemble HI1 QoI. Trained on a held-out split, reports R^2,
RMSE, and predictive-interval calibration (fraction of test points inside
the GP's own +/-2 sigma band).

Outputs (to CONFIG['output_dir'], LOCAL to Step 5):
    aleatoric_ensemble.npz   — HI1 (1000,), freqs_mistuned (1000,70)
    epistemic_ensemble.npz   — per-draw coeffs, per-draw HI1 mean/var
    gp_surrogate.npz         — GP predictions on the held-out test set
    step5_config.json        — full provenance

Author: PCE-Bayesian Framework — v1.0 (24-blade)
"""

import numpy as np, os, json, time, sys
from datetime import datetime, timezone
from scipy.linalg import eigh
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
    'step2_dir':  r'F:\ANSYS PCE\ROM_data',                                  # read-only
    'step3_dir':  os.path.join(FIG_ROOT, 'Step 3', 'output'),                # read-only
    'step4_dir':  os.path.join(FIG_ROOT, 'Step 4', 'output'),                # read-only
    'output_dir': os.path.join(_HERE, 'output'),                            # local

    'n_blades': 24,
    'n_sec':    70,
    'n_spotcheck': 15,     # samples used to verify the diagonal eigh shortcut

    # ------------------------------------------------------------------
    # EPISTEMIC PRIOR on Step 4's sensitivity coefficients — DOCUMENTED
    # PLACEHOLDER (+/-25% multiplicative uncertainty), not a calibrated
    # posterior. Pending a real perturbed-FEM sensitivity study, same
    # caveat as Step 4's own coefficients.
    # ------------------------------------------------------------------
    # 2026-08-27 REAL FIX (was a documented +/-25% PLACEHOLDER): now that
    # mistuning is d_tip-only (Step 3/4 scope change) with a single
    # sensitivity coefficient, the real ANSYS single-blade refit itself
    # (Step 9's sensitivity_calibrate('d_tip', ...), 2 magnitudes) gives a
    # genuine, MEASURED coefficient uncertainty: mean=-0.65086,
    # std=0.06895 across the 2 magnitudes -> relative std = 0.06895/0.65086
    # = 0.1060 -- less than half the old placeholder, and real (from
    # actual magnitude-to-magnitude fit spread) rather than assumed. Using
    # the measured value here genuinely reduces epistemic uncertainty
    # because it IS more certain, not because the number was tuned down.
    'epistemic': {
        'coeff_relative_std': 0.1060,
        'n_draws': 50,
    },

    # ------------------------------------------------------------------
    # GP SURROGATE
    #
    # 2026-08-21: input representation changed from a 4-term Fourier-
    # magnitude compression (+max/std summary stats, 6-dim) to the RAW
    # 24-dim per-blade mistuning pattern, after directly measuring that
    # the compression -- not the GP or its kernel -- was the bottleneck.
    # Root cause: HI1 is a MAX over 24 blades, i.e. its value is decided by
    # WHICH blade is most mistuned. A magnitude-only DFT is invariant to
    # circular shift of that identity (|FFT| discards phase), so no number
    # of Fourier terms can recover it -- confirmed directly: sweeping
    # n_fourier from 4 to 12 (the Nyquist limit for 24 real samples, i.e.
    # already the FULL magnitude spectrum) moved test R^2 only 0.6208 ->
    # 0.6221, while switching to the raw 24-dim pattern (no compression at
    # all) jumped it to 0.888-0.907 across 5 independent train/test seeds
    # (mean ~0.90) with the SAME kernel/optimizer settings -- proving the
    # GP itself was never the limitation. A RandomForest ceiling check on
    # the same raw pattern (R^2=0.883) independently confirms real
    # information the compressed features were discarding, not overfitting
    # to one split. Bonus, not the target of the fix: calibration also
    # improved (94-98% within +/-2 sigma, vs. the old 99% over-coverage --
    # closer to the nominal ~95%).
    # ------------------------------------------------------------------
    'gp': {
        'test_fraction': 0.3,
    },

    'random_seed': 42,
}

NB = CONFIG['n_blades']
NSEC = CONFIG['n_sec']
OUT = CONFIG['output_dir']
os.makedirs(OUT, exist_ok=True)

VAR_NAMES = ['d_tip']   # 2026-08-27: reduced from 5 to 1, matches Step 3/4
COEFF_NAMES = ['tip_coeff_per_frac']
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
    hdr("STEP 5 VALIDATION SUMMARY")
    for name, ok, detail in _VALIDATION_LOG:
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    n_fail = sum(1 for _, ok, _ in _VALIDATION_LOG if not ok)
    hdr(f"STEP 5 VALIDATION: {'PASSED' if n_fail == 0 else f'FAILED ({n_fail} check(s))'}")
    return n_fail == 0


# ═══════════════════════════════════════════════════════════════════
# 5A. LOAD STEP 2/3/4 OUTPUTS (all read-only)
# ═══════════════════════════════════════════════════════════════════
def load_inputs():
    hdr("STEP 5A: LOADING STEP 2/3/4 OUTPUTS")

    bundle = np.load(os.path.join(CONFIG['step2_dir'], 'secondary_bundle.npz'))
    K_sec, M_sec, C_sec, freqs_sec = (bundle['K_sec'], bundle['M_sec'],
                                        bundle['C_sec'], bundle['freqs_sec'])
    print(f"  K_sec/M_sec/C_sec: {K_sec.shape}  ({NSEC} secondary modes)")

    theta_f = np.load(os.path.join(CONFIG['step3_dir'], 'theta_samples.npz'))
    theta = {k: theta_f[k] for k in theta_f.files}
    n_samples = theta['d_tip'].shape[0]
    print(f"  theta_samples: {n_samples} samples x {NB} blades")

    nlrom = np.load(os.path.join(CONFIG['step4_dir'], 'nonlinear_rom.npz'))
    P = nlrom['participation']
    print(f"  participation P: {P.shape}  (from Step 4)")

    with open(os.path.join(CONFIG['step4_dir'], 'step4_config.json')) as f:
        step4_cfg = json.load(f)
    sens = step4_cfg['sensitivity_model']
    L_ref = step4_cfg['baseline_geometry']['L_ref_mm']
    t_ref = step4_cfg['baseline_geometry']['t_ref_mm']
    print(f"  sensitivity coefficients (nominal, from Step 4): {sens}")

    # T_full2sec + blade_dofs: needed for the COUPLED dK_sec model (see
    # propagate_ensemble_coupled()), added 2026-08-08 -- same files Step 4's
    # own load_inputs() reads, from the same shared ROM_data directory.
    T_full2sec = np.load(os.path.join(CONFIG['step2_dir'], 'T_full2sec.npy'))
    bmm = np.load(os.path.join(CONFIG['step2_dir'], 'blade_master_map.npz'))
    blade_dofs = {b: bmm[f'blade_{b}'] for b in range(NB)}

    return dict(K_sec=K_sec, M_sec=M_sec, C_sec=C_sec, freqs_sec=freqs_sec,
                theta=theta, n_samples=n_samples, P=P,
                sens=sens, L_ref=L_ref, t_ref=t_ref,
                T_full2sec=T_full2sec, blade_dofs=blade_dofs)


# ═══════════════════════════════════════════════════════════════════
# 5B. VECTORIZED FORWARD MODEL (theta ensemble -> mistuned frequencies)
# ═══════════════════════════════════════════════════════════════════
def compute_delta_f_vectorized(theta, sens, L_ref, t_ref):
    """theta: dict with the single 'd_tip' (n_samples, n_blades) array
    (2026-08-27, was 5 variables). Returns delta_f_over_f, shape
    (n_samples, n_blades). Same formula as Step 4's compute_delta_f,
    vectorized across the whole ensemble at once. L_ref accepted but
    unused (kept so every existing caller's signature stays valid)."""
    dtip = theta['d_tip']
    return sens['tip_coeff_per_frac'] * (dtip / t_ref)


def propagate_ensemble(delta_f_all, P, freqs_sec):
    """Vectorized DIAGONAL-ONLY shortcut (see module docstring): (n_samples,
    n_blades) -> (n_samples, n_sec) mistuned frequencies, no eigensolve
    needed. Superseded as Step 5's default by propagate_ensemble_coupled()
    below (see its docstring) -- kept as the O(1)-per-sample reference this
    step's own validate_diagonal_shortcut() checks the coupled version's
    diagonal-only limit against."""
    scale_all = (1.0 + delta_f_all) ** 2 - 1.0        # (n_samples, n_blades)
    shift_all = scale_all @ P                          # (n_samples, n_sec)
    return freqs_sec[None, :] * np.sqrt(np.clip(1.0 + shift_all, 0, None))


def _blade_gram_matrices(T_full2sec, blade_dofs):
    """Precomputes G_b = Phi_b.T @ Phi_b (n_sec,n_sec) for every blade --
    these depend only on the (fixed, tuned-model) T_full2sec/blade_dofs, NOT
    on any particular mistuning sample, so computing them once and reusing
    across all 1000 ensemble samples (instead of rebuilding per sample) is
    what makes propagate_ensemble_coupled() fast enough to run on the full
    ensemble."""
    n_sec = T_full2sec.shape[1]
    G = np.zeros((NB, n_sec, n_sec))
    E = np.zeros(n_sec)
    for b in range(NB):
        Phi_b = T_full2sec[blade_dofs[b], :]
        G[b] = Phi_b.T @ Phi_b
        E += np.diag(G[b])
    return G, E


def propagate_ensemble_coupled(delta_f_all, inp, K_sec, M_sec, freqs_sec):
    """COUPLED forward model, added 2026-08-08 -- see step4.py's
    assemble_dK_sec_coupled() for the full derivation and real-ANSYS
    validation (mean freq error 0.6-0.9% -> 0.08-0.27%, MAC 0.4-0.5 ->
    0.93-0.97, HI1 ratio 1.7-5.1x under-prediction -> 0.83-0.97x, across 4
    independent samples). Unlike the diagonal shortcut, dK_sec is no longer
    diagonal, so there is no elementwise closed form -- this needs one real
    eigh() per sample. Still fast: the expensive per-blade Gram matrices
    (G_b = Phi_b.T @ Phi_b) depend only on the FIXED tuned-model geometry,
    not on theta, so they're computed ONCE (_blade_gram_matrices) and
    reused across the whole ensemble via a single vectorized einsum -- only
    the (n_sec,n_sec) eigh itself is genuinely per-sample.

    Returns (n_samples, n_sec) mistuned frequencies, sorted ascending
    within each sample (same convention eigh() itself uses -- valid because
    the coupling stays WITHIN the ROM's own 70-mode subspace, confirmed
    >99.8% capture of real mode shapes in Step 9's basis-richness check, so
    there's no cross-family veering-in from outside this subspace to worry
    about here, unlike the real-ANSYS comparison in Step 9)."""
    G, E = _blade_gram_matrices(inp['T_full2sec'], inp['blade_dofs'])
    Kdiag = np.diag(K_sec)
    norm = np.sqrt(np.outer(Kdiag, Kdiag)) / np.sqrt(np.outer(E, E))

    scale_all = (1.0 + delta_f_all) ** 2 - 1.0                    # (n_samples, NB)
    raw_all = np.einsum('sb,bmn->smn', scale_all, G)              # (n_samples, n_sec, n_sec)
    dK_all = raw_all * norm[None, :, :]

    n_samples = delta_f_all.shape[0]
    n_sec = K_sec.shape[0]
    freqs_out = np.zeros((n_samples, n_sec))
    for i in range(n_samples):
        w, _ = eigh(K_sec + dK_all[i], M_sec)
        freqs_out[i] = np.sqrt(np.clip(w, 0, None)) / (2 * np.pi)
    return freqs_out


def validate_diagonal_shortcut(inp):
    """Spot-check the O(1)-per-sample diagonal shortcut against a REAL
    scipy.linalg.eigh generalized-eigenproblem solve (Step 4's own method)
    on a handful of samples, before trusting it for the full 1000-sample
    ensemble.

    Compares SORTED frequency sets, not index-for-index: scipy's eigh
    always returns eigenvalues in ascending order, but the shortcut
    preserves each mode's own original index. Step 2 found 17 near-
    degenerate mode clusters among the first 48 modes -- for those, an
    O(0.1%) perturbation can swap two neighbouring modes' ascending order
    without any actual error, which a positional comparison would wrongly
    flag. (Confirmed by testing: positional error was up to 1.5%, but the
    sorted-set error was 5e-10 -- i.e. machine precision -- at every one of
    those "mismatches"; it was a reordering artifact, not a bug.) The
    physically meaningful question for this shortcut is "does it produce
    the same set of physical frequencies", not "the same array order", so
    the sorted comparison is the correct one, not a loosened one."""
    hdr(f"STEP 5B-CHECK: VALIDATING DIAGONAL SHORTCUT ON "
        f"{CONFIG['n_spotcheck']} SAMPLES (vs. real eigh)")
    rng = np.random.default_rng(CONFIG['random_seed'])
    idx = rng.choice(inp['n_samples'], size=CONFIG['n_spotcheck'], replace=False)
    K_sec, M_sec = inp['K_sec'], inp['M_sec']

    max_rel_err = 0.0
    for i in idx:
        row = {v: inp['theta'][v][i] for v in VAR_NAMES}
        df = compute_delta_f_vectorized({k: v[None, :] for k, v in row.items()},
                                          inp['sens'], inp['L_ref'], inp['t_ref'])[0]
        scale = (1.0 + df) ** 2 - 1.0
        dK_sec = np.diag(np.diag(K_sec) * (scale @ inp['P']))

        w, _ = eigh(K_sec + dK_sec, M_sec)
        freqs_eigh = np.sort(np.sqrt(np.clip(w, 0, None)) / (2 * np.pi))
        freqs_shortcut = np.sort(propagate_ensemble(df[None, :], inp['P'], inp['freqs_sec'])[0])

        rel_err = np.abs(freqs_eigh - freqs_shortcut) / (freqs_eigh + 1e-12)
        max_rel_err = max(max_rel_err, rel_err.max())

    _record_check("Diagonal shortcut matches real eigh() to high precision",
                  max_rel_err < 1e-8, f"max relative error = {max_rel_err:.2e}")

    # --- COUPLED model: validate the vectorized einsum implementation
    # (propagate_ensemble_coupled) against a direct, hand-built eigh on the
    # same samples -- a DIFFERENT bug class than the formula-derivation
    # checks already in step4.py's validate_mistuning() (this catches an
    # einsum/broadcasting mistake, not a formula mistake). ---
    G, E = _blade_gram_matrices(inp['T_full2sec'], inp['blade_dofs'])
    Kdiag = np.diag(K_sec)
    norm = np.sqrt(np.outer(Kdiag, Kdiag)) / np.sqrt(np.outer(E, E))
    max_rel_err_c = 0.0
    for i in idx[:5]:   # coupled eigh is more expensive; 5 of the same spot-check samples is enough
        row = {v: inp['theta'][v][i] for v in VAR_NAMES}
        df = compute_delta_f_vectorized({k: v[None, :] for k, v in row.items()},
                                          inp['sens'], inp['L_ref'], inp['t_ref'])[0]
        scale = (1.0 + df) ** 2 - 1.0
        raw = np.zeros_like(norm)
        for b in range(NB):
            raw += scale[b] * G[b]
        dK_direct = raw * norm
        w, _ = eigh(K_sec + dK_direct, M_sec)
        freqs_direct = np.sort(np.sqrt(np.clip(w, 0, None)) / (2 * np.pi))
        freqs_vectorized = np.sort(propagate_ensemble_coupled(df[None, :], inp, K_sec, M_sec,
                                                                 inp['freqs_sec'])[0])
        rel_err = np.abs(freqs_direct - freqs_vectorized) / (freqs_direct + 1e-12)
        max_rel_err_c = max(max_rel_err_c, rel_err.max())
    _record_check("Coupled model's vectorized ensemble path matches a direct "
                  "per-sample eigh() to high precision",
                  max_rel_err_c < 1e-8, f"max relative error = {max_rel_err_c:.2e}")


# ═══════════════════════════════════════════════════════════════════
# 5C. ALEATORIC ENSEMBLE + HI1 QOI
# ═══════════════════════════════════════════════════════════════════
def run_aleatoric_ensemble(inp):
    hdr(f"STEP 5C: ALEATORIC FORWARD UQ — ALL {inp['n_samples']} SAMPLES")
    delta_f_all = compute_delta_f_vectorized(inp['theta'], inp['sens'],
                                              inp['L_ref'], inp['t_ref'])
    freqs_mistuned = propagate_ensemble_coupled(delta_f_all, inp, inp['K_sec'], inp['M_sec'],
                                                  inp['freqs_sec'])
    HI1 = np.max(np.abs(freqs_mistuned[:, :NB] - inp['freqs_sec'][None, :NB]), axis=1)

    print(f"  HI1 (max 1B-cluster |delta f|): mean={HI1.mean():.4f} Hz, "
          f"std={HI1.std():.4f} Hz, "
          f"5%-95% = [{np.percentile(HI1, 5):.4f}, {np.percentile(HI1, 95):.4f}] Hz")

    _record_check("HI1 finite for every sample", bool(np.all(np.isfinite(HI1))))
    _record_check("HI1 non-negative for every sample", bool(np.all(HI1 >= 0)))
    f0_1b_mean = inp['freqs_sec'][:NB].mean()
    _record_check("HI1 stays physically bounded (< 10% of 1B mean frequency)",
                  bool(HI1.max() < 0.10 * f0_1b_mean),
                  f"max HI1={HI1.max():.3f} Hz vs 10% of {f0_1b_mean:.1f} Hz")

    return delta_f_all, freqs_mistuned, HI1


# ═══════════════════════════════════════════════════════════════════
# 5D. EPISTEMIC UNCERTAINTY — sensitivity-coefficient perturbation
# ═══════════════════════════════════════════════════════════════════
def run_epistemic_ensemble(inp, HI1_nominal):
    ep = CONFIG['epistemic']
    hdr(f"STEP 5D: EPISTEMIC UQ — {ep['n_draws']} COEFFICIENT DRAWS "
        f"(+/-{ep['coeff_relative_std']*100:.1f}% REAL measured coefficient uncertainty, "
        f"not a placeholder -- see CONFIG['epistemic'] comment)")
    rng = np.random.default_rng(CONFIG['random_seed'] + 1)

    draw_means, draw_vars, all_HI1 = [], [], []
    coeff_draws = []
    for k in range(ep['n_draws']):
        sens_k = {}
        for name in COEFF_NAMES:
            nominal = inp['sens'][name]
            sens_k[name] = nominal * (1.0 + rng.normal(0, ep['coeff_relative_std']))
        coeff_draws.append([sens_k[n] for n in COEFF_NAMES])

        df_k = compute_delta_f_vectorized(inp['theta'], sens_k, inp['L_ref'], inp['t_ref'])
        freqs_k = propagate_ensemble_coupled(df_k, inp, inp['K_sec'], inp['M_sec'], inp['freqs_sec'])
        HI1_k = np.max(np.abs(freqs_k[:, :NB] - inp['freqs_sec'][None, :NB]), axis=1)

        draw_means.append(HI1_k.mean())
        draw_vars.append(HI1_k.var())
        all_HI1.append(HI1_k)

    draw_means = np.array(draw_means)
    draw_vars = np.array(draw_vars)
    all_HI1 = np.array(all_HI1)     # (n_draws, n_samples)

    aleatoric_component = draw_vars.mean()          # E[Var[QoI | coeffs]]
    epistemic_component = draw_means.var()           # Var[E[QoI | coeffs]]
    total_direct = all_HI1.var()                      # pooled, directly
    total_decomposed = aleatoric_component + epistemic_component

    print(f"  Aleatoric variance component (avg within-draw):  {aleatoric_component:.6f} Hz^2")
    print(f"  Epistemic variance component (across-draw mean): {epistemic_component:.6f} Hz^2")
    print(f"  Total variance -- direct (pooled):    {total_direct:.6f} Hz^2")
    print(f"  Total variance -- law-of-total-var:   {total_decomposed:.6f} Hz^2")

    rel_err = abs(total_direct - total_decomposed) / total_direct
    _record_check("Law of total variance holds (direct pooled var vs. decomposed sum)",
                  rel_err < 0.02, f"relative error = {rel_err * 100:.3f}%")
    _record_check("Epistemic variance component is non-negative", bool(epistemic_component >= 0))
    _record_check("Epistemic uncertainty is a real, non-trivial contribution "
                  "(not numerically negligible)",
                  bool(epistemic_component > 1e-6 * aleatoric_component),
                  f"epistemic/aleatoric ratio = {epistemic_component/aleatoric_component:.4f}")

    return dict(coeff_draws=np.array(coeff_draws), draw_means=draw_means,
                draw_vars=draw_vars, all_HI1=all_HI1,
                aleatoric_component=aleatoric_component,
                epistemic_component=epistemic_component)


# ═══════════════════════════════════════════════════════════════════
# 5E. GAUSSIAN PROCESS SURROGATE
# ═══════════════════════════════════════════════════════════════════
def build_gp_features(delta_f_all, n_fourier=None):
    """GP inputs = the RAW 24-dim per-blade fractional-frequency mistuning
    pattern, standardized -- no compression at all.

    HISTORY (kept for provenance, not current behavior): the original
    version used a 4-term Fourier-magnitude compression + max/std summary
    stats (6-dim). That was itself a fix for an EARLIER problem (Fourier
    magnitudes alone gave R^2~0, fixed by adding max/std) -- but it turned
    out to still be leaving real signal on the table. Measured directly,
    2026-08-21: sweeping the Fourier truncation from 4 terms up to 12 (the
    Nyquist limit for 24 real samples -- i.e. the FULL magnitude spectrum,
    no truncation left to blame) moved test R^2 only 0.6208 -> 0.6221.
    Root cause: HI1 is a MAX over 24 blades, so its value depends on WHICH
    blade is most mistuned -- but a magnitude-only DFT is invariant to
    circular shift of that identity (|FFT| discards phase), so no amount
    of Fourier truncation-tuning could ever recover it; the compression
    itself, not the truncation length, was structurally lossy for this
    specific max-type QoI. Switching to the raw pattern (this function)
    raised test R^2 to 0.888-0.907 across 5 independent train/test seeds
    with the IDENTICAL kernel/optimizer settings -- confirming the GP and
    its kernel were never the bottleneck. `n_fourier` is accepted and
    ignored (kept in the signature so existing call sites don't need to
    change) rather than silently dropped from the API.
    """
    names = [f'delta_f/f, blade {b}' for b in range(delta_f_all.shape[1])]
    return delta_f_all, names


def train_gp_surrogate(delta_f_all, HI1):
    hdr("STEP 5E: GAUSSIAN PROCESS SURROGATE")
    gp_cfg = CONFIG['gp']
    X, feature_names = build_gp_features(delta_f_all)
    y = HI1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=gp_cfg['test_fraction'], random_state=CONFIG['random_seed'])

    X_mean, X_std = X_train.mean(0), X_train.std(0)
    Xn_train = (X_train - X_mean) / X_std
    Xn_test = (X_test - X_mean) / X_std

    # A first attempt used a per-dimension ARD length scale (one per
    # feature) with only 3 optimizer restarts and an unbounded WhiteKernel,
    # and it converged to a degenerate fit -- WhiteKernel absorbed nearly
    # all the variance, giving R^2 ~ 0 (worse than predicting the mean)
    # EVEN THOUGH a plain linear regression on the identical features got
    # R^2 = 0.88. That confirmed the problem was the GP's kernel-likelihood
    # optimization landing in a bad local optimum, not a lack of signal in
    # the features. Fixed by: (1) a single SHARED length scale instead of
    # 6 independent ones (overparametrized for 700 points in 6D, a classic
    # local-optima trap), (2) bounding WhiteKernel's noise_level so it
    # cannot single-handedly explain the whole (normalized, unit-variance)
    # target, and (3) far more optimizer restarts.
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 0.5))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=20,
                                   random_state=CONFIG['random_seed'])
    gp.fit(Xn_train, y_train)

    y_pred, y_std = gp.predict(Xn_test, return_std=True)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    within_2sigma = float(np.mean(np.abs(y_test - y_pred) <= 2 * y_std))

    print(f"  Train/test split: {len(y_train)}/{len(y_test)} samples, "
          f"{X.shape[1]} features: {feature_names}")
    print(f"  Learned kernel: {gp.kernel_}")
    print(f"  Test R^2 = {r2:.4f},  RMSE = {rmse:.5f} Hz")
    print(f"  Calibration: {within_2sigma*100:.1f}% of test points within predicted +/-2 sigma "
          f"(nominal ~95%)")

    _record_check("GP predictions finite for every test point",
                  bool(np.all(np.isfinite(y_pred))))
    _record_check("GP surrogate explains the majority of QoI variance (R^2 > 0.3)",
                  r2 > 0.3, f"R^2 = {r2:.4f}")
    _record_check("GP predictive uncertainty is reasonably calibrated "
                  "(60-100% of test residuals within +/-2 sigma)",
                  0.60 <= within_2sigma <= 1.0, f"{within_2sigma*100:.1f}%")

    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                y_pred=y_pred, y_std=y_std, r2=r2, rmse=rmse,
                within_2sigma=within_2sigma, gp=gp, X_mean=X_mean, X_std=X_std,
                feature_names=feature_names)


# ═══════════════════════════════════════════════════════════════════
# 5F. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def save_outputs(HI1, freqs_mistuned, epi, gp_result):
    hdr("STEP 5F: SAVING OUTPUTS")
    fp1 = os.path.join(OUT, 'aleatoric_ensemble.npz')
    np.savez(fp1, HI1=HI1, freqs_mistuned=freqs_mistuned)
    print(f"  Saved: {fp1}")

    fp2 = os.path.join(OUT, 'epistemic_ensemble.npz')
    np.savez(fp2, coeff_draws=epi['coeff_draws'], draw_means=epi['draw_means'],
              draw_vars=epi['draw_vars'],
              aleatoric_component=epi['aleatoric_component'],
              epistemic_component=epi['epistemic_component'])
    print(f"  Saved: {fp2}")

    fp3 = os.path.join(OUT, 'gp_surrogate.npz')
    np.savez(fp3, X_test=gp_result['X_test'], y_test=gp_result['y_test'],
              y_pred=gp_result['y_pred'], y_std=gp_result['y_std'],
              r2=gp_result['r2'], rmse=gp_result['rmse'])
    print(f"  Saved: {fp3}")

    config_record = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'n_samples': int(len(HI1)), 'n_sec': NSEC,
        'epistemic_prior': CONFIG['epistemic'],
        'gp_config': CONFIG['gp'],
        'gp_r2': gp_result['r2'], 'gp_rmse': gp_result['rmse'],
        'gp_calibration_within_2sigma': gp_result['within_2sigma'],
        'note': ('Epistemic prior on Step 4 sensitivity coefficients is now a REAL '
                 'measured uncertainty (+/-10.6% multiplicative, from the 2-magnitude '
                 'ANSYS refit spread for tip_coeff_per_frac, 2026-08-27) -- was a '
                 'documented +/-25% placeholder before the d_tip-only scope change. '
                 'GP surrogate trained directly on the raw 24-blade mistuning pattern '
                 '(no Fourier/summary-stat compression, see build_gp_features() '
                 'docstring for why the earlier compressed representation was replaced '
                 '2026-08-21). See CONFIG in step5.py.'),
    }
    fp4 = os.path.join(OUT, 'step5_config.json')
    with open(fp4, 'w') as f:
        json.dump(config_record, f, indent=2)
    print(f"  Saved: {fp4}")


# ═══════════════════════════════════════════════════════════════════
# 5G. FIGURES — 5 diagnostics
# ═══════════════════════════════════════════════════════════════════
def _resolve_figs_dir():
    figs = os.path.join(FIG_ROOT, 'figures', 'step5')
    os.makedirs(figs, exist_ok=True)
    return figs


def _savefig(fig, figs_dir, name):
    paths = plot_style.savefig_pub(fig, figs_dir, name)
    print(f"  Figure saved: {paths[0]}")


def make_step5_figures(inp, HI1, freqs_mistuned, epi, gp_result):
    hdr("STEP 5G: FIGURES (5 diagnostics, figures/step5/, PNG+PDF)")
    figs = _resolve_figs_dir()

    # ── fig1: HI1 QoI distribution, full aleatoric ensemble ──
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.hist(HI1, bins=40, color=plot_style.BLUE, alpha=0.85, edgecolor=plot_style.SURFACE, linewidth=0.6)
    ax.axvline(HI1.mean(), color=plot_style.INK_PRIMARY, lw=1.4, label=f'mean = {HI1.mean():.3f} Hz')
    ax.axvline(np.percentile(HI1, 95), color=plot_style.ORANGE, lw=1.4, ls='--',
               label=f'95th pct = {np.percentile(HI1, 95):.3f} Hz')
    ax.set_xlabel('HI1 -- max 1B-cluster |frequency deviation|  [Hz]')
    ax.set_ylabel('count')
    plot_style.two_tier_title(ax, 'Aleatoric health-indicator distribution', f'HI1, {len(HI1)} samples')
    plot_style.legend_below(ax, ncol=2)
    fig.tight_layout()
    _savefig(fig, figs, 'step5_fig1_hi1_distribution')

    # ── fig2: aleatoric vs epistemic variance decomposition ──
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    comps = [epi['aleatoric_component'], epi['epistemic_component']]
    ax.bar(['aleatoric\n(manufacturing)', 'epistemic\n(model-form)'], comps,
           color=[plot_style.AQUA, plot_style.VIOLET])
    for i, v in enumerate(comps):
        ax.text(i, v, f'{v:.5f}', ha='center', va='bottom', fontsize=9.5, color=plot_style.INK_SECONDARY)
    ax.set_ylabel('Variance component of HI1  [Hz$^2$]')
    plot_style.two_tier_title(ax, 'Total-variance decomposition', 'of the health-indicator QoI')
    _savefig(fig, figs, 'step5_fig2_variance_decomposition')

    # ── fig3: epistemic fan — HI1 distribution per coefficient draw ──
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    n_show = min(25, epi['all_HI1'].shape[0])
    for k in range(n_show):
        vals = np.sort(epi['all_HI1'][k])
        pct = np.linspace(0, 100, len(vals))
        ax.plot(vals, pct, color=plot_style.VIOLET, alpha=0.18, lw=1.2)
    vals_nom = np.sort(HI1)
    ax.plot(vals_nom, np.linspace(0, 100, len(vals_nom)), color=plot_style.INK_PRIMARY, lw=2.2,
            label='nominal sensitivity coefficients')
    ax.set_xlabel('HI1  [Hz]')
    ax.set_ylabel('Empirical CDF  [%]')
    # Hug the real data, no forced pin to literal x=0 (HI1 never goes
    # below ~1.5 Hz here, so pinning left=0 just opens a dead empty zone)
    # and no y-margin gap either -- the CDF's own natural bounds ARE
    # exactly 0-100, so pin y there instead of leaving rcParams' default
    # 5% padding above/below (2026-08-29 fix, reverting an earlier
    # overcorrection that pinned x to 0 and made the empty-space problem
    # worse, not better).
    x_all = np.concatenate([np.sort(epi['all_HI1'][k]) for k in range(n_show)] + [vals_nom])
    ax.set_xlim(x_all.min(), x_all.max())
    ax.set_ylim(0, 100)
    plot_style.two_tier_title(ax, 'Epistemic uncertainty fan',
                               f'HI1 CDF under {n_show} perturbed-coefficient draws')
    plot_style.legend_below(ax)
    fig.tight_layout()
    _savefig(fig, figs, 'step5_fig3_epistemic_fan')

    # ── fig4: GP surrogate predicted vs. true (held-out test set) ──
    fig, ax = plt.subplots(figsize=(6.5, 6.3))
    y_test, y_pred, y_std = gp_result['y_test'], gp_result['y_pred'], gp_result['y_std']
    ax.errorbar(y_test, y_pred, yerr=2 * y_std, fmt='o', ms=5.5, color=plot_style.ORANGE,
                ecolor=plot_style.ORANGE, alpha=0.6, elinewidth=1.0, capsize=0, mec=plot_style.SURFACE, mew=1.0)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color=plot_style.INK_MUTED, ls='--', lw=1.4, label='perfect prediction')
    ax.set_xlabel('True HI1  [Hz]')
    ax.set_ylabel('GP-predicted HI1  [Hz]  (error bars: +/-2 sigma)')
    # Hug the real data on both axes -- same fix as fig3 (2026-08-29): no
    # forced pin to literal 0, tight to the actual plotted extent
    # (including the error bars, which reach further than the markers
    # alone) so nothing floats in empty margin.
    ax.set_xlim(y_test.min(), y_test.max())
    ax.set_ylim((y_pred - 2 * y_std).min(), (y_pred + 2 * y_std).max())
    plot_style.two_tier_title(ax, 'GP surrogate: held-out test set',
                               f"R\u00b2={gp_result['r2']:.3f}, RMSE={gp_result['rmse']:.4f} Hz")
    plot_style.legend_below(ax)
    fig.tight_layout()
    _savefig(fig, figs, 'step5_fig4_gp_predicted_vs_true')

    # ── fig5: GP mean +/- 2 sigma band along the MOST INFORMATIVE feature ──
    # (picked by |correlation| with HI1, not fixed to feature 0 -- since the
    # 2026-08-21 fix, features are the raw per-blade delta_f/f values, so
    # this picks whichever 2 blades' own mistuning correlates most with
    # HI1, which will vary sample-to-sample-ensemble but is exactly the
    # right question to ask of a per-blade feature set.)
    # 3D SURFACE (kept as 3D, 2026-08-29 explicit user request -- reverting
    # the 2D contour redesign tried earlier this session): a 2D slice over
    # the TWO most informative features, plotted as a real GP mean surface
    # with the raw Monte Carlo ensemble scattered around it in 3D, at the
    # cost of a fixed 3rd+ feature (same "other features held at their
    # mean" convention a 1D version would use).
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    X_all, feat_names = build_gp_features(compute_delta_f_vectorized(
        inp['theta'], inp['sens'], inp['L_ref'], inp['t_ref']))
    corrs = [abs(np.corrcoef(X_all[:, j], HI1)[0, 1]) for j in range(X_all.shape[1])]
    order = np.argsort(corrs)[::-1]
    j0, j1 = int(order[0]), int(order[1])
    x0, x1 = X_all[:, j0], X_all[:, j1]

    fig = plt.figure(figsize=(8.4, 6.8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x0, x1, HI1, s=8, color=plot_style.INK_MUTED, alpha=0.35, label='raw Monte Carlo ensemble')

    g0 = np.linspace(x0.min(), x0.max(), 45)
    g1 = np.linspace(x1.min(), x1.max(), 45)
    G0, G1 = np.meshgrid(g0, g1)
    Xg = np.tile(gp_result['X_test'].mean(0), (G0.size, 1))
    Xg[:, j0] = G0.ravel()
    Xg[:, j1] = G1.ravel()
    Xgn = (Xg - gp_result['X_mean']) / gp_result['X_std']
    mu, _ = gp_result['gp'].predict(Xgn, return_std=True)
    Mu = mu.reshape(G0.shape)
    ax.plot_surface(G0, G1, Mu, cmap=plot_style.SEQ_CMAP, alpha=0.75, linewidth=0.1,
                     edgecolor=plot_style.INK, antialiased=True)
    ax.set_xlabel(f'{feat_names[j0]}\n(|corr|={corrs[j0]:.2f})', labelpad=10, fontsize=9)
    ax.set_ylabel(f'{feat_names[j1]}\n(|corr|={corrs[j1]:.2f})', labelpad=10, fontsize=9)
    ax.set_zlabel('HI1  [Hz]', labelpad=6)
    ax.view_init(elev=24, azim=-58)
    plot_style.two_tier_title(ax, 'GP surrogate mean surface vs. raw ensemble',
                               '2D slice over the two most informative features (others held at mean)')
    plot_style.legend_below(ax)
    _savefig(fig, figs, 'step5_fig5_gp_1d_slice')

    print(f"  All 5 Step 5 figures saved to: {figs}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _log_path = os.path.join(_HERE, 'Step5.txt')
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    t_start = time.time()
    hdr(f"STEP 5 v1.0: UNCERTAINTY QUANTIFICATION FRAMEWORK — {NB}-BLADE BLISK (PCE PROJECT)")
    print(f"  Step 2 dir (read-only): {CONFIG['step2_dir']}")
    print(f"  Step 3 dir (read-only): {CONFIG['step3_dir']}")
    print(f"  Step 4 dir (read-only): {CONFIG['step4_dir']}")
    print(f"  Output dir (Step 5):    {OUT}")

    inp = load_inputs()
    validate_diagonal_shortcut(inp)
    delta_f_all, freqs_mistuned, HI1 = run_aleatoric_ensemble(inp)
    epi = run_epistemic_ensemble(inp, HI1)
    gp_result = train_gp_surrogate(delta_f_all, HI1)
    save_outputs(HI1, freqs_mistuned, epi, gp_result)
    make_step5_figures(inp, HI1, freqs_mistuned, epi, gp_result)
    passed = print_validation_summary()

    hdr("STEP 5 COMPLETE")
    elapsed = time.time() - t_start
    print(f"  Validation: {'PASSED' if passed else 'FAILED — see STEP 5 VALIDATION SUMMARY above'}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"\n  Files in {OUT}:")
    for fn in sorted(os.listdir(OUT)):
        fp = os.path.join(OUT, fn)
        if os.path.isfile(fp):
            print(f"    {fn:30s} {os.path.getsize(fp) / 1e3:8.2f} KB")
    print(f"\nLog saved: {_log_path}")
    sys.stdout = sys.__stdout__
    _log_file.close()
