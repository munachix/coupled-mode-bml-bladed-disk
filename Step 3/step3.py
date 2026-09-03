"""
STEP 3 v1.0: Geometric Mistuning Parameterization — 24-Blade Blisk (PCE Project)
==================================================================================

Implements PHASE 3 of the roadmap: per-blade geometric mistuning variables,
sampled as a spatially-correlated Gaussian random field over the cyclic ring
of blades (Karhunen-Loeve expansion), NOT a bare stiffness-matrix perturbation.

Five per-blade variables (matching the roadmap MD's Phase 3 list):
    d_length     — blade length deviation from nominal        [mm]
    d_thickness  — blade thickness deviation from nominal     [mm]
    d_le_te      — leading/trailing-edge profile deviation    [mm]
    d_twist_deg  — twist angle deviation from nominal         [deg]
    d_tip        — tip geometry deviation from nominal        [mm]

All five are DEVIATIONS about zero (Step 1's extracted model is the nominal/
tuned baseline) — this step does not touch K/M or the ROM; it only produces
the random geometric parameter vectors theta that Phase 4 (nonlinear ROM)
will later map onto stiffness/mass perturbations.

Spatial correlation model
--------------------------
The 24 blades sit on a cyclic ring, so blade-to-blade correlation should only
depend on angular separation: a CIRCULANT covariance matrix. A circulant
matrix's eigenbasis is exactly the discrete Fourier / nodal-diameter harmonic
basis for this ring — the closed-form Karhunen-Loeve expansion of a cyclic
random field (standard in bladed-disk mistuning literature). All 24 real KL
coefficients are retained (no arbitrary truncation — dimensionality is fixed
by blade count), and a single `correlation_length_blades` knob per variable
controls how quickly the kernel decays with blade-index distance, so high
nodal diameters get small eigenvalues automatically rather than being cut off
by hand.

IMPORTANT — tolerance magnitudes are PLACEHOLDERS, not measured data
----------------------------------------------------------------------
Step 1's blade_geometry.json only recorded two GLOBAL scalars for the whole
(nominally tuned) blisk (outer_radius_mm, tip_z_extent_mm) — there is no real
per-blade manufacturing/inspection data anywhere in this project. The std-dev
values in CONFIG['tolerances'] below are generic literature-typical aerospace
blade manufacturing tolerances, clearly flagged here so they are trivial to
replace with real spec/drawing tolerances later.

Outputs (to CONFIG['output_dir'], LOCAL to Step 3 — not F:\\ANSYS PCE\\ROM_data,
which is ANSYS/Step-1/2 territory and already has leftover clutter from a
prior iteration):
    theta_samples.npz   — d_length, d_thickness, d_le_te, d_twist_deg, d_tip
                           each shape (n_samples, n_blades); blade_angles_deg
    mistuning_config.json — full provenance: tolerances, correlation lengths,
                           baseline geometry pulled from Step 1, seed, count

Author: PCE-Bayesian Framework — v1.0 (24-blade)
"""

import numpy as np, os, json, time, sys
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import plot_style   # noqa: E402  (shared publication style, see PCE project/plot_style.py)
plot_style.apply_style()

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — edit paths / tolerances if real data becomes available
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    'input_dir':   r'F:\ANSYS PCE\ROM_data',           # Step 1 output (read-only)
    'output_dir':  os.path.join(_HERE, 'output'),       # Step 3's own output (local)

    'n_blades':    24,
    'n_samples':   1000,
    'random_seed': 42,

    # Fallback nominal geometry, used only if blade_geometry.json is missing
    # (values match Step 1's own printed log for this model).
    'fallback_outer_radius_mm': 302.93,
    'fallback_tip_z_extent_mm': 52.00,

    # ------------------------------------------------------------------
    # TOLERANCES.
    # 2026-08-27 SCOPE CHANGE (explicit user decision): mistuning is now
    # d_tip ONLY -- d_length/d_thickness/d_le_te/d_twist_deg are dropped,
    # not carried as unused dead code. Rationale: d_tip is the one
    # variable with a REAL ANSYS-measured sensitivity coefficient tied to
    # this project's real Green-Lagrange nonlinear calibration (Step 4),
    # and the one whose tolerance magnitude can be anchored to a REAL
    # measured dimension of this exact part (Step 1's
    # measure_per_blade_tip_geometry(), blade_tip_geometry.json) rather
    # than an assumption disconnected from the actual geometry. The other
    # 4 variables' coefficients were also real ANSYS measurements (Section
    # 9a of PROJECT_STATUS.md), but this is a deliberate scope narrowing,
    # not a claim they were wrong.
    #
    # 'std_frac_of_thickness' is still a DOCUMENTED, generic-aerospace-
    # practice FRACTION (not measured) -- what changed is the DENOMINATOR
    # it's applied to: Step 1's real per-blade tip z-extent measurement
    # (blade_geometry.json's tip_z_extent_mm, itself corrected 2026-08-27
    # from a contaminated 52.0mm whole-model value to the real, radially-
    # filtered per-blade value of 36.43mm -- see measure_blade_geometry's
    # own comment in step1.py). The real, measured cross-blade geometric
    # variation in this idealized CAD is ~0 (confirmed via
    # blade_tip_geometry.json, CV~4e-11) -- there is no physical
    # inspection data for an unbuilt design study, so the SPREAD here
    # necessarily remains an assumed statistical model, now anchored to a
    # real dimension of this specific part rather than a disconnected
    # generic number.
    # 'correlation_length_blades' is in units of blade-to-blade spacing
    # (how many neighbouring blades stay strongly correlated).
    # ------------------------------------------------------------------
    'tolerances': {
        'd_tip':       {'std_frac_of_thickness': 0.02,   'correlation_length_blades': 2.0},
    },
}

NB = CONFIG['n_blades']
OUT = CONFIG['output_dir']
os.makedirs(OUT, exist_ok=True)

# Figures follow the same <fig_root>/figures/step{N}/ convention referenced
# in step1.py/step2.py. FIG_ROOT is the PCE project root (parent of Step 3/).
FIG_ROOT = os.path.dirname(_HERE)

# 2026-08-27: reduced to d_tip only, see CONFIG['tolerances'] comment.
VAR_NAMES = ['d_tip']
VAR_UNITS = {'d_tip': 'mm'}
VAR_LABELS = {
    'd_tip':       'Blade tip',
}

_VALIDATION_LOG = []   # list of (name, bool_passed, detail_str)


class _Tee:
    """Duplicates writes to multiple streams (console + a UTF-8 log file),
    so every run auto-saves its own console output without manual redirection.
    Each stream's encoding is handled independently: if a stream can't encode
    a character (e.g. a non-UTF-8 Windows console codepage choking on '→'),
    that ONE stream falls back to a replaced-character write instead of
    crashing the whole run — the other streams (e.g. the UTF-8 log file)
    still get the full-fidelity text."""
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


def _record_check(name, ok, detail=''):
    _VALIDATION_LOG.append((name, bool(ok), detail))
    status = 'OK' if ok else 'FAIL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    return ok


def hdr(t):
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")


# ═══════════════════════════════════════════════════════════════════
# 3A. LOAD NOMINAL BASELINE GEOMETRY (from Step 1, read-only)
# ═══════════════════════════════════════════════════════════════════
def load_baseline_geometry():
    hdr("STEP 3A: LOADING BASELINE GEOMETRY (from Step 1)")
    fp = os.path.join(CONFIG['input_dir'], 'blade_geometry.json')
    if os.path.exists(fp):
        with open(fp) as f:
            geo = json.load(f)
        r_out = geo.get('outer_radius_mm', CONFIG['fallback_outer_radius_mm'])
        t_tip = geo.get('tip_z_extent_mm', CONFIG['fallback_tip_z_extent_mm'])
        print(f"  Loaded blade_geometry.json from {CONFIG['input_dir']}")
    else:
        r_out = CONFIG['fallback_outer_radius_mm']
        t_tip = CONFIG['fallback_tip_z_extent_mm']
        print(f"  blade_geometry.json not found — using fallback values "
              f"(outer_radius={r_out} mm, tip_z_extent={t_tip} mm)")
    print(f"  outer_radius_mm   = {r_out:.2f}")
    print(f"  tip_z_extent_mm   = {t_tip:.2f}  (thickness proxy)")
    return {'outer_radius_mm': r_out, 'tip_z_extent_mm': t_tip}


# ═══════════════════════════════════════════════════════════════════
# 3B. RESOLVE TOLERANCES TO PHYSICAL UNITS
# ═══════════════════════════════════════════════════════════════════
def resolve_tolerances(baseline):
    hdr("STEP 3B: RESOLVING TOLERANCES TO PHYSICAL UNITS")
    r_out, t_tip = baseline['outer_radius_mm'], baseline['tip_z_extent_mm']
    tol = CONFIG['tolerances']
    resolved = {}

    resolved['d_tip'] = {
        'std': tol['d_tip']['std_frac_of_thickness'] * t_tip,
        'unit': 'mm',
        'correlation_length_blades': tol['d_tip']['correlation_length_blades'],
    }

    for name, r in resolved.items():
        print(f"  {name:14s}  std = {r['std']:.5f} {r['unit']:4s}  "
              f"corr_len = {r['correlation_length_blades']:.1f} blades")
    return resolved


# ═══════════════════════════════════════════════════════════════════
# 3C. CIRCULANT KARHUNEN-LOEVE BASIS (exact, per variable)
# ═══════════════════════════════════════════════════════════════════
def circulant_kl_basis(std, correlation_length_blades, n_blades=NB):
    """Exact KL expansion of a circulant (cyclic) squared-exponential random
    field over `n_blades` equally-spaced positions. Returns (eigvecs, eigvals)
    of the n_blades x n_blades covariance matrix C, with C[i,j] = std^2 *
    exp(-0.5 * (circular_dist(i,j) / correlation_length_blades)^2).
    Eigendecomposition of a real symmetric circulant matrix IS the closed-form
    KLE for this cyclic domain — no truncation: all n_blades modes kept."""
    idx = np.arange(n_blades)
    delta = np.abs(idx[:, None] - idx[None, :])
    circ_dist = np.minimum(delta, n_blades - delta)          # shortest way round the ring
    C = (std ** 2) * np.exp(-0.5 * (circ_dist / correlation_length_blades) ** 2)
    eigvals, eigvecs = np.linalg.eigh(C)                      # exact, symmetric, tiny (24x24)
    eigvals = np.clip(eigvals, 0.0, None)                     # guard tiny negative numerical noise
    return eigvecs, eigvals


def sample_field(eigvecs, eigvals, n_samples, rng):
    """Draw n_samples spatially-correlated realizations from the KL basis.
    Returns array (n_samples, n_blades) in the same physical units as std."""
    sqrt_lam = np.sqrt(eigvals)
    transform = eigvecs * sqrt_lam[None, :]                  # (n_blades, n_blades)
    xi = rng.standard_normal((n_samples, len(eigvals)))       # iid N(0,1)
    return xi @ transform.T                                   # (n_samples, n_blades)


# ═══════════════════════════════════════════════════════════════════
# 3D. GENERATE MISTUNING SAMPLES
# ═══════════════════════════════════════════════════════════════════
def generate_mistuning_samples(resolved_tol):
    hdr(f"STEP 3D: GENERATING {CONFIG['n_samples']} MISTUNING REALIZATIONS")
    rng = np.random.default_rng(CONFIG['random_seed'])
    samples = {}
    for name, r in resolved_tol.items():
        eigvecs, eigvals = circulant_kl_basis(r['std'], r['correlation_length_blades'])
        samples[name] = sample_field(eigvecs, eigvals, CONFIG['n_samples'], rng)
        realized_std = samples[name].std()
        print(f"  {name:14s}  target std = {r['std']:.5f} {r['unit']:4s}  "
              f"realized std = {realized_std:.5f} {r['unit']:4s}")
    samples['blade_angles_deg'] = (360.0 / NB) * np.arange(NB)
    return samples


# ═══════════════════════════════════════════════════════════════════
# 3E. VALIDATION — statistical sanity checks against the design targets
# ═══════════════════════════════════════════════════════════════════
def sample_covariance(X):
    """(n_blades, n_blades) sample covariance of X (n_samples, n_blades)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def circular_distance_matrix(n_blades=NB):
    idx = np.arange(n_blades)
    delta = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(delta, n_blades - delta)


def validate_samples(samples, resolved_tol):
    hdr("STEP 3E: VALIDATION — STATISTICAL CHECKS AGAINST DESIGN TARGETS")
    circ_dist = circular_distance_matrix()

    for name, r in resolved_tol.items():
        X = samples[name]                                    # (n_samples, n_blades)
        target_std = r['std']

        # Theoretical circulant covariance (needed below for both the mean
        # check's standard error and the covariance-shape check).
        eigvecs, eigvals = circulant_kl_basis(target_std, r['correlation_length_blades'])
        C_theory = (eigvecs * eigvals[None, :]) @ eigvecs.T

        # Mean should be ~0 (these are deviations about the nominal baseline).
        # NOTE: a flat "X% of std" threshold is statistically wrong here --
        # blade-to-blade correlation (correlation_length_blades) shrinks the
        # EFFECTIVE sample size of the pooled (n_samples x n_blades) mean well
        # below n_samples*n_blades, so the true standard error must come from
        # the covariance itself, not just the marginal std. For one row
        # (one 24-blade draw), Var[row_mean] = (1/NB^2) * sum(C_theory); the
        # pooled mean over n_samples independent rows then has variance
        # row_var / n_samples. Gate at 5 sigma (~3e-7 two-sided false-positive
        # rate) so this only trips on a real bias, not sampling noise.
        realized_mean = float(X.mean())
        row_var = float(C_theory.sum()) / (NB ** 2)
        se_mean = np.sqrt(row_var / CONFIG['n_samples'])
        _record_check(f"{name}: realized mean ~ 0 (within 5 sigma of standard error)",
                       abs(realized_mean) < 5 * se_mean,
                       f"mean={realized_mean:.5f} {r['unit']}, "
                       f"SE={se_mean:.5f}, |mean|/SE={abs(realized_mean) / se_mean:.2f}")

        # Realized std should match the configured tolerance within 10%
        # (finite-sample Monte Carlo noise at n_samples=1000 is a few percent)
        realized_std = float(X.std())
        rel_err = abs(realized_std - target_std) / target_std
        _record_check(f"{name}: realized std matches target (within 10%)",
                       rel_err < 0.10,
                       f"target={target_std:.5f}, realized={realized_std:.5f} "
                       f"{r['unit']} ({rel_err * 100:.2f}% off)")
        cov = sample_covariance(X)
        theory_by_k = np.array([C_theory[circ_dist == k].mean() for k in range(NB // 2 + 1)])
        sample_by_k = np.array([cov[circ_dist == k].mean() for k in range(NB // 2 + 1)])
        max_abs_err = float(np.max(np.abs(sample_by_k - theory_by_k)))
        _record_check(f"{name}: sample covariance matches circulant theory by blade-distance",
                       max_abs_err < 0.15 * target_std ** 2,
                       f"max|sample-theory| = {max_abs_err:.6f} vs. tol "
                       f"{0.15 * target_std ** 2:.6f} (15% of variance)")
        print(f"    cov(k=0..4) theory = " +
              ", ".join(f"{v:.5f}" for v in theory_by_k[:5]))
        print(f"    cov(k=0..4) sample = " +
              ", ".join(f"{v:.5f}" for v in sample_by_k[:5]))

        # Covariance matrix must be symmetric PSD (guards the KL construction itself)
        sym_err = float(np.max(np.abs(C_theory - C_theory.T)))
        _record_check(f"{name}: theoretical covariance symmetric",
                       sym_err < 1e-9, f"max asymmetry = {sym_err:.2e}")
        _record_check(f"{name}: theoretical covariance PSD (eigvals >= 0)",
                       bool(np.all(eigvals >= -1e-9)), f"min eigval = {eigvals.min():.3e}")

    # Shape / dtype sanity
    for name in resolved_tol:
        ok = samples[name].shape == (CONFIG['n_samples'], NB)
        _record_check(f"{name}: output shape is (n_samples, n_blades)", ok,
                       f"shape={samples[name].shape}")
    _record_check("blade_angles_deg shape is (n_blades,)",
                   samples['blade_angles_deg'].shape == (NB,),
                   f"shape={samples['blade_angles_deg'].shape}")


def print_validation_summary():
    hdr("STEP 3 VALIDATION SUMMARY")
    for name, ok, detail in _VALIDATION_LOG:
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    n_fail = sum(1 for _, ok, _ in _VALIDATION_LOG if not ok)
    hdr(f"STEP 3 VALIDATION: {'PASSED' if n_fail == 0 else f'FAILED ({n_fail} check(s))'}")
    return n_fail == 0


# ═══════════════════════════════════════════════════════════════════
# 3F. SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def save_outputs(samples, resolved_tol, baseline):
    hdr("STEP 3F: SAVING OUTPUTS")
    fp_npz = os.path.join(OUT, 'theta_samples.npz')
    np.savez(fp_npz, **samples)
    print(f"  Saved: {fp_npz}")

    config_record = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'n_blades': NB,
        'n_samples': CONFIG['n_samples'],
        'random_seed': CONFIG['random_seed'],
        'baseline_geometry': baseline,
        'tolerances_resolved': resolved_tol,
        'note': ('Tolerance magnitudes are generic literature-typical placeholders, '
                 'not measured manufacturing/inspection data. See CONFIG in step3.py.'),
    }
    fp_json = os.path.join(OUT, 'mistuning_config.json')
    with open(fp_json, 'w') as f:
        json.dump(config_record, f, indent=2)
    print(f"  Saved: {fp_json}")


# ═══════════════════════════════════════════════════════════════════
# 3G. FIGURES — 5 diagnostic plots of the mistuning parameterization
# ═══════════════════════════════════════════════════════════════════
def _resolve_figs_dir():
    figs = os.path.join(FIG_ROOT, 'figures', 'step3')
    os.makedirs(figs, exist_ok=True)
    return figs


def _savefig(fig, figs_dir, name):
    paths = plot_style.savefig_pub(fig, figs_dir, name)
    print(f"  Figure saved: {paths[0]}")


def make_step3_figures(samples, resolved_tol):
    hdr("STEP 3G: FIGURES (5 diagnostics, figures/step3/, PNG+PDF)")
    figs = _resolve_figs_dir()
    circ_dist = circular_distance_matrix()
    colors = dict(zip(VAR_NAMES, plot_style.CATEGORICAL[:5]))

    # ── fig1: correlation-vs-blade-distance decay ──
    # NOTE: the normalized decay curve C(k)/C(0) depends ONLY on
    # correlation_length_blades, not on std -- so variables sharing the same
    # L (d_length & d_thickness both L=3; d_le_te & d_tip both L=2) produce
    # mathematically IDENTICAL curves. Plotting one line per variable made
    # two of the five lines invisible (silently overdrawn by whichever line
    # was plotted later at the same L). Fixed by grouping: one line per
    # UNIQUE L, with the legend listing every variable that shares it.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ks = np.arange(NB // 2 + 1)
    by_L = {}
    for name, r in resolved_tol.items():
        by_L.setdefault(r['correlation_length_blades'], []).append(name)
    for (L, names_here), c in zip(sorted(by_L.items()), plot_style.CATEGORICAL):
        eigvecs, eigvals = circulant_kl_basis(1.0, L)   # std=1: shape depends only on L
        C_theory = (eigvecs * eigvals[None, :]) @ eigvecs.T
        by_k = np.array([C_theory[circ_dist == k].mean() for k in ks]) / C_theory[0, 0]
        ax.plot(ks, by_k, marker='o', color=c,
                label=f"L={L:.0f}   ({', '.join(VAR_LABELS[n] for n in names_here)})")
    ax.set_xlabel('Circular blade distance, $k$')
    ax.set_ylabel('Normalized correlation  $C(k)/C(0)$')
    plot_style.two_tier_title(ax, 'Mistuning spatial correlation',
                               'decay vs. blade distance, by KL correlation length')
    plot_style.legend_below(ax, title='correlation length (blades)')
    fig.tight_layout()
    _savefig(fig, figs, 'step3_fig1_correlation_decay')

    # ── fig2: marginal histograms, one standalone PNG per variable ──
    # SPLIT (2026-08-19, explicit user request): was one 2x3 grid (6th panel
    # always blank -- only 5 variables) -- now 5 standalone PNGs (fig2a-e).
    panel_letters2 = ['a', 'b', 'c', 'd', 'e']
    for name, letter in zip(VAR_NAMES, panel_letters2):
        X = samples[name].ravel()
        r = resolved_tol[name]
        mu, sd = float(X.mean()), float(X.std())
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        ax.hist(X, bins=40, density=True, color=colors[name], alpha=0.8, edgecolor='white', linewidth=0.4)
        xs = np.linspace(X.min(), X.max(), 200)
        pdf = np.exp(-0.5 * (xs / r['std']) ** 2) / (r['std'] * np.sqrt(2 * np.pi))
        ax.plot(xs, pdf, color=plot_style.TRUTH_COLOR, ls='--', lw=1.3, label='target $N(0,\\sigma)$')
        ax.axvline(mu, color=plot_style.TRUTH_COLOR, lw=1.0, alpha=0.7)
        ax.set_xlabel(f"{VAR_LABELS[name]}  [{r['unit']}]")
        ax.set_ylabel('probability density')
        plot_style.two_tier_title(ax, 'Realized mistuning distribution',
                                   f"{VAR_LABELS[name]}: mu={mu:.4f}, sigma={sd:.4f} {r['unit']} "
                                   f"(pooled 1000 samples x 24 blades, vs. target Gaussian)")
        ax.legend(fontsize=8.5, frameon=False)
        fig.tight_layout()
        _savefig(fig, figs, f'step3_fig2{letter}_histogram_{name}')
    print(f"  step3_fig2a-e: marginal histograms, {', '.join(VAR_NAMES)}")

    # ── fig3: one example realization, as a ring pattern around the disk ──
    # A polar layout matches the actual cyclic geometry (24 blades on a
    # ring) far better than a linear blade-index axis: the dashed circle is
    # the nominal (zero-deviation) baseline, and the solid trace shows how
    # far each blade departs from it, radially, at its true angular position.
    # SPLIT (2026-08-19, explicit user request): was one 2x3 polar grid
    # (6th panel always blank) -- now 5 standalone polar PNGs (fig3a-e).
    theta = np.radians(samples['blade_angles_deg'])
    theta_c = np.append(theta, theta[0])
    sample_row = 0
    panel_letters3 = ['a', 'b', 'c', 'd', 'e']
    for name, letter in zip(VAR_NAMES, panel_letters3):
        r = resolved_tol[name]
        offset = 4.0 * r['std']                     # keeps the radius positive
        vals = samples[name][sample_row]
        vals_c = np.append(vals, vals[0])
        fig, ax = plt.subplots(figsize=(6.8, 6.4), subplot_kw={'projection': 'polar'})
        ax.plot(theta_c, offset + vals_c, marker='o', ms=3.5, mew=0.6, color=colors[name])
        ax.plot(theta_c, np.full_like(theta_c, offset), color=plot_style.TRUTH_COLOR,
                ls='--', lw=1.0, alpha=0.6, label='nominal (zero deviation)')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        fig.suptitle(f"Example mistuning realization -- {VAR_LABELS[name]} [{r['unit']}]",
                     fontsize=12, fontweight='bold', color=plot_style.INK, x=0.02, ha='left', y=0.98)
        fig.text(0.02, 0.935, f"sample #{sample_row}, radial deviation from nominal around the 24-blade ring",
                  fontsize=9.5, color=plot_style.INK_SECONDARY, ha='left')
        ax.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(1.28, 1.12), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        _savefig(fig, figs, f'step3_fig3{letter}_realization_{name}')
    print(f"  step3_fig3a-e: example realization (polar), {', '.join(VAR_NAMES)}")

    # ── fig4: KL eigenvalue (variance) spectrum per variable ──
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    for name, r in resolved_tol.items():
        _, eigvals = circulant_kl_basis(r['std'], r['correlation_length_blades'])
        spec = np.sort(eigvals)[::-1]
        spec = np.clip(spec, 1e-12, None)
        ax.semilogy(np.arange(NB), spec, marker='o',
                    color=colors[name], label=f"{VAR_LABELS[name]} (std={r['std']:.3f}, L={r['correlation_length_blades']:.0f})")
    ax.set_xlabel('KL mode index (sorted by variance)')
    ax.set_ylabel('Eigenvalue (variance), log scale')
    plot_style.two_tier_title(ax, 'Karhunen-Loeve eigenvalue spectrum',
                               'all 24 modes retained, no truncation -- curve shape set by L, level by std$^2$')
    plot_style.legend_below(ax, ncol=2, y=-0.22)
    fig.tight_layout()
    _savefig(fig, figs, 'step3_fig4_kl_spectrum')

    # ── fig5: covariance matrix, one standalone 2D heatmap per variable ──
    # 2D HEATMAP (2026-08-29, explicit user request: no 3D plots in this
    # project). Each panel keeps its own independent color scale (d_le_te
    # and d_twist_deg have ~40x smaller variance than d_tip, so a shared
    # scale would flatten their panels to near-blank). The circulant
    # structure (constant-value diagonal bands, by construction of a
    # cyclic-symmetric covariance) reads directly as diagonal bands in the
    # flat heatmap -- no 3D projection needed to see it.
    panel_letters = ['a', 'b', 'c', 'd', 'e']
    for name, letter in zip(VAR_NAMES, panel_letters):
        r = resolved_tol[name]
        eigvecs, eigvals = circulant_kl_basis(r['std'], r['correlation_length_blades'])
        C_theory = (eigvecs * eigvals[None, :]) @ eigvecs.T
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        im = ax.imshow(C_theory, cmap=plot_style.SEQ_CMAP, origin='lower',
                        interpolation='nearest', aspect='equal')
        ax.set_xlabel('blade j')
        ax.set_ylabel('blade i')
        plot_style.two_tier_title(ax, f'Spatial covariance -- {VAR_LABELS[name]}',
                                   f'theoretical circulant covariance, std={r["std"]:.3f} {r["unit"]}')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('covariance')
        fig.tight_layout()
        _savefig(fig, figs, f'step3_fig5{letter}_covariance_{name}')
    print(f"  step3_fig5a-e: covariance heatmaps, {', '.join(VAR_NAMES)}")

    print(f"  All 5 Step 3 figures saved to: {figs}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _log_path = os.path.join(_HERE, 'Step3.txt')
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    t_start = time.time()
    hdr(f"STEP 3 v1.0: GEOMETRIC MISTUNING PARAMETERIZATION — {NB}-BLADE BLISK (PCE PROJECT)")
    print(f"  Input dir  (Step 1, read-only): {CONFIG['input_dir']}")
    print(f"  Output dir (Step 3):            {OUT}")

    baseline = load_baseline_geometry()
    resolved_tol = resolve_tolerances(baseline)
    samples = generate_mistuning_samples(resolved_tol)
    validate_samples(samples, resolved_tol)
    save_outputs(samples, resolved_tol, baseline)
    make_step3_figures(samples, resolved_tol)
    passed = print_validation_summary()

    hdr("STEP 3 COMPLETE")
    elapsed = time.time() - t_start
    print(f"  Validation: {'PASSED' if passed else 'FAILED — see STEP 3 VALIDATION SUMMARY above'}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"\n  Files in {OUT}:")
    for fn in sorted(os.listdir(OUT)):
        fp = os.path.join(OUT, fn)
        if os.path.isfile(fp):
            print(f"    {fn:30s} {os.path.getsize(fp) / 1e3:8.2f} KB")
    print(f"\nLog saved: {_log_path}")
    sys.stdout = sys.__stdout__
    _log_file.close()
