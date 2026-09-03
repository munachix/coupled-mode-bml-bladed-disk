import numpy as np


def _fit_given_wd(t, q, Omega, omega_d, decay_rate):
    env = np.exp(-decay_rate * t)
    X = np.column_stack([np.cos(Omega * t), np.sin(Omega * t),
                          env * np.cos(omega_d * t), env * np.sin(omega_d * t)])
    coef, *_ = np.linalg.lstsq(X, q, rcond=None)
    resid = q - X @ coef
    a1, b1, a2, b2 = coef
    return float(np.hypot(a1, b1)), float(np.hypot(a2, b2)), float(np.sum(resid ** 2))


def fit_A_ss(t, q, Omega, omega0_guess, zeta, wd_search_pct=3.0, n_grid=201):
    """Extracts the steady-state forced-response amplitude from a transient
    (still-beating) time history: q(t) ~ A_ss*cos(Omega t - phi) +
    A_tr(t)*cos(omega_d t - psi), A_tr(t)=A_tr0*exp(-decay_rate*t).

    decay_rate = zeta*omega0 is treated as KNOWN/EXACT, not fit -- it's
    imposed directly via BETAD (stiffness-proportional Rayleigh damping),
    not an independent uncertain physical parameter, unlike the transient's
    own oscillation frequency omega_d, which the real nonlinear dynamics
    could shift away from the naive linear omega0 -- so only omega_d is
    grid-searched (validated synthetically: <1.6% error even with omega_d
    off by 2% from the naive guess; assuming decay_rate is ALSO unknown and
    grid-searching it too made things WORSE, up to ~9.5% error, from
    overfitting noise with an unjustified extra free parameter -- decay
    rate is controlled by us, not guessed).

    Returns dict with A_ss, A_tr0, omega_d_fit, ssr (fit residual)."""
    decay_rate = zeta * omega0_guess
    omega_d_nom = omega0_guess * np.sqrt(1 - zeta ** 2)
    candidates = omega_d_nom * (1 + np.linspace(-wd_search_pct / 100, wd_search_pct / 100, n_grid))
    best = None
    for wd in candidates:
        A_ss, A_tr, ssr = _fit_given_wd(t, q, Omega, wd, decay_rate)
        if best is None or ssr < best[3]:
            best = (A_ss, A_tr, wd, ssr)
    A_ss, A_tr, wd_best, ssr = best
    q_var = float(np.var(q))
    r2 = 1 - (ssr / len(q)) / q_var if q_var > 0 else float('nan')
    return dict(A_ss=A_ss, A_tr0=A_tr, omega_d_fit=wd_best, ssr=ssr, r2=r2,
                omega_d_mismatch_pct=(wd_best / omega_d_nom - 1) * 100)


def consistency_check(t, q, Omega, omega0_guess, zeta, label=''):
    """Self-validation with NO independent ground truth needed: fit on the
    first half of the record and on the full record separately. If the
    method is extracting something real (not an artifact of window
    length), both should agree. This is the actual 'make sure it's
    correct' check -- run on REAL data, not just synthetic tests."""
    n = len(t)
    half = fit_A_ss(t[:n // 2], q[:n // 2], Omega, omega0_guess, zeta)
    full = fit_A_ss(t, q, Omega, omega0_guess, zeta)
    third_q = fit_A_ss(t[:3 * n // 4], q[:3 * n // 4], Omega, omega0_guess, zeta)
    rel_diff_half_full = abs(full['A_ss'] - half['A_ss']) / full['A_ss'] * 100
    rel_diff_3q_full = abs(full['A_ss'] - third_q['A_ss']) / full['A_ss'] * 100
    print(f"  [{label}] consistency check:")
    print(f"    half-record   A_ss={half['A_ss']:.5f}  R2={half['r2']:.4f}")
    print(f"    3/4-record    A_ss={third_q['A_ss']:.5f}  R2={third_q['r2']:.4f}")
    print(f"    full-record   A_ss={full['A_ss']:.5f}  R2={full['r2']:.4f}")
    print(f"    |half-full|/full = {rel_diff_half_full:.2f}%,  |3/4-full|/full = {rel_diff_3q_full:.2f}%")
    passed = rel_diff_half_full < 15.0 and rel_diff_3q_full < 10.0 and full['r2'] > 0.9
    print(f"    PASS={passed} (thresholds: half<15%, 3/4<10%, R2>0.9)")
    return dict(passed=passed, half=half, third_q=third_q, full=full,
                rel_diff_half_full=rel_diff_half_full, rel_diff_3q_full=rel_diff_3q_full)
