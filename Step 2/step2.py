"""
STEP 2 v10.0: PC-CMS ROM + Secondary Eigenbasis — 24-Blade Blisk (PCE Project)
==============================================================================

v10.0 changes vs v9.0 (July 2026) -- FI-TRUNCATION AUTO-SWEEP + FIGURE CLEANUP:
  - REMOVED every blisk/CAD geometry figure and ALL of its supporting code:
    fig0a-0d (CAD hero shots), fig3 (master-DOF-on-CAD overlay), fig4
    (tip-amplitude-on-CAD heatmap), render_cad_blisk, load_cad_mesh,
    _tessellate_step, _resolve_stp_path, _upright_cad, _face_normals_cad,
    _lambert_shade_cad, blade_tip_xyz_from_mesh, make_cad_hero_figures, the
    STP_PATH/_CAD_* constants, and the Poly3DCollection import. Step 2 no
    longer touches Blisk.stp or OCP at all; it only produces data/diagnostic
    figures (freq. correlation/error, MAC, FI-convergence, FRF).
  - REDESIGNED step2_fig2_mac: was a dark 'inferno' colormap on a cream axes
    background (low-MAC cells rendered near-black, reading as "a black
    background"); now a light-yellow -> blue -> near-black colormap on an
    explicitly light canvas, so background/low values stay light and only
    genuinely high MAC cells go dark. Also ADDED a degenerate-pair
    "subspace MAC" diagnostic: a 24-blade cyclic-symmetric disk has
    frequency-degenerate nodal-diameter mode PAIRS, so full-order vs. ROM
    solvers can pick an arbitrary rotation within a pair's 2-D eigenspace,
    giving a low POINTWISE MAC even when the 2-D subspace is reproduced
    exactly. This used to just show up as an unexplained low min-diagonal
    MAC (e.g. 0.21); v10.0 detects near-degenerate clusters from the
    full-order frequency spacing and reports the principal-angle-based
    subspace MAC for each one directly on the figure.
  - FIXED the n_fi_modes=50 / n_sec=50 hard-coding. The 2B-CHECK validation
    in v8.x/v9.x showed >1% frequency error above secondary-mode ~37 (max
    1.52% at mode 45) -- a Craig-Bampton fixed-interface TRUNCATION error
    that v8.2a's own annotation already flagged and recommended checking at
    n_sec=70. v10.0 runs that check AUTOMATICALLY every run: it computes a
    buffer of n_fi_compute=110 fixed-interface modes, then re-projects
    K_r/M_r at several candidate truncations using EXACT block-matrix
    algebra (see sweep_fi_convergence() / _assemble_bordered_eig() -- the
    FI-FI block is exactly diag(eigenvalues) and the FI mass block is
    exactly the identity, both for free, so only ONE expensive sparse
    projection is ever computed; each candidate truncation is then a cheap
    dense re-assembly + eigh). It auto-selects the smallest n_fi meeting
    the error tolerances, floored at n_fi_keep_min=70 (this was verified
    against a synthetic brute-force reference to machine precision before
    being used on the real model -- see the accompanying assessment notes).
    n_sec (the secondary/nonlinear-analysis basis size fed to Steps 3-6)
    is raised from 50 to 70 to match. NEW outputs: step2_fi_convergence.json
    and figures/step2/step2_fig6_fi_convergence.png.

v9.0 changes vs v8.3 (July 2026) -- CAD RENDER MERGE:
  - REMOVED the FEA-facet blisk renderer (_load_blisk_mesh, _add_blisk,
    _surface_from_connectivity, _detect_node_layout, _style_axis(_bbox),
    _cmap_face_rgb, _view_dir). That renderer drew the blisk from the
    coarse ANSYS solver mesh (node_coordinates.npy + element_connectivity
    .npy) and silently fell back to a bare ax.scatter() of raw node
    coordinates whenever the connectivity file wasn't found at exactly
    the expected OUT path -- this was the "mess" in fig3_master_dofs.png.
  - ADDED a true CAD B-rep renderer that reads Blisk.stp DIRECTLY with
    the free OpenCASCADE kernel (`pip install cadquery-ocp`, the `OCP`
    package) -- no ANSYS/SpaceClaim/license needed. It tessellates the
    216 real analytic/B-spline surfaces at fine display resolution
    (~220k verts / 436k triangles, vs the FE mesh's few thousand facets)
    and shades with a two-light Lambertian model. Cached to
    cad_verts.npy / cad_faces.npy in OUT after the first call (~10s),
    so re-runs are near-instant and don't need OCP again.
  - make_step2_figures() now draws fig3 (master-DOF overlay) and fig4
    (tip-amplitude heatmap) on the true CAD surface instead of the FE
    mesh. Four new "hero" figures (fig0a isometric, fig0b top view,
    fig0c front elevation, fig0d single-blade detail) are generated
    from Blisk.stp alone, independent of any ANSYS run, via
    make_cad_hero_figures().
  - render_cad_blisk(..., scalar=...) is a reusable hook: colour the
    true CAD surface by an arbitrary per-vertex scalar (mode-shape
    amplitude, mistuning pattern, ...) -- intended for reuse in Steps
    3-6 wherever a physics result needs to be shown on the real geometry.
  - Sections 2A-2D (ROM assembly: load_step1, build_pccms_rom,
    build_secondary_basis, save_rom) are BYTE-IDENTICAL to v8.3 --
    NO solver/numerical logic changed anywhere in this file.

v8.3 changes vs v8.2 (July 2026):
  - VALIDATION-FIGURE RESTYLE ONLY. Figures now match Step 3's "ink-wave"
    look: opaque white/cream canvas (was transparent PNG), the shared
    palette (INK/C_1B/C_HF/C_WARN/...), bold left-aligned two-tier titles,
    horizontal legends below each axis, and the _wave_stems glyph for
    per-mode bar/stem plots. Fig 1 (freq comparison) is now split into two
    files (fig1a correlation, fig1b per-mode error) to match Step 3's
    one-plot-per-PNG convention. NO numerical/solver logic changed anywhere
    in this file -- sections 2A-2D (ROM assembly, save_rom) are untouched;
    only the plotting calls inside make_step2_figures were rewritten.
    (v9.0 note: the _save_transparent() alias mentioned below was removed
    when the FEA-mesh renderer was replaced by the CAD renderer; _savefig()
    is the only save helper now.)

v8.2a ANNOTATION (May 2026) -- Q1 audit notes; NO solver-physics change here.
  The secondary-eigenbasis reduction (181k -> 770 -> 50 modes) is correct, but
  two facts must be DISCLOSED when this ROM feeds the v5.0 multi-harmonic HBM
  (Step 6) so reviewers can judge fidelity:
    (1) HF-mode frequency error.  Above secondary-mode ~37 the retained basis
        no longer reproduces the full-order modal frequencies to <1% (the
        Craig-Bampton fixed-interface truncation degrades the high tail).  The
        v5.0 forced response is driven in the 1B band (~285-370 Hz) where the
        low modes dominate, so this does not bias the QoI -- but the cubic K3
        now couples ALL 50 modes, so any energy that reaches modes >37 is only
        order-of-magnitude accurate.  A 50->70 secondary-basis convergence run
        is the recommended robustness check (see note 2).
    (2) Basis-size convergence.  The choice n_sec=50 was made for the linear
        response; with full K3 coupling the recommended check is to re-run the
        Step 2 secondary reduction at n_sec=70 and confirm the Step 6 log_LAMF
        mean/std shift by < a few %.  This is cheap relative to the dataset.
  Output contract is unchanged and already includes secondary_bundle.npz
  (K_sec/M_sec/C_sec) which Step 6 v5.0 loads directly.
==============================================================================
v8.2 changes vs v8.1:
  - Robust 3-tier fixed-interface eigensolver (Section 4) replaces the
    fragile single eigsh call that crashed with SuperLU MemoryError on
    Windows. New strategy:
      Tier 1: reuse existing K_ss LU via LinearOperator (no new factorisation)
      Tier 2: fresh shift-invert at σ>0 with symmetric MMD ordering and
              SPD-friendly SuperLU options
      Tier 3: LOBPCG with Jacobi preconditioner (matrix-free, no LU)
    The old which='SM' fallback was removed — it called splu(M_ss)
    internally, which was the exact cause of the v8.1 MemoryError.

v8.1 inherited from v8.0 (36-blade → 24-blade port):
  - n_blades = 24 → 1B cluster = 24 modes (indices 0..23)
  - System: ~183,321 DOFs (61107 nodes × 3)
  - Material: E=96000 MPa, nu=0.36, rho=4.62e-6 kg/mm^3
  - f_ref for beta_mis uses mean of 1B cluster (modes 0..23)
  - Output dir: F:\\ANSYS PCE\\ROM_data

Everything else is identical to v8.0/v8.1:
  - Modal damping C_sec = diag(2·ζ·ω_k), exact ζ=0.002 for all modes
  - C_mis = C_sec + β_mis · (K_mis - K_sec)
  - β_mis = ζ / (π · f_ref)
  - PC-CMS: constraint modes + n_fi fixed-interface modes
  - Secondary eigenbasis: n_sec = 50 modes

Outputs (to output_dir = F:\\ANSYS PCE\\ROM_data):
  K_r.npy, M_r.npy, C_r.npy            — ROM matrices
  T_pccms.npz                           — projection basis
  freqs_rom.npy                         — ROM frequencies
  blade_rom_indices.npz
  Phi_sec.npy, freqs_sec.npy
  T_full2sec.npy                        — (n_full, n_sec)
  secondary_bundle.npz                  — K_sec, M_sec, C_sec, K3_sec stub
  rom_manifest.json
  damping_alpha.npy, damping_beta.npy   — Rayleigh (reference only)
  damping_beta_mis.npy, damping_zeta.npy

Author: PCE-Bayesian Framework — v8.3 (24-blade, robust FI solver, ink-wave figs)
Date:   July 2026
"""

import numpy as np, os, time, json, gc, sys
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu, lobpcg, LinearOperator
from scipy.linalg import eigh

CONFIG = {
    'output_dir':      r'F:\ANSYS PCE\ROM_data',   # CHANGED
    'n_blades':        24,    # CHANGED: 36 → 24

    # ------------------------------------------------------------------
    # v10.0 FI-MODE / SECONDARY-BASIS SIZE  (was hard-coded 50/50)
    # ------------------------------------------------------------------
    # v8.x/v9.x kept a fixed n_fi_modes=50 and only discovered, post-hoc,
    # that the 770-DOF ROM's own eigenvalue error exceeds 1% above
    # secondary-mode ~37 (max 1.52% at mode 45) -- a classic Craig-Bampton
    # fixed-interface TRUNCATION error. v8.2a's own annotation recommended
    # a 50->70 convergence check before trusting the cubic-coupled (Step 3)
    # ROM out to mode 70. v10.0 runs that check AUTOMATICALLY, every run,
    # using EXACT block-matrix algebra (see sweep_fi_convergence()) so the
    # extra candidates cost a few cheap dense eigensolves, not repeated
    # expensive sparse projections:
    #   1. Compute a generous buffer of FI modes ('n_fi_compute').
    #   2. Re-assemble K_r/M_r at each 'n_fi_sweep_candidates' truncation
    #      and compare against the Step-1 full-order truth frequencies.
    #   3. Auto-select the smallest n_fi meeting the tolerances below,
    #      floored at 'n_fi_keep_min' (the user's requested >=70).
    # Result is REPORTED, not assumed -- see step2_fi_convergence.json and
    # figures/step2/step2_fig6_fi_convergence.png.
    'n_fi_compute':          110,  # raw FI eigenpairs computed (one solve)
    'n_fi_sweep_candidates': [50, 60, 70, 80, 90, 100],
    # v10.1 MEMORY FIX: the master+constraint projection in the sweep below
    # (T_mc.T @ K @ T_mc etc.) is now computed in column-chunks of T_mc
    # instead of ever materialising the full (n_dof, n_m) dense product
    # K@T_mc / M@T_mc (~1 GB EACH at n_dof=181,473, n_m=720 -- holding both
    # of those plus T_mc itself simultaneously caused a MemoryError on at
    # least one real run). Lower this if you still see MemoryError; raising
    # it trades a bit more peak memory for slightly fewer chunk iterations
    # (the per-chunk cost is dominated by the sparse matmul either way, so
    # this is not a significant speed lever).
    'sweep_chunk_cols':      90,
    'n_fi_keep_min':         70,   # never go BELOW this many FI modes
    'fi_tol_mean_pct':       0.30, # target: mean freq error < this ...
    'fi_tol_max_pct':        1.00, # ... and max freq error < this ...
    'fi_tol_check_nmodes':   50,   # ... over the first N full-order modes

    'n_sec':           70,    # secondary eigenbasis dimension (was 50)

    'damping_ratio':   0.002, # ζ — exact modal damping for all modes
    'cm_batch':        64,    # constraint mode solve batch size

    'mac_min_threshold': 0.85,  # validation gate for the MAC diagnostic
}

OUT = CONFIG['output_dir']
NB  = CONFIG['n_blades']
os.makedirs(OUT, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_VALIDATION_LOG = []   # list of (name, bool_passed, detail_str) — accumulated through the run


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


def print_validation_summary():
    hdr("STEP 2 VALIDATION SUMMARY")
    for name, ok, detail in _VALIDATION_LOG:
        status = 'OK' if ok else 'FAIL'
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    n_fail = sum(1 for _, ok, _ in _VALIDATION_LOG if not ok)
    hdr(f"STEP 2 VALIDATION: {'PASSED' if n_fail == 0 else f'FAILED ({n_fail} check(s))'}")
    return n_fail == 0


def hdr(t): print(f"\n{'='*70}\n  {t}\n{'='*70}")


def reconstruct_symmetric(A):
    """Enforce exact symmetry: return (A + A^T) / 2 via sparse ops."""
    A = A.tocsc()
    return A + A.T - sparse.diags(A.diagonal(), format='csc')


def parse_mapping(path):
    labels = {'UX': 0, 'UY': 1, 'UZ': 2}
    nd2r = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            try:
                eqn, node = int(p[0]), int(p[1])
                di = labels.get(p[2].strip(), -1)
                if di < 0:
                    continue
                nd2r[(node, di)] = eqn - 1
            except:
                continue
    return nd2r


# ═══════════════════════════════════════════════════════════════════
# 2A. LOAD STEP 1 DATA
# ═══════════════════════════════════════════════════════════════════
def load_step1():
    hdr("2A: LOADING STEP 1 DATA")
    d = {}

    # K and M
    K = reconstruct_symmetric(sparse.load_npz(os.path.join(OUT, 'K_full.npz')))
    M = reconstruct_symmetric(sparse.load_npz(os.path.join(OUT, 'M_full.npz')))
    d['K'], d['M'], d['n_dof'] = K, M, K.shape[0]
    print(f"  K: {K.shape},  M: {M.shape}")
    print(f"  K nnz: {K.nnz:,}  (sparsity {K.nnz/K.shape[0]**2*100:.2f}%)")

    # DOF mapping
    mapping_file = os.path.join(OUT, 'Stiff_mapped.mapping')
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(
            f"Stiff_mapped.mapping not found at {OUT}\n"
            "Run Step 1 first.")
    nd2r = parse_mapping(mapping_file)
    d['nd2r'] = nd2r
    print(f"  {len(nd2r):,} DOFs mapped")

    # Frequencies from modal solve
    for fn in ['frequencies_all.npy', 'freqs_full.npy']:
        fp = os.path.join(OUT, fn)
        if os.path.exists(fp):
            d['freqs_full'] = np.load(fp)
            print(f"  {len(d['freqs_full'])} full modes: "
                  f"{d['freqs_full'][0]:.2f}–{d['freqs_full'][-1]:.2f} Hz")
            n_1b = int((d['freqs_full'] < 700).sum())
            print(f"  1B cluster: {n_1b} modes below 700 Hz  "
                  f"(expected {NB} for {NB}-blade disk)")
            break

    # Blade tip maps
    d['tip_maps'] = {}
    for b in range(NB):
        fp = os.path.join(OUT, f'bladetip_blade{b}_nodes.npy')
        if os.path.exists(fp):
            d['tip_maps'][b] = np.load(fp)
    print(f"  {len(d['tip_maps'])} blade tip maps loaded, "
          f"{len(d['tip_maps'].get(0, []))} tip nodes/blade")

    return d


# ═══════════════════════════════════════════════════════════════════
# 2B. PC-CMS ROM
# ═══════════════════════════════════════════════════════════════════
def _assemble_bordered_eig(A, B, A_cross_p, B_cross_p, ev_fi_p):
    """Assemble the bordered (master+constraint | leading-p FI-mode) ROM and
    solve the generalised eigenproblem.

    Exact block structure (T = [T_mc | Phi_fi[:, :p]], Phi_fi mass-normalised
    fixed-interface eigenvectors of (K_ss, M_ss)):
        K_r = [[A,            A_cross_p     ],     M_r = [[B,            B_cross_p],
               [A_cross_p.T,  diag(ev_fi_p)]]            [B_cross_p.T,  I_p      ]]
    The FI-FI blocks are EXACT (no new sparse solve) because the FI modes are
    already M_ss-orthonormal eigenvectors of (K_ss, M_ss): phi_i^T K_ss phi_j =
    ev_fi_j * phi_i^T M_ss phi_j = ev_fi_j * delta_ij. This lets every
    candidate truncation p be evaluated from ONE pair of expensive sparse
    projections (A, B, A_cross_full, B_cross_full), computed once for the
    largest p and simply sliced for smaller ones -- an O(1)-expensive-solve,
    O(#candidates)-cheap-assembly convergence study instead of re-running the
    full projection per candidate."""
    p = ev_fi_p.shape[0]
    n_m = A.shape[0]
    n = n_m + p
    K_r = np.zeros((n, n)); M_r = np.zeros((n, n))
    K_r[:n_m, :n_m] = A;            K_r[:n_m, n_m:] = A_cross_p
    K_r[n_m:, :n_m] = A_cross_p.T;  K_r[n_m:, n_m:] = np.diag(ev_fi_p)
    M_r[:n_m, :n_m] = B;            M_r[:n_m, n_m:] = B_cross_p
    M_r[n_m:, :n_m] = B_cross_p.T;  M_r[n_m:, n_m:] = np.eye(p)
    K_r = 0.5 * (K_r + K_r.T); M_r = 0.5 * (M_r + M_r.T)
    eigvals, eigvecs = eigh(K_r, M_r)
    return eigvals, eigvecs, K_r, M_r


def sweep_fi_convergence(K, M, T_mc, slave, ev_fi_buf, phi_fi_buf, freqs_full):
    """Automatic Craig-Bampton fixed-interface truncation study (v10.0).

    v8.x/v9.x hard-coded n_fi_modes=50 and only ever validated THAT ONE
    choice, post-hoc, against the full-order truth -- discovering (but not
    correcting) a >1% error above secondary-mode ~37. v8.2a's own annotation
    recommended re-running at n_sec=70 to check; this function does that
    check FOR EVERY RUN, automatically, and picks the smallest fixed-interface
    truncation that (a) meets the error tolerances in CONFIG and (b) is never
    smaller than CONFIG['n_fi_keep_min'] (the user's requested >=70 floor).
    """
    hdr("2B-SWEEP: FI-MODE TRUNCATION CONVERGENCE  (exact block algebra)")
    n_dof_full, n_m = T_mc.shape
    n_fi_compute = phi_fi_buf.shape[1]
    t0 = time.time()

    # v10.1 MEMORY FIX: compute A = T_mc.T@K@T_mc, B = T_mc.T@M@T_mc, and the
    # FI cross-terms in COLUMN-CHUNKS of T_mc, so the big (n_dof, n_m) dense
    # product (~1 GB at this model's size) is NEVER fully materialised --
    # only one (n_dof, chunk) slice of it exists at a time, and K's and M's
    # products are processed in fully separate passes so they are never both
    # alive simultaneously either. This is algebraically identical to the
    # monolithic 'KT_mc = K @ T_mc; MT_mc = M @ T_mc' version it replaces
    # (matrix multiplication distributes over column blocks of the right
    # operand, so each output column is exactly the same regardless of how
    # the columns are batched) -- verified bit-for-bit identical against the
    # monolithic version on a synthetic stand-in before being trusted here.
    # This was needed because the monolithic version held T_mc + K@T_mc +
    # M@T_mc simultaneously (~3 GB) and crashed with MemoryError on a real
    # (181,473-DOF, 720-master-DOF) run; chunking keeps peak memory to
    # roughly T_mc alone plus one small chunk.
    chunk = max(1, int(CONFIG['sweep_chunk_cols']))
    A = np.zeros((n_m, n_m)); A_cross_full = np.zeros((n_m, n_fi_compute))
    for c0 in range(0, n_m, chunk):
        c1 = min(c0 + chunk, n_m)
        KTc = K @ T_mc[:, c0:c1]
        A[:, c0:c1] = T_mc.T @ KTc
        A_cross_full[c0:c1, :] = KTc[slave, :].T @ phi_fi_buf
        del KTc
    gc.collect()
    A = 0.5 * (A + A.T)
    print(f"  Master+constraint block A {A.shape} + FI cross-terms "
          f"(K side) projected in {chunk}-column chunks in "
          f"{time.time()-t0:.1f}s")

    t1 = time.time()
    B = np.zeros((n_m, n_m)); B_cross_full = np.zeros((n_m, n_fi_compute))
    for c0 in range(0, n_m, chunk):
        c1 = min(c0 + chunk, n_m)
        MTc = M @ T_mc[:, c0:c1]
        B[:, c0:c1] = T_mc.T @ MTc
        B_cross_full[c0:c1, :] = MTc[slave, :].T @ phi_fi_buf
        del MTc
    gc.collect()
    B = 0.5 * (B + B.T)
    print(f"  Master+constraint block B {B.shape} + FI cross-terms "
          f"(M side) projected in {chunk}-column chunks in "
          f"{time.time()-t1:.1f}s")

    cands = sorted(set(int(c) for c in CONFIG['n_fi_sweep_candidates']
                        if 1 <= c <= n_fi_compute))
    if n_fi_compute not in cands:
        cands.append(n_fi_compute)
    tol_mean = CONFIG['fi_tol_mean_pct']; tol_max = CONFIG['fi_tol_max_pct']
    nchk = CONFIG['fi_tol_check_nmodes']; p_min = CONFIG['n_fi_keep_min']

    print(f"  {'n_fi_keep':>9}  {'n_rom':>6}  {'mean err%':>10}  {'max err%':>9}  status")
    table, solved, chosen_p = [], {}, None
    for p in cands:
        eigvals, eigvecs, K_r, M_r = _assemble_bordered_eig(
            A, B, A_cross_full[:, :p], B_cross_full[:, :p], ev_fi_buf[:p])
        freqs_p = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
        nc = min(len(freqs_full), len(freqs_p), nchk)
        err = np.abs(freqs_p[:nc] - freqs_full[:nc]) / (freqs_full[:nc] + 1e-10) * 100
        row = {'n_fi_keep': p, 'n_rom': int(A.shape[0] + p),
               'err_mean_pct': float(err.mean()), 'err_max_pct': float(err.max())}
        table.append(row); solved[p] = (eigvals, eigvecs, K_r, M_r)
        ok = row['err_mean_pct'] <= tol_mean and row['err_max_pct'] <= tol_max and p >= p_min
        if ok and chosen_p is None:
            chosen_p = p
        print(f"  {p:9d}  {row['n_rom']:6d}  {row['err_mean_pct']:10.4f}  "
              f"{row['err_max_pct']:9.4f}  {'<- SELECTED' if ok and p == chosen_p else ('meets tol' if ok else '')}")

    if chosen_p is None:
        chosen_p = max(p_min, cands[-1])
        if chosen_p not in solved:
            eigvals, eigvecs, K_r, M_r = _assemble_bordered_eig(
                A, B, A_cross_full[:, :chosen_p], B_cross_full[:, :chosen_p],
                ev_fi_buf[:chosen_p])
            freqs_p = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
            nc = min(len(freqs_full), len(freqs_p), nchk)
            err = np.abs(freqs_p[:nc]-freqs_full[:nc])/(freqs_full[:nc]+1e-10)*100
            table.append({'n_fi_keep': chosen_p, 'n_rom': int(A.shape[0]+chosen_p),
                          'err_mean_pct': float(err.mean()), 'err_max_pct': float(err.max())})
            solved[chosen_p] = (eigvals, eigvecs, K_r, M_r)
        print(f"  WARNING: no candidate up to n_fi_compute={n_fi_compute} met "
              f"tol (mean<{tol_mean}%, max<{tol_max}%) at n_fi_keep>={p_min}. "
              f"Falling back to n_fi_keep={chosen_p} -- consider raising "
              f"CONFIG['n_fi_compute'] and re-running.")

    eigvals_sel, eigvecs_sel, K_r_sel, M_r_sel = solved[chosen_p]
    row_sel = next(r for r in table if r['n_fi_keep'] == chosen_p)
    print(f"\n  SELECTED n_fi_keep = {chosen_p}  (n_rom = {row_sel['n_rom']})  "
          f"-> mean err {row_sel['err_mean_pct']:.4f}%, "
          f"max err {row_sel['err_max_pct']:.4f}%  "
          f"(targets: mean<{tol_mean}%, max<{tol_max}%, floor n_fi_keep>={p_min})")
    return dict(n_fi_selected=chosen_p, K_r=K_r_sel, M_r=M_r_sel,
                eigvals=eigvals_sel, eigvecs=eigvecs_sel, sweep_table=table)


def build_pccms_rom(d):
    hdr("2B: PC-CMS ROM CONSTRUCTION")
    K, M, nd2r = d['K'], d['M'], d['nd2r']
    n_dof = d['n_dof']

    # 1. Master DOFs (blade-tip, ordered blade by blade)
    print("  1. Master DOFs (blade-tip)...")
    btmd = {}
    for b in range(NB):
        nodes = d['tip_maps'].get(b, np.array([]))
        dofs  = sorted({nd2r[(int(nid), di)]
                        for nid in nodes
                        for di in range(3)
                        if (int(nid), di) in nd2r})
        btmd[b] = dofs

    # NOTE (mesh is NOT cyclic-symmetric): a full 360-degree ANSYS blisk
    # mesh is generated as one whole-disk mesh, not stamped out as 24
    # identical rotated copies of a single sector. Per-blade tip-node counts
    # can legitimately differ slightly from blade to blade (meshing
    # tolerances, local refinement, etc.) -- Step 1's own angular-window
    # partition already respects this (see replicate_to_all_blades). This
    # ROM construction must NOT force a uniform per-blade master-DOF count:
    # truncating every blade down to min(len(btmd[b])) silently throws away
    # real tip DOFs from every blade that has more nodes than the smallest
    # one, which quietly imposes an artificial cyclic-symmetry assumption
    # onto a genuinely non-conforming, non-symmetric full-disk mesh.
    blade_sizes = np.array([len(btmd[b]) for b in range(NB)])
    print(f"    Tip master DOFs/blade: min={blade_sizes.min()}, "
          f"max={blade_sizes.max()}, mean={blade_sizes.mean():.1f}")
    if blade_sizes.min() != blade_sizes.max():
        print(f"    NOTE: per-blade counts are NOT uniform (mesh is a whole-disk "
              f"model, not a stamped cyclic sector) -- keeping each blade's true "
              f"count rather than truncating to the minimum.")
    n_m = int(blade_sizes.sum())
    print(f"    {n_m} master DOFs total ({NB} blades, "
          f"{blade_sizes.min()}-{blade_sizes.max()} DOFs/blade)")

    # Ordered master array: blade 0 first, then 1, ... NB-1.
    # bri[b] now indexes a variable-length block (cumulative offsets)
    # instead of assuming a fixed-width ntp-sized block per blade.
    master_ordered, bri = [], {}
    offset = 0
    for b in range(NB):
        nb_dofs = len(btmd[b])
        bri[b] = np.arange(offset, offset + nb_dofs, dtype=int)
        master_ordered.extend(btmd[b])
        offset += nb_dofs
    master = np.array(master_ordered, dtype=int)
    slave  = np.setdiff1d(np.arange(n_dof), master).astype(int)
    print(f"    Slave: {len(slave):,} DOFs")

    # 2. Partition K and M
    print("  2. Partitioning K, M...")
    K_ss = K[np.ix_(slave, slave)].tocsc()
    K_sm = K[np.ix_(slave, master)]
    M_ss = M[np.ix_(slave, slave)].tocsc()
    print(f"    K_ss: {K_ss.shape},  K_sm: {K_sm.shape}")

    # 3. Constraint modes: Psi_C = -K_ss^{-1} K_sm, streamed directly into T
    # MEMORY-SAFE VERSION:
    #   * splu with SPD-friendly options reduces L+U fill-in 2–4× vs default
    #   * Stream each column straight into the projection matrix T — we avoid
    #     ever holding the full (180753, 720) constraint-mode block in RAM
    #     (~993 MB). T must exist anyway, so this costs nothing extra.
    #   * Each solve uses only one ~1.4 MB dense RHS column.
    print(f"  3. Constraint modes ({n_m} solves, streamed into T)...")
    t0 = time.time()
    K_sm_csc = K_sm.tocsc()
    lu = splu(
        K_ss,
        permc_spec='MMD_AT_PLUS_A',
        options={'SymmetricMode': True, 'DiagPivotThresh': 0.0},
    )

    # v10.2 MEMORY FIX: "Finalising T" (step 6 below) used to allocate a
    # SECOND full-size dense array (n_dof, n_rom) ~1.12 GiB and copy T_mc
    # into it, while T_mc (~0.97 GiB) was still alive -- peak ~2.1 GiB on
    # top of everything else already resident (K_ss/M_ss, phi_fi, K_r/M_r/
    # eigvecs_r from the sweep), which is exactly the "two big dense arrays
    # at once" anti-pattern the v10.1 fix above already eliminated once,
    # just reintroduced one step later. Fix: allocate ONE array at its
    # maximum possible final width up front -- n_fi_compute (CONFIG,
    # known now) is an upper bound on n_fi, so n_m + n_fi_compute is an
    # upper bound on n_rom -- and write the constraint-mode block directly
    # into its first n_m columns. sweep_fi_convergence only READS T_mc (to
    # build A=T_mc.T@K@T_mc, B=T_mc.T@M@T_mc), so a view is safe. Step 6
    # then just SLICES this array down to (n_dof, n_rom) -- a free view,
    # not a copy -- instead of allocating+copying a second big array.
    n_fi_compute_max = CONFIG['n_fi_compute']
    T = np.zeros((n_dof, n_m + n_fi_compute_max), dtype=np.float64)
    T_mc = T[:, :n_m]          # view, not a copy
    for i, mdof in enumerate(master):
        T_mc[mdof, i] = 1.0

    for j in range(n_m):
        rhs_j = -K_sm_csc[:, j].toarray().ravel()
        T_mc[slave, j] = lu.solve(rhs_j)
        if (j + 1) % 100 == 0 or (j + 1) == n_m:
            pct = (j + 1) / n_m * 100
            print(f"      {j+1}/{n_m}  ({pct:.0f}%,  {time.time()-t0:.0f}s)")

    del K_sm_csc; gc.collect()
    print(f"    Constraint modes: {time.time()-t0:.1f}s")

    # 4. Fixed-interface modes (vibrations with master DOFs fixed)
    #    Solve generalised eigenproblem  K_ss φ = λ M_ss φ  for smallest λ.
    #
    #    Robust 3-tier strategy (the v8.1 single-attempt code crashed on
    #    Windows when SuperLU ran out of memory factorising K_ss or M_ss):
    #
    #      Tier 1 : eigsh shift-invert with small positive σ. We REUSE the
    #               existing LU of K_ss (sigma is set close to 0; the LU of
    #               K_ss is a valid factorisation for σ→0⁺ since OPinv =
    #               (K_ss - σ M_ss)⁻¹ ≈ K_ss⁻¹ for σ small relative to the
    #               lowest eigenvalue). This avoids a second sparse LU.
    #      Tier 2 : eigsh with a fresh shift-invert at σ = small positive
    #               value, providing an explicit OPinv via a new splu of
    #               (K_ss - σ M_ss). Uses SuperLU options that reduce peak
    #               memory.
    #      Tier 3 : LOBPCG with diagonal (Jacobi) preconditioner — fully
    #               matrix-free, no LU required. Slower but bullet-proof.
    #
    #    We NEVER fall back to which='SM' because internally SciPy calls
    #    splu(M_ss), which is exactly what blew up in v8.1.
    #
    #    v10.0: we no longer decide "how many to keep" here -- we compute a
    #    generous BUFFER of n_fi_compute modes and let sweep_fi_convergence()
    #    (step 5 below) pick the smallest truncation that meets the accuracy
    #    targets, floored at CONFIG['n_fi_keep_min'].
    print(f"  4. Fixed-interface modes ({CONFIG['n_fi_compute']} computed as "
          f"a buffer; truncation chosen by the convergence sweep below)...")
    t0 = time.time()
    n_compute = CONFIG['n_fi_compute']
    ev_fi = phi_fi = None

    # ---- Tier 1: reuse existing LU of K_ss (zero-shift approx) ----------
    try:
        print("    Tier 1: eigsh shift-invert reusing existing K_ss LU ...")
        OPinv = LinearOperator(K_ss.shape, matvec=lu.solve, dtype=np.float64)
        ev_fi, phi_fi = eigsh(
            K_ss, k=n_compute, M=M_ss,
            sigma=0.0, which='LM', OPinv=OPinv,
            tol=1e-8, maxiter=2000,
        )
        print(f"    Tier 1 succeeded ({time.time()-t0:.1f}s)")
    except Exception as e1:
        print(f"    Tier 1 failed: {type(e1).__name__}: {e1}")

        # ---- Tier 2: fresh shift-invert with small positive σ -----------
        # Pick σ well below the lowest expected FI eigenvalue. Using the
        # smallest diagonal-ratio gives an order-of-magnitude lower bound.
        try:
            print("    Tier 2: fresh splu(K_ss - σ M_ss) with σ>0 ...")
            # Estimate a safe σ: ~1% of the smallest expected eigenvalue.
            # Lowest physical FI freq is well above 0 Hz, so σ=1.0 (in
            # ω² units → ~0.16 Hz) is safely below anything physical.
            sigma_shift = 1.0
            # Free anything we can before the big factorisation
            del lu
            gc.collect()
            A_shift = (K_ss - sigma_shift * M_ss).tocsc()
            # SuperLU memory-friendly options
            lu_shift = splu(
                A_shift,
                permc_spec='MMD_AT_PLUS_A',   # less fill-in than COLAMD here
                options={
                    'SymmetricMode': True,
                    'DiagPivotThresh': 0.0,    # no partial pivoting (SPD)
                },
            )
            del A_shift; gc.collect()
            OPinv = LinearOperator(K_ss.shape, matvec=lu_shift.solve,
                                   dtype=np.float64)
            ev_fi, phi_fi = eigsh(
                K_ss, k=n_compute, M=M_ss,
                sigma=sigma_shift, which='LM', OPinv=OPinv,
                tol=1e-8, maxiter=2000,
            )
            del lu_shift; gc.collect()
            print(f"    Tier 2 succeeded ({time.time()-t0:.1f}s)")
        except Exception as e2:
            print(f"    Tier 2 failed: {type(e2).__name__}: {e2}")

            # ---- Tier 3: LOBPCG with Jacobi preconditioner --------------
            print("    Tier 3: LOBPCG (matrix-free, diagonal preconditioner)...")
            n_s = K_ss.shape[0]
            # Diagonal (Jacobi) preconditioner: M⁻¹ ≈ diag(K_ss)⁻¹
            d_k = K_ss.diagonal()
            d_k[d_k <= 0] = d_k[d_k > 0].min() if (d_k > 0).any() else 1.0
            d_inv = 1.0 / d_k

            def precond(x):
                return d_inv[:, None] * x if x.ndim == 2 else d_inv * x
            Minv_op = LinearOperator((n_s, n_s), matvec=precond,
                                     matmat=precond, dtype=np.float64)

            # Random initial guess, then orthonormalise
            rng = np.random.default_rng(0)
            X0 = rng.standard_normal((n_s, n_compute)).astype(np.float64)
            X0, _ = np.linalg.qr(X0)

            ev_fi, phi_fi = lobpcg(
                K_ss, X0, B=M_ss, M=Minv_op,
                tol=1e-7, maxiter=500, largest=False, verbosityLevel=0,
            )
            print(f"    Tier 3 succeeded ({time.time()-t0:.1f}s)")
    # keep ALL n_compute buffer modes, ascending, mass-normalised -- the
    # convergence sweep below decides how many of them to actually use.
    idx    = np.argsort(ev_fi)
    ev_fi  = ev_fi[idx[:n_compute]]
    phi_fi = phi_fi[:, idx[:n_compute]]
    for i in range(n_compute):
        nm = np.sqrt(abs(phi_fi[:, i] @ (M_ss @ phi_fi[:, i])))
        if nm > 1e-10:
            phi_fi[:, i] /= nm
    freqs_fi = np.sqrt(np.abs(ev_fi)) / (2 * np.pi)
    print(f"    FI modes (buffer): {freqs_fi[0]:.2f}–{freqs_fi[-1]:.2f} Hz  "
          f"({time.time()-t0:.1f}s)")

    # 5. FI-truncation convergence sweep -- picks n_fi (and, with it, K_r/M_r)
    #    from the buffer above using exact block algebra (one solve, many
    #    cheap candidate assemblies -- see sweep_fi_convergence()).
    sweep = sweep_fi_convergence(K, M, T_mc, slave, ev_fi, phi_fi,
                                  d['freqs_full'])
    n_fi        = sweep['n_fi_selected']
    K_r, M_r    = sweep['K_r'], sweep['M_r']
    eigvals_r, eigvecs_r = sweep['eigvals'], sweep['eigvecs']
    fi_sweep_table = sweep['sweep_table']
    n_rom = n_m + n_fi
    freqs_rom = np.sqrt(np.abs(eigvals_r)) / (2 * np.pi)

    # 6. Finalise projection matrix T at the selected truncation.
    # v10.2: T was already allocated at its max width (n_m + n_fi_compute_max)
    # back in step 3, and T_mc is just T[:, :n_m] -- a view, not a separate
    # array. So there is nothing left to copy: write the selected n_fi
    # FI-mode columns into T[:, n_m:n_m+n_fi], then SLICE T down to
    # (n_dof, n_rom). That slice is a view of the same underlying buffer
    # (no allocation, no copy) since n_rom <= n_m + n_fi_compute_max always.
    print(f"\n  6. Finalising T at the selected n_fi={n_fi}...")
    T[slave, n_m:n_m + n_fi] = phi_fi[:, :n_fi]
    T = T[:, :n_rom]           # view: drops unused buffer columns, no copy
    del T_mc, phi_fi; gc.collect()
    print(f"    T: {T.shape}  ({n_dof:,} → {n_rom}, "
          f"{n_dof//n_rom}× reduction)")

    # ── MODAL DAMPING (exact ζ for every mode) ──────────────────────
    zeta      = CONFIG['damping_ratio']
    omega_rom = 2 * np.pi * freqs_rom
    # C_r = M_r Φ diag(2ζωk) Φ^T M_r  (Φ = mass-normalised eigenvectors)
    MrPhi = M_r @ eigvecs_r
    C_r   = MrPhi @ np.diag(2.0 * zeta * omega_rom) @ MrPhi.T
    C_r   = 0.5 * (C_r + C_r.T)

    # Rayleigh coefficients (reference only — not used in Steps 4-7)
    f1 = freqs_rom[0]; w1 = 2 * np.pi * f1
    idx2 = next((i for i in range(1, len(freqs_rom))
                 if freqs_rom[i] > 2 * f1), len(freqs_rom) - 1)
    w2 = 2 * np.pi * freqs_rom[idx2]
    A  = np.array([[1 / (2 * w1), w1 / 2], [1 / (2 * w2), w2 / 2]])
    alpha, beta = np.linalg.solve(A, [zeta, zeta])
    print(f"    Rayleigh (reference): α={alpha:.4e}, β={beta:.4e}")

    # β_mis for mistuning damping perturbation
    # CHANGED: use mean of first NB modes (24-mode 1B cluster)
    f_ref    = float(np.mean(freqs_rom[:NB]))
    beta_mis = zeta / (np.pi * f_ref)
    print(f"    Modal damping: ζ={zeta} exact for ALL {len(freqs_rom)} modes")
    print(f"    β_mis = ζ/(π·f_ref) = {beta_mis:.6e}  "
          f"(f_ref = mean of 1B cluster = {f_ref:.2f} Hz)")

    # Verify modal damping
    print(f"    Damping ratio check (spot modes):")
    C_modal_check = eigvecs_r.T @ C_r @ eigvecs_r
    check_indices = [0, 1, NB-1, NB, NB+5, min(n_rom-1, 49)]
    for i in check_indices:
        if i >= len(freqs_rom): continue
        wi    = omega_rom[i]
        zeta_i = C_modal_check[i, i] / (2 * wi) if wi > 0 else 0
        label = "1B" if i < NB else "HF"
        print(f"      Mode {i:3d} ({label}): f={freqs_rom[i]:8.2f} Hz, "
              f"ζ_modal={zeta_i:.6f}  "
              f"({'✓' if abs(zeta_i - zeta) < 1e-8 else '⚠'})")
    offdiag = C_modal_check - np.diag(np.diag(C_modal_check))
    off_rn  = np.linalg.norm(offdiag) / (np.linalg.norm(C_modal_check) + 1e-30)
    print(f"    Off-diagonal norm: {off_rn:.2e}  "
          f"({'✓' if off_rn < 1e-10 else '⚠'})")

    # ── Eigenvalue validation ─────────────────────────────────────────
    hdr("2B-CHECK: EIGENVALUE VALIDATION")
    ff  = d['freqs_full']
    # v10.0 FIX: use the SAME mode-count window as the sweep's own tolerance
    # check (CONFIG['fi_tol_check_nmodes']), not an independent hard-coded 50
    # -- otherwise this final report can flag a spurious ">tol" warning (or
    # miss a real one) simply because it looked at a different set of modes
    # than the sweep used to pick n_fi. Found via synthetic end-to-end
    # testing before this script was handed off.
    nc  = min(len(ff), len(freqs_rom), CONFIG['fi_tol_check_nmodes'])
    errs = np.abs(freqs_rom[:nc] - ff[:nc]) / (ff[:nc] + 1e-10) * 100
    print(f"  {'Mode':>5}  {'Full(Hz)':>10}  {'ROM(Hz)':>10}  {'Err%':>8}")
    for i in range(0, nc, max(1, nc // 10)):
        label = "1B" if i < NB else "HF"
        print(f"  {i+1:5d}  {ff[i]:10.2f}  {freqs_rom[i]:10.2f}  "
              f"{errs[i]:8.4f}%  ({label})")
    print(f"\n  Mean error: {errs.mean():.4f}%,  Max: {errs.max():.4f}%  "
          f"(first {nc} modes, matching the sweep's own check window)")
    eig_ok = errs.max() <= CONFIG['fi_tol_max_pct'] and errs.mean() <= CONFIG['fi_tol_mean_pct']
    _record_check(
        f"Eigenvalue error within tolerance (mean<{CONFIG['fi_tol_mean_pct']}%, "
        f"max<{CONFIG['fi_tol_max_pct']}%) at n_fi={n_fi}",
        eig_ok, f"mean={errs.mean():.4f}%, max={errs.max():.4f}%")
    if not eig_ok:
        print(f"  -- raise CONFIG['n_fi_compute'] and/or widen "
              f"'n_fi_sweep_candidates', then re-run.")

    d.update({
        'T': T, 'M_r': M_r, 'K_r': K_r, 'C_r': C_r,
        'freqs_rom': freqs_rom, 'master_dofs': master, 'slave_dofs': slave,
        'n_rom': n_rom, 'n_master': n_m, 'n_fi': n_fi,
        'n_fi_compute': CONFIG['n_fi_compute'], 'fi_sweep_table': fi_sweep_table,
        'alpha': alpha, 'beta': beta, 'beta_mis': beta_mis,
        'btmd': btmd, 'bri': bri, 'eigvecs_rom': eigvecs_r,
    })
    return d


# ═══════════════════════════════════════════════════════════════════
# 2C. SECONDARY EIGENBASIS
# ═══════════════════════════════════════════════════════════════════
def build_secondary_basis(d):
    hdr("2C: SECONDARY EIGENBASIS")
    n_sec = CONFIG['n_sec']
    K_r, M_r = d['K_r'], d['M_r']
    zeta = CONFIG['damping_ratio']

    print(f"  Computing {n_sec} eigenmodes of (K_r, M_r)...")
    t0 = time.time()
    eigvals, Phi = eigh(K_r, M_r, subset_by_index=[0, n_sec - 1])
    freqs = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Freq range: {freqs[0]:.2f}–{freqs[-1]:.2f} Hz")
    n_1b_sec = int((freqs < 700).sum())
    print(f"  1B cluster in secondary basis: {n_1b_sec} modes below 700 Hz  "
          f"(expected {NB})")

    # Mass-normalise: Phi^T M_r Phi = I
    for k in range(n_sec):
        nm = np.sqrt(abs(Phi[:, k] @ (M_r @ Phi[:, k])))
        if nm > 1e-10:
            Phi[:, k] /= nm

    # T_full2sec: maps full DOF vector to secondary coordinates
    print("  Computing T_full2sec = T_pccms @ Phi_sec ...")
    t0  = time.time()
    T_sp = sparse.csr_matrix(d['T'])
    T_f2s = T_sp.dot(Phi)
    print(f"  T_full2sec: {T_f2s.shape},  "
          f"{T_f2s.nbytes/1e6:.1f} MB,  {time.time()-t0:.1f}s")

    col_norms = np.array([np.linalg.norm(T_f2s[:, k]) for k in range(n_sec)])
    print(f"  T_f2s column norms: min={col_norms.min():.2f}, "
          f"max={col_norms.max():.2f}, mean={col_norms.mean():.2f}")
    print(f"  q_sec=1e-4 → physical displacement ≈ "
          f"{1e-4*col_norms[0]*1000:.3f} mm")

    # Secondary basis K_sec, M_sec, C_sec
    K_sec = np.diag(np.diag(Phi.T @ K_r @ Phi))   # diagonal (modal stiffness)
    M_sec = np.eye(n_sec)                           # identity (mass-normalised)

    # Modal damping: C_sec = diag(2·ζ·ω_k) — exact ζ for every mode
    omega_sec = 2 * np.pi * freqs
    C_sec = np.diag(2.0 * zeta * omega_sec)

    # Verify damping in secondary basis
    print(f"\n  Damping verification in secondary basis:")
    print(f"  {'Mode':>4}  {'f(Hz)':>8}  {'ζ_modal':>10}  label")
    alpha, beta = d['alpha'], d['beta']
    C_sec_rayleigh = alpha * M_sec + beta * K_sec   # reference
    for k in [0, 1, NB-1, NB, NB+5, n_sec-1]:
        if k >= n_sec: continue
        wk        = omega_sec[k]
        zm        = C_sec[k, k] / (2 * wk) if wk > 0 else 0
        zr        = C_sec_rayleigh[k, k] / (2 * wk) if wk > 0 else 0
        label     = "1B" if k < NB else "HF"
        print(f"  {k:4d}  {freqs[k]:8.2f}  {zm:10.6f}  ({label})  "
              f"Rayleigh would give {zr:.6f}")
    print(f"  Modal damping: ALL {n_sec} modes have ζ = {zeta:.6f} exactly ✓")

    # K3_sec placeholder — filled by Step 3
    # Included in bundle so Step 6 can load a single file
    K3_sec_stub = np.zeros((n_sec, n_sec, n_sec), dtype=np.float32)
    print(f"\n  K3_sec placeholder: {K3_sec_stub.shape}  (filled by Step 3)")

    # Save secondary bundle
    # DEFENSIVE FIX: re-assert OUT exists immediately before writing. This
    # guards against the directory having been removed AFTER the module-
    # level os.makedirs(OUT) ran at import time but BEFORE this save point
    # -- observed in practice (FileNotFoundError here despite os.makedirs
    # succeeding at startup) most likely because the long-running PyMAPDL
    # session from Step 1 (constraint-mode solves alone took ~3 min in this
    # run) or an external process (AV scan, cloud-sync client, a parallel
    # ANSYS working-directory cleanup) removed/relocated F:\ANSYS PCE\
    # ROM_data partway through this script's ~10+ minute runtime. Every
    # np.save/np.savez call in this module from this point on is now
    # preceded by the same guard (see save_rom below) rather than relying
    # solely on the one-time makedirs at import.
    os.makedirs(OUT, exist_ok=True)
    bundle_path = os.path.join(OUT, 'secondary_bundle.npz')
    np.savez(bundle_path,
             K_sec=K_sec, M_sec=M_sec, C_sec=C_sec,
             K3_sec=K3_sec_stub, freqs_sec=freqs)
    print(f"  Saved: secondary_bundle.npz  "
          f"({os.path.getsize(bundle_path)/1e6:.1f} MB)")

    d.update({
        'Phi_sec': Phi, 'freqs_sec': freqs,
        'T_full2sec': T_f2s, 'T_f2s_col_norms': col_norms,
        'K_sec': K_sec, 'M_sec': M_sec, 'C_sec': C_sec,
        'n_sec': n_sec,
    })
    return d


# ═══════════════════════════════════════════════════════════════════
# 2D. SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════════════════════════
def save_rom(d):
    hdr("2D: SAVING ROM OUTPUTS")
    # DEFENSIVE FIX (see build_secondary_basis's identical guard above for
    # the full explanation): re-assert OUT exists before this function's
    # dozen-plus np.save/np.savez calls, in case the directory was removed
    # by an external process sometime after the module-level makedirs ran.
    os.makedirs(OUT, exist_ok=True)

    # Core arrays
    for name, arr in [
        ('M_r',              d['M_r']),
        ('K_r',              d['K_r']),
        ('C_r',              d['C_r']),
        ('freqs_rom',        d['freqs_rom']),
        ('master_dofs',      d['master_dofs']),
        ('slave_dofs',       d['slave_dofs']),
        ('Phi_sec',          d['Phi_sec']),
        ('freqs_sec',        d['freqs_sec']),
        ('T_full2sec',       d['T_full2sec']),
        ('K_sec',            d['K_sec']),
        ('M_sec',            d['M_sec']),
        ('C_sec',            d['C_sec']),
        ('T_f2s_col_norms',  d['T_f2s_col_norms']),
    ]:
        path = os.path.join(OUT, f'{name}.npy')
        np.save(path, arr)
        sz = os.path.getsize(path) / 1e6
        print(f"  {name}.npy  {str(arr.shape):<25}  {sz:.1f} MB")

    # Damping
    np.save(os.path.join(OUT, 'damping_alpha.npy'),    np.float64(d['alpha']))
    np.save(os.path.join(OUT, 'damping_beta.npy'),     np.float64(d['beta']))
    np.save(os.path.join(OUT, 'damping_beta_mis.npy'), np.float64(d['beta_mis']))
    np.save(os.path.join(OUT, 'damping_zeta.npy'),     np.float64(CONFIG['damping_ratio']))
    print(f"  damping_beta_mis = {d['beta_mis']:.6e}  (for mistuning C_mis)")
    print(f"  damping_zeta     = {CONFIG['damping_ratio']:.6f}  (exact modal ζ)")

    # Sparse projection basis
    sparse.save_npz(os.path.join(OUT, 'T_pccms.npz'), sparse.csr_matrix(d['T']))

    # Blade ROM index maps
    np.savez(os.path.join(OUT, 'blade_rom_indices.npz'),
             **{f'blade_{b}': d['bri'][b] for b in range(NB)})
    np.savez(os.path.join(OUT, 'blade_rom_indices_step4.npz'),
             **{f'blade_{b}': d['bri'][b] for b in range(NB)})
    np.savez(os.path.join(OUT, 'blade_tip_matrix_dofs.npz'),
             **{f'blade_{b}': np.array(d['btmd'][b]) for b in range(NB)})

    # Manifest
    manifest = {
        'step':              2,
        'version':           '10.0',
        'n_blades':          NB,
        'n_dof':             int(d['n_dof']),
        'n_rom':             int(d['n_rom']),
        'n_master':          int(d['n_master']),
        'n_fi':              int(d['n_fi']),
        'n_fi_compute':      int(d['n_fi_compute']),
        'n_fi_keep_min':     int(CONFIG['n_fi_keep_min']),
        'n_fi_auto_selected': True,
        'n_sec':             int(d['n_sec']),
        'freq_range_hz':     [float(d['freqs_rom'][0]),  float(d['freqs_rom'][-1])],
        'sec_freq_range_hz': [float(d['freqs_sec'][0]),  float(d['freqs_sec'][-1])],
        'n_1b_modes':        int((d['freqs_sec'] < 700).sum()),
        'T_f2s_col_norm_mean': float(d['T_f2s_col_norms'].mean()),
        'damping_alpha':     float(d['alpha']),
        'damping_beta':      float(d['beta']),
        'damping_beta_mis':  float(d['beta_mis']),
        'damping_ratio':     float(CONFIG['damping_ratio']),
        'damping_model':     'modal',
        'damping_note':      'C_sec=diag(2*zeta*omega_k); C_mis=C_sec+beta_mis*(K_mis-K_sec)',
        'material':          'Ti-alloy: E=96GPa, nu=0.36, rho=4620 kg/m3',
    }
    with open(os.path.join(OUT, 'rom_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print("  rom_manifest.json")
    print(f"\n  All outputs saved to: {OUT}")




# ═══════════════════════════════════════════════════════════════════
# 2E. VALIDATION FIGURES — publication-ready ROM diagnostics
# ------------------------------------------------------------------
# Self-contained figure suite added on top of the existing ROM build.
# It NEVER alters the ROM computation; it only reads arrays that are
# either already in the working dict `d` OR loadable from the saved
# .npy files in OUT — so it also runs standalone / with checkpoints.
#
# v10.0 CHANGE (per user request): ALL blisk-geometry / CAD renders are
# REMOVED from this script -- fig0a-d (CAD hero shots), fig3 (master-DOF
# overlay on the CAD blisk) and fig4 (tip-amplitude heatmap on the CAD
# blisk), together with every line of their supporting code (the
# B-rep/OpenCASCADE tessellation pipeline, render_cad_blisk, load_cad_mesh,
# make_cad_hero_figures, blade_tip_xyz_from_mesh, etc.) have been deleted,
# not just skipped. Step 2 now produces ONLY data/diagnostic figures:
# frequency correlation (1a), per-mode error (1b), MAC (2, redesigned --
# light canvas, black/blue colormap), FI-truncation convergence (6, new),
# and FRF (5). No 3D geometry is rendered anywhere in this file.
# ═══════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')                       # headless / file-only backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as _cm

# Optional preferred figure root (identical logic to Step 1 / Step 3).
FIG_ROOT = r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project'

# Standing 3/4 camera (matches the ANSYS view — blisk stands, not flat)
_ELEV, _AZIM = 22, -68

# ── matplotlib style (v9: matches Step 3's "ink-wave" look — solid cream
#    canvas, bold left-aligned titles, horizontal legends below the axes —
#    instead of the old transparent-PNG / default-matplotlib look) ─────────
plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':       10,
    'axes.labelsize':  11,
    'axes.titlesize':  11,
    'legend.fontsize': 9,
    'figure.dpi':      150,
    # No default x-padding: first/last data point should touch the axes
    # frame, not float in empty margin (2026-08-13, explicit user request).
    'axes.xmargin':    0.0,
    'axes.ymargin':    0.05,
})

# ── palette (identical to Step 3's _make_figures_core palette) ────────────
INK     = '#22303c'   # near-black foreground text / spines
CREAM   = '#ffffff'   # canvas
GRIDCOL = '#d9cfc0'   # faint grid on the cream
C_1B    = '#2f6f8f'   # 1B cluster  (deep teal-blue)
C_HF    = '#e07a3f'   # HF modes    (warm burnt orange)
C_ACC   = '#7a5195'   # tertiary accent (violet)
C_WARN  = '#c1443c'   # tolerance / warning red
C_OK    = '#3f8f6b'   # pass green
FADE    = '#b9c6cd'   # faded reference line grey-blue
_PALETTE = [C_1B, C_HF, C_ACC, C_OK, '#8c6d4a']

def _new_ax(figsize=(8.4, 4.6)):
    """A single-axes figure with the cream canvas + minimalist spines
    (byte-for-byte the same helper Step 3 uses, so Step 2 figures match)."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.1)
        ax.spines[s].set_position(('outward', 6))
    ax.tick_params(colors=INK, labelcolor=INK, length=4, width=1.0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, which='major', color=GRIDCOL, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    return fig, ax


def _new_axes_row(n, figsize=(11, 4.6)):
    """n side-by-side cream axes sharing one figure (for multi-panel figs)."""
    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.patch.set_facecolor(CREAM)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor(CREAM)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(INK)
            ax.spines[s].set_linewidth(1.1)
            ax.spines[s].set_position(('outward', 6))
        ax.tick_params(colors=INK, labelcolor=INK, length=4, width=1.0)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
        ax.grid(True, which='major', color=GRIDCOL, lw=0.7, alpha=0.9)
        ax.set_axisbelow(True)
    return fig, axes


def _title(ax, text, sub=None):
    """Left-aligned two-tier title: bold headline (+ optional grey subline).
    3D-axes fix (2026-08-19): Axes3D.text() takes (x,y,z,s,...), not
    (x,y,s,...) -- use text2D for axes-fraction overlay text on 3D axes."""
    ax.set_title(text, loc='left', fontsize=13, fontweight='bold',
                 color=INK, pad=(20 if sub else 10))
    if sub:
        text_fn = ax.text2D if hasattr(ax, 'zaxis') else ax.text
        text_fn(0.0, 1.015, sub, transform=ax.transAxes, ha='left',
                va='bottom', fontsize=9.5, color='#6b7a83')


def _legend_below(ax, handles=None, labels=None, ncol=None, y=-0.18):
    """Horizontal legend anchored under the x-axis. Returns the legend."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if labels is None:
        labels = [h.get_label() for h in handles]
    if ncol is None:
        ncol = min(len(handles), 4)
    leg = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                    fontsize=9.5, handlelength=1.6, columnspacing=1.8,
                    borderaxespad=0.0)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def _wave_stems(ax, x, y, color, baseline=None, lw=1.4, ms=5.5,
                fill_alpha=0.14, log=False, marker='o'):
    """A ridgeline-inspired glyph: a faint filled sheet from a baseline up to
    an envelope, thin vertical stems, and round markers at the tips. Ported
    from Step 3 so per-mode bar/stem figures share the same visual language."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if baseline is None:
        baseline = (np.nanmin(y[y > 0]) * 0.5) if log else 0.0
    ax.fill_between(x, baseline, y, color=color, alpha=fill_alpha,
                    linewidth=0, zorder=1)
    ax.vlines(x, baseline, y, color=color, lw=lw, alpha=0.55, zorder=2)
    ax.plot(x, y, marker, ms=ms, mfc=color, mec=CREAM, mew=0.8,
            linestyle='none', zorder=3)


def _resolve_figs_dir(step_name):
    root = FIG_ROOT if (FIG_ROOT and os.path.isdir(FIG_ROOT)) else OUT
    figs = os.path.join(root, 'figures', step_name)
    os.makedirs(figs, exist_ok=True)
    return figs

def _savefig(fig, figs_dir, name, dpi=None):
    """Opaque cream-canvas save (matches Step 3's _savefig). Replaces the old
    _save_transparent: Step 2 figures are no longer transparent PNGs — they
    use the same solid white canvas, DPI, and bbox handling as Step 3."""
    path = os.path.join(figs_dir, name + '.png')
    fig.savefig(path, dpi=(dpi or 150), bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Validation-figure data helpers  (robust to checkpoints / standalone)
# ═══════════════════════════════════════════════════════════════════
def _G(d, key, fname=None):
    """Return d[key] if present, else load OUT/(fname or key.npy)."""
    if d is not None and key in d and d[key] is not None:
        return d[key]
    fp = os.path.join(OUT, fname if fname else f'{key}.npy')
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    return np.load(fp, allow_pickle=True)


def _get_freqs_full(d):
    if d is not None and 'freqs_full' in d and d['freqs_full'] is not None:
        return np.asarray(d['freqs_full'])
    for fn in ('freqs_full.npy', 'frequencies_all.npy'):
        fp = os.path.join(OUT, fn)
        if os.path.exists(fp):
            return np.load(fp)
    raise FileNotFoundError('freqs_full.npy / frequencies_all.npy')


def _rom_eig(d):
    """Return (freqs_rom, eigvecs_rom) — reuse d if available, else recompute
    from the saved K_r / M_r (cheap 770x770 dense eigensolve)."""
    if d is not None and 'eigvecs_rom' in d and d['eigvecs_rom'] is not None:
        return _G(d, 'freqs_rom'), d['eigvecs_rom']
    K_r = _G(d, 'K_r'); M_r = _G(d, 'M_r')
    ev, evec = eigh(K_r, M_r)
    return np.sqrt(np.abs(ev)) / (2 * np.pi), evec


def _dof_map():
    return np.load(os.path.join(OUT, 'dof_mapping.npy'))   # (n_eq, 2)[node, dir]


def _full_modes_at_dofs(dof_eq):
    """Full-order mode-shape values at the given solver DOF (equation)
    indices → array (len(dof_eq), n_full_modes)."""
    Phi  = np.load(os.path.join(OUT, 'Phi_all_modes.npy'))      # (n,3,nm)
    nnum = np.load(os.path.join(OUT, 'mode_node_ids.npy'))
    dmap = _dof_map()
    id2row = {int(n): i for i, n in enumerate(nnum)}
    nm  = Phi.shape[2]
    out = np.zeros((len(dof_eq), nm), float)
    for r, m in enumerate(dof_eq):
        node, di = int(dmap[m, 0]), int(dmap[m, 1])
        row = id2row.get(node)
        if row is not None and 0 <= di < 3:
            out[r] = Phi[row, di, :]
    return out


def _mac_matrix(A, B):
    """Modal Assurance Criterion between column-mode sets A (n,na), B (n,nb)."""
    An = np.einsum('ij,ij->j', A, A)
    Bn = np.einsum('ij,ij->j', B, B)
    C  = A.T @ B
    den = np.outer(An, Bn)
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(den > 0, C ** 2 / den, 0.0)
    return M


def _node_dir_to_eq():
    """{node_id: [eq_ux, eq_uy, eq_uz]} (missing dir -> -1) from dof_mapping."""
    dmap = _dof_map()
    nd = {}
    for eq in range(dmap.shape[0]):
        node, di = int(dmap[eq, 0]), int(dmap[eq, 1])
        if 0 <= di < 3:
            nd.setdefault(node, [-1, -1, -1])[di] = eq
    return nd


def _tip_node_ids():
    """All blade-tip (master) node IDs, unioned across blades."""
    ids = []
    got = False
    for b in range(NB):
        fp = os.path.join(OUT, f'bladetip_blade{b}_nodes.npy')
        if os.path.exists(fp):
            got = True
            ids.extend(int(x) for x in np.load(fp))
    if not got:
        # fall back to master_dofs -> node ids
        try:
            md = _G(None, 'master_dofs')
            dmap = _dof_map()
            ids = [int(dmap[m, 0]) for m in md]
        except Exception:
            pass
    return np.unique(ids).astype(int)


# ═══════════════════════════════════════════════════════════════════
# 2E. VALIDATION FIGURES — main entry
# ═══════════════════════════════════════════════════════════════════
def make_step2_figures(d=None):
    hdr("2E: ROM VALIDATION FIGURES (ink-wave restyle, figures/step2/)")
    figs = _resolve_figs_dir('step2')
    saved_names = []

    # ---- gather core arrays (dict or disk) ---------------------------
    try:
        freqs_full = _get_freqs_full(d)
        freqs_rom, evec_rom = _rom_eig(d)
        master = np.asarray(_G(d, 'master_dofs')).astype(int)
    except FileNotFoundError as e:
        print(f"  Skipping validation figures — missing input: {e}")
        return
    n_m  = len(master)
    # NOTE: blade tip-DOF counts are NOT assumed uniform (whole-disk mesh,
    # not a stamped cyclic sector) -- load blade 0's TRUE master-DOF block
    # from blade_rom_indices.npz rather than slicing with n_m // NB, which
    # would silently grab the wrong DOFs once per-blade counts differ.
    try:
        bri0 = np.load(os.path.join(OUT, 'blade_rom_indices.npz'))['blade_0']
        ntp = len(bri0)
    except Exception:
        ntp = n_m // NB   # fallback only if the per-blade index file is missing
        print(f"  NOTE: blade_rom_indices.npz not found -- falling back to "
              f"average master-DOFs/blade ({ntp}); this fallback assumes "
              f"uniform blade size and may be wrong for a non-symmetric mesh.")
    zeta = CONFIG['damping_ratio']
    zp   = os.path.join(OUT, 'damping_zeta.npy')
    if os.path.exists(zp):
        try: zeta = float(np.load(zp))
        except Exception: pass

    # ── FIG 1a: full-order vs ROM frequency correlation ──────────────
    N = int(min(len(freqs_full), len(freqs_rom), 50))
    ff, fr = freqs_full[:N], freqs_rom[:N]
    rel = np.abs(fr - ff) / (ff + 1e-12) * 100.0
    is_1b = np.arange(N) < NB
    n_1b_here = int(is_1b.sum())

    fig, ax = _new_ax()
    lo, hi = min(ff.min(), fr.min()), max(ff.max(), fr.max())
    pad = (hi - lo) * 0.04
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '-', color=INK,
            lw=1.3, alpha=0.55, zorder=2)
    for mask, col in [(is_1b, C_1B), (~is_1b, C_HF)]:
        if mask.any():
            ax.scatter(ff[mask], fr[mask], s=42, color=col, edgecolors=CREAM,
                       linewidths=0.8, zorder=4)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel('Full-order frequency  (Hz)')
    ax.set_ylabel('ROM frequency  (Hz)')
    _title(ax, 'Frequency correlation',
           f'first {N} modes; PC-CMS ROM vs. full-order 181,473-DOF solve')
    import matplotlib.lines as mlines
    h1 = mlines.Line2D([], [], color=C_1B, marker='o', linestyle='none',
                       ms=7, mec=CREAM, mew=0.8, label=f'1B cluster ({n_1b_here} modes)')
    h2 = mlines.Line2D([], [], color=C_HF, marker='o', linestyle='none',
                       ms=7, mec=CREAM, mew=0.8, label=f'HF modes ({N - n_1b_here} modes)')
    ref = mlines.Line2D([], [], color=INK, lw=1.3, alpha=0.6, label='1:1 reference')
    _legend_below(ax, [h1, h2, ref], None, ncol=3)
    fig.tight_layout()
    _savefig(fig, figs, 'step2_fig1a_freq_correlation')
    saved_names.append('step2_fig1a_freq_correlation')

    # ── FIG 1b: per-mode relative error (wave-stem glyph) ─────────────
    fig, ax = _new_ax()
    xax = np.arange(N)
    for mask, col in [(is_1b, C_1B), (~is_1b, C_HF)]:
        if mask.any():
            _wave_stems(ax, xax[mask], rel[mask], col)
    ax.axhline(1.0, ls=(0, (4, 2)), color=C_WARN, lw=1.4, zorder=3)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Relative error  (%)')
    _title(ax, 'Per-mode frequency error',
           f'mean {rel.mean():.3f}%, max {rel.max():.3f}%  (first {N} modes)')
    h1 = mlines.Line2D([], [], color=C_1B, marker='o', linestyle='none',
                       ms=7, mec=CREAM, mew=0.8, label=f'1B cluster ({n_1b_here} modes)')
    h2 = mlines.Line2D([], [], color=C_HF, marker='o', linestyle='none',
                       ms=7, mec=CREAM, mew=0.8, label=f'HF modes ({N - n_1b_here} modes)')
    htol = mlines.Line2D([], [], color=C_WARN, lw=1.4, ls=(0, (4, 2)),
                        label='1% tolerance')
    _legend_below(ax, [h1, h2, htol], None, ncol=3)
    fig.tight_layout()
    _savefig(fig, figs, 'step2_fig1b_freq_error')
    saved_names.append('step2_fig1b_freq_error')

    # ── FIG 2: MAC (interface / blade-tip master DOFs) — v10.1 REDESIGN ──
    # v10.1: figure now uses the SAME _new_ax()/_title()/_savefig() helpers as
    # every other Step 2/3 figure (white CREAM canvas, ink spines, left-
    # aligned two-tier title) instead of a bespoke '#fffef2' pale-yellow
    # canvas + boxed text annotation that didn't match the rest of the deck.
    # The degenerate-pair "subspace MAC" numbers are still computed and
    # printed to the console (they explain a low diagonal MAC on a mesh with
    # frequency-degenerate nodal-diameter pairs), but are no longer stamped
    # as an on-figure text box -- they belong in the log/report, not baked
    # into the PNG.
    try:
        Full_m = _full_modes_at_dofs(master)          # (n_m, n_full)
        ROM_m  = evec_rom[:n_m, :]                     # (n_m, n_rom)
        Nm = int(min(Full_m.shape[1], ROM_m.shape[1], N))
        MAC = _mac_matrix(Full_m[:, :Nm], ROM_m[:, :Nm])
        diag = np.diag(MAC)

        # -- degenerate-pair / subspace MAC (explains a low min diagonal) --
        ff_here = freqs_full[:Nm]
        gap_rel = np.diff(ff_here) / (ff_here[:-1] + 1e-12)
        clusters, cur = [], [0]
        for i, g in enumerate(gap_rel):
            if g < 0.005:      # <0.5% relative spacing -> same degenerate cluster
                cur.append(i + 1)
            else:
                clusters.append(cur); cur = [i + 1]
        clusters.append(cur)
        sub_rows = []
        for cl in clusters:
            if len(cl) < 2:
                continue
            Fc = Full_m[:, cl]; Rc = ROM_m[:, cl]
            Qf, _ = np.linalg.qr(Fc); Qr, _ = np.linalg.qr(Rc)
            s = np.linalg.svd(Qf.T @ Qr, compute_uv=False)
            sub_mac = float(np.mean(np.clip(s, -1, 1) ** 2))
            sub_rows.append((cl, sub_mac))
        n_clusters = len(sub_rows)
        sub_mac_min = min((r[1] for r in sub_rows), default=float('nan'))
        sub_mac_mean = (float(np.mean([r[1] for r in sub_rows]))
                         if sub_rows else float('nan'))
        print(f"  Degenerate-pair check: {n_clusters} near-degenerate "
              f"cluster(s) found (<0.5% freq. spacing)")
        for cl, sm in sub_rows:
            print(f"    modes {cl[0]}-{cl[-1]}: pointwise MAC diag = "
                  f"{np.round(diag[cl], 3)}  ->  subspace MAC = {sm:.4f}")

        # Degeneracy-aware validation gate: non-degenerate modes are judged
        # on their pointwise MAC diagonal; degenerate clusters (where the
        # ROM/full solvers can pick an arbitrary rotation within the shared
        # eigenspace) are judged on the rotation-invariant subspace MAC.
        clustered_idx = set(i for cl in clusters if len(cl) >= 2 for i in cl)
        effective_macs = [float(diag[i]) for i in range(Nm) if i not in clustered_idx]
        effective_macs += [sm for _, sm in sub_rows]
        mac_eff_min = min(effective_macs) if effective_macs else float('nan')
        _record_check(
            f"MAC agreement (degeneracy-aware) >= {CONFIG['mac_min_threshold']}",
            mac_eff_min >= CONFIG['mac_min_threshold'],
            f"effective min MAC = {mac_eff_min:.4f} across {Nm} modes, "
            f"{n_clusters} degenerate cluster(s)")

        # 2D BAR CHART (2026-08-29 REDESIGN, replacing the 3D surface added
        # 2026-08-19): the 3D surface was hard to read on the page and in a
        # reviewer's printout -- a reader has to mentally rotate the plot to
        # tell whether a given bar is at 0.9 or 0.6. The MAC matrix is
        # near-diagonal by construction (each ROM mode should match ONE
        # full-order mode), so the only number that actually matters per
        # mode is diag(MAC) -- a plain bar chart of that diagonal, one bar
        # per mode index, shows exactly the same information (which modes
        # match well, which don't) in a flat, unambiguous 2D read. Degenerate
        # near-pairs (where pointwise MAC is misleadingly low because the
        # ROM's eigensolver picked an arbitrary rotation within a shared
        # 2-D eigenspace) are marked explicitly with their own subspace-MAC
        # value printed above the bar pair, rather than left for the reader
        # to puzzle out from an off-diagonal bump.
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        fig.patch.set_facecolor(CREAM)
        ax.set_facecolor(CREAM)
        mode_idx = np.arange(len(diag))
        bar_colors = [INK if d >= CONFIG['mac_min_threshold'] else C_WARN for d in diag]
        ax.bar(mode_idx, diag, color=bar_colors, edgecolor=INK, linewidth=0.4, width=0.75)
        ax.axhline(CONFIG['mac_min_threshold'], color=C_1B, linestyle='--', linewidth=1.1,
                   label=f"validation gate ({CONFIG['mac_min_threshold']:.2f})")
        # Mark degenerate-pair clusters with a thin bracket (rotation-
        # invariant subspace MAC explains a low pointwise MAC inside a
        # cluster, since the ROM/full solvers can pick an arbitrary
        # rotation within the shared eigenspace). A per-cluster text stamp
        # was tried first but with 17 near-degenerate clusters packed into
        # 48 modes the labels collided into an unreadable smear (2026-08-29
        # fix) -- one consolidated annotation carries the same information
        # without the overlap, since every cluster's own console-printed
        # subspace MAC (see log) turns out to be numerically identical here.
        for cl, sm in sub_rows:
            y_ann = max(diag[cl].max(), CONFIG['mac_min_threshold']) + 0.03
            ax.plot([cl[0], cl[-1]], [y_ann, y_ann], color=C_1B, linewidth=1.4)
        if sub_rows:
            sm_vals = [sm for _, sm in sub_rows]
            sm_lo, sm_hi = min(sm_vals), max(sm_vals)
            sm_label = (f'subspace MAC = {sm_lo:.3f}' if round(sm_lo, 3) == round(sm_hi, 3)
                        else f'subspace MAC = {sm_lo:.3f}-{sm_hi:.3f}')
            # Plain data-coordinate placement (NOT axes-fraction via
            # get_xaxis_transform, which put this text 10% of the axes
            # HEIGHT above the top edge -- close enough to collide with the
            # title sitting just outside the axes, 2026-08-29 fix): y=1.30
            # sits comfortably inside the data ylim below, well clear of
            # both the degenerate-pair brackets (~1.03-1.05) and the title.
            ax.text((len(diag) - 1) / 2, 1.30, f'brackets: {len(sub_rows)} near-degenerate mode '
                    f'clusters (<0.5% freq. spacing), {sm_label}',
                    ha='center', va='bottom', color=C_1B, fontsize=8.5)
        ax.set_xlabel('Mode index')
        ax.set_ylabel('MAC')
        ax.set_ylim(0, 1.45)
        ax.set_xlim(-0.6, len(diag) - 0.4)
        ax.legend(loc='lower right', framealpha=0.9)
        _title(ax, 'Modal Assurance Criterion',
               'full-order vs. ROM mode shapes at the blade-tip interface DOFs, per mode')
        fig.tight_layout()
        _savefig(fig, figs, 'step2_fig2_mac')
        saved_names.append('step2_fig2_mac')
        print(f"  MAC diagonal: mean={diag.mean():.4f}, min={diag.min():.4f}  "
              f"(low min is expected/explained by degenerate pairs above, "
              f"see subspace MAC)")
    except FileNotFoundError as e:
        print(f"  [fig2 MAC] skipped — need full mode shapes: {e}")

    # NOTE (v10.0): the CAD-geometry figures previously here (fig0a-d hero
    # renders, fig3 master-DOF-on-CAD overlay, fig4 tip-amplitude-on-CAD
    # heatmap) have been REMOVED at the user's request, along with all of
    # their supporting code (the B-rep/OpenCASCADE tessellation pipeline,
    # render_cad_blisk, load_cad_mesh, make_cad_hero_figures, etc.). Step 2
    # now only produces data/diagnostic figures (frequency correlation and
    # error, MAC, FI-truncation convergence, FRF) -- no blisk geometry
    # renders anywhere in this script.

    # ── FIG 6: fixed-interface truncation convergence (NEW, v10.0) ───
    # Direct visual evidence for the n_fi=50->70(+) decision made in 2B:
    # mean/max frequency error vs. how many FI modes are retained.
    try:
        table = d.get('fi_sweep_table') if d is not None else None
        if table is None:
            jp = os.path.join(OUT, 'step2_fi_convergence.json')
            if os.path.exists(jp):
                with open(jp) as f:
                    table = json.load(f)['sweep_table']
        if table:
            ps      = np.array([r['n_fi_keep'] for r in table])
            e_mean  = np.array([r['err_mean_pct'] for r in table])
            e_max   = np.array([r['err_max_pct'] for r in table])
            order   = np.argsort(ps)
            ps, e_mean, e_max = ps[order], e_mean[order], e_max[order]
            n_fi_sel = int(d['n_fi']) if (d is not None and 'n_fi' in d) else int(ps[np.argmin(e_max)])

            fig, ax = _new_ax(figsize=(8.4, 4.9))
            ax.plot(ps, e_mean, '-o', color=C_1B, lw=1.8, ms=6,
                    mec=CREAM, mew=0.8, label='mean error (first modes)', zorder=4)
            ax.plot(ps, e_max, '-o', color=C_HF, lw=1.8, ms=6,
                    mec=CREAM, mew=0.8, label='max error (first modes)', zorder=4)
            ax.axhline(CONFIG['fi_tol_mean_pct'], color=C_1B, ls=(0, (4, 2)),
                       lw=1.2, alpha=0.7, zorder=2)
            ax.axhline(CONFIG['fi_tol_max_pct'], color=C_HF, ls=(0, (4, 2)),
                       lw=1.2, alpha=0.7, zorder=2)
            ax.axvline(n_fi_sel, color=INK, lw=1.3, alpha=0.55, zorder=1)
            ax.text(n_fi_sel, ax.get_ylim()[1] * 0.92, f'  selected n_fi={n_fi_sel}',
                    fontsize=9, color=INK, va='top', ha='left')
            ax.set_xlabel('Fixed-interface modes retained (n_fi)')
            ax.set_ylabel('Frequency error vs. full order  (%)')
            _title(ax, 'ROM fixed-interface truncation convergence',
                   'block-exact re-projection at each candidate n_fi (no re-solve)')
            _legend_below(ax, ncol=2)
            fig.tight_layout()
            _savefig(fig, figs, 'step2_fig6_fi_convergence')
            saved_names.append('step2_fig6_fi_convergence')
        else:
            print("  [fig6 FI convergence] skipped — no sweep table available "
                  "(run build_pccms_rom in this session, or check "
                  "step2_fi_convergence.json).")
    except Exception as e:
        print(f"  [fig6 FI convergence] skipped — {e}")

    # ── FIG 5: FRF dynamic validation (full modal vs ROM) ────────────
    try:
        Full_m = Full_m if 'Full_m' in dir() else _full_modes_at_dofs(master)
    except FileNotFoundError as e:
        print(f"  [fig5 FRF] skipped — need full mode shapes: {e}")
        Full_m = None
    if Full_m is not None:
        # choose the most responsive blade-0 tip DOF (max |phi_full| in mode 0)
        # ntp above is blade 0's TRUE master-DOF count (from blade_rom_indices
        # .npz), so this slice is blade 0's actual block even when per-blade
        # counts are not uniform across the disk.
        p0 = int(np.argmax(np.abs(Full_m[:ntp, 0]))) if ntp > 0 else 0
        w  = 2 * np.pi * np.linspace(250.0, 500.0, 2000)     # 1B band
        # full-order modal FRF (mass-normalised modes assumed, ANSYS default)
        wk_f = 2 * np.pi * freqs_full
        phi_f = Full_m[p0, :]                                 # (n_full,)
        Hf = np.zeros_like(w, dtype=complex)
        for k in range(len(wk_f)):
            Hf += (phi_f[k] * phi_f[k]) / (wk_f[k] ** 2 - w ** 2
                                           + 2j * zeta * wk_f[k] * w)
        # ROM modal FRF (mass-normalised in full space by construction)
        wk_r = 2 * np.pi * freqs_rom
        psi_r = evec_rom[p0, :]                               # value at DOF p0
        Hr = np.zeros_like(w, dtype=complex)
        for k in range(len(wk_r)):
            Hr += (psi_r[k] * psi_r[k]) / (wk_r[k] ** 2 - w ** 2
                                           + 2j * zeta * wk_r[k] * w)
        fdB = 20 * np.log10(np.abs(Hf) + 1e-30)
        rdB = 20 * np.log10(np.abs(Hr) + 1e-30)
        # 2026-08-31 RESTYLE (explicit user request): this figure predates
        # the project-wide plot_style.py house look ("ink-wave" palette,
        # warm-tan gridlines, Times New Roman) and was still using step2's
        # own older local _new_ax/_title/_legend_below helpers, which read
        # as visibly inconsistent with every other figure in the paper.
        # Switched to plot_style directly, and both axes are now bounded
        # tightly to the actual plotted data (no autoscaled empty margin
        # above/below or beyond the 1B band).
        import plot_style as _ps
        _ps.apply_style()
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        band = (w / (2 * np.pi) >= 250) & (w / (2 * np.pi) <= 500)
        ax.plot(w / (2 * np.pi), fdB, color=_ps.C_1B, lw=2.2,
                label='Full-order (modal superposition)', zorder=3)
        ax.plot(w / (2 * np.pi), rdB, color=_ps.C_HF, lw=1.6, ls=(0, (4, 2)),
                label='ROM (secondary modal basis)', zorder=4)
        ax.set_xlabel('Frequency  (Hz)')
        ax.set_ylabel('Receptance  |H|  (dB)')
        # Title/subtitle removed from the image itself (2026-08-31, explicit
        # user request) -- the docx caption carries this text instead.
        ax.set_xlim(250, 500)
        y_lo = min(fdB[band].min(), rdB[band].min())
        y_hi = max(fdB[band].max(), rdB[band].max())
        pad = (y_hi - y_lo) * 0.05
        ax.set_ylim(y_lo - pad, y_hi + pad)
        _ps.legend_inside(ax, loc='upper right')
        fig.tight_layout()
        _ps.savefig_pub(fig, figs, 'step2_fig5_frf')
        saved_names.append('step2_fig5_frf')
        band = (w / (2 * np.pi) >= 250) & (w / (2 * np.pi) <= 500)
        rmsd = np.sqrt(np.mean((fdB[band] - rdB[band]) ** 2))
        print(f"  FRF full-vs-ROM RMS diff over 250-500 Hz: {rmsd:.3f} dB")

    print(f"  All {len(saved_names)} Step 2 validation figures saved to: {figs}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    _log_path = os.path.join(_HERE, 'Step2.txt')
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    t_start = time.time()
    hdr(f"STEP 2 v10.0: PC-CMS ROM + SECONDARY BASIS — {NB}-BLADE BLISK  "
        f"(auto FI-truncation sweep, CAD figures removed)")

    d = load_step1()
    d = build_pccms_rom(d)
    d = build_secondary_basis(d)
    save_rom(d)

    # Persist the FI-truncation convergence table on its own (small, human-
    # readable) JSON so fig6 / a reviewer can inspect it without re-running
    # the whole ROM build.
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'step2_fi_convergence.json'), 'w') as f:
        json.dump({
            'n_fi_compute':  int(d['n_fi_compute']),
            'n_fi_selected': int(d['n_fi']),
            'n_fi_keep_min': int(CONFIG['n_fi_keep_min']),
            'fi_tol_mean_pct': CONFIG['fi_tol_mean_pct'],
            'fi_tol_max_pct':  CONFIG['fi_tol_max_pct'],
            'sweep_table':   d['fi_sweep_table'],
        }, f, indent=2)
    print("  step2_fi_convergence.json")

    try:
        make_step2_figures(d)
    except Exception as _e:
        print(f"  [validation figures] skipped due to: {_e}")

    passed = print_validation_summary()

    elapsed = time.time() - t_start
    hdr("STEP 2 COMPLETE")
    print(f"  {d['n_dof']:,} DOFs  →  {d['n_rom']} ROM DOFs  →  {d['n_sec']} secondary DOFs")
    print(f"  FI modes: {d['n_fi']} kept of {d['n_fi_compute']} computed "
          f"(auto-selected; floor={CONFIG['n_fi_keep_min']})")
    nc = min(len(d['freqs_rom']), len(d['freqs_full']), 50)
    err_first = (np.abs(d['freqs_rom'][:nc] - d['freqs_full'][:nc])
                 / (d['freqs_full'][:nc] + 1e-12) * 100.0)
    print(f"  Freq error (first {nc}): mean={err_first.mean():.4f}%, "
          f"max={err_first.max():.4f}%")
    print(f"  Damping: MODAL (ζ={CONFIG['damping_ratio']} for ALL modes)")
    print(f"  β_mis = {d['beta_mis']:.6e}")
    print(f"  Rayleigh α={d['alpha']:.6e}, β={d['beta']:.6e} (reference only)")
    print(f"  Validation: {'PASSED' if passed else 'FAILED — see STEP 2 VALIDATION SUMMARY above'}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\nNEXT: python step3.py  (K3 cubic stiffness tensor)")
    print(f"\nLog saved: {_log_path}")
    sys.stdout = sys.__stdout__
    _log_file.close()