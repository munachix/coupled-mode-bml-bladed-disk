"""HF-band (modes 24-69) nonlinear ROM campaign (2026-08-21).

Real ANSYS work to replace Step 4's extrapolated (hardening_ratio-based)
diagonal K3 for the 46 high-frequency secondary modes with real measured
data, and to replace the diagonal-only (SDOF) nonlinear model with real
cross-mode coupling wherever the modes are actually near-degenerate --
exactly the same method already validated on the 1B cluster (modes 0-23,
Section 9j/9k of PROJECT_STATUS.md), just extended to the rest of the
70-mode secondary basis.

Real frequency-gap scan (2026-08-21, same method as the 1B-cluster scan:
each mode's own real half-power bandwidth ~2*zeta*f, real zeta=0.002,
against real freqs_sec) found the HF band is NOT mostly isolated, contrary
to the working assumption up to this point:
  - 2 genuinely isolated singles: modes 24, 37
  - 10 clean isolated near-degenerate pairs: (25,26),(27,28),(29,30),
    (31,32),(33,34),(35,36),(38,39),(40,41),(42,43),(44,45)
  - 1 small chain: 46-48 (3 modes, 2 adjacent-pair identifications)
  - 1 large chain: 49-69 (21 modes, 20 adjacent-pair identifications)
44 of 46 HF modes (95.7%) need real cross-mode coupling data -- only 2 are
genuinely isolated.

Total real ANSYS work: 46 independent K3 identifications (ALL HF modes,
including the 2 isolated ones -- replaces their extrapolated placeholder
with real measured data even though they stay diagonal) + 32 cross-mode
coupling identifications (10 clean pairs + 2 small-chain pairs + 20
large-chain pairs).

Resumable by design, same pattern as _full_multimode_campaign.py: every
item checks for its own already-saved output file first and skips if
present, so a crash/restart doesn't repeat finished ANSYS work.

Run with:
  C:\\Users\\Ronin\\AppData\\Local\\Programs\\Python\\Python312\\python.exe _hf_multimode_campaign.py
"""
import sys, time, os, subprocess
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

# All 46 HF modes -- independent K3 for every one (including the 2 isolated
# singles: real measured diagonal K3 still replaces their extrapolated
# placeholder, even though they correctly stay un-coupled).
HF_MODES = list(range(24, 70))

# Real, measured topology from the 2026-08-21 gap scan.
CLEAN_PAIRS = [(25, 26), (27, 28), (29, 30), (31, 32), (33, 34),
               (35, 36), (38, 39), (40, 41), (42, 43), (44, 45)]
CHAIN_SMALL_PAIRS = [(46, 47), (47, 48)]
CHAIN_LARGE_PAIRS = [(i, i + 1) for i in range(49, 69)]   # (49,50) ... (68,69)
ALL_HF_PAIRS = CLEAN_PAIRS + CHAIN_SMALL_PAIRS + CHAIN_LARGE_PAIRS

assert len(HF_MODES) == 46
assert len(ALL_HF_PAIRS) == 32


def _cleanup_stale_ansys():
    """Same reasoning as the 1B campaign (Section 8a): mapdl.exit() doesn't
    reliably kill the OS process on this machine, so leftover batch workers
    can accumulate across ~78 sequential launches and starve the next one."""
    try:
        out = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            low = line.lower()
            if 'ansys' in low and '.exe' in low:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    pid = parts[1]
                    subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True, timeout=10)
    except Exception as e:
        print(f"  (cleanup skipped: {e})", flush=True)


def run_hf_k3():
    print(f"\n{'='*70}\nPHASE 1: HF independent static K3, {len(HF_MODES)} modes "
          f"({HF_MODES[0]}-{HF_MODES[-1]})\n{'='*70}", flush=True)
    for m in HF_MODES:
        dst = os.path.join(s9.OUT, f'case3_k3_identification_mode{m}.npz')
        if os.path.exists(dst):
            print(f"  mode {m}: already done, skipping ({dst})", flush=True)
            continue
        t0 = time.time()
        print(f"\n--- mode {m} ---", flush=True)
        _cleanup_stale_ansys()
        try:
            s9.run_case3_identification(mode_index=m, amplitudes=(0.02, 0.05, 0.08, 0.11))
        except Exception as e:
            print(f"  MODE {m} FAILED: {e}", flush=True)
            continue
        src = os.path.join(s9.OUT, 'case3_k3_identification.npz')
        if os.path.exists(src):
            os.replace(src, dst)
        elapsed = time.time() - t0
        print(f"  mode {m} done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


def run_hf_cross_pairs():
    print(f"\n{'='*70}\nPHASE 2: HF cross-mode coupling, {len(ALL_HF_PAIRS)} pairs\n{'='*70}", flush=True)
    for (m0, m1) in ALL_HF_PAIRS:
        dst = os.path.join(s9.OUT, f'case3_cross_k3_modes{m0}{m1}.npz')
        if os.path.exists(dst):
            print(f"  pair ({m0},{m1}): already done, skipping ({dst})", flush=True)
            continue
        t0 = time.time()
        print(f"\n--- pair ({m0},{m1}) ---", flush=True)
        _cleanup_stale_ansys()
        try:
            s9.run_case3_cross_identification(mode_pair=(m0, m1))
        except Exception as e:
            print(f"  PAIR ({m0},{m1}) FAILED: {e}", flush=True)
            continue
        elapsed = time.time() - t0
        print(f"  pair ({m0},{m1}) done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == '__main__':
    t_start = time.time()
    print(f"HF campaign starting: {len(HF_MODES)} K3 IDs + {len(ALL_HF_PAIRS)} cross-coupling IDs "
          f"= {len(HF_MODES) + len(ALL_HF_PAIRS)} total real ANSYS solves", flush=True)
    run_hf_k3()
    run_hf_cross_pairs()
    total = time.time() - t_start
    print(f"\n{'='*70}\nHF CAMPAIGN COMPLETE in {total:.1f}s ({total/60:.1f} min = {total/3600:.2f} hr)\n{'='*70}", flush=True)
