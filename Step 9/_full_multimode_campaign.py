"""Full 24-mode nonlinear ROM campaign (2026-08-13).

Real ANSYS work needed to give BPINN genuine, measured coverage across
all 24 1B-cluster modes, per the real frequency-gap scan done this
session (not assumed):
  - 5 clean isolated near-degenerate pairs: (0,1) [DONE], (3,4),(5,6),
    (7,8),(9,10)
  - 1 genuinely isolated single: mode 2
  - a 13-mode continuously-overlapping chain, modes 11-23, coupled via
    12 adjacent-pair identifications: (11,12)...(22,23)
  - independent (diagonal) K3 for every mode not yet measured (modes
    0-3 already done in Section 9i; this adds 4-23)

Resumable by design: every item checks for its own already-saved output
file first and skips if present, so a crash/restart doesn't repeat
finished ANSYS work (the exact lesson from this project's earlier
overnight-run crash, Section 9f/PROJECT_STATUS.md).

Run with the anaconda (PyMAPDL) interpreter:
  C:\\Users\\Ronin\\anaconda3\\python.exe _full_multimode_campaign.py
"""
import sys, time, os, subprocess
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
import step9 as s9

INDEPENDENT_MODES = list(range(4, 24))   # 0-3 already measured (Section 9i)
CLEAN_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10)]
CHAIN_PAIRS = [(i, i + 1) for i in range(11, 23)]   # (11,12) ... (22,23)
ALL_NEW_PAIRS = CLEAN_PAIRS + CHAIN_PAIRS


def _cleanup_stale_ansys():
    """mapdl.exit() doesn't reliably kill the OS process on this machine
    (Section 8a) -- across ~36 sequential launches, leftover batch
    workers would otherwise accumulate and can starve/fail the next
    launch. Kill any lingering ANSYS*.exe / ansyscl.exe between items."""
    try:
        out = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            low = line.lower()
            if 'ansys' in low and ('.exe' in low):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    pid = parts[1]
                    subprocess.run(['taskkill', '/PID', pid, '/F'], capture_output=True, timeout=10)
    except Exception as e:
        print(f"  (cleanup skipped: {e})", flush=True)


def run_independent_k3():
    print(f"\n{'='*70}\nPHASE 1: independent static K3, modes {INDEPENDENT_MODES}\n{'='*70}", flush=True)
    for m in INDEPENDENT_MODES:
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


def run_cross_pairs():
    print(f"\n{'='*70}\nPHASE 2: cross-mode coupling, {len(ALL_NEW_PAIRS)} pairs\n{'='*70}", flush=True)
    for (m0, m1) in ALL_NEW_PAIRS:
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
    run_independent_k3()
    run_cross_pairs()
    total = time.time() - t_start
    print(f"\n{'='*70}\nFULL CAMPAIGN COMPLETE in {total:.1f}s ({total/60:.1f} min = {total/3600:.2f} hr)\n{'='*70}", flush=True)
