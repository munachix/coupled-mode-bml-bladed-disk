# -*- coding: utf-8 -*-
"""
Summary of every real finite-element damage-injection case (Section 3.6).

Round 1 (ten cases) established a rate and left two explanations for the misses
open. Round 2 was designed to separate them, so this figure is laid out around
that question rather than as a list of cases.

  (a) THE FULL RING at a fixed -4.5% severity, ring distance against blade
      index. This is the discriminating panel. If the misses were caused by
      aliasing between nodal-diameter harmonics, as round 1's errors at ring
      distances of exactly 24/4 and 24/3 suggested, failures would be periodic
      in blade index and this panel would show it outright. If they are
      scattered, that explanation is dead.

  (b) SEVERITY LADDERS on two blades, one that failed at -4.5% (blade 12) and
      one that succeeded (blade 14) pushed on to -6% and -8%. If the linear
      frequency-shift map inside the support search is what degrades, both
      should break down as severity grows, regardless of blade.

  (c) WHAT EACH MODEL ANSWERS, every case pooled. Round 1's most telling
      observation was that the diagonal-only model is not localizing badly so
      much as barely localizing at all, returning one fixed blade regardless of
      where the damage is; this panel is where that shows.

Every case is a genuine full-order extraction on the perturbed mesh, MAC-matched
to the ROM's own mode ordering before either localizer sees it.

Output: figures/step9/step9_fig17_damage_sweep_summary.png
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style  # noqa: E402

FIGS = os.path.join(ROOT, "figures", "step9")
OUT = os.path.join(HERE, "output")
NB = 24
TOL = 2
REF_SEV = -0.045

# The first two cases predate the sweep script; they are recorded in the
# manuscript and in validation3_real_ansys_health_id.json / the blade 10 log.
PRIOR = [
    dict(blade=5, severity=-0.045, diagonal_blade=21, diagonal_ring=8,
         coupled_blade=5, coupled_ring=0, coupled_rank=1, axis="ring"),
    dict(blade=10, severity=-0.030, diagonal_blade=4, diagonal_ring=6,
         coupled_blade=2, coupled_ring=8, coupled_rank=2, axis="severity"),
]
sweep = json.load(open(os.path.join(OUT, "validation3_damage_sweep.json")))
cases = PRIOR + [r for r in sweep if not r.get("failed")]
n = len(cases)

c_ring = np.array([c["coupled_ring"] for c in cases])
d_ring = np.array([c["diagonal_ring"] for c in cases])
c_pick = np.array([c["coupled_blade"] for c in cases])
d_pick = np.array([c["diagonal_blade"] for c in cases])
true = np.array([c["blade"] for c in cases])
sev = np.array([c["severity"] for c in cases])

c_exact, c_within = int((c_ring == 0).sum()), int((c_ring <= TOL).sum())
d_exact, d_within = int((d_ring == 0).sum()), int((d_ring <= TOL).sum())
mode_pick, mode_count = Counter(d_pick.tolist()).most_common(1)[0]

# ---- panel (a): the ring at the reference severity --------------------------
ring_mask = np.isclose(sev, REF_SEV)
r_blade = true[ring_mask]
order = np.argsort(r_blade)
r_blade = r_blade[order]
r_coup = c_ring[ring_mask][order]
r_diag = d_ring[ring_mask][order]

# ---- periodicity of the ring result ----------------------------------------
# Round 1's misses sat at ring distances of 24/4 and 24/3, which suggested the
# failures might be periodic in blade index. With the full ring measured this is
# testable rather than suggestive: for each candidate period that divides 24,
# score how consistently the success/failure label agrees within each residue
# class. Chance sits near half; a real period stands well clear of it.
r_ok = (r_coup <= TOL).astype(int)


def periodicity(period):
    agree = total = 0
    for res in range(period):
        lab = r_ok[r_blade % period == res]
        if len(lab) < 2:
            continue
        agree += int((lab == round(lab.mean())).sum())
        total += len(lab)
    return agree, total


PERIODS = [2, 3, 4, 6, 8, 12]
scores = {p_: periodicity(p_) for p_ in PERIODS}
best_p = max(PERIODS, key=lambda p_: (scores[p_][0] / scores[p_][1]) if scores[p_][1] else 0)
best_a, best_t = scores[best_p]
runner = max((p_ for p_ in PERIODS if p_ != best_p),
             key=lambda p_: (scores[p_][0] / scores[p_][1]) if scores[p_][1] else 0)
run_a, run_t = scores[runner]

# the contiguous arcs of failure, for shading
fail_blades = set(r_blade[r_ok == 0].tolist())

# ---- panel (b): severity ladders -------------------------------------------
ladders = {}
for b in sorted(set(true.tolist())):
    m = true == b
    if m.sum() >= 3:
        s_ = sev[m]
        o = np.argsort(-s_)
        ladders[b] = (-100.0 * s_[o], c_ring[m][o])

plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

INK = plot_style.INK
MUTED = plot_style.INK_SECONDARY
C_COUP = plot_style.C_1B
C_DIAG = plot_style.C_WARN
OK = plot_style.C_OK

# Stacked 3x1, matching Figure 27's layout (plt.subplots(n, 1, ...) plus
# tight_layout). Every explanatory annotation that used to sit inside the axes
# has been moved to the caption: the panels were carrying a periodicity score, a
# tolerance label, a note about the diagonal model's habitual answer and the
# tallies inside the legend entries, and together they crowded the data.
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.0, 13.2))

# ------------------------------------------------------------------ panel (a)
ax1.axhspan(-0.4, TOL + 0.4, color=OK, alpha=0.10, zorder=0)
for _b in sorted(fail_blades):
    ax1.axvspan(_b - 0.5, _b + 0.5, color=C_DIAG, alpha=0.07, zorder=0)
ax1.plot(r_blade, r_diag, marker="o", ms=6, lw=0, color=C_DIAG, alpha=0.75,
         label="diagonal-only", zorder=3)
ax1.plot(r_blade, r_coup, marker="D", ms=7, lw=1.6, color=C_COUP,
         label="coupled", zorder=4)
ax1.set_xlabel("Damaged blade index")
ax1.set_ylabel("Ring distance from the true blade")
ax1.set_xticks(range(0, NB, 2))
ax1.set_xlim(-0.8, NB - 0.2)
plot_style.two_tier_title(ax1, "(a)  the full ring at one severity")
plot_style.legend_inside(ax1, loc="upper right", fontsize=12.5)

# ------------------------------------------------------------------ panel (b)
ax2.axhspan(-0.4, TOL + 0.4, color=OK, alpha=0.10, zorder=0)
for i, (b, (xs, ys)) in enumerate(sorted(ladders.items())):
    ax2.plot(xs, ys, marker="D", ms=8, lw=1.7,
             color=[C_COUP, plot_style.C_ACC, plot_style.C_HF][i % 3],
             label=f"blade {b}", zorder=3)
ax2.set_xlabel("Injected severity  [%]")
ax2.set_ylabel("Ring distance, coupled model")
plot_style.two_tier_title(ax2, "(b)  driving one blade harder")
plot_style.legend_inside(ax2, loc="upper left", fontsize=12.5)

# ------------------------------------------------------------------ panel (c)
ax3.plot([-1, NB], [-1, NB], color=plot_style.FADE, lw=1.2, ls=(0, (5, 4)),
         zorder=1)
ax3.axhline(mode_pick, color=C_DIAG, lw=1.0, ls=(0, (2, 3)), zorder=1)
ax3.scatter(true, d_pick, s=70, color=C_DIAG, alpha=0.85, zorder=3,
            label="diagonal-only")
ax3.scatter(true, c_pick, s=70, color=C_COUP, marker="D", alpha=0.9, zorder=4,
            label="coupled")
ax3.set_xlim(-1.5, NB + 0.5)
ax3.set_ylim(-1.5, NB + 0.5)
ax3.set_xlabel("True damaged blade")
ax3.set_ylabel("Blade the model picks")
ax3.set_xticks(range(0, NB, 4))
ax3.set_yticks(range(0, NB, 4))
plot_style.two_tier_title(ax3, "(c)  what each model answers")
plot_style.legend_inside(ax3, loc="upper left", fontsize=12.5)

fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, "step9_fig17_damage_sweep_summary")

# ------------------------------------------------------------------ numbers
print("=== Real finite-element damage-injection summary ===", flush=True)
print(f"  cases: {n}", flush=True)
for c, dr, cr in sorted(zip(cases, d_ring, c_ring),
                        key=lambda t: (t[0]["blade"], t[0]["severity"])):
    print(f"    blade {c['blade']:2d} @ {c['severity']*100:5.1f}%  "
          f"diagonal ring {dr:2d} -> blade {c['diagonal_blade']:2d}   "
          f"coupled ring {cr:2d} -> blade {c['coupled_blade']:2d}", flush=True)
print(f"  coupled : {c_exact}/{n} exact, {c_within}/{n} within {TOL}, "
      f"worst {c_ring.max()}", flush=True)
print(f"  diagonal: {d_exact}/{n} exact, {d_within}/{n} within {TOL}, "
      f"worst {d_ring.max()}", flush=True)
print(f"  diagonal-only answers blade {mode_pick} in {mode_count}/{n} cases",
      flush=True)
print(f"\n  --- full ring at {abs(REF_SEV)*100:.1f}% ({len(r_blade)} blades) ---",
      flush=True)
print(f"  coupled exact: {int((r_coup==0).sum())}/{len(r_coup)}   "
      f"within {TOL}: {int((r_coup<=TOL).sum())}/{len(r_coup)}", flush=True)
fails = r_blade[r_coup > TOL]
print(f"  blades missed: {fails.tolist()}", flush=True)
print("  periodicity of the success/fail label (agreement within residue "
      "classes):", flush=True)
for p_ in PERIODS:
    a_, t_ = scores[p_]
    print(f"    period {p_:2d}: {a_}/{t_}", flush=True)
print(f"  -> best period {best_p} ({best_a}/{best_t}), "
      f"next best {runner} ({run_a}/{run_t})", flush=True)
for b, (xs, ys) in sorted(ladders.items()):
    print(f"  blade {b} severity ladder: "
          + ", ".join(f"{x:.1f}%->ring {int(y)}" for x, y in zip(xs, ys)),
          flush=True)
print("Saved step9_fig17_damage_sweep_summary.png", flush=True)
