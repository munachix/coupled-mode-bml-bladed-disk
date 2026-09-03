# -*- coding: utf-8 -*-
"""
Builds a single combined 4-panel waveform gallery (Section 3.3.4),
replacing the two separate standalone waveform PNGs, per explicit user
request: "we don't have a waveform picture for when mistuning and
nonlinearity are present... we need 4 pictures... they will all be
Figure 19 but we will add (a,b,c,d) to the title within the PNG."

All four panels use the REAL coupled time-domain ODE solver
(step4.duffing_forced_response_coupled, modes 0-1, the same physics that
generated the severity classifier's own training data) -- genuine
simulated signals, not sinusoids drawn from a predicted amplitude, and
not the old panel (d)'s BPINN-illustrative substitute.

(a) Ideal tuned baseline (df = 0 on every blade).
(b) A real healthy manufacturing-mistuning realization (same rng draw as
    the original step8_fig8b panel).
(c) The damage trajectory's final state (blade from Step 8's own real
    trajectory, single-blade fault, 15% severity) -- reconstructed exactly
    from damage_trajectory.npz's own df_baseline/damaged_blade/severity
    arrays (df_traj[t] = df_baseline with df_traj[:, damaged_blade] +=
    severity[t], the same formula step8.py itself uses).
(d) Mistuning and nonlinearity together, the gap none of (a)-(c) covers:
    the same real healthy realization as (b), driven at 16x forcing at
    1.3*omega_0 so the response sits on the hardened branch. See the
    comment above D_FORCE_MULT for why raising single-blade severity is
    the wrong lever here.

WINDOW (verified 2026-09-03, and stated in the manuscript). Every panel is
the last ten forcing cycles of a 60-cycle run started from rest. None is an
asymptotic steady state, and (c) and (d) do not have one: re-running the
same solver, panel (a) settles from 0.0398 mm in this window to 0.0253 mm
by 150 cycles, while (c) and (d) escape past roughly 150 cycles because the
identified cubic force polynomial stops being restoring outside the
amplitude range it was fitted over. That escape survives rtol 1e-10, so it
is a property of the fitted model rather than of the integrator. Keeping
one identical window for all four is what makes the panels comparable; do
not raise n_cycles without re-reading this note.
"""
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 5")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 7")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project")
import step6 as s6
import step4 as s4
import step5 as s5
import step7 as s7
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step8")
RANDOM_SEED = 42
PAIR = (0, 1)

inp = s6.load_inputs()
NB = inp["n_blades"] if "n_blades" in inp else 24

traj = np.load(os.path.join(OUT, "damage_trajectory.npz"))
df_baseline = traj["df_baseline"]
damaged_blade = int(traj["damaged_blade"])
severity = traj["severity"]

# ---- real coupled-physics setup (matches step8.py's own fig8 block) ----
cc = s4.CONFIG["nonlinear"]["cross_coupling"][PAIR]
Ki0 = inp["K_sec"][0, 0]; Kj0 = inp["K_sec"][1, 1]
Mi = inp["M_sec"][0, 0]; Mj = inp["M_sec"][1, 1]
Ci = inp["C_sec"][0, 0]; Cj = inp["C_sec"][1, 1]
omega0_i = math.sqrt(Ki0 / Mi)

pair_model, pair_norm = s7.load_bpinn_coupled(PAIR)
if pair_norm.get("is_forcing_aware", False):
    tp = float(pair_norm.get("default_target_peak", 1.0))
    zeta_i0 = Ci / (2 * math.sqrt(Ki0 * Mi)); zeta_j0 = Cj / (2 * math.sqrt(Kj0 * Mj))
    Fg_i = tp * 2 * zeta_i0 * Ki0; Fg_j = tp * 2 * zeta_j0 * Kj0
else:
    Fg_i = float(pair_norm["f_gen_i"]); Fg_j = float(pair_norm["f_gen_j"])

rng_wf = np.random.default_rng(RANDOM_SEED + 70_000)
healthy_idx = int(rng_wf.integers(0, inp["n_samples"]))
df_healthy = s5.compute_delta_f_vectorized(
    {k: inp["theta"][k][healthy_idx:healthy_idx + 1] for k in s7.VAR_NAMES},
    inp["sens"], inp["L_ref"], inp["t_ref"])[0]

df_damage_final = df_baseline.copy()
df_damage_final[damaged_blade] += severity[-1]

# Panel (d), corrected 2026-09-02. The previous version raised single-blade
# severity to 30% and expected more nonlinearity. That is backwards: detuning a
# blade moves it AWAY from resonance, which lowers its amplitude, and cubic
# distortion is amplitude-driven. The old panel (d) therefore came out with a
# LOWER peak than panel (c) and looked no more nonlinear than the tuned panel.
#
# Nonlinear distortion is reached the way the physics actually reaches it: by
# driving harder, and by driving above the linear resonance where the hardened
# branch lives. Panel (d) keeps a real manufacturing-mistuning realization and
# drives it at 16x forcing at 1.3*omega_0. That is the highest combination this
# coupled time-domain solver holds together at -- 32x diverges numerically, as
# does 32x at 1.5*omega_0 -- and it produces clearly non-sinusoidal crests and
# an uneven envelope rather than a clean two-mode beat.
D_FORCE_MULT = 16.0
D_DRIVE_RATIO = 1.3
df_mistuned_nonlinear = df_healthy.copy()


def solve_waveform(df_state, force_mult=1.0, drive_ratio=1.0, n_cycles=60):
    scale = (1.0 + df_state) ** 2 - 1.0
    Ki = Ki0 * (1.0 + float(scale @ inp["P"][:, 0]))
    Kj = Kj0 * (1.0 + float(scale @ inp["P"][:, 1]))
    Omega = omega0_i * drive_ratio
    r = s4.duffing_forced_response_coupled(PAIR, (Ki, Kj), (Mi, Mj), (Ci, Cj),
                                            cc["coef0"], cc["coef1"],
                                            (Fg_i * force_mult, Fg_j * force_mult),
                                            Omega, n_cycles=n_cycles, steps_per_cycle=60)
    tail = r["t"] > r["t"].max() - 10 * (2 * np.pi / Omega)
    t_show = r["t"][tail] - r["t"][tail].min()
    return t_show, r["q_i"][tail]


panels = [
    ("(a) Ideal tuned baseline", np.zeros(NB), plot_style.C_1B, 1.0, 1.0),
    (f"(b) Real healthy sample (Step 3 #{healthy_idx})", df_healthy, plot_style.C_OK, 1.0, 1.0),
    (f"(c) Damage trajectory, final state (blade {damaged_blade}, 15% loss)",
     df_damage_final, plot_style.C_HF, 1.0, 1.0),
    (f"(d) Mistuning + nonlinearity ({D_FORCE_MULT:.0f}x forcing at {D_DRIVE_RATIO:g}f$_0$)",
     df_mistuned_nonlinear, plot_style.C_WARN, D_FORCE_MULT, D_DRIVE_RATIO),
]

plot_style.apply_style()
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
for ax, (label, df_state, color, fmul, wrat) in zip(axes.flat, panels):
    t_show, q_show = solve_waveform(df_state, fmul, wrat)
    ax.plot(t_show, q_show, color=color, lw=1.8)
    ax.axhline(0, color=plot_style.INK_MUTED, lw=0.8)
    ax.set_xlabel("time [s]", fontsize=15)
    ax.set_ylabel(r"Mode 0 displacement, $q_0(t)$  [mm]", fontsize=15)
    ax.set_title(label, loc="left", fontsize=15, fontweight="bold", color=plot_style.INK, pad=8)
    ax.tick_params(labelsize=13)

fig.tight_layout()
out_path = os.path.join(FIGS, "step8_fig8_waveform_gallery.png")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, facecolor=fig.get_facecolor())
print("Saved:", out_path)
print(f"real-solver check: panel (d) uses a real healthy mistuning realization at "
      f"{D_FORCE_MULT:.0f}x forcing driven at {D_DRIVE_RATIO:g}*omega_0 -- the strongest "
      f"combination this coupled solver integrates without diverging")
