# -*- coding: utf-8 -*-
"""
Campbell (engine-order crossing) diagram for the 1B cluster, Section 3.4.2.

Why this figure exists. The paper repeatedly refers to "the 1B cluster band
this blisk operates in" and, in Section 3.4.1, reports that the mode-0 fold
sits above that band. Neither statement is anchored anywhere: the manuscript
never says which rotor speeds the band corresponds to, nor which engine orders
can reach it, so "the margin between the operating band and the fold" that
Section 4.3 recommends designing against is asserted rather than quantified.
This figure supplies both, and it does so from the project's own rotating-frame
matrices rather than from a textbook approximation.

Physics. The rotating eigenproblem solved at each speed is

    [ -w^2 M + i w (Omega G) + (K + Omega^2 (K_sigma - K_cs)) ] phi = 0

with, all projected onto the same 70-mode secondary basis every other result in
this paper uses:
    K_sigma  centrifugal stress stiffening, from a real prestressed full-order
             static solve (raises blade frequencies with speed),
    K_cs     spin softening (lowers them; stiffening wins here, net rise),
    G        the Coriolis/gyroscopic operator, which splits each
             nodal-diameter pair into a forward and a backward travelling wave.

Validation. At the one speed where an independent prestressed full-order modal
solution exists, 7200 rpm, this reduced rotating model reproduces the 24
fundamental-cluster frequencies to 1.25% mean and 1.44% maximum error. That
number is read from the project's own saved comparison, not recomputed here.

Honest scope note, carried into the caption and the section text: every
forced-response, mistuning and health-monitoring result elsewhere in this paper
is computed for the non-rotating structure, i.e. at the left-hand edge of this
diagram. The diagram is operating-envelope context for those results, not a
rotating extension of them.

Output: figures/step2/step2_fig_campbell_1B.png
"""
import json
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import plot_style  # noqa: E402

ROM = r"F:\ANSYS PCE\ROM_data"
FIGS = os.path.join(ROOT, "figures", "step2")

N_1B = 24                 # fundamental-cluster modes
RPM_MAX = 20000.0
N_SPEED = 81
ENGINE_ORDERS = [2, 4, 6, 8, 12, 16, 24]
FOLD_LO, FOLD_HI = 551.0, 1116.0     # Section 3.4.1, lowest and highest fold
RPM_VALIDATED = 7200.0               # the speed with a full-order prestressed solve

# ------------------------------------------------------------------ physics
M = np.load(os.path.join(ROM, "M_sec.npy"))
K = np.load(os.path.join(ROM, "K_sec.npy"))
bundle = np.load(os.path.join(ROM, "rotating_secondary_bundle.npz"))
G, Kcs, Ksig = bundle["G_sec"], bundle["Kcs_sec"], bundle["Ksigma_sec"]
n = M.shape[0]
_I, _Z = np.eye(n), np.zeros((n, n))


def whirl_frequencies(rpm):
    """Undamped whirl frequencies (Hz) at a given rotor speed, Coriolis included.

    Solved as a first-order companion problem rather than a symmetric
    eigenproblem because the gyroscopic term is skew and makes the quadratic
    pencil non-symmetric; the eigenvalues come in conjugate pairs, so every
    second sorted |imag| is kept.
    """
    om = rpm * 2.0 * np.pi / 60.0
    Kt = K + om ** 2 * (Ksig - Kcs)
    Ct = om * G
    A = np.block([[_Z, _I],
                  [-np.linalg.solve(M, Kt), -np.linalg.solve(M, Ct)]])
    f = np.sort(np.abs(np.linalg.eigvals(A).imag)) / (2.0 * np.pi)
    return f[f > 1e-6][::2]


rpm = np.linspace(0.0, RPM_MAX, N_SPEED)
F = np.array([whirl_frequencies(r) for r in rpm])       # (N_SPEED, n)
lo_1b, hi_1b = F[:, 0], F[:, N_1B - 1]
# the next family up, for context on how isolated the 1B band is
lo_2, hi_2 = F[:, N_1B], F[:, N_1B + 5]

val = json.load(open(os.path.join(ROM, "step2_rotating_validation.json")))
a_val = np.array(val["freqs_stiff_ansys_hz"])[:N_1B]
r_val = np.array(val["freqs_stiff_rom_hz"])[:N_1B]
err = np.abs(100.0 * (r_val - a_val) / a_val)
err_mean, err_max = float(err.mean()), float(err.max())


def crossings(order):
    """Speeds (rpm) at which engine order `order` enters and leaves the band."""
    line = order * rpm / 60.0
    out = []
    for band in (lo_1b, hi_1b):
        s = np.sign(line - band)
        idx = np.where(np.diff(s) != 0)[0]
        for i in idx:                      # linear interpolation on the crossing
            g0, g1 = (line - band)[i], (line - band)[i + 1]
            out.append(rpm[i] + (rpm[i + 1] - rpm[i]) * g0 / (g0 - g1))
    return sorted(out)


# ------------------------------------------------------------------ figure
plot_style.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

INK = plot_style.INK
MUTED = plot_style.INK_SECONDARY

fig, ax = plt.subplots(figsize=(10.4, 7.0))
ymax = 1250.0

# Section 3.4.1's fold is deliberately NOT drawn as a horizontal band here.
# The 551-1116 Hz fold range was traced for the non-rotating structure, and the
# same centrifugal stiffening that lifts the cluster in this diagram also lifts
# the fold, by an amount that depends on forcing through the nondimensional
# hardening parameter. Overlaying a rest-state fold on rotating mode lines
# would invite the reader to measure a margin that is not being measured. The
# fold margin is reported in the text as a ratio to the mode's own linear
# resonance instead, which is the quantity that survives the speed change.

# engine-order rays
for eo in ENGINE_ORDERS:
    line = eo * rpm / 60.0
    ax.plot(rpm, line, color=plot_style.FADE, lw=1.0, ls=(0, (5, 4)), zorder=1)
    yend = eo * RPM_MAX / 60.0
    if yend <= ymax:
        # Kept INSIDE the axes. Drawing it outside the right spine (the previous
        # x = RPM_MAX * 1.006) added content on one side only, and because
        # savefig uses bbox="tight" the saved image then cropped asymmetrically
        # and the plot read as shifted left. Figure 22, which this now matches,
        # places nothing outside its axes.
        xlab = RPM_MAX * 0.955
        ax.text(xlab, eo * xlab / 60.0 + 22, f"EO {eo}", ha="center",
                va="bottom", fontsize=12.5, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.15", fc=plot_style.SURFACE,
                          ec="none"))
    else:
        # label on the ray itself, just below the frame, rotated to its slope
        ylab = ymax * 0.955
        ax.text(ylab * 60.0 / eo, ylab, f"EO {eo}", ha="center", va="top",
                fontsize=12.5, color=MUTED,
                bbox=dict(boxstyle="round,pad=0.15", fc=plot_style.SURFACE,
                          ec="none"))

# the 1B cluster band and the family above it
ax.fill_between(rpm, lo_2, hi_2, color=plot_style.C_HF, alpha=0.16, zorder=2)
ax.text(RPM_MAX * 0.40, (lo_2[32] + hi_2[32]) / 2 + 40,
        "next secondary-mode family", fontsize=12.5, color=plot_style.C_HF)

ax.fill_between(rpm, lo_1b, hi_1b, color=plot_style.C_1B, alpha=0.30, zorder=3)
ax.plot(rpm, lo_1b, color=plot_style.C_1B, lw=2.0, zorder=4)
ax.plot(rpm, hi_1b, color=plot_style.C_1B, lw=2.0, zorder=4)
ax.text(RPM_MAX * 0.30, hi_1b[32] + 26, "1B cluster, 24 modes",
        fontsize=14, color=plot_style.C_1B, fontweight="bold")

# crossings: where an engine order actually drives the cluster
xs, ys = [], []
for eo in ENGINE_ORDERS:
    for c in crossings(eo):
        if 0.0 < c <= RPM_MAX:
            xs.append(c)
            ys.append(eo * c / 60.0)
ax.scatter(xs, ys, s=48, facecolor=plot_style.SURFACE,
           edgecolor=plot_style.C_ACC, linewidth=1.8, zorder=7)
ax.scatter([], [], s=48, facecolor=plot_style.SURFACE,
           edgecolor=plot_style.C_ACC, linewidth=1.8,
           label=f"engine-order crossings of the 1B band ({len(xs)})")

# the state everything else in the paper is computed at
ax.axvline(0.0, color=INK, lw=1.6, zorder=5)
ax.annotate("every other result in this paper is computed\n"
            "here: at rest, 292.8 to 355.9 Hz",
            xy=(0, 324), xytext=(RPM_MAX * 0.085, 700),
            fontsize=13, color=INK,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3,
                            shrinkA=2, shrinkB=4,
                            connectionstyle="arc3,rad=-0.30"))

# the one speed with an independent full-order check
ax.plot([RPM_VALIDATED], [np.interp(RPM_VALIDATED, rpm, lo_1b)], marker="D",
        ms=9, color=plot_style.C_OK, zorder=8)
ax.annotate(f"prestressed full-order check at {RPM_VALIDATED:,.0f} rpm\n"
            f"{err_mean:.2f}% mean, {err_max:.2f}% max over the 24 modes",
            xy=(RPM_VALIDATED, np.interp(RPM_VALIDATED, rpm, lo_1b)),
            xytext=(RPM_MAX * 0.30, 120), fontsize=13, color=plot_style.C_OK,
            arrowprops=dict(arrowstyle="-|>", color=plot_style.C_OK, lw=1.3,
                            shrinkA=2, shrinkB=6,
                            connectionstyle="arc3,rad=0.20"))

ax.set_xlim(0, RPM_MAX)
ax.set_ylim(0, ymax)
ax.set_xlabel("Rotor speed  [rpm]")
ax.set_ylabel("Frequency  [Hz]")
# Subtitle dropped: it ran wider than the axes, so bbox="tight" padded the
# right-hand side out to reach it and left an empty gutter there. The same
# information is already in the caption.
plot_style.two_tier_title(
    ax, "Which engine orders drive the 1B cluster, and at what speed")
plot_style.legend_inside(ax, loc="lower right", fontsize=13, framealpha=0.95)
plot_style.savefig_pub(fig, FIGS, "step2_fig_campbell_1B")

# ------------------------------------------------------------------ numbers
print("=== Campbell diagram, 1B cluster ===", flush=True)
print(f"  speeds            : 0 to {RPM_MAX:,.0f} rpm, {N_SPEED} points", flush=True)
print(f"  1B band at rest   : {lo_1b[0]:.2f} to {hi_1b[0]:.2f} Hz", flush=True)
print(f"  1B band at max    : {lo_1b[-1]:.2f} to {hi_1b[-1]:.2f} Hz", flush=True)
print(f"  net rise, lowest  : {100*(lo_1b[-1]/lo_1b[0]-1):.1f}% "
      f"(stress stiffening beats spin softening)", flush=True)
print(f"  Coriolis split at max speed, mode pair 1: "
      f"{F[-1,1]-F[-1,0]:.2f} Hz", flush=True)
print(f"  engine-order crossings inside the band: {len(xs)}", flush=True)
for eo in ENGINE_ORDERS:
    cs = [c for c in crossings(eo) if 0 < c <= RPM_MAX]
    if cs:
        print(f"    EO {eo:2d}: enters/leaves at "
              + ", ".join(f"{c:,.0f}" for c in cs) + " rpm", flush=True)
print(f"  fold margin as a ratio to the mode's own linear resonance: "
      f"{FOLD_LO/292.82:.2f}x (lowest forcing) to {FOLD_HI/292.82:.2f}x "
      f"(highest); engine-order crossings sit at 1.00x by definition",
      flush=True)
print(f"  rotating-ROM check at {RPM_VALIDATED:,.0f} rpm: "
      f"{err_mean:.3f}% mean, {err_max:.3f}% max", flush=True)
print("Saved step2_fig_campbell_1B.png", flush=True)
