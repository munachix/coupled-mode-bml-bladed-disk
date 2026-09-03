# -*- coding: utf-8 -*-
"""
Real ROM-vs-FEM frequency-response validation for the tuned (cyclic-
symmetric) blisk, for the new Section 2.1.1. Uses the real ANSYS linear
HARMIC solve already computed and saved for Case 1
(F:\\ANSYS PCE\\ROM_data_case1_harmonic\\harmonic_frf.npz, 41 points,
261.7-330 Hz, 2500N generalized force on mode 0, node 1171 UZ) against
the secondary modal ROM's own linear complex-solve prediction
(step9.rom_predicted_frf, the same 70-secondary-mode ROM used throughout
the rest of the paper) -- no new ANSYS run, no fabricated numbers.

Only TWO real curves are plotted (real ANSYS FEM, 181473 DOFs; and the
70-mode secondary ROM). A separate, independently-validated curve for the
intermediate 830-DOF Craig-Bampton stage does not exist in this project's
saved outputs and is not fabricated here -- the Craig-Bampton reduction
is the standard, well-established first stage of this pipeline (physical
mode-shape truncation), and its own fidelity is implicit in the final
70-mode ROM's agreement with real FEM, not separately re-validated.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 9')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import step9 as s9
import plot_style

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(HERE), "figures", "step9")
OUT = os.path.join(HERE, "output")

d = np.load(r'F:\ANSYS PCE\ROM_data_case1_harmonic\harmonic_frf.npz')
freqs, amp_fem = d['freqs'], d['amplitude']
force_scale = float(d['force_scale'])
target_node = int(d['target_node'])
target_dir = str(d['target_dir'])

inp_s4 = s9.s4.load_inputs()
amp_rom = s9.rom_predicted_frf(freqs, inp_s4, target_node, target_dir, force_scale=force_scale)

peak_fem_idx = int(np.argmax(amp_fem))
peak_rom_idx = int(np.argmax(amp_rom))
peak_fem, freq_fem = float(amp_fem[peak_fem_idx]), float(freqs[peak_fem_idx])
peak_rom, freq_rom = float(amp_rom[peak_rom_idx]), float(freqs[peak_rom_idx])
peak_amp_err_pct = abs(peak_rom - peak_fem) / peak_fem * 100.0
peak_freq_shift_hz = freq_rom - freq_fem
rms_err_pct = float(np.sqrt(np.mean(((amp_rom - amp_fem) / amp_fem) ** 2)) * 100.0)
corr = float(np.corrcoef(amp_fem, amp_rom)[0, 1])

# Real per-mode natural-frequency error, ROM (K_sec/M_sec diagonal) vs.
# nothing fabricated: the ROM's own secondary-mode frequencies ARE the
# frequencies the whole training set and every other section's ROM is
# built on (inp['K_sec'], inp['M_sec']) -- there is no independently
# re-solved "full-order FEM modal frequency list" saved separately from
# what generated K_sec/M_sec in the first place (K_sec/M_sec are the
# Craig-Bampton/modal-reduction OUTPUT, referenced to the real full-order
# model at extraction time), so a distinct "ROM vs FEM modal frequency
# error" table is not fabricated here beyond what's already reported
# throughout Section 2.1 and 3.1.4 (resonance-location error <2%, RMSE
# 0.0046, on 100 held-out samples) -- this figure's own quantitative
# claim is restricted to what is directly, freshly computed above: peak
# amplitude error and RMS error across the measured real ANSYS band.

print(f"Real ANSYS FEM peak: {peak_fem:.4f} mm at {freq_fem:.2f} Hz")
print(f"ROM (70-mode) peak:  {peak_rom:.4f} mm at {freq_rom:.2f} Hz")
print(f"Peak amplitude error: {peak_amp_err_pct:.2f}%")
print(f"Peak frequency shift: {peak_freq_shift_hz:.2f} Hz")
print(f"RMS error across band: {rms_err_pct:.2f}%")
print(f"Correlation coefficient: {corr:.4f}")

np.savez(os.path.join(OUT, "rom_vs_fem_validation.npz"),
         freqs=freqs, amp_fem=amp_fem, amp_rom=amp_rom,
         peak_fem=peak_fem, freq_fem=freq_fem, peak_rom=peak_rom, freq_rom=freq_rom,
         peak_amp_err_pct=peak_amp_err_pct, peak_freq_shift_hz=peak_freq_shift_hz,
         rms_err_pct=rms_err_pct, corr=corr, force_scale=force_scale,
         target_node=target_node, target_dir=target_dir)

# ---- figure ----
plot_style.apply_style()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.4, 6.4))
ax.plot(freqs, amp_fem, 'o-', color=plot_style.BLUE, lw=2.4, ms=6,
        mec=plot_style.SURFACE, mew=0.9, label='Full-order FEM solution')
ax.plot(freqs, amp_rom, '--', color=plot_style.C_OK, lw=2.4,
        label='Secondary modal ROM')
ax.set_xlabel('Frequency  [Hz]')
ax.set_ylabel(f'|U$_Z$| at node {target_node}  [mm]')
plot_style.two_tier_title(ax, 'ROM validation against full-order FEM',
                           f'peak error {peak_amp_err_pct:.1f}%, RMS error {rms_err_pct:.1f}% across band')
ax.set_ylim(0, max(amp_fem.max(), amp_rom.max()) * 1.18)
plot_style.legend_inside(ax, loc='upper right')
fig.tight_layout()
plot_style.savefig_pub(fig, FIGS, 'step9_fig2_rom_vs_fem_validation')
print("Saved step9_fig2_rom_vs_fem_validation.png")
