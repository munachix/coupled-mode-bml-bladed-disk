"""Real relationship: geometric mistuning magnitude -> nonlinear forced-
response behavior (2026-08-13, explicit user request -- 'no relationship
plots showing the effect geometric nonlinearity has on mistuning').

Uses ONLY already-validated machinery (Step 4's exact HBM continuation,
Step 5's exact delta-f model, Step 6's own per-sample SDOF parametrization)
across a real ensemble of Step 3's mistuning samples -- no new physics,
a new analysis/figure layer on top of what's already measured and
validated.
"""
import sys, os, json
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 5')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 6')
sys.path.insert(0, r'C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project')
import numpy as np
import matplotlib.pyplot as plt
import step4 as s4
import step5 as s5
import step6 as s6
import plot_style

plot_style.apply_style()
FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures', 'step9')
os.makedirs(FIGS, exist_ok=True)

N_SAMPLES = 200
TARGET_PEAK = 1.0   # fixed forcing level (linear-estimate peak / q_ref) -- same for every sample

inp = s6.load_inputs()
n_available = inp['theta']['d_length'].shape[0]
sample_idx = np.arange(min(N_SAMPLES, n_available))

max_abs_df = np.zeros(len(sample_idx))
shift_m = np.zeros(len(sample_idx))
peak_amp = np.zeros(len(sample_idx))
res_freq = np.zeros(len(sample_idx))

f0_linear = inp['freqs_sec'][0]

for i, idx in enumerate(sample_idx):
    row = {v: inp['theta'][v][idx] for v in s6.VAR_NAMES}
    df = s5.compute_delta_f_vectorized({k: v[None, :] for k, v in row.items()},
                                        inp['sens'], inp['L_ref'], inp['t_ref'])[0]
    max_abs_df[i] = np.max(np.abs(df)) * 100   # %, worst-blade fractional freq shift

    params = s6.per_sample_sdof_params(inp, idx)
    shift_m[i] = params['features'][0] * 100   # %, mode-0 participation-weighted shift

    cont = s4.duffing_forced_response_continuation(
        params['omega0'], params['M'], params['C'], params['K'], params['K3'],
        params['q_ref'], TARGET_PEAK)
    peak_i = int(np.argmax(cont['amplitude']))
    peak_amp[i] = cont['amplitude'][peak_i]
    res_freq[i] = cont['Omega'][peak_i] / (2 * np.pi) - f0_linear   # Hz, nonlinear peak vs. NOMINAL linear f0

# The raw res_freq above is dominated by a large, roughly-constant offset
# (~170-190 Hz) from nonlinear hardening at this forcing level (see fig5's
# backbone -- the same real effect), NOT by mistuning; the true mistuning-
# driven linear shift is only a few Hz on top of that (checked directly:
# f0*shift_m/2 predicts a ~1-9 Hz range, matching what's left after
# centering). Centering on the ensemble mean isolates the real, mistuning-
# attributable part instead of conflating it with the shared hardening
# offset -- reporting the raw, uncentered number would have been
# misleading (it doesn't even carry the right sign vs. shift_m).
res_freq_centered = res_freq - res_freq.mean()

print(f"n samples: {len(sample_idx)}")
print(f"max|df/f| range: [{max_abs_df.min():.3f}, {max_abs_df.max():.3f}] %")
print(f"peak nonlinear amplitude range: [{peak_amp.min():.4f}, {peak_amp.max():.4f}] mm")
print(f"raw resonance-peak freq range (incl. hardening offset): [{res_freq.min():.2f}, {res_freq.max():.2f}] Hz")
print(f"mistuning-attributable resonance shift (centered) range: [{res_freq_centered.min():.4f}, {res_freq_centered.max():.4f}] Hz")
print(f"corr(max|df/f|, peak amplitude) = {np.corrcoef(max_abs_df, peak_amp)[0,1]:.4f}")
print(f"corr(shift_m, centered resonance shift) = {np.corrcoef(shift_m, res_freq_centered)[0,1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

ax = axes[0]
ax.scatter(max_abs_df, peak_amp, s=26, color=plot_style.BLUE, alpha=0.65,
           edgecolors=plot_style.SURFACE, linewidths=0.5)
ax.set_xlabel('Worst-blade geometric mistuning  max|df$_b$/f|  [%]')
ax.set_ylabel('Nonlinear peak forced-response amplitude  [mm]')
plot_style.two_tier_title(ax, 'Mistuning magnitude vs. nonlinear response amplitude',
                           f'mode 0, fixed forcing (linear peak target={TARGET_PEAK}), '
                           f'{len(sample_idx)} real Step-3 samples')

ax = axes[1]
ax.scatter(shift_m, res_freq_centered, s=26, color=plot_style.ORANGE, alpha=0.65,
           edgecolors=plot_style.SURFACE, linewidths=0.5)
ax.axhline(0, color=plot_style.INK_MUTED, lw=1.0, ls='--')
ax.axvline(0, color=plot_style.INK_MUTED, lw=1.0, ls='--')
ax.set_xlabel("Mode-0 participation-weighted stiffness shift  [%]")
ax.set_ylabel('Resonance-peak shift vs. ensemble mean  [Hz]')
plot_style.two_tier_title(ax, 'Mistuning direction vs. resonance shift',
                           f'mode 0, same {len(sample_idx)} samples, mistuning-attributable part only '
                           f'(common nonlinear-hardening offset removed)')

fig.tight_layout(w_pad=4.0)
plot_style.savefig_pub(fig, FIGS, 'step9_fig9_mistuning_nonlinearity_relationship')
print(f"Figure saved: {os.path.join(FIGS, 'step9_fig9_mistuning_nonlinearity_relationship.png')}  (+ .pdf)")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',
                         'mistuning_nonlinearity_relationship.npz')
np.savez(out_path, max_abs_df=max_abs_df, shift_m=shift_m, peak_amp=peak_amp,
         res_freq=res_freq, res_freq_centered=res_freq_centered)
print(f"Saved: {out_path}")
