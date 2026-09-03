"""
Cross-mode coupling matrix (2026-08-18: first version, a chord diagram; RESTYLED once already
2026-08-19 with node-arc footprints; REDESIGNED AGAIN 2026-08-19, explicit user request for a
genuinely different plot type after two chord-diagram styles): visualizes the real, ANSYS-
measured nonlinear cross-mode coupling topology across the 24-mode 1B cluster.

WHY A HEATMAP INSTEAD OF A CHORD DIAGRAM: a chord diagram is the right tool for a DENSE,
many-to-many network where the geometry of "who connects to whom" is itself interesting. This
topology is the opposite: every one of the 17 real edges connects two ADJACENT mode indices
(never a skip), so a chord diagram's curved paths all collapse into short arcs hugging the rim
-- the circular layout was fighting the data instead of showing it. A 24x24 coupling-strength
MATRIX (heatmap) puts that same fact in the most direct possible form: real coupling shows up
ONLY as two thin bands immediately next to the diagonal (i=j+/-1), and every other cell is
exactly zero (no measurement exists / no coupling assumed) -- the sparsity and locality are
now literally what the eye sees, not something to infer from ribbon geometry.

Coupling strength per pair, same definition as before: identified cross-coupling coefficients'
magnitude relative to each mode's own diagonal (Duffing) coefficient, from Step 9's own cached
`case3_cross_k3_modes{m0}{m1}.npz` fits (same data step9_fig10's bar chart reports; this is a
different VIEW of already-validated numbers, not new physics).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project\Step 4")
sys.path.insert(0, r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project")
import step4 as s4
import plot_style

plot_style.apply_style()

ROOT = r"C:\Users\Ronin\PycharmProjects\Bladed Disk\New folder\PCE project"
OUT9 = os.path.join(ROOT, 'Step 9', 'output')

pairs_ordered = [(0, 1)] + s4.MODE_GROUPS['pairs'][1:] + \
    [(m, m + 1) for m in s4.MODE_GROUPS['chain'][:-1]]
clean_pairs = set(s4.MODE_GROUPS['pairs'])

NB_MODES = 24
M = np.full((NB_MODES, NB_MODES), np.nan)  # NaN = no measurement / not coupled
edges = []
for (m0, m1) in pairs_ordered:
    fp = os.path.join(OUT9, f'case3_cross_k3_modes{m0}{m1}.npz')
    if not os.path.exists(fp):
        continue
    d = np.load(fp)
    coef0, coef1 = d['coef0'], d['coef1']
    diag = abs(coef0[0]) + abs(coef1[3]) + 1e-12
    cross = abs(coef0[1]) + abs(coef0[2]) + abs(coef0[3]) + abs(coef1[0]) + abs(coef1[1]) + abs(coef1[2])
    strength = cross / diag
    M[m0, m1] = strength
    M[m1, m0] = strength
    edges.append((m0, m1, strength, (m0, m1) not in clean_pairs))

single_mode = s4.MODE_GROUPS['single'][0]

fig, ax = plt.subplots(figsize=(9.6, 8.6))
cmap = plot_style.SEQ_CMAP.copy()
cmap.set_bad(plot_style.SURFACE)
im = ax.imshow(M, cmap=cmap, vmin=0, origin='lower', aspect='equal')

# thin frame around the diagonal to make "this axis IS the same 24 modes
# twice" visually explicit, and light gridlines at every mode so isolated
# vs. chain-adjacent structure is countable, not just colorable.
ax.set_xticks(np.arange(NB_MODES))
ax.set_yticks(np.arange(NB_MODES))
ax.set_xticklabels(range(NB_MODES), fontsize=7.5)
ax.set_yticklabels(range(NB_MODES), fontsize=7.5)
ax.set_xticks(np.arange(-0.5, NB_MODES, 1), minor=True)
ax.set_yticks(np.arange(-0.5, NB_MODES, 1), minor=True)
ax.grid(which='minor', color=plot_style.GRID_HAIRLINE, linewidth=0.5, alpha=0.6)
ax.grid(which='major', visible=False)
ax.plot([-0.5, NB_MODES - 0.5], [-0.5, NB_MODES - 0.5], color=plot_style.INK_MUTED, lw=0.8,
        alpha=0.5, ls=':', zorder=1)

# mark the uncoupled mode's row/col with a light X so its absence from the
# sparsity pattern reads as "checked, genuinely zero" not "just missing data"
ax.scatter([single_mode], [single_mode], marker='x', s=70, color=plot_style.INK_MUTED, linewidths=1.6, zorder=5)

ax.set_xlabel('Mode index')
ax.set_ylabel('Mode index')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label('Coupling strength (cross-term / own-diagonal coefficient magnitude)', color=plot_style.INK)
cb.ax.tick_params(colors=plot_style.INK)

plot_style.two_tier_title(ax, 'Measured cross-mode nonlinear coupling matrix',
                           f'24-mode 1B cluster, {len(edges)} real ANSYS-measured pairs -- white = no '
                           f'coupling measured/assumed; x = mode {single_mode}, confirmed uncoupled')
fig.tight_layout()

figs9 = os.path.join(ROOT, 'figures', 'step9')
os.makedirs(figs9, exist_ok=True)
# RENUMBERED 2026-08-29: this standalone script picked 'fig11' independently
# of step9.py's own main flow, which already uses fig11 for
# step9_fig11_case3_full_multimode_resolution -- moved to fig15 (the next
# free number) to resolve the collision. Content unchanged.
plot_style.savefig_pub(fig, figs9, 'step9_fig15_cross_mode_coupling_matrix')
print(f"Saved: {os.path.join(figs9, 'step9_fig15_cross_mode_coupling_matrix.png')}")
print(f"Edges: {len(edges)}  strength range [{min(e[2] for e in edges):.3f}, {max(e[2] for e in edges):.3f}]")
for (m0, m1, s, is_chain) in sorted(edges, key=lambda e: -e[2])[:5]:
    print(f"  strongest: modes {m0}-{m1}  strength={s:.3f}  ({'chain' if is_chain else 'clean pair'})")
