"""
Shared publication-quality plotting style for Steps 2-8 (PCE project).
====================================================================

Step 2's own script (`Step 2/step2.py`) already defines the project's
house style -- INK/CREAM palette, left-aligned two-tier titles, borderless
legends below the axes, warm-tan gridlines ("v9: matches Step 3's
'ink-wave' look"). This module is a faithful PORT of those exact constants
and helpers (not a re-derivation from a different design system) so Steps
3-8 read as the SAME paper's figures as Step 2, not six more ad hoc
styles. Where Step 2's own `_new_ax`/`_title`/`_legend_below` helpers are
per-axes Python functions, the parts that are expressible as matplotlib
rcParams (colors, fonts, grid, spine color/width, tick styling) are set
globally here via apply_style() so existing `plt.subplots()` call sites
across Steps 3-8 don't all need individual rewiring; two_tier_title() /
figure_title() / legend_below() port the two remaining non-rcParams
pieces (the left-aligned two-tier title text and the borderless
below-axes legend) directly.

One deliberate, disclosed simplification: Step 2's `_new_ax` also nudges
the left/bottom spines outward by 6pt (`set_position(('outward', 6))`) --
that specific offset has no rcParams equivalent and would require
touching every individual axes-creation call across ~35 figures to
replicate exactly. Left as the one un-ported visual detail; everything
else (palette, fonts, titles, legends, grid) matches exactly.

Every figure is saved as BOTH a PNG (150 dpi, matching Step 2's own
`_savefig`) and a vector PDF (print-ready, resolution-independent) via
savefig_pub().
"""

import os
import matplotlib
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════
# Palette -- verbatim from Step 2/step2.py's own module-level constants
# ═══════════════════════════════════════════════════════════════════
INK = '#22303c'      # near-black foreground text / spines
SURFACE = '#ffffff'  # canvas (Step 2 calls this CREAM)
GRID_HAIRLINE = '#d9cfc0'   # faint warm-tan grid on the white canvas
C_1B = '#2f6f8f'     # deep teal-blue    (Step 2's "1B cluster" series)
C_HF = '#e07a3f'     # warm burnt orange (Step 2's "HF modes" series)
C_ACC = '#7a5195'    # tertiary accent, violet
C_WARN = '#c1443c'   # tolerance / warning red
C_OK = '#3f8f6b'     # pass green
FADE = '#b9c6cd'     # faded reference line grey-blue

INK_PRIMARY = INK
INK_SECONDARY = '#6b7a83'    # Step 2's subtitle color
INK_MUTED = FADE
AXIS_BASELINE = INK

# Fixed-order categorical palette, identical to Step 2's own `_PALETTE`.
CATEGORICAL = [C_1B, C_HF, C_ACC, C_OK, '#8c6d4a']
BLUE, ORANGE, VIOLET, GREEN = C_1B, C_HF, C_ACC, C_OK
AQUA = C_OK           # alias -- Step 2's palette has no separate aqua slot
RED = C_WARN

TRUTH_COLOR = C_1B
COMPARE_COLOR = C_HF
HIGHLIGHT_COLOR = C_WARN
BAND_ALPHA = 0.16

# Sequential ramp: Step 2's own MAC colormap (CREAM -> C_1B -> INK), reused
# here for any heatmap/continuous-magnitude figure so it matches the MAC
# figure's look exactly.
SEQUENTIAL_BLUE = None   # placeholder list kept for call sites expecting a list; see SEQ_CMAP
SEQ_CMAP = None


def apply_style():
    """Call once per script, before creating any figure."""
    global SEQ_CMAP, SEQUENTIAL_BLUE
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',  # Times-compatible math font (figures use $...$ math text)
        # 2026-08-29: bumped across the board (paper reviewer feedback --
        # legends and axis text were unreadable at the docx's final
        # embedded size). Was font.size=10/axes.labelsize=11/
        # axes.titlesize=11/legend.fontsize=9/xtick+ytick unset (~10).
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,

        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': INK,
        'axes.linewidth': 1.1,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,

        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.top': False,
        'ytick.right': False,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,

        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID_HAIRLINE,
        'grid.linewidth': 0.7,
        'grid.alpha': 0.9,

        # No default x-padding: the first/last data point should touch the
        # axes frame, not float in empty margin (2026-08-13, explicit user
        # request). Y keeps a small margin since bars/markers at the very
        # top edge look clipped without one.
        'axes.xmargin': 0.0,
        'axes.ymargin': 0.05,

        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.0,
        'lines.markeredgecolor': SURFACE,

        'legend.frameon': False,
        'legend.labelcolor': INK,

        'axes.prop_cycle': matplotlib.cycler(color=CATEGORICAL),
    })
    SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('mac_ink_wave', [SURFACE, C_1B, INK])
    # A 12-step light->dark sampling of the same ramp, for call sites that
    # want a discrete sequence (e.g. an ordered family of curves) rather
    # than a continuous colormap.
    SEQUENTIAL_BLUE = [matplotlib.colors.to_hex(SEQ_CMAP(x)) for x in
                        [0.30, 0.38, 0.46, 0.54, 0.62, 0.68, 0.74, 0.80, 0.86, 0.90, 0.94, 0.98]]


def two_tier_title(ax, title, subtitle=None):
    """Step 2's own `_title()` helper, ported verbatim: a bold left-aligned
    headline (matplotlib's native loc='left' title) with an optional
    smaller grey subline directly beneath it.

    3D axes fix (2026-08-19): Axes3D.text() has a different signature
    (x, y, z, s, ...) than Axes.text() (x, y, s, ...) -- calling it the 2D
    way silently mis-binds `s` and raises "missing s". Axes3D provides
    text2D(x, y, s, ...) specifically for axes-fraction overlay text like
    this subtitle; detect 3D axes (via the 'zaxis' attribute, present only
    on Axes3D) and route to the right call."""
    ax.set_title(title, loc='left', fontsize=18, fontweight='bold', color=INK,
                 pad=(24 if subtitle else 12))
    if subtitle:
        text_fn = ax.text2D if hasattr(ax, 'zaxis') else ax.text
        text_fn(0.0, 1.015, subtitle, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=13.5, color=INK_SECONDARY)


def figure_title(fig, title, subtitle=None, x=0.02, y_title=0.99, y_subtitle=0.955):
    """Same two-tier headline, anchored to the FIGURE (not one axes) -- for
    multi-panel figures where the headline should sit above all panels."""
    fig.text(x, y_title, title, fontsize=20, fontweight='bold', color=INK, ha='left', va='top')
    if subtitle:
        fig.text(x, y_subtitle, subtitle, fontsize=14.5, color=INK_SECONDARY, ha='left', va='top')


def legend_below(ax, ncol=None, y=-0.18, **kwargs):
    """Step 2's own `_legend_below()` helper, ported verbatim: a horizontal,
    borderless legend centered under the axes."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = min(len(handles), 4)
    leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y),
                     ncol=ncol, frameon=False, fontsize=16, handlelength=1.6,
                     columnspacing=1.8, borderaxespad=0.0, **kwargs)
    for t in leg.get_texts():
        t.set_color(INK)


def legend_inside(ax, loc='best', **kwargs):
    """2026-08-29 (explicit user request, replacing legend_below as the
    default): a stacked (single-column), borderless legend placed INSIDE
    the axes' own empty space, rather than reserving a horizontal strip
    below the plot -- the below-plot strip was itself a source of the
    "too much empty space" complaint on several figures. `loc` should be
    picked per-figure to land in whichever corner/region the plotted data
    actually leaves empty (e.g. 'upper left', 'lower right'); 'best' lets
    matplotlib guess if the caller hasn't checked which corner is free."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    # frameon=True with a translucent white card (2026-08-29, found necessary once
    # legends moved onto the plot itself): an inside-plot legend can land over data
    # or tick labels depending on what empty space the particular figure has --
    # a subtle white backing keeps the text legible either way instead of relying
    # on every call site guessing a truly empty corner.
    opts = dict(ncol=1, frameon=True, facecolor=SURFACE, edgecolor=GRID_HAIRLINE,
                framealpha=0.92, borderpad=0.6, fontsize=16, handlelength=1.6, labelspacing=0.5)
    opts.update(kwargs)  # per-figure overrides (e.g. smaller fontsize for a dense legend)
    leg = ax.legend(handles, labels, loc=loc, **opts)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def savefig_pub(fig, figs_dir, name, formats=('png',)):
    """Saves the same figure at publication quality in every requested
    format. PNG-only by default (150 dpi, matching Step 2's own
    `_savefig`) -- PDF output was dropped project-wide (2026-08-29,
    explicit user request); pass formats=('png', 'pdf') at a call site if
    a vector copy is ever needed again. Returns the list of paths written."""
    paths = []
    for ext in formats:
        path = os.path.join(figs_dir, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', pad_inches=0.05, facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths
