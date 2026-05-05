"""
plot_fault_model.py
===================
Standalone diagnostic plots for the fault-slip model.

Plots produced
--------------
1. Grid & geometry          – staggered mesh + fault trace + layer bands
2. Initial stress field     – σn, τ, pore pressure vs depth
3. Shear / normal stress 2D – tauqs and sigmaqs colormaps on the grid
4. Slip rate vs depth       – V(y) on a log axis (initial + evolution snapshots)

Run standalone:
    python plot_fault_model.py

Or import and call individual functions:
    from plot_fault_model import plot_grid, plot_stress_field, plot_stress_2d, plot_slip_rate
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ── make sure the model module is importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    ModelParameters, Grid, FrictionalZones,
    StressState, FaultState,
    compute_stress_fields,
)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "figure.dpi":       130,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# Layer definitions (model y-coordinates, ysize=2000 m)
LAYERS = {
    "Rocksalt":    {"y_top":    0, "y_bot":  730, "color": "#d4e8f7"},
    "Basal Zech.": {"y_top":  730, "y_bot":  780, "color": "#ffe0b2"},
    "Ten Boer":    {"y_top":  780, "y_bot":  850, "color": "#e8f5e9"},
    "Sandstone":   {"y_top":  850, "y_bot": 1050, "color": "#fff9c4"},
    "Carbonif.":   {"y_top": 1050, "y_bot": 2000, "color": "#f3e5f5"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw layer bands on a y-axis plot
# ─────────────────────────────────────────────────────────────────────────────
def _add_layer_bands(ax, orientation="horizontal", alpha=0.25):
    """Shade geological layers.  orientation='horizontal' → bands on x-axes."""
    patches = []
    for name, L in LAYERS.items():
        if orientation == "horizontal":
            ax.axhspan(L["y_top"], L["y_bot"],
                       color=L["color"], alpha=alpha, linewidth=0)
        else:
            ax.axvspan(L["y_top"], L["y_bot"],
                       color=L["color"], alpha=alpha, linewidth=0)
        patches.append(mpatches.Patch(color=L["color"], label=name, alpha=0.7))
    return patches


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Grid & geometry
# ─────────────────────────────────────────────────────────────────────────────
def plot_grid(grid: Grid, show: bool = True) -> plt.Figure:
    """
    Draw the staggered 2-D mesh with:
      - main grid lines (τ / σ nodes)
      - fault trace at centre column
      - geological layer colour bands
      - legend for node types
    """
    p  = grid.p
    x, y = grid.x, grid.y
    xp, yp = grid.xp, grid.yp
    X,  Y  = np.meshgrid(x,  y)
    Xp, Yp = np.meshgrid(xp, yp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Model Grid & Geometry", fontsize=13, fontweight="bold", y=1.01)

    # ── left panel: full mesh ──────────────────────────────────────────
    ax = axes[0]

    # Layer shading (horizontal bands = depth bands)
    patches = _add_layer_bands(ax, orientation="horizontal", alpha=0.20)

    # Main grid
    ax.plot(X,  Y,  color="#9e9e9e", linewidth=0.35, zorder=2)
    ax.plot(X.T, Y.T, color="#9e9e9e", linewidth=0.35, zorder=2)

    # Staggered pressure nodes (subset for clarity)
    step = max(1, p.Nx // 20)
    ax.plot(Xp[::step, ::step], Yp[::step, ::step],
            "s", ms=2.5, color="#e67e22", alpha=0.6,
            label="Pressure nodes (xp, yp)", zorder=3)

    # τ nodes (basic grid, subset)
    ax.plot(X[::step, ::step], Y[::step, ::step],
            "o", ms=2.5, color="#2980b9", alpha=0.6,
            label="Stress nodes (x, y)", zorder=3)

    # Fault trace (centre column)
    x_fault = 0.0
    ax.axvline(x_fault, color="#c0392b", linewidth=2.0,
               linestyle="--", label=f"Fault (x = {x_fault} m)", zorder=4)

    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]  (depth increases upward in model coords)")
    ax.set_title(f"Full mesh  ({p.Nx} × {p.Ny} nodes,  α = {p.alpha}°)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ── right panel: layer column ──────────────────────────────────────
    ax2 = axes[1]
    for name, L in LAYERS.items():
        height = L["y_bot"] - L["y_top"]
        ax2.barh(L["y_top"] + height / 2, 1,
                 height=height, color=L["color"],
                 edgecolor="#555", linewidth=0.6, left=0)
        ax2.text(0.5, L["y_top"] + height / 2, name,
                 ha="center", va="center", fontsize=8.5, fontweight="bold")

    ax2.set_xlim(0, 1); ax2.set_ylim(0, p.ysize)
    ax2.set_xticks([])
    ax2.set_ylabel("y  [m]")
    ax2.set_title("Stratigraphy")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # Grid stats annotation
    stats = (f"dx = {grid.dx:.1f} m\ndy = {grid.dy:.1f} m\n"
             f"N_DOF = {grid.N:,}")
    axes[0].text(0.01, 0.99, stats, transform=axes[0].transAxes,
                 va="top", fontsize=8, color="#333",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Initial stress field (1-D profiles)
# ─────────────────────────────────────────────────────────────────────────────
def plot_stress_field(grid: Grid, stress: StressState,
                      fric: FrictionalZones,
                      show: bool = True) -> plt.Figure:
    """
    Four-panel figure of depth profiles:
      (a) Effective normal stress σn  [MPa]
      (b) Initial shear stress τ₀     [MPa]
      (c) Pore pressure Pl, Pr        [MPa]
      (d) Rate-and-state params a, b  [–]
    """
    y  = grid.y
    s  = stress

    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True)
    fig.suptitle("Initial Stress & Friction State vs Depth", fontsize=13,
                 fontweight="bold")

    kw_line = dict(linewidth=2.0)

    # ── (a) Effective normal stress ────────────────────────────────────
    ax = axes[0]
    _add_layer_bands(ax)
    ax.plot(s.sigman0 / 1e6, y, color="#1565c0", **kw_line, label="σₙ₀")
    ax.set_xlabel("σₙ  [MPa]");  ax.set_ylabel("y  [m]")
    ax.set_title("(a) Eff. normal stress")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # ── (b) Initial shear stress ───────────────────────────────────────
    ax = axes[1]
    _add_layer_bands(ax)
    ax.plot(s.tau0 / 1e6, y, color="#b71c1c", **kw_line, label="τ₀")
    ax.set_xlabel("τ₀  [MPa]")
    ax.set_title("(b) Initial shear stress")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # friction ratio τ/σ
    ratio = s.tau0 / s.sigman0
    ax2b  = ax.twiny()
    ax2b.plot(ratio, y, color="#6a1a9a", linewidth=1.2,
              linestyle="--", label="τ/σ")
    ax2b.set_xlabel("τ₀ / σₙ  [–]", color="#6a1a9a")
    ax2b.tick_params(axis="x", labelcolor="#6a1a9a")
    ax2b.spines["top"].set_visible(True)
    ax2b.spines["top"].set_color("#6a1a9a")

    # ── (c) Pore pressure ─────────────────────────────────────────────
    ax = axes[2]
    _add_layer_bands(ax)
    ax.plot(s.Pl0 / 1e6, y, color="#00838f", **kw_line, label="Pl (left)")
    ax.plot(s.Pr0 / 1e6, y, color="#ef6c00", linewidth=1.5,
            linestyle="--", label="Pr (right)")
    ax.set_xlabel("Pore pressure  [MPa]")
    ax.set_title("(c) Pore pressure")
    ax.legend(fontsize=8, loc="lower right")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # ── (d) Friction a, b ─────────────────────────────────────────────
    ax = axes[3]
    _add_layer_bands(ax)
    ax.plot(fric.a, y, color="#2e7d32", **kw_line, label="a")
    ax.plot(fric.b, y, color="#e65100", **kw_line,
            linestyle="--", label="b")
    ax.plot(fric.a - fric.b, y, color="#555", linewidth=1.2,
            linestyle=":", label="a − b")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="-")
    ax.set_xlabel("a,  b,  a−b  [–]")
    ax.set_title("(d) Rate-and-state params")
    ax.legend(fontsize=8, loc="lower right")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Layer legend on rightmost panel
    patches = [mpatches.Patch(color=L["color"], label=name, alpha=0.7)
               for name, L in LAYERS.items()]
    axes[3].legend(handles=patches, fontsize=7, loc="upper right",
                   title="Layer", title_fontsize=7, framealpha=0.85)

    for ax in axes:
        ax.set_ylim(0, grid.p.ysize)
        ax.invert_yaxis()           # depth increases downward visually

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  2-D stress colormaps
# ─────────────────────────────────────────────────────────────────────────────
def plot_stress_2d(grid: Grid, stress: StressState,
                   tauqs: np.ndarray = None,
                   sigmaqs: np.ndarray = None,
                   show: bool = True) -> plt.Figure:
    """
    Two-panel colourmap:
      left  – shear stress tauqs  (Ny × Nx)
      right – normal stress sigmaqs  (Ny-1 × Nx-1)

    If tauqs / sigmaqs are not supplied, the zero-displacement initial
    fields are computed and shown as a reference.
    """
    p  = grid.p
    Ny, Nx = p.Ny, p.Nx

    if tauqs is None or sigmaqs is None:
        uy = np.zeros((Ny, Nx + 1))
        ux = np.zeros((Ny + 1, Nx))
        tauqs, sigmaqs = compute_stress_fields(
            uy, ux, grid.dx, grid.dy,
            p.lam, p.G, grid.cosa, grid.sina, Ny, Nx)

    # Add the lithostatic background to sigmaqs for interpretability
    # sigmaqs is the *perturbation*; add sigman0 for total
    sigman0_2d = stress.sigman0[1:Ny, np.newaxis] * np.ones((Ny - 1, Nx - 1))
    sigma_total = sigman0_2d - sigmaqs          # effective normal (compression +)
    tau_total   = stress.tau0[:, np.newaxis] * np.ones((Ny, Nx)) + tauqs

    X_tau, Y_tau     = np.meshgrid(grid.x,         grid.y)
    X_sig, Y_sig     = np.meshgrid(grid.x[1:Nx],   grid.yp[1:Ny])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.suptitle("2-D Stress Fields on the Grid", fontsize=13, fontweight="bold")

    # ── shear stress ──────────────────────────────────────────────────
    ax = axes[0]
    vmax_tau = np.percentile(np.abs(tau_total), 99)
    norm_tau = TwoSlopeNorm(vcenter=np.median(tau_total),
                            vmin=tau_total.min(), vmax=tau_total.max())
    cf = ax.pcolormesh(X_tau, Y_tau, tau_total / 1e6,
                       cmap="RdBu_r", norm=norm_tau,
                       shading="auto", rasterized=True)
    plt.colorbar(cf, ax=ax, label="τ  [MPa]", pad=0.02)
    ax.axvline(0, color="k", linewidth=1.5, linestyle="--",
               label="Fault trace")
    ax.set_xlabel("x  [m]");  ax.set_ylabel("y  [m]")
    ax.set_title("(a) Shear stress  τ(x, y)")
    ax.legend(fontsize=8)

    # ── normal stress ─────────────────────────────────────────────────
    ax = axes[1]
    cf2 = ax.pcolormesh(X_sig, Y_sig, sigma_total / 1e6,
                        cmap="viridis", shading="auto", rasterized=True)
    plt.colorbar(cf2, ax=ax, label="σₙ  [MPa]", pad=0.02)
    ax.axvline(0, color="w", linewidth=1.5, linestyle="--",
               label="Fault trace")
    ax.set_xlabel("x  [m]")
    ax.set_title("(b) Effective normal stress  σₙ(x, y)")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_ylim(0, p.ysize)
        # Layer boundaries as horizontal dashed lines
        for L in LAYERS.values():
            ax.axhline(L["y_top"], color="gray", linewidth=0.6,
                       linestyle=":", alpha=0.7)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Slip rate vs depth
# ─────────────────────────────────────────────────────────────────────────────
def plot_slip_rate(grid: Grid, fault: FaultState,
                   V_snapshots: list[np.ndarray] = None,
                   t_labels:    list[str]        = None,
                   show: bool = True) -> plt.Figure:
    """
    Slip rate V(y) on a log₁₀ x-axis vs depth y.

    Parameters
    ----------
    V_snapshots : list of 1-D arrays shape (Ny,), optional
        Slip-rate profiles at different times (from OutputManager.Vm columns).
        If None, only the current fault.V is plotted.
    t_labels : list of str
        Labels for each snapshot (e.g. ["t=0", "t=100 yr"]).
    """
    y  = grid.y
    p  = grid.p

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle("Slip Rate vs Depth", fontsize=13, fontweight="bold")

    # ── left panel: current / initial V ───────────────────────────────
    ax = axes[0]
    _add_layer_bands(ax)

    V = np.maximum(fault.V, 1e-40)
    ax.semilogx(V, y, color="#c0392b", linewidth=2.2,
                label="V  (current)")
    ax.axvline(p.Vi, color="gray", linewidth=1.0,
               linestyle="--", label=f"Vi = {p.Vi:.0e} m/s")
    ax.axvline(p.V0, color="#27ae60", linewidth=1.0,
               linestyle="-.", label=f"V₀ = {p.V0:.0e} m/s")

    ax.set_xlabel("Slip rate V  [m/s]")
    ax.set_ylabel("y  [m]")
    ax.set_title("(a) Current slip-rate profile")
    ax.legend(fontsize=8)
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.35)
    ax.set_ylim(0, p.ysize)
    ax.invert_yaxis()

    # Layer boundary ticks on right side
    for name, L in LAYERS.items():
        ax.axhline(L["y_top"], color="gray", linewidth=0.6,
                   linestyle=":", alpha=0.5)
        ax.text(ax.get_xlim()[0] * 1.05, L["y_top"] + 5,
                name, fontsize=6.5, color="#444", va="bottom")

    # ── right panel: time snapshots (if provided) ──────────────────────
    ax2 = axes[1]
    _add_layer_bands(ax2)
    ax2.set_title("(b) Evolution over time")
    ax2.set_xlabel("log₁₀(V)  [m/s]")

    if V_snapshots:
        cmap   = plt.cm.plasma
        n_snap = len(V_snapshots)
        labels = t_labels or [f"snapshot {i}" for i in range(n_snap)]
        for i, (Vs, lbl) in enumerate(zip(V_snapshots, labels)):
            Vs = np.maximum(Vs, 1e-40)
            color = cmap(i / max(n_snap - 1, 1))
            ax2.plot(np.log10(Vs), y,
                     color=color, linewidth=1.4, label=lbl)
        ax2.legend(fontsize=7, ncol=2, loc="lower right")
    else:
        # Show a − b profile instead as supplementary info
        fric = FrictionalZones(p, y)
        ax2.fill_betweenx(y, fric.a - fric.b, 0,
                          where=(fric.a - fric.b > 0),
                          color="#e53935", alpha=0.35,
                          label="Velocity weakening (a−b > 0)")
        ax2.fill_betweenx(y, fric.a - fric.b, 0,
                          where=(fric.a - fric.b < 0),
                          color="#1e88e5", alpha=0.35,
                          label="Velocity strengthening (a−b < 0)")
        ax2.plot(fric.a - fric.b, y, color="#333",
                 linewidth=1.5, label="a − b")
        ax2.axvline(0, color="k", linewidth=0.9)
        ax2.set_xlabel("a − b  [–]  (no snapshots provided)")
        ax2.legend(fontsize=8, loc="lower right")

    for ax in axes:
        ax.set_ylim(0, p.ysize)
        ax.invert_yaxis()
        for L in LAYERS.values():
            ax.axhline(L["y_top"], color="gray", linewidth=0.5,
                       linestyle=":", alpha=0.5)

    ax2.set_ylim(0, p.ysize)
    ax2.invert_yaxis()

    # Shared layer legend
    patches = [mpatches.Patch(color=L["color"], label=nm, alpha=0.7)
               for nm, L in LAYERS.items()]
    axes[0].legend(handles=patches + axes[0].get_legend_handles_labels()[0],
                   fontsize=7, loc="lower right",
                   title="Layer / curves", title_fontsize=7)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main: run all four plots
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out = Path("plots")
    out.mkdir(exist_ok=True)

    p      = ModelParameters(Nx=101, Ny=101)
    grid   = Grid(p)
    fric   = FrictionalZones(p, grid.y)
    stress = StressState(p, grid.y)
    fault  = FaultState(p, stress, fric)

    print("Plotting grid …")
    fig1 = plot_grid(grid, show=False)
    fig1.savefig(out / "01_grid.png", dpi=150, bbox_inches="tight")

    print("Plotting stress field …")
    fig2 = plot_stress_field(grid, stress, fric, show=False)
    fig2.savefig(out / "02_stress_field.png", dpi=150, bbox_inches="tight")

    print("Plotting 2-D stress colormaps …")
    fig3 = plot_stress_2d(grid, stress, show=False)
    fig3.savefig(out / "03_stress_2d.png", dpi=150, bbox_inches="tight")

    print("Plotting slip rate …")
    fig4 = plot_slip_rate(grid, fault, show=False)
    fig4.savefig(out / "04_slip_rate.png", dpi=150, bbox_inches="tight")

    print(f"Saved four figures to  {out.resolve()}/")
    plt.show()


if __name__ == "__main__":
    main()