"""
plot_cloudmap.py
================
Cloud-map (filled contour / pcolormesh) plots and GIF animations
for every major field in the fault-slip model.

Fields supported
----------------
  "V"       – slip rate            (Ny × n_snaps)   [m/s]  log scale
  "U"       – cumulative slip      (Ny × n_snaps)   [m]
  "tau"     – on-fault shear stress (Ny × n_snaps)  [MPa]
  "sigma"   – on-fault normal stress(Ny × n_snaps)  [MPa]
  "tauqs"   – 2-D shear stress     (Ny × Nx × n_snaps) [MPa]
  "sigmaqs" – 2-D normal stress    (Ny-1×Nx-1×n_snaps) [MPa]

Usage
-----
  python plot_cloudmap.py                  # uses synthetic demo data
  python plot_cloudmap.py dataall.npz      # load real model output

API
---
  cm = CloudMapper(data, grid_info)
  cm.plot_field("tauqs")                  # single static cloud map
  cm.animate_field("tauqs", fps=12)       # GIF animation
  cm.animate_all(fps=12)                  # animate every field
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "figure.facecolor":  "white",
    "figure.dpi":        130,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── geological layers (model y-coords, ysize = 2000 m) ───────────────────────
LAYERS = {
    "Rocksalt":    {"y_top":    0, "y_bot":  730, "color": "#d4e8f7"},
    "Basal Zech.": {"y_top":  730, "y_bot":  780, "color": "#ffe0b2"},
    "Ten Boer":    {"y_top":  780, "y_bot":  850, "color": "#e8f5e9"},
    "Sandstone":   {"y_top":  850, "y_bot": 1050, "color": "#fff9c4"},
    "Carbonif.":   {"y_top": 1050, "y_bot": 2000, "color": "#f3e5f5"},
}

# ── field metadata ────────────────────────────────────────────────────────────
FIELDS = {
    "V": {
        "label":  "Slip rate  V  [m/s]",
        "cmap":   "inferno",
        "scale":  "log",
        "unit":   "m/s",
        "title":  "Slip Rate",
        "dim":    "1d",
    },
    "U": {
        "label":  "Cumulative slip  U  [m]",
        "cmap":   "plasma",
        "scale":  "linear",
        "unit":   "m",
        "title":  "Cumulative Slip",
        "dim":    "1d",
    },
    "tau": {
        "label":  "Shear stress  τ  [MPa]",
        "cmap":   "RdBu_r",
        "scale":  "linear",
        "unit":   "MPa",
        "title":  "On-fault Shear Stress",
        "dim":    "1d",
    },
    "sigma": {
        "label":  "Normal stress  σₙ  [MPa]",
        "cmap":   "viridis",
        "scale":  "linear",
        "unit":   "MPa",
        "title":  "On-fault Normal Stress",
        "dim":    "1d",
    },
    "tauqs": {
        "label":  "2-D Shear stress  τ  [MPa]",
        "cmap":   "seismic",
        "scale":  "diverging",
        "unit":   "MPa",
        "title":  "2-D Shear Stress Field",
        "dim":    "2d",
        "x_key":  "x",
        "y_key":  "y",
    },
    "sigmaqs": {
        "label":  "2-D Normal stress  σₙ  [MPa]",
        "cmap":   "RdYlBu_r",
        "scale":  "linear",
        "unit":   "MPa",
        "title":  "2-D Normal Stress Field",
        "dim":    "2d",
        "x_key":  "xp",
        "y_key":  "yp",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CloudMapper
# ─────────────────────────────────────────────────────────────────────────────
class CloudMapper:
    """
    Wraps model output arrays and exposes cloudmap + animation methods.

    Parameters
    ----------
    data      : dict with keys matching field names above, plus 'tm', 'y', 'x', 'xp', 'yp'
    out_dir   : where to save figures and GIFs
    """

    def __init__(self, data: dict, out_dir: Path = Path("plots")):
        self.data    = data
        self.out     = out_dir
        self.out.mkdir(parents=True, exist_ok=True)

        self.tm      = data["tm"]                      # (n_snaps,)
        self.yr      = 365 * 24 * 3600
        self.n_snaps = len(self.tm)
        self.y       = data["y"]                       # (Ny,)
        self.x       = data["x"]                       # (Nx,)
        self.xp      = data.get("xp", data["x"])
        self.yp      = data.get("yp", data["y"])

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_field(self, name: str) -> np.ndarray:
        """Return the raw array (unit conversion where needed)."""
        arr = self.data[name]
        meta = FIELDS[name]
        if meta["unit"] == "MPa":
            arr = arr / 1e6
        elif meta["unit"] == "m/s" and meta["scale"] == "log":
            arr = np.maximum(arr, 1e-40)
        return arr

    def _norm(self, name: str, arr: np.ndarray, frame: int = None):
        """Return a Normalize instance appropriate for the field."""
        meta  = FIELDS[name]
        data  = arr if frame is None else arr[..., frame]
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)

        if meta["scale"] == "log":
            vmin = max(vmin, 1e-40)
            return LogNorm(vmin=vmin, vmax=vmax)
        elif meta["scale"] == "diverging":
            vcen = np.nanmedian(arr)
            return TwoSlopeNorm(vcenter=vcen, vmin=vmin, vmax=vmax)
        else:
            return Normalize(vmin=vmin, vmax=vmax)

    def _layer_patches(self):
        return [mpatches.Patch(color=L["color"], label=nm, alpha=0.7)
                for nm, L in LAYERS.items()]

    def _draw_layers(self, ax):
        for L in LAYERS.values():
            ax.axhspan(L["y_top"], L["y_bot"],
                       color=L["color"], alpha=0.18, linewidth=0)
            ax.axhline(L["y_top"], color="gray",
                       linewidth=0.5, linestyle=":", alpha=0.5)

    def _draw_layers_2d(self, ax):
        """Horizontal dashed lines on a 2-D pcolormesh."""
        for L in LAYERS.values():
            ax.axhline(L["y_top"], color="white",
                       linewidth=0.7, linestyle="--", alpha=0.6)

    # ── time label ────────────────────────────────────────────────────────────
    def _tlabel(self, frame: int) -> str:
        t_yr = self.tm[frame] / self.yr
        if t_yr < 1:
            return f"t = {self.tm[frame]:.1f} s"
        return f"t = {t_yr:.1f} yr"

    # ─────────────────────────────────────────────────────────────────────────
    # Static cloud map  (single frame = last snapshot, or chosen index)
    # ─────────────────────────────────────────────────────────────────────────
    def plot_field(self, name: str, frame: int = -1,
                   save: bool = True) -> plt.Figure:
        """
        Draw a filled-contour / pcolormesh cloud map of *name* at time *frame*.
        Works for both 1-D (time × depth) and 2-D (x × y) fields.
        """
        meta = FIELDS[name]
        arr  = self._get_field(name)
        norm = self._norm(name, arr)

        if meta["dim"] == "1d":
            fig = self._plot_1d_cloudmap(name, arr, norm, frame, meta)
        else:
            fig = self._plot_2d_cloudmap(name, arr, norm, frame, meta)

        if save:
            fig.savefig(self.out / f"cloudmap_{name}.png",
                        dpi=150, bbox_inches="tight")
        return fig

    # ── 1-D cloud map: depth × time ───────────────────────────────────────────
    def _plot_1d_cloudmap(self, name, arr, norm, frame, meta) -> plt.Figure:
        tm_yr  = self.tm / self.yr
        y      = self.y

        fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                                 gridspec_kw={"width_ratios": [3, 1]})
        fig.suptitle(f"Cloud Map — {meta['title']}", fontsize=13,
                     fontweight="bold")

        # ── left: depth–time cloudmap ──────────────────────────────────
        ax = axes[0]
        T, Y = np.meshgrid(tm_yr, y)
        cf = ax.pcolormesh(T, Y, arr, norm=norm,
                           cmap=meta["cmap"], shading="auto",
                           rasterized=True)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.06)
        cb  = fig.colorbar(cf, cax=cax, label=meta["label"])
        cb.ax.tick_params(labelsize=8)

        # Layer boundaries
        for L in LAYERS.values():
            ax.axhline(L["y_top"], color="white", linewidth=0.8,
                       linestyle="--", alpha=0.6)

        # Snapshot marker
        fi = frame if frame >= 0 else self.n_snaps + frame
        ax.axvline(tm_yr[fi], color="lime", linewidth=1.5,
                   linestyle="--", label=self._tlabel(fi))

        ax.set_xlabel("Time  [yr]")
        ax.set_ylabel("Depth  y  [m]")
        ax.set_title("(a) Depth × time cloud map")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right")

        # ── right: profile at chosen snapshot ─────────────────────────
        ax2 = axes[1]
        self._draw_layers(ax2)
        profile = arr[:, fi]
        if meta["scale"] == "log":
            ax2.semilogx(profile, y, color="#c0392b", linewidth=2.0)
            ax2.set_xlabel(meta["label"])
        else:
            ax2.plot(profile, y, color="#c0392b", linewidth=2.0)
            ax2.set_xlabel(meta["label"])

        ax2.set_title(f"(b) Profile at\n{self._tlabel(fi)}")
        ax2.invert_yaxis()
        ax2.set_ylim(y.min(), y.max())
        ax2.grid(axis="x", linestyle=":", alpha=0.4)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        # Layer legend
        patches = self._layer_patches()
        axes[1].legend(handles=patches, fontsize=7,
                       loc="lower right", title="Layer", title_fontsize=7)

        fig.tight_layout()
        return fig

    # ── 2-D cloud map: x × y at one snapshot ─────────────────────────────────
    def _plot_2d_cloudmap(self, name, arr, norm, frame, meta) -> plt.Figure:
        fi = frame if frame >= 0 else self.n_snaps + frame
        x  = self.xp if name == "sigmaqs" else self.x
        y  = self.yp[1:len(self.y)] if name == "sigmaqs" else self.y
        X, Y = np.meshgrid(x, y)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        fig.suptitle(f"Cloud Map — {meta['title']}  |  {self._tlabel(fi)}",
                     fontsize=13, fontweight="bold")

        # ── left: full 2-D field ───────────────────────────────────────
        ax = axes[0]
        data_2d = arr[:, :, fi] if arr.ndim == 3 else arr
        # Trim to match coordinate grid if needed
        nr, nc = min(data_2d.shape[0], Y.shape[0]), min(data_2d.shape[1], X.shape[1])
        cf = ax.pcolormesh(X[:nr, :nc], Y[:nr, :nc], data_2d[:nr, :nc],
                           norm=norm, cmap=meta["cmap"],
                           shading="auto", rasterized=True)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.06)
        cb  = fig.colorbar(cf, cax=cax, label=meta["label"])
        cb.ax.tick_params(labelsize=8)

        self._draw_layers_2d(ax)
        ax.axvline(0, color="lime", linewidth=1.5,
                   linestyle="--", label="Fault (x=0)")
        ax.set_xlabel("x  [m]");  ax.set_ylabel("y  [m]")
        ax.set_title("(a) Full 2-D field")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="upper right")

        # ── right: fault-column profile vs depth ──────────────────────
        ax2 = axes[1]
        self._draw_layers(ax2)
        mid_col = data_2d.shape[1] // 2
        profile = data_2d[:, mid_col]
        ax2.plot(profile, y[:nr], color="#1565c0", linewidth=2.0,
                 label="Fault column")

        # Also plot left/right neighbours
        if mid_col > 0:
            ax2.plot(data_2d[:nr, mid_col-1], y[:nr],
                     color="#e57373", linewidth=1.0, linestyle="--",
                     label="Left neighbour")
        if mid_col < data_2d.shape[1] - 1:
            ax2.plot(data_2d[:nr, mid_col+1], y[:nr],
                     color="#66bb6a", linewidth=1.0, linestyle="--",
                     label="Right neighbour")

        ax2.set_xlabel(meta["label"])
        ax2.set_title("(b) Fault-column\nprofile")
        ax2.invert_yaxis()
        ax2.set_ylim(y.min(), y.max())
        ax2.legend(fontsize=7)
        ax2.grid(axis="x", linestyle=":", alpha=0.4)
        ax2.yaxis.tick_right()

        # Layer legend
        patches = self._layer_patches()
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
                  fontsize=7, loc="upper right",
                  title="Layer / curves", title_fontsize=7)

        fig.tight_layout()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # Animation
    # ─────────────────────────────────────────────────────────────────────────
    def animate_field(self, name: str, fps: int = 10,
                      dpi: int = 100) -> str:
        """
        Produce a GIF animation of *name* sweeping through all time snapshots.
        Returns the path to the saved GIF.
        """
        meta  = FIELDS[name]
        arr   = self._get_field(name)
        norm  = self._norm(name, arr)           # fixed colour scale for all frames
        fname = str(self.out / f"anim_{name}.gif")

        if meta["dim"] == "1d":
            fig, anim_obj = self._animate_1d(name, arr, norm, meta, fps)
        else:
            fig, anim_obj = self._animate_2d(name, arr, norm, meta, fps)

        print(f"  Saving {fname} …", end=" ", flush=True)
        anim_obj.save(fname, writer="pillow", fps=fps, dpi=dpi)
        print("done.")
        plt.close(fig)
        return fname

    # ── 1-D animation ─────────────────────────────────────────────────────────
    def _animate_1d(self, name, arr, norm, meta, fps):
        tm_yr = self.tm / self.yr
        y     = self.y

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")

        # Draw layer bands once
        self._draw_layers(ax)

        if meta["scale"] == "log":
            line, = ax.semilogx(arr[:, 0], y, color="#e53935",
                                linewidth=2.2, zorder=3)
            ax.set_xlim(norm.vmin, norm.vmax)
        else:
            line, = ax.plot(arr[:, 0], y, color="#e53935",
                            linewidth=2.2, zorder=3)
            ax.set_xlim(np.nanmin(arr), np.nanmax(arr))

        ax.invert_yaxis()
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel("y  [m]")
        ax.set_xlabel(meta["label"])
        ax.set_title(meta["title"], fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        # Time text and progress bar
        t_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         va="top", fontsize=10, color="#333",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", alpha=0.85))
        # Coloured velocity indicator bar on right edge
        ax2 = ax.twinx()
        ax2.set_ylim(y.max(), y.min())
        fill_obj = [ax2.fill_betweenx(y, arr[:, 0], arr[:, 0],
                                       color="#c0392b", alpha=0.15)]
        ax2.set_yticks([])
        ax2.spines["right"].set_visible(False)

        # Layer legend
        patches = self._layer_patches()
        ax.legend(handles=patches, fontsize=7, loc="lower right",
                  title="Layer", title_fontsize=7, framealpha=0.85)

        def update(frame):
            profile = arr[:, frame]
            line.set_xdata(profile)
            t_text.set_text(self._tlabel(frame) +
                            f"\nmax = {profile.max():.3g} {meta['unit']}")
            # Update fill
            for coll in fill_obj[0].get_paths():
                pass
            ax2.cla()
            ax2.set_ylim(y.max(), y.min())
            ax2.fill_betweenx(y, np.nanmin(arr), profile,
                               color="#c0392b", alpha=0.12)
            ax2.set_yticks([])
            ax2.spines["right"].set_visible(False)
            return line, t_text

        ani = animation.FuncAnimation(
            fig, update, frames=self.n_snaps,
            interval=1000 // fps, blit=False)

        return fig, ani

    # ── 2-D animation ─────────────────────────────────────────────────────────
    def _animate_2d(self, name, arr, norm, meta, fps):
        x   = self.xp if name == "sigmaqs" else self.x
        y_c = self.yp[1:len(self.y)] if name == "sigmaqs" else self.y
        X, Y = np.meshgrid(x, y_c)
        nr   = min(arr.shape[0], Y.shape[0])
        nc   = min(arr.shape[1], X.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                                 gridspec_kw={"width_ratios": [3, 1]},
                                 sharey=True)
        fig.patch.set_facecolor("white")
        fig.suptitle(meta["title"], fontsize=12, fontweight="bold")

        # ── left: 2-D mesh ────────────────────────────────────────────
        ax   = axes[0]
        data0 = arr[:nr, :nc, 0]
        mesh = ax.pcolormesh(X[:nr, :nc], Y[:nr, :nc], data0,
                             norm=norm, cmap=meta["cmap"],
                             shading="auto", rasterized=True)

        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3%", pad=0.06)
        cb  = fig.colorbar(mesh, cax=cax, label=meta["label"])
        cb.ax.tick_params(labelsize=8)

        ax.axvline(0, color="lime", linewidth=1.5,
                   linestyle="--", label="Fault (x=0)")
        self._draw_layers_2d(ax)
        ax.set_xlabel("x  [m]");  ax.set_ylabel("y  [m]")
        ax.set_title("(a) 2-D field")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="upper right")

        t_text = ax.text(0.02, 0.97, self._tlabel(0),
                         transform=ax.transAxes, va="top", fontsize=9,
                         color="white", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="#333", alpha=0.7))

        # ── right: fault-column profile ────────────────────────────────
        ax2  = axes[1]
        self._draw_layers(ax2)
        mid  = data0.shape[1] // 2
        line_fault, = ax2.plot(data0[:, mid], y_c[:nr],
                               color="#1565c0", linewidth=2.0,
                               label="Fault col.")
        ax2.set_xlabel(meta["label"])
        ax2.set_title("(b) Profile")
        ax2.invert_yaxis()
        ax2.set_ylim(y_c.min(), y_c.max())
        ax2.legend(fontsize=8, loc="lower right")
        ax2.grid(axis="x", linestyle=":", alpha=0.4)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        # Fixed x-limits for the profile plot
        ax2.set_xlim(np.nanmin(arr), np.nanmax(arr))

        # Progress bar at bottom
        bar_ax = fig.add_axes([0.08, 0.01, 0.84, 0.015])
        bar_ax.set_xlim(0, self.n_snaps - 1)
        bar_ax.set_ylim(0, 1)
        bar_ax.axis("off")
        bar_bg = bar_ax.barh(0.5, self.n_snaps - 1, height=1,
                             color="#ddd", align="center")[0]
        bar_fg = bar_ax.barh(0.5, 0, height=1,
                             color="#e53935", align="center")[0]

        def update(frame):
            data_f = arr[:nr, :nc, frame]
            mesh.set_array(data_f.ravel())
            t_text.set_text(self._tlabel(frame))
            line_fault.set_xdata(data_f[:, mid])
            bar_fg.set_width(frame)
            return mesh, t_text, line_fault, bar_fg

        ani = animation.FuncAnimation(
            fig, update, frames=self.n_snaps,
            interval=1000 // fps, blit=False)

        return fig, ani

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: animate every supported field
    # ─────────────────────────────────────────────────────────────────────────
    def animate_all(self, fps: int = 10, dpi: int = 100) -> list[str]:
        """Generate GIF animations for all fields present in data."""
        saved = []
        for name in FIELDS:
            if name in self.data or name in {"V", "U", "tau", "sigma",
                                              "tauqs", "sigmaqs"}:
                # map storage key names
                key_map = {"V": "Vm", "U": "Um", "tau": "taum",
                           "sigma": "sigmam", "tauqs": "taumall",
                           "sigmaqs": "sigmamall"}
                src = key_map.get(name, name)
                if src not in self.data:
                    continue
                print(f"Animating {name} …")
                path = self.animate_field(name, fps=fps, dpi=dpi)
                saved.append(path)
        return saved

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience: plot all static cloud maps
    # ─────────────────────────────────────────────────────────────────────────
    def plot_all(self, frame: int = -1) -> list[plt.Figure]:
        figs = []
        for name in FIELDS:
            key_map = {"V": "Vm", "U": "Um", "tau": "taum",
                       "sigma": "sigmam", "tauqs": "taumall",
                       "sigmaqs": "sigmamall"}
            src = key_map.get(name, name)
            if src not in self.data:
                continue
            print(f"  Plotting cloudmap: {name}")
            fig = self.plot_field(name, frame=frame)
            figs.append(fig)
        return figs


# ─────────────────────────────────────────────────────────────────────────────
# Data loader helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_npz(path: str) -> dict:
    """Load a dataall.npz saved by OutputManager.save_all()."""
    raw = dict(np.load(path))
    # Remap storage names to field names for CloudMapper
    key_map = {"Vm": "V", "Um": "U", "taum": "tau", "sigmam": "sigma",
               "taumall": "tauqs", "sigmamall": "sigmaqs"}
    data = {}
    for k, v in raw.items():
        data[key_map.get(k, k)] = v
    return data


def _make_demo_data() -> dict:
    """
    Build synthetic time-evolving fields without running the full model.
    Used when no data file is supplied.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import (ModelParameters, Grid, FrictionalZones,
                                  StressState, FaultState)

    p      = ModelParameters(Nx=101, Ny=101)
    grid   = Grid(p)
    fric   = FrictionalZones(p, grid.y)
    stress = StressState(p, grid.y)

    yr       = 365 * 24 * 3600
    n_snaps  = 60
    tm       = np.linspace(0, 500 * yr, n_snaps)
    Ny, Nx   = p.Ny, p.Nx
    y        = grid.y

    # Slip velocity: nucleation in sandstone then propagation
    Vm = np.zeros((Ny, n_snaps))
    for i in range(n_snaps):
        f = i / (n_snaps - 1)
        V = np.full(Ny, p.Vi)
        nuc = (y > 850) & (y <= 1050)
        yn  = (y[nuc] - 850) / 200.0
        V[nuc] = p.Vi * np.exp(f * 28 * yn * (1 - yn) * 4)
        if f > 0.45:
            spread = (y > 700) & (y <= 1250)
            V[spread] = np.maximum(V[spread], p.Vi * np.exp((f - 0.45) * 12))
        Vm[:, i] = np.maximum(V, p.Vi)

    # Cumulative slip
    dts = np.diff(np.append(0, tm))
    Um  = np.cumsum(Vm * dts[None, :], axis=1)

    # 1-D stress profiles
    taum   = stress.tau0[:, None] + 0.4e6 * np.sin(np.pi * y[:, None] / p.ysize) \
             * np.sin(np.linspace(0, np.pi, n_snaps)[None, :])
    sigmam = stress.sigman0[:, None] - 0.25e6 * np.outer(
             (y > 850) & (y <= 1050), np.linspace(0, 1, n_snaps))

    # 2-D shear stress (tauqs)
    X2, Y2 = np.meshgrid(grid.x, y)
    tauqs = np.zeros((Ny, Nx, n_snaps))
    for i in range(n_snaps):
        f = i / (n_snaps - 1)
        bg   = stress.tau0[:, None] * np.ones((Ny, Nx))
        blob = 0.9e6 * np.exp(-((Y2 - 950)**2 / 130**2
                                 + X2**2 / 280**2)) * np.sin(f * np.pi)
        wave = 0.15e6 * np.sin(2 * np.pi * X2 / p.xsize) \
                      * np.sin(np.pi * Y2 / p.ysize) * f
        tauqs[:, :, i] = bg + blob + wave

    # 2-D normal stress (sigmaqs)
    Xm, Ym = np.meshgrid(grid.x[1:Nx], grid.yp[1:Ny])
    sigmaqs = np.zeros((Ny - 1, Nx - 1, n_snaps))
    for i in range(n_snaps):
        f = i / (n_snaps - 1)
        bg        = stress.sigman0[1:Ny, None] * np.ones((Ny - 1, Nx - 1))
        depletion = -0.6e6 * f * np.exp(-((Ym - 950)**2 / 180**2
                                           + Xm**2 / 350**2))
        rebound   = 0.1e6 * f * np.exp(-((Ym - 1200)**2 / 250**2
                                          + Xm**2 / 500**2))
        sigmaqs[:, :, i] = bg + depletion + rebound

    return {
        "V":       Vm,        "U":       Um,
        "tau":     taum,      "sigma":   sigmam,
        "tauqs":   tauqs,     "sigmaqs": sigmaqs,
        "tm": tm,
        "y":  grid.y,   "x":  grid.x,
        "xp": grid.xp,  "yp": grid.yp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cloud-map plots and GIF animations for the fault-slip model.")
    parser.add_argument("datafile", nargs="?", default=None,
                        help="Path to dataall.npz (optional; uses demo data if omitted)")
    parser.add_argument("--field", default=None,
                        choices=list(FIELDS.keys()),
                        help="Animate only this field (default: all)")
    parser.add_argument("--fps", type=int, default=12,
                        help="Frames per second for GIF (default: 12)")
    parser.add_argument("--no-anim", action="store_true",
                        help="Skip animation, static cloud maps only")
    parser.add_argument("--out", default="plots",
                        help="Output directory (default: plots/)")
    args = parser.parse_args()

    out = Path(args.out)

    # ── load data ─────────────────────────────────────────────────────
    if args.datafile:
        print(f"Loading {args.datafile} …")
        data = load_npz(args.datafile)
    else:
        print("No data file supplied — generating synthetic demo data …")
        data = _make_demo_data()

    cm = CloudMapper(data, out_dir=out)

    # ── static cloud maps ─────────────────────────────────────────────
    print("\nGenerating static cloud maps …")
    cm.plot_all(frame=-1)

    # ── animations ───────────────────────────────────────────────────
    if not args.no_anim:
        print("\nGenerating GIF animations …")
        if args.field:
            cm.animate_field(args.field, fps=args.fps)
        else:
            # Animate all fields
            key_map = {"V": "V", "U": "U", "tau": "tau", "sigma": "sigma",
                       "tauqs": "tauqs", "sigmaqs": "sigmaqs"}
            for name in FIELDS:
                if name in data:
                    cm.animate_field(name, fps=args.fps)

    print(f"\nAll output saved to  {out.resolve()}/")
    plt.show()


if __name__ == "__main__":
    main()