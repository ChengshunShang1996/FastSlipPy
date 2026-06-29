"""
plot_results.py
---------------
Reads dataall.npz produced by OutputManager.save_all() and plots
several fault-midpoint quantities as a function of time.

Usage
-----
    python plot_results.py                        # looks for output/dataall.npz
    python plot_results.py path/to/dataall.npz    # explicit path
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 0.  Load data
# ─────────────────────────────────────────────────────────────

npz_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/dataall.npz")

if not npz_path.exists():
    raise FileNotFoundError(f"Cannot find {npz_path}. "
                            "Pass the path as a command-line argument.")

d = np.load(npz_path)

tm      = d["tm"]          # (n,)  time [s]
taum    = d["taum"]        # (Ny, n)  shear stress [Pa]
sigmam  = d["sigmam"]      # (Ny, n)  effective normal stress [Pa]
Vm      = d["Vm"]          # (Ny, n)  slip rate [m/s]
Um      = d["Um"]          # (Ny, n)  cumulative slip [m]
thetam  = d["thetam"]      # (Ny, n)  state variable [s]

yr   = 365 * 24 * 3600.0
t_yr = tm / yr             # time in years

Ny = taum.shape[0]
mid = Ny // 2             # mid-depth fault node

# ─────────────────────────────────────────────────────────────
# 1.  Extract mid-point time series
# ─────────────────────────────────────────────────────────────

tau_mid   = taum[mid, :]
sigma_mid = sigmam[mid, :]
V_mid     = Vm[mid, :]
U_mid     = Um[mid, :]
theta_mid = thetam[mid, :]

ratio_mid = tau_mid / (sigma_mid + 1e-30)   # τ / σ_n

# ─────────────────────────────────────────────────────────────
# 2.  Plot
# ─────────────────────────────────────────────────────────────
fig = plt.figure()
plt.plot(tm, ratio_mid, color="steelblue", lw=1.2)
plt.ylabel(r"Friction")
plt.title(r"Shear-to-normal stress ratio  $\tau / \sigma_n$")
plt.grid(True, alpha=0.4)
plt.xlabel("Time [s]")
plt.grid(True, alpha=0.4)
#plt.ylim(0.65, 0.78)
plt.tight_layout()
plt.show()

fig = plt.figure()
plt.plot(U_mid, ratio_mid, color="steelblue", lw=1.2)
plt.ylabel(r"Friction")
#plt.title(r"Shear-to-normal stress ratio  $\tau / \sigma_n$")
plt.grid(True, alpha=0.4)
plt.xlabel("Cumulative Slip [m]")
plt.grid(True, alpha=0.4)
plt.ylim(0.65, 0.78)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 1, sharex=True)
fig.suptitle(f"Fault mid-point time series  (node {mid} / {Ny-1})", fontsize=13)

# ── (a) τ / σ ratio ──────────────────────────────────────────
ax = axes[0]
ax.plot(tm, ratio_mid, color="steelblue", lw=1.2)
ax.set_ylabel(r"$\tau\,/\,\sigma_n$")
ax.set_title(r"Shear-to-normal stress ratio  $\tau / \sigma_n$")
ax.grid(True, alpha=0.4)

# ── (b) shear stress τ ───────────────────────────────────────
ax = axes[1]
ax.plot(tm, tau_mid / 1e6, color="tomato", lw=1.2)
ax.set_ylabel(r"$\tau$ [MPa]")
ax.set_title("Shear stress")
ax.grid(True, alpha=0.4)

# # ── (c) effective normal stress σ ────────────────────────────
# ax = axes[2]
# ax.plot(tm, sigma_mid / 1e6, color="seagreen", lw=1.2)
# ax.set_ylabel(r"$\sigma_n$ [MPa]")
# ax.set_title("Effective normal stress")
# ax.grid(True, alpha=0.4)

# ax.set_xlabel("Time [s]")
# ax.grid(True, alpha=0.4)

ax = axes[2]
ax.semilogy(tm, np.abs(V_mid) + 1e-40, color="darkorange", lw=1.2)
ax.set_ylabel(r"$V$ [m/s]")
ax.set_title("Slip rate  (log scale)")
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(True, which="both", alpha=0.3)

# ── (e) cumulative slip U ────────────────────────────────────
ax = axes[3]
ax.plot(tm, U_mid, color="purple", lw=1.2)
ax.set_ylabel(r"$U$ [m]")
ax.set_title("Cumulative slip")
ax.set_xlabel("Time [s]")
ax.grid(True, alpha=0.4)

plt.tight_layout()
out_png = npz_path.parent / "midpoint_timeseries_part1.png"
fig.savefig(out_png, dpi=150)
print(f"Saved → {out_png}")
plt.show()

fig, axes = plt.subplots(2, 1, sharex=True)
fig.suptitle(f"Fault mid-point time series  (node {mid} / {Ny-1})", fontsize=13)

# ── (d) slip rate V  (log scale) ─────────────────────────────
ax = axes[0]
ax.semilogy(tm, np.abs(V_mid) + 1e-40, color="darkorange", lw=1.2)
ax.set_ylabel(r"$V$ [m/s]")
ax.set_title("Slip rate  (log scale)")
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(True, which="both", alpha=0.3)

# ── (e) cumulative slip U ────────────────────────────────────
ax = axes[1]
ax.plot(tm, U_mid, color="purple", lw=1.2)
ax.set_ylabel(r"$U$ [m]")
ax.set_title("Cumulative slip")
ax.set_xlabel("Time [s]")
ax.grid(True, alpha=0.4)

plt.tight_layout()
out_png = npz_path.parent / "midpoint_timeseries_part2.png"
fig.savefig(out_png, dpi=150)
print(f"Saved → {out_png}")
plt.show()

# ─────────────────────────────────────────────────────────────
# 3.  Bonus: depth–time colour map of τ / σ
# ─────────────────────────────────────────────────────────────

ratio_all = taum / (sigmam + 1e-30)   # (Ny, n)

fig2, ax2 = plt.subplots(figsize=(10, 4))
pcm = ax2.pcolormesh(tm,
                     np.arange(Ny),
                     ratio_all,
                     shading="auto",
                     cmap="RdBu_r")
ax2.axhline(mid, color="k", lw=1, ls="--", label=f"mid node ({mid})")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Fault node index  (0 = shallow)")
ax2.set_title(r"$\tau / \sigma_n$ along fault over time")
ax2.legend(fontsize=8)
fig2.colorbar(pcm, ax=ax2, label=r"$\tau / \sigma_n$")
plt.tight_layout()
out_png2 = npz_path.parent / "ratio_depth_time.png"
fig2.savefig(out_png2, dpi=150)
print(f"Saved → {out_png2}")
plt.show()