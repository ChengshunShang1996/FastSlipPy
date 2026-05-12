import numpy as np
import matplotlib.pyplot as plt


def plot_results_from_file(filename="output/dataall.npz"):
    data = np.load(filename)

    Um = data["Um"]
    Vm = data["Vm"]
    taum = data["taum"]
    sigmam = data["sigmam"]
    tm = data["tm"]

    Ny, Nt = Um.shape
    y = np.linspace(2000, 4000, Ny)  # adjust if needed
    yr = 365 * 24 * 3600
    t_yr = tm / yr

    # ─────────────────────────────────────
    # 1. τ / σ vs depth (last timestep)
    # ─────────────────────────────────────
    tau = taum[:, -1]
    sigma = sigmam[:, -1]
    ratio = tau / (sigma + 1e-12)

    plt.figure(figsize=(5, 4))
    plt.plot(ratio, y, lw=2)
    plt.gca().invert_yaxis()
    plt.xlabel(r"$\tau / \sigma$")
    plt.ylabel("Depth [m]")
    plt.title("Stress ratio at final time")
    plt.grid()

    # ─────────────────────────────────────
    # 2. Slip velocity evolution
    # ─────────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.contourf(t_yr, y, np.abs(Vm) + 1e-40)
    plt.gca().invert_yaxis()
    plt.xlabel("Time [yr]")
    plt.ylabel("Depth [m]")
    plt.title("Slip velocity")
    plt.colorbar()

    plt.figure(figsize=(6, 4))
    

    plt.figure(figsize=(6, 4))
    plt.plot(np.abs(Vm[:, -1]), y, lw=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Slip velocity at final time")
    plt.ylabel("Depth [m]")
    plt.title("Slip velocity at final time")
    plt.grid()


    # ─────────────────────────────────────
    # 3. Shear stress evolution
    # ─────────────────────────────────────
    plt.figure(figsize=(6, 4))
    plt.contourf(t_yr, y, taum / 1e6)
    plt.gca().invert_yaxis()
    plt.xlabel("Time [yr]")
    plt.ylabel("Depth [m]")
    plt.title("Shear stress [MPa]")
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results_from_file()