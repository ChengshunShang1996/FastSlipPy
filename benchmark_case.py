import numpy as np
import matplotlib.pyplot as plt

from main import (
    ModelParameters,
    Grid,
    compute_stress_fields
)

# ============================================================
# Benchmark parameters
# ============================================================

params = ModelParameters(
    Nx=21,
    Ny=21,
    xsize=2000,
    ysize=2000
)

grid = Grid(params)

dx = grid.dx
dy = grid.dy

lam = params.lam
G   = params.G

cosa = grid.cosa
sina = grid.sina

Nx = params.Nx
Ny = params.Ny

# ============================================================
# CASE SELECTION
# ============================================================

CASE = "shear"

# CASE = "linear"

# ============================================================
# Allocate staggered fields
# ============================================================

ux = np.zeros((Ny + 1, Nx))
uy = np.zeros((Ny, Nx + 1))

# ============================================================
# CASE A : linear extension
# ============================================================

if CASE == "linear":

    A = 1e-5
    B = -2e-5

    # ux on ux-grid
    ux = A * grid.Xux

    # uy on uy-grid
    uy = B * grid.Yuy

    # ----------------------------------------
    # Analytical stresses
    # ----------------------------------------

    sigma_exact = (
        (lam + 2*G) * A
        + lam * B
    )

    tau_exact = 0.0

# ============================================================
# CASE B : pure shear
# ============================================================

elif CASE == "shear":

    gamma = 1e-5

    # ux = gamma * y
    ux = gamma * grid.Yux

    # uy = 0
    uy[:] = 0.0

    # ----------------------------------------
    # Analytical stresses
    # ----------------------------------------

    tau_exact = (
            G / sina
            * (1 - 2*cosa**2)
            * gamma
        )

    sigma_exact = 0.0

else:
    raise ValueError("Unknown CASE")

# ============================================================
# Compute numerical stresses
# ============================================================

tauqs, sigmaqs = compute_stress_fields(
    uy,
    ux,
    dx,
    dy,
    lam,
    G,
    cosa,
    sina,
    Ny,
    Nx
)

# ============================================================
# Error analysis
# ============================================================

tau_error = tauqs - tau_exact

sigma_error = sigmaqs - sigma_exact

print("\n==============================")
print("Manufactured Solution Benchmark")
print("==============================")

print(f"\nCASE = {CASE}")

print("\n--- Tau error ---")
print(f"Max error : {np.max(np.abs(tau_error)):.6e}")
print(f"L2 error  : {np.sqrt(np.mean(tau_error**2)):.6e}")

print("\n--- Sigma error ---")
print(f"Max error : {np.max(np.abs(sigma_error)):.6e}")
print(f"L2 error  : {np.sqrt(np.mean(sigma_error**2)):.6e}")

# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ----------------------------------------
# tau numerical
# ----------------------------------------

im = axes[0,0].pcolormesh(
    grid.Xtau,
    grid.Ytau,
    tauqs,
    shading='auto'
)

axes[0,0].set_title("Numerical Tau")
fig.colorbar(im, ax=axes[0,0])

# ----------------------------------------
# tau error
# ----------------------------------------

im = axes[0,1].pcolormesh(
    grid.Xtau,
    grid.Ytau,
    tau_error,
    shading='auto'
)

axes[0,1].set_title("Tau Error")
fig.colorbar(im, ax=axes[0,1])

# ----------------------------------------
# sigma numerical
# ----------------------------------------

im = axes[1,0].pcolormesh(
    grid.Xsigma,
    grid.Ysigma,
    sigmaqs,
    shading='auto'
)

axes[1,0].set_title("Numerical Sigma")
fig.colorbar(im, ax=axes[1,0])

# ----------------------------------------
# sigma error
# ----------------------------------------

im = axes[1,1].pcolormesh(
    grid.Xsigma,
    grid.Ysigma,
    sigma_error,
    shading='auto'
)

axes[1,1].set_title("Sigma Error")
fig.colorbar(im, ax=axes[1,1])

for ax in axes.flat:
    ax.set_aspect('equal')
    ax.invert_yaxis()

plt.tight_layout()
plt.show()