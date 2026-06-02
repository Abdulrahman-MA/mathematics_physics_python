import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm

# ================================
# PARAMETERS
# ================================
basepath = "ParticleInBox/"
os.makedirs(basepath, exist_ok=True)

Lx, Ly = 1.0, 1.0   # Box size is exactly 1
N = 200
xs = np.linspace(0, Lx, N)
ys = np.linspace(0, Ly, N)

hbar = 1.0
m = 1.0

fps = 60
seconds_per_state = 5
timesteps = seconds_per_state * fps

state_list = [(nx, ny) for nx in range(1, 4) for ny in range(1, 4)]  # (1,1) .. (3,3)
total_frames = len(state_list) * timesteps

# ================================
# FUNCTIONS
# ================================
def energy(n_x, n_y, Lx, Ly, hbar, m):
    return (hbar**2 * np.pi**2 / (2*m)) * ((n_x**2 / Lx**2) + (n_y**2 / Ly**2))

@njit(parallel=True, fastmath=True)
def apply_phase(spatial_grid, t, E, hbar):
    """Multiply spatial grid by time-dependent phase."""
    N = spatial_grid.shape[0]
    out = np.empty((N, N), dtype=np.complex128)
    phase = np.exp(-1j * E * t / hbar)
    for i in prange(N):
        for j in range(N):
            out[i, j] = spatial_grid[i, j] * phase
    return out

# ================================
# MAIN SIMULATION
# ================================
data_path = os.path.join(basepath, "wavefunction_evolution.dat")
data = np.memmap(data_path, dtype=np.float32, mode="w+", shape=(total_frames, N, N, 3))

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((N, N)), extent=[xs.min(), xs.max(), ys.min(), ys.max()],
               origin="lower", cmap="viridis", vmin=0, vmax=1)
ax.set_title("Wavefunction |ψ|² (evolving)")

frame_index = 0
global_mag_max = 0.0

for (n_x, n_y) in state_list:
    # Normalization factor √(2/Lx) √(2/Ly)
    norm_factor = np.sqrt(2/Lx) * np.sqrt(2/Ly)

    # Spatial part (time-independent, normalized)
    spatial_grid = norm_factor * np.outer(
        np.sin(n_x * np.pi * xs / Lx),
        np.sin(n_y * np.pi * ys / Ly)
    )

    # Probability density |ψ|² (time-independent)
    mag2_grid = (spatial_grid**2).astype(np.float32)
    local_max = float(mag2_grid.max())
    if local_max > global_mag_max:
        global_mag_max = local_max

    E = energy(n_x, n_y, Lx, Ly, hbar, m)

    for frame in tqdm(range(timesteps), desc=f"State ({n_x},{n_y})"):
        t = frame / fps
        psi_grid = apply_phase(spatial_grid, t, E, hbar)

        # Store [Re, Im, |ψ|²]
        data[frame_index, :, :, 0] = np.real(psi_grid)
        data[frame_index, :, :, 1] = np.imag(psi_grid)
        data[frame_index, :, :, 2] = mag2_grid

        # Live plot every ~5 frames
        if frame % 5 == 0:
            im.set_data(mag2_grid)
            ax.set_title(f"State ({n_x},{n_y}), frame {frame}")
            plt.pause(0.001)

        frame_index += 1

plt.ioff()
plt.show()

# Flush to disk
data.flush()
del data

# Reopen read-only
data = np.memmap(data_path, dtype=np.float32, mode="r", shape=(total_frames, N, N, 3))
print("Saved data with shape:", data.shape)
