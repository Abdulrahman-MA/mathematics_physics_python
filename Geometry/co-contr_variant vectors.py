import numpy as np
import matplotlib.pyplot as plt

# Set up original basis vectors
e1 = np.array([1, 0])
e2 = np.array([0, 1])
standard_basis = np.column_stack((e1, e2))

# Define a vector in standard basis
v = np.array([2, 1])

# Define a linear transformation (new basis)
T = np.array([[2, 1],
              [1, 3]])  # New basis matrix (columns are the new basis vectors)

# Contravariant transformation (vector in new coordinates)
T_inv = np.linalg.inv(T)
v_contra = T_inv @ v

# Covariant transformation (e.g. gradient or dual)
v_co = T.T @ v

# Setup plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', color='lightgray')
ax.set_title("Original and Transformed Grids with Vector Transformations")

# Plot original basis vectors
ax.quiver(0, 0, *e1, color='gray', scale=1, scale_units='xy', angles='xy', label='e1 (Standard Basis)')
ax.quiver(0, 0, *e2, color='gray', scale=1, scale_units='xy', angles='xy', label='e2 (Standard Basis)')

# Plot new basis vectors (columns of T)
ax.quiver(0, 0, *T[:, 0], color='orange', scale=1, scale_units='xy', angles='xy', label="New Basis Vector 1")
ax.quiver(0, 0, *T[:, 1], color='purple', scale=1, scale_units='xy', angles='xy', label="New Basis Vector 2")

# Plot original vector
ax.quiver(0, 0, *v, color='black', angles='xy', scale_units='xy', scale=1, label='Original Vector')

# Plot contravariant transformation
ax.quiver(0, 0, *v_contra, color='blue', angles='xy', scale_units='xy', scale=1, label='Contravariant (T⁻¹v)')

# Plot covariant transformation
ax.quiver(0, 0, *v_co, color='red', angles='xy', scale_units='xy', scale=1, label='Covariant (Tᵗv)')

# Draw transformed grid
grid_range = np.arange(-5, 6, 1)
for x in grid_range:
    points = np.array([T @ [x, y] for y in grid_range])
    ax.plot(points[:, 0], points[:, 1], color='orange', linewidth=0.5, alpha=0.5)

for y in grid_range:
    points = np.array([T @ [x, y] for x in grid_range])
    ax.plot(points[:, 0], points[:, 1], color='orange', linewidth=0.5, alpha=0.5)

# Legend and show
ax.legend()
plt.savefig("co-contra_variant_vectors.png",dpi=900)
plt.show()
