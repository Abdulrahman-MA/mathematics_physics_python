import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# Create grid
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

# === CASE 1: Scalar function f(x, y) = x^2 + y^2
f = X**2 + Y**2

# Compute grad f
df_dx, df_dy = np.gradient(f, x[1]-x[0], y[1]-y[0])

# Compute div(grad f) = Laplacian
d2f_dx2 = np.gradient(df_dx, x[1]-x[0], axis=1)
d2f_dy2 = np.gradient(df_dy, y[1]-y[0], axis=0)
laplacian = d2f_dx2 + d2f_dy2

# === CASE 2: Vector field F = (x^2, y^2)
U = X**2
V = Y**2

# Compute divergence of F
dU_dx = np.gradient(U, x[1]-x[0], axis=1)
dV_dy = np.gradient(V, y[1]-y[0], axis=0)
div_F = dU_dx + dV_dy

# Compute grad(div F)
grad_div_F_x = np.gradient(div_F, x[1]-x[0], axis=1)
grad_div_F_y = np.gradient(div_F, y[1]-y[0], axis=0)

# === Plotting ===
fig = plt.figure(figsize=(18, 6))

# 1) Plot Laplacian (div grad f)
ax1 = fig.add_subplot(1, 3, 1)
c1 = ax1.contourf(X, Y, laplacian, levels=50, cmap='viridis')
ax1.set_title("div(grad f) = Laplacian")
fig.colorbar(c1, ax=ax1)
ax1.grid(True)

# 2) Plot grad(div F) as vectors
ax2 = fig.add_subplot(1, 3, 2)
magnitude = np.sqrt(grad_div_F_x**2 + grad_div_F_y**2)
q = ax2.quiver(X, Y, grad_div_F_x, grad_div_F_y, magnitude, cmap='plasma', scale=50)
ax2.set_title("grad(div F)")
ax2.grid(True)

# 3) 3D surface plot of f(x,y)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf = ax3.plot_surface(X, Y, f, cmap='coolwarm', edgecolor='k', alpha=0.8)
ax3.set_title("Surface plot of f(x,y) = x² + y²")
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('f(x,y)')
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10)

plt.tight_layout()
plt.savefig('vector fields.png',dpi=1600)
plt.show()
