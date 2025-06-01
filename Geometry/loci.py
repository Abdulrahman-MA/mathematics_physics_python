import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 1  # x and y axis scaling
c = 1  # z axis scaling
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(-2, 2, 100)
U, V = np.meshgrid(u, v)

# Parametric equations for one-sheeted hyperboloid
X = a * np.cosh(V) * np.cos(U)
Y = a * np.cosh(V) * np.sin(U)
Z = c * np.sinh(V)

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, color='red', alpha=0.8)

# Labels and aesthetics
ax.set_title("One-Sheeted Hyperboloid: $\\frac{x^2}{a^2} + \\frac{y^2}{b^2} - \\frac{z^2}{c^2} = 1$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=30, azim=30)
plt.tight_layout()
plt.savefig("one_sheeted_hyperboloid.png", dpi=300)
plt.show()
