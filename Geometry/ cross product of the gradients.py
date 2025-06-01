import matplotlib.pyplot as plt
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width

# Curve one
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

fig = plt.figure(figsize=(16, 12))

ax = [fig.add_subplot(2, 2, 1, projection='3d'), fig.add_subplot(2, 2, 2, projection='3d'),
      fig.add_subplot(2, 2, 3, projection='3d'), fig.add_subplot(2, 2, 4, projection='3d'), ]

surf1 = ax[1].plot_surface(x, y, z,color='green', rstride=5, cstride=5,linewidth=0.5, edgecolor='k', alpha=0.1)

ax[1].set_title('Unit Sphere: $x^2 + y^2 + z^2 = 1$')
fig.colorbar(surf1, shrink=0.5, aspect=10)

#Curve Two
x2 = np.linspace(-1.2, 1.2, 50)
y2 = np.linspace(-1.2, 1.2, 50)
x2, y2 = np.meshgrid(x2, y2)
z2 = y2

surf2 = ax[1].plot_surface(x2, y2, z2, color='black', rstride=5, cstride=5,linewidth=0.5, edgecolor='k', alpha=0.5)
'''ax[1].set_title('Plane: $z=y$')
fig.colorbar(surf2, shrink=0.5, aspect=10)'''

for i in range(1, 2):
    ax[i].set_xlabel('X axis')
    ax[i].set_ylabel('Y axis')
    ax[i].set_zlabel('Z axis')

plt.tight_layout()

plt.show()
