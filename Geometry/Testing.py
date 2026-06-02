import numpy as np
import pyvista as pv

# Schwarzschild radius
M = 1.0
r_s = 2 * M

# Define grid in r, theta (2D slice for visualization)
r = np.linspace(0.5, 5, 30)  # radial range
theta = np.linspace(0, 2*np.pi, 60)  # angular
R, Theta = np.meshgrid(r, theta)

# Convert to Cartesian (for plotting)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Killing vector field: ∂t (magnitude depends on g_tt = -(1 - 2M/r))
def killing_norm(r):
    return -(1 - 2*M/r)

# Compute norm on grid
Norm = killing_norm(R)

# Define direction of Killing vector (just in +z)
U = np.zeros_like(X)
V = np.zeros_like(Y)
W = np.ones_like(X)

# Scale direction by sign of norm (timelike/spacelike)
scale = np.sign(Norm)
W = W * scale

# Flatten arrays
points = np.column_stack([X.flatten(), Y.flatten(), np.zeros_like(X).flatten()])
vectors = np.column_stack([U.flatten(), V.flatten(), W.flatten()])
scalars = Norm.flatten()

# Create PyVista point cloud
pdata = pv.PolyData(points)
pdata["vectors"] = vectors
pdata["scalars"] = scalars

# Create arrows via glyphs
arrows = pdata.glyph(orient="vectors", scale=False, factor=0.2)

# Plot
plotter = pv.Plotter()
# Draw horizon
horizon = pv.Disc(inner=r_s-0.01, outer=r_s+0.01, c_res=200)
plotter.add_mesh(horizon, color="black", opacity=0.5)

# Add vector field
plotter.add_mesh(arrows, scalars="scalars", cmap=["blue", "white", "red"], clim=[-1,1])

plotter.add_text("Killing Vector Field Norm", font_size=14)
plotter.show_axes()
plotter.show()
