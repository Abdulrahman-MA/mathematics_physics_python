import numpy as np
import pyvista as pv

theta = np.linspace(0.01,np.pi-0.01,20)
phi = np.linspace(0.01,2*np.pi,60)
theta, phi = np.meshgrid(theta, phi)

# Sphere coords
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Unit vectors
e_theta = np.stack([np.cos(theta)*np.cos(phi),
                    np.cos(theta)*np.sin(phi),
                   -np.sin(theta)], axis=-1)
e_phi = np.stack([-np.sin(phi),
                  np.cos(phi),
                  np.zeros_like(phi)], axis=-1)

# Vector field: -sin(phi) θ^ - cot(theta)cos(phi) φ^
V = - np.sin(phi)[..., None] * e_theta - (np.cos(theta)/np.sin(theta))[..., None] * np.cos(phi)[..., None] * e_phi
V = V/np.linalg.norm(V,axis=-1,keepdims=True)

# Flatten
points = np.c_[x.flatten(), y.flatten(), z.flatten()]
vectors = np.c_[V[...,0].flatten(), V[...,1].flatten(), V[...,2].flatten()]

# Build PyVista dataset
pdata = pv.PolyData(points)
pdata["vectors"] = vectors

# Create arrows from vectors
arrows = pdata.glyph(orient="vectors", scale=False, factor=0.1)

# Plot
plotter = pv.Plotter()
sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)
plotter.add_mesh(sphere, color="lightgray", opacity=0.6)
plotter.add_mesh(arrows, color="blue")
plotter.export_gltf(r"C:\Users\Abdul\OneDrive\Desktop\TKillingVectorFieldShwartz.gltf")

# Save screenshots
plotter.show(screenshot="TKillingVectorFieldShwartz.png", window_size=[1920, 1080])   # Full HD
plotter.screenshot("TKillingVectorFieldShwartz_4K.png", window_size=[3840, 2160])   # 4K
