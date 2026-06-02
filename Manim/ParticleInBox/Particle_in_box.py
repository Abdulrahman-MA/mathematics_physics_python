from manim import *
import numpy as np

class SingleSurfac(ThreeDScene):
    def construct(self):
        N = 200
        X = np.linspace(0, 1, N)
        Y = np.linspace(0, 1, N)
        Z = np.sin(2 * np.pi * X[:, None]) * np.cos(2 * np.pi * Y[None, :])

        # Surface generator
        def make_surface(Z, axes):
            def func(u, v):
                i = int(u * (N - 1))
                j = int(v * (N - 1))
                return axes.c2p(u, v, Z[i, j])
            return Surface(
                func,
                u_range=[0, 1], v_range=[0, 1],
                resolution=(40, 40),
                fill_opacity=0.8,
                checkerboard_colors=[BLUE_D, BLUE_E],  # add grid
                stroke_width=0                        # restore hair
            )

        # Axes
        axes = ThreeDAxes(
            x_range=[0, 1],
            y_range=[0, 1],
            z_range=[-1, 1],
        )
        surface = make_surface(Z, axes)

        group = VGroup(axes, surface)

        # Camera orientation
        self.set_camera_orientation(phi=65*DEGREES, theta=45*DEGREES, distance=6)
        self.add(group)

        # Spin
        self.play(group.animate.rotate(angle=2 * PI, axis=UP), run_time=6)
        self.wait()
