from manim import *
import numpy as np
import random

class GaspresureGravityPalance(Scene):
    """Visualizes gravitational contraction, pressure, and temperature increase."""

    N_BODIES = 50
    N_BODIES2 = 2000
    BODY_RADIUS = 0.2
    BODY_RADIUS2 = 0.1
    INIT_RADIUS = 3.5
    CENTER = np.array([0, 0, 0])
    MOVE_TIME = 6

    def RandPosition(self):
        """Random position within a circle of INIT_RADIUS, for overlap avoidance."""
        while True:
            x = random.uniform(-self.INIT_RADIUS, self.INIT_RADIUS)
            y = random.uniform(-self.INIT_RADIUS, self.INIT_RADIUS)
            if x**2 + y**2 <= self.INIT_RADIUS**2:
                return np.array([x, y, 0])

    def color_temperature(self, frac):
        """Interpolate color from BLUE (cold) to YELLOW (medium) to RED (hot)."""
        if frac < 0.5:
            return interpolate_color(BLUE, YELLOW, frac * 2)
        else:
            return interpolate_color(YELLOW, RED, (frac - 0.5) * 2)

    def construct(self):



        N_LAYERS = 25
        INIT_RADIUS = 0
        WRAP_RADIUS = 3.5
        DELTA_RADIUS = WRAP_RADIUS / (N_LAYERS - 1)
        DOTS_PER_LAYER = 40
        SCALE_FACTOR = 0.7
        ARROW_LENGTH = 0.5
        N_ARROWS = 24

                # --- Color Bar Legend ---
        color_bar_width = 4
        n_steps = 100  # number of color steps in gradient

        # Create rectangles with interpolated colors
        gradient = VGroup()
        for i in range(n_steps):
            frac = i / (n_steps - 1)
            color = interpolate_color(BLUE_E, RED_E, frac)
            rect = Rectangle(
                width=color_bar_width / n_steps,
                height=0.3,
                fill_color=color,
                fill_opacity=1,
                stroke_width=0
            )
            rect.move_to(LEFT * color_bar_width / 2 + RIGHT * (i * color_bar_width / n_steps))
            gradient.add(rect)

        # Place the color bar at bottom right
        gradient.move_to(DOWN * 3.2 + RIGHT * 4.2)

        # Add text labels
        cold_label = Text("Cold", font_size=24).next_to(gradient, LEFT, buff=0.2)
        hot_label = Text("Hot", font_size=24).next_to(gradient, RIGHT, buff=0.2)

        color_bar_group = VGroup(gradient, cold_label, hot_label).scale(0.8)

        # Create star layers
        all_dots = VGroup()
        wrapping_circle = Dot(radius=WRAP_RADIUS, color=YELLOW_E, stroke_width=2)
        self.play(Create(wrapping_circle))

        for i in range(N_LAYERS):
            r = INIT_RADIUS + i * DELTA_RADIUS
            color = interpolate_color(RED_E, BLUE_E, i / N_LAYERS)
            radius = interpolate(0.08, 0.03, i / N_LAYERS)

            layer = VGroup()
            for j in range(DOTS_PER_LAYER):
                theta = 2 * PI * j / DOTS_PER_LAYER
                pos = r * np.array([np.cos(theta), np.sin(theta), 0])
                dot = Dot(point=pos, radius=radius, color=color)
                layer.add(dot)
            all_dots.add(layer)

        self.play(FadeIn(all_dots, lag_ratio=0.1), run_time=3)
        self.play(Write(color_bar_group))
        # Gravity arrows
        arrow_group = VGroup()
        arrow_directions = []

        for i in range(N_ARROWS):
            theta = 2 * PI * i / N_ARROWS
            direction = np.array([np.cos(theta), np.sin(theta), 0])
            arrow_directions.append(direction)

            end = WRAP_RADIUS * direction
            start = (WRAP_RADIUS + ARROW_LENGTH) * direction
            arrow = Arrow(start=start, end=end, stroke_width=2.5, buff=0, color=TEAL)
            arrow_group.add(arrow)

        self.play(LaggedStart(*[GrowArrow(a) for a in arrow_group], lag_ratio=0.05), run_time=2)

        # Legend
        g_arrow = Arrow(LEFT * 0.5, ORIGIN, buff=0, color=TEAL)
        g_label = Text("Gravity", font_size=24).next_to(g_arrow, RIGHT, buff=0.2)
        p_arrow = Arrow(LEFT * 0.5, ORIGIN, buff=0, color=RED_E)
        p_label = Text("Pressure", font_size=24).next_to(p_arrow, RIGHT, buff=0.2)
        legend = VGroup(VGroup(g_arrow, g_label), VGroup(p_arrow, p_label))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend.to_corner(UL).shift(DOWN * 0.3 + RIGHT * 0.3)
        self.play(FadeIn(legend), run_time=1)
        self.wait()

        # Contract star and gravity arrows
        self.play(
            all_dots.animate.scale(SCALE_FACTOR),
            wrapping_circle.animate.scale(SCALE_FACTOR),
            *[
                arrow.animate.put_start_and_end_on(
                    (WRAP_RADIUS * SCALE_FACTOR + ARROW_LENGTH) * dir,
                    WRAP_RADIUS * SCALE_FACTOR * dir
                )
                for arrow, dir in zip(arrow_group, arrow_directions)
            ],
            run_time=2
        )
        self.wait()

        # Pressure wave and big arrows
        final_radius = WRAP_RADIUS * SCALE_FACTOR
        pressure_wave = Circle(radius=0.1, color=RED_E, stroke_width=4)
        pressure_wave.set_fill(RED_E, opacity=0.4)
        pressure_wave.move_to(ORIGIN)

        big_arrows = VGroup()
        for dir in arrow_directions:
            a = Arrow(
                start=ORIGIN,
                end=ORIGIN,
                buff=0,
                color=RED_E,
                stroke_width=2.5
            )
            big_arrows.add(a)

        self.add(pressure_wave, big_arrows)

        target_arrows = VGroup()
        for dir in arrow_directions:
            new_arrow = Arrow(
                start=ORIGIN,
                end=final_radius * dir,
                color=RED_E,
                buff=0,
                stroke_width=2.5
            )
            target_arrows.add(new_arrow)

        self.play(
            AnimationGroup(
                pressure_wave.animate.set(width=2 * final_radius).set(opacity=0),
                *[
                    Transform(a, new_a)
                    for a, new_a in zip(big_arrows, target_arrows)
                ],
                lag_ratio=0,
                run_time=2
            )
        )


        # Big arrows disappear tail to tip
        self.play(
            AnimationGroup(
                *[
                    a.animate.put_start_and_end_on(
                        final_radius * dir * 0.9,
                        final_radius * dir
                    )
                    for a, dir in zip(big_arrows, arrow_directions)
                ],
                FadeOut(pressure_wave),
                lag_ratio=0,
                run_time=1.5
            )
        )

        # Expand star and replace with final small pressure arrows
        expanded_radius = WRAP_RADIUS  # Back to original size
        expanded_arrow_group = VGroup()

        for dir in arrow_directions:
            start = (expanded_radius - ARROW_LENGTH) * dir
            end = expanded_radius * dir
            new_arrow = Arrow(start=start, end=end, color=RED_E, buff=0, stroke_width=2.5)
            expanded_arrow_group.add(new_arrow)

        self.play(
            all_dots.animate.scale(1 / SCALE_FACTOR),
            wrapping_circle.animate.scale(1 / SCALE_FACTOR),
            *[
                Transform(arrow, new_arrow)
                for arrow, new_arrow in zip(big_arrows, expanded_arrow_group)
            ],
            *[
                arrow.animate.put_start_and_end_on(
                    (expanded_radius + ARROW_LENGTH) * dir,
                    expanded_radius * dir
                )
                for arrow, dir in zip(arrow_group, arrow_directions)
            ],
            run_time=2
        )

        # Now final small pressure arrows (fade in over existing ones)
        final_pressure_arrows = VGroup()
        for dir in arrow_directions:
            start = (expanded_radius - ARROW_LENGTH) * dir
            end = expanded_radius * dir
            arrow = Arrow(start=start, end=end, color=RED_E, buff=0, stroke_width=2.5)
            final_pressure_arrows.add(arrow)

        self.play(
            FadeOut(big_arrows),
            FadeIn(final_pressure_arrows),
            run_time=1.5
        )
        # Fade out all elements
        self.play(
            FadeOut(all_dots),
            FadeOut(wrapping_circle),
            FadeOut(arrow_group),
            FadeOut(color_bar_group),
            FadeOut(legend),
            FadeOut(final_pressure_arrows),
            run_time=1.5
        )
        self.wait()