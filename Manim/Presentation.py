from manim import *
import random
import numpy as np

class GravitationalAttraction(Scene):

    def RandPosition(self):
        # Generate a random position within a rectangle for overlap avoidance
        x = random.uniform(-6, 6)
        y = random.uniform(-3, 3)
        return np.array([x, y, 0])

    def construct(self):
        radius = 0.5

        # === Two main bodies ===
        body1 = Circle(radius=radius, color=BLUE)
        body2 = Circle(radius=radius, color=RED)

        body1.move_to(LEFT * 3)
        body2.move_to(RIGHT * 3)

        label1 = Text("Body 1", font_size=24).next_to(body1, DOWN)
        label2 = Text("Body 2", font_size=24).next_to(body2, DOWN)

        body1_group = VGroup(body1, label1)
        body2_group = VGroup(body2, label2)

        # Animate creation of the two bodies and their labels
        self.play(Create(body1), Write(label1), run_time=2)
        self.play(Create(body2), Write(label2), run_time=2)
        self.wait(0.5)

        # Function to create an arrow from the surface of one circle to another
        def surface_arrow(start_circle, target_circle, length=0.8):
            direction = normalize(target_circle.get_center() - start_circle.get_center())
            start = start_circle.get_center() + direction * radius
            end = start + direction * length
            return Arrow(start=start, end=end, color=YELLOW, stroke_width=6, buff=0)

        # Always redraw arrows as the bodies move
        arrow1 = always_redraw(lambda: surface_arrow(body1, body2))
        arrow2 = always_redraw(lambda: surface_arrow(body2, body1))

        self.play(Create(arrow1), Create(arrow2))
        self.wait(0.5)
        
        # Display Newton's law of gravitation
        equation1 = MathTex(
            "F = G \\frac{m_1 m_2}{r_{12}^2}", font_size=40
        )

        equation1.to_corner(UP + LEFT)
        self.play(Write(equation1), run_time=2)
        self.wait(0.5)

        # Create axes for force vs distance graph
        axes = Axes(
            x_range=[0.4472135955, 10, 1],
            y_range=[0, 5, 1],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": [2, 4, 6, 8, 10]},
            y_axis_config={"numbers_to_include": [1, 2, 3, 4, 5]},
        )
        axes.scale(0.6).to_corner(DOWN + LEFT)

        # Axis labels
        x_label = axes.get_x_axis_label("r", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("F", edge=UP)

        # Constants for force calculation
        G = 1
        m1 = 1
        m2 = 1

        # Force as a function of distance
        def force_func(r):
            return G * m1 * m2 / (r**2)

        # Plot the force function
        force_graph = axes.plot(force_func, color=YELLOW)

        # Track the distance between the two bodies
        distance_tracker = ValueTracker(
            np.linalg.norm(body1.get_center() - body2.get_center())
        )

        # Update the tracker as the bodies move
        distance_tracker.add_updater(
            lambda m, dt: m.set_value(
                np.linalg.norm(body1.get_center() - body2.get_center())
            )
        )

        self.add(distance_tracker)

        # Dot showing the current force value on the graph
        moving_dot = Dot(color=RED).move_to(axes.c2p(distance_tracker.get_value(), force_func(distance_tracker.get_value())))

        # Label showing the current force value
        force_label = always_redraw(lambda: 
            MathTex(f"F = {force_func(distance_tracker.get_value()):.2f}", font_size=32)
            .next_to(moving_dot, UP)
        )

        # Update the dot position as the distance changes
        moving_dot.add_updater(
            lambda d: d.move_to(axes.c2p(distance_tracker.get_value(), force_func(distance_tracker.get_value())))
        )

        # Animate creation of axes, graph, and moving dot/label
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=2)
        self.play(Create(force_graph), run_time=2)
        self.add(moving_dot, force_label)
        self.wait()
        # Animate the two bodies moving closer together and then apart
        self.play(
            body1_group.animate.shift(RIGHT * 2.5),
            body2_group.animate.shift(LEFT * 2.5),
            run_time=3,
            rate_func=smooth
        )

        self.wait(1)

        self.play(
            body1_group.animate.shift(LEFT * 2.5),
            body2_group.animate.shift(RIGHT * 2.5),
            run_time=3,
            rate_func=smooth
        )

        self.wait()
        # Remove all objects related to the graph and bodies
        self.play(
            Uncreate(axes),
            Uncreate(x_label),
            Uncreate(y_label),
            Uncreate(moving_dot),
            Uncreate(force_label),
            Uncreate(force_graph),
            Uncreate(body1_group),
            Uncreate(body2_group),
            Uncreate(arrow1),
            Uncreate(arrow2)
        )
  
        # Show the multi-body gravitational equation
        equation2 = MathTex(
            "+", 
            "G \\frac{m_1 m_3}{r_{13}^2}", "+", 
            "G \\frac{m_1 m_4}{r_{14}^2}", "+", 
            " \\dots", 
            font_size=40
        )
        equation3 = MathTex("F","=","G","\\sum_{i=1}^{n} x_i","\\frac{m_1 m_i}{r_{1i}^2}",font_size=40)
        equation3.to_corner(UP + LEFT)

        # Create random bodies with overlap avoidance
        bodies = []
        min_dist = 2 * (radius - 0.2) + 0.1  

        for _ in range(20):
            max_attempts = 100
            for attempt in range(max_attempts):
                position = self.RandPosition()
                if all(np.linalg.norm(position - b.get_center()) > min_dist for b in bodies):
                    break  
            else:
                continue  

            color = random.choice([
                RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK, TEAL, MAROON, GOLD
            ])
            body = Circle(radius=radius - 0.2, color=color)
            body.move_to(position)
            self.play(DrawBorderThenFill(body), run_time=0.1)
            bodies.append(body)

        # Show the extended equation next to the first
        equation2.next_to(equation1[-1], RIGHT, buff=0.1)
        self.play(Write(equation2), run_time=2)
        self.wait(0.5)

        # Find the furthest body from the center to use as a reference
        ref_point = np.array([0, 0, 0])
        furthest_body = max(bodies, key=lambda b: np.linalg.norm(b.get_center() - ref_point))
        arrows = []
        for body in bodies:
            if body == furthest_body:
                continue
            start = furthest_body.get_center()
            end = body.get_center()
            arrow = Arrow(
                start=start,
                end=end,
                color=body.get_color(),
                stroke_width=4,
                max_tip_length_to_length_ratio=0.08
            )
            arrows.append(arrow)

        # Animate arrows from the furthest body to all others
        self.play(*[GrowArrow(arrow) for arrow in arrows], run_time=1.5)

        # Split each arrow into x and y components
        x_components = []
        y_components = []

        for arrow in arrows:
            start = arrow.get_start()
            end = arrow.get_end()

            x_end = np.array([end[0]/2, start[1], 0])
            dy = end[1] - start[1]
            y_end =np.array([start[0], dy/2, 0])

            color = arrow.get_color()

            x_arrow = Arrow(
                start=start,
                end=x_end,
                color=color,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.08
            )

            y_arrow = Arrow(
                start=start,
                end=y_end,
                color=color,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.08
            )

            x_components.append(x_arrow)
            y_components.append(y_arrow)

        # Animate the splitting of arrows into components
        split_anims = []
        for i in range(len(arrows)):
            self.play(
                Transform(arrows[i], x_components[i]),
                GrowArrow(y_components[i]),
                run_time=0.2
            )

        virtual_unit = 0.2

        # Sum the virtual x and y components for net force
        sum_virtual_x = np.array([0.0, 0.0, 0.0])
        sum_virtual_y = np.array([0.0, 0.0, 0.0])

        net_start = furthest_body.get_center()
        adjusted_x_arrows = []
        adjusted_y_arrows = []

        for i in range(len(x_components)):
            x_vec = x_components[i].get_end() - x_components[i].get_start()
            y_vec = y_components[i].get_end() - y_components[i].get_start()

            x_dir = x_vec / (np.linalg.norm(x_vec) + 1e-6)
            y_dir = y_vec / (np.linalg.norm(y_vec) + 1e-6)

            v_x = x_dir * virtual_unit
            v_y = y_dir * virtual_unit

            sum_virtual_x += v_x
            sum_virtual_y += v_y

            adjusted_x_arrows.append(
                Arrow(
                    start=net_start,
                    end=net_start + v_x,
                    color=x_components[i].get_color(),
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.08
                )
            )
            adjusted_y_arrows.append(
                Arrow(
                    start=net_start + v_x,
                    end=net_start + v_x + v_y,
                    color=y_components[i].get_color(),
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.08
                )
            )

        # Animate the addition of all x and y components
        self.play(
            *[GrowArrow(a) for a in adjusted_x_arrows + adjusted_y_arrows],
            run_time=2
        )

        # Remove old equations and show the vector sum equation
        self.play(Uncreate(equation1), Uncreate(equation2), run_time=2)
        self.play(Write(equation3),run_time=2)

        # Draw net force arrows and labels
        net_x_arrow = Arrow(
            start=net_start,
            end=net_start + sum_virtual_x,
            color=WHITE,
            stroke_width=5
        )
        
        net_y_arrow_moved = DashedVMobject(
            Arrow(
                start=net_start + sum_virtual_x,
                end=net_start + sum_virtual_x + sum_virtual_y,
                color=WHITE,
                stroke_width=5
            )
        )

        net_y_arrow = Arrow(
            start=net_start,
            end=net_start + sum_virtual_y,
            color=WHITE,
            stroke_width=5
        )
        resultant_arrow = Arrow(
            start=net_start,
            end=net_start + sum_virtual_x + sum_virtual_y,
            color=YELLOW,
            stroke_width=6
        )

        # Labels for the net force components
        label_x = MathTex(r"\sum F_x", color=WHITE).next_to(net_x_arrow, DOWN)
        label_y = MathTex(r"\sum F_y", color=WHITE).next_to(VGroup(net_y_arrow), LEFT)
        label_r = MathTex(r"\vec{F}_{\text{net}}", color=YELLOW).next_to(resultant_arrow, UP + RIGHT)

        # Keep only the relevant objects on screen
        to_keep = VGroup(
            equation3,
            furthest_body, 
            net_x_arrow, 
            net_y_arrow,        
            resultant_arrow,
            label_x,
            label_y,            
            label_r
        )

        all_mobs = self.mobjects

        to_remove = [mob for mob in all_mobs if mob not in to_keep]

        # Fade out everything except the summary
        self.play(*[FadeOut(mob) for mob in to_remove], run_time=2)
        self.wait(0.5)
        # Animate the net force arrows and labels
        self.play(GrowArrow(net_x_arrow), Write(label_x), run_time=1.2)
        self.play(GrowArrow(net_y_arrow), Write(label_y), run_time=1.2)
        self.play(Create(net_y_arrow_moved), run_time=1.2)
        self.play(GrowArrow(resultant_arrow), Write(label_r), run_time=1.2)
        self.wait()


class GravitationalForceResult(Scene):
    """Visualizes gravitational contraction, pressure, and temperature increase."""

    N_BODIES = 50
    BODY_RADIUS = 0.2
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
        # --- Create non-overlapping bodies in a disk ---
        bodies = VGroup()
        positions = []
        min_dist = 2 * self.BODY_RADIUS + 0.02
        for _ in range(self.N_BODIES):
            for _ in range(100):  # Try up to 100 times to avoid overlap
                pos = self.RandPosition()
                if all(np.linalg.norm(pos - p) > min_dist for p in positions):
                    positions.append(pos)
                    break
            body = Dot(point=pos, radius=self.BODY_RADIUS, color=BLUE)
            body.initial_scale = 1.0
            bodies.add(body)
        self.play(LaggedStart(*[DrawBorderThenFill(b) for b in bodies], lag_ratio=0), run_time=2)
        self.wait(0.5)

        # --- Pressure and Temperature Equations ---
        temp_eq = MathTex("T = \\frac{PV}{nR}", font_size=30).to_corner(UP + LEFT)
        pres_eq = MathTex("P = \\frac{nRT}{V}", font_size=30).next_to(temp_eq, DOWN, aligned_edge=LEFT)
        self.play(Write(temp_eq), Write(pres_eq))
        self.wait(0.5)

        # --- Animate contraction and color change ---
        start_positions = [body.get_center() for body in bodies]
        end_positions = [self.CENTER for _ in start_positions]

        # --- Pressure Graph Setup ---
        axes = Axes(
            x_range=[0, self.MOVE_TIME, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": [0, 1, 2, 3, 4, 5, 6]},
            y_axis_config={"numbers_to_include": [1, 2, 3, 4, 5]},
        )

        x_label = axes.get_x_axis_label("T", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("P", edge=UP)

        graph_group = VGroup(axes, x_label, y_label)
        graph_group.scale(0.7).to_corner(DOWN + LEFT)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # --- Trackers for Pressure and Temperature ---
        pressure_tracker = ValueTracker(0)

        # --- Animate contraction and color change ---
        def update_bodies(mob, alpha):
            for i, body in enumerate(mob):
                pos = interpolate(start_positions[i], end_positions[i], smooth(alpha))
                body.move_to(pos)
                body.set_color(self.color_temperature(alpha))

                # Shrink smoothly
                target_scale = interpolate(1.0, 0.3, smooth(alpha))
                scale_ratio = target_scale / body.initial_scale
                body.scale(scale_ratio)
                body.initial_scale = target_scale

        # Now add updater AFTER it's defined
        bodies.add_updater(lambda mob: update_bodies(mob, pressure_tracker.get_value() / self.MOVE_TIME))
        self.add(bodies)

        # --- Pressure function (simulate contraction: P ~ 1/r^3) ---
        def pressure_func(t):
            min_r = 0.7
            r = self.INIT_RADIUS - (self.INIT_RADIUS - min_r) * (t / self.MOVE_TIME)
            return min(5, 1 / (r ** 3))


        pressure_graph = always_redraw(
            lambda: axes.plot(
                pressure_func,
                x_range=[0, pressure_tracker.get_value()],
                color=YELLOW,
            )
        )
        moving_dot = always_redraw(
            lambda: Dot(color=RED).move_to(
                axes.c2p(
                    pressure_tracker.get_value(),
                    pressure_func(pressure_tracker.get_value())
                )
            )
        )

        self.add(pressure_graph)
        self.add(moving_dot)

        # --- Animate pressure and temperature increase ---
        temp_label = always_redraw(
            lambda: MathTex(
                f"T = {1 + 4 * (pressure_tracker.get_value() / self.MOVE_TIME):.1f} K",
                font_size=30
            ).next_to(temp_eq, RIGHT)
        )
        pres_label = always_redraw(
            lambda: MathTex(
                f"P = {pressure_func(pressure_tracker.get_value()):.2f} Pa",
                font_size=30
            ).next_to(pres_eq, RIGHT)
        )
        self.add(temp_label, pres_label)

        self.play(
            pressure_tracker.animate.set_value(self.MOVE_TIME),
            rate_func=smooth,
            run_time=self.MOVE_TIME
        )
        bodies.clear_updaters()
        self.wait(1)

        # --- Final state: all bodies red, high pressure/temperature ---
        for body in bodies:
            body.set_color(RED)
        self.wait(2)

        # --- Clean up ---
        self.play(
            Uncreate(axes),
            Uncreate(x_label),
            Uncreate(y_label),
            Uncreate(pressure_graph),
            Uncreate(moving_dot),
            Uncreate(temp_eq),
            Uncreate(pres_eq),
            Uncreate(temp_label),
            Uncreate(pres_label),
            FadeOut(bodies)
        )
        self.wait(1)

        