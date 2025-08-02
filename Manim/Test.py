from manim import *
import numpy as np

class ElectricForce(Scene):
    def construct(self):
        radius = 0.5

        # Title
        title = Text("Electric Force", font_size=48)
        title.to_edge(UP)  # Moves to top center of screen
        self.play(Write(title))

        # === Two main bodies ===
        body1 = Circle(radius=radius, color=BLUE)
        body2 = Circle(radius=radius, color=RED)

        body1.move_to(LEFT * 3)
        body2.move_to(RIGHT * 3)

        label1 = Text("Body 1", font_size=24).next_to(body1, DOWN)
        label2 = Text("Body 2", font_size=24).next_to(body2, DOWN)

        plus1 = Text("+", font_size=40, weight=BOLD).move_to(body1.get_center())
        plus2 = Text("+", font_size=40, weight=BOLD).move_to(body2.get_center())

        body1_group = VGroup(body1, label1, plus1)
        body2_group = VGroup(body2, label2, plus2)

        self.play(Create(body1), Write(label1), 
                  Create(body2), Write(label2), 
                  Create(plus1), Create(plus2),
                  run_time=2)
        
        self.wait(0.5)

        # Function to create arrow from surface of one body to another
        def surface_arrow(start_circle, target_circle, length=0.8):
            direction = normalize(target_circle.get_center() - start_circle.get_center())
            start = start_circle.get_center() + direction * radius
            end = start + direction * length
            return Arrow(start=start, end=end, color=YELLOW, stroke_width=6, buff=0)


        self.wait(0.5)

        small_scale = 0.5
        corner_pos = UR * 2.5

        bodies_group = VGroup(body1_group, body2_group,)

        self.play(
            bodies_group.animate.scale(0.7).to_corner(UP + RIGHT),
            title.animate.shift(LEFT*0.5),
            run_time=2
        )
        self.wait(0.5)

        # Newton's gravitational law
        equation1 = MathTex("F = k \\frac{q_1 q_2}{r_{12}^2}", font_size=40)
        equation1.to_corner(UP + LEFT)
        self.play(Write(equation1), run_time=2)
        self.wait(0.5)

        # Gravitational force graph
        axes = Axes(
            x_range=[0.5, 10, 1],
            y_range=[0, 5, 1],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": [2, 4, 6, 8, 10]},
            y_axis_config={"numbers_to_include": [1, 2, 3, 4, 5]},
        )
        axes.scale(0.8).to_corner(DOWN + LEFT)

        x_label = axes.get_x_axis_label("r", edge=RIGHT, direction=RIGHT)
        y_label = axes.get_y_axis_label("F", edge=UP)

        G = 1
        m1 = 1
        m2 = 1

        def force_func(r):
            return G * m1 * m2 / (r ** 2)

        force_graph = axes.plot(force_func, color=YELLOW)
        
        def get_distance():
            return np.linalg.norm(body1.get_center() - body2.get_center())

        moving_dot = always_redraw(lambda:
            Dot(color=RED).move_to(
                axes.c2p(
                    get_distance(),
                    force_func(get_distance())
                )
            )
        )

        force_label = always_redraw(lambda:
            MathTex(f"F = {force_func(get_distance()):.2f}", font_size=32)
            .next_to(moving_dot, UP)
        )

        # --- ELECTRIC POTENTIAL ENERGY ---

        k = 1
        q1 = 1
        q2 = 1

        def potential_func(r):
            return k * q1 * q2 / r  # Positive potential (repulsive)

        # Axes for potential energy vs. distance
        potential_axes = Axes(
            x_range=[0.5, 10, 1],
            y_range=[0, 10, 2],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": [2, 4, 6, 8, 10]},
            y_axis_config={"numbers_to_include": [2, 4, 6, 8, 10]},
        )
        potential_axes.scale(1).to_corner(DOWN + RIGHT)

        # Axis labels
        x_label_U = potential_axes.get_x_axis_label("r", edge=RIGHT, direction=RIGHT)
        y_label_U = potential_axes.get_y_axis_label("U", edge=UP)

        # Plot the potential energy function
        potential_graph = potential_axes.plot(potential_func, color=BLUE)

        # Dot and label
        potential_dot = always_redraw(lambda:
            Dot(color=BLUE).move_to(
                potential_axes.c2p(
                    get_distance(),
                    potential_func(get_distance())
                )
            )
        )
        potential_label = always_redraw(lambda:
            MathTex(f"U = {potential_func(get_distance()):.2f}", font_size=32)
            .next_to(potential_dot, UP)
        )

        # Show axes and graph
        self.play(
            Create(potential_axes), Write(x_label_U), Write(y_label_U),
            Create(potential_graph), run_time=2
        )
        self.wait()

        # --- Barrier and tunneling visual ---
        barrier = DashedLine(
            start=potential_axes.c2p(0.7, 0),
            end=potential_axes.c2p(0.7, 6),
            color=WHITE
        )
        barrier_label = Text("Barrier", font_size=24).next_to(barrier, UP).shift(RIGHT * 0.5)

        tunnel_arrow = Arrow(
            start=potential_axes.c2p(0.4, 2),
            end=potential_axes.c2p(1.2, 2),
            color=GREEN,
            buff=0,
            stroke_width=6
        )
        tunnel_label = Text("Tunneling", font_size=24, color=GREEN).next_to(tunnel_arrow, UP).shift(LEFT * 1)

        self.play(Create(barrier), Write(barrier_label),
                   Create(axes), Write(x_label), Write(y_label), 
                   run_time=2)
        self.play(GrowArrow(tunnel_arrow), Write(tunnel_label), 
                  Create(force_graph), run_time=2)
        
        self.wait()
        
        self.add(potential_dot, potential_label)
        self.add(moving_dot, force_label)

        # Animate attraction and separation
        def move_bodies(distance):
            self.play(
                body1_group.animate.shift(RIGHT * distance),
                body2_group.animate.shift(LEFT * distance),
                run_time=3,
                rate_func=smooth
            )
            self.wait(1)
            self.play(
                body1_group.animate.shift(LEFT * distance),
                body2_group.animate.shift(RIGHT * distance),
                run_time=3,
                rate_func=smooth
            )
            self.wait(1)

        move_bodies(1.5)

        self.play(
            *[Uncreate(element) for element in [moving_dot, force_label, force_graph, 
                                                axes, x_label, y_label, x_label_U, 
                                                y_label_U]]
        )
        self.wait(1)

        self.play(
            *[Uncreate(element) for element in [tunnel_label, tunnel_arrow, barrier_label, barrier]]
        )

        # Updated potential function (safe near zero)
        def potential_func(r):
            if abs(r) < 0.1:
                return 10  # Avoid division near r=0
            U = k * q1 * q2 / r
            return abs(U)  # Ensure potential is always positive

        # New extended axes for potential
        potential_axes_extended = Axes(
            x_range=[-10, 10, 2],
            y_range=[0, 10, 2],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": False},
            x_axis_config={"numbers_to_include": [-10, -5, 0, 5, 10]},
            y_axis_config={"numbers_to_include": [0, 5, 10]},
        ).scale(0.9).to_edge(DOWN)

        x_label_ext = potential_axes_extended.get_x_axis_label("r", edge=RIGHT, direction=RIGHT)
        y_label_ext = potential_axes_extended.get_y_axis_label("U", edge=UP)

        new_potential_graph = potential_axes_extended.plot(potential_func, color=BLUE)

        # After new axes and graph are displayed:
        # Remove old dots
        self.remove(potential_dot, potential_label, moving_dot, force_label)

        # Redefine dot logic for new axes
        new_moving_dot = always_redraw(lambda:
            Dot(color=RED).move_to(
                potential_axes_extended.c2p(
                    get_distance(),
                    potential_func(get_distance())
                )
            )
        )

        new_moving_dot2 = always_redraw(lambda:
            Dot(color=BLUE).move_to(
                potential_axes_extended.c2p(
                    -get_distance(),                 
                    potential_func(-get_distance())
                )
            )
        )

        new_force_label = always_redraw(lambda:
            MathTex(f"U_{{\\text{{RED}}}} = {potential_func(get_distance()):.2f}", font_size=32)
            .next_to(new_moving_dot, UP)
        )
        new_force_label2 = always_redraw(lambda:
            MathTex(f"U_{{\\text{{BLUE}}}} = -{potential_func(get_distance()):.2f}", font_size=32)
            .next_to(new_moving_dot2, UP)
        )

        self.add(new_moving_dot, new_force_label, new_moving_dot2, new_force_label2)

        # Transition from old to new graph
        self.play(
            Transform(potential_axes, potential_axes_extended),
            Transform(potential_graph, new_potential_graph),
            FadeIn(x_label_ext),
            FadeIn(y_label_ext),
            run_time=2
        )
        self.wait(1)

        
        self.wait()

        def wiggle_bodies(n):
            for i in range(n):
                self.play(body2_group.animate.shift(LEFT * 3.4),
                        run_time=0.1)
                self.play(body2_group.animate.shift(RIGHT * 3.4),
                        run_time=0.1)
                i+= 1
        wiggle_bodies(10)

        self.play(body2_group.animate.shift(LEFT * 3.4),
                        run_time=0.1)
        
        bodyies_group = VGroup(body1_group, body2_group)

        last_circle = Circle(radius=radius+0.3, color=YELLOW)
        last_label = Text("Body 3", font_size=24).next_to(last_circle, DOWN)
        last_sign = Text("2+", font_size=40, weight=BOLD).move_to(last_circle.get_center())
        last_group = VGroup(last_circle, last_label, last_sign)
        last_group.scale(0.7).next_to(body1_group, RIGHT, buff=0)

        self.play(Transform(bodyies_group, last_group),
                  run_time=1)

        self.wait(1)

        self.play(FadeOut(*self.mobjects))

        self.wait(1)