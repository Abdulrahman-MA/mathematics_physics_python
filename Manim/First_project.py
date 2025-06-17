from manim import *

class MyFirstScene(Scene):
    def construct(self):
        circle = Circle(radius=5, color=BLUE)
        self.play(Create(circle), run_time=5)
        self.wait(1)


class BasicShapes(Scene):
    def construct(self):
        # Create shapes
        circle = Circle(radius=1, color=BLUE)
        square = Square(side_length=2, color=RED)
        triangle = Triangle().set_color(GREEN)

        # Positioning
        square.next_to(circle, RIGHT, buff=0.5)
        triangle.next_to(circle, LEFT, buff=0.5)

        # Animation
        self.play(Create(circle))
        self.play(Create(square))
        self.play(Create(triangle))
        self.wait(1)

class LayoutScene(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        square = Square(color=RED)
        triangle = Triangle().set_color(GREEN)

        # Absolute positioning
        circle.move_to(LEFT * 3 + UP * 2)
        square.move_to(DOWN * 2)
        triangle.move_to(RIGHT * 3 + UP * 1)

        # Add to screen
        self.play(Create(circle))
        self.play(Create(square))
        self.play(Create(triangle))
        self.wait(1)

class GridLayoutExample(Scene):
    def construct(self):
        # Step 1: Create shapes
        shapes = VGroup(
            Circle(color=BLUE),
            Square(color=RED),
            Triangle(color=GREEN),
            Star(color=YELLOW, n=5),
            RegularPolygon(n=6, color=PURPLE),
            Annulus(inner_radius=0.3, outer_radius=0.8, color=ORANGE)
        )

        # Step 2: Arrange in grid
        shapes.arrange_in_grid(rows=2, cols=3, buff=1.0)

        # General row-by-row animation
        rows = 2
        cols = 3
        for i in range(rows):
            row = shapes[i*cols:(i+1)*cols]
            self.play(*[Create(shape) for shape in row])
            self.wait(0.5)

class TransformExample(Scene):
    def construct(self):
        square = Square(color=BLUE).shift(LEFT * 2)
        circle = Circle(color=GREEN).shift(RIGHT * 2)

        self.play(Create(square))
        self.wait(0.5)
        self.play(Transform(square, circle), run_time=2)
        self.wait(1)

class ShapeSwap(Scene):
    def construct(self):
        square = Square(color=RED).shift(LEFT * 2)
        triangle = Triangle(color=GREEN).shift(RIGHT * 2)

        # Step 1: Show square
        self.play(Create(square))
        self.wait(0.5)

        # Step 2: Morph square into triangle (destroys square)
        self.play(ReplacementTransform(square, triangle), run_time=2)

        self.wait(1)

        # Step 3: Fade triangle away
        self.play(FadeOut(triangle))
        self.wait(1)

class CopyAndRestore(Scene):
    def construct(self):
        square = Square(color=RED)
        self.play(Create(square))
        self.wait(0.5)

        # Save the current state of the square
        square.save_state()

        # Move and scale the square
        self.play(square.animate.shift(RIGHT * 3).scale(0.5).set_color(BLUE))
        self.wait(0.5)

        # Restore to original state
        self.play(Restore(square), run_time=2)
        self.wait(1)

class CopyShape(Scene):
    def construct(self):
        triangle = Triangle(color=GREEN)
        triangle_copy = triangle.copy().shift(RIGHT * 3)

        self.play(Create(triangle))
        self.wait(0.5)
        self.play(Create(triangle_copy))
        self.wait(1)

class CoordinatedAnimations(Scene):
    def construct(self):
        # Create three circles and arrange them horizontally
        circles = VGroup(*[Circle(radius=0.5).shift(RIGHT * i) for i in range(-2, 3)])

        # Color them differently
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        for circle, color in zip(circles, colors):
            circle.set_color(color)

        self.play(LaggedStart(*[Create(c) for c in circles], lag_ratio=0.3))
        self.wait(1)

        # Scale them all at once with AnimationGroup
        self.play(AnimationGroup(*[c.animate.scale(1.5) for c in circles], lag_ratio=0))
        self.wait(1)

        # Fade them out one after another
        self.play(Succession(*[FadeOut(c) for c in circles], lag_ratio=0.2))
        self.wait(1)

class Chalenge1(Scene):
    def construct(self):
        triangle = Triangle().set_color(GREEN)
        circle  = Circle(radius=1, color=BLUE).shift(UP*2)
        square = Square(side_length=2, color=RED).shift(DOWN*1.5+ RIGHT*1.9)
        mob = Annulus(inner_radius=0.5, outer_radius=1.2,fill_color= DARK_BLUE, stroke_color=YELLOW, stroke_width=2) 
        mob.shift(LEFT*2 + DOWN*1)
        self.play(Create(triangle))
        self.play(Create(circle))
        self.play(Create(square))
        self.play(Create(mob))

class Chalenge2(Scene):
    def construct(self):
        triangle = Triangle().set_color(GREEN)
        circle  = Circle(radius=1, color=BLUE)
        square = Square(side_length=2, color=RED)
        group = VGroup(triangle, circle, square).arrange(LEFT, buff=1).move_to(LEFT *10)
        self.play(group.animate.move_to(ORIGIN))
        self.play(group.animate.scale(0.5))

class Chalenge3(Scene):
    def construct(self):
        circle = Circle(radius=1, color=BLUE).move_to(LEFT * 2)
        triangle = Triangle(color=GREEN).move_to(RIGHT * 2)

        self.play(Create(triangle))
        self.play(Transform(triangle, circle), run_time=2)
        self.play(FadeOut(triangle), run_time=1)
        self.wait(1)

class Chalenge4(Scene):
    def construct(self):
         triangle =  Triangle(color=GREEN).move_to(LEFT * 2)
         star = Star(color=YELLOW, n=6).move_to(RIGHT * 2)

         self.play(Create(triangle))
         self.play(ReplacementTransform(triangle,star))
         self.play(star.animate.scale(0.5).rotate(10))
         self.play(FadeOut(star))


class Chalenge5(Scene):
    def construct(self):
        # Step 1: Create objects
        circle = Circle(radius=0.5, color=BLUE)
        square = Square(side_length=1, color=RED).shift(RIGHT * 2)
        triangle = Triangle().set_color(GREEN).shift(LEFT * 2)

        # Step 2: Grouping
        group = VGroup(circle, square, triangle)

        # Step 3: Center the group around origin
        group.move_to(ORIGIN)

        # Step 4: Animate group rotation
        self.play(Create(group))
        self.wait(0.5)
        self.play(group.animate.rotate(PI), run_time=2)
        self.wait(0.5)
        for shape in group:
            self.play(shape.animate.scale(1.5), run_time=1.5)
            self.wait(0.5)
        self.play(group.animate.shift(UP * 2), run_time=1.5)
        self.play(FadeOut(group))
        self.wait(1)