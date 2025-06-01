import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.latex import parse_latex
from IPython.display import display, Math

# Define the symbol
x = sp.symbols('x')

# Define the function
f = parse_latex(r"\frac{x}{6\sqrt{x^2 + 2500}} - \frac{80 - x}{2\sqrt{(80 - x)^2 + 10000}}")

numerator, dominator = sp.fraction(f)
# Compute the derivative
f_prime = sp.diff(f, x)
f_2prime = sp.diff(f_prime, x)

print(f_prime)
print(f_2prime)


# Convert symbolic expressions to numeric functions
f_lambdified = sp.lambdify(x, f, modules='numpy')
f_prime_lambdified = sp.lambdify(x, f_prime, modules='numpy')
f_2prime_lambdified = sp.lambdify(x, f_2prime, modules='numpy')

# Create a range of x values
# Create a safe domain that excludes x = ±1
x_vals = np.linspace(-1000, 1000, 1000000)
x_vals = x_vals[(np.abs(x_vals - 1) > 0.1) & (np.abs(x_vals + 1) > 0.1)]
y_vals = f_lambdified(x_vals)
y_prime_vals = f_prime_lambdified(x_vals)
y_2prime_vals = f_2prime_lambdified(x_vals)
# Plot the function and its derivative
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label=fr"${sp.latex(f)}$", color='blue')
plt.plot(x_vals, y_prime_vals, label="f'(x)", linestyle='--', color='red')
plt.plot(x_vals, y_2prime_vals, label="f''(x)", linestyle=':', color='green')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title("Function and its Derivative")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
