import matplotlib.pyplot as plt
from sympy import Function, dsolve, Eq, Derivative, symbols, simplify, cancel
import sympy as sp
from sympy.parsing.latex import parse_latex
import numpy as np

'''
# Define symbols
t = symbols('t')
v = Function('v')(t)
a, g = symbols('a g', positive=True)

# Input LaTeX and parse just for display (not used in solving)
latex_str = r"\frac{dv(t)}{dt} - g + a v(t)^2"
latex_expr = parse_latex(latex_str)
ode = Eq(latex_expr, 0)
sol = dsolve(ode, v)
sol = simplify(sol)
sol = cancel(sol)'''

# Plot the parsed LaTeX as visual math
plt.figure(figsize=(6, 1.5))
plt.axis("off")
plt.text(0.01, 0.5, r"$\vec{F} \cdot \vec{r}\,' = (-\sin\theta, \cos\theta) \cdot (-\sin\theta, \cos\theta) = \sin^2\theta + \cos^2\theta = 1$", fontsize=10)
plt.savefig("solution_display.png", dpi=1600, bbox_inches='tight')
plt.show()
plt.close()

'''
params = {a:1, g: 9.8}

# Automatically map symbols in sol.rhs to known symbols by name:
def unify_symbols(expr, known_syms):
    name_to_sym = {str(s): s for s in known_syms}
    replacement_map = {sym: name_to_sym[str(sym)] for sym in expr.free_symbols if str(sym) in name_to_sym}
    return expr.subs(replacement_map)

v_sol_rhs = sol.rhs
v_sol_rhs_corrected = unify_symbols(v_sol_rhs, params.keys())

# Substitute numerical values
v_sol_rhs_num = v_sol_rhs_corrected.subs(params)
v_sol_rhs_num = simplify(v_sol_rhs_num)

print("After substitution:", v_sol_rhs_num)
print("Expression passed to lambdify (sol):", v_sol_rhs_num)

# Lambdify for plotting
v_func = sp.lambdify(t, v_sol_rhs_num, modules=['numpy','sympy'])

# Time values (fix here)
t_vals = np.linspace(0, 10, 1000)
y_vals = v_func(t_vals)



# Plot the evaluated numerical solution
plt.figure(figsize=(10, 6))
plt.plot(t_vals, y_vals, label=fr"${sp.latex(sol)}$", color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Velocity vs Time")
plt.xlabel('Time t')
plt.ylabel('Velocity v(t)')
plt.grid(True)
plt.legend()
plt.show()'''
