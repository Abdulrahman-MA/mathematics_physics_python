import sympy as sp

x = sp.symbols('x')
f = x**3 / (x**2 - 1)

# Horizontal asymptotes: limits at infinity
limit_pos_inf = sp.limit(f, x, sp.oo)
limit_neg_inf = sp.limit(f, x, -sp.oo)
if limit_pos_inf == abs(limit_neg_inf) != sp.oo:
    limit_pos_inf = limit_neg_inf = "No horizontal asymptote"
print("Limit as x → ∞:", limit_pos_inf)
print("Limit as x → -∞:", limit_neg_inf)
