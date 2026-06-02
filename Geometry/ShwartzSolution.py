import sympy as sp
import itertools


class GREngine:
    """
    A symbolic General Relativity engine for computing spacetime geometry tensors.

    This class evaluates and caches the fundamental geometric quantities of a
    provided spacetime metric, including Christoffel symbols, the Riemann
    curvature tensor, the Ricci tensor, and the Kretschmann scalar.
    """

    def __init__(self, coords, metric):
        """
        Initializes the geometry engine.

        Args:
            coords (list): A list of SymPy symbols representing the coordinates.
            metric (sp.Matrix): The covariant metric tensor as a SymPy Matrix.
        """
        self.coords = coords
        self.g = metric
        self.n = len(coords)
        self.g_inv = metric.inv()

        self._Gamma = None
        self._Riemann = None
        self._Ricci = None
        self._Kretschmann = None

    def christoffel(self):
        """
        Computes the Christoffel symbols of the second kind.

        Returns:
            sp.MutableDenseNDimArray: A 3D array containing the Christoffel symbols.
        """
        if self._Gamma is not None:
            return self._Gamma

        n, coords, g, g_inv = self.n, self.coords, self.g, self.g_inv
        Gamma = sp.MutableDenseNDimArray.zeros(n, n, n)

        for rho, mu, nu in itertools.product(range(n), repeat=3):
            term = 0
            for sigma in range(n):
                if g_inv[rho, sigma] != 0:
                    term += g_inv[rho, sigma] * (
                            sp.diff(g[sigma, mu], coords[nu]) +
                            sp.diff(g[sigma, nu], coords[mu]) -
                            sp.diff(g[mu, nu], coords[sigma])
                    )
            if term != 0:
                Gamma[rho, mu, nu] = sp.simplify(term / 2)

        self._Gamma = Gamma
        return Gamma

    def riemann(self):
        """
        Computes the Riemann curvature tensor.

        Returns:
            sp.MutableDenseNDimArray: A 4D array containing the Riemann tensor components.
        """
        if self._Riemann is not None:
            return self._Riemann

        Gamma = self.christoffel()
        n, coords = self.n, self.coords
        Riem = sp.MutableDenseNDimArray.zeros(n, n, n, n)

        for rho, sigma, mu, nu in itertools.product(range(n), repeat=4):
            term = sp.diff(Gamma[rho, nu, sigma], coords[mu]) \
                   - sp.diff(Gamma[rho, mu, sigma], coords[nu])
            for lam in range(n):
                term += Gamma[rho, mu, lam] * Gamma[lam, nu, sigma]
                term -= Gamma[rho, nu, lam] * Gamma[lam, mu, sigma]
            if term != 0:
                Riem[rho, sigma, mu, nu] = sp.simplify(term)

        self._Riemann = Riem
        return Riem

    def ricci(self):
        """
        Computes the Ricci tensor by contracting the Riemann tensor.

        Returns:
            sp.MutableDenseMatrix: A 2D matrix containing the Ricci tensor components.
        """
        if self._Ricci is not None:
            return self._Ricci

        Riem = self.riemann()
        n = self.n
        Ric = sp.MutableDenseMatrix.zeros(n, n)

        for mu, nu in itertools.product(range(n), repeat=2):
            term = sum(Riem[rho, mu, rho, nu] for rho in range(n))
            if term != 0:
                Ric[mu, nu] = sp.simplify(term)

        self._Ricci = Ric
        return Ric

    def kretschmann(self):
        """
        Computes the Kretschmann scalar.

        The scalar is found by fully contracting the Riemann tensor with itself.
        Index raising and lowering is performed sequentially prior to contraction.

        Returns:
            sp.Expr: The simplified symbolic expression of the Kretschmann scalar.
        """
        if self._Kretschmann is not None:
            return self._Kretschmann

        R_uddd = self.riemann()
        n = self.n
        g, g_inv = self.g, self.g_inv

        R_dddd = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for a, b, c, d in itertools.product(range(n), repeat=4):
            term = sum(g[a, e] * R_uddd[e, b, c, d] for e in range(n) if g[a, e] != 0)
            if term != 0: R_dddd[a, b, c, d] = term

        R_uddd_raised = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for a, b, c, d in itertools.product(range(n), repeat=4):
            term = sum(g_inv[a, e] * R_dddd[e, b, c, d] for e in range(n) if g_inv[a, e] != 0)
            if term != 0: R_uddd_raised[a, b, c, d] = term

        R_uudd = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for a, b, c, d in itertools.product(range(n), repeat=4):
            term = sum(g_inv[b, e] * R_uddd_raised[a, e, c, d] for e in range(n) if g_inv[b, e] != 0)
            if term != 0: R_uudd[a, b, c, d] = term

        R_uuud = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for a, b, c, d in itertools.product(range(n), repeat=4):
            term = sum(g_inv[c, e] * R_uudd[a, b, e, d] for e in range(n) if g_inv[c, e] != 0)
            if term != 0: R_uuud[a, b, c, d] = term

        R_uuuu = sp.MutableDenseNDimArray.zeros(n, n, n, n)
        for a, b, c, d in itertools.product(range(n), repeat=4):
            term = sum(g_inv[d, e] * R_uuud[a, b, c, e] for e in range(n) if g_inv[d, e] != 0)
            if term != 0: R_uuuu[a, b, c, d] = term

        K = sum(R_dddd[a, b, c, d] * R_uuuu[a, b, c, d]
                for a, b, c, d in itertools.product(range(n), repeat=4))

        self._Kretschmann = sp.simplify(K)
        return self._Kretschmann


def derive_schwarzschild_metric():
    """
    Executes the symbolic derivation and validation of the Schwarzschild metric.
    """
    t, r, th, ph = sp.symbols('t r theta phi', real=True)
    coords = [t, r, th, ph]
    alpha, beta = sp.Function('alpha')(r), sp.Function('beta')(r)

    g_ansatz = sp.diag(-sp.exp(2 * alpha), sp.exp(2 * beta), r ** 2, r ** 2 * sp.sin(th) ** 2)
    engine = GREngine(coords, g_ansatz)
    Ric = engine.ricci()

    print("--- Step 1: Solving R^t_t - R^r_r = 0 ---")
    eq_alpha_beta = sp.simplify(engine.g_inv[0,0]*Ric[0,0] - engine.g_inv[1,1]*Ric[1,1])
    print(f"Relationship eq: {eq_alpha_beta} = 0")
    print("=> alpha'(r) = -beta'(r) => beta(r) = -alpha(r) (setting integration constant to 0 for asymptotic flatness)\n")

    print("--- Step 2: Solving R_thth = 0 to find alpha(r) ---")
    y = sp.Function('y')(r)

    R_thth_sub = Ric[2, 2].subs(beta, -alpha).subs(alpha, sp.log(y)/2).doit()
    expr_y = sp.simplify(R_thth_sub)
    print(f"ODE for y(r) = exp(2*alpha): {expr_y} = 0")

    sol = sp.dsolve(sp.Eq(expr_y, 0), y)
    print(f"General solution for y(r): {sol.rhs}")

    constants = list(sol.rhs.free_symbols - {r})
    if constants:
        C1 = constants[0]
        rs = sp.symbols('r_s', real=True)
        y_final = sol.rhs.subs(C1, -rs)
        print(f"Applying physical boundary conditions (constant = -r_s): y(r) = {y_final}\n")
    else:
        raise ValueError("SymPy did not return a constant of integration.")

    print("--- Step 3: Constructing and Validating the Derived Metric ---")
    alpha_final = sp.log(y_final)/2
    beta_final = -alpha_final

    g_derived = sp.simplify(sp.diag(-sp.exp(2 * alpha_final), sp.exp(2 * beta_final), r ** 2, r ** 2 * sp.sin(th) ** 2))

    val_engine = GREngine(coords, g_derived)
    ricci_final = val_engine.ricci()

    is_vacuum = True
    for i, j in itertools.product(range(4), repeat=2):
        if sp.simplify(ricci_final[i, j]) != 0:
            is_vacuum = False

    print(f"Einstein Vacuum Equations (R_mu_nu = 0) satisfied dynamically: {is_vacuum}")

    print("\nLine Element (ds^2):")
    dt, dr, dth, dph = sp.symbols('dt dr dtheta dphi')
    ds2 = g_derived[0,0]*dt**2 + g_derived[1,1]*dr**2 + g_derived[2,2]*dth**2 + g_derived[3,3]*dph**2
    sp.pprint(sp.Eq(sp.Symbol('ds^2'), ds2))

    print("\nKretschmann Scalar (K = R_abcd R^abcd):")
    k_scalar = val_engine.kretschmann()
    sp.pprint(k_scalar)

if __name__ == "__main__":
    derive_schwarzschild_metric()