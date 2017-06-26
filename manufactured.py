from dolfin import *
from decimal import Decimal
import sys
import sympy as smp
from sympy.printing import ccode

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

class GrayScottEquations(NonlinearProblem):
    def __init__(self, a, L): # a, L, bilinear and linear form respectively
        NonlinearProblem.__init__(self)
        self.a = a
        self.L = L

    def F(self, b, x): # calculates residual vector b
        assemble(self.L, tensor=b)
    def J(self, A, x): # computes Jacobian matrix A
        assemble(self.a, tensor=A)

class PeriodicBoundary(SubDomain):
    def __init__(self, domain_size):
        SubDomain.__init__(self)
        self.domain_size = domain_size

    # Domain is left
    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.domain_size):
            y[0] = x[0] - self.domain_size
        else:
            y[0] = x[0]


def grayScottSolver(F_input, k_input, degree, end_time = "100.0", time_step = "1.0",
                    mesh_size = 128, domain_size = 2.5, save_solution = False, output = None):
    """Some documentation here"""

    # First we parse the save destination to make sure we have no error when we write to file
    if save_solution:
        try:
            if output[-4:] != ".pvd":
                print "Invalid output name, ending should be .pvd"
                sys.exit()
        except:
            print "Please provide filename"
            sys.exit()


    # Time variables
    T = Decimal(end_time)
    t = Decimal("0.0") # current time we are solving for
    h = Decimal(time_step)
    dt = Constant(float(h)) # for use in the form
    ntimestep = 0 # number of timesteps solved

    # Gray-Scott parameters
    F = F_input
    k = k_input
    D_u = 2e-5
    D_v = 1e-5


    mesh = UnitIntervalMesh(mesh_size)

    element = FiniteElement("CG", mesh.ufl_cell(), degree)
    pbd = PeriodicBoundary(domain_size)
    W = FunctionSpace(mesh, MixedElement([element, element]), constrained_domain=pbd)

    w = Function(W); (u, v) = split(w)
    (r, s) = split(TestFunction(W))

    # To store solutions to previous iterations
    w1 = Function(W); (u1, v1) = split(w1);
    w2 = Function(W); (u2, v2) = split(w2)
    u_prevs = [u1, u2]; v_prevs = [v1, v2]

    # Manufactured source term
    x, tt = smp.symbols('x[0], tt') # change tt to t
    # D_uu, D_vv, FF, kk = smp.symbols('D_u, D_v, F, k')
    u_exact = 3.0/4 + smp.cos(2*pi*x)/4 + tt
    v_exact = 1.0/8 - smp.cos(2*pi*x)/8 + tt
    # print ccode(u_exact), ccode(v_exact)

    # The equations applied to the exact solutions
    du = u_exact.diff(tt) - ( D_u * u_exact.diff(x,2) - u_exact*v_exact**2 + F*(1-u_exact) )
    dv = v_exact.diff(tt) - ( D_v * v_exact.diff(x,2) + u_exact*v_exact**2 - (F+k)*v_exact )

    # Initial conditions
    expression_w_exact = Expression([ccode(u_exact), ccode(v_exact)], tt = float(t), degree=degree+1) #element = W.ufl_element())
    w_exact = interpolate(expression_w_exact, W)
    u_exact, v_exact = split(w_exact)
    # print expression_w_exact.cppcode
    # u0, _ = w_exact.split()
    # File("pvd/u0.pvd") << u0

    assign(w, w_exact)
    assign(w1, w_exact) # the previous iteration

    # The forms

    # F = Constant(F_input)
    # k = Constant(k_input)
    # D_u = Constant(2e-5)
    # D_v = Constant(1e-5)

    rhs_u = Expression(ccode(du), tt = float(t), degree = degree+1)
    rhs_v = Expression(ccode(dv), tt = float(t), degree = degree+1)
    # expression_rhs = Expression([du, dv], t=float(t), D_u=D_u, D_v=D_v, F=F, k=k, element=W.ufl_element())


    # Midpoints used in Crank-Nicolson
    u_mid = 0.5*(u + u_prevs[0])
    v_mid = 0.5*(v + v_prevs[0])

    L_cn = (
            u*r*dx - u_prevs[0]*r*dx
            - dt*(
                  - D_u * inner(grad(u_mid), grad(r))
                  - u_mid * v_mid**2 * r
                  + F * (1 - u_mid) * r
                  )*dx
            - dt*rhs_u*r*dx

            + v*s*dx - v_prevs[0]*s*dx
            - dt*(
                  - D_v * inner(grad(v_mid), grad(s))
                  + u_mid * v_mid**2 * s
                  - (F + k) * v_mid * s
                  )*dx
            - dt*rhs_v*s*dx
            )


    L_bdf = (
             u*r*dx - 4.0/3 * u_prevs[0]*r*dx + 1.0/3 * u_prevs[1]*r*dx
             - 2.0/3*dt*(
                         - D_u * inner(grad(u), grad(r))
                         - u * v**2 * r
                         + F * (1 - u) * r
                         )*dx
            - 2.0/3*dt*rhs_u*r*dx

             + v*s*dx - 4.0/3 * v_prevs[0]*s*dx + 1.0/3 * v_prevs[1]*s*dx
             - 2.0/3*dt*(
                         - D_v * inner(grad(v), grad(s))
                         + u * v**2 * s
                         - (F + k)*v  * s
                         )*dx
            - 2.0/3*dt*rhs_v*s*dx
            )


    # Compute directional derivative of w in the direction of dw (Jacobian)
    a_bdf = derivative(L_bdf, w, TrialFunction(W))
    a_cn = derivative(L_cn, w, TrialFunction(W))

    # Create nonlinear problem and Newton solver
    problem_bdf = GrayScottEquations(a_bdf, L_bdf)
    solver_bdf = NewtonSolver()
    solver_bdf.parameters["linear_solver"] = "gmres"

    problem_cn = GrayScottEquations(a_cn, L_cn)
    solver_cn = NewtonSolver()
    solver_cn.parameters["linear_solver"] = "gmres"

    if save_solution:
        output = File("pvd/" + output)
        uu, vv = w.split()
        output << (uu, float(t))

    while t < T:

        # Update the time we're solving for
        t += h; print "Solving for time: ", float(t)

        # Check if we have enough initial data for BDF2, otherwise use Crank-Nicolson
        if ntimestep < 1:
            print "Crank-Nicolson time is: ", float(t - h/2)
            rhs_u.tt = float(t - h/2)
            rhs_v.tt = float(t - h/2)
            solver_cn.solve(problem_cn, w.vector())
        else:
            rhs_u.tt = float(t)
            rhs_v.tt = float(t)
            solver_bdf.solve(problem_bdf, w.vector())

        # Cycle the variables
        w2.assign(w1)
        w1.assign(w)

        ntimestep += 1

        if save_solution:# and ntimestep % 10 == 0:
            uu, vv = w.split()
            output << (uu, float(t))


    # The last saved solution
    u, _ = w.split(deepcopy=True) # deepcopy needed for u.vector()
    # We update the exact solution to the end time
    expression_w_exact.tt = float(t)
    w_exact = interpolate(expression_w_exact, W)
    u_exact, _ = w_exact.split(deepcopy=True)

    l2_err = errornorm(u_exact, u, "l2")
    infty_err = abs(u_exact.vector().array() - u.vector().array()).max()

    return l2_err, infty_err




if __name__ == "__main__":
    import sys
    end_time = sys.argv[1]
    time_step = sys.argv[2]
    F_input = float(sys.argv[3]) # forcing term
    k_input = float(sys.argv[4]) # rate constant
    output = sys.argv[5] # name of file to save solution

    degree = 1

    l2_err, infty_err = grayScottSolver( F_input, k_input, degree, end_time=end_time, time_step=time_step,
                     save_solution = True, output = output, mesh_size = 32, domain_size = 1.0)
    print l2_err, infty_err
