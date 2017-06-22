from dolfin import *
import numpy as np
from decimal import Decimal
import sys

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

    # Domain is left and bottom boundary, and not the two corners (0, 1) and (1, 0)
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                (not ((near(x[0], 0) and near(x[1], self.domain_size)) or
                        (near(x[0], self.domain_size) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.domain_size) and near(x[1], self.domain_size):
            y[0] = x[0] - self.domain_size
            y[1] = x[1] - self.domain_size
        elif near(x[0], self.domain_size):
            y[0] = x[0] - self.domain_size
            y[1] = x[1]
        else:   # near(x[1], self.domain_size)
            y[0] = x[0]
            y[1] = x[1] - self.domain_size

class PerturbationSpots(SubDomain):
    def __init__(self, domain_size):
        SubDomain.__init__(self)
        self.domain_size = domain_size
        self.center = (self.domain_size * np.random.rand(), self.domain_size * np.random.rand())
        # TODO add radius (and number of spots but perhaps not here)

    def inside(self, x, on_boundary):
        r_squared = (self.center[0] - x[0])**2  + (self.center[1] - x[1])**2
        return bool(r_squared < (self.domain_size/20)**2)

class PerturbationSquare(SubDomain):
    def __init__(self, domain_size):
        SubDomain.__init__(self)
        self.domain_size = domain_size
        self.size = domain_size / 10 # TODO let the user input this
        self.center = (domain_size/2, domain_size/2) # TODO and this

    def inside(self, x, on_boundary):
        return bool( abs(self.center[0] - x[0]) < self.size
                    and abs(self.center[1] - x[1]) < self.size )

class InitialConditions(Expression):
    def __init__(self, **kwargs):
        self.cell_function = kwargs["cell_function"]
        self.sigma = kwargs["sigma"]
        # np.random.seed(0) # TODO this doesn't work

    def eval_cell(self, values, x, ufc_cell):
        if self.cell_function.array()[ufc_cell.index] == 1:  # inside PerturbationSpots
            values[0] = 0.5 + self.sigma * np.random.randn()
            values[1] = 0.25 + self.sigma * np.random.randn()
        else:
            values[0] = 1.0 #+ sigma * np.random.randn() These perturbation don't appear to be useful
            values[1] = 0.0 #+ sigma * np.random.randn()

    def value_shape(self):
        return (2,)


def grayScottSolver(F_input, k_input, degree, end_time = "100.0", time_step = "1.0",
                    mesh_size = 128, domain_size = 2.5, initial_condition = "spots",
                    save_solution = False, output = None, exact_solution = None):
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

    # We also test that we have correct exact solution
    if exact_solution != None and exact_solution.saved == False:
        print "The exact solution provided has not been saved"
        sys.exit()


    # Time variables
    T = Decimal(end_time)
    t = Decimal("0.0") # current time we are solving for
    h = Decimal(time_step)
    dt = Constant(float(h)) # for use in the form
    ntimestep = 0 # number of timesteps solved


    mesh = RectangleMesh( Point(0, 0), Point(domain_size, domain_size),
                         mesh_size, mesh_size, "crossed" )

    element = FiniteElement("CG", triangle, degree)
    pbd = PeriodicBoundary(domain_size)
    V = FunctionSpace(mesh, element, constrained_domain=pbd)
    W = FunctionSpace(mesh, MixedElement([element, element]), constrained_domain=pbd)

    w = Function(W); (u, v) = split(w)
    (r, s) = split(TestFunction(W))

    # To store solutions to previous iterations
    w1 = Function(W); (u1, v1) = split(w1);
    w2 = Function(W); (u2, v2) = split(w2)
    u_prevs = [u1, u2]; v_prevs = [v1, v2]


    # Initial conditions
    cf = CellFunction('size_t', mesh, 0)
    if initial_condition == "spots":
        for _ in range(50): # TODO let user input this
            PerturbationSpots(domain_size).mark(cf, 1)
            sigma = 0.01
    elif initial_condition == "square":
        PerturbationSquare(domain_size).mark(cf, 1)
        sigma = 0.0

    w0 = interpolate(InitialConditions(cell_function = cf, sigma = sigma, degree = degree), W)
    # u0, v0 = w0.split()
    # File("pvd/u0.pvd") << u0

    assign(w, w0)
    assign(w1, w0) # the previous iteration


    # The forms

    F = Constant(F_input)
    k = Constant(k_input)
    D_u = Constant(2e-5)
    D_v = Constant(1e-5)

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
            + v*s*dx - v_prevs[0]*s*dx
            - dt*(
                  - D_v * inner(grad(v_mid), grad(s))
                  + u_mid * v_mid**2 * s
                  - (F + k) * v_mid * s
                  )*dx
            )


    L_bdf = (
             u*r*dx - 4.0/3 * u_prevs[0]*r*dx + 1.0/3 * u_prevs[1]*r*dx
             - 2.0/3*dt*(
                         - D_u * inner(grad(u), grad(r))
                         - inner(u, v**2) * r
                         + F * (1 - u) * r
                         )*dx

             + v*s*dx - 4.0/3 * v_prevs[0]*s*dx + 1.0/3 * v_prevs[1]*s*dx
             - 2.0/3*dt*(
                         - D_v * inner(grad(v), grad(s))
                         + inner(u, v**2) * s
                         - (F + k)*v  * s
                         )*dx
            )


    # Compute directional derivative of w in the direction of dw (Jacobian)
    a_bdf = derivative(L_bdf, w, TrialFunction(W))
    a_cn = derivative(L_cn, w, TrialFunction(W))

    # Create nonlinear problem and Newton solver
    problem_bdf = GrayScottEquations(a_bdf, L_bdf)
    solver_bdf = NewtonSolver()
    solver_bdf.parameters["linear_solver"] = "gmres"
    # solver_bdf.parameters["preconditioner"] = "sor"
    # solver_bdf.parameters["convergence_criterion"] = "residual" # "incremental"

    problem_cn = GrayScottEquations(a_cn, L_cn)
    solver_cn = NewtonSolver()
    solver_cn.parameters["linear_solver"] = "gmres"

    if save_solution:
        output = File("pvd/" + output)
        uu, _ = w.split()
        output << (uu, float(t))

    while t < T:
        # Update the time we're solving for
        t += h; print "Solving for time: ", float(t)

        # Check if we have enough initial data for BDF2, otherwise use Crank-Nicolson
        if ntimestep < 1:
            solver_cn.solve(problem_cn, w.vector())
        else:
            solver_bdf.solve(problem_bdf, w.vector())

        # Cycle the variables
        assign(w2, [w1.sub(0), w1.sub(1)])
        assign(w1, [w.sub(0), w.sub(1)])

        ntimestep += 1

        if save_solution and ntimestep % 10 == 0:
            uu, _ = w.split()
            output << (uu, float(t))

    # If we have an exact solution (from a finer mesh) then we calculate the errors
    if exact_solution != None: #exact_solution.saved has already been tested
        # The last saved solution
        u, _ = w.split(deepcopy=True) # deepcopy needed for u.vector()

        mesh_exact = Mesh("xml/mesh_exact.xml")
        W_exact = FunctionSpace(mesh_exact, MixedElement([element, element]), constrained_domain=pbd)
        w_exact = Function(W_exact, "xml/w_exact.xml")
        u_exact = interpolate(w_exact.sub(0), W.sub(0).collapse()) # not the same as using V!

        l2_err = errornorm(u_exact, u, "l2")
        infty_err = abs(u_exact.vector().array() - u.vector().array()).max()

        return w, mesh, l2_err, infty_err # cannot save sub-function as xml file so no point returning u
    else:
        return w, mesh


class ExactSolution():
    def __init__(self):
        self.saved = False
    def save(self, F, k, degree, end_time = "100.0", time_step = "1.0", mesh_size = 128):
        self.saved = True
        w, mesh = grayScottSolver(F, k, degree, end_time = end_time,
                                        time_step = time_step, mesh_size = mesh_size,
                                        initial_condition = "square" )
        File("xml/w_exact.xml") << w
        File("xml/mesh_exact.xml") << mesh



if __name__ == "__main__":
    import sys
    end_time = sys.argv[1]
    time_step = sys.argv[2]
    F_input = float(sys.argv[3]) # forcing term
    k_input = float(sys.argv[4]) # rate constant
    output = sys.argv[5] # name of file to save solution

    degree = 1

    grayScottSolver( F_input, k_input, degree, end_time=end_time, time_step=time_step,
                     save_solution = True, output = output, mesh_size = 128)
