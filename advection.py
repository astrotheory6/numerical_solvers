#from this import d
from types import CellType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem

from dolfinx import geometry

from ufl import ds, dx, grad, inner
from mpi4py import MPI

from petsc4py import PETSc

from matplotlib import pyplot as plt

#import pyvista

from utils import plot_fn_of_x

from collections import OrderedDict


# du/dt + adv* dx/dt = 0
# u(x, 0) = u0(x)
# analytical sol:   u_ex(x, t) = u0(x-adv*t)

start = 0
stop = 1
num_intervals = 100

t = 0.0
T = 2.0
num_timesteps = 25
dt = (T - t) / num_timesteps

adv = 0.4

domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_intervals)
degree = 1
P1 = ufl.FiniteElement("Lagrange", ufl.interval, degree)
V = FunctionSpace(domain, P1)

#V = FunctionSpace(domain, ("CG", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def evaluate(fn, d):
    points = np.zeros((3, num_intervals+1))
    points[0] = d

    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    cells = []
    points_on_proc = []
    cell_candidates = geometry.compute_collisions(bb_tree, points.T) # Find cells whose bounding-box collide with the the points
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T) # Choose one of the cells that contains the point
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])   


    fn_values = fn.eval(points_on_proc, cells)

    return fn_values


def plotter_1d(fn, fn_ex):

    d = np.linspace(start, stop, num_intervals+1)

    fn_values = evaluate(fn, d)
    fn_ex_values = evaluate(fn_ex, d)

    fig1, ax1 = plt.subplots()

    ax1.plot(d, fn_values, label='uh')
    ax1.plot(d, fn_ex_values, label='uex')
    ax1.legend()
    #ax1.ylabel('f')
    #ax1.xlabel('x')
    ax1.set(ylim=(-1, 1))
    fig1.savefig('test' + str(t)+ '.png')
    plt.close(fig1)


def plot_errors_per_timestep(errors_dict):
    x = []
    y = []
    for key, value in errors_dict.items():
        x.append(key)
        y.append(value)

    fig0, ax0 = plt.subplots()

    ax0.plot(x, y)
    ax0.set_ylabel('L2_error')
    ax0.set_xlabel('time')
    fig0.savefig('errors_' + str(degree) + '_.png')
    plt.close(fig0)


u0 = lambda x : 0.5 * np.sin(15 * x[0] + 0.1)

def u0_for_plt(x):
    z = 5
    return x ** 2
    #return np.sin(z * x)

ut = fem.Function(V)
ut.interpolate(u0)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
#bc = fem.dirichletbc(uD, boundary_dofs)

uh = fem.Function(V)
uh.interpolate(u0)

a = (u * v * ufl.dx) + (dt * adv * ufl.grad(u)[0] * v * ufl.dx)
L = ut * v * ufl.dx

bilinear = fem.form(a)
linear = fem.form(L)

A = fem.petsc.assemble_matrix(bilinear, bcs=[])
A.assemble()
b = fem.petsc.create_vector(linear)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# L2 error testing
#V2 = fem.FunctionSpace(domain, ("CG", 1))
uex = fem.Function(V)

L2_errors_per_timestep = OrderedDict()

for i in range(num_timesteps):

    uex.interpolate(lambda x: u0(x-adv*t))

    plotter_1d(uh, uex)

    # Update L reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, linear)


    # set bcs, solve
    fem.petsc.set_bc(b, [])
    #solver.convergence_criterion = "residual"
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # print L2-error metric
    error = fem.assemble_scalar(fem.form(ufl.inner(ut-uex, ut-uex) * ufl.dx))
    L2_error = np.sqrt(domain.comm.allreduce(error, op=MPI.SUM))

    L2_errors_per_timestep[t] = L2_error
    #print("t=", t, "L2:", L2_error)


    error_max = np.max(np.abs(uex.x.array-ut.x.array))
    #print("max difference between dofs:", error_max)

    # Update solution at previous time step (ut)
    ut.x.array[:] = uh.x.array

    t += dt


print(uh.x.array)
plot_errors_per_timestep(L2_errors_per_timestep)
plotter_1d(uh, uex)
