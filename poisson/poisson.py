from types import CellType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner, SpatialCoordinate, TestFunction, TrialFunction, div, dot, dx, grad, inner
from mpi4py import MPI

from dolfinx import geometry

from petsc4py.PETSc import ScalarType

from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

#import pyvista

# - del^2 u(x) = f(x)
# u(x) = u_D(x)
# u_ex = 1 + x^2 + 2y^2

np.set_printoptions(suppress=True)

nx = 100
ny = 100

domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
delx = 1 / nx

V = FunctionSpace(domain, ("CG", 1))

uD = fem.Function(V)
# uD.interpolate(lambda x: x[0]**3 + x[1]**3)
uD.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)


tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType(-6))

# f = fem.Function(V)
# f.interpolate(lambda x: 6*x[0] + 6*x[1])

#variational problem
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# L2 error testing
V2 = fem.FunctionSpace(domain, ("CG", 3))
u_ex = fem.Function(V2)
u_ex.interpolate(lambda x: 1 + x[0]**2 + 2*x[1]**2)
# u_ex.interpolate(lambda x: x[0]**3 + x[1]**3)


e = fem.assemble_scalar(fem.form(ufl.inner(uh-u_ex, uh-u_ex) * ufl.dx))
L2_error = np.sqrt(domain.comm.allreduce(e, op=MPI.SUM))
print("L2_error:", L2_error)
error_max = np.max(np.abs(uD.x.array-uh.x.array))

x = np.linspace(0, 1, nx+1)
y = np.linspace(0, 1, ny+1)
X, Y = np.meshgrid(x, y)

def uex(x, y):
    return 1 + x**2 + 2*y**3
    # return x**3 + y**3)

uexact = uex(X, Y)

points = np.array([X.ravel(), Y.ravel(), np.zeros(X.shape).ravel()])
points = points.T

def evaluate(uh, points):
    u_values = []
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])       
    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    u_values = uh.eval(points_on_proc, cells)
    u_values_transform = [q[0] for q in u_values]

    u_eval = np.zeros((nx+1, nx+1))
    for i in range(0, nx+1):
        slice = u_values_transform[(nx+1)*i : (nx+1)*(i+1)]
        u_eval[i, :] = slice

    return u_eval

uh_eval = evaluate(uh, points)
print("u_eval:", uh_eval)

g = np.array(np.gradient(uh_eval, delx))
divrgnce = utils.divergence(g, delx)
print("divergence:", divrgnce)
residual_mat = -divrgnce - uexact

print(residual_mat)

def plot_eval():
    fig, ax = plt.subplots(1, 2)

    fig1 = ax[0].imshow(uh_eval, cmap='viridis')
    fig2 = ax[1].imshow(uexact, cmap='viridis')

    ax[0].set_title('computed eval')
    ax[1].set_title('analytical eval')

    ax[0].set_xlabel("x")
    ax[1].set_ylabel("y")

    cbar0 = fig.colorbar(fig1, ax=ax[0], shrink=0.4)
    cbar1 = fig.colorbar(fig2, ax=ax[1], shrink=0.4)

    fig.tight_layout()

    dir = 'plots'
    if not os.path.exists(dir):
        os.mkdir(os.path.join(dir))
        plt.savefig('plots/poisson_eval_heatmap.png')
    else: 
        plt.savefig('plots/poisson_eval_heatmap.png')

def plot_residual():
    fig, ax = plt.subplots(1, 2)

    fig1 = ax[0].imshow(residual_mat, cmap='viridis')
    fig2 = ax[1].imshow(np.zeros((nx+1, nx+1)), cmap='viridis')

    ax[0].set_title('computed residual')
    ax[1].set_title('analytical residual')

    ax[0].set_xlabel("x")
    ax[1].set_ylabel("y")

    cbar0 = fig.colorbar(fig1, ax=ax[0], shrink=0.4)
    cbar1 = fig.colorbar(fig2, ax=ax[1], shrink=0.4)

    fig.tight_layout()

    dir = 'plots'
    if not os.path.exists(dir):
        os.mkdir(os.path.join(dir))
        plt.savefig('plots/poisson_residual_heatmap.png')
    else: 
        plt.savefig('plots/poisson_residual_heatmap.png')

plot_eval()
plot_residual()




    

