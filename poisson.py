from types import CellType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner
from mpi4py import MPI

from petsc4py.PETSc import ScalarType

from matplotlib import pyplot as plt

import utils

from dolfinx import geometry

#import pyvista

# - del^2 u(x) = f(x)
# u(x) = u_D(x)
# u_ex = 1 + x^2 + 2y^2

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
delx = 0.125

V = FunctionSpace(domain, ("CG", 1))

u_D = fem.Function(V)
u_D.interpolate(lambda x: 1+ x[0]**2 + 2*x[1]**2)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)


boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(u_D, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
#f = fem.Constant(domain, ScalarType(-6))
f = -6

#variational problem
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = fem.petsc.LinearProblem(a, 
L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

#print(uh)

# L2 error testing
V2 = fem.FunctionSpace(domain, ("CG", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

e = fem.assemble_scalar(fem.form(ufl.inner(uh-uex, uh-uex) * ufl.dx))
L2_error = np.sqrt(domain.comm.allreduce(e, op=MPI.SUM))
print(L2_error)
error_max = np.max(np.abs(u_D.x.array-uh.x.array))

#print(error_L2)
#print(error_max)

#print(error_max)

def evaluate_uh(x, y):

    points_transform = np.zeros((3, len(x)))
    points_transform[0] = x
    points_transform[1] = y

    #cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T)
    #colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T)

    cells = []
    points_on_proc = []

    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)

    cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T) # Find cells whose bounding-box collide with the the points
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_transform.T) # Choose one of the cells that contains the point
    for i, point in enumerate(points_transform.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    uh_eval = uh.eval(points_on_proc, cells)

    return uh_eval

def uex(x, y):
   return 1 + x**2 + 2*y**2

x = np.linspace(0, 1, 8)
y = np.linspace(0, 1, 8)
X, Y = np.meshgrid(x, y)

uh_eval = evaluate_uh(x, y)
print("uh eval: \n", uh_eval)

uh_eval_transform = [v[0] for v in uh_eval]

g = np.array(np.gradient(uh_eval_transform, delx))
div = utils.divergence(g, delx)
print("div: \n", div)

residual = -div - f
print("residual:", np.linalg.norm(residual))

