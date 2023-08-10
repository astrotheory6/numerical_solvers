import numpy as np
#import pyvista

from dolfinx.fem import (Constant, Function, FunctionSpace, 
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, ds, dx, grad

from dolfinx import geometry

import matplotlib.pyplot as plt

import os


mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
V = FunctionSpace(mesh, ("CG", 1))
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx

def u_exact(x):
    return 1 + x[0]**2 + 2*x[1]**2

def boundary_D(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0],1))

dofs_D = locate_dofs_geometrical(V, boundary_D)
u_bc = Function(V)
u_bc.interpolate(u_exact)
bc = dirichletbc(u_bc, dofs_D)

x = SpatialCoordinate(mesh)
g = -4 * x[1]
f = Constant(mesh, ScalarType(-6))
L = f * v * dx - g * v * ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = FunctionSpace(mesh, ("CG", 2))
uex = Function(V2)
uex.interpolate(u_exact)

e = assemble_scalar(form((uh - uex)**2 * dx))
L2_error = np.sqrt(MPI.COMM_WORLD.allreduce(e, op=MPI.SUM))

u_vertex_values = uh.x.array
uex_1 = Function(V)
uex_1.interpolate(uex)
u_ex_vertex_values = uex_1.x.array
error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
error_max = MPI.COMM_WORLD.allreduce(error_max, op=MPI.MAX)
print(f"Error_L2 : {L2_error:.2e}")
print(f"Error_max : {error_max:.2e}")

def evaluate_uh(x_points, y_points):
    
    points_transform = np.zeros((3, len(x_points)))
    points_transform[0] = x_points
    points_transform[1] = y_points


    #cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T)
    #colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T)

    cells = []
    points_on_proc = []

    bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)

    cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T) # Find cells whose bounding-box collide with the the points
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points_transform.T) # Choose one of the cells that contains the point
    for i, point in enumerate(points_transform.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])   


    uh_eval = uh.eval(points_on_proc, cells)

    return uh_eval

def evaluate_uex(x_points, y_points):

    points_transform = np.zeros((3, len(x_points)))
    points_transform[0] = x_points
    points_transform[1] = y_points

    #cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T)
    #colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T)

    cells = []
    points_on_proc = []

    bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)

    cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T) # Find cells whose bounding-box collide with the the points
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points_transform.T) # Choose one of the cells that contains the point
    for i, point in enumerate(points_transform.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])   


    #uh_eval = uh.eval(points_on_proc, cells)
    uex_eval = uex.eval(points_on_proc, cells)

    return uex_eval


"""x_points = np.linspace(0, 10, 101)
y_points = np.linspace(0, 10, 101)

uh_values = evaluate_uh(x_points, y_points)
uex_values = evaluate_uex(x_points, y_points)

# Create a 3D figure
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')

# Plot the data as a 3D scatter plot
ax.scatter(x_points, y_points, uh_values, c='r', marker='o')

# Set labels for the axes
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Show the plot
plt.show()

if not os.path.exists(dir):
    os.mkdir(os.path.join(dir))
    fig1.savefig('poisson_dirichlet_neumann.png')
else: 
    fig1.savefig('poisson_dirichlet_neumann.png')
plt.close(fig1)
"""
