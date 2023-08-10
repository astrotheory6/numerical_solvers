#from this import d
from types import CellType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical)

from dolfinx.fem.petsc import LinearProblem

from dolfinx import geometry

from dolfinx.common import Timer, TimingType, list_timings

from ufl import ds, dx, grad, inner, div, SpatialCoordinate
from mpi4py import MPI

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from matplotlib import pyplot as plt
from collections import OrderedDict
import os


degree = 1
nx = 10
ny = 10
delta_x = 1/nx
delta_y = 1/ny

domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", degree))

# f = fem.Function(V)
# f.interpolate(lambda x: 2*x[0]**2 + x[1])
# f = fem.Constant(domain, -1.0)
f = -1

k = fem.Function(V)
k.interpolate(lambda x: x[0] + x[1])

def boundary_x(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

def boundary_y(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

dofs_x = locate_dofs_geometrical(V, boundary_x)
bcs_x = dirichletbc(0.0, dofs_x, V)

dofs_y = locate_dofs_geometrical(V, boundary_y)
bcs_y = dirichletbc(0.0, dofs_y, V)

bcs = [bcs_x, bcs_y]

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = (ufl.inner(ufl.grad(k), ufl.grad(u)) + k * ufl.div(ufl.grad(u))) * v * ufl.dx
L = -f * v * ufl.dx

problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

print(uh)