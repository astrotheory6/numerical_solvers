import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical)

from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from dolfinx import geometry
from dolfinx.common import Timer, TimingType, list_timings

from ufl import ds, dx, grad, inner, div, SpatialCoordinate
from mpi4py import MPI

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

from Heat_2d import Heat_2d

# u = 1 + x**2 + a*y**2 + beta*t
# f(x, y, t) = beta - 2 - 2*a
# u0 = 1 + x**2 + a*y**2

t = 0
T = 1.0
num_timesteps = 10
dt = (T - t) / num_timesteps

x_start, x_end = -2, 2
y_start, y_end = -2, 2

nx, ny = 75, 75
xs = np.linspace(x_start, x_end, nx+1)
ys = np.linspace(y_start, y_end, ny+1)

alpha, beta = 3, 1.2

def u0(x, a=5):
    return 1 + x[0]**2 + alpha*x[1]**2 + beta*t
    # return np.exp(-a * (x[0]**2 + x[1]**2))

degree = 2

beta = 1.2
alpha = 3
f = lambda x: x[0] - x[0] + beta - 2 - 2*alpha

heat = Heat_2d(x_start, x_end, y_start, y_end, nx, ny, t, T, num_timesteps, u0, f, degree)

heat.solve()
heat.construct_eval_cube()
heat.residual_cube()
