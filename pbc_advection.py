#from this import d
from types import CellType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import (CellType, create_unit_cube, locate_entities_boundary,meshtags)

from dolfinx import geometry

import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint

from dolfinx.common import Timer, TimingType, list_timings

from ufl import ds, dx, grad, inner, SpatialCoordinate
from mpi4py import MPI

from petsc4py import PETSc

from matplotlib import pyplot as plt
from collections import OrderedDict
import os

start = 0
stop = 1
num_intervals = 100
delta_x = (start - stop) / num_intervals
t = 0.0
T = 2
num_timesteps = 100
ts = np.linspace(0, 1, num_timesteps+1)
dt = (T - t) / num_timesteps
adv = 0.4
u0 = lambda x: np.sin(x[0])
# u0 = lambda x: x[0] ** 2

degree = 1
domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_intervals)
V = FunctionSpace(domain, ("CG", degree))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

ut = fem.Function(V)
ut.interpolate(u0)

a = (u + dt * adv * ufl.grad(u)[0])*v*ufl.dx
L = ut * v * ufl.dx

uh = fem.Function(V)
uh.interpolate(u0)

bilinear = fem.form(a)
linear = fem.form(L)

bcs = []

def periodic_boundary(x):
    return np.array(np.isclose(x[0], 1))

def periodic_relation(x):
    out_x = np.zeros(x.shape)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x

facets = locate_entities_boundary(domain, domain.topology.dim - 1, periodic_boundary)
arg_sort = np.argsort(facets)
mt = meshtags(domain, domain.topology.dim - 1, facets[arg_sort], np.full(len(facets), 2, dtype=np.int64))

with Timer("~PERIODIC: Initialize MPC"):
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint_topological(V, mt, 2, periodic_relation, bcs, 1.0)
    mpc.finalize()

A = fem.petsc.assemble_matrix(bilinear, bcs=bcs)
A.assemble()
b = fem.petsc.create_vector(linear)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

L2_errors_per_timestep = {}

for i in range(num_timesteps):
    t += dt
    t = round(t, 4)

    # Update L reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    fem.petsc.assemble_vector(b, linear)

    # set bcs, solve
    fem.petsc.set_bc(b, bcs)
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (ut)
    ut.x.array[:] = uh.x.array