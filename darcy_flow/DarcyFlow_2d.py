import pyvista
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot, nls
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical)

from dolfinx.fem.petsc import LinearProblem

from dolfinx import geometry
from dolfinx.common import Timer, TimingType, list_timings

from ufl import ds, dx, grad, inner, div, SpatialCoordinate
from mpi4py import MPI

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from dolfinx.plot import create_vtk_mesh

from matplotlib import pyplot as plt
from collections import OrderedDict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

np.set_printoptions(threshold=np.inf)


class DarcyFlow_2d():
    def __init__(self, nx, ny, degree, k, f):
        self.nx = nx
        self.ny = ny
        self.delta_x = 1/nx
        self.delta_y  = 1/ny

        self.xs = np.linspace(0, 1, nx+1)
        self.ys = np.linspace(0, 1, ny+1)

        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
        self.V = FunctionSpace(self.domain, ("CG", degree))

        self.degree = degree

        self.k = fem.Function(self.V)
        self.k.interpolate(k)
        self.k.x.scatter_forward()
        # self.k = k
        
        self.f = fem.Function(self.V)
        self.f.interpolate(f)
        self.f.x.scatter_forward()

        # self.f = f

        self.uh = fem.Function(self.V)

    def boundary_x(self, x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

    def boundary_y(self, x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

    def solve(self):

        dofs_x = locate_dofs_geometrical(self.V, self.boundary_x)
        bcs_x = dirichletbc(0.0, dofs_x, self.V)

        dofs_y = locate_dofs_geometrical(self.V, self.boundary_y)
        bcs_y = dirichletbc(0.0, dofs_y, self.V)

        bcs = [bcs_x, bcs_y]
        # bcs = []

        # LINEAR SOLVE
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # a = ufl.div(self.k * ufl.grad(u)) * v * ufl.dx
        a = (-ufl.inner(ufl.grad(self.k * v), ufl.grad(u)) + ufl.inner(ufl.grad(self.k), ufl.grad(u)) * v) * ufl.dx
        L = -self.f * v * ufl.dx

        # problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu" })
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "gmres"})

        copy = fem.Function(self.V)
        copy = problem.solve()
        self.uh.x.array[:] = copy.x.array

        solver = problem.solver
        viewer = PETSc.Viewer().createASCII("solver_output.txt")
        solver.view(viewer)
        solver_output = open("solver_output.txt", "r")
        for line in solver_output.readlines():
            print(line)


        # NON LINEAR SOLVE
        # u = fem.Function(self.V)
        # v = ufl.TestFunction(self.V)

        # F = ufl.div(self.k * ufl.grad(u)) * v * ufl.dx + self.f*v*ufl.dx
        # problem = fem.petsc.NonlinearProblem(F, u, bcs=bcs)

        # solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
        # solver.convergence_criterion = "incremental"
        # solver.rtol = 1e-6
        # solver.report = True

        # print(u.x.array)

        # self.uh.x.array[:] = u.x.array

        return self.uh
    
    def evaluate_fn(self, fn, xs, y):
        
        points_transform = np.zeros((len(xs), 3))
        points_transform[:, 0] = xs
        points_transform[:, 1] = y

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, points_transform) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform) # Choose one of the cells that contains the point
        for i, point in enumerate(points_transform):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])   

        fn_eval = fn.eval(points_on_proc, cells)

        return fn_eval 

    def evaluate_fn_2d(self, fn, xs, ys):

        mtrx = np.zeros((len(xs),len(ys)))

        for i, y in enumerate(ys): 
        
            points_transform = np.zeros((len(xs), 3))
            points_transform[:, 0] = xs
            points_transform[:, 1] = y

            fn_values = self.evaluate_fn(fn, xs, y)
            fn_values_transform = [q[0] for q in fn_values]
            mtrx[i, :] = fn_values_transform

        return mtrx 

    def plot_fn_eval(self, fn, xs, ys, name):
        mtrx = self.evaluate_fn_2d(fn, xs, ys)

        if name == "residual":
            print(name, np.mean(np.abs(mtrx)))

        fig, ax = plt.subplots(1, 1)
        fig1 = ax.imshow(mtrx, cmap='viridis', extent=[0, 1, 0, 1])

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cbar0 = fig.colorbar(fig1, ax=ax, shrink=0.4)
        fig.tight_layout()

        dir = 'plots'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/' + name + '.png')
        else: 
            plt.savefig('plots/' + name + '.png')
        plt.close(fig)

    def plot_uh_eval(self, xs, ys):
        self.plot_fn_eval(self.k, xs, ys, "k")
        self.plot_fn_eval(self.uh, xs, ys, "uh")
    
    def residual_uh(self, xs, ys):
        uh = self.uh
        k = self.k
        f = self.f
        V = self.V

        r = fem.Function(V)
        e = ufl.div(k * ufl.grad(uh)) + f
        expr = fem.Expression(e, V.element.interpolation_points())
        r.interpolate(expr)
        r.x.scatter_forward()
        self.plot_fn_eval(r, xs, ys, "residual")
