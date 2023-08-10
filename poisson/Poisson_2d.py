import numpy as np
import pyvista
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar, dirichletbc, form, locate_dofs_geometrical)

from dolfinx.fem.petsc import LinearProblem

from dolfinx import geometry

from dolfinx.common import Timer, TimingType, list_timings

from ufl import ds, dx, grad, inner, div, SpatialCoordinate, FacetNormal
from mpi4py import MPI

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from matplotlib import pyplot as plt
from collections import OrderedDict
import os

np.set_printoptions(threshold=np.inf)

class Poisson_2d():
    def __init__(self, nx, ny, degree, f):
        self.nx = nx
        self.ny = ny
        self.delta_x = 1/nx
        self.delta_y  = 1/ny

        self.xs = np.linspace(0, 1, nx+1)
        self.ys = np.linspace(0, 1, ny+1)

        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
        self.V = FunctionSpace(self.domain, ("CG", degree))
        self.degree = degree

        self.f = fem.Function(self.V)
        self.f.interpolate(f)
        
        self.uh = fem.Function(self.V)

    def boundary_x(self, x):
        return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], 1))

    def boundary_y(self, x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

    def solve(self):

        dofs_x = locate_dofs_geometrical(self.V, self.boundary_x)
        bcs_x = dirichletbc(0.0, dofs_x, self.V)

        # dofs_y = locate_dofs_geometrical(self.V, self.boundary_y)
        # bcs_y = dirichletbc(0.0, dofs_y, self.V)

        bcs = [bcs_x]

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        x = SpatialCoordinate(self.domain)
        g = -4 * x[1]

        # a = ufl.div(ufl.grad(u)) * v * ufl.dx
        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = self.f * v * ufl.dx - g * v * ufl.ds

        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "cg", "ksp_rtol":1e-8, "ksp_atol":1e-12, "ksp_max_it": 6000})
        copy = fem.Function(self.V)
        copy = problem.solve()
        self.uh.x.array[:] = copy.x.array

        return self.uh

    def evaluate_uh(self, uh, xs, y):
        
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

        uh_eval = uh.eval(points_on_proc, cells)

        return uh_eval 
    
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
            print(name, np.mean(mtrx))

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
        V = self.V
        uh = self.uh
        domain = self.domain

        # V2 = fem.VectorFunctionSpace(domain, ("CG", 1))

        # n = ufl.FacetNormal(domain)
        # uh_n = fem.Function(V2)
        # e = ufl.dot(ufl.grad(uh), n)
        # expr = fem.Expression(e, V2.element.interpolation_points())
        # uh_n.interpolate(expr)
        # self.plot_fn_eval(uh_n, xs, ys, "uh_normal")

        self.plot_fn_eval(self.f, xs, ys, "f")
        self.plot_fn_eval(self.uh, xs, ys, "uh")
    
    def residual_uh(self, xs, ys):
        uh = self.uh
        f = self.f
        V = self.V

        r = fem.Function(V)
        e = -ufl.div(ufl.grad(uh)) - f
        expr = fem.Expression(e, V.element.interpolation_points())
        r.interpolate(expr)

        self.plot_fn_eval(r, xs, ys, "residual")




