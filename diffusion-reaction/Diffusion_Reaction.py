from math import radians
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

np.set_printoptions(threshold=np.inf)

class Diffusion_Reaction():
    def __init__(self, nx, ny, T, num_timesteps, degree, u0, v0, Du, Dv, k):

        self.degree = degree

        self.nx = nx
        self.ny = ny
        self.T = T
        self.num_timesteps = num_timesteps
        self.ts = np.linspace(0, T, num_timesteps+1)
        self.dt = T / num_timesteps

        self.xs = np.linspace(0, 1, nx+1)
        self.ys = np.linspace(0, 1, ny+1)

        self.msh = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)

        P1 = ufl.FiniteElement("Lagrange", self.msh.ufl_cell(), degree)
        self.V = FunctionSpace(self.msh, P1 * P1)

        self.zh = fem.Function(self.V)
        self.uh, self.vh = ufl.split(self.zh)

        self.u0 = u0
        self.v0 = v0

        self.Du = Du
        self.Dv = Dv
        self.k = k

        self.dict_of_zh = {}

    def solve(self):
        p, q = ufl.TestFunctions(self.V)

        z = Function(self.V)
        u, v = ufl.split(z)

        z0 = Function(self.V)
        u0, v0 = ufl.split(z0)        

        z0.sub(0).interpolate(self.u0)
        z0.sub(1).interpolate(self.v0)

        zt = fem.Function(self.V)
        ut, vt = ufl.split(zt)

        dt = self.dt
        Du = self.Du
        Dv = self.Dv
        k = self.k
        # F0 = (u*q - dt*Du*ufl.div(ufl.grad(u))*q - (u - u**3 - k - v)*dt*q - ut*q)*ufl.dx
        # F1 = (v*p - dt*Dv*ufl.div(ufl.grad(v))*p - (u - v)*dt*p - vt*p)*ufl.dx

        F0 = (u*q + dt*Du*ufl.inner(ufl.grad(q), ufl.grad(u))- (u - u**3 - k - v)*dt*q - ut*q)*ufl.dx
        F1 = (v*p + dt*Dv*ufl.inner(ufl.grad(p), ufl.grad(v)) - (u - v)*dt*p - vt*p)*ufl.dx

        F = F0 + F1

        problem = NonlinearProblem(F, z)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = 1e-6

        z.x.array[:] = 0.0
        zt.x.array[:] = z0.x.array

        self.dict_of_zh[0] = z0

        t = 0
        for i in range(self.num_timesteps):
            t += dt
            t = np.round(t, 4)
            s = solver.solve(z)

            copy = Function(self.V)
            copy.x.array[:] = z.x.array[:]
            self.dict_of_zh[t] = copy

            zt.x.array[:] = z.x.array
            
        return self.dict_of_zh

    def evaluate_fn(self, fn, xs, y):
        
        points_transform = np.zeros((len(xs), 3))
        points_transform[:, 0] = xs
        points_transform[:, 1] = y

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.msh, self.msh.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, points_transform) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.msh, cell_candidates, points_transform) # Choose one of the cells that contains the point
        for i, point in enumerate(points_transform):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])   

        fn_eval = fn.eval(points_on_proc, cells)

        return fn_eval 

    def evaluate_fn_2d(self, fn):

        xs = self.xs
        ys = self.ys

        mtrx = np.zeros((len(xs),len(ys)))

        for i, y in enumerate(ys): 
        
            points_transform = np.zeros((len(xs), 3))
            points_transform[:, 0] = xs
            points_transform[:, 1] = y

            fn_values = self.evaluate_fn(fn, xs, y)
            fn_values_transform = [q[0] for q in fn_values]
            mtrx[i, :] = fn_values_transform

        return mtrx 

    def construct_eval_cube(self):
        xs = self.xs
        ys = self.ys
        u_cube = np.zeros((self.num_timesteps+1, len(xs), len(ys)))
        v_cube = np.zeros((self.num_timesteps+1, len(xs), len(ys)))
        for i, k in enumerate(self.dict_of_zh.keys()): 
            zh = self.dict_of_zh[k]
            uh = zh.sub(0)
            vh = zh.sub(1)
            u_mtrx = self.evaluate_fn_2d(uh)
            v_mtrx = self.evaluate_fn_2d(vh)

            # self.plot_fn_eval(fn, str(k), mtrx)

            u_cube[i, :, :] = u_mtrx
            v_cube[i, :, :] = v_mtrx

        return u_cube, v_cube # (num_timesteps, nx+1, ny+1)

    def plot_fn_eval(self, fn, name, *args):
        # *args can be a matrix, so if args, directly plot the matrix
        xs = self.xs
        ys = self.ys
        if (args):
            mtrx = args[0]
        else:
            mtrx = self.evaluate_fn_2d(fn)

        fig, ax = plt.subplots(1, 1)
        fig1 = ax.imshow(mtrx, cmap='viridis', extent=[0, 1, 0, 1], vmin=0.0, vmax=1.0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cbar0 = fig.colorbar(fig1, ax=ax, shrink=0.4)
        fig.tight_layout()

        dir = 'plots/'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/' + name + '.png')
        else: 
            plt.savefig('plots/' + name + '.png')
        plt.close(fig)

    def residual_zh(self):

        Du = self.Du
        Dv = self.Dv
        k = self.k

        V = self.V

        u_cube, v_cube = self.construct_eval_cube()
        u3_cube = u_cube**3

        du_dt = np.gradient(u_cube, self.dt, edge_order=2, axis=0)
        # ddu_ddt = np.gradient(du_dt, self.dt, edge_order=2, axis=0)
        sum_of_second_spatial_partials_cube = np.zeros((self.num_timesteps+1, len(self.xs), len(self.ys)))

        for _ in range(self.num_timesteps+1):

            du_dx = np.gradient(u_cube[_, :, :], 1/self.nx, edge_order=2, axis=1)
            ddu_ddx = np.gradient(du_dx, 1/self.nx, edge_order=2, axis=1)

            du_dy = np.gradient(u_cube[_, :, :], 1/self.ny, edge_order=2, axis=0)
            ddu_ddy = np.gradient(du_dy, 1/self.ny, edge_order=2, axis=0)

            sum_of_second_spatial_partials_cube[_, :, :] =  ddu_ddx + ddu_ddy

        r = du_dt - Du*sum_of_second_spatial_partials_cube - (u_cube - u3_cube - k - v_cube)

        for _ in range(self.num_timesteps+1):
            slice = r[_, :, :]
            print("residual at t=" + str(_) + ":", np.mean(np.abs(slice)))
            # self.plot_fn_eval("placeholder", "residuals/uh/" + str(_), slice)
