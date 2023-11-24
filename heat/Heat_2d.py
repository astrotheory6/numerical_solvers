from cProfile import label
from re import S
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

# du_dt = div(grad(u)) + f
# BC: u = uD
# IC: u = u0

class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t

class Heat_2d():
    def __init__(self, x_start, x_end, y_start, y_end, nx, ny, t, T, num_timesteps, u0, f, degree):
        self.degree = degree

        self.t = t # Start time
        self.T = T # Final time
        self.num_timesteps = num_timesteps # number of timesteps
        self.dt = (T-t) / num_timesteps # timestep size

        # Define mesh
        self.x_start, self.x_end = x_start, x_end, 
        self.y_start, self.y_end = y_start, y_end,
        self.nx, self.ny = nx, ny

        self.xs = np.linspace(x_start, x_end, nx+1)
        self.ys = np.linspace(y_start, y_end, ny+1)
        self.domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([x_start, y_start]), np.array([x_end, y_end])], [nx, ny], mesh.CellType.triangle)

        self.V = fem.FunctionSpace(self.domain, ("CG", degree))

        self.u0 = u0
        self.dict_of_uh = {}

        self.f = fem.Function(self.V)
        self.f.interpolate(f)

        alpha, beta = 1.2, 3.0
        self.f1 = fem.Constant(self.domain, beta - 2 - 2 * alpha)
        print(self.f1.value)


    def solve(self):
        V = self.V
        dt = self.dt

        t_ = self.t

        # un = fem.Function(self.V)
        # un.interpolate(self.u0)

        # Create boundary condition
        alpha, beta = 1.2, 3
        u_exact = exact_solution(alpha, beta, t_)
        uD = fem.Function(V)
        uD.interpolate(u_exact)

        un = fem.Function(V)
        un.interpolate(u_exact)

        uh = fem.Function(self.V)
        uh.interpolate(u_exact)

        copy0 = fem.Function(V)
        copy0.x.array[:] = uh.x.array
        self.dict_of_uh[t_] = copy0

        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        bc = fem.dirichletbc(uD, fem.locate_dofs_topological(V, fdim, boundary_facets))

        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        # F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (un + dt * self.f1) * v * ufl.dx
        # a = fem.form(ufl.lhs(F))
        # L = fem.form(ufl.rhs(F))
        
        a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = (un + dt * self.f1) * v * ufl.dx

        A = fem.petsc.assemble_matrix(a, bcs=[bc])
        A.assemble()
        b = fem.petsc.create_vector(L)

        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        for i in range(self.num_timesteps):

            t_ += self.dt
            t_ = np.round(t_, 4)

            u_exact.t = t_
            uD.interpolate(u_exact)

            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(b, L)
                    
            # Apply Dirichlet boundary condition to the vector
            fem.petsc.apply_lifting(b, [a], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, uh.vector)
            uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            un.x.array[:] = uh.x.array

            copy = fem.Function(V)
            copy.x.array[:] = uh.x.array
            self.dict_of_uh[t_] = copy

            # print(np.mean(np.abs(copy.x.array - uD.x.array)))


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

        alpha, beta = 3, 1.2

        xs = self.xs
        ys = self.ys
        cube = np.zeros((self.num_timesteps+1, len(xs), len(ys)))
        for i, k in enumerate(self.dict_of_uh.keys()): 
            u_exact = exact_solution(alpha, beta, k)

            fn = self.dict_of_uh[k]
            mtrx = self.evaluate_fn_2d(fn)
            self.plot_fn_eval(fn, str(k), mtrx)
            cube[i, :, :] = mtrx

        return cube # (num_timesteps, nx+1, ny+1)


    def plot_exact_fn(self, fn, name):
        xs = self.xs
        ys = self.ys

        x, y = np.meshgrid(xs, ys)

        mtrx = fn(x, y)

        fig, ax = plt.subplots(1, 1)
        fig1 = ax.imshow(mtrx, cmap='viridis', extent=[self.x_start, self.x_end, self.y_start, self.y_end])

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cbar0 = fig.colorbar(fig1, ax=ax, shrink=0.4)
        fig.tight_layout()

        dir = 'plots/evals'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/evals/' + name + '.png')
        else: 
            plt.savefig('plots/evals/' + name + '.png')
        plt.close(fig)

    def plot_fn_eval(self, fn, name, *args):
        # *args can be a matrix, so if args, directly plot the matrix
        xs = self.xs
        ys = self.ys
        if (args):
            mtrx = args[0]
        else:
            mtrx = self.evaluate_fn_2d(fn, xs, ys)

        fig, ax = plt.subplots(1, 1)
        fig1 = ax.imshow(mtrx, cmap='viridis', extent=[self.x_start, self.x_end, self.y_start, self.y_end], vmin=0.0, vmax=15.0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        cbar0 = fig.colorbar(fig1, ax=ax, shrink=0.4)
        fig.tight_layout()

        dir = 'plots/evals'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/evals/' + name + '.png')
        else: 
            plt.savefig('plots/evals/' + name + '.png')
        plt.close(fig)
    
    def residual_cube(self):

        cube = self.construct_eval_cube()

        print("eval:", np.mean(np.abs(cube)))

        # f_eval = self.evaluate_fn_2d(self.f1)
        # f_eval_transform = [q[0] for q in f_eval]
        f_cube = np.zeros((self.num_timesteps+1, len(self.xs), len(self.ys)))

        du_dt = np.gradient(cube, self.dt, edge_order=2, axis=0)
        ddu_ddt = np.gradient(du_dt, self.dt, edge_order=2, axis=0)

        laplace_cube = np.zeros((self.num_timesteps+1, len(self.xs), len(self.ys)))

        for _ in range(self.num_timesteps+1):

            du_dx = np.gradient(cube[_, :, :], self.x_end-self.x_start/self.nx, edge_order=2, axis=1)
            ddu_ddx = np.gradient(du_dx, self.x_end-self.x_start/self.nx, edge_order=2, axis=1)

            du_dy = np.gradient(cube[_, :, :], self.y_end-self.y_start/self.ny, edge_order=2, axis=0)
            ddu_ddy = np.gradient(du_dy, self.y_end-self.y_start/self.ny, edge_order=2, axis=0)

            laplace_cube[_, :, :] = ddu_ddt[_, :, :] + ddu_ddx + ddu_ddy
            print("du_dt:", np.mean(np.abs(du_dt[_, :, :])))
            print("laplace cube:", np.mean(np.abs(laplace_cube[_, :, :])))

            f_cube[_, :, :] = self.f1
        
        residual = du_dt - laplace_cube - f_cube

        for _ in range(residual.shape[0]):
            print("residual at [", str(_), "]:", np.mean(np.abs(residual[_, :, :])))

        return residual 

