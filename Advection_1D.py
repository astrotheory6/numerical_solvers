#from this import d
from types import CellType
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem

from dolfinx import geometry

from ufl import ds, dx, grad, inner
from mpi4py import MPI

from petsc4py import PETSc

from matplotlib import pyplot as plt

#import pyvista

from utils import plot_fn_of_x

from collections import OrderedDict


import os

class Advection_1D():
    def __init__(self, start, stop, num_intervals, t, T, num_timesteps, adv, degree, u0):
        self.start = start
        self.stop = stop
        self.num_intervals = num_intervals

        self.delx = (self.start - self.stop) / self.num_intervals

        self.t = t
        self.T = T
        self.num_timesteps = num_timesteps
        self.dt = (self.T - self.t) / self.num_timesteps

        self.ts = np.linspace(self.t, self.T, self.num_timesteps+1)

        self.adv = adv

        self.domain = mesh.create_unit_interval(MPI.COMM_WORLD, self.num_intervals)
        self.degree = degree
        P1 = ufl.FiniteElement("Lagrange", ufl.interval, self.degree)
        self.V = FunctionSpace(self.domain, P1)

        self.u0 = u0
       
        self.dict_of_uex = self.populate_dict_uex()

        self.dict_of_uh = {}

    def populate_dict_uex(self):
        d = {}
        for step in self.ts:
            uex = fem.Function(self.V)
            uex.interpolate(lambda x: self.u0(x - self.adv * step))
            d[step] = uex
        return d
    
    def solve(self):

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        ut = fem.Function(self.V)
        ut.interpolate(self.u0)

        a = (u * v * ufl.dx) + (self.dt * self.adv * ufl.grad(u)[0] * v * ufl.dx)
        L = ut * v * ufl.dx

        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)

        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        #bc = fem.dirichletbc(uD, boundary_dofs)

        uh = fem.Function(self.V)
        uh.interpolate(self.u0)

        bilinear = fem.form(a)
        linear = fem.form(L)

        A = fem.petsc.assemble_matrix(bilinear, bcs=[])
        A.assemble()
        b = fem.petsc.create_vector(linear)

        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        # L2 error testing
        #V2 = fem.FunctionSpace(domain, ("CG", 1))
        #uex = fem.Function(self.V)
        L2_errors_per_timestep = OrderedDict()

        for step in self.ts:
            # Update L reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(b, linear)

            # set bcs, solve
            fem.petsc.set_bc(b, [])
            #solver.convergence_criterion = "residual"
            solver.solve(b, uh.vector)
            uh.x.scatter_forward()

            # print L2-error metric
            uex = self.dict_of_uex[step]
            error = fem.assemble_scalar(fem.form(ufl.inner(ut-uex, ut-uex) * ufl.dx))
            L2_error = np.sqrt(self.domain.comm.allreduce(error, op=MPI.SUM))
 
            L2_errors_per_timestep[step] = L2_error
            #print("t=", t, "L2:", L2_error)

            error_max = np.max(np.abs(uex.x.array-ut.x.array))
            #print("max difference between dofs:", error_max)

            # store solved in dict at timestep
            copy = fem.Function(self.V)
            copy.x.array[:] = ut.x.array
            self.dict_of_uh[step] = copy

            # Update solution at previous time step (ut)
            ut.x.array[:] = uh.x.array

        return self.dict_of_uh, L2_errors_per_timestep


    def evaluate_uh(self, t, points):
        uh = self.dict_of_uh[t]

        points_transform = np.zeros((3, len(points)))
        points_transform[0] = points

        #cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T)
        #colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T)

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T) # Choose one of the cells that contains the point
        for i, point in enumerate(points_transform.T):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])   


        uh_eval = uh.eval(points_on_proc, cells)

        return uh_eval

    def evaluate_uex(self, t, points=np.linspace(0, 1, 101)):
        #points = np.linspace(0, 1, 101)

        uex = self.dict_of_uex[t]

        points_transform = np.zeros((3, len(points)))
        points_transform[0] = points

        #cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T)
        #colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T)

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T) # Choose one of the cells that contains the point
        for i, point in enumerate(points_transform.T):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])   


        #uh_eval = uh.eval(points_on_proc, cells)
        uex_eval = uex.eval(points_on_proc, cells)

        return uex_eval
    
    def plot_eval_at_t(self, t, points):

        uh_values = self.evaluate_uh(t, points)
        uex_values = self.evaluate_uex(t)

        dir = 'plots'
        fig1, ax1 = plt.subplots()
        ax1.plot(np.linspace(0, 1, 101), uex_values, label='uex')
        ax1.plot(points, uh_values, label='uh')
        ax1.legend()

        ax1.set(ylim=(-1, 1))
        #ax1.ylabel('f')
        #ax1.xlabel('x')

        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            fig1.savefig('plots/test' + str(t)+ '.png')
        else: 
            fig1.savefig('plots/test' + str(t)+ '.png')
        plt.close(fig1)

    def plot_eval_at_all_timesteps(self, points):

        for step in self.dict_of_uh:
            #uh_values, uex_values = self.evaluate(self, step, points)
            self.plot_eval_at_t(step, points)

    def residual(self, t, points):
        dt = self.dt
        adv = self.adv

        uh_values = self.evaluate_uh(t, points)

        uh_values_transform = [v[0] for v in uh_values]
        r = dt + adv * np.gradient(uh_values_transform, 0.01)
        residual = np.mean(np.abs(r))

        return residual

    def plot_residual_over_all_ts(self, points):

        rs = [self.residual(t, points) for t in self.dict_of_uh.keys()]

        fig1, ax1 = plt.subplots()
        ax1.plot(self.dict_of_uh.keys(), rs, label='residual')
        ax1.legend()
        ax1.set_ylabel('residual')
        ax1.set_xlabel('time')
        fig1.savefig('residuals.png')

    def plot_eval_heatmap(self, points):

        v = np.zeros((self.num_timesteps+1, self.num_intervals+1))

        for i, time in enumerate(self.dict_of_uh.keys()):
            e = self.evaluate_uh(time, points)
            e_transform = [m[0] for m in e]

            v[-i-1, :] = e_transform

        #print("v:", v)
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = [3.50, 3.50]
        #plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])
        plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])

        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()
        plt.savefig("eval_heatmap.png")
    
    def plot_gradient_heatmap(self, points):

        v = np.zeros((self.num_timesteps+1, self.num_intervals+1))

        for i, time in enumerate(self.dict_of_uh.keys()):
            e = self.evaluate_uh(time, points)
            e_transform = [m[0] for m in e]

            v[-i-1, :] = np.gradient(e_transform, self.delx)

        #print("v:", v)
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = [3.50, 3.50]
        #plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])
        plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])

        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()
        plt.savefig("gradient_heatmap.png")
    
    def plot_residual_heatmap(self, points):

        v = np.zeros((self.num_timesteps+1, self.num_intervals+1))

        for i, time in enumerate(self.dict_of_uh.keys()):
            e = self.evaluate_uh(time, points)
            e_transform = [m[0] for m in e]

            v[-i-1, :] = self.residual(time, points)

        #print("v:", v)
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = [3.50, 3.50]
        #plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])
        plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])

        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()
        plt.savefig("residual_heatmap.png")