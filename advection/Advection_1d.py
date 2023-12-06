from ast import Constant
from types import CellType
from unicodedata import east_asian_width
import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import (CellType, create_unit_cube, locate_entities_boundary,meshtags)
from dolfinx import geometry


import dolfinx_mpc

from dolfinx.common import Timer, TimingType, list_timings
from ufl import ds, dx, grad, inner, SpatialCoordinate
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
from collections import OrderedDict
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#import utils

class Advection_1d():
    def __init__(self, x_start, x_end, nx, t, T, num_timesteps, adv, degree, u0):
        self.x_start = x_start
        self.x_end = x_end
        self.nx = nx

        self.xs = np.linspace(x_start, x_end, nx+1)

        self.delx = (self.x_start - self.x_end) / self.nx

        self.t = t
        self.T = T
        self.num_timesteps = num_timesteps
        self.dt = (self.T - self.t) / self.num_timesteps

        ts = np.linspace(self.t, self.T, self.num_timesteps+1)
        self.ts = np.round(ts, 4)

        self.adv = adv

        self.domain = mesh.create_unit_interval(MPI.COMM_WORLD, self.nx)
        self.degree = degree
        
        # P1 = ufl.FiniteElement("Lagrange", ufl.interval, self.degree)
        # self.V = FunctionSpace(self.domain, P1)

        self.V = fem.FunctionSpace(self.domain, ("CG", degree))

        self.u0 = u0
       
        self.dict_of_uex = self.populate_dict_uex()

        self.dict_of_uh = {}

    def populate_dict_uex(self):
        d = {}
        for t in self.ts:
            uex = fem.Function(self.V)
            uex.interpolate(lambda x: self.u0(x - self.adv * t))
            d[t] = uex

        return d

    # def generate_pbc_slave_to_master_map(self, i):
    #     def pbc_slave_to_master_map(x):
    #         out_x = x.copy()
    #         out_x[i] = x[i] - self.x_end
    #         return out_x
    #     return pbc_slave_to_master_map

    # def generate_pbc_is_slave(self, i):
    #     return lambda x: np.isclose(x[i], self.x_end)

    # def generate_pbc_is_master(self, i):
    #     return lambda x: np.isclose(x[i], 0.0)
    
    def solve(self):

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        u_0 = fem.Function(self.V)
        u_0.interpolate(self.u0)
        ut = fem.Function(self.V)
        ut.x.array[:] = u_0.x.array
        # ut.interpolate(self.u0)

        x = SpatialCoordinate(self.domain)

        # a = (u + self.dt * self.adv * ufl.grad(u)[0])*v*ufl.dx
        # a = (u*v*ufl.dx) + (self.dt * self.adv * np.inner(v, ufl.grad(u)[0]) * ufl.dx)
        a = (u + self.dt * self.adv * ufl.grad(u)[0]) * v * ufl.dx
        L = (2 * self.dt + ut) * v * ufl.dx

        uh = fem.Function(self.V)
        uh.interpolate(self.u0)

        copy0_h = fem.Function(self.V)
        copy0_h.x.array[:] = uh.x.array
        self.dict_of_uh[0.0] = copy0_h
        
        bilinear = fem.form(a)
        linear = fem.form(L)

        def dirichletboundary(x):
            return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

        # Create Dirichlet boundary condition
        # facets = mesh.locate_entities_boundary(self.domain, self.domain.geometry.dim - 1, dirichletboundary)
        # topological_dofs = fem.locate_dofs_topological(self.V, self.domain.geometry.dim - 1, facets)
        # print("top dofs:", topological_dofs)
        # zero = np.array([0, 0], dtype=PETSc.ScalarType)
        # # bc = fem.dirichletbc(zero, topological_dofs, self.V)
        bcs = []

        def periodic_boundary(x):
            return np.isclose(x[0], 1)

        def periodic_relation(x):
            out_x = np.zeros(x.shape)
            out_x[0] = 1 - x[0]
            out_x[1] = x[1]
            # out_x[2] = x[2]
            return out_x

        V = self.V
        facets = locate_entities_boundary(self.domain, self.domain.topology.dim - 1, periodic_boundary)
        arg_sort = np.argsort(facets)
        mt = meshtags(self.domain, self.domain.topology.dim - 1, facets[arg_sort], np.full(len(facets), 2, dtype=np.int32))

        with Timer("~PERIODIC: Initialize MPC"):
            mpc = dolfinx_mpc.MultiPointConstraint(V)
            mpc.create_periodic_constraint_topological(V, mt, 2, periodic_relation, bcs, 1.0)
            mpc.finalize()

        A = fem.petsc.assemble_matrix(bilinear, bcs=bcs)
        A.assemble()
        b = fem.petsc.create_vector(linear)

        solver = PETSc.KSP().create(self.domain.comm)
        solver.setOperators(A)
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)

        # L2 error testing
        # V2 = fem.FunctionSpace(domain, ("CG", 1))
        # uex = fem.Function(self.V)
        L2_errors_per_timestep = OrderedDict()

        t = self.t
        dt = self.dt

        for i in range(self.num_timesteps):
            t += dt
            t = round(t, 4)

            # L = ut * v * ufl.dx
            # linear = fem.form(L)
            # b = fem.petsc.create_vector(linear)

            # Update L reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(b, linear)

            # set bcs, solve
            fem.petsc.set_bc(b, bcs)
            petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
            # petsc_options = {"ksp_type": "cg", "pc_type": "hypre"}

            problem = dolfinx_mpc.LinearProblem(a, L, mpc, bcs, petsc_options=petsc_options)
            uh = problem.solve()
            uh.x.scatter_forward()

            # print L2-error metric
            uex = self.dict_of_uex[t]
            error = fem.assemble_scalar(fem.form(ufl.inner(uh-uex, uh-uex) * ufl.dx))
            L2_error = np.sqrt(self.domain.comm.allreduce(error, op=MPI.SUM))
 
            L2_errors_per_timestep[t] = L2_error
            print("L2:", L2_error)

            error_max = np.max(np.abs(uex.x.array-uh.x.array))
            print("max difference between dofs:", error_max)

            # Update solution at previous time step (ut)
            ut.x.array[:] = uh.x.array

            # store solved in dict at timestep
            copy = fem.Function(self.V)
            copy.x.array[:] = uh.x.array
            self.dict_of_uh[t] = copy
       
        return self.dict_of_uh, L2_errors_per_timestep


    def evaluate_uh(self, t):
        uh = self.dict_of_uh[t]

        ps = np.zeros((len(self.xs), 3))
        ps[:, 0] = self.xs

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, ps) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, ps) # Choose one of the cells that contains the point
        for i, point in enumerate(ps):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])   

        uh_eval = uh.eval(points_on_proc, cells)

        return uh_eval

    def evaluate_uex(self, t):
        uex = self.dict_of_uex[t]

        ps = np.zeros((len(self.xs), 3))
        ps[:, 0] = self.xs

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, ps) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, ps) # Choose one of the cells that contains the point
        for i, point in enumerate(ps):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])    

        uex_eval = uex.eval(points_on_proc, cells)

        return uex_eval

    def construct_uh_eval_cube(self):

        xs = self.xs
        cube = np.zeros((self.num_timesteps+1, len(xs)))
        for k, t in enumerate(self.ts):
            e = self.evaluate_uh(t)
            e_tf = [q[0] for q in e]
            cube[-k-1] = e_tf

        return cube # (num_timesteps, nx+1) 

    def construct_uex_eval_cube(self):

        xs = self.xs
        cube = np.zeros((self.num_timesteps+1, len(xs)))
        for k, t in enumerate(self.ts):
            e = self.evaluate_uex(t)
            e_tf = [q[0] for q in e]
            cube[-k-1] = e_tf

        return cube # (num_timesteps, nx+1) 
    
    def plot_eval_at_t(self, t):

        uh_values = self.evaluate_uh(t)
        uex_values = self.evaluate_uex(t)

        dir = 'plots/evals'
        fig1, ax1 = plt.subplots()
        ax1.plot(self.xs, uex_values, label='uex')
        ax1.plot(self.xs, uh_values, label='uh')
        ax1.legend()

        ax1.set(ylim=(-1, 1))
        #ax1.ylabel('f')
        #ax1.xlabel('x')

        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            fig1.savefig('plots/evals/test' + str(t)+ '.png')
        else: 
            fig1.savefig('plots/evals/test' + str(t)+ '.png')
        plt.close(fig1)

    def plot_eval_at_all_timesteps(self):

        self.dict_of_uh.keys()
        for t in self.dict_of_uh.keys():
            #uh_values, uex_values = self.evaluate(self, step, points)
            self.plot_eval_at_t(t)

    def residual_uh(self):
        dt = self.dt
        adv = self.adv

        uh_eval_cube = self.construct_uh_eval_cube()

        du_dt = np.gradient(uh_eval_cube, self.dt, axis=0, edge_order=2)
        du_dx = np.gradient(uh_eval_cube, self.delx, axis=1, edge_order=2)

        r = du_dt + adv * du_dx
        print("uh r:", np.mean(np.abs(r)))
        # print(r)

        return r

    def residual_uex(self):
        dt = self.dt
        adv = self.adv

        uex_eval_cube = self.construct_uex_eval_cube()

        du_dt = np.gradient(uex_eval_cube, dt, axis=0, edge_order=2)
        du_dx = np.gradient(uex_eval_cube, self.delx, axis=1, edge_order=2)

        r = du_dt + adv * du_dx
        print("uex r:", np.mean(np.abs(r)))
        # print(r)

        return r

    def plot_eval_heatmap(self):

        dt = self.dt
        adv = self.adv

        uh_eval_cube = self.construct_uh_eval_cube()
        uex_eval_cube = self.construct_uex_eval_cube()

        fig, ax = plt.subplots(1, 2)

        fig0 = ax[0].imshow(uh_eval_cube, cmap='viridis', extent=[0, 1, 0, 2])
        fig1 = ax[1].imshow(uex_eval_cube, cmap='viridis', extent=[0, 1, 0, 2])

        ax[0].set_title('solution eval')
        ax[1].set_title('exact eval')

        ax[0].set_xlabel("x")
        ax[1].set_ylabel("t")

        cbar0 = fig.colorbar(fig0, ax=ax[0], shrink=0.4)
        fig0.set_clim(vmin=-1, vmax=1)
        cbar1 = fig.colorbar(fig1, ax=ax[1], shrink=0.4)
        fig1.set_clim(vmin=-1, vmax=1)

        fig.tight_layout()

        dir = 'plots'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/eval_heatmap.png')
        else: 
            plt.savefig('plots/eval_heatmap.png')
    
    def plot_gradient_heatmap(self):
        
        uh_eval_cube = self.construct_uh_eval_cube()
        uex_eval_cube = self.construct_uex_eval_cube()

        grad_uh = np.gradient(uh_eval_cube, axis=1, edge_order=2)
        grad_uex = np.gradient(uex_eval_cube, axis=1, edge_order=2)

        fig, ax = plt.subplots(1, 2)

        fig1 = ax[0].imshow(grad_uh, cmap='viridis', extent=[0, 1, 0, 2])
        fig2 = ax[1].imshow(grad_uex, cmap='viridis', extent=[0, 1, 0, 2])

        ax[0].set_title('solution gradient')
        ax[1].set_title('analytical gradient')

        ax[0].set_xlabel("x")
        ax[1].set_ylabel("t")

        cbar0 = fig.colorbar(fig1, ax=ax[0], shrink=0.4)
        cbar1 = fig.colorbar(fig2, ax=ax[1], shrink=0.4)

        fig.tight_layout()

        dir = 'plots'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/gradient_heatmap.png')
        else: 
            plt.savefig('plots/gradient_heatmap.png')
    
    def plot_residual_heatmap(self):

        uh_residual_cube = self.residual_uh()
        uex_residual_cube = self.residual_uex()

        fig, ax = plt.subplots(1, 2)

        fig1 = ax[0].imshow(uh_residual_cube, cmap='viridis', extent=[0, 1, 0, 2])
        fig2 = ax[1].imshow(uex_residual_cube, cmap='viridis', extent=[0, 1, 0, 2])

        ax[0].set_title('solver residual')
        ax[1].set_title('exact residual')

        ax[0].set_xlabel("x")
        ax[1].set_ylabel("t")

        cbar0 = fig.colorbar(fig1, ax=ax[0], shrink=0.4)
        cbar1 = fig.colorbar(fig2, ax=ax[1], shrink=0.4)

        fig.tight_layout()

        dir = 'plots'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/residual_heatmap.png')
        else: 
            plt.savefig('plots/residual_heatmap.png')