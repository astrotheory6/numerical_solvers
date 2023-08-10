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
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#import utils

class Advection_1d():
    def __init__(self, start, stop, num_intervals, t, T, num_timesteps, adv, degree, u0):
        self.start = start
        self.stop = stop
        self.num_intervals = num_intervals

        self.delx = (self.start - self.stop) / self.num_intervals

        self.t = t
        self.T = T
        self.num_timesteps = num_timesteps
        self.dt = (self.T - self.t) / self.num_timesteps

        ts = np.linspace(self.t, self.T, self.num_timesteps+1)
        self.ts = np.round(ts, 4)

        self.adv = adv

        self.domain = mesh.create_unit_interval(MPI.COMM_WORLD, self.num_intervals)
        self.degree = degree
        
        # P1 = ufl.FiniteElement("Lagrange", ufl.interval, self.degree)
        # self.V = FunctionSpace(self.domain, P1)

        self.V = fem.FunctionSpace(self.domain, ("CG", 1))


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

        x = SpatialCoordinate(self.domain)

        a = (u + self.dt * self.adv * ufl.grad(u)[0])*v*ufl.dx
        L = ut * v * ufl.dx

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
            return np.array(np.isclose(x[0], 1))

        def periodic_relation(x):
            out_x = np.zeros(x.shape)
            out_x[0] = 1 - x[0]
            out_x[1] = x[1]
            out_x[2] = x[2]
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

            L = ut * v * ufl.dx
            linear = fem.form(L)
            b = fem.petsc.create_vector(linear)

            # Update L reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            fem.petsc.assemble_vector(b, linear)

            # set bcs, solve
            fem.petsc.set_bc(b, bcs)
            #solver.convergence_criterion = "residual"
            solver.solve(b, uh.vector)
            uh.x.scatter_forward()

            # print L2-error metric
            uex = self.dict_of_uex[t]
            error = fem.assemble_scalar(fem.form(ufl.inner(uh-uex, uh-uex) * ufl.dx))
            L2_error = np.sqrt(self.domain.comm.allreduce(error, op=MPI.SUM))
 
            L2_errors_per_timestep[t] = L2_error
            #print("t=", t, "L2:", L2_error)

            error_max = np.max(np.abs(uex.x.array-uh.x.array))
            #print("max difference between dofs:", error_max)

            # store solved in dict at timestep
            copy = fem.Function(self.V)
            copy.x.array[:] = uh.x.array
            self.dict_of_uh[t] = copy

            # Update solution at previous time step (ut)
            ut.x.array[:] = uh.x.array
       
        return self.dict_of_uh, L2_errors_per_timestep


    def evaluate_uh(self, t, points):
        uh = self.dict_of_uh[t]

        points_transform = np.zeros((len(points), 3))
        points_transform[:, 0] = points

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

    def evaluate_uex(self, t, points):
        uex = self.dict_of_uex[t]

        points_transform = np.zeros((3, len(points)))
        points_transform[0] = points

        cells = []
        points_on_proc = []

        bb_tree = geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)

        cell_candidates = geometry.compute_collisions(bb_tree, points_transform.T) # Find cells whose bounding-box collide with the the points
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points_transform.T) # Choose one of the cells that contains the point
        for i, point in enumerate(points_transform.T):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])   

        uex_eval = uex.eval(points_on_proc, cells)

        return uex_eval
    
    def plot_eval_at_t(self, t, points):

        uh_values = self.evaluate_uh(t, points)
        uex_values = self.evaluate_uex(t, points)

        dir = 'plots/evals'
        fig1, ax1 = plt.subplots()
        ax1.plot(points, uex_values, label='uex')
        ax1.plot(points, uh_values, label='uh')
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

    def plot_eval_at_all_timesteps(self, points):

        self.dict_of_uh.keys()
        for step in self.dict_of_uh.keys():
            #uh_values, uex_values = self.evaluate(self, step, points)
            self.plot_eval_at_t(step, points)

    def residual_uh(self, t, points, delx):
        dt = self.dt
        adv = self.adv

        """if t == 0:
            uh_values = self.evaluate_uh(t, points)
            uh_values_transform = [q[0] for q in uh_values]

            du_dt = 0
            #du_dt = self.T / self.num_timesteps

            r = du_dt + adv * np.gradient(uh_values_transform, delx)
            residual = np.mean(np.abs(r))
            return residual

        else: 
            uh_values = self.evaluate_uh(t, points)
            u_hminus1_values = self.evaluate_uh(np.round(t-self.dt, 4), points)

            uh_values_transform = np.array([p[0] for p in uh_values])
            uhminus1_values_transform = np.array([q[0] for q in u_hminus1_values])

            du_dt = np.gradient(uh_values_transform, self.dt)

            r = du_dt + adv * np.gradient(uh_values_transform, delx)
            residual = np.mean(np.abs(r))

            return residual"""
        uh_values = self.evaluate_uh(t, points)
        u_hminus1_values = self.evaluate_uh(np.round(t, 4), points)

        uh_values_transform = np.array([p[0] for p in uh_values])
        uhminus1_values_transform = np.array([q[0] for q in u_hminus1_values])

        du_dt = np.gradient(uh_values_transform, self.dt)

        print("du_dt h:", du_dt)

        r = du_dt + adv * np.gradient(uh_values_transform, delx)
        print(r)
        residual = np.mean(np.abs(r))
        return residual

    def residual_uex(self, t, points, delx):
        dt = self.dt
        adv = self.adv

        """if t == 0:
            uex_values = self.evaluate_uex(t, points)
            uex_values_transform = [q[0] for q in uex_values]

            du_dt = 0
            #du_dt = self.T / self.num_timesteps

            r = du_dt + adv * np.gradient(uex_values_transform, delx)
            residual = np.mean(np.abs(r))

            return residual

        else: 
            uh_values = self.evaluate_uex(t, points)
            u_hminus1_values = self.evaluate_uex(np.round(t-self.dt, 4), points)

            uh_values_transform = np.array([p[0] for p in uh_values])
            uhminus1_values_transform = np.array([q[0] for q in u_hminus1_values])

            du_dt = np.gradient(uh_values_transform, self.dt)

            r = du_dt + adv * np.gradient(uh_values_transform, delx)
            residual = np.mean(np.abs(r))

            return residual"""

        uex_values = self.evaluate_uex(t, points)
        u_exminus1_values = self.evaluate_uex(np.round(t, 4), points)

        uex_values_transform = np.array([p[0] for p in uex_values])
        uexminus1_values_transform = np.array([q[0] for q in u_exminus1_values])

        du_dt = np.gradient(uex_values_transform, self.dt)

        print("du_dt ex:", du_dt)

        r = du_dt + adv * np.gradient(uex_values_transform, delx)
        residual = np.mean(np.abs(r))

        return residual

    def plot_residual_uh_over_all_ts(self, points, delx):

        rs = [self.residual_uh(t, points, delx) for t in self.dict_of_uh.keys()]

        fig1, ax1 = plt.subplots()
        ax1.plot(self.dict_of_uh.keys(), rs, label='residual')
        ax1.legend()
        ax1.set_ylabel('residual')
        ax1.set_xlabel('time')
        ax1.set_yscale("log")


        fig1.savefig('residuals.png')

    def plot_eval_heatmap(self, points):

        vh = np.zeros((self.num_timesteps+1, self.num_intervals+1))
        vex = np.zeros((self.num_timesteps+1, self.num_intervals+1))

        for i, time in enumerate(self.dict_of_uh.keys()):
            eh = self.evaluate_uh(time, points)
            eh_transform = [q[0] for q in eh]
            vh[-i-1, :] = eh_transform

            eex = self.evaluate_uex(time, points)
            eex_transform = [q[0] for q in eex]
            vex[-i-1, :] = eex_transform


        #print("v:", v)
        fig, ax = plt.subplots(1, 2)

        fig1 = ax[0].imshow(vh, cmap='viridis', extent=[0, 1, 0, 2])
        fig2 = ax[1].imshow(vex, cmap='viridis', extent=[0, 1, 0, 2])

        ax[0].set_title('solution eval')
        ax[1].set_title('analytical eval')

        ax[0].set_xlabel("x")
        ax[1].set_ylabel("t")

        cbar0 = fig.colorbar(fig1, ax=ax[0], shrink=0.4)
        cbar1 = fig.colorbar(fig2, ax=ax[1], shrink=0.4)

        fig.tight_layout()

        dir = 'plots'
        if not os.path.exists(dir):
            os.mkdir(os.path.join(dir))
            plt.savefig('plots/eval_heatmap.png')
        else: 
            plt.savefig('plots/eval_heatmap.png')
    
    def plot_gradient_heatmap(self, points):
        
        vh = np.zeros((self.num_timesteps+1, self.num_intervals+1))
        vex = np.zeros((self.num_timesteps+1, self.num_intervals+1))


        for i, time in enumerate(self.dict_of_uh.keys()):
            eh = self.evaluate_uh(time, points)
            eh_transform = [m[0] for m in eh]
            vh[-i-1, :] = np.gradient(eh_transform, self.delx)

            eex = self.evaluate_uex(time, points)
            eex_transform = [m[0] for m in eex]
            vex[-i-1, :] = np.gradient(eex_transform, self.delx)

        #print("v:", v)
        #fig = plt.figure()
        #plt.rcParams["figure.figsize"] = [3.50, 3.50]
        #plt.imshow(v, cmap='viridis', extent=[0, 1, 0, 2])

        fig, ax = plt.subplots(1, 2)

        fig1 = ax[0].imshow(vh, cmap='viridis', extent=[0, 1, 0, 2])
        fig2 = ax[1].imshow(vex, cmap='viridis', extent=[0, 1, 0, 2])

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
    
    def plot_residual_heatmap(self, points, delx):

        vh = np.zeros((self.num_timesteps+1, self.num_intervals+1))
        vex = np.zeros((self.num_timesteps+1, self.num_intervals+1))


        for i, time in enumerate(self.dict_of_uh.keys()):
            #eh = self.evaluate_uh(time, points)
            #eh_transform = [m[0] for m in eh]
            vh[-i-1, :] = self.residual_uh(time, points, delx)
            vex[-i-1, :] = self.residual_uex(time, points, delx)

        #print("v:", v)
        fig, ax = plt.subplots(1, 2)

        fig1 = ax[0].imshow(vh, cmap='viridis', extent=[0, 1, 0, 2])
        fig2 = ax[1].imshow(vex, cmap='viridis', extent=[0, 1, 0, 2])

        ax[0].set_title('solution residual')
        ax[1].set_title('analytical residual')

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