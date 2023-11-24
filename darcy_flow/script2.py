import numpy as np
import dolfinx
from dolfinx import fem, io, mesh, plot, nls, geometry
from mpi4py import MPI
import ufl

nx = 200
ny = 200
xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
V = fem.FunctionSpace(domain, ("CG", 2))
uh = fem.Function(V)

content = np.fromfile("uh_dofs.txt")
uh.x.array[:] = content

k = lambda x: ((x[0] - 0.5)**2 / 81) + ((x[1] - 0.5)**2 / 16) - 1
f = lambda x: np.sin(x[0] + x[1])

r = fem.Function(V)
e = ufl.div(k * ufl.grad(uh)) + f
expr = fem.Expression(e, V.element.interpolation_points())
r.interpolate(expr)
r.x.scatter_forward()

def evaluate_fn(fn, xs, y):
    points_transform = np.zeros((len(xs), 3))
    points_transform[:, 0] = xs
    points_transform[:, 1] = y

    cells = []
    points_on_proc = []

    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)

    cell_candidates = geometry.compute_collisions(bb_tree, points_transform) # Find cells whose bounding-box collide with the the points
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_transform) # Choose one of the cells that contains the point
    for i, point in enumerate(points_transform):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])   

    fn_eval = fn.eval(points_on_proc, cells)

    return fn_eval 

def evaluate_fn_2d(fn, xs, ys):

    mtrx = np.zeros((len(xs),len(ys)))

    for i, y in enumerate(ys): 
        points_transform = np.zeros((len(xs), 3))
        points_transform[:, 0] = xs
        points_transform[:, 1] = y

        fn_values = evaluate_fn(fn, xs, y)
        fn_values_transform = [q[0] for q in fn_values]
        mtrx[i, :] = fn_values_transform
    return mtrx 

mtrx = evaluate_fn_2d(uh, xs, ys)
print("residual:", np.mean(np.abs(mtrx)))
