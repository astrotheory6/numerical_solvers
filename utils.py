import numpy as np
import matplotlib.pyplot as plt

from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

from mpi4py import MPI

#from advection import u0

def initial_condition(x):
    mean = 0.5
    std_dev = 0.1

    print(x.shape)
    return np.exp(-(x - mean)**2 / (2 * std_dev**2))

def u0(x):
    z = 5
    return np.sin(z * x)

def u1(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def plot_fn_of_x(f, X, name):
    Y = f(X)
    
    plt.plot(X, Y)
    plt.ylabel('f')
    plt.xlabel('x')
    plt.savefig(name)
    plt.show()

def plot_fn_of_x_y(f, X, Y, name):
    
    assert len(X) == len(Y)

    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)

    plt.savefig(name)
    plt.show()


"""def residual(advection, t, points):

    dt = advection.dt
    adv = advection.adv

    uh_values = advection.evaluate_uh(0.6, points)

    uh_values_transform = [v[0] for v in uh_values]
    r = dt + adv * np.gradient(uh_values_transform)
    residual = np.linalg.norm(r, 2)"""

def divergence(f, delx):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)

    if len(f.shape) == 1:
        return np.gradient(f, delx)
    else:
        return np.ufunc.reduce(np.add, [np.gradient(f[i], delx, axis=i, edge_order=2) for i in range(num_dims)])