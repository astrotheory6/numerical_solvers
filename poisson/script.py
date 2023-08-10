import numpy as np
from Poisson_2d import Poisson_2d

np.set_printoptions(suppress=True)


nx = 100
ny = 100
degree = 2
delta_x = 1/nx
delta_y = 1/ny
xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)

# f = lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 - 4

# GAUUSIAN
f = lambda x: np.exp(
    -(((x[0] - 0.5) ** 2)/2 + ((x[1] - 0.5) **2 )/2)
) 

xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)

poisson = Poisson_2d(nx, ny, degree, f)
uh = poisson.solve()

poisson.plot_uh_eval(xs, ys)
poisson.residual_uh(xs, ys)
