import numpy as np
from DarcyFlow_2d import DarcyFlow_2d

np.set_printoptions(suppress=True)

# increase resolution to reduce residual

nx = 150
ny = 150
degree = 2
delta_x = 1/nx
delta_y = 1/ny
xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)

# f = lambda x: ((x[0] - 0.5)**2 / 9) + ((x[1] - 0.5)**2 / 25) - 1
f = lambda x: np.sin(x[0] + x[1])

# k = lambda x: np.sin(x[0]/5 + x[1]/5)
# CIRCLE
# k = lambda x: (x[0] - 0.5) ** 2 + (x[1] - 0.5)**2 - 16
# ELLIPSE
k = lambda x: ((x[0] - 0.5)**2 / 81) + ((x[1] - 0.5)**2 / 16) - 1
# k = lambda x: np.exp(x[0]) + np.exp(x[1])
# # ANGLED ELLIPSE
# pi = np.pi
# k = lambda x: ((np.cos(pi/6)**2 / 4) + ((np.sin(pi/6)**2) / 16)*(x[0] - 0.5)**2) + ((np.cos(pi/6)**2 / 4) + ((np.sin(pi/6)**2) / 16)*(x[1]-0.5)**2) - 1

xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)

darcy_flow = DarcyFlow_2d(nx, ny, degree, k, f)
uh = darcy_flow.solve()

# a = np.array(uh.x.array[:])
# f = open("uh_dofs.txt", "w+")
# content = str(a)
# f.write(content)
# f.close()

#darcy_flow.plot_uh_eval(xs, ys)
darcy_flow.residual_uh(xs, ys)
