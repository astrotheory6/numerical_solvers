#from this import d
import numpy as np
from Advection_1d import Advection_1d

import matplotlib.pyplot as plt

# np.set_printoptions(suppress=True)

start = 0
stop = 1
num_intervals = 100

delx = (start - stop) / num_intervals

t = 0.0
T = 2
num_timesteps = 100

ts = np.linspace(0, 1, num_timesteps+1)

dt = (T - t) / num_timesteps

adv = 0.4

n_max = 8
Lx = 1

A = np.random.uniform(0, 1)
k = (2*np.pi * np.random.randint(1, n_max)) / Lx
phi = np.random.uniform(0, 2* np.pi)
sign = np.random.choice([-1, 1])
print("A:", A, "k:", k, "phi:", phi, "sign:", sign)

def sin1(x):
    return sign * A * np.sin(k * x + phi)

A2 = np.random.uniform(0, 1)
k2 = (2*np.pi * np.random.randint(1, n_max)) / Lx
phi2 = np.random.uniform(0, 2* np.pi)
sign2 = np.random.choice([-1, 1])
print("A2:", A2, "k2:", k2, "phi2:", phi2, "sign2:", sign2)

def sin2(x):
    return sign2 * A2 * np.sin(k2 * x + phi2)

#u0 = lambda x: 0.25 * np.sin(8 * x[0] + 0.02)
#u0 = lambda x: sin1(x[0])
#u0 = lambda x: np.tan(x[0])
#u0 = lambda x: sin1(x[0]) + sin2(x[0])
u0 = lambda x: x[0] ** 2

degree = 1
advection = Advection_1d(start, stop, num_intervals, t, T, num_timesteps, adv, degree, u0)
uh, errors = advection.solve()

points = np.linspace(start, stop, num_intervals+1)

advection.plot_residual_uh_over_all_ts(points, delx)
advection.plot_eval_at_all_timesteps(points)

advection.plot_eval_heatmap(points)
advection.plot_gradient_heatmap(points)
advection.plot_residual_heatmap(points, delx)


#def plot_residual():
    # eq: du/dt + beta * dx/dt = 0  ----> call this R(u)
    # ||R(u) - 0|| ^ 2
    # 