#from this import d
import numpy as np
from Advection_1D import Advection_1D

import matplotlib.pyplot as plt

start = 0
stop = 1
num_intervals = 100

t = 0.0
T = 2.0
num_timesteps = 100

ts = np.linspace(0, 1, num_timesteps+1)

dt = (T - t) / num_timesteps

adv = 0.4

n_max = 8
Lx = 1

Ai = np.random.uniform(0, 1)
ki = (2*np.pi * np.random.randint(1, n_max)) / Lx
phi = np.random.uniform(0, 2* np.pi)
sign = np.random.choice([-1, 1])
print("Ai:", Ai, "ki:", ki, "phi:", phi, "sign:", sign)

def sin1(x):
    return sign * Ai * np.sin(ki * x + phi)


Ai2 = np.random.uniform(0, 1)
ki2 = (2*np.pi * np.random.randint(1, n_max)) / Lx
phi2 = np.random.uniform(0, 2* np.pi)
sign2 = np.random.choice([-1, 1])
print("Ai2:", Ai2, "ki2:", ki2, "phi2:", phi2, "sign2:", sign2)

def sin2(x):
    return sign2 * Ai2 * np.sin(ki2 * x + phi2)

#u0 = lambda x: 0.25 * np.sin(8 * x[0] + 0.02)
#u0 = lambda x: sin1(x[0])
u0 = lambda x: sin1(x[0]) + sin2(x[0])

degree = 1
advection = Advection_1D(start, stop, num_intervals, t, T, num_timesteps, adv, degree, u0)
uh, errors = advection.solve()

points = np.linspace(0, 1, num_intervals+1)

#advection.plot_residual_over_all_ts(points)
#advection.plot_eval_at_all_timesteps(points)

advection.plot_eval_heatmap(points)
advection.plot_gradient_heatmap(points)
advection.plot_residual_heatmap(points)


#def plot_residual():
    # eq: du/dt + beta * dx/dt = 0  ----> call this R(u)
    # ||R(u) - 0|| ^ 2
    # 