import numpy as np
import matplotlib.pyplot as plt
from FyeldGenerator import generate_field

from Diffusion_Reaction import Diffusion_Reaction

np.random.seed(64)

# increase residual to decrease residual

nx = 64
ny = 64
xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)

T = 1.0
num_timesteps = 20
ts = np.linspace(0, T, num_timesteps+1)
dt = T / num_timesteps

Du = 0.001
Dv = 0.005
k = 0.005

np.set_printoptions(suppress=True)

def _u0(x):
    def Pkgen(n):
        def Pk(k):
            return np.power(k, -n)
        return Pk

    def distrib(shape):
        a = np.random.normal(loc=0, scale=1, size=shape)
        b = np.random.normal(loc=0, scale=1, size=shape)
        return a+ 1j * b

    shape = (32, 32)
    # shape = (1, x.shape[1])

    field = generate_field(distrib, Pkgen(3), shape)

    fig, ax = plt.subplots(1, 1)
    fig1 = ax.imshow(field, cmap='viridis', extent=[0, 1, 0, 1])
    cbar0 = fig.colorbar(fig1, ax=ax, shrink=0.7)
    fig.tight_layout()
    plt.savefig("plots/field.png")
    plt.close(fig)

    field = np.reshape(field, (1, x.shape[1]))
    
    return field

# u0 = lambda x: _u0(x)
# u0 = lambda x: (x[0]-0.5)**2 +(x[1]-0.5)**2
# u0 = lambda x: np.random.normal(0, 1, x.shape[1])
v0 = lambda x: np.random.normal(0, 1, x.shape[1])

degree = 2
dr = Diffusion_Reaction(nx, ny, T, num_timesteps, degree, _u0, v0, Du, Dv, k)
d = dr.solve()

dr.residual_zh()