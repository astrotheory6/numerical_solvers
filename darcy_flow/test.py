import numpy as np
from matplotlib import pyplot as plt
import os

np.set_printoptions(suppress=True)

nx = 10
ny = 10
xs = np.linspace(0, 1, nx+1)
ys = np.linspace(0, 1, ny+1)

X, Y = np.meshgrid(xs, ys)

def u(X, Y):
    return (X - 0.5)**2 + (Y- 0.5)**2

m = u(X, Y)
print(np.gradient(m, edge_order=2))

fig, ax = plt.subplots(1, 1)

fig1 = ax.imshow(m, cmap='viridis', extent=[0, 1, 0, 1])

cbar0 = fig.colorbar(fig1, ax=ax, shrink=0.4)

dir = 'plots'
if not os.path.exists(dir):
    os.mkdir(os.path.join(dir))
    plt.savefig('plots/TESTTESTSET.png')
else: 
    plt.savefig('plots/TESTEESTS.png')
plt.close(fig)

