import numpy as np

tol = 0.001  # Avoid hitting the outside of the domain
y = np.linspace(-1 + tol, 1 - tol, 101)
points = np.zeros((101, 3))
points[:, 1] = y

print(points)
