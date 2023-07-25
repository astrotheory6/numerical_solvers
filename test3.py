import numpy as np

np.set_printoptions(suppress=True)

x = np.linspace(0, 1, 11)
y = np.linspace(0, 1, 11)

delx = 1 / (len(x) - 1)

def divergence(f):
    """
    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
    """
    num_dims = len(f)
    
    print(len(f.shape))

    if len(f.shape) == 1:
        return np.gradient(f, delx)
    else:
        return np.ufunc.reduce(np.add, [np.gradient(f[i], delx, axis=i) for i in range(num_dims)])

X, Y = np.meshgrid(x, y)

def u(x, y):
    #return 1 + x**2
    return 1 + x**2 + 2*y**2

f= -6

def dx(x):
    return 2*x
def dy(y):
    return 4*y

u_X_Y = u(X, Y)

#print("f(mesh): \n", u_X_Y)

g = np.array(np.gradient(u_X_Y, delx))
#print("g: \n", g)

partial_x = dx(X)
partial_y = dy(Y)

g_ex = np.empty((2, len(x), len(x)))
g_ex[0, :, :] = partial_y
g_ex[1, :, :] = partial_x

#print("v: \n", v)

print("g-v: \n", g-g_ex)

div = divergence(g_ex)
print("div: \n", div)

residual = -div - f

print("residual: \n", residual)