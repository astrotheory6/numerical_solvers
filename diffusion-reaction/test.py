import numpy as np
from scipy import fftpack

# Define the size of the field
size = (100, 100)

# Generate a 2D Gaussian random field
x = np.linspace(-1,1,size[0])
y = np.linspace(-1,1,size[1])
X,Y = np.meshgrid(x,y)
d = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
fft_g = fftpack.fft2(g)
field = np.real(fftpack.ifft2(fft_g))

