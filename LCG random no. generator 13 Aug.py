"""""Shivam Kumar 2311166"""""

#LCG random no. generator

import matplotlib.pyplot as plt
from all_lib import lcg

# LCG parameters
a = 1103515245
c = 12345
m = 32768


# Generating numbers by calling LCG function
seed = 1
N = 1000
L = lcg(seed, N)
print(L)

# Delay k=5
k = 5
xi = L[:-k]
xi_k = L[k:]

# Plot xi vs xi+k
plt.scatter(xi, xi_k, s=10, color='b')
plt.title(f"LCG Correlation Plot ($x_i$ vs $x_{{i+{k}}}$)")
plt.xlabel("$x_i$")
plt.ylabel(f"$x_{{i+{k}}}$")
plt.grid(True)
plt.show()