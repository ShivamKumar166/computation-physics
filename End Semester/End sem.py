
# Shivam Kumar – 2311166
# End Semester Examination


import numpy as np
import matplotlib.pyplot as plt
from all_lib import (
    lcgg, gauss_seidel, read_labeled_matrices, newton_raphson, simpson,
    rk4, solve_equations
)
import math


# Q1 — Particle Exchange Model Using Random Numbers

# We simulate particles hopping between two chambers.

gen = lcgg(seed=1234)
# start with all particles on the left
left = 5000
right = 0  # none on the right initially
# long enough to reach steady state
steps = 300000

left_history = []
right_history = []

for _ in range(steps):
    r = next(gen)

# probability of selecting a particle from the left compartment
    pL = left / (left + right)

    # decides which particle is selected
    if r < pL:
        left -= 1
        right += 1
    else:
        right -= 1
        left += 1

    left_history.append(left)
    right_history.append(right)

# Plotting the evolution of particle distribution
plt.figure(figsize=(10,6))
plt.plot(left_history, label="Count on Left", linewidth=1.2)
plt.plot(right_history, label="Count on Right", linewidth=1.2)
plt.axhline(2500, linestyle='--', color='black', label="Expected Equilibrium")
plt.title("Random Particle Exchange Between Two Chambers", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Number of Particles", fontsize=12)
plt.legend()
plt.grid(alpha=0.6, linestyle='--')
plt.tight_layout()
plt.show()


# Q2 — Solving Linear System Using Gauss–Seidel

# Matrix A and vector b are stored in an external text file.
# The read_labeled_matrices() function loads them cleanly.

matrices = read_labeled_matrices("queston2matrx.txt")
A = matrices["A"]
b = matrices["b"][0]   # extracted as 1D list

solution, iters = gauss_seidel(A, b, tol=1e-6, max_iter=10000)

print("\nQ2 — Gauss–Seidel Results:")
for i, val in enumerate(solution, 1):
    print(f"x{i} = {val:.8f}")
print("Iterations used:", iters)



# Q3 — Newton–Raphson Method for Spring Stretch

# Force balance equation: F(x) = 2.5 − x·e^x.


# The solution is obtained by solving F(x)=0.

def f(x):
    return 2.5 - x*math.exp(x)

def df(x):
    return -math.exp(x)*(x+1)

root, iters = newton_raphson(f, df, x0=0.5)

print("\nQ3 — Spring Extension via Newton–Raphson:")
print("Computed root =", root)
print("Total iterations =", iters)



# Q4 — Center of Mass of Beam with λ(x)=x²


# Center of mass x_cm = ( ∫ x·λ(x) dx ) / ( ∫ λ(x) dx )

def lambda_x(x):
    return x**2

def x_lambda(x):
    return x**3

a, b_ = 0, 2
N = 2000
# Simpson’s rule requires even N

M_total  = simpson(lambda_x, a, b_, N)
M_moment = simpson(x_lambda, a, b_, N)

x_cm = M_moment / M_total

print("\nQ4 — Numerical Center of Mass:")
print(f"x_cm = {x_cm:.4f} m")


# Q5 — Maximum Height with Air Resistance (RK4)

# The relation dv/dy = (-γv − g)/v is integrated upward until v becomes zero.

g = 10.0
gamma = 0.02
v0 = 10.0

# Maximum height without drag (to set an upper integration limit)
y_limit = v0**2 / (2*g)

def dv_dy(y, v):
    return (-gamma*v - g)/v

# Small step size for accuracy
h = 0.001
y_vals, v_vals = rk4(dv_dy, 0.0, v0, h, y_limit)

v_arr = np.array(v_vals)
y_arr = np.array(y_vals)

idx = np.where(v_arr <= 0)[0][0]
y_max = y_arr[idx]

print("\nQ5 — Maximum Height with Drag:")
print(f"Maximum height ≈ {y_max:.4f} m")

# plot v vs y
plt.figure(figsize=(7,5))
plt.plot(y_arr[:idx+1], v_arr[:idx+1], label="v(y)")
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Height y (m)")
plt.ylabel("Velocity v (m/s)")
plt.title("Velocity Profile During Upward Motion (RK4)")
plt.grid(True)
plt.legend()
plt.show()


# Q6 – Numerical solution of 1D heat equation u_tt = u_xx

import numpy as np
import matplotlib.pyplot as plt


#  PARAMETERS

L  = 2.0        # length of rod
nx = 20         # number of spatial grid points
nt = 5000       # number of time steps

dx = L / (nx - 1)       # spatial step size
dt = 4 / nt             # total time = 4 seconds >> dt = 4/nt
s  = dt / dx**2         # stability parameter
print("\nQ6 — Numerical solution of 1D heat equation u_tt = u_xx:")
print("Stability parameter s =", s)

# Stability requirement for FTCS: s <= 0.5
assert s <= 0.5, "FTCS scheme unstable (s must be <= 0.5)"

#   SPATIAL GRID

x = np.linspace(0, L, nx)


#   INITIAL CONDITION
#   u(x,0) = 20 | sin(pi x) |
u = 20 * np.abs(np.sin(np.pi * x))


# Time indices to store snapshots
snapshot_times = [0, 10, 20, 50, 100, 200, 500, 1000]
snapshots = [u.copy()]   # store u(x,0)


#   FTCS TIME EVOLUTION

for t in range(1, nt + 1):

    # create a new array for u at next time step
    u_new = u.copy()

    # apply FTCS formula to internal points
    for i in range(1, nx - 1):
        u_new[i] = u[i] + s * (u[i+1] - 2*u[i] + u[i-1])

    # apply boundary conditions: ends fixed at 0°C
    u_new[0]  = 0
    u_new[-1] = 0

    # update for next iteration
    u = u_new

    # store snapshot if needed
    if t in snapshot_times:
        snapshots.append(u.copy())


#   PLOTTING RESULTS

plt.figure(figsize=(9,5))

for idx, t in enumerate(snapshot_times):
    plt.plot(x, snapshots[idx], label=f"t = {t}")

plt.xlabel("Position x along the rod")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Distribution for 1D Heat Equation (FTCS)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Q7 — Quartic Polynomial Fit


# Fit a quartic curve to (x,y) data using normal equations and Gauss–Jordan.

data = np.loadtxt("Q.7 DATA.txt")
xdata, ydata = data[:,0], data[:,1]

n = len(xdata)
X = np.vstack([
    np.ones(n),
    xdata,
    xdata**2,
    xdata**3,
    xdata**4
]).T

A = (X.T @ X).tolist()
b_vec = (X.T @ ydata).tolist()

coeffs = solve_equations(A, b_vec)
a0, a1, a2, a3, a4 = coeffs

print("\nQ7 — Quartic Polynomial Coefficients:")
print("a0 =", a0)
print("a1 =", a1)
print("a2 =", a2)
print("a3 =", a3)
print("a4 =", a4)

xx = np.linspace(min(xdata), max(xdata), 400)
yy = a0 + a1*xx + a2*xx**2 + a3*xx**3 + a4*xx**4

plt.figure(figsize=(8,5))
plt.scatter(xdata, ydata, color='blue', label="Raw Data")
plt.plot(xx, yy, 'r', linewidth=2, label="Quartic Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Quartic Fit to Provided Dataset")
plt.grid(True)
plt.legend()
plt.show()

#Answers

'''
Q.1 From the graph we can see that at eq. both sides will have approx. 2500 particles

Q2 — Gauss–Seidel Results:
x1 = 0.99999975
x2 = 0.99999979
x3 = 0.99999991
x4 = 0.99999985
x5 = 0.99999987
x6 = 0.99999995
Iterations used: 16

Q3 — Spring Extension via Newton–Raphson:
Computed root = 0.9585863567288397
Total iterations = 6

Q4 — Numerical Center of Mass:
x_cm = 1.5000 m

Q5 — Maximum Height with Drag:
Maximum height ≈ 4.9350 m

Q6 — Numerical solution of 1D heat equation u_tt = u_xx:
Stability parameter s = 0.07220000000000001

Q7 — Quartic Polynomial Coefficients:
a0 = 0.25462950721153693
a1 = -1.1937592138092312
a2 = -0.4572554123829478
a3 = -0.8025653910658175
a4 = 0.013239427477391276'''

