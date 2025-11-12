# Shivam Kumar 2311166


# Random Number Generator

def random():
    x = 0.1
    n=0
    l=[]
    while n<=1000:
        c= 2.89
        x=c*x*(1-x)
        n+=1

        l.append(x)
    return l


# LCG Random no. generator

def lcg(seed, n):
    a = 1103515245
    c = 12345
    m = 32768
    x = seed
    numbers = []
    for _ in range(n):
        x = (a * x + c) % m
        d=x/m
        numbers.append(d)
    return numbers



#LCG generator

def lcgg(seed):

    a = 1103515245
    c = 12345
    m = 32768
    while True:
        seed = (a * seed + c) % m
        yield seed / m



#For implementation of the Gauss–Jordan elimination



def gauss_jordan(matrix, vector, eps=1e-12):
    n = len(matrix)
    mat = [[float(x) for x in row] for row in matrix]

    # Augment with RHS vector
    for i in range(n):
        mat[i].append(float(vector[i]))

    # Elimination
    for i in range(n):
        # Partial pivoting
        max_row = max(range(i, n), key=lambda r: abs(mat[r][i]))
        if abs(mat[max_row][i]) < eps:
            raise ValueError("Matrix is singular or nearly singular")

        if max_row != i:
            mat[i], mat[max_row] = mat[max_row], mat[i]

        # Normalize pivot row
        pivot = mat[i][i]
        for j in range(i, n+1):
            mat[i][j] /= pivot

        # Eliminate other rows
        for k in range(n):
            if k == i:
                continue
            factor = mat[k][i]
            for j in range(i, n+1):
                mat[k][j] -= factor * mat[i][j]

    return mat


#for solving gauss jordan eqn.

def solve_equations(A_matrix, B_vector):
    reduced = gauss_jordan(A_matrix, B_vector)
    n = len(A_matrix)
    return [reduced[i][-1] for i in range(n)]

# Inverse Matrix via Gauss-Jordan
def inverse_matrix(A):
    n = len(A)
    inverse = []
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for col in range(n):
        e = [I[row][col] for row in range(n)]
        x = solve_equations([row[:] for row in A], e)
        inverse.append(x)

    # Transpose to get proper inverse
    return [[inverse[j][i] for j in range(n)] for i in range(n)]





# LU decomposition by using Doolittle method

def lu_decomp(a):
    n = len(a)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = a[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (a[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U




# Solving linear system with LU decomposition (Doolittle)



def forward_sub(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def back_sub(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x



# Code for Cholesky Decomposition

def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:  # diagonal
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = val ** 0.5
            else:  # off-diagonal
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    return x



#Function for Gauss-Seidel Iteration

def gauss_seidel(A, b, tol=1e-6, max_iter=10000):
    n = len(A)
    x = [0.0] * n
    for k in range(max_iter):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        # check convergence
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter


#Function for jacobi_iteration

def jacobi_iteration(A, b, tol=1e-6, max_iter=1000):
    n = len(b)
    x = [0.0] * n   # initial guess
    x_new = [0.0] * n

    for it in range(max_iter):
        # Compute next iteration values
        for i in range(n):
            total = 0.0
            for j in range(n):
                if j != i:
                    total += A[i][j] * x[j]
            x_new[i] = (b[i] - total) / A[i][i]

        # Check for convergence
        max_diff = 0.0
        for i in range(n):
            if abs(x_new[i] - x[i]) > max_diff:
                max_diff = abs(x_new[i] - x[i])

        if max_diff < tol:
            return x_new, it + 1

        # Update for next iteration
        for i in range(n):
            x[i] = x_new[i]

    return x, max_iter

#Function for making diagonally dominant

def make_diagonally_dominant(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            row_sum = sum(abs(A[j][k]) for k in range(n) if k != i)
            if abs(A[j][i]) >= row_sum:
                # Swap rows in both A and b
                A[i], A[j] = A[j], A[i]
                b[i], b[j] = b[j], b[i]
                break
    return A, b


#To read text file

def read_labeled_matrices(filename):
    matrices = {}
    current_label = None
    current_matrix = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:  # blank line = separator
                if current_label and current_matrix:
                    matrices[current_label] = current_matrix
                    current_label = None
                    current_matrix = []

            elif line.endswith(":"):  # label line
                current_label = line[:-1]  # remove ":"

            else:
                row = [float(num) for num in line.split()]
                current_matrix.append(row)

    # Add last matrix
    if current_label and current_matrix:
        matrices[current_label] = current_matrix


    return matrices

# Root Finding

import math


# Bisection method

def bisection(f1,a, b, tol=1e-6):
# Checking if root is bracketed
    if f1(a) * f1(b) > 0:
        raise ValueError("No root in this interval for Bisection")
# Repeat until interval is smaller than tolerance
    iterations = 0
    while (b - a) / 2 > tol:
        iterations += 1
        # midpoint
        c = (a + b) / 2
        # exact root
        if f1(c) == 0:
            return c, iterations
        # checking root lies in [a, c]
        elif f1(a) * f1(c) < 0:
            b = c
            # checking root lies in [c, b]
        else:
            a = c
    return (a + b) / 2, iterations

# Regula Falsi method


def regula_falsi(f1,a, b, tol=1e-6):
    # Checking if root is bracketed
    if f1(a) * f1(b) > 0:
        raise ValueError("No root in this interval for Regula Falsi")

    c = a
    iterations = 0
    while True:
        iterations += 1
        c_old = c
        # Formula for false position
        c = (a * f1(b) - b * f1(a)) / (f1(b) - f1(a))
        # stop if root doesn't change much
        if abs(c - c_old) < tol:
            return c, iterations
        if f1(c) == 0:
            return c, iterations
        elif f1(a) * f1(c) < 0:
            b = c
        else:
            a = c



# Functions for Question 2 FINDING BRACKET INTERVAL

def g(x):

    return -x - math.cos(x)

def bracket_interval(g,a, b, beta=0.1, max_iter=20):


    fa, fb = g(a), g(b)

    for i in range(max_iter):
        if fa * fb < 0:  # root is bracketed
            print(f"Bracket found: [{a:.6f}, {b:.6f}] with g(a)={fa:.6f}, g(b)={fb:.6f}")
            return a, b

        # Shift whichever side has smaller magnitude
        if abs(fa) < abs(fb):
            a = a - beta * (b - a)
            fa = g(a)
        else:
            b = b + beta * (b - a)
            fb = g(b)


        # Show iteration progress
        print(f"Iter {i+1}: a={a:.6f}, b={b:.6f}, g(a)={fa:.6f}, g(b)={fb:.6f}")

    # If not bracketed after max iterations
    print("No bracket found within max iterations")
    return None, None

#CODE FOR Newton Raphson
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    # Initial guess
    x = x0
    for i in range(max_iter):
        # Function value at x
        fx = f(x)
        # Derivative value at x
        dfx = df(x)

        # If function value is close enough to 0, root found
        if abs(fx) < tol:
            return x, i + 1

        # To avoid division by zero if derivative = 0
        if dfx == 0:
            break

        # Updating x using Newton–Raphson formula
        x = x - fx / dfx

    return None, max_iter

#CODE for fxed pont method
def fixed_point(g, x0, tol=1e-6, max_iter=100):
    # Initial guess
    x = x0
    for i in range(max_iter):
 # Computng next approximation using iteration function g(x)
        x_new = g(x)


        if abs(x_new - x) < tol:
            return x_new, i + 1

        x = x_new

    # If no convergence within max_iter, return None
    return None, max_iter






#  Fixed Point Method
def fixed_point_method(x0, tol=1e-6, max_iter=100):
    # starting guess values
    x1, x2, x3 = x0

    for i in range(max_iter):
        try:
            # finding new x1 from 1st eqn
            x1_new = math.sqrt(37 - x2)
        except ValueError:
            print(f"bad math for x1 in iter {i}")

            return None

        if x1_new <= 5:
            print(f"x1_new <= 5, cant do sqrt for x2 in iter {i}")
            return None

        try:

         # finding new x2 from 2nd eqn
            x2_new = math.sqrt(x1_new - 5)
        except ValueError:

            print(f"bad math for x2 in iter {i}")
            return None

        # finding x3 from 3rd eqn
        x3_new = 3 - x1_new - x2_new

        # checking for convergence
        diff = max(abs(x1_new - x1), abs(x2_new - x2), abs(x3_new - x3))

# updating the values
        x1, x2, x3 = x1_new, x2_new, x3_new

        print(f"FixedPoint iter {i + 1}: x1={x1:.6f}, x2={x2:.6f}, x3={x3:.6f}, diff={diff:.2e}")

        if diff < tol:
            return [x1, x2, x3]

    print("Fixed Point not converge in max iters")
    return [x1, x2, x3]





#  Newton Raphson Method

def newton_raphson_method(x0, tol=1e-6, max_iter=100):


    def f(x):
        x1, x2, x3 = x
        return [
            x1 ** 2 + x2 - 37,  # eqn1
            x1 - x2 ** 2 - 5,  # eqn2
            x1 + x2 + x3 - 3  # eqn3
        ]

    # DEfining the jacobian matrix
    def jacobian(x):
        x1, x2, x3 = x
        return [
            [2 * x1, 1, 0],
            [1, -2 * x2, 0],
            [1, 1, 1]
        ]


    x = list(x0)

    for i in range(max_iter):
        f_val = f(x)
        J = jacobian(x)

        try:
         # inverse of jacobian
            J_inv = inverse_matrix(J)
        except Exception as e:
            print(f"Jacobian inv fail at iter {i + 1}: {e}")
            return None


        delta = [sum(J_inv[r][c] * (-f_val[c]) for c in range(len(J))) for r in range(len(J))]

        # update x
        x_new = [x[j] + delta[j] for j in range(3)]

        # checking max differnce
        diff = max(abs(x_new[j] - x[j]) for j in range(3))
        x = x_new

        print(f"NewtonRaphson iter {i + 1}: x1={x[0]:.6f}, x2={x[1]:.6f}, x3={x[2]:.6f}, diff={diff:.2e}")

        if diff < tol:
            return x

    print("Newton Raphson not converge in max iters")
    return x


#Laguerre method

# This fun. give us value of poly. and its 1st and 2nd derivative at x
def poly_and_derivs(coeffs, x):
    n = len(coeffs) - 1
    # p is poly value, dp is first deriv, ddp is second deriv
    p = coeffs[0]
    dp = 0
    ddp = 0
    for i in range(1, n+1):
        ddp = ddp * x + 2 * dp
        dp = dp * x + p
        p = p * x + coeffs[i]
    return p, dp, ddp


# Laguerre method to find one root of poly

def laguerre(coeffs, x0, tol=1e-12, max_iter=100):
    n = len(coeffs) - 1
 # we take starting guess
    x = x0
    for _ in range(max_iter):
        p, dp, ddp = poly_and_derivs(coeffs, x)
     # if value is very small then root found
        if abs(p) < tol:
            return x
        G = dp / p
        H = G*G - ddp / p
    # we have 2 possible denom. So we pick the larger one
        denom1 = G + math.sqrt((n-1)*(n*H - G*G))
        denom2 = G - math.sqrt((n-1)*(n*H - G*G))
        if abs(denom1) > abs(denom2):
            a = n / denom1
        else:
            a = n / denom2
        x_new = x - a

        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    # return last value if no converge
    return x


# this function is deflation using synthetic division to reduce degree of polynomial

def deflate(coeffs, root):
    n = len(coeffs) - 1
    new_coeffs = [coeffs[0]]
    for i in range(1, n):
        new_coeffs.append(coeffs[i] + root*new_coeffs[-1])
    return new_coeffs

# this fun finds all roots of a poly one by one

def all_roots(coeffs):
  #we make copy so original not change
    coeffs = coeffs[:]
    roots = []
    while len(coeffs) > 2:
        # starting guess 1.5
        root = laguerre(coeffs, 1.6)
        roots.append(round(root, 6))
        coeffs = deflate(coeffs, root)
# final left linear eqn. ax+b = 0
    if len(coeffs) == 2:
        roots.append(round(-coeffs[1]/coeffs[0], 6))
    return roots


# Midpoint rule

def midpoint(f, a, b, N):
    h = (b - a) / N
    total = 0.0
    for n in range(1, N + 1):
# mid point of sub interval
        x_mid = a + (n - 0.5) * h
        total += f(x_mid)
    return h * total


# Trapezoidal rule

def trapezoidal(f, a, b, N):
    h = (b - a) / N
    total = 0.0
    for n in range(1, N + 1):
        x_prev = a + (n - 1) * h
        x_curr = a + n * h
        total += (f(x_prev) + f(x_curr))
    return (h / 2) * total

# SIMPSON’S 1/3 RULE
def simpson(f,a,b,N):
    if N % 2 == 1:    # Simpson’s rule needs even N
        N += 1
    h = (b - a) / N
    s = f(a) + f(b)
    # Apply Simpson’s coefficients
    for i in range(1, N):
        coef = 4 if i % 2 == 1 else 2
        s += coef * f(a + i * h)
    return s * h / 3

#MONTE CARLO INTEGRATION FUNCTION
def monte_sin2(N, seed=55):
    rands = lcg(seed, N)
    s = 0
    for r in rands:
        # Map r in [0,1) → x in [-1,1]
        x = -1 + 2 * r
        s += math.sin(x)**2
    return 2 * s / N   # multiply by (b-a) = 2



#Gaussian Quadrature method

def gauss_quad_integration(func, start, end, nodes, coeffs):
    # Calculate midpoint and half of the interval length
    midpoint = (start + end) / 2
    half_range = (end - start) / 2

    # Initialize total sum
    integral_sum = 0.0

    # Loop over all roots (Legendre nodes)
    for i in range(len(nodes)):
        # Transform node from [-1,1] interval to [start,end]
        x_value = midpoint + half_range * nodes[i]

        # Accumulate weighted function values
        integral_sum += coeffs[i] * func(x_value)

    # Multiply by the scaling factor to get the final integral value
    return integral_sum * half_range


import numpy as np

# Forward Euler method
def forward_euler(func, y0, x_start, x_end, h):
    # finds how many points are there in the given range
    n_steps = int((x_end - x_start) / h) + 1

    # make x and y arrays
    x_vals = np.linspace(x_start, x_end, n_steps)
    y_vals = np.zeros(n_steps)
    y_vals[0] = y0  # starting value (given)

    # loop to get all next y values
    for i in range(n_steps - 1):
        # apply Euler formula: y(i+1) = y(i) + h*f(x(i), y(i))
        y_vals[i + 1] = y_vals[i] + h * func(x_vals[i], y_vals[i])
        # basically taking small step using slope

    # return the arrays to plot or print later
    return x_vals, y_vals


# Predictor–Corrector (2 step method)

def predictor_corrector(func, y0, x_start, x_end, h):
    n_steps = int((x_end - x_start) / h) + 1
    x_vals = np.linspace(x_start, x_end, n_steps)
    y_vals = np.zeros(n_steps)
    y_vals[0] = y0

    # loop for all intervals
    for i in range(n_steps - 1):
        # predictor using euler
        y_pred = y_vals[i] + h * func(x_vals[i], y_vals[i])
        # corrector – average slope kind of thing
        y_vals[i + 1] = y_vals[i] + (h / 2) * (func(x_vals[i], y_vals[i]) + func(x_vals[i + 1], y_pred))

    return x_vals, y_vals



# Runge-Kutta 4th order method implementation FOR 1ST Order Dfferental eqn
def rk4(f, x0, y0, h, x_end):
    n = int((x_end - x0) / h)          # number of steps
    x_vals = [x0]                      # list to store x values
    y_vals = [y0]                      # list to store y values

    for i in range(n):
        # current values
        x, y = x_vals[-1], y_vals[-1]
# RK4 intermediate slopes
        k1 = f(x, y)
        k2 = f(x + h/2, y + k1*h/2)
        k3 = f(x + h/2, y + k2*h/2)
        k4 = f(x + h, y + k3*h)
    # combine slopes to find next y
        y_new = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        # update x and y lists
        x_vals.append(x + h)
        y_vals.append(y_new)
    return np.array(x_vals), np.array(y_vals)


# RK4 method for a system of two equations
def rk4_system(f, t0, state0, h, t_end):
    t_vals = [t0]          # list to store time values
    states = [state0]      # list to store [x, v] at each step
    while t_vals[-1] < t_end:
        t, s = t_vals[-1], states[-1]
        # four slopes as per RK4 formula
        k1 = f(t, s)
        k2 = f(t + h/2, s + h*k1/2)
        k3 = f(t + h/2, s + h*k2/2)
        k4 = f(t + h, s + h*k3)
        # combine to get next state
        s_new = s + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t_vals.append(t + h)
        states.append(s_new)
    return np.array(t_vals), np.array(states)



import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0                # Length of rod
T_center = 300.0       # Initial temperature at center
Nx = 40                # Number of spatial grid points (can increase for better resolution)
dx = L / (Nx - 1)      # Spatial step
dt = 0.0005            # Time step chosen so dt/(dx^2) << 0.5
Nt = 2000              # Number of time steps (adjust as needed for desired simulation time)
s = dt / dx**2

# Print stability parameter for verification
print("Stability parameter s =", s)
assert s < 0.5

# Prepare grid
x = np.linspace(0, L, Nx)

# Initial condition: all 0C except at the center
u = np.zeros(Nx)
center_index = np.argmin(np.abs(x - L/2))
u[center_index] = T_center

# Prepare to record selected snapshots
snapshots = [u.copy()]
snapshot_times = [0, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]  # time steps to plot

# FTCS time-stepping
for n in range(1, Nt+1):
    u_new = u.copy()
    for i in range(1, Nx-1):
        u_new[i] = u[i] + s * (u[i+1] - 2*u[i] + u[i-1])
    # Boundary conditions remain zero
    u_new[0] = u_new[-1] = 0
    u = u_new
    if n in snapshot_times:
        snapshots.append(u.copy())

# Plot results
plt.figure(figsize=(8,5))
for idx, snap in enumerate(snapshots):
    plt.plot(x, snap, label=f't={snapshot_times[idx]*dt:.3f}')
plt.title('1D Heat Equation: Central Pulse, Explicit FTCS')
plt.xlabel('x')
plt.ylabel('Temperature (°C)')
plt.legend(title='Time')
plt.show()



# Shooting method function

def shooting_method(guess1, guess2, tol=1e-5, max_iter=15, N=1000):
    h = L / N  # step size
    for i in range(max_iter):
     # first integration with slope guess1
        xs1, sol1 = rk4_system(heat_eq, 0, np.array([T0, guess1]), h, L)
        TL1 = sol1[-1, 0]  # temperature at x=L for first guess

# second integration with slope guess2
        xs2, sol2 = rk4_system(heat_eq, 0, np.array([T0, guess2]), h, L)
        TL2 = sol2[-1, 0]

         # update slope using secant method
        new_guess = guess2 + (TL - TL2) * (guess2 - guess1) / (TL2 - TL1)

        # print progress (optional)
        # print(f"Iter {i+1}: slope = {new_guess:.4f}, TL = {TL2:.2f}")

        if abs(TL2 - TL) < tol:
            return xs2, sol2  # done when close enough

        # shift guesses for next iteration
        guess1, guess2 = guess2, new_guess

    return xs2, sol2  # return last solution if not converged
#given constants
L = 10.0  # rod length (m)
alpha = 0.01  # heat transfer coefficient (1/m^2)
Ta = 20.0  # ambient temperature (°C)
T0 = 40.0  # temperature at x=0 (°C)
TL = 200.0  # temperature at x=L (°C)


# define the system of ODEs
# y1 = T, y2 = dT/dx
# dy1/dx = y2
# dy2/dx = -alpha * (Ta - y1)





# initial slope guesses

guess1 = 0.0
guess2 = 40.0

def heat_eq(x, state):
    T, dTdx = state
    dT = dTdx
    d2T = -alpha * (Ta - T)
    return np.array([dT, d2T])




# Function to find interpolated value using lagrange formula
def lagrange_interpoltion(x_vals, y_vals, x):
    total = 0
    n = len(x_vals)
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
    # multiply by all terms (x - xj)/(xi - xj)
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
    # add up each term to total
        total += term
        # return final estimated value
    return total


#  linear regression
def linear_regresion(x, y):
# calculate slope and intercept using basic formula
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


# Calculating Pearson’s r square value
def r_squared(x, y, slope, intercept):
  # predicted y from the regression line
    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred)**2)

    ss_tot = np.sum((y - np.mean(y))**2)
    # return coefficient of determination
    return 1 - (ss_res / ss_tot)
