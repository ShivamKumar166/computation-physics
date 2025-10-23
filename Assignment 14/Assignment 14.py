# Shivam Kumar 2311166

# Assignment 14 – Solving ODEs usng RK-4 Method


# Runge-Kutta 4th Order Method for a First-Order ODE
# Equation: dy/dx = (x + y)^2,  y(0) = 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function representing dy/dx
def f(x, y):
    return (x + y)**2

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

# Analytical (exact) solution for comparison
def y_true(x):
    return np.tan(x + np.pi/4) - x

# Step sizes to test
h_values = [0.1, 0.25, 0.45]

# Loop over each step size
for h in h_values:
    # Compute RK4 numerical solution
    x, y = rk4(f, 0, 1, h, np.pi/5)
    # Compute analytical values at same x
    y_exact = y_true(x)

    # Display results in tabular form
    table = pd.DataFrame({
        'x': x,
        'RK4 y': y,
        'Analytical y': y_exact,
        'Error': np.abs(y - y_exact)
    })
    print(f"\n--- Results for step size h = {h} ---")
    print(table.to_string(index=False, float_format="%.5f"))

    # Plot RK4 results
    plt.plot(x, y, 'o--', label=f'RK4 (h={h})')

# Plot analytical curve for comparison
x_true = np.linspace(0, np.pi/5, 100)
plt.plot(x_true, y_true(x_true), 'k', label='Analytical')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison: RK4 vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()











#Queston 2
# Runge-Kutta 4th Order Method for a Second-Order ODE
# Equation: ẍ + μẋ + ω²x = 0
# Initial conditions: x(0) = 1,  v(0) = 0

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Given parameters ---
mu = 0.15          # damping coefficient
omega = 1.0        # natural frequency
x0, v0 = 1.0, 0.0  # initial displacement and velocity
t_end, h = 40, 0.01  # total time and time step

#System of first-order ODEs
# x' = v
# v' = -μv - ω²x
def f(t, state):
    x, v = state
    dxdt = v
    dvdt = -mu * v - omega**2 * x
    return np.array([dxdt, dvdt])

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

#Solve the system using RK4
t, sol = rk4_system(f, 0, np.array([x0, v0]), h, t_end)
x, v = sol[:,0], sol[:,1]   # separate displacement and velocity
E = 0.5 * (v**2 + omega**2 * x**2)  # total mechanical energy

#Show sample results in a table
sample_idx = np.arange(0, len(t), 20)  # print every 20th point
table = pd.DataFrame({
    't': t[sample_idx],
    'x': x[sample_idx],
    'v': v[sample_idx],
    'E': E[sample_idx]
})
print("\n--- Sample Results (every 20th point) ---")
print(table.to_string(index=False, float_format="%.5f"))

#Separate Plots

# Displacement vs Time
plt.figure(figsize=(7,4))
plt.plot(t, x, color='blue')
plt.xlabel('Time (t)')
plt.ylabel('Displacement x(t)')
plt.title('Displacement vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("displacement_vs_time.png", dpi=300)
plt.show()

#  Velocity vs Time
plt.figure(figsize=(7,4))
plt.plot(t, v, color='orange')
plt.xlabel('Time (t)')
plt.ylabel('Velocity v(t)')
plt.title('Velocity vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("velocity_vs_time.png", dpi=300)
plt.show()

#  Energy vs Time
plt.figure(figsize=(7,4))
plt.plot(t, E, color='green')
plt.xlabel('Time (t)')
plt.ylabel('Energy E(t)')
plt.title('Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_vs_time.png", dpi=300)
plt.show()

#  Phase Space Spiral (Scatter)
plt.figure(figsize=(6,6))
plt.scatter(x, v, s=10, color='purple')
plt.xlabel('Displacement x')
plt.ylabel('Velocity v')
plt.title('Phase Space Spiral (v vs x)')
plt.grid(True)
plt.tight_layout()
plt.savefig("phase_spiral_scatter.png", dpi=300)
plt.show()
