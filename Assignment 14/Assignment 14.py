# Shivam Kumar 2311166

# Assignment 14 – Solving ODEs usng RK-4 Method


# Runge-Kutta 4th Order Method for a First-Order ODE
# Equation: dy/dx = (x + y)^2,  y(0) = 1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from all_lib import rk4,rk4_system
# Function representing dy/dx
def f(x, y):
    return (x + y)**2


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
