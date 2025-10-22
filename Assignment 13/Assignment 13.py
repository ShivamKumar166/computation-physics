# Shivam Kumar 2311166


# Assignment 13 – Solving ODEs
# Methods used: Forward Euler & Predictor–Corrector
# Compare results with the analytical solution and plot the difference between the result of Forward Euler & Predictor–Corrector method

from all_lib import forward_euler,predictor_corrector
import numpy as np

import matplotlib.pyplot as plt


# Eqn 1: dy/dx = y - x^2 , y(0)=0 , range [0,2]

def ode1(x, y):
    # fn for 1st diff eqn.
    return y - x**2

def exact_sol1(x):
    # given analytical ans for checking error
    return x**2 + 2*x + 2 - 2*np.exp(x)


# Eqn 2: dy/dx = (x + y)^2 , y(0)=1 , range [0, π/5]

def ode2(x, y):
    # 2nd eqn fn
    return (x + y)**2

def exact_sol2(x):
    # analytical formula for comparision
    return np.tan(x + np.pi/4) - x







# Solve 1st ODE

x1_euler, y1_euler = forward_euler(ode1, 0, 0, 2, 0.1)
x1_corr, y1_corr = predictor_corrector(ode1, 0, 0, 2, 0.1)
y1_exact = exact_sol1(x1_euler)


# Solve 2nd ODE

x2_euler, y2_euler = forward_euler(ode2, 1, 0, np.pi / 5, 0.1)
x2_corr, y2_corr = predictor_corrector(ode2, 1, 0, np.pi / 5, 0.1)
y2_exact = exact_sol2(x2_euler)


# print the data points for checking

print("---- Data points for Equation 1 (dy/dx = y - x^2) ----")
for i in range(len(x1_euler)):
    print(f"x = {x1_euler[i]:.2f},  Euler y = {y1_euler[i]:.5f},  PC y = {y1_corr[i]:.5f},  Exact y = {y1_exact[i]:.5f}")

print("\n---- Data points for Equation 2 (dy/dx = (x + y)^2) ----")
for i in range(len(x2_euler)):
    print(f"x = {x2_euler[i]:.2f},  Euler y = {y2_euler[i]:.5f},  PC y = {y2_corr[i]:.5f},  Exact y = {y2_exact[i]:.5f}")


# Plot both results

plt.figure(figsize=(11, 4))

# for eqn 1
plt.subplot(1, 2, 1)
plt.plot(x1_euler, y1_euler, 'c--', label='Forward Euler')
plt.plot(x1_corr, y1_corr, 's--', label='Predictor–Corrector')
plt.plot(x1_euler, y1_exact, 'm-', label='Analytical')
plt.title(r"$\frac{dy}{dx} = y - x^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# for eqn 2
plt.subplot(1, 2, 2)
plt.plot(x2_euler, y2_euler, 'c', label='Forward Euler')
plt.plot(x2_corr, y2_corr, 's--', label='Predictor–Corrector')
plt.plot(x2_euler, y2_exact, 'm', label='Analytical')
plt.title(r"$\frac{dy}{dx} = (x + y)^2$")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

#Solution

'''
---- Data points for Equation 1 (dy/dx = y - x^2) ----
x = 0.00,  Euler y = 0.00000,  PC y = 0.00000,  Exact y = 0.00000
x = 0.10,  Euler y = 0.00000,  PC y = -0.00050,  Exact y = -0.00034
x = 0.20,  Euler y = -0.00100,  PC y = -0.00310,  Exact y = -0.00281
x = 0.30,  Euler y = -0.00510,  PC y = -0.01013,  Exact y = -0.00972
x = 0.40,  Euler y = -0.01461,  PC y = -0.02414,  Exact y = -0.02365
x = 0.50,  Euler y = -0.03207,  PC y = -0.04798,  Exact y = -0.04744
x = 0.60,  Euler y = -0.06028,  PC y = -0.08476,  Exact y = -0.08424
x = 0.70,  Euler y = -0.10231,  PC y = -0.13796,  Exact y = -0.13751
x = 0.80,  Euler y = -0.16154,  PC y = -0.21140,  Exact y = -0.21108
x = 0.90,  Euler y = -0.24169,  PC y = -0.30930,  Exact y = -0.30921
x = 1.00,  Euler y = -0.34686,  PC y = -0.43632,  Exact y = -0.43656
x = 1.10,  Euler y = -0.48155,  PC y = -0.59764,  Exact y = -0.59833
x = 1.20,  Euler y = -0.65070,  PC y = -0.79894,  Exact y = -0.80023
x = 1.30,  Euler y = -0.85977,  PC y = -1.04653,  Exact y = -1.04859
x = 1.40,  Euler y = -1.11475,  PC y = -1.34736,  Exact y = -1.35040
x = 1.50,  Euler y = -1.42222,  PC y = -1.70914,  Exact y = -1.71338
x = 1.60,  Euler y = -1.78944,  PC y = -2.14035,  Exact y = -2.14606
x = 1.70,  Euler y = -2.22439,  PC y = -2.65038,  Exact y = -2.65789
x = 1.80,  Euler y = -2.73583,  PC y = -3.24962,  Exact y = -3.25929
x = 1.90,  Euler y = -3.33341,  PC y = -3.94953,  Exact y = -3.96179
x = 2.00,  Euler y = -4.02775,  PC y = -4.76279,  Exact y = -4.77811

---- Data points for Equation 2 (dy/dx = (x + y)^2) ----
x = 0.00,  Euler y = 1.00000,  PC y = 1.00000,  Exact y = 1.00000
x = 0.10,  Euler y = 1.10000,  PC y = 1.12257,  Exact y = 1.13018
x = 0.21,  Euler y = 1.24513,  PC y = 1.30779,  Exact y = 1.33043
x = 0.31,  Euler y = 1.45671,  PC y = 1.59441,  Exact y = 1.64845
x = 0.42,  Euler y = 1.77031,  PC y = 2.05918,  Exact y = 2.18621
x = 0.52,  Euler y = 2.24957,  PC y = 2.87721,  Exact y = 3.20845
x = 0.63,  Euler y = 3.01861,  PC y = 4.54224,  Exact y = 5.68543
'''
