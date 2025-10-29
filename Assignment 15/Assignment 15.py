# Assignment 15 - Shooting Method using RK4
# Shivam Kumar (Roll No: 2311166)


import numpy as np
import matplotlib.pyplot as plt
from all_lib import rk4_system,shooting_method,heat_eq

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

# solve using shooting method
x_vals, states = shooting_method(guess1, guess2)
T_vals = states[:, 0]


# find x where T = 100°C (linear interpolation)

idx = np.where(np.diff(np.sign(T_vals - 100)))[0][0]
x_100 = x_vals[idx] + (100 - T_vals[idx]) * (x_vals[idx + 1] - x_vals[idx]) / (T_vals[idx + 1] - T_vals[idx])
print(f"\nT = 100°C occurs at x ≈ {x_100:.2f} m\n")


# print table for every 1 meter position

print(" x (m)     T(x) (°C)")
for x_target in np.arange(0, L + 1, 1):
    idx = np.argmin(np.abs(x_vals - x_target))
    print(f"{x_vals[idx]:5.1f}     {T_vals[idx]:8.2f}")


# plot the temperature profile

plt.figure(figsize=(7, 4))
plt.plot(x_vals, T_vals, 'b-', label='Temperature Profile')
plt.axhline(100, color='r', linestyle='--', label='T = 100°C')
plt.xlabel('x (m)')
plt.ylabel('T (°C)')
plt.title('Heat Conduction in a Rod (Shooting Method)')
plt.legend()
plt.grid(True)
plt.show()

'''ANSWERS

Stability parameter s = 0.19012500000000002

T = 100°C occurs at x ≈ 4.43 m

 x (m)     T(x) (°C)
  0.0        40.00
  1.0        52.79
  2.0        65.91
  3.0        79.50
  4.0        93.67
  5.0       108.58
  6.0       124.38
  7.0       141.23
  8.0       159.29
  9.0       178.74
 10.0       199.78
'''