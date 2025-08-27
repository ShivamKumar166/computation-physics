"""""Shivam Kumar 2311166"""""
#Estimation of π using LCG

#importing the functions
import math
import matplotlib.pyplot as plt
from all_lib import lcgg

# Parameters
max_N = 2000
seed = 123456789
d= lcgg(seed)

hits = 0
Ns = []
pi_values = []

#ESTIMATING PI USING RANDOM NUMBERS
for N in range(1, max_N + 1):
    x = next(d)
    y = next(d)

    if x**2 + y**2 <= 1.0:
        hits += 1

    if N % 20 == 0:
        pi_estimate = 4.0 * hits / N
        print(f"N = {N}, π ≈ {pi_estimate}")
        Ns.append(N)
        pi_values.append(pi_estimate)

# Plotting no. of throws v/s estimated value of pi
plt.figure(figsize=(8, 5))
plt.plot(Ns, pi_values, marker='o', linestyle='-', color='blue', label="Estimated π")
plt.axhline(math.pi, linestyle='--', color='red', linewidth=1, label="True π")
plt.xlabel("Number of Throws (N)")
plt.ylabel("Estimated π")
plt.title("Estimation of π using LCG ")
plt.legend()
plt.grid(True)
plt.show()