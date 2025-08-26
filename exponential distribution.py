import math
import matplotlib.pyplot as plt
from all_lib import lcgg


# Generating exponential distribution numbers
N = 5000
seed = 987654321
rng = lcgg(seed)

exp_numbers = []
for _ in range(N):
    u = next(rng)
    x = -math.log(u)
    exp_numbers.append(x)

# Plotting histogram

plt.figure(figsize=(8,5))
plt.hist(exp_numbers, bins=50, density=True, alpha=0.7, color='blue', label="Generated data")
plt.plot([0, max(exp_numbers)], [math.exp(-0), math.exp(-max(exp_numbers))], 'r--', label="Theoretical exp(-x)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Exponential Distribution from Uniform [0,1) via LCG")
plt.legend()
plt.grid(True)
plt.show()
