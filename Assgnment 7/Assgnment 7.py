# Shivam Kumar 2311166

import math
from all_lib import bisection,regula_falsi,newton_raphson,fixed_point

# Queston 1

#Finding the root of the following function in the interval [−1.5, 1.5] to an accuracy of 10−6 using Bisection, Regula Falsi and Newton-Raphson methods
def f1(x):
    return 3 * x + math.sin(x) - math.exp(x)

def df1(x):
    return 3 + math.cos(x) - math.exp(x)


bisection_result = bisection(f1, 0, 1)
regula_falsi_result = regula_falsi(f1, 0, 1)
newton_result = newton_raphson(f1, df1, 0.5)

# Print results (rounded to 6 decimals)
print("Bisection Method: Root = {:.6f}, Iterations = {}".format(bisection_result[0], bisection_result[1]))
print("Regula Falsi Method: Root = {:.6f}, Iterations = {}".format(regula_falsi_result[0], regula_falsi_result[1]))
print("Newton-Raphson Method: Root = {:.6f}, Iterations = {}".format(newton_result[0], newton_result[1]))

"""Answer;

Bisection Method: Root = 0.360421, Iterations = 19
Regula Falsi Method: Root = 0.360422, Iterations = 8
Newton-Raphson Method: Root = 0.360422, Iterations = 4
"""

#Question 2

#Finding the root of the following function using fixed point method

def f2(x):
    return x**2 - 2*x - 3



def g(x):
    return math.sqrt(2*x + 3)


# Run
x0 = 2.40  # initial guess
root, iterations = fixed_point(g, x0)

print("Approximate root:", root)
print("Number of iterations:", iterations)

"""Answer 2

Approximate root: 2.9999996032983853
Number of iterations: 13
"""