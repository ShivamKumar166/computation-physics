# Shivam Kumar 2311166

# This program demonstrates numerical integration using
# both Simpson’s Rule and Gaussian Quadrature methods.


from all_lib import simpson,gauss_quad_integration
import numpy as np


# Question 1
# Function to integrate: f(x) = (x^2) / (1 + x^4)


def func_q1(x):
    return (x ** 2) / (1 + x ** 4)


print("Question 1: Gaussian Quadrature for different orders (n)")
for n_points in [4, 8, 12, 16]:
    # Get Legendre polynomial roots and weights for n_points
    roots, weights = np.polynomial.legendre.leggauss(n_points)

    # Compute integration using Gaussian Quadrature
    result = gauss_quad_integration(func_q1, -1, 1, roots, weights)

    print(f"n = {n_points:2d}, Approx. Integral = {result:.9f}")



# Question 2
# Function to integrate: f(x) = sqrt(1 + x^4)

def func_q2(x):
    return np.sqrt(1 + x ** 4)

# Integration bounds
lower_limit, upper_limit = 0, 1

print("\nQuestion 2:")
print("Simpson’s Rule for different subintervals (N):")

# Using Simpson’s Rule for various numbers of subintervals
for num_intervals in [1, 5, 10, 15, 25, 30, 32, 40]:
    result = simpson(func_q2, lower_limit, upper_limit, num_intervals)
    print(f"N = {num_intervals:2d}, Simpson's Rule Result = {result:.9f}")

# Using Gaussian Quadrature for comparison
print("\nGaussian Quadrature Results:")
for n_points in [4, 8, 12, 16]:
    roots, weights = np.polynomial.legendre.leggauss(n_points)
    result = gauss_quad_integration(func_q2, lower_limit, upper_limit, roots, weights)
    print(f"n = {n_points:2d}, Gaussian Quadrature Result = {result:.9f}")

'''Question 1: Gaussian Quadrature for different orders (n)
n =  4, Approx. Integral = 0.481635482
n =  8, Approx. Integral = 0.487501940
n = 12, Approx. Integral = 0.487495520
n = 16, Approx. Integral = 0.487495494

Question 2:
Simpson’s Rule for different subintervals (N):
N =  1, Simpson's Rule Result = 1.089553198
N =  5, Simpson's Rule Result = 1.089428758
N = 10, Simpson's Rule Result = 1.089429384
N = 15, Simpson's Rule Result = 1.089429412
N = 25, Simpson's Rule Result = 1.089429413
N = 30, Simpson's Rule Result = 1.089429413
N = 32, Simpson's Rule Result = 1.089429413
N = 40, Simpson's Rule Result = 1.089429413

Gaussian Quadrature Results:
n =  4, Gaussian Quadrature Result = 1.089424360
n =  8, Gaussian Quadrature Result = 1.089429413
n = 12, Gaussian Quadrature Result = 1.089429413
n = 16, Gaussian Quadrature Result = 1.089429413'''