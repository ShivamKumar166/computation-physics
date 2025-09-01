from all_lib import cholesky_decomposition,forward_substitution,backward_substitution,jacobi_iteration,read_labeled_matrices

#  Loading matrices from text file
matrices = read_labeled_matrices("Assignment 4 matrix.txt")
print("Loaded matrices:", list(matrices.keys()))

# Extract matrices
A = matrices["A"]

b = matrices["b"][0]

# Q1: Cholesky Decomposition: Using Cholesky factorization for the matrix A and solving for Ax = b

L = cholesky_decomposition(A)

# Compute L^T
n = len(A)
LT = [[L[j][i] for j in range(n)] for i in range(n)]

# Solve L*y = b
y = forward_substitution(L, b)

# Solve L^T * x = y
x_cholesky = backward_substitution(LT, y)

print("Solution using Cholesky:", x_cholesky)

#solution

"""Solution using Cholesky: [-1.1102230246251565e-16, 0.9999999999999999, 1.0, 1.0000000000000004]"""


# Q2: Jacobi Iteration: Using Jacobi iterative method to solve the above matrix to a precision of 10−6

x_jacobi, iterations = jacobi_iteration(A, b, tol=1e-6)

print("Solution using Jacobi:", x_jacobi)
print("Jacobi iterations:", iterations)

#Solution

"""Solution using Jacobi: [0.0, 0.9999994039535522, 0.9999997019767761, 0.9999997019767761]
Jacobi iterations: 42"""