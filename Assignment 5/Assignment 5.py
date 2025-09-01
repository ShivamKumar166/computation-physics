'''Shivam Kumar 2311166'''

 #1st question
from all_lib import cholesky_decomposition, forward_substitution, backward_substitution, gauss_seidel,read_labeled_matrices

#  Loading matrices from text file
matrices = read_labeled_matrices("Assignment_5_A_and_b.txt")
print("Loaded matrices:", list(matrices.keys()))

# Extract matrices
A = matrices["A"]

P = matrices["P"]
q = matrices["q"][0]
b = matrices["b"][0]  # b and q are single rows




# Function to check whether a matrix is symmetric (needed for Cholesky decomposition)
def is_symmetric(A, tol=1e-9):
    n = len(A)
    for i in range(n):


        for j in range(i + 1, n):  # only checking upper vs lower elements
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


n = len(A)

# Checking and confirm symmetry of matrix A
if is_symmetric(A):
    print("Matrix is symmetric.")
else:
    print("Matrix is NOT symmetric.")




# Solve system using Cholesky decomposition (A = L * L^T)
L = cholesky_decomposition(A)


y = forward_substitution(L, b)
U = [[L[j][i] for j in range(n)] for i in range(n)]  # transpose of L gives U




x_cholesky = backward_substitution(U, y)

print("Cholesky solution:", x_cholesky)


# Solve the same system using Gauss-Seidel iterative method
x_gs, its = gauss_seidel(A, b, tol=1e-6, max_iter=10000)

print("Gauss-Seidel solution:", x_gs)
print("Converged in", its, "iterations")


#Solutions

'''Matrix is symmetric.
Cholesky solution: [1.0, 0.9999999999999999, 1.0, 1.0, 1.0, 1.0]
Gauss-Seidel solution: [0.9999997530614102, 0.9999997892247294, 0.9999999100460266, 0.9999998509593769, 0.9999998727858708, 0.9999999457079743]
Converged in 16 iterations'''


print("\nSecond Question:\n")

from all_lib import jacobi_iteration, make_diagonally_dominant

# Ensuring P is diagonally dominant for guaranteed convergence of iterative methods
P, q = make_diagonally_dominant(P, q)

# Solving system using Jacobi method
x_jacobi, iter_j = jacobi_iteration(P, q)
print("Solution by Jacobi:", x_jacobi, "in", iter_j, "iterations")

# Solving the same system using Gauss-Seidel method and compare efficiency
x_gs, iter_gs = gauss_seidel(P, q)
print("Solution by Gauss-Seidel:", x_gs, "in", iter_gs, "iterations")

#Solution

'''Solution by Jacobi: [2.9791649583226008, 2.215599258220273, 0.21128373337161171, 0.15231661140963978, 5.71503326456748] in 58 iterations
Solution by Gauss-Seidel: [2.979165086347139, 2.215599676186742, 0.21128402698819157, 0.15231700827754802, 5.715033568811629] in 12 iterations'''
