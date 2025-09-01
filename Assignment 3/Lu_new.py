# Shivam Kumar 2311166

from all_lib import lu_decomp, forward_sub, back_sub,read_labeled_matrices


# FUNCTION FOR CALLING THE TEXT FILE IS IN THE LIBRARY
# Loading matrices from text file
matrices = read_labeled_matrices("LU_matrix.txt")
print("Loaded matrices:", list(matrices.keys()))

# Extract matrices
A = matrices["A"]
Matrix = matrices["Matrix"]
b = matrices["b"][0]  # b is a single row

# LU decomposition by using Doolittle method

L, U = lu_decomp(A)

# Verify A = L * U
n = len(A)
A_creates = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

print("\nL matrix:")
for row in L:
    print(row)

print("\nU matrix:")
for row in U:
    print(row)

print("\nReconstructed A (L*U):")
for row in A_creates:
    print(row)

"""
L matrix:
[1.0, 0.0, 0.0]
[3.0, 1.0, 0.0]
[2.0, 1.0, 1.0]

U matrix:
[1.0, 2.0, 4.0]
[0.0, 2.0, 2.0]
[0.0, 0.0, 3.0]

Reconstructed A (L*U):
[1.0, 2.0, 4.0]
[3.0, 8.0, 14.0]
[2.0, 6.0, 13.0]
"""

print('STARTING OF 2ND ANSWER---->')

######## Q.2 Programm
# Solve linear system with LU decomposition (Doolittle)

L, U = lu_decomp(Matrix)
y = forward_sub(L, b)
x = back_sub(U, y)

n = len(Matrix)
A_crea = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

print("\nReconstructed Matrix (L*U):")
for row in A_crea:
    print(row)

print("\nSolution (a1 to a6):", x)

"""
Reconstructed Matrix (L*U):
[1.0, -1.0, 4.0, 0.0, 2.0, 9.0]
[0.0, 5.0, -2.0, 7.0, 8.0, 4.0]
[1.0, 0.0, 5.0, 7.0, 3.0, -2.000000000000001]
[6.0, -1.0, 2.0, 3.0, 0.0, 8.0]
[-4.0, 2.0, -6.661338147750939e-16, 5.0000000000000036, -5.0, 3.0000000000000013]
[0.0, 7.0, -1.0000000000000002, 5.000000000000001, 4.000000000000001, -2.0000000000000018]

Solution (a1 to a6): [-1.7618170439978584, 0.8962280338740118, 4.051931404116156, -1.6171308025395417, 2.041913538501913, 0.15183248715593536]
"""

