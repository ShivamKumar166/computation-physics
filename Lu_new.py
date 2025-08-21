# Shivam Kumar 2311166

# LU decomposition by using Doolittle method

from A_LU import A
from all_lib import lu_decomp

L, U = lu_decomp(A)

# Verify A = L * U
n = len(A)
A_creates = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

print("L matrix:")
for row in L:
    print(row)
print("U matrix:")
for row in U:
    print(row)
print("Reconstructed A (L*U):")
for row in A_creates:
    print(row)



"""L matrix:
[1.0, 0.0, 0.0]
[3.0, 1.0, 0.0]
[2.0, 1.0, 1.0]
U matrix:
[1, 2, 4]
[0.0, 2.0, 2.0]
[0.0, 0.0, 3.0]
Reconstructed A (L*U):
[1.0, 2.0, 4.0]
[3.0, 8.0, 14.0]
[2.0, 6.0, 13.0]"""

print('STARTING OF 2ND ANSWER---->')

######## Q.2 Programm
# Solve linear system with LU decomposition (Doolittle)

from A_LU import b,Matrix
from all_lib import lu_decomp
from all_lib import forward_sub,back_sub


L, U = lu_decomp(Matrix)
y = forward_sub(L, b)
x = back_sub(U, y)

n = len(Matrix)
A_crea = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

print("Reconstructed A (L*U):")
for row in A_crea:
    print(row)

print("Solution (a1 to a6):", x)


"""Matrix A:
[1, -1, 4, 0, 2, 9]
[0, 5, -2, 7, 8, 4]
[1, 0, 5, 7, 3, -2]
[6, -1, 2, 3, 0, 8]
[-4, 2, 0, 5, -5, 3]
[0, 7, -1, 5, 4, -2]
Solution (a1 to a6): [-1.7618170439978584, 0.8962280338740118, 4.051931404116156, -1.6171308025395417, 2.041913538501913, 0.15183248715593536]"""
