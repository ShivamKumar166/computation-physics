# Shivam Kumar 2311166


# Random Number Generator

def random():
    x = 0.1
    n=0
    l=[]
    while n<=1000:
        c= 2.89
        x=c*x*(1-x)
        n+=1

        l.append(x)
    return l


# LCG Random no. generator

def lcg(seed, n):
    a = 1103515245
    c = 12345
    m = 32768
    x = seed
    numbers = []
    for _ in range(n):
        x = (a * x + c) % m
        d=x/m
        numbers.append(d)
    return numbers



#LCG generator

def lcgg(seed):

    a = 1103515245
    c = 12345
    m = 32768
    while True:
        seed = (a * seed + c) % m
        yield seed / m



#For implementation of the Gauss–Jordan elimination

def gauss_jordan(matrix, vector, eps=1e-12):
    n = len(matrix)

    mat = [[float(x) for x in row] for row in matrix]

    # Augmenting with B
    for i in range(n):
        mat[i].append(float(vector[i]))

    print_mat(mat, "Initial Augmented Matrix [A|b]")

    # Elimination
    for i in range(n):
        # Partial pivoting
        max_row = max(range(i, n), key=lambda r: abs(mat[r][i]))
        if abs(mat[max_row][i]) < eps:
            raise ValueError("Matrix is singular or nearly singular")

        if max_row != i:
            mat[i], mat[max_row] = mat[max_row], mat[i]
            print(f"\nSwapped row {i} with row {max_row}")
            print_mat(mat, f"After swapping for column {i}")

        # normalize pivot row
        pivot = mat[i][i]
        for j in range(i, n+1):
            mat[i][j] /= pivot

        # eliminate other rows
        for k in range(n):
            if k == i:
                continue
            factor = mat[k][i]
            for j in range(i, n+1):
                mat[k][j] -= factor * mat[i][j]

        print_mat(mat, f"After elimination step {i}")

    return mat

#for solving gauss jordan eqn.

def solve_equations(A_matrix, B_vector):
    reduced = gauss_jordan(A_matrix, B_vector)
    n = len(A_matrix)
    return [reduced[i][-1] for i in range(n)]

# LU decomposition by using Doolittle method

def lu_decomp(a):
    n = len(a)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = a[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (a[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U




# Solving linear system with LU decomposition (Doolittle)



def forward_sub(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def back_sub(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x



# Code for Cholesky Decomposition

def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:  # diagonal
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = val ** 0.5
            else:  # off-diagonal
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    return x



#Function for Gauss-Seidel Iteration

def gauss_seidel(A, b, tol=1e-6, max_iter=10000):
    n = len(A)
    x = [0.0] * n
    for k in range(max_iter):
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        # check convergence
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter


#Function for Gauss-Seidel Iteration

def jacobi_iteration(A, b, tol=1e-6, max_iter=1000):
    n = len(b)
    x = [0.0] * n   # initial guess
    x_new = [0.0] * n

    for it in range(max_iter):
        # Compute next iteration values
        for i in range(n):
            total = 0.0
            for j in range(n):
                if j != i:
                    total += A[i][j] * x[j]
            x_new[i] = (b[i] - total) / A[i][i]

        # Check for convergence
        max_diff = 0.0
        for i in range(n):
            if abs(x_new[i] - x[i]) > max_diff:
                max_diff = abs(x_new[i] - x[i])

        if max_diff < tol:
            return x_new, it + 1

        # Update for next iteration
        for i in range(n):
            x[i] = x_new[i]

    return x, max_iter

#Function for making diagonally dominant

def make_diagonally_dominant(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            row_sum = sum(abs(A[j][k]) for k in range(n) if k != i)
            if abs(A[j][i]) >= row_sum:
                # Swap rows in both A and b
                A[i], A[j] = A[j], A[i]
                b[i], b[j] = b[j], b[i]
                break
    return A, b

#To read text file

def read_labeled_matrices(filename):
    matrices = {}
    current_label = None
    current_matrix = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:  # blank line = separator
                if current_label and current_matrix:
                    matrices[current_label] = current_matrix
                    current_label = None
                    current_matrix = []

            elif line.endswith(":"):  # label line
                current_label = line[:-1]  # remove ":"

            else:
                row = [float(num) for num in line.split()]
                current_matrix.append(row)

    # Add last matrix
    if current_label and current_matrix:
        matrices[current_label] = current_matrix

    return matrices
