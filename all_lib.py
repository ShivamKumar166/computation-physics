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



