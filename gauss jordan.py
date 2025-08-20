#SHIVAM KUMAR 2311166


from gauss_jordan_A import A
from gauss_jordan_B import B
def print_mat(mat, title="Matrix"):
    print(f"\n{title}:")
    for row in mat:
        print(["{:.6f}".format(x) for x in row])

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

def solve_equations(A_matrix, B_vector):
    reduced = gauss_jordan(A_matrix, B_vector)
    n = len(A_matrix)
    return [reduced[i][-1] for i in range(n)]


if __name__ == "__main__":
    print("Solving equations Ax = B")
    solution = solve_equations(A,B)
    print("\nFinal solution vector x =", solution)


