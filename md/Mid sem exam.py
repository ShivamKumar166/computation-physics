#Shivam Kumar 2311166

from all_lib import lcg,newton_raphson,lu_decomp,forward_sub,back_sub,read_labeled_matrices,gauss_seidel
import math

#1st question

# semi-major axis
a_ellipse = 2
# semi-minor axis
b_ellipse = 1
true_area = math.pi * a_ellipse * b_ellipse   # analytical area = 2π
seed = 7
N = 1000

rand_nums = lcg(seed, 2 * N)   # generate 2N random numbers
inside = 0

for i in range(N):
    # map random numbers to rectangle (-a,a) × (-b,b)
    x = (2 * rand_nums[2*i] - 1) * a_ellipse
    y = (2 * rand_nums[2*i + 1] - 1) * b_ellipse

    # check if inside ellipse
    if (x**2 / a_ellipse**2 + y**2 / b_ellipse**2) <= 1:
        inside += 1

# estimate area
rect_area = 4 * a_ellipse * b_ellipse
est_area = rect_area * inside / N
error = abs(est_area - true_area) / true_area * 100

print(f"N = {N}")
print(f"Estimated Area = {est_area:.5f}")
print(f"Analytical Area = {true_area:.5f}")
print(f"Error = {error:.3f}%")

'''ANSWER 1
N = 1000
Estimated Area = 6.43200
Analytical Area = 6.28319
Error = 2.368%'''

#2ND QUESTION
print('')
print('2nd question')



# --- Define the Wein's displacement function and its derivative ---
def f(x):
    return (x - 5) * math.exp(x) + 5

def df(x):
    return math.exp(x) * (x - 4)


# --- Initial guess ---
x0 = 5

# --- Solve using Newton–Raphson ---
root, iterations = newton_raphson(f, df, x0)

if root is not None:
    print(f"Root x = {root:.6f} found in {iterations} iterations")

    # --- Constants ---
    h = 6.626e-34  # Planck's constant
    c = 3e8        # speed of light
    k = 1.381e-23  # Boltzmann constant

    # --- Compute Wien's constant ---
    b = (h * c) / (k * root)
    print(f"Wien's displacement constant b = {b:.6e} m·K")

else:
    print("Newton-Raphson did not converge.")



'''Answer 2
Root x = 4.965114 found in 4 iterations
Wien's displacement constant b = 2.899010e-03 m·K'''





#3rd QUESTION
print('')
print('3rd question')

# Read matrices from file
matrices = read_labeled_matrices("Q.3rd_matirx.txt")

# Extract matrix A
A = matrices["A"]

# LU decomposition
L, U = lu_decomp(A)

# Compute inverse using LU
n = len(A)
A_inv = []

for i in range(n):
    e = [0.0] * n
    e[i] = 1.0
    y = forward_sub(L, e)
    x = back_sub(U, y)
    A_inv.append(x)

# Transpose the result (so columns correspond to each e_i)
A_inv = list(map(list, zip(*A_inv)))

# Print the inverse (rounded to 3 decimals)
print("Inverse of A (rounded to 3 decimals):")
for row in A_inv:
    print([round(val, 3) for val in row])

'''ANS 3
Inverse of A (rounded to 3 decimals):
[-0.708, 2.531, 2.431, 0.967, -3.902]
[-0.193, 0.31, 0.279, 0.058, -0.294]
[0.022, 0.365, 0.286, 0.051, -0.29]
[0.273, -0.13, 0.132, -0.141, 0.449]
[0.782, -2.875, -2.679, -0.701, 4.234]
'''




#4TH QUESTION
print('')
print('4TH question')




# Load the matrix A and vector b from a file
A= read_labeled_matrices("comp1.txt")
b= read_labeled_matrices("comp2.txt")
# Set tolerance and maximum number of iterations
tolerance = 1e-6
max_iterations = 1000

# Solve the system using the Gauss-Seidel method
solution, num_iterations = gauss_seidel(A, b, tolerance, max_iterations)

# Display results
print("Question 4:")
print(f"Solution vector x (Gauss-Seidel, tolerance={tolerance}):")
for idx, value in enumerate(solution, start=1):
    print(f"x{idx} = {value:.6f}")
print("Number of iterations:", num_iterations)
