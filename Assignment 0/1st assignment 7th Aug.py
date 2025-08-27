# Warm up exercise, Name= Shivam Kumar , Roll No.= 2311166

# 1.1 Sum of first 20 odd numbers using while loop

l = []
s_odds = 0

for p in range(1, 40):
    if p % 2 == 1:
        s_odds = s_odds + p
        l.append(p)

# 1.2 Factorial of 8 using for loop

factorial = 1
for p in range(1, 9):
    factorial *= p

print("Set of 1st 20 odd numbers=", l)
print("Sum of first 20 odd numbers =", s_odds)
print("Factorial of 8 =", factorial)

# 2. Sum of GP and HP (15 terms) using loop

# GP first term = 1.25, ratio = 0.5
f_term = 1.25
gp_s = 0
for _ in range(15):
    gp_s += f_term
    f_term *= 0.5

# HP first term = 1.25, difference = 1.5
f_term = 1.25
hp_s = 0
for _ in range(15):
    hp_s += 1 / f_term
    f_term += 1.5

print("GP sum of 15 terms =", gp_s)
print("HP sum of 15 terms =", hp_s)


# 3. multiplication and dot product of matrices

def read_matrix(assign0_gen):
    with open(assign0_gen, 'r') as f:
        matrix = []
        for line in f:
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix


mtx_A = read_matrix('asgn0_matA')
mtx_B = read_matrix('asgn0_matB')
mtx_C = read_matrix('asgn0_vecC')

mtx_D = read_matrix('asgn0_vecD')

# Dot product of D.C
s = 0
for p in range(len(mtx_D)):
    s += mtx_D[p][0] * mtx_C[p][0]
print("Dot product of DC=", s)

# doing multipication of AB
AB = []
for p in range(len(mtx_A)):
    row = []
    for q in range(len(mtx_B[0])):
        s = 0
        for k in range(len(mtx_B)):
            s += mtx_A[p][k] * mtx_B[k][q]
        row.append(s)
    AB.append(row)
print('AB =', AB)

#  BC multiplication
BC = []
for p in range(len(mtx_B)):
    row = []
    for q in range(len(mtx_C[0])):
        s = 0
        for k in range(len(mtx_C)):
            s += mtx_B[p][k] * mtx_C[k][q]
        row.append(s)
    BC.append(row)
print('BC =', BC)

"""
Results

Set of 1st 20 odd numbers= [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
Sum of first 20 odd numbers = 400
Factorial of 8 = 40320
GP sum of 15 terms = 2.4999237060546875
HP sum of 15 terms = 2.4139570733659186
Dot product of DC= -3.5
AB = [[-0.3000000000000007, -3.5, 5.2], [-4.5, -2.0, 4.5], [9.3, 0.8, -7.0]]
BC = [[1.0], [-5.75], [-9.0]]"""

