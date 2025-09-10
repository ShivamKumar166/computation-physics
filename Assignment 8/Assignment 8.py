#Shivam Kumar 2311166

from all_lib import gauss_jordan,solve_equations,inverse_matrix,newton_raphson_method,fixed_point_method

import math



#Gauss Jordan method is used

initial_guess = [1, 3, 2.1]
#Solving by Fixed Point Method
print("Fixed Point Method Solution:")
fp_solution = fixed_point_method(initial_guess)
print("Solution:", fp_solution)

#Solving by Newton-Raphson Method 
print("\nNewton-Raphson Method Solution:")
nr_solution = newton_raphson_method(initial_guess)
print("Solution:", nr_solution)


#Solutions

"""
Fixed Point Method Solution:
FixedPoint iter 1: x1=5.830952, x2=0.911566, x3=-3.742518, diff=5.84e+00
FixedPoint iter 2: x1=6.007365, x2=1.003676, x3=-4.011041, diff=2.69e-01
FixedPoint iter 3: x1=5.999694, x2=0.999847, x3=-3.999541, diff=1.15e-02
FixedPoint iter 4: x1=6.000013, x2=1.000006, x3=-4.000019, diff=4.79e-04
FixedPoint iter 5: x1=5.999999, x2=1.000000, x3=-3.999999, diff=1.99e-05
FixedPoint iter 6: x1=6.000000, x2=1.000000, x3=-4.000000, diff=8.31e-07
Solution: [6.000000022160193, 1.0000000110800964, -4.000000033240289]

Newton-Raphson Method Solution:
NewtonRaphson iter 1: x1=17.230769, x2=3.538462, x3=-17.769231, diff=1.99e+01
NewtonRaphson iter 2: x1=9.618767, x2=2.421883, x3=-9.040650, diff=8.73e+00
NewtonRaphson iter 3: x1=6.652031, x2=1.552005, x3=-5.204036, diff=3.84e+00
NewtonRaphson iter 4: x1=6.023996, x2=1.105897, x3=-4.129893, diff=1.07e+00
NewtonRaphson iter 5: x1=5.999640, x2=1.004908, x3=-4.004548, diff=1.25e-01
NewtonRaphson iter 6: x1=5.999999, x2=1.000012, x3=-4.000011, diff=4.90e-03
NewtonRaphson iter 7: x1=6.000000, x2=1.000000, x3=-4.000000, diff=1.15e-05
NewtonRaphson iter 8: x1=6.000000, x2=1.000000, x3=-4.000000, diff=6.36e-11
Solution: [6.0, 1.0000000000000002, -4.0]
"""


