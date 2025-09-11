# Shivam Kumar 2311166

#Using Laguerre’s method and deflation to find the roots (all real) of the following three polynomials

from all_lib import poly_and_derivs,laguerre,deflate,all_roots
import math



# polynomials from assignmnt
"""
 x^4 - x^3 - 7x^2 + x + 6
 x^4 - 5x^2 + 4
 2x^5 - 19.5x^3 + 0.5x^2 + 13.5x - 4.5
"""
P1 = [1, -1, -7, 1, 6]
P2 = [1, 0, -5, 0, 4]
P3 = [2, 0, -19.5, 0.5, 13.5, -4.5]

print("Roots of P1:", all_roots(P1))
print("Roots of P2:", all_roots(P2))
print("Roots of P3:", all_roots(P3))


#Solutions
"""
Roots of P1: [1.0, 3.0, -1.0, -2.0]
Roots of P2: [2.0, 1.0, -1.0, -2.0]
Roots of P3: [0.5, 0.5, 3.0, -1.0, -3.0]

"""