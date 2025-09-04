"""Shivam Kumar 2311166"""

import all_lib
import math

# Question 1

# Functions for Question 1
def f1(x):

    return math.log(x/2) - math.sin(5*x/2)

print("Bisection root:", all_lib.bisection(f1,1.5, 3.0))
print("Regula Falsi root:", all_lib.regula_falsi(f1,1.5, 3.0))

"""ANSWER 1--------

Bisection root: (2.623141050338745, 20)
Regula Falsi root: (2.6231403358555436, 6)
"""

# Question 2
def g(x):

    return -x - math.cos(x)
# Test inside [2, 4]
a, b = 2, 4
print(all_lib.bracket_interval(g,a,b))


"""
Answer 2 

Iter 1: a=1.800000, b=4.000000, g(a)=-1.572798, g(b)=-3.346356
Iter 2: a=1.580000, b=4.000000, g(a)=-1.570796, g(b)=-3.346356
Iter 3: a=1.338000, b=4.000000, g(a)=-1.568699, g(b)=-3.346356
Iter 4: a=1.071800, b=4.000000, g(a)=-1.550344, g(b)=-3.346356
Iter 5: a=0.778980, b=4.000000, g(a)=-1.490611, g(b)=-3.346356
Iter 6: a=0.456878, b=4.000000, g(a)=-1.354312, g(b)=-3.346356
Iter 7: a=0.102566, b=4.000000, g(a)=-1.097311, g(b)=-3.346356
Iter 8: a=-0.287178, b=4.000000, g(a)=-0.671870, g(b)=-3.346356
Iter 9: a=-0.715895, b=4.000000, g(a)=-0.038611, g(b)=-3.346356
Iter 10: a=-1.187485, b=4.000000, g(a)=0.813491, g(b)=-3.346356
Bracket found: [-1.187485, 4.000000] with g(a)=0.813491, g(b)=-3.346356
(-1.1874849202000002, 4)
"""







