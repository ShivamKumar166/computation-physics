import all_lib


# Question 1

print("Bisection root:", all_lib.bisection(1.5, 3.0))
print("Regula Falsi root:", all_lib.regula_falsi(1.5, 3.0))

# Question 2
# Test inside [2, 4]
a, b = 2, 4
print(all_lib.bracket_interval(a,b))



