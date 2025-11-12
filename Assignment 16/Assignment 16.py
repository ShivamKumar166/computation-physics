# Shivam Kumar 2311166
# Assignment 16: Lagrange Interpolation and Data Fitting
from all_lib import lagrange_interpoltion,linear_regresion,r_squared
import numpy as np
import matplotlib.pyplot as plt


# Question 1: Lagrange Interpolation

x1 = np.array([2, 3, 5, 8, 12])
y1 = np.array([10, 15, 25, 40, 60])



x_eval = 6.7
y_eval = lagrange_interpoltion(x1, y1, x_eval)

print("Question 1: Lagrange Interpolation")
print(f"Estimated value of y(6.7) = {y_eval:.4f}\n")




# Question 2: Data Fitting - Power Law and Exponential Models

x2 = np.array([2.5, 3.5, 5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.5])
y2 = np.array([13.0, 11.0, 8.5, 8.2, 7.0, 6.2, 5.2, 4.8, 4.6, 4.3])


# Power law model: y = a * x^b
logx = np.log(x2)
logy = np.log(y2)
slope_p, intercept_p = linear_regresion(logx, logy)
a_p = np.exp(intercept_p)
b_p = slope_p
r2_power = r_squared(logx, logy, slope_p, intercept_p)

# Exponential model: y = a * e^(-b*x)
logy2 = np.log(y2)
slope_e, intercept_e = linear_regresion(x2, logy2)
a_e = np.exp(intercept_e)
b_e = -slope_e
r2_exp = r_squared(x2, logy2, slope_e, intercept_e)

# plot both fitted curves
x_fit = np.linspace(min(x2), max(x2), 100)
y_fit_power = a_p * x_fit**b_p
y_fit_exp = a_e * np.exp(-b_e * x_fit)

plt.scatter(x2, y2, color='r', label='Data Points')
plt.plot(x_fit, y_fit_power, label=f'Power law fit (r²={r2_power:.4f})')
plt.plot(x_fit, y_fit_exp, label=f'Exponential fit (r²={r2_exp:.4f})')
plt.title('Data Fitting Comparison: Power Law vs Exponential')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# print results
print("Question 2: Data Fitting Results")
print("Power Law Model: y = a * x^b")
print(f"  a = {a_p:.6f}")
print(f"  b = {b_p:.6f}")
print(f"  r² = {r2_power:.6f}")
print()
print("Exponential Model: y = a * e^(-b*x)")
print(f"  a = {a_e:.6f}")
print(f"  b = {b_e:.6f}")
print(f"  r² = {r2_exp:.6f}")

# check which fit is better
better_model = "Power Law" if r2_power > r2_exp else "Exponential"
print(f"\nBetter Fit Based on r² is {better_model} Model")

"""   
Question 1: Lagrange Interpolation
Estimated value of y(6.7) = 33.5000

Question 2: Data Fitting Results
Power Law Model: y = a * x^b
  a = 21.046352
  b = -0.537409
  r² = 0.994518

Exponential Model: y = a * e^(-b*x)
  a = 12.212993
  b = 0.058456
  r² = 0.901792

Better Fit Based on r² is Power Law Model 
"""