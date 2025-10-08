''''Shivam Kumar 2311166'''

import math

from all_lib import midpoint,trapezoidal

# This code finds numerical integration using Midpoint and Trapezoidal rules. We also compare both the results


# define the given functions
def f1(x): return 1/x
def f2(x): return x * math.cos(x)
def f3(x): return x * math.atan(x)

# exact analytical values of integrals
integrals = [
    (f1, 1, 2, 0.69314718, "∫₁² (1/x) dx = 0.69314718"),
    (f2, 0, math.pi/2, math.pi/2 - 1, "∫₀^{π/2} x cos(x) dx = π/2 - 1"),
    (f3, 0, 1, math.pi/4 - 0.5, "∫₀¹ x tan⁻¹x dx = π/4 - 1/2")
]



# now loopping over each function

N_values = [4, 8, 15, 20]

for f, a, b, exact, label in integrals:
    print(f"\n================ {label} ================")
    print(f"{'N':<5}{'Midpoint':>13}{'Trapezoidal':>15}{'Exact':>13}{'|Mid Err|':>13}{'|Trap Err|':>13}{'Rel Mid (%)':>14}{'Rel Trap (%)':>14}")
    print("-" * 100)

    for N in N_values:
        I_mid = midpoint(f, a, b, N)
        I_trap = trapezoidal(f, a, b, N)
        abs_err_mid = abs(I_mid - exact)
        abs_err_trap = abs(I_trap - exact)
        rel_err_mid = abs_err_mid / abs(exact) * 100
        rel_err_trap = abs_err_trap / abs(exact) * 100

        print(f"{N:<5}{I_mid:13.8f}{I_trap:15.8f}{exact:13.8f}{abs_err_mid:13.3e}{abs_err_trap:13.3e}{rel_err_mid:14.5f}{rel_err_trap:14.5f}")

'''
solutions
================ ∫₁² (1/x) dx = 0.69314718 ================
N         Midpoint    Trapezoidal        Exact    |Mid Err|   |Trap Err|   Rel Mid (%)  Rel Trap (%)
----------------------------------------------------------------------------------------------------
4       0.69121989     0.69702381   0.69314718    1.927e-03    3.877e-03       0.27805       0.55928
8       0.69266055     0.69412185   0.69314718    4.866e-04    9.747e-04       0.07021       0.14062
15      0.69300843     0.69342480   0.69314718    1.388e-04    2.776e-04       0.02002       0.04005
20      0.69306910     0.69330338   0.69314718    7.808e-05    1.562e-04       0.01126       0.02254


================ ∫₀^{π/2} x cos(x) dx = π/2 - 1 ================
N         Midpoint    Trapezoidal        Exact    |Mid Err|   |Trap Err|   Rel Mid (%)  Rel Trap (%)
----------------------------------------------------------------------------------------------------
4       0.58744792     0.53760713   0.57079633    1.665e-02    3.319e-02       2.91726       5.81454
8       0.57493427     0.56252752   0.57079633    4.138e-03    8.269e-03       0.72494       1.44864
15      0.57197166     0.56844624   0.57079633    1.175e-03    2.350e-03       0.20591       0.41172
20      0.57145729     0.56947459   0.57079633    6.610e-04    1.322e-03       0.11580       0.23156


================ ∫₀¹ x tan⁻¹x dx = π/4 - 1/2 ================
N         Midpoint    Trapezoidal        Exact    |Mid Err|   |Trap Err|   Rel Mid (%)  Rel Trap (%)
----------------------------------------------------------------------------------------------------
4       0.28204605     0.29209835   0.28539816    3.352e-03    6.700e-03       1.17454       2.34766
8       0.28456102     0.28707220   0.28539816    8.371e-04    1.674e-03       0.29332       0.58656
15      0.28516010     0.28587426   0.28539816    2.381e-04    4.761e-04       0.08341       0.16682
20      0.28526426     0.28566596   0.28539816    1.339e-04    2.678e-04       0.04692       0.09383
'''