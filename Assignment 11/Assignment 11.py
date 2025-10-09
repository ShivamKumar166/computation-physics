#ASSIGNMENT 10 : MIDPOINT, SIMPSON’S 1/3 & MONTE CARLO METHODS
from all_lib import midpoint, lcg,simpson,monte_sin2
import math
import matplotlib.pyplot as plt

#DEFINING FUNCTIONS FOR THE GIVEN INTEGRALS

def f1(x):

    return 1/x

def f2(x):
    return x*math.cos(x)


#EXACT (ANALYTICAL) VALUES OF INTEGRALS

I1 = math.log(2)          # ∫ 1→2 (1/x) dx
I2 = math.pi/2 - 1        # ∫ 0→π/2 x*cos(x) dx

# INTEGRATION LIMITS AND N VALUES
# (N values were chosen from error bound formula calculations)

a1, b1 = 1, 2
a2, b2 = 0, math.pi/2

N1m, N1s = 289, 20      # For 1/x integral
N2m, N2s = 610, 22      # For x*cosx integral


# CALCULATE NUMERICAL INTEGRALS

I1_mid = midpoint(f1, a1, b1, N1m)
I1_sim = simpson(f1, a1, b1, N1s)
I2_mid = midpoint(f2, a2, b2, N2m)
I2_sim = simpson(f2, a2, b2, N2s)


#ERROR CALCULATIONS

def err(val, true):
    absE = abs(val - true)
    relE = absE / abs(true)
    return absE, relE

e1m, r1m = err(I1_mid, I1)
e1s, r1s = err(I1_sim, I1)
e2m, r2m = err(I2_mid, I2)
e2s, r2s = err(I2_sim, I2)


#DISPLAY MIDPOINT & SIMPSON RESULTS

print("\n=== MIDPOINT & SIMPSON RESULTS ===")
print(f"Integral (1/x) from 1 to 2")
print(f" Midpoint N={N1m:4d}: {I1_mid:.8f}  |err|={e1m:.2e}  Rel={r1m:.2e}")
print(f" Simpson  N={N1s:4d}: {I1_sim:.8f}  |err|={e1s:.2e}  Rel={r1s:.2e}")

print(f"\nIntegral (x*cosx) from 0 to pi/2")
print(f" Midpoint N={N2m:4d}: {I2_mid:.8f}  |err|={e2m:.2e}  Rel={r2m:.2e}")
print(f" Simpson  N={N2s:4d}: {I2_sim:.8f}  |err|={e2s:.2e}  Rel={r2s:.2e}")
'''
Answer 1

=== MIDPOINT & SIMPSON RESULTS ===
Integral (1/x) from 1 to 2
 Midpoint N= 289: 0.69314681  |err|=3.74e-07  Rel=5.40e-07
 Simpson  N=  20: 0.69314737  |err|=1.94e-07  Rel=2.80e-07

Integral (x*cosx) from 0 to pi/2
 Midpoint N= 610: 0.57079704  |err|=7.10e-07  Rel=1.24e-06
 Simpson  N=  22: 0.57079699  |err|=6.61e-07  Rel=1.16e-06
'''
# MONTE CARLO INTEGRATION

# ∫ sin²(x) dx from -1 to 1 using own LCG (pseudo-random generator)

print("\n=== MONTE CARLO sin^2(x) ===")


# Exact analytical value of ∫ sin²x dx from -1→1
I_exact = 1 - math.sin(2) / 2

# Try for different N values
Ns = [1000,1500,4000, 5000,7000 ,10000,14000,15000,20000,25000]

vals, errs = [], []

# Compute integral for each N
for N3 in Ns:
    I3_mc = monte_sin2(N3)
    e3 = abs(I3_mc - I_exact)
    vals.append(I3_mc)
    errs.append(e3)
    print(f"N={N3:<8d}  I={I3_mc:.3f}  |err|={e3:.3e}")


# PLOT CONVERGENCE OF MONTE CARLO METHOD

plt.figure()
plt.plot(Ns, vals, "o-", label="Monte Carlo estimate")
plt.axhline(I_exact, color="r", linestyle="--", label="Exact value")
plt.xscale("log")
plt.xlabel("Number of samples (N)")
plt.ylabel("Integral estimate")
plt.title("Convergence of Monte Carlo integration for ∫sin²x dx (-1→1)")
plt.legend()
plt.grid(True)
plt.show()
'''
Answer 2

=== MONTE CARLO sin^2(x) ===
N=1000      I=0.511  |err|=3.477e-02
N=1500      I=0.516  |err|=2.924e-02
N=4000      I=0.536  |err|=9.087e-03
N=5000      I=0.536  |err|=9.018e-03
N=7000      I=0.538  |err|=7.664e-03
N=10000     I=0.539  |err|=6.296e-03
N=14000     I=0.538  |err|=7.589e-03
N=15000     I=0.538  |err|=7.401e-03
N=20000     I=0.541  |err|=4.087e-03
N=25000     I=0.542  |err|=2.953e-03
'''
