import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

M = 1.0

# flow parameters
alpha = 0.1
beta = 0.01
q = 1.5


# spin grid
a_vals = np.linspace(0.0,0.99,120)

# Kerr geometry
def Delta(r,a):
    return r**2 - 2*M*r + a**2

def r_plus(a):
    return M + np.sqrt(M**2 - a**2)

def r_minus(a):
    return M - np.sqrt(M**2 - a**2)

# Kepler-like rotation
def Omega(r):
    return beta * r**(-q)

def dOmega(r):
    return -q * beta * r**(-q-1)

# stretching matrix
def stretching_matrix(r,a):

    Δ = Delta(r,a)

    A = -alpha - alpha*r*(r*Δ - r**2*(r-M))/(r**2*Δ)

    B = -(Δ/r**2)*(r - M*a**2/r**2)*Omega(r)

    C = dOmega(r) + (r - M*a**2/r**2)/(r**2 + a**2 + 2*M*a**2/r)*Omega(r)

    D = -alpha*r*(r - M*a**2/r**2)/(r**2 + a**2 + 2*M*a**2/r)

    return np.array([[A,B],[C,D]])

# largest eigenvalue
def lambda_max(r,a):

    S = stretching_matrix(r,a)
    eigs = np.linalg.eigvals(S)

    return np.max(np.real(eigs))

# solve lambda_max = 0
def find_r_dyn(a):

    rp = r_plus(a)

    r1 = rp + 1e-4
    r2 = 10.0

    try:
        return brentq(lambda r: lambda_max(r,a), r1, r2)
    except:
        return np.nan

# ISCO radius (prograde)
def r_ISCO(a):

    Z1 = 1 + (1 - a**2)**(1/3) * ((1+a)**(1/3) + (1-a)**(1/3))
    Z2 = np.sqrt(3*a**2 + Z1**2)

    return 3 + Z2 - np.sqrt((3-Z1)*(3+Z1+2*Z2))

# compute curves
r_dyn_vals = np.array([find_r_dyn(a) for a in a_vals])
r_plus_vals = r_plus(a_vals)
r_isco_vals = np.array([r_ISCO(a) for a in a_vals])

# plot
plt.figure(figsize=(7,5))

plt.plot(a_vals, r_dyn_vals, label="Dynamo boundary $r_{dyn}$", linewidth=2)
plt.plot(a_vals, r_isco_vals, "--", label="ISCO $r_{ISCO}$", linewidth=2)
plt.plot(a_vals, r_plus_vals, ":", label="Event horizon $r_+$", linewidth=2)

plt.xlabel("Spin ($a/M$)")
plt.ylabel("Radius ($M$)")
#plt.title("Spin dependence of dynamo boundary")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
