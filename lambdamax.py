import numpy as np
import matplotlib.pyplot as plt

M = 1.0

# Flow parameters (you can tweak these)
alpha = 0.1
beta = 0.5
q = 1.5  # Kepler-like

def Delta(r, a):
    return r**2 - 2*M*r + a**2

def Omega(r):
    return beta * r**(-q)

def dOmega(r):
    return -q * beta * r**(-q - 1)

def stretching_matrix(r, a):
    D = Delta(r, a)

    # Metric factors
    gphiphi = r**2 + a**2 + 2*M*a**2 / r
    factor = (r - M*a**2 / r**2)

    # Components
    A = -alpha - alpha * r * (r*D - r**2*(r - M)) / (r**2 * D)
    B = -(D / r**2) * factor * Omega(r)
    C = dOmega(r) + (factor / gphiphi) * Omega(r)
    Dcomp = -alpha * r * factor / gphiphi

    return np.array([[A, B],
                     [C, Dcomp]])

def lambda_max(r, a):
    S = stretching_matrix(r, a)
    eigvals = np.linalg.eigvals(S)
    return np.max(np.real(eigvals))

# Spin range
a_vals = np.linspace(0.0, 0.99, 100)

# Choose discrete radii
r_values = [2.2, 3.0, 5.0, 10.0]

plt.figure()

for r in r_values:
    lam = []
    for a in a_vals:
        r_plus = M + np.sqrt(M**2 - a**2)

        # Skip if radius is inside horizon
        if r <= r_plus:
            lam.append(np.nan)
        else:
            lam.append(lambda_max(r, a))

    plt.plot(a_vals, lam, label=f"r = {r}")

plt.axhline(0, linestyle='--')

plt.xlabel("a/M")
plt.ylabel(r"$\lambda_{\max}$")
plt.legend()
#plt.title("Spin dependence of dynamo growth rate at fixed radii")

plt.show()
