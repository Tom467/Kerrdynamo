import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Model parameters
# -----------------------

M = 1.0
alpha = 0.1
beta = 1.0
q = 1.5

# spin grid
spins = np.linspace(0,0.99,40)

# radial grid
r = np.linspace(1.01,10,4000)

def Omega(r):
    return beta*r**(-q)

def Omega_prime(r):
    return -q*beta*r**(-q-1)

def lambda_max(r,a):

    Delta = r**2 - 2*M*r + a**2

    A = -alpha - alpha*r*(r*Delta - r**2*(r-M))/(r**2*Delta)

    B = -(Delta/r**2)*(r - M*a**2/r**2)*Omega(r)

    C = Omega_prime(r) + (r - M*a**2/r**2)/(r**2 + a**2 + 2*M*a**2/r)*Omega(r)

    D = -alpha*r*(r - M*a**2/r**2)/(r**2 + a**2 + 2*M*a**2/r)

    trace = A + D
    det = A*D - B*C

    disc = np.sqrt(trace**2 - 4*det)

    lam1 = 0.5*(trace + disc)
    lam2 = 0.5*(trace - disc)

    return np.maximum(lam1,lam2)

# store dynamo radii
r_dyn = []

for a in spins:

    r_plus = M + np.sqrt(M**2 - a**2)

    r_grid = r[r > r_plus + 0.02]

    lam = lambda_max(r_grid,a)

    # find first positive eigenvalue
    idx = np.where(lam>0)[0]

    if len(idx)>0:
        r_dyn.append(r_grid[idx[0]])
    else:
        r_dyn.append(np.nan)

# convert to array
r_dyn = np.array(r_dyn)

# -----------------------
# Plot
# -----------------------

plt.figure(figsize=(7,5))

plt.plot(spins,r_dyn,label="Dynamo boundary")

# horizon radius
r_plus_curve = M + np.sqrt(M**2 - spins**2)
plt.plot(spins,r_plus_curve,'--',label="Event horizon")

plt.xlabel("Spin ")
plt.ylabel("Inner dynamo radius")

#plt.title("Inner dynamo radius vs black hole spin")

plt.legend()
plt.tight_layout()

plt.show()
