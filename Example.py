'''
@author: Covariant Joe

This code provides an example about how to use the module SingularitySolver with a solution for the Schwarzschild metric
'''

# Imports
from SingularitySolver import GeneralPDE
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols

# Read Example.mw in Maple for explanation about how to produce the exact solution and find boundary conditions analytically

# Constants
I = 1j # Just for comfort
r = symbols('r')

# Define F(r) and P(r) as sympy expressions of the symbol 'r'

F = 1-2/r
P = 80*r**2

# Define L (angular index) as a positive integer, less than 4 for accuracy
L = 1

# Initialize an instance of the Partial Differencial Equation object:
Problem = GeneralPDE(F,P,L, analytic="Example.json") # Analytic is optional

# Print the problem being solved
Problem.PDE()

# Define the mesh
r0 = 0.001; dr = 1e-2
r_vec = np.arange(r0,5,1e-2)
t_vec = np.linspace(0,11,220)

# Define the boundary conditions using the following structure:
# Conditions = [ [array with frequencies (must be real positive)] , [array with the amplitude at each corresponding frequency (may be complex)] , [array with the amplitude of the derivative at each corresponding frequency (may be complex)] ]
# that is, you need to know the Fourier transforms of: U(r = r_0,t) and dU/dr (r = r_0, t).
# Important: If your problem can't be solved analitically, you may still find these conditions from approximating F(r) near r = r_0 which may allow you to get an exact solution with Maple, then evaluating at r_0. This is the case with Hayward near r_0 = 0 and was a major point in the article.

Conditions = [ [0,0.7071067811865476,-0.7071067811865476] , [11063.21645 + 374.6554219*I,-21979.07530 + 3597.530826*I,-21979.07530 + 3597.530826*I] , [-1.826652030*10**6 - 62574.12938*I,3.663051444*10**6 - 617938.0833*I,3.663051444*10**6 - 617938.0833*I] ]

# Solve
# The third argument is the name of Scipy's runge kuta method (see Scipy doc), "DOP853" is an 8th order explicit solver which is the best usually. Some Scipy methods don't work with complex boundary conditions.
# max_step is the argument for Scipy's solver (see Scipy doc), default is np.inf (chosen by the solver)
# Homogeneous_Cond is required to use the Frobrenius method (solving with singularities), it is the following: the Fourier transform of the boundary condition (WITH FREQUENCY = 0) to the same problem when the source is zero (that is, taking P(r) = 0) 
# In all my testing with Maple, this is usually 0 for both U and dU/dr (thus [0,0])

Solution = Problem.Solve(r_vec,t_vec,Conditions, method = "DOP853", max_step = np.inf, Homogeneous_Cond=[0,0])[1:]
# The first element is what we are interested in, the other values returned are useful for finding which frequencies are introducing error in your solution, or to solve in parallel the whole problem

# Access the exact solution for plotting, need to pass a time vector where the sol. will be evaluated
Analytic = Problem.analytic(t_vec)
error = np.abs(Analytic)/np.abs(Solution) # Matrix with error, make sure Analytic was produced at the same values of r_vec.
print("Finished. Max error: " + str(round(100*np.abs(1-np.max(error)),3)) + " %")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# A bunch of code just to plot. Change the value below to the position of your singularity
Singularity = 2
Index = np.argmin(np.abs(r_vec - Singularity))
T,R = np.meshgrid(t_vec,r_vec[:Index-1])
T2,R2 = np.meshgrid(t_vec,r_vec[Index+1:])

fig, ax = plt.subplots(2,2)
ax[0][0].contourf(R,T,np.abs(Analytic[:Index-1,:]))
ax[0][0].set_title("Exact solution before singularity")
ax[0][0].set_xlabel("r")
ax[0][0].set_ylabel("t")

ax[0][1].contourf(R,T,np.abs(Solution[:Index-1,:]))
ax[0][1].set_title("Numerical solution before singularity")
ax[0][1].set_xlabel("r")
ax[0][1].set_ylabel("t")

ax[1][0].contourf(R2,T2,np.abs(Analytic[Index+1:,:]))
ax[1][0].set_title("Exact solution after singularity")
ax[1][0].set_xlim([Singularity,r_vec[-1]])
ax[1][0].set_xlabel("r")
ax[1][0].set_ylabel("t")

ax[1][1].contourf(R2,T2,np.abs(Analytic[Index+1:,:]))
ax[1][1].set_title("Numerical solution after singularity")
ax[1][1].set_xlim([Singularity,r_vec[-1]])
ax[1][1].set_xlabel("r")
ax[1][1].set_ylabel("t")

fig.subplots_adjust(
top=0.93,
bottom=0.11,
left=0.105,
right=0.92,
hspace=0.43,
wspace=0.325)
