"""
@author: Covariant Joe

The purpose of this code is to show the process to obtain the results in the article. To understand the solver see Example.py

UNLESS DISABLING THE ACCURACY THIS CODE TAKES HOURS, IT IS IMPORTANT TO RUN THIS IN A MACHINE WITH ABOUT 12 GB OF RAM !
Google Colab allows more RAM, but takes significantly longer to run this
"""

from SingularitySolver import GeneralPDE
from sympy import sqrt,diff,Abs,simplify, solve
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols

# Function to solve in parallel for each frequency. Higher RAM consumption but good accuracy much faster
def Solve_frequency(Problem, mesh, Conditions, index, step):
    Conditions = [ [Conditions[0][index]], [Conditions[1][index]], [Conditions[2][index]] ]
    return Problem.Solve(mesh[0], mesh[1], Conditions, method = "DOP853", max_step = step, Homogeneous_Cond = [0,0] )[1:]  # Returns the sol. to the ODE with that frequency rather than solution to the PDE

if __name__ == "__main__":
    # Constants and definitions
    code = ""
    G = 6.67384e-11; c = 299792458
    I = 1j # Just for comfort
    r = symbols('r')
    l = 0.001 # Hayward metric parameter in meters.
    M = (1.3466837364048547e+27)*r**2/(r**2+l**2) # Mass function of radius in kg, approx Jupiter mass asymptotically
    Lambda = 1/l
    
    L = 1 # Angular index in PDE. Either 1, 2 or 3
    F = simplify( 1- (2*(G*M/c**2)*r**2)/(r**3 + 2*(G*M/c**2)*l**2) ) # F(r) with mass as function M(r)
    P = Lambda*(sqrt(F))*(r**2*diff(M)/M + 4*r) # Back-reaction souce term after multiplying by r^2
    
    Problem = GeneralPDE(F,P,L)
    # Print the problem being solved
    Problem.PDE()

    # Mesh. The range from r = 0.0001 to r = 7 is divided in 5 sections in order to save ram and have different steps for each section
    t_vec = np.linspace(0,11,220)
    Mesh = [ [np.linspace(l/10,1,100),t_vec], [np.linspace(1,2+1e-7,100),t_vec], [np.linspace(2+1e-7,2.7,70),t_vec], [np.linspace(2.7,5,230),t_vec], [np.linspace(5,6.8,180),t_vec] ]
    
    # step for each section of the Mesh
    steps_0 = [7e-7,3e-7,7e-7,5e-6,5e-6] # For the equation with frequency = 0  
    steps = [3.5e-7,2e-7,4.5e-7,5e-6,5e-6] # For other frequencies

# -----------------------------------------------------------------------------------------------------------
    # Uncomment this is you just want to run it fast without any accuracy, the solution is visually similar
    #steps = 5*[np.inf]; steps_0 = steps; code = "No accuracy"
# -----------------------------------------------------------------------------------------------------------
    
    # Boundary conditions, found in Maple when solving the PDE for the F(r) defined above expanded near r = 0:
    Conditions = []
    frequencies = [0, np.sqrt(0.5),-np.sqrt(0.5)]
    if L == 1:
        Conditions.append([frequencies,[-518.5874848 + 212.4653996j,820.7150414-344.2142232j,344.2142232+820.7150414j],[5439187.13+5874029.808j,499417.0096+12674398.638j,-12674398.638+499417.0096j]])
    elif L == 2:        
        Conditions.append([frequencies,[-41.4548606+483.8615546j, 68.21215724+57.33190192j,-57.33190192+68.21215724j],[11891088.754-1390666.9456j,-1722760.3912+1361934.1002j,-1361934.1002-1722760.3912j]])
    elif L == 3:
        Conditions.append([frequencies,[13.466021594-90.13851528j,-71.50012314-53.2336142j,53.2336142-71.50012314j],[-3170252.9039999996-8225.654014j,2190091.98-2201474.6459999997j,2201474.6459999997+2190091.98j]])
        
    Solution = []

    # Solve in parallel for each section of the mesh
    for i in range(0,len(Mesh)):
        len_sect = len(Mesh[i][0])
        with multiprocessing.Pool(processes=len(frequencies)) as pool:
            print("Solving in interval r = " + str(Mesh[i][0][0]) + " to " + str(Mesh[i][0][-1]) )
            process1 = pool.apply_async(Solve_frequency, (Problem, Mesh[i], Conditions[i], 0, steps_0[i]))
            process2 = pool.apply_async(Solve_frequency, (Problem, Mesh[i], Conditions[i], 1, steps[i]))
            process3 = pool.apply_async(Solve_frequency, (Problem, Mesh[i], Conditions[i], 2, steps[i]))
            
            result1 = process1.get()
            result2 = process2.get()
            result3 = process3.get()
            
            # Save temporary solution in case execution is halted. First array is the solution everywhere, second is derivative evaluated just at the end of the section
            if code != "No accuracy":
                np.save("L"+str(L)+"_Freq0_section"+str(i)+".npy",np.array([result1[0][:,0], result1[1]*len(result1[0][:,0])]) )
                np.save("L"+str(L)+"_Freq1_section"+str(i)+".npy",np.array([result2[0][:,0], result2[1]*len(result2[0][:,0])]) )
                np.save("L"+str(L)+"_Freq2_section"+str(i)+".npy",np.array([result3[0][:,0], result3[1]*len(result3[0][:,0])]) )
           
            
        result = np.array([result1[0][:,0][:-1],result2[0][:,0][:-1],result3[0][:,0][:-1] ]).T.reshape([len_sect-1,3])
        # Solution at the end of section is initial condition for the next one
        Conditions.append([ frequencies , [result[-1,0],result[-1,1],result[-1,2]], [result1[1][0],result2[1][0],result3[1][0]] ])
        # Inverse fourier to convert the solution for each frequency to solution to the whole PDE in this section of the mesh
        Solution.append( Problem.FourierInv(Mesh[i][0][:-1], Mesh[i][1], result ,frequencies ) )
        
    Solution = np.vstack([ Solution[0], Solution[1], Solution[2], Solution[3], Solution[4] ])
    r_vec = np.hstack([Mesh[0][0][:-1], Mesh[1][0][:-1], Mesh[2][0][:-1], Mesh[3][0][:-1], Mesh[4][0][:-1]])
   

'''

Some code just to plot

'''
Index_begin_plot = 199 # Plots after singularity
Index_end_plot = -1
R,T=np.meshgrid(r_vec[Index_begin_plot:Index_end_plot],t_vec[:])
Solution_to_plot = np.abs(Solution[Index_begin_plot:Index_end_plot,:].T)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(R, T, Solution_to_plot, edgecolor='black', lw=0.5, rstride=8, cstride=8,
alpha=0.3)

ax.contourf(R, T, Solution_to_plot, zdir='z', offset=np.min(Solution_to_plot[:,:])*0.9, cmap='coolwarm',levels=10)
ax.set(xlim=(r_vec[Index_begin_plot], r_vec[Index_end_plot]*1.05), ylim=(-1, 11),zlim=(np.min(Solution_to_plot[:,:])*0.9,np.max(Solution_to_plot[:,:])*1.01),
xlabel='r', ylabel='t', zlabel='|R|')
ax.set_zlabel('|R|',fontsize=12)
ax.set_ylabel('t',fontsize=12)
ax.set_xlabel('r',fontsize=12)
ax.set_title(f"L = {L} {code}",fontsize=14)
plt.show()

