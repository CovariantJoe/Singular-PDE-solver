# Wave-like equation solver for singular spacetimes

## Description
This code was used in the article "Back-reaction spacetime waves from a radially varying mass in a Hayward black hole"

General solver of the non-homogeneous Partial Differential Equation with singular points:
![Alt text](General.png)
Implemented and tested only for equations of form:
![Alt text](Particular.png)
    
As long as the Fourier transform of boundary conditions (r = r_0, t) exists. Uses scipy.solve_ivp back-end.
The solver uses Frobenius method to bypass singular points in the "r" coordinate, even when the solution to the PDE is discontinuous.

## Characteristics
- The most important characteristic is that it can solve beyond one or more singular points in the spacelike coordinate that make the solution discontinuous, like black hole event horizons, as long as the equation remains regular (see Frobenius method).
- F(r) is the metric function g_tt, this PDE is obtained by applying separation of variables to the General Relativity D'alembertian. This allows modelling scalar fields in Schwarzschild, Hayward, Reissner-Nördstrom, etc, backgrounds.
- Can read a JSON file containing the exact solution to the problem, if known, in order to facilitate testing. This file can be generated in Maple. Schwarzschild exact solutions can often be found.

## Usage
- Documentation is in the solver file SingularitySolver.py.
- A complete example for the Schwarzschild metric (which has an analytical solution to compare: E. Avila-Vargas, C. Moreno, and R. Hernández-Jiménez, Journal of Mathematical Physics 64 (2023).) is in Example.py. The file Example.json contains this exact solution. Example.mw is a Maple worksheet explaining how to find initial conditions and export the exact solution to JSON so that it is read by the solver.
- Article.py contains the code to replicate the results in our article. This file is computationally expensive.
