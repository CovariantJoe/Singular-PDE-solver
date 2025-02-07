"""
Created on Thu Dec 21 10:03:04 2023

@author: CovariantJoe

General solver of the non-homogeneous partial differential equation:
    Alpha(r) U_tt + Beta(r) U_rr + Gamma(r) U_r + Delta(r) U = P(r)
    
Implemented and tested only for equations of type:
    -r^2/F(r) U_tt + r^2 F(r) U_rr + r^2 (2F(r)/r + dF/dr) U_r + L(L+1) U = P(r)
    
As long as the Fourier transform of boundary conditions (r = r_0, t) exists. Uses scipy.solve_ivp back-end.
The solver uses Frobenius method to bypass singular points in the "r" coordinate, even when the solution to the PDE is discontinuous.

Limitations: See GeneralPDE.Solve()

Constructor:
F -> F(r) sympy expression
P -> P(r) sympy expression
L -> L positive integer (suggested < 4, scipy can't reliably handle more)
boundaryCond -> [ [frequencies],[amplitudes],[derivative's amplitudes] ] ( Fourier transform of U(r=r_0,t), dU/dr (r=r_0,t) )
analytic -> defaults to False. Expects "file.json" containing several lines, where each line must be a string function (e.g: "1 + t^2"), that corresponds to the analytic solution to the problem evaluated at a certain radius. You must make sure the radii correspond to the same ones you are using when calling the solver. 

"""
from sympy import symbols, simplify, diff, sympify, lambdify, Eq, solve
import numpy as np
class GeneralPDE:
    r_symbol = symbols('r')
    def __init__(self, F, P, L, analytic = False):
        testF = str(type(lambdify(self.r_symbol,F)(1.123)))
        testP = str(type(lambdify(self.r_symbol,P,modules=np.lib.scimath)(1.123)))
        if L not in [1,2,3]:
            print("Warning: for physical meaning L should be a positive integer and for numerical stability less than 4. See documentation")
        elif "sympy" not in str(type(F)) or "sympy" not in str(type(P)):
            try:
                diff(F,self.r_symbol)
                diff(P,self.r_symbol)
            except:
                raise Exception("F and P should be either constants or sympy expressions")
        elif testF not in ["<class 'int'>","<class 'float'>","<class 'numpy.float64'>", "<class 'numpy.complex128'>","<class 'complex'>"] or testP not in ["<class 'int'>","<class 'float'>","<class 'numpy.float64'>", "<class 'numpy.complex128'>","<class 'complex'>"]:
            raise Exception("Couldn't evaluate F(r) and/or P(r) to a value, you either used a variable different to r, or have another unkown")    

        self.F = F
        self.P = P
        self.alpha = -self.r_symbol**2/F
        self.beta = self.r_symbol**2*F
        self.gamma = self.r_symbol**2 * (2*F/self.r_symbol + diff(F,self.r_symbol))
        self.L = L
        
        if analytic != False:
            def sol_analytic(filename):
                '''
                Basic checks, then loads exact solution to a lambda function of time that generates a matrix when called
                '''
                if "str" not in str(type(filename)) or "json" not in filename:
                    raise Exception("The argument 'analytic' is a name such as 'file.json' containing the exact solution as several functions of time saved as strings (you get this after evaluating U(r,t) at some radii you'll use)")
                import json
                file_path = filename 
                with open(file_path, 'r') as file:
                    solucion_schw = json.load(file)
                    solucion_schw = np.array(solucion_schw,dtype='str')
                    
                sol_an = sympify(solucion_schw)
                sol_an = lambdify(symbols('t'),sol_an,"numpy")
                return sol_an
            self.analytic = sol_analytic(analytic)
        else:
            self.analytic = np.nan
        
    def FourierInv(self,r,t, sol,freq):
        '''
        Exact inverse Fourier transform in column axis. Assumes entry is sum of Dirac deltas (e.g: 2*Dirac(pi) + 4*Dirac(2*pi))
        '''
        freq_delta = np.array(freq).reshape(len(freq),1)
        try:
            transform = np.zeros([len(r),len(t)],dtype='complex128')
        except MemoryError:
            t = np.linspace(t[0],t[-1],10000)
            print("Time exceeds memory, will use a 10,000 length time")
            transform = np.zeros([len(r),len(t)],dtype='complex128')
        t2 = t.reshape(1,len(t))
        for row in range(np.shape(sol)[0]):
            coefs = sol[row,:]
            transform[row,:] = np.dot(np.exp(1j*np.matmul(freq_delta,t2)).T,coefs)*1/(2*np.pi)
        return transform
    
    def PDE(self):
        '''
        Method to print the partial differential equation that is being solved, as simplified as possible.
        '''
        print("The PDE is: ")
        print(str(simplify(self.alpha)) + " U_tt + " + str(simplify(self.beta)) + " U_rr + " + "(" + str(simplify(self.gamma)) + ")" + " U_r + " +str(self.L*(self.L+1))+"U = " + str(self.P))

    
    def Solve(self, r, t, boundaryCond, method, max_step = np.inf,Homogeneous_Cond = np.nan):
        '''
        Main method of the class, used to solve the Fourier transformation of the equation
        Parameters: 
        r, t: Mesh where the solution will be returned .
        method: Method of scipy.solve_ivp, not every one is supported for this type of problem.
        max_step: Max_step of scipy.solve_ivp, defaults to np.inf (chosen by the solver).
        homogeniousCond: required to use the Frobenius method (solving with singularities), it is the following: the Fourier transform of the boundary condition (WITH FREQUENCY = 0) to the same problem when the source is zero (that is, taking P(r) = 0)
        
        Returns: An array with: 
        sol_pde: A complex128 matrix of the numerical solution to the PDE in the mesh, after calling FourierInv() on sol_ode. 
        sol_ode: An array containing the solution to the Fourier transform of the problem, one column for each frequency in the Fourier transform.
        sol_deriv: An array containing the derivative of the solution to the Fourier transform of the problem, one column for each frequency.

        Known limitations:
        - Scipy.solve_ivp becomes inaccurate for large L ( >= 4)
        - Those in Frobenius()
        '''
        if len(boundaryCond) != 3:
            raise Exception("Boundary condition doesn't have the right size, it should be an array with 3 arrays [frequencies, amplitudes, derivative's amplitudes]. See documentation")
        elif np.size(boundaryCond[1]) != np.size(boundaryCond[2]) or np.size(boundaryCond[0]) != np.size(boundaryCond[1]) or np.size(boundaryCond[0]) != np.size(boundaryCond[2]):
            raise Exception("Inconsistent boundary conditions, should have the same number of frequencies as amplitudes.")
            
        alf = lambdify(self.r_symbol, self.alpha,"numpy")
        bet = lambdify(self.r_symbol, self.beta,"numpy")
        gam = lambdify(self.r_symbol, self.gamma,"numpy")
        p = lambdify(self.r_symbol, self.P,modules = np.lib.scimath)
        L = self.L
        freq = boundaryCond[0]
        u0 = boundaryCond[1]
        u0p = boundaryCond[2]
        Dirac = lambda x: 1 if x==0 else 0
        
        def ODE(r,y0):
            # Function to pass to Scipy.solve_ivp
            u0, u0p = y0
            y = [u0p, frec**2 * alf(r)/bet(r)*u0 -gam(r)/bet(r)*u0p -(L**2+L)/bet(r)*u0 +2*np.pi*p(r)/bet(r)*Dirac(frec)*Dirac(Homogenea)]
            return y
         
        from sympy import Function, Sum,simplify,factorial
        from scipy.integrate import quad
            
        def Frobenius(self,N,w_val,l_val,x0,A,B,A_h = np.nan,B_h = np.nan):    
            '''
            Method to solve numerically beyond regular singular points. 
            Very strongly suggested to understand the Frobenius method for equal and different roots to understand this function.

            Logic:
            RK can't solve beyond singular points in the ODE, so to accomplish this the logic is the following: 
            -RK solves up to near singularity's left and returns. 
            -This code attempts to find an exact solution y1 to the homogeneous ODE, a Frobenius series (key point: the series converges near both sides of the singularity).
            -This code attempts to find a second, linearly independent Frobenius solution y2 from the first one.
            -This code constructs the general solution C1*y1 + C2*y2. The constants are calibrated using the solution and solution's derivative returned by RK in the vicinity of the singularity (A and B).
            -To solve the non-homogeneous problem, the code uses variation of parameters. This requires integrating P*y1/W and P*y2/W where P is the source and W is the Wronskian across the singularity (may not converge). I
             It can be shown that the limits of this integral used here give the right solution, the code integrates from -x0 to x0.
            -The code evaluates C1*y1 + C2*y2 + Integrals and the first derivative of this at the right of the singularity to use as a new initial condition in the RK method.
            
            Parameters:
            N: Number of terms to use in the Frobenius series, more ensure better convergence but significantly impact speed.
            w_val: The frequency in the Fourier transform that is being solved at this moment.
            l_val: The value of L in the equation.
            x0: The distance from the singularity to the point where RK stops at its left (negative) . Strongly suggested |x0| < 1e-5, or else use a very high N, but no guarantee.
            A: The solution to the ODE returned by RK evaluated at r = r_singularity + x0.
            B: The derivative of the solution to the ODE returned by RK evaluated at r = r_singularity + x0.
            A_h: The solution to the homogeneous version of the ODE returned by RK evaluated at r = r_singularity + x0. Required only when solving for w_val = 0 to construct a solution to the non-homogeneous problem.
            B_h: The derivative of the solution to the homogeneous version of the ODE returned by RK evaluated at r = r_singularity + x0. Required only when solving for w_val = 0 to construct a solution to the non-homogeneous problem.

            Returns, an array with:
            sol: The solution to the non-homogeneous ODE evaluated at -x0 (the other side of the singularity) to use as initial condition.
            sol_deriv: The derivative of the solution to the non-homogeneous ODE evaluated at -x0 (the other side of the singularity) to use as initial condition.

            Known limitations:
            - Strange inconsistency when doing the linear combination of 2 independent Frobenius solutions. 
            Sometimes you need to take the complex conjugate of both Frobenius solutions and sometimes you don't, even for the same PDE with different conditions.
            the code handles this empirically according to several tests with known solutions with different F(r).
            '''
            from sympy import LT,expand
            if w_val == 0 and (np.isnan(A_h) or np.isnan(B_h)):
                raise Exception("The solution at singularity's left is required with and without source (P(r)) for w_val = 0")

            #factorize F(r) to explicitly remove regular singularities 
            num, den = simplify(self.F).as_numer_denom()
            coef = LT(expand(num)).as_coeff_mul()[1][0]
            pre_exp = 1
            for j in range(len(singularidad)):
                pre_exp *= (self.r_symbol-singularidad[j])
                
            x = symbols('x')
            F = simplify(pre_exp / den)                
            F = simplify(self.F).subs(simplify(self.F).as_numer_denom()[0],pre_exp)
            
            if ( np.abs( lambdify(self.r_symbol,self.F)(0.5) - lambdify(self.r_symbol,F)(0.5) )>1e-3 ):
                F = -F*coef
            
            F = F.subs(self.r_symbol,x+sing)


            def Taylors(N,l_val,w_val):
                '''
                The ODE is assumed in form x**2*y(x)'' + x*P(x)y(x)' + Q(x)y(x), where P and Q are analytical functions at x = 0 (the singularity).
                This function finds the coefficients of the Taylor series that approximates P(x) and Q(x) near x = 0 (the singularity)
                This means the ODE has to be regular-singular from the Frobenius perspective.  

                Parameters: N, l_val, w_val as defined above.
                Returns: Two length N complex128 arrays with the first N coefficients of the Taylor series of P(x) and Q(x).

                Warning: this function is very slow and may be changed to Cython later.
                '''
                w,l = symbols('w l')
                P = x*self.gamma.subs(self.r_symbol,x+sing)/(F*(x+sing)**2)
                Q = simplify(w**2*x**2/F**2 + x**2*(self.L*(self.L+1))/(F*(x+sing)**2))
                P_vals = np.zeros(N)
                Q_vals = np.zeros(N)
                diffnP = diff(P,x,0)
                diffnQ = diff(Q,x,0)
                for j in range(1,N+1):
                    P_vals[j-1] = 1/factorial(j-1)*diffnP.subs({l:l_val,w:w_val,x:0})
                    Q_vals[j-1] = 1/factorial(j-1)*diffnQ.subs({l:l_val,w:w_val,x:0})
                    if j != N:
                        diffnP = diff(diffnP,x,1)
                        diffnQ = diff(diffnQ,x,1)
                return np.complex128(P_vals), np.complex128(Q_vals)

            def r():
                '''
                Finds roots of Frobenius indicial equation. Degenerate roots are supported, roots differing by an integer are not.
                '''
                rs = solve(x**2-x+x*(P_vals[0])+Q_vals[0])
                if len(rs) == 1:
                    rs.append(rs[0])
                return np.complex128([round(rs[0],13),round(rs[1],13)])
                
            def Series(r,N, equal = False):
                '''
                Calculates the Frobenius series coefficients with N terms corresponding to the ODE. The general recursive relation for the coefficients was found by hand.
                Parameters: r, A Frobenius root from r(). N, the number of terms. equal, a boolean expressing whether the roots are repeated. 
                Returns: An array with the Frobenius coefficients An. If Frobenius roots are equal returns an array with Bn too, the coefficients for the second independent solution.
                '''
                a0 = 1
                coefficients = {a(0): a0}
                An = np.zeros(N,dtype="complex128"); An[0] = a0
                Fi = lambda r: r*(r-1) + P_vals[0]*r + Q_vals[0] 
                for i in range(1, N+1):
                    coefficients[P(i)] = P_vals[i]
                    coefficients[Q(i)] = Q_vals[i]
                    relation = Eq(a(n), -Sum(a(k)*((r+k)*P(n-k) + Q(n-k))/(Fi(r+n)), (k, 0, n-1))) # recursive relation for Frobenius coefficients
                    expr = relation.rhs.subs(n, i)
                    expr = expr.doit().subs(coefficients)
                    value = expr.evalf()
                    coefficients[a(i)] = value
                    An[i-1] = np.complex128(value)
                if not equal: 
                    return An
                else:
                    b = Function("b")
                    Bn = np.zeros(N,dtype="complex128")
                    coefficients[Q(0)] = Q_vals[0] 
                    coefficients[a(0)] = a0
                    for i in range(1,N+1):
                        relation = Eq(b(n),(-Sum( b(k)*(k*P(n-k) + Q(n-k)) + a(k)*P(n-k),(k,1,n-1) ) -P(n)*a(0) -a(n)*(P_vals[0] +2*n-1))/(n*P_vals[0]+Q_vals[0]+n*(n-1)) ) # recursive relation
                        expr = relation.rhs.subs(n,i)
                        expr = expr.doit().subs(coefficients)
                        value = expr.evalf()
                        coefficients[b(i)] = value
                        Bn[i-1] = np.complex128(value)
                    return An,Bn

            def x_potencia(x,N, power = 0):
                '''
                Function to perform x^j for j = 1,2..N.
                It is also used to evaluate x^(complex power) keeping a consistent branch
                Returns: If x is scalar returns an array with x^j (x^1, x^2 ... x^N). If x is a vector, a matrix composed of the results for the scalar case.
                '''
                if power != 0:
                # Evaluate x^power where power is complex
                    return np.exp(power*( np.log(np.abs(x)) + np.angle(x)*1j) )
                
                if np.size(x) == 1:
                    xp = np.zeros(N,dtype="complex128")
                    for j in range(1,N+1):
                        if True: #x > 0:
                            xp[j-1] = x**j
                        else:
                            xp[j-1] = np.exp(1j*np.pi*j)*(-x)**j
                    return xp
                else:
                    xp = np.zeros([len(x),N])
                    for i in range(len(x)):
                        for j in range(1,N+1):
                            if True: #x[i] > 0:
                                xp[i,j-1] = x[i]**j
                            else:
                                xp[i,j-1] = np.exp(1j*np.pi*j)*(-x[i])**j
                    return xp
                
            def Sumatoria(coefs,x,N):
                '''
                Function to perform the sum: coefs[0]*x**0 + coefs[1]*x**1 + ... coefs[N]*x**N by calling x_potencia() and its dot product with coefs.
                Returns: A scalar or a vector with this calculation according to the shape of x. 
                '''
                xp = x_potencia(x,N)
                if np.size(x) == 1:
                    return np.dot(coefs,xp)
                else:
                    suma = np.zeros(len(x),dtype="complex128")
                    for j in range(len(x)):
                        suma[j] = np.dot(coefs,xp[j])
                    return suma
                
            def Solution(x,r1,r2,coefficients,coefficients2 = 0,sign=1):
                '''
                Function to evaluate a Frobenius solution to the ODE at the point x.
                Parameters: x, the point to evaluate. r1 & r2, the roots from r(). coefficients & coefficients2, Frobenius series coeficients, the second is only needed if r1 = r2. sign, this changes depending on the singularity (solving from a region with F > 0 to F < 0 or viceversa). 
                Returns: Either an array with the Frobenius solution and its derivative evaluated at x or two arrays with that (that is, a second independent solution, for the case r1 = r2)
                Assumes: That if r1 and r2 are closer than 1e-13 then the roots are the same.
                '''

                if float(int(np.abs(r1-r2))) == round(np.abs(r1-r2),13) and np.abs(r1-r2) != 0:
                    raise Exception("The Frobenius indicial equation roots differ by an integer, this case is not supported. This depends only on the form of F(r).")
                elif np.abs(r1-r2) > 1e-13:
                    y = x_potencia(x,0,r1) * (1 + Sumatoria(coefficients,x,N))
                    yp = x_potencia(x,0,r1-1)*(r1*(1 + Sumatoria(coefficients,x,N)) + Sumatoria(np.arange(1,N+1,1)*coefficients,x,N) )
                elif np.abs(r1-r2) < 1e-13 and np.abs(r1) < 1e-13:
                    y1 = (1+Sumatoria(coefficients,x,N))
                    y1p = (Sumatoria(np.arange(1,N+1,1)*coefficients,x,N))/x
                    y2 =  Sumatoria(coefficients2,x,N) + y1*np.log(sign*x+0j)
                    y2p = Sumatoria(np.arange(1,N+1,1)*coefficients2,x,N)/x + y1p*np.log(sign*x+0j) + (y1)/x
                    return [y1,y1p],[y2,y2p]
                else: 
                    raise Exception ("The roots of the indicial equation are equal but not to zero, this is not implemented. This depends only on the form of F(r).")
                return [y,yp]
            
            n, k = symbols('n k')
            a = Function('a') # place holder
            P = Function('P'); Q = Function('Q') # Will become mathematical expressions for P and Q
            P_vals, Q_vals = Taylors(N+1,l_val,w_val) # Taylor expansion of P and Q near x = 0
            sign = np.sign(F.subs(x,-x0)) # Changes the second solution depending on the sign of F near the singularity
            r1,r2 = r()
            
            # Both independent solutions y1,y2 to the ODE are constructed according to whether or not the roots are equal (to zero) or they are different to each other.
            if np.abs(r1-r2) > 1e-13:
                coefs_r1 = Series(r1,N)
                coefs_r2 = Series(r2,N)
                y1 = lambda x: Solution(x,r1,r2,coefs_r1)
                y2 = lambda x: Solution(x,r2,r1,coefs_r2)
            else:
                coefs_r1, coefs_r2 = Series(r1,N,equal = True)
                Call = lambda x: Solution(x,r1,r2,coefs_r1,coefs_r2,sign)
                y1 = lambda x: Call(x)[0]
                y2 = lambda x: Call(x)[1]
            
            # Find the constants in: y(x) = C1*y1(x) + C2*y2(x) by requiring y(x) to be the known Runge Kuta solution at x = x0.
            # Some times you need to use the conjugate of y1, y2 (C1, C2) and sometimes you don't (C3, C4), even for the exact same PDE with different initial conditions. This is still mysterious, but the choice below works for every case tested.
            C1,C2 = np.linalg.solve(np.matrix([[np.conjugate(y1(x0)[0]),np.conjugate(y2(x0)[0])],[np.conjugate(y1(x0)[1]),np.conjugate(y2(x0)[1])]]),[A,B])
            C3,C4 = np.linalg.solve(np.matrix([[y1(x0)[0],y2(x0)[0]],[y1(x0)[1],y2(x0)[1]]]),[A,B])
            
            # Decision to use C1,C2 or C3,C4. See just above.
            if w_val != 0:   
                valA = np.abs(np.angle(A) - np.angle(C3*y1(x0)[0]*np.exp(np.pi*1j*r1) + C4*y2(x0)[0]*np.exp(np.pi*1j*r2) )); valB = np.abs( 2*np.angle(A) - np.angle(C3*y1(-x0)[0] + C4*y2(-x0)[0]) - np.angle(C3*y1(x0)[0]*np.exp(np.pi*1j*r1) + C4*y2(x0)[0]*np.exp(np.pi*1j*r2) )  )                
                if (np.abs(C1/C3-1) < 1e-2 and np.abs(np.angle(C1/C3)) < 1e-5) or (np.abs(C1/C4-1) < 1e-2 and np.abs(np.angle(C1/C4)) < 1e-5):
                    sol = lambda x: np.complex128(C1*np.conjugate(y1(x)[0]) + C2*np.conjugate(y2(x)[0]))
                    sol_deriv = lambda x: np.complex128(C1*np.conjugate(y1(x)[1]) + C2*np.conjugate(y2(x)[1]))
                elif  valA < 1e-2 or np.abs(valA-2*np.pi) < 1e-3 or np.abs(valA - valB) < 1e-2:
                    sol = lambda x: np.complex128(C3*y1(x)[0] + C4*y2(x)[0])
                    sol_deriv = lambda x: np.complex128(C3*y1(x)[1] + C4*y2(x)[1])
        
                elif  valB < 1e-3 or np.abs(valB-2*np.pi) < 1e-3:
                    sol = lambda x: np.complex128(C3*y1(-x)[0]*np.exp(np.pi*1j*r1) + C4*y2(-x)[0]*np.exp(np.pi*1j*r2))
                    sol_deriv = lambda x: np.complex128(-C3*y1(-x)[1]*np.exp(np.pi*1j*r1) - C4*y2(-x)[1]*np.exp(np.pi*1j*r2))
                else:
                    raise Exception("The way to combine Frobenius solutions could not be determined")
            
            # This part of the code calculates the integrals in the variation of parameters method to construct the solution to the non-homogeneous problem from the Frobenius ones which are only for homogeneous ODEs.    
            # The integrals are solved numerically integrating from x0 to -x0, which includes the singular point x = 0. Scipy.quad can handle this unless the value printed on screen isn't small.    
            # Notice how the source term has P(x)/(F(x)*r**2) instead of P(x), this is because we got the Frobenius solutions from x**2*y(x)'' + x*P(x)y(x)' + Q(x)y(x), after dividing everything by the original factor in y''.
            else:
                yp =  A_h - A; yp_deriv =  B_h - B # The particular solution and its derivative are the solution to the non-homogeneous minus the solution to the homogeneous problem at the same point (x0)
                P0 = self.P.subs(self.r_symbol,x+sing)
                P = lambdify(x,2*np.pi*P0/(F*(x+sing)**2),modules = np.lib.scimath) # Source term
                I_s =np.matmul(np.linalg.inv( np.matrix([ [y1(x0)[0],y2(x0)[0]], [y1(x0)[1],y2(x0)[1]] ],dtype="complex128") ),[yp,yp_deriv]) # The value of the integrals at x0, found from known things without integrating
                Integrand1 = lambda x: (x*y1(x)[0]*P(x) )/( y1(x)[0]**2 + Sumatoria(np.arange(1,N+1,1)*coefs_r2,x,N)*y1(x)[0] - y1(x)[1]*x*Sumatoria(coefs_r2,x,N) ) # The denominator here is the Wronskian simplified by hand
                Integrand2 = lambda x: (x*y2(x)[0]*P(x) )/( y1(x)[0]**2 + Sumatoria(np.arange(1,N+1,1)*coefs_r2,x,N)*y1(x)[0] - y1(x)[1]*x*Sumatoria(coefs_r2,x,N) ) # The denominator here is the Wronskian simplified by hand
                I1_n = quad(Integrand1, x0, -x0,points=0,complex_func=True)
                I2_n = quad(Integrand2, x0, -x0,points=0,complex_func=True)
                print("Scipy.integrate.quad convergence value. Singularity bypassing needs both to be very small: \n" + str(I1_n[1])+ str(I2_n[1]))
                I1_n = I1_n[0]
                I2_n = I2_n[0]
                # Construct the solution. sol = homogeneous solution (Frobenius linear combination) + Variation of parameters solution
                # Example if singularity is at r = 2. The integral at 2.01 (and thus the solution at 2.01) is found from adding the integral at 1.99 + the integral from 1.99 to 2.01. Both quantities found above.
                sol = lambda x: np.complex128(C1*y1(x)[0] + C2*y2(x)[0]) + np.complex128( y2(x)[0]*(I_s[0,1]+I1_n) + y1(x)[0]*(I_s[0,0]-I2_n) )
                sol_deriv = lambda x: np.complex128(C1*y1(x)[1] + C2*y2(x)[1]) + np.complex128( y2(x)[1]*(I_s[0,1]+I1_n) + y1(x)[1]*(I_s[0,0]-I2_n) )
            # Return solution at -x0 to restart Runge Kuta beyond the singular point.
            return [sol(-x0),sol_deriv(-x0)]
       

        # Actual execution within the Solve() method begins:
        from scipy.integrate import solve_ivp
        sol_ode = np.zeros([len(r),len(u0)],dtype='complex128')
        Homogenea = 0
        singularidad = np.real(np.complex128(solve(self.F,self.r_symbol)))
        singularidad = np.sort(singularidad)
        domain_sing = []
        for s in singularidad:
            if s > r[0] and s < r[-1]:
                domain_sing.append(s)
        if len(domain_sing) == 0:
            print("No singularities in the domain, proceeding")
        
        sol_deriv = []
        for i in range(len(freq)):
            frec = freq[i]
            if len(freq) > 1:
                print("Solving for the frequency: ",frec)
            
            if np.isnan(Homogeneous_Cond).any():
                raise Exception("To bypass the singularity you need the initial conditions to the problem removing the source term P(r).")
            cond0 = [u0[i],u0p[i]]
            section_solution = []
            
            if len(domain_sing) == 0:
                try:
                    rk_sol = solve_ivp(ODE,[r[0],r[-1]],[cond0[0],cond0[1]],method=method,dense_output=True,max_step=max_step,atol=1e-12,rtol=1e-8)
                    section_solution.append(rk_sol.sol(r)[0,:])
                    sol_deriv.append(rk_sol.sol(r[-1])[1])
                except ValueError:
                    raise Exception("No singularities were found, but scipy.solve_ivp couldn't return correctly, maybe the roots (singularities) in F(r) are failing to be detected")
            else:
                for count in range(0,len(domain_sing)+1):
                    if count != len(domain_sing):
                        sing = domain_sing[count]
                        rsolve = r[np.logical_and(r<sing, r>domain_sing[count-1])] if count !=0 else r[r<sing] 
                        rmin = 1e-7
                        r1 = sing - rmin
                        if frec != 0:
                            Homogenea = 1
                            rk_sol = solve_ivp(ODE,[rsolve[0],rsolve[-1]],[cond0[0],cond0[1]],method=method,dense_output=True,max_step=max_step,atol=1e-12,rtol=1e-8) if r1 < rsolve[-1] else solve_ivp(ODE,[rsolve[0],r1],[cond0[0],cond0[1]],method=method, dense_output=True,max_step=max_step,atol=1e-12,rtol=1e-8)
                            cond0 = Frobenius(self,2*self.L+3,w_val = frec, l_val = self.L,x0 =-rmin,A= rk_sol.sol(r1)[0],B = rk_sol.sol(r1)[1])
                        else:
                            Homogenea = 0
                            rk_sol = solve_ivp(ODE,[rsolve[0],rsolve[-1]],[cond0[0],cond0[1]],method=method,dense_output=True,max_step=max_step,atol=1e-12,rtol=1e-8) if r1 < rsolve[-1] else solve_ivp(ODE,[rsolve[0],r1],[cond0[0],cond0[1]],method=method, dense_output=True,max_step=max_step,atol=1e-12,rtol=1e-8)      
                            cond0 = Frobenius(self,2*self.L+3,w_val = frec, l_val = self.L,x0 =-rmin,A = 0, B = 0, A_h=rk_sol.sol(r1)[0] ,B_h=rk_sol.sol(r1)[1] )             

                        section_solution.append(rk_sol.sol(rsolve)[0,:])
                        #sol_deriv.append(rk_sol.sol(rsolve[-1])[1])
                    else:
                        r0 = np.max([r1,rsolve[-1]])
                        rsolve = r[np.logical_and(r>sing,r<domain_sing[count+1])] if count <= len(domain_sing)-2 else r[r>sing]
                        if len(rsolve) == 0:
                            continue

                        # Initial condition in the next section = what Frobenius() returned
                        rk_sol = solve_ivp(ODE,[2*sing-r0,rsolve[-1]],[cond0[0],cond0[1]],method=method,dense_output=True,max_step=max_step)
                        section_solution.append(rk_sol.sol(rsolve)[0,:])
                        sol_deriv.append(rk_sol.sol(rsolve[-1])[1])
              
            sol_ode[:,i] = np.hstack(section_solution)
        sol_pde = self.FourierInv(r,t, sol_ode,freq)
        return sol_pde, sol_ode, sol_deriv
