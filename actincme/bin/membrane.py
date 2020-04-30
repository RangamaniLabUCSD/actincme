import logging
import numpy as np
# import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Membrane(object):

    def __init__(self, x, k=320, R0=20,  g=20, c0=0.02, a0=8, lam0=0.002, aOut=200, aIn=100):
        """
        Membrane object is fully defined given
        x -> mesh points (default: (0:1/2000:1)^2)*200
        k -> bending rigidity (default: 320 pNnm)
        R0 -> non-dimensionalization length (default: 20nm)
        g -> smoothing parameter (default: 20)
        c0 -> spontaneous curvature (default: 0.02)
        a0 -> area of coat (default: 8)
        lam0 -> membrane tension (default: 0.002 pN/nm)
        aOut -> area of upward applied force (default: 200)
        aIn -> area of inward applied force (default: 100)
        BVPs (Boundary Value Problems) that this class solves for are:
        1. c0, a0 -> shape
            fun_no_f, bc_no_f
        2. c0, a0, f -> shape
            fun, bc 
        """
        self.x = x
        self.g = g
        self.k = k
        self.R0 = R0
        self.c0 = c0
        self.a0 = a0
        self.lam0 = lam0
        self.aOut = aOut
        self.aIn = aIn

        self._aF = self.a0
        self._disp = 0

    def init(self, x):
        """
        Initial condition for the membrane shape 
        """ 
        ds = 1e-4
        return np.vstack((ds + np.sqrt(2*x), \
                          np.zeros(len(x)), \
                          np.zeros(len(x)), \
                          np.zeros(len(x)), \
                          np.zeros(len(x)), \
                          (self.lam0 * self.R0**2/self.k) * np.ones(len(x)), \
                          ))

    def fun_no_f(self, x, y):
        """
        System of equations to solve for membrane shape given spontaneous curvature
        and area of coat, no additional parameters
        """
        b = 1
        ds = 1e-4
        c = 0.5*(self.c0*self.R0)*(1 - np.tanh(self.g*(x - self.a0)))
        dc = 0.5*(self.c0*self.R0)*self.g*(np.tanh(self.g*(x - self.a0))**2 - 1)
        
        return np.vstack((np.cos(y[2])/(y[0] ), \
                          np.sin(y[2])/(y[0]), \
                          (2*y[0]*y[3] - np.sin(y[2]))/(y[0] )**2, \
                          y[4]/(y[0] )**2 + dc, \
                          2*y[3]*((y[3] - c)**2 + y[5]/ b) - 2*(y[3] - c)*(y[3]**2 + (y[3] - np.sin(y[2])/(y[0] ))**2), \
                          2*b*(y[3] - c)* dc))

    def bc_no_f(self, ya, yb):
        """
        Boundary condition for fun_no_f
        """
        ds  =1e-4
        return np.array([ya[0] - ds, yb[1], ya[2], yb[2], ya[4], yb[5] - self.lam0 * self.R0**2/self.k])
    
    def fun(self, x, y, p):
        """
        System of equations to solve for membrane shape given spontaneous curvature
        and area of coat. Applied force is an additional parameter that is estimated via an
        additional boundary condition
        """

        b = 1
        ds = 1e-4
        c = 0.5*(self.c0*self.R0)*(1 - np.tanh(self.g*(x - self.a0)))
        dc = 0.5*(self.c0*self.R0)*self.g*(np.tanh(self.g*(x - self.a0))**2 - 1)
        #applied force

        fbar = p*(0.5*((1 - np.tanh(self.g*(x - self._aF)))/self._aF - (np.tanh(self.g*(x - self.aIn)) - np.tanh(self.g*(x - self.aOut)))/(self.aOut-self.aIn)))
        
        return np.vstack((np.cos(y[2])/(y[0] ), \
                          np.sin(y[2])/(y[0]), \
                          (2*y[0]*y[3] - np.sin(y[2]))/(y[0] )**2, \
                          y[4]/(y[0] )**2 + dc, \
                          fbar*np.cos(y[2]) + 2*y[3]*((y[3] - c)**2 + y[5]/ b) - 2*(y[3] - c)*(y[3]**2 + (y[3] - np.sin(y[2])/(y[0] ))**2), \
                          2*b*(y[3] - c)* dc - fbar*np.sin(y[2])/y[0]))

    def bc(self, ya, yb, p):
        """
        Boundary condition for fun
        """
        ds  =1e-4
        return np.array([ya[0] - ds,  ya[1] - self._disp/self.R0, yb[1], ya[2], yb[2], ya[4], yb[5] - self.lam0 * self.R0**2/ self.k])
    
    def compute(self, x, initial = None, p_guess = None,  type = 'fun_1'):
        """
        MAIN COMPUTE CALL
        type == None calls extra parameter function
        type == 'c' or 'a' calls normal function
        """
        if initial is None:
            initial = self.init(x)
        if p_guess is None:
            p_guess = np.array([1])
    
        if type == "fun_1":
            sol = solve_bvp(self.fun, self.bc, x,  initial, p = p_guess, tol = 1e-5, max_nodes = 10000, verbose = 1)
            print(sol.success, sol.status)
        elif type == "fun_2":
            sol = solve_bvp(self.fun_no_f, self.bc_no_f, x,  initial, max_nodes = 10000, verbose = 1, tol = 1e-5)
            print(sol.success, sol.status)

        return sol

    def loop_curvature_c(self, x, rng, initial = None):
        """
        Looping through a range of c0 for fixed a0
        """
        loop_sol = []
        for c in rng:
            self.c0 = c
            sol = self.compute(x, initial, type = 'fun_2')

            if sol.success == False:
                return loop_sol
                raise Exception('Failed')

            if len(loop_sol) == 0:
                loop_sol = sol.sol(x)
            else:
                loop_sol = np.vstack((loop_sol, sol.sol(x)))

            initial = sol.sol(x)
        return loop_sol  

    def loop_curvature_a(self, x, rng, initial = None):
        """
        Looping through a range of a0 for fixed c0
        """
        loop_sol = []
        for a in rng:
            self.a0 = a

            sol = self.compute(x, initial, type = 'c')

            if sol.success == False:
                return loop_sol
                raise Exception('Failed')

            if len(loop_sol) == 0:
                loop_sol = sol.sol(x)
            else:
                loop_sol = np.vstack((loop_sol, sol.sol(x)))
             
            initial = sol.sol(x)
        return loop_sol  

    def loop_force(self, x, number_of_iterations, Z_spacing, initial, p_guess = None): 
        loop_sol, p_sol = [], []

        for j, i in enumerate(range(number_of_iterations)):

            if j == 0:
                self._disp = initial[1,0]*self.R0
                print(self._disp)
            else:
                self._disp = initial[1,0]*self.R0 - j* Z_spacing

            sol = self.compute(x, initial, p_guess)

            if sol.success == False:
                return loop_sol, p_sol
                raise Exception('Failed')

            if len(loop_sol) == 0:
                loop_sol = sol.sol(x)
            else:
                loop_sol = np.vstack((loop_sol, sol.sol(x))) 
            initial = sol.sol(x)
            p_guess = sol.p
            p_sol.append(sol.p * (self.k/self.R0**3) * 2* np.pi * self.R0**2)
        return loop_sol, p_sol

    def loop_area_of_force(self, x, initial, range_aF, p_guess = None): 
        loop_sol, p_sol = [], []

        for j, i in enumerate(range(len(range_aF))):


            self._disp = initial[1,0]*self.R0

            self._aF = range_aF[j]

            sol = self.compute(x, initial, p_guess)

            if sol.success == False:
                return loop_sol, p_sol
                raise Exception('Failed')

            if len(loop_sol) == 0:
                loop_sol = sol.sol(x)
            else:
                loop_sol = np.vstack((loop_sol, sol.sol(x))) 
            initial = sol.sol(x)
            p_guess = sol.p
            p_sol.append(sol.p * (self.k/self.R0**3) * 2* np.pi * self.R0**2)
        return loop_sol, p_sol

    
    def plot(self, ax, x,  sol):

        x_plot = self.R0*sol[0]
        y_plot = self.R0 * sol[1]
        ax.plot(x_plot, y_plot)
        ax.set_xlabel("x")
        ax.set_ylabel("y")