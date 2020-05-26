# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize


class BoundaryValueProblem:
    """
    Boundary value problem

    solve a system of first order differential equations

    u(x)' = f(u,x)

    f      -> function w/ derivative f(u,x,*f_args)
    f_args -> list of extra arguments for f
    x      -> array w/ grid points
    s      -> array w/ information on boundary condition location
              if s =  1, b.c. is at x[0]
              if s = -1, b.c. is at x[-1]
    u0     -> value of function u at boundaries
    order  -> should be 2 (default) or 4, to set the order of the method
    """

    def __init__(self, f, x, s, u0, Guess, order=2):

        self.__FirstCall = True

        if callable(f):
            self.f = f
        else:
            raise TypeError("f must be of the type f = f(u,x)")

        self.order = order
        self.s = s
        self.Guess = Guess

        # make sure x is a numpy array instead of a list
        if not type(x) == np.ndarray:
            self.x = np.array(x)
        else:
            self.x = x

        # same for boundary conditions array
        if not type(u0) == np.ndarray:
            self.u0 = np.array(u0)
        else:
            self.u0 = u0

        # number of equations and number of grid points
        self.NEQ = len(u0)
        self.NGRID = len(x)

        self.shape = (self.NEQ, self.NGRID)

        # residual function
        self.F = np.zeros((self.NEQ, self.NGRID))

        # check which order of the method should be used
        if self.order == 2:
            self.D = self.diff2
        elif self.order == 4:
            self.D = self.diff4
        else:
            err = "The order of the method should be 2 or 4"
            raise NotImplementedError(err)

        # step of finite differences, Um sobre
        self.US = 1./(self.x[1]-self.x[0])

        # indexes for B.V.
        self.bvp_i = [0 if xx == 1 else -1 for xx in self.s]
        self.IndexZero = np.where(np.asarray(self.bvp_i) == 0)[0].tolist()
        self.IndexMinusOne = np.where(np.asarray(self.bvp_i) == -1)[0].tolist()

        # finite differences coefficient
        self.CoefsCenter2 = [-1./2., 1./2.]
        self.CoefsCenter4 = [1./12., -2./3., 2./3., -1./12.]
        self.CoefsForward2 = [-3./2., 2., -1./2.]
        self.CoefsForward4 = [-25./12., 4., -3., 4./3., -1./4.]

        return

    def residual(self, u):
        """
        Calculate de residual function w/ b.c. imposed
        """

        if self.__FirstCall:
            self.diff = np.zeros_like(u)
            self.__FirstCall = False

        # calculate function to be integrated
        Derivs = self.f(u, self.x)

        # calculate derivatives
        self.D(u)

        # impose boundary condition
        self.F[self.IndexZero, 0] = (u[self.IndexZero, 0]
                                     - self.u0[self.IndexZero])

        self.F[self.IndexMinusOne, -1] = (u[self.IndexMinusOne, -1]
                                          - self.u0[self.IndexMinusOne])

        self.F[self.IndexZero, 1:] = (self.diff[self.IndexZero, 1:]
                                      - Derivs[self.IndexZero, 1:])

        self.F[self.IndexMinusOne, :-1] = (self.diff[self.IndexMinusOne, :-1]
                                           - Derivs[self.IndexMinusOne, :-1])

        return self.F

    def Loss(self, u):
        return 0.5 * np.linalg.norm(self.residual(u.reshape(self.shape)))**2

    def Solver(self, u):
        return self.residual(u.reshape(self.shape)).reshape(
            (self.NEQ*self.NGRID))

    def diff2(self, u):
        """
        Calculate second order finite differences method for
        first derivative of u
        """

        self.diff[:, 0] = self.US * (self.CoefsForward2[0] * u[:, 0]
                                     + self.CoefsForward2[1] * u[:, 1]
                                     + self.CoefsForward2[2] * u[:, 2])

        self.diff[:, -1] = - self.US * (self.CoefsForward2[0] * u[:, -1]
                                        + self.CoefsForward2[1] * u[:, -2]
                                        + self.CoefsForward2[2] * u[:, -3])

        self.diff[:, 1:-1] = self.US * (self.CoefsCenter2[0] * u[:, :-2]
                                        + self.CoefsCenter2[1] * u[:, 2:])

        return self.diff

    def diff4(self, u):
        """
        Calculate fourth order finite differences method for
        first derivative of u
        """

        self.diff[:, 0] = self.US * (self.CoefsForward4[0] * u[:, 0]
                                     + self.CoefsForward4[1] * u[:, 1]
                                     + self.CoefsForward4[2] * u[:, 2]
                                     + self.CoefsForward4[3] * u[:, 3]
                                     + self.CoefsForward4[4] * u[:, 4])

        self.diff[:, -1] = - self.US * (self.CoefsForward4[0] * u[:, -1]
                                        + self.CoefsForward4[1] * u[:, -2]
                                        + self.CoefsForward4[2] * u[:, -3]
                                        + self.CoefsForward4[3] * u[:, -4]
                                        + self.CoefsForward4[4] * u[:, -5])

        # second order for second grid points
        self.diff[:, 1] = self.US * (self.CoefsCenter2[0] * u[:, 0]
                                     + self.CoefsCenter2[1] * u[:, 2])

        self.diff[:, -2] = self.US * (self.CoefsCenter2[0] * u[:, -3]
                                      + self.CoefsCenter2[1] * u[:, -1])

        # fourth order for everyone else
        self.diff[:, 2:-2] = self.US * (self.CoefsCenter4[0] * u[:, :-4]
                                        + self.CoefsCenter4[1] * u[:, 1:-3]
                                        + self.CoefsCenter4[2] * u[:, 3:-1]
                                        + self.CoefsCenter4[3] * u[:, 4:])

        return self.diff

    def Solve(self):
        """
        Solve(f,f_args)
        """

        # guess = self.Guess.reshape((self.NEQ*self.NGRID))
        # BH = basinhopping(self.Loss,guess,disp=True,niter=10,T=0.00001,
        # niter_success=3,stepsize=0.001)
        # print BH.message
        # print "F=",BH.fun
        # return newton_krylov(self.residual,BH.x.reshape(self.shape),
        # f_tol=1.0e-6,line_search='wolfe',verbose=True)

        # return newton_krylov(self.residual,self.Guess,f_tol=1.0e-4,
        # line_search='wolfe',verbose=True)

        # Different alternatives, this one was the most efficient
        # guess = self.Guess.reshape((self.NEQ*self.NGRID))
        # MIN = minimize(self.Loss,guess,options={"disp":True})
        # return newton_krylov(self.residual,MIN.x.reshape(self.shape),
        # f_tol=1.0e-4,line_search='wolfe',verbose=True)

        guess = self.Guess.reshape((self.NEQ*self.NGRID))
        MIN = minimize(self.Loss, guess, options={"disp": False})
        return MIN.x.reshape(self.shape)
