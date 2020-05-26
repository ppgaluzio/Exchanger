"""Definition of class PlateExchanger based on the work of Gut 2003

All standard values are taken from Gut 2003

"""

import numpy as np
import matplotlib.pyplot as pl
import Exchanger.BoundaryValueProblem as bvp


class PlateExchanger:
    """
    Classe que modela um trocador de calor do tipo placas planas

    Plate and Frame Heat Exchanger parameters
    -----------------------------------------

    N_C : # of channels
    P_I : # of passes at side I
    P_II: # of passes at side II
    phi : feed connection relative location (1,2,3,4)
    Y_h : hot fluid location (1 or 0)
    Y_f : type of flow in channels (1 or 0)

    """

    def __init__(self, N_C=36, P_I=2, P_II=2, phi=3, Y_h=1,
                 Y_f=1, NGRID=100, verbose=False):

        self.vp = print if verbose else lambda *a, **k: None

        self.Count = 0

        # reset calling variables
        self.__reset__()

        # number of grid points for integration
        self.NGRID = NGRID

        # Number of channels
        if isinstance(N_C, int):
            self.N_C = N_C
        else:
            raise TypeError("N_C should be int")

        self.vp("\nIn PLATE HEAT EXCHANGER\n")
        self.vp("Modelling a parallel plate heat exchanger based on Gut 2003")
        self.vp("Number of channels:", self.N_C)

        # define array w/ temperature function
        self.T = np.ones((self.N_C, self.NGRID))
        self.UnNormalizedT = np.zeros_like(self.T)

        # Fluid density
        self.rho = np.zeros_like(self.T)

        # fluid viscosity
        self.mu = np.zeros_like(self.T)

        # fluid specific heat at constant pressure
        self.C_p = np.zeros_like(self.T)

        # fluid thermal conductivity
        self.k = np.zeros_like(self.T)

        # convective heat transfer coefficient
        self.h = np.zeros_like(self.T)

        # Prandtl number
        self.Pr = np.zeros_like(self.T)

        # Reynolds number
        self.Re = np.zeros_like(self.T)

        # Nusselt number (Nu = h De / k)
        self.Nu = np.zeros_like(self.T)

        # Fluid mass flow rate
        self.WW = np.zeros_like(self.T)

        # Channel mass velocity
        self.Gc = np.zeros(self.N_C)

        # Boundary conditions temperature
        self.T_BC = np.zeros(self.N_C)

        # define array w/ grid points
        self.x = np.linspace(0, 1, self.NGRID)

        # number of passes at side I
        if isinstance(P_I, int) and P_I < N_C:
            self.P_I = P_I
        else:
            raise TypeError("P_I should be int and smaller than N_C")

        # number of passes at side II
        if isinstance(P_II, int) and P_II < N_C:
            self.P_II = P_II
        else:
            raise TypeError("P_II should be int and smaller than N_C")

        # define number of channels and passes
        # NC_# --> number of channels at side # (I or II)
        # N_# --> number of channels per pass at side # (I or II)

        if self.N_C & 1:       # odd

            self.NC_I = (self.N_C + 1)/2
            self.NC_II = (self.N_C - 1)/2

            self.N_I = (self.N_C + 1)/(2*self.P_I)
            self.N_II = (self.N_C - 1)/(2*self.P_II)

        else:                  # even

            self.NC_I = self.N_C / 2
            self.NC_II = self.N_C / 2

            self.N_I = self.N_C//(2*self.P_I)
            self.N_II = self.N_C//(2*self.P_II)

        self.vp("\nNumber of channels per pass I:", self.N_I)
        self.vp("Total number of channels I:", self.NC_I)
        self.vp("\nNumber of channels per pass II:", self.N_II)
        self.vp("Total number of channels II:", self.NC_II)

        def factors(N):
            return [i for i in range(1, N+1) if N % i == 0]

        # check if the values are allowed

        if not (self.NC_I % self.P_I == 0):
            err = (f"Remainder of NC_I / P_I = {self.NC_I % self.P_I}\n"
                   f"Allowed values for P_I : {factors(self.NC_I)}\n"
                   "P_I is not a factor of NC_I")
            raise ValueError(err)

        if not (self.NC_II % self.P_II == 0):
            err = (f"Remainder of NC_II / P_II = {self.NC_II % self.P_II}\n"
                   f"Allowed values for P_II : {factors(self.NC_II)}\n"
                   "P_II is not a factor of NC_II")
            raise ValueError(err)

        if not self.N_C == (self.NC_I + self.NC_II):
            raise ValueError("NC should be equal to NC_I + NC_II")

        # feed connection relative location
        if phi == 1 or phi == 2 or phi == 3 or phi == 4:
            self.phi = phi
        else:
            raise ValueError("phi must be 1, 2, 3 or 4")

        # even index
        self.SideI = range(0, self.N_C, 2)

        # odd index
        self.SideII = range(1, self.N_C, 2)

        # Hot fluid location
        if Y_h == 1:
            self.Y_h = 1        # Hot fluid at side I
            self.hi = self.SideI
            self.ci = self.SideII
        elif Y_h == 0:
            self.Y_h = 0        # hot fluid at side II
            self.hi = self.SideII
            self.ci = self.SideI
        else:
            raise ValueError("Y_h must be either 0 or 1")

        # type of flow in channels
        if Y_f == 1:
            self.Y_f = 1        # Flow diagonal in all channels
        elif Y_f == 0:
            self.Y_f = 0        # Flow vertical in all channels
        else:
            raise ValueError("Y_f must be either 0 or 1")

        return

    def __reset__(self):
        # initialization varibles
        self.__calledDefineGeometry = False
        self.__CalledDefineHotFluid = False
        self.__CalledDefineColdFluid = False
        self.__CalledDefineHeatTransferCoef = False
        self.__CalledDetermineS = False
        self.__CalledDefineHeatTransferCoef = False
        self.__CalledDerivative = False

        return

    def DefineGeometry(self, L=0.74, w=0.236, b=2.7e-3, D_p=0.059,
                       epss_p=0.7e-3, Phi=1.17, k_p=17, verbose=False):
        """
        Define geometrical factors of the PHE

        L      : effective plate length        (m)
        w      : effective plate width         (m)
        b      : channel average thickness     (m)
        D_p    : port diameter of plate        (m)
        epss_p : thickness of metal plate      (m)
        Phi    : plate area enlargement factor (m)
        k_p    : plate thermal conductivity    (W/m ºC)
        """

        self.__calledDefineGeometry = True

        self.L = L
        self.w = w
        self.b = b
        self.D_p = D_p
        self.epss_p = epss_p
        self.Phi = Phi
        self.k_p = k_p

        self.Area = self.Phi * self.L * self.w
        self.De = 2 * self.b / self.Phi

        if not self.__CalledDetermineS:
            self.DetermineS()

        if verbose:
            print("\nDefine Geometry:\n")
            print("L      = ", self.L, " m")
            print("w      = ", self.w, " m")
            print("b      = ", self.b, " m")
            print("D_p    = ", self.D_p, " m")
            print("epss_p = ", self.epss_p, " m")
            print("k_p    = ", self.k_p, " W/m ºC")
            print("Phi    = ", self.Phi, " m")

        return

    def DefineHotFluid(self, T_inh=35, W_h=1.30, R_fh=8.6e-5,
                       rho_h=1286, mu_h=5.15e-2, C_ph=2803, k_h=0.407,
                       a_h=[0.400, 0.598, 0.33, 0.000, 18.29, 0.652],
                       verbose=False):
        """
        Define Hot Fluid Parameters

        All h indexes refer to hot fluid

        T_inh : inlet temperature                        (ºC)
        W_h   : fluid mass flow rate                     (kg/s)
        R_fh  : Fluid fowling factor                     (m^2 ºC/W)
        rho_h : fluid density                            (kg/m^3)
        mu_h  : fluid viscosity                          (Pa s)
        C_ph  : fluid specific heat at constant pressure (J/kg ºC)
        k_h   : fluid thermal conductivity               (W/m ºC)
        a_h   : general model parameter

        """

        self.__CalledDefineHotFluid = True
        self.__HotFluidUpdate = False

        self.T_inh = T_inh
        self.W_h = W_h
        self.R_fh = R_fh

        if self.Y_h == 0:
            self.Gc[self.hi] = self.W_h / (self.N_I * self.b * self.w)
            self.WW[self.hi, :] = self.W_h / self.N_I

        else:
            self.Gc[self.hi] = self.W_h / (self.N_II * self.b * self.w)
            self.WW[self.hi, :] = self.W_h / self.N_II

        # test wether the arguments are functions or numbers
        if callable(rho_h):
            self.__HotFluidUpdate = True
            self.rho_h_f = rho_h
            self.rho[self.hi, :] = self.rho_h_f(self.T[self.hi, :])
        else:
            self.rho[self.hi, :] = rho_h

        if callable(mu_h):
            self.__HotFluidUpdate = True
            self.mu_h_f = mu_h
            self.mu[self.hi, :] = self.mu_h_f(self.T[self.hi, :])

            mu_average = np.average(  # noqa:F841
                self.mu_h_f(np.linspace(0, 1, 100)))
        else:
            self.mu[self.hi, :] = mu_h

        if callable(C_ph):
            self.__HotFluidUpdate = True
            self.C_ph_f = C_ph
            self.C_p[self.hi, :] = self.C_ph_f(self.T[self.hi, :])

            C_p_average = np.average(  # noqa:F841
                self.C_ph_f(np.linspace(0, 1, 100)))
        else:
            self.C_p[self.hi, :] = C_ph

        if callable(k_h):
            self.__HotFluidUpdate = True
            self.k_h_f = k_h
            self.k[self.hi, :] = self.k_h_f(self.T[self.hi, :])

            k_average = np.average(  # noqa:F841
                self.k_h_f(np.linspace(0, 1, 100)))
        else:
            self.k[self.hi, :] = k_h

        # dimension for a
        Na = 6
        try:
            if len(a_h) == Na:
                self.a_h = a_h
            else:
                print("Na = ", Na)
                raise IndexError("a_h must have dimension equal to Na")
        except TypeError:
            print("a_h is neither a numpy array nor a list")
            raise

        # Average prandt number

        if verbose:
            print("\nDefine Hot Fluid Parameters:\n")
            print("T_inh = ", T_inh, "ºC")
            print("W_h   = ", W_h, "kg/s")
            print("R_fh  =", R_fh,  "m^2 ºC/W")
            print("rho_h =", rho_h, "kg/m^3")
            print("mu_h  =", mu_h,  "Pa s")
            print("C_ph  =", C_ph,  "J/kg ºC")
            print("k_h   =", k_h,  "W/m ºC")
            print("a_h   =", a_h)
            print()

        return

    def DefineColdFluid(self, T_inc=1, W_c=1.30, R_fc=8.6e-5,
                        rho_c=1286, mu_c=5.15e-2, C_pc=2803,
                        k_c=0.407, a_c=[0.400, 0.598, 0.33, 0.000,
                                        18.29, 0.652], verbose=False):
        """
        Define Cold Fluid Parameters

        All c indexes refer to cold fluid

        T_inc : inlet temperature                        (ºC)
        W_c   : fluid mass flow rate                     (kg/s)
        R_fc  : Fluid fowling factor                     (m^2 ºC/W)
        rho_c : fluid density                            (kg/m^3)
        mu_c  : fluid viscosity                          (Pa s)
        C_pc  : fluid specific heat at constant pressure (J/kg ºC)
        k_c   : fluid thermal conductivity               (W/m ºC)
        a_c   : general model parameter

        """

        self.__CalledDefineColdFluid = True
        self.__ColdFluidUpdate = False

        self.T_inc = T_inc
        self.W_c = W_c
        self.R_fc = R_fc

        if self.Y_h == 0:
            self.Gc[self.ci] = self.W_c / (self.N_II * self.b * self.w)
            self.WW[self.ci, :] = self.W_c / self.N_I
        else:
            self.Gc[self.ci] = self.W_c / (self.N_I * self.b * self.w)
            self.WW[self.ci, :] = self.W_c / self.N_I

        if callable(rho_c):
            self.__ColdFluidUpdate = True
            self.rho_c_f = rho_c
            self.rho[self.ci, :] = self.rho_c_f(self.T[self.ci, :])

        else:
            self.rho[self.ci, :] = rho_c

        if callable(mu_c):
            self.__ColdFluidUpdate = True
            self.mu_c_f = mu_c
            self.mu[self.ci, :] = self.mu_c_f(self.T[self.ci, :])

        else:
            self.mu[self.ci, :] = mu_c

        if callable(C_pc):
            self.__ColdFluidUpdate = True
            self.C_pc_f = C_pc
            self.C_p[self.ci, :] = self.C_pc_f(self.T[self.ci, :])

        else:
            self.C_p[self.ci, :] = C_pc

        if callable(k_c):
            self.__ColdFluidUpdate = True
            self.k_c_f = k_c
            self.k[self.ci, :] = self.k_c_f(self.T[self.ci, :])

        else:
            self.k[self.ci, :] = k_c

        # dimension for a
        Na = 6
        try:
            if len(a_c) == Na:
                self.a_c = a_c
            else:
                print("Na = ", Na)
                raise IndexError("a_c must have dimension equal to Na")
        except TypeError:
            print("a_c is neither a numpy array nor a list")
            raise

        if verbose:
            print("\nDefine Cold Fluid Parameters:\n")
            print("T_inc = ", T_inc, "ºC")
            print("W_c   = ", W_c, "kg/s")
            print("R_fc  =", R_fc,  "m^2 ºC/W")
            print("rho_c =", rho_c, "kg/m^3")
            print("mu_c  =", mu_c,  "Pa s")
            print("C_pc  =", C_pc,  "J/kg ºC")
            print("k_c   =", k_c,  "W/m ºC")
            print("a_c   =", a_c)

        return

    def UpdateFluids(self):

        self.UnNormalizedT = (self.T_inh-self.T_inc) * self.T + self.T_inc

        if self.__HotFluidUpdate:
            # XXX: verify exception
            self.rho[self.hi, :] = self.rho_h_f(self.UnNormalizedT[self.hi, :])
            self.mu[self.hi, :] = self.mu_h_f(self.UnNormalizedT[self.hi, :])
            self.C_p[self.hi, :] = self.C_ph_f(self.UnNormalizedT[self.hi, :])
            self.k[self.hi, :] = self.k_h_f(self.UnNormalizedT[self.hi, :])

            # try:
            #     self.rho[self.hi, :] = \self.rho_h_f(self.UnNormalizedT[self.hi, :])
            # except:
            #     pass
            # try:
            #     self.mu[self.hi, :] = self.mu_h_f(self.UnNormalizedT[self.hi, :])
            # except:
            #     pass
            # try:
            #     self.C_p[self.hi, :] = self.C_ph_f(self.UnNormalizedT[self.hi, :])
            # except:
            #     pass
            # try:
            #     self.k[self.hi, :] = self.k_h_f(self.UnNormalizedT[self.hi, :])
            # except:
            #     pass

        if self.__ColdFluidUpdate:
            # XXX: verify exception
            self.rho[self.ci, :] = self.rho_c_f(self.UnNormalizedT[self.ci, :])
            self.mu[self.ci, :] = self.mu_c_f(self.UnNormalizedT[self.ci, :])
            self.C_p[self.ci, :] = self.C_pc_f(self.UnNormalizedT[self.ci, :])
            self.k[self.ci, :] = self.k_c_f(self.UnNormalizedT[self.ci, :])

            # try:
            #     self.rho[self.ci, :] = self.rho_c_f(self.UnNormalizedT[self.ci, :])
            # except:
            #     pass
            # try:
            #     self.mu[self.ci, :] = self.mu_c_f(self.UnNormalizedT[self.ci, :])
            # except:
            #     pass
            # try:
            #     self.C_p[self.ci, :] = self.C_pc_f(self.UnNormalizedT[self.ci, :])
            # except:
            #     pass
            # try:
            #     self.k[self.ci, :] = self.k_c_f(self.UnNormalizedT[self.ci, :])
            # except:
            #     pass

        return

    def FluidParameters(self):

        self.Pr = self.C_p * self.mu / self.k

        for i in range(self.N_C):
            self.Re[i, :] = self.Gc[i] * self.De / self.mu[i, :]

        self.Nu[self.hi, :] = (self.a_h[0] * self.Re[self.hi, :]**self.a_h[1]
                               * self.Pr[self.hi, :]**self.a_h[2])
        self.Nu[self.ci, :] = (self.a_c[0] * self.Re[self.ci, :]**self.a_c[1]
                               * self.Pr[self.ci, :]**self.a_c[2])

        self.h = self.Nu * self.k / self.De

        return

    def DefineHeatTransferCoef(self):
        """
        Define the value for the Heat Transfer Coefficient U

        k_p    : plate thermal conductivity
        epss_p : thickness of the plate
        R_h    : Fouling factor for hot stream
        R_c    : Fouling factor for cold stream
        """

        if not self.__calledDefineGeometry:
            self.DefineGeometry(verbose=True)

        if not self.__CalledDefineHotFluid:
            self.DefineHotFluid(verbose=True)

        if not self.__CalledDefineColdFluid:
            self.DefineColdFluid(verbose=True)

        # if first call, create the array
        if not self.__CalledDefineHeatTransferCoef:
            self.U = np.zeros((self.N_C-1, self.NGRID))
            self.__CalledDefineHeatTransferCoef = True

        # calculate fluid parameters
        self.UpdateFluids()
        self.FluidParameters()

        hi = np.array([x**(-1) for x in self.h[:-1, :]])

        hip1 = np.array([x**(-1) for x in self.h[1:, :]])

        self.U = np.power(hi + hip1 + self.epss_p / self.k_p + self.R_fh
                          + self.R_fc, -1)

        return

    def DetermineS(self):
        """
        Determine the value of S, according w/ gut 2003
        """

        self.__CalledDetermineS = True

        self.s = np.zeros_like(self.T)

        for p in range(self.P_I):
            for n in range(self.N_I):
                ii = self.SideI[p*self.N_I:(p+1)*self.N_I]
                self.s[ii, :] = (-1)**p

        # for p in range(self.P_I) :
        #     for n in range(self.N_I) :
        #         i = 2*p*self.N_I + 2*n
        #         self.s[i,:] = (-1)**(p)

        for p in range(self.P_II):
            for n in range(self.N_II):
                if self.phi == 1:
                    ii = self.SideII[p*self.N_II:(p+1)*self.N_II]
                    self.s[ii, :] = (-1)**p
                elif self.phi == 2:
                    ii = self.SideII[p*self.N_II:(p+1)*self.N_II]
                    self.s[ii, :] = (-1)**(p+1)
                elif self.phi == 3:
                    ii = self.SideII[-(p+1)*self.N_II:-(p)*self.N_II or None]
                    self.s[ii, :] = (-1)**p
                elif self.phi == 4:
                    ii = self.SideII[-(p+1)*self.N_II:-(p)*self.N_II or None]
                    self.s[ii, :] = (-1)**(p+1)

        if 0 in self.s:
            print("s = ", self.s)
            raise ValueError("S must be either 1 or -1")

        self.OutletIndex = [-1 if xx == 1 else 0 for xx in self.s[:, 0]]
        self.InletIndex = [0 if xx == 1 else -1 for xx in self.s[:, 0]]

        self.PreviousIndex = []
        self.Previous = [[] for i in range(self.N_C)]
        for p in range(1, self.P_I):
            Previous = self.SideI[(p-1)*self.N_I:p*self.N_I]
            for i in self.SideI[p*self.N_I:(p+1)*self.N_I]:
                self.Previous[i] = Previous
                self.PreviousIndex.append(i)

        if self.phi == 1 or self.phi == 2:
            for p in range(1, self.P_II):
                Previous = self.SideII[(p-1)*self.N_II:p*self.N_II]
                for i in self.SideII[p*self.N_II:(p+1)*self.N_II]:
                    self.Previous[i] = Previous
                    self.PreviousIndex.append(i)

        elif self.phi == 3 or self.phi == 4:
            for p in range(1, self.P_II):
                Previous = self.SideII[-(p)*self.N_II:-(p-1)*self.N_II or None]
                for i in self.SideII[-(p+1)*self.N_II:-(p)*self.N_II or None]:
                    self.Previous[i] = Previous
                    self.PreviousIndex.append(i)

        return

    def BoundaryConditions(self):
        """
        Update boundary conditions

        at a given pass, the inlet temperature corresponds to
        the average outlet temperature of the previous pass
        """

        if not self.__CalledDefineHotFluid:
            self.DefineHotFluid()

        if not self.__CalledDefineColdFluid:
            self.DefineColdFluid()

        if not self.__CalledDetermineS:
            self.DetermineS()

        # define inlet temperatures
        if self.Y_h == 1:       # Hot fluid at side I
            T_in_I = 1.0        # self.T_inh
            T_in_II = 0.0       # self.T_inc
        else:                   # Hot fluid at side II
            T_in_I = 0.0        # self.T_inc
            T_in_II = 1.0       # self.T_inh

        # Obs.:
        # if s = 1, b.c. at x = 0
        # if s = -1, b.c. at x = 1

        #  Side I
        # --------

        for i in self.SideI[:self.N_I]:
            self.T_BC[i] = T_in_I
        if self.phi == 1 or self.phi == 2:

            for i in self.SideII[:self.N_II]:
                self.T_BC[i] = T_in_II
        elif self.phi == 3 or self.phi == 4:
            for i in self.SideII[-self.N_II:]:
                self.T_BC[i] = T_in_II

        for i in self.PreviousIndex:
            Out = [self.OutletIndex[ii] for ii in self.Previous[i]]
            self.T_BC[i] = np.average(self.T[self.Previous[i], Out])

        return

    def Derivative(self, T, x):
        """
        calculate the derivative of the function at position x, w/ T(x)
        """

        if not self.__CalledDerivative:
            self.f = np.zeros_like(T)
            self.DetermineS()
            self.__CalledDerivative = True
            self.DefineHeatTransferCoef()

        self.T = T
        self.BoundaryConditions()

        if self.__ColdFluidUpdate or self.__HotFluidUpdate:
            self.DefineHeatTransferCoef()

        # self.UpdateFluids()
        # self.FluidParameters()
        # self.DefineHeatTransferCoef()

        # first channel
        self.f[0, :] = (self.s[0, :] * self.Area * self.U[0, :]
                        * (self.T[1, :] - self.T[0, :])
                        / (self.WW[0, :] * self.C_p[0, :]))

        # ineer channels
        self.f[1:-1, :] = (self.s[1:-1, :] * self.Area
                           * (self.U[0:-1, :] * (self.T[:-2, :]
                                                 - self.T[1:-1, :])
                              + self.U[1:, :] * (self.T[2:, :]
                                                 - self.T[1:-1, :]))
                           / (self.WW[1:-1, :] * self.C_p[1:-1, :]))

        # last channel
        self.f[-1, :] = (self.s[-1, :] * self.Area * self.U[-1, :]
                         * (self.T[-2, :]-self.T[-1, :])
                         / (self.WW[-1, :] * self.C_p[-1, :]))

        return self.f

    def PlotaTemp(self, Salva=False, Filename="exchanger.png"):
        self.BoundaryConditions()
        fig = pl.figure()
        ax = fig.add_subplot(111)
        for i in self.SideI:
            ax.plot(self.x, self.T_inc+self.T[i, :]*(self.T_inh-self.T_inc),
                    "k-", label=f"{i}")
        for i in self.SideII:
            ax.plot(self.x, self.T_inc+self.T[i, :]*(self.T_inh-self.T_inc),
                    "r-", label=r"{i}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$T\,(^oC)$")
        if Salva:
            fig.savefig(Filename, dpi=300)
        else:
            fig.show()

        return

    def Solve(self):

        # guess
        self.T = self.AnalyticalSolution()

        SOL = bvp.BoundaryValueProblem(self.Derivative, self.x, self.s[:, 0],
                                       self.T_BC, Guess=self.T, order=2)

        self.sol = SOL.Solve()
        return self.sol

    def AnalyticalSolution(self):

        if not self.__CalledDefineHeatTransferCoef:
            self.DefineHeatTransferCoef()

        #  properly define dd

        AverageU = np.average(self.U)
        dd = np.zeros(self.N_C)

        Alpha_hot = 1
        Alpha_cold = 1

        # XXX: here you need to check how to properly implement the
        # alpha coefficients

        dd[self.SideI] = self.s[self.SideI, 0] * Alpha_hot
        # self.Area * AverageU * self.N_I  / \
        # (np.average(self.WW[self.SideI]) * np.average(self.C_p[self.SideI]))

        dd[self.SideII] = self.s[self.SideII, 0] * Alpha_cold
        #  self.Area * AverageU * self.N_II / \
        # (np.average(self.WW[self.SideII]) *
        # np.average(self.C_p[self.SideII]))

        M = np.zeros((self.N_C, self.N_C))

        M[0, 0] = -dd[0]
        M[0, 1] = dd[0]

        for i in range(1, self.N_C-1):
            M[i, i] = -2*dd[i]
            M[i, i-1] = M[i, i+1] = dd[i]

        M[-1, -1] = -dd[-1]
        M[-1, -2] = dd[-1]

        # m is good!!!

        # w eigenvalue
        # v[:,i] corresponding eigenvector

        w, v = np.linalg.eig(M)

        # set boundary conditions
        # define inlet temperatures
        if self.Y_h == 1:       # Hot fluid at side I
            T_in_I = 1.0
            T_in_II = 0.0
        else:                  # Hot fluid at side II
            T_in_I = 0.0
            T_in_II = 1.0

        # vector w/ boundary conditions to implement analytical solution
        B = np.zeros(self.N_C)

        # fluid inlet

        # side I
        for n in range(self.N_I):
            B[n] = T_in_I

        # side II
        if self.phi == 1 or self.phi == 2:
            for n in range(self.N_II):
                ii = 2*n + 1
                B[ii] = T_in_II
        elif self.phi == 3 or self.phi == 4:
            for n in range(self.N_II):
                ii = 2*(self.P_II-1)*self.N_II + 2*n+1
                B[ii] = T_in_II

        # Definition of matrix for analytical solution

        A = np.zeros_like(v)
        v_previous = np.zeros_like(v)

        E = np.asarray([[np.exp(w[i] * (1-self.s[j, 0])/2)
                         for i in range(self.N_C)]
                        for j in range(self.N_C)]).T

        for i in range(self.N_C):
            for j in range(self.N_C):
                if not self.Previous[i]:
                    # list is empty
                    v_previous[i, j] = 0
                else:
                    v_previous[i, j] = np.average(v[self.Previous[i], j])

        # define the matrix in the system to be solved
        A = (v-v_previous)*E.T

        # solve it!!!
        c = np.linalg.solve(A, B)

        guess = np.zeros_like(self.T)

        for i in range(self.N_C):
            guess[i, :] = 0
            for j in range(self.N_C):
                # guess[i,:] += c[j] * v[i,j] * np.exp(w[j]*self.x)
                guess[i, :] = np.add(guess[i, :],
                                     c[j] * v[i, j] * np.exp(w[j]*self.x),
                                     out=guess[i, :],
                                     casting='unsafe')
        self.T = guess
        return guess

    def OutputTemp(self):

        # Side I
        Index = []
        TempI = 0
        for i in range(self.N_I):
            index = 2 * (self.P_I - 1) * self.N_I + 2 * i
            Index.append(index)
            TempI += self.T[index, self.OutletIndex[index]]
        TempI /= self.N_I

        # Side II
        Index = []
        TempII = 0
        for i in range(self.N_II):
            if self.phi == 1 or self.phi == 2:
                index = 2 * (self.P_II - 1) * self.N_II + 2 * i + 1
                Index.append(index)
            elif self.phi == 3 or self.phi == 4:
                index = 2*i + 1
                Index.append(index)
            TempII += self.T[index, self.OutletIndex[index]]

        TempII /= self.N_II

        TempI = self.T_inc + TempI * (self.T_inh-self.T_inc)
        TempII = self.T_inc + TempII*(self.T_inh-self.T_inc)

        if self.Y_h == 1:
            Th = TempI
            Tc = TempII
        else:
            Tc = TempI
            Th = TempII

        return Tc, Th

    def Efficiency(self):
        # W fluid mass flow rate
        # Cpm Fluid specific heat average
        # Theta_in Theta)out

        Tc, Th = self.OutputTemp()

        # Renormalize
        DeltaT = self.T_inh - self.T_inc
        Tc = (Tc - self.T_inc) / DeltaT
        Th = (Th - self.T_inc) / DeltaT

        C_phm = np.average(self.C_p[self.hi, :])
        C_pcm = np.average(self.C_p[self.ci, :])

        Denominator = min(self.W_c * C_pcm, self.W_h * C_phm)

        Eh = self.W_h * C_phm * abs(Th-1) / Denominator
        Ec = self.W_c * C_pcm * abs(Tc) / Denominator

        return np.average([Ec, Eh])  # Ec, Eh

    def PressureDrop(self):

        if self.Y_h == 1:
            Ph = self.P_I
            Pc = self.P_II
        else:
            Pc = self.P_I
            Ph = self.P_II

        g = 9.8

        # Hot Fluid
        Re = np.average(self.Re[self.hi, :])
        f = self.a_h[3] + self.a_h[4] / Re**self.a_h[5]
        Gc = np.average(self.Gc[self.hi])
        Gp = 4 * self.W_h / (np.pi * self.D_p**2)
        RhoM = np.average(self.rho[self.hi, :])

        self.DeltaP_h = (2 * f * (self.L + self.D_p) * Ph * Gc**2
                         / (RhoM * self.De) + 1.4 * (Ph * Gp**2 / (2 * RhoM))
                         + RhoM * g * (self.L + self.D_p))

        # Cold Fluid
        Re = np.average(self.Re[self.ci, :])
        f = self.a_c[3] + self.a_c[4] / Re**self.a_c[5]
        Gc = np.average(self.Gc[self.ci])
        Gp = 4 * self.W_c / (np.pi * self.D_p**2)
        RhoM = np.average(self.rho[self.ci, :])

        self.DeltaP_c = (2 * f * (self.L + self.D_p) * Pc * Gc**2
                         / (RhoM * self.De) + 1.4 * (Pc * Gp**2 / (2 * RhoM))
                         + RhoM * g * (self.L + self.D_p))

        return self.DeltaP_c, self.DeltaP_h
