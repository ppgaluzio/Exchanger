#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
import PlateHeatExchanger as phe
from argparse import ArgumentParser
import pickle

#%% command line arguments

parser = ArgumentParser()
parser.add_argument("-f", "--fig", dest="fig_filename",default="exchanger.png",
                    help="Figure File Name")
parser.add_argument("-o", "--out", dest="out",default="phe_out.obj",
                    help="Filename for final output")


args = parser.parse_args()

#%% define funções dos fluidos

def RhoHot(T) :
    return -1.451e-3 * T**2 - 0.4281 * T + 1296

def MuHot(T) :
    return np.power(10, -4.513 + 421.8 / (T + 108.5))

def CpHot(T) :
    return 4.803 * T + 2696

def KHot(T) :
    return -3.696e-6 * T**2 + 1.201e-3 * T + 0.3825

def RhoCold(T) :
    return 2.080e-5 * T**3 - 6.668e-3 * T**2 + 0.04675 * T + 999.9

def MuCold(T) :
    return np.power( 21.482 * ((T-8.435)+np.sqrt(8078.4+(T-8.435)**2))-1200, -1)

def CpCold(T) :
    return 5.2013e-7 * T**4 - 2.1528e-4 * T**3 + 4.1758e-2 * T**2 - \
        2.6171 * T + 4227.1

def KCold(T) :
    return 0.5691 + T/538 - T**2 / 133333

#%% PASSO 1: Leitura dos dados --> ok

x = phe.PlateExchanger(N_C=4,NGRID=10)
x.DefineGeometry(verbose=True)

x.DefineHotFluid(rho_h=RhoHot, mu_h=MuHot, C_ph=CpHot, k_h=KHot, verbose=True)
x.DefineColdFluid(rho_c=RhoCold, mu_c=MuCold, C_pc=CpCold, k_c=KCold,
                  verbose=True)

#%% PASSO 2: Designio das propriedades do fluído ao lado I e II

x.Solve()
x.PlotaTemp(Salva=True,Filename=args.fig_filename)

Tc, Th = x.OutputTemp()
print "temp c = ",Tc
print "temp h = ",Th

DPC, DPH = x.PressureDrop()

print "DP C = ", DPC
print "DP H = ", DPH

print "Eff = ",x.Efficiency()

FileOut = open(args.out, 'w')
pickle.dump(x, FileOut)

print "DONE"
