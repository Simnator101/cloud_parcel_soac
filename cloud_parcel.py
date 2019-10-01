#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:40:54 2019

Cloud parcel module class
"""

import numpy as np
import sounding as snd

# Constants
gval = 9.81 # m / s^2
Rideal = 8.31446261815324	# J / k / mol
Rair = 287.058  # J / k / kg
cpa = 1004.0    # J / kg / K
cpv = 716.0     # J / kg / K

def trial_lapse_rate(z):
    if 0.0 <= z < 100.0:
        return -20.0e-3
    elif 100.0 <= z < 800.0:
        return -9.8e-3
    else:
        return 20e-3


class CloudParcel(object):
    def __init__(self, T0=300.0, z0=0.0, w0=0.0):
        self.__t0 = T0
        self.__w0 = w0
        self.__z0 = z0
        self.induced_mass = 0.5
        self.storage = None
        
        
    def run(self, dt, NT, environ):
        assert(environ is not None)
        assert(NT > 0)
        assert(dt > 0.0)
        T = np.full(NT, self.__t0)
        w = np.full(NT, self.__w0)
        z = np.full(NT, self.__z0)
        acc = gval / (1 + self.induced_mass)
        
        for i in range(1, NT):
            # Predictor
            Te = environ.sample(z[i-1])
            Buoy = acc * (T[i-1] - Te) / Te
            wstar = w[i-1] + dt * Buoy
            Tstar = T[i-1] + dt * (-gval / cpa * w[i-1])
            #zstar = z[i-1] + dt * w[i-1] + 0.5 * Buoy * dt * dt
            zstar = z[i-1] + dt * w[i-1]
            
            # Corrector
            Te = environ.sample(zstar)
            Buoy = acc * (Tstar - Te) / Te
            w[i] = w[i-1] + dt * Buoy
            T[i] = T[i-1] + dt * (-gval / cpa * wstar)
            #z[i] = z[i-1] + dt * wstar + 0.5 * Buoy * dt * dt
            z[i] = z[i-1] + dt * wstar
        self.storage = (T, w, z)
        return T, w, z
            
if __name__ == "__main__":
    snd = snd.Sounding(None, None)
    snd.from_lapse_rate(trial_lapse_rate, 0, 2e3, 2000)
    
    parcel = CloudParcel(T0 = 301.)
    T, w, z = parcel.run(1.0, 10000, snd)
    plt.plot(w)
    plt.show()
