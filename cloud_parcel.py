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
        print(self.__t0, self.__w0, self.__z0)
        
        T = np.full(NT, self.__t0)
        w = np.full(NT, self.__w0)
        z = np.full(NT, self.__z0)
        acc = gval / (1 + self.induced_mass)
        
        def wf(T, z):
            return acc * (T - environ.sample(z)) / environ.sample(z)
        
        def Tf(w):
            return -gval / cpa * w
        
        def zf(w):
            return w
        
        for i in range(1, NT):
            # K1
            k1w = dt * wf(T[i-1], z[i-1])
            k1T = dt * Tf(w[i-1])
            k1z = dt * zf(w[i-1])
            # K2
            k2w = dt * wf(T[i-1] + k1T / 2, z[i-1] + k1z / 2)
            k2T = dt * Tf(w[i-1] + k1w / 2)
            k2z = dt * zf(w[i-1] + k1w / 2)
            # K3
            k3w = dt * wf(T[i-1] + k2T / 2, z[i-1] + k2z / 2)
            k3T = dt * Tf(w[i-1] + k2w / 2)
            k3z = dt * zf(w[i-1] + k2w / 2)
            # K4
            k4w = dt * wf(T[i-1] + k3T, z[i-1] + k3z)
            k4T = dt * Tf(w[i-1] + k3w)
            k4z = dt * zf(w[i-1] + k3w)
            # Update
            w[i] = w[i-1] + 1. / 6. * (k1w + 2.0 * (k2w + k3w) + k4w)
            T[i] = T[i-1] + 1. / 6. * (k1T + 2.0 * (k2T + k3T) + k4T)
            z[i] = z[i-1] + 1. / 6. * (k1z + 2.0 * (k2z + k3z) + k4z)
            
        self.storage = (T, w, z)
        return T, w, z
            
if __name__ == "__main__":
    snd = snd.Sounding(None, None)
    snd.from_lapse_rate(trial_lapse_rate, 0, 2e3, 2000)
    
    parcel = CloudParcel(T0 = 301.)
    T, w, z = parcel.run(1.0, 10000, snd)
    plt.plot(z)
    plt.show()
