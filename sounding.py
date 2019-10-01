#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:43:28 2019

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
gval = 9.81 # m / s^2
Rideal = 8.31446261815324	# J / k / mol
Rair = 287.058  # J / k / kg
Rv = 461.5      # J / k / kg
Lv = 2.5e6      # J / kg
cpa = 1004.0    # J / kg / K
cpv = 716.0     # J / kg / K


def saturation_pressure(T):
    T0 = 273.16
    return 610.78 * np.exp(Lv / Rv / T0 * (T - T0) / T)
    

class Sounding(object):
    def __init__(self, z, T):
        if (z is None):
            assert(z == T)
        self.__z = z
        self.__T = T
        
    def from_lapse_rate(self, g_rate, zb, zt, nsamps, T0=300.0):
        self.__z = np.linspace(zb, zt, nsamps)
        self.__T = np.full(nsamps, T0)
        dz = np.diff(self.__z)[0]
        for i, zv in enumerate(self.__z):
            self.__T[i] = self.__T[i-1] + g_rate(zv) * dz
        
    def sample(self, z):
        #assert(self.__z.max() + 10.0 >= z and self.__z.min() - 10.0 <= z)
        return np.interp(z, self.__z, self.__T)
    
            
    def add_points(self, z, T):
        z = np.array(z, dtype=np.float)
        T = np.array(T, dtype=np.float)
        self.__T = np.r_[self.__T, T] if self.__T is not None else T
        self.__z = np.r_[self.__z, z] if self.__z is not None else z
        
    def pressure_profile(self, p0=1e5):
        assert(self.__z is not None and self.__T is not None)
        p = np.empty(self.__z.size)
        p[0] = p0
        for i in range(1, p.size):
            p[i] = p0 * np.exp(np.trapz(-gval / Rair / self.__T[:i], self.__z[:i]))
        return p
    
    
    @property
    def height(self):
        return self.__z
        
    def plot(self):
        f, ax = plt.subplots()
        ax.plot(self.__T, self.__z)
        plt.show()
        
        
if __name__ == "__main__":
    def trial_lapse_rate(z):
        if 0.0 <= z < 100.0:
            return -20.0e-3
        elif 100.0 <= z < 2000.0:
            return -9.8e-3
        else:
            return 20e-3
    
    snd = Sounding(None, None)
    snd.from_lapse_rate(trial_lapse_rate, 0, 2e3, 1000)
    snd.plot()
    snd.sample(1e3)
    plt.plot(snd.pressure_profile(), snd.height)