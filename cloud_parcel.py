#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:40:54 2019

Cloud parcel module class
"""

import matplotlib.pyplot as plt
import numpy as np
import sounding as snd
import matplotlib.pyplot as plt

# Constants
gval = 9.81 # m / s^2
Rideal = 8.31446261815324	# J / k / mol
Rair = 287.058  # J / k / kg
Rv = 461.5      # J / k / kg
cpa = 1004.0    # J / kg / K
cpv = 716.0     # J / kg / K
Lv = 2.5e6      # J / kg

def trial_lapse_rate(z):
    if 0.0 <= z < 1500.0:
        return -10.0e-3
    elif 1500.0 <= z < 10000.0:
        return -6.5e-3
    else:
        return 5e-3


class CloudParcel(object):
    def __init__(self, T0=300.0, z0=0.0, w0=0.0, q0=0.0, Kmix=1.):
        self.__t0 = T0
        self.__w0 = w0
        self.__z0 = z0
        self.__q0 = q0
        self.__Kmix = Kmix
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
        q = np.full(NT, self.__q0)
        l = np.zeros(NT)
        p = environ.pressure_profile()
        acc = gval / (1 + self.induced_mass)
        
        def wf(T, z, l):
            return acc * ((T - environ.sample(z)) / environ.sample(z) - l)
        
        def flux(wv, wl, T, z):
            wmax = snd.saturation_pressure(T) / np.interp(z, environ.height, p) * Rair / Rv
            #return min(wmax - wv, wl)
            if wv/wmax > 1.01:
                fl = wmax - wv
            elif wv/wmax < 0.99:
                fl = min(wmax - wv, wl)
            else:
                fl = 0
            return fl*(1-np.exp(-dt/self.__Kmix))
=======
            return min(wmax - wv, wl)
>>>>>>> b3cfa883833e4c63d0e0a24be9cfb0e2ee4e5708
        
        def Tf(w, wv, wl, T, z):
            return -gval / cpa * w - Lv / cpa * flux(wv, wl, T, z)
        
        def zf(w):
            return w
        
        def qf(w, wv, wl, T, z):
            return flux(wv, wl, T, z)
        
        def lf(w, wv, wl, T, z):
            return -flux(wv, wl, T, z)
        
        for i in range(1, NT):
            # K1
            k1w = dt * wf(T[i-1], 
                          z[i-1], 
                          l[i-1])
            k1T = dt * Tf(w[i-1], 
                          q[i-1], 
                          l[i-1], 
                          T[i-1], 
                          z[i-1])
            k1z = dt * zf(w[i-1])
            k1q = dt * qf(w[i-1], 
                          q[i-1], 
                          l[i-1], 
                          T[i-1], 
                          z[i-1])
            k1l = dt * lf(w[i-1], 
                          q[i-1], 
                          l[i-1], 
                          T[i-1], 
                          z[i-1])
            # K2
            k2w = dt * wf(T[i-1] + k1T / 2,
                          z[i-1] + k1z / 2,
                          l[i-1] + k1l / 2)
            k2T = dt * Tf(w[i-1] + k1w / 2,
                          q[i-1] + k1q / 2,
                          l[i-1] + k1l / 2,
                          T[i-1] + k1T / 2,
                          z[i-1] + k1z / 2)
            k2z = dt * zf(w[i-1] + k1w / 2)
            k2q = dt * qf(w[i-1] + k1w / 2,
                          q[i-1] + k1q / 2,
                          l[i-1] + k1l / 2,
                          T[i-1] + k1T / 2,
                          z[i-1] + k1z / 2)
            k2l = dt * lf(w[i-1] + k1w / 2,
                          q[i-1] + k1q / 2,
                          l[i-1] + k1l / 2,
                          T[i-1] + k1T / 2,
                          z[i-1] + k1z / 2)
            # K3
            k3w = dt * wf(T[i-1] + k2T / 2,
                          z[i-1] + k2z / 2,
                          l[i-1] + k2l / 2)
            k3T = dt * Tf(w[i-1] + k2w / 2,
                          q[i-1] + k2q / 2,
                          l[i-1] + k2l / 2, 
                          T[i-1] + k2T / 2,
                          z[i-1] + k2z / 2)
            k3z = dt * zf(w[i-1] + k2w / 2)
            k3q = dt * qf(w[i-1] + k2w / 2,
                          q[i-1] + k2q / 2,
                          l[i-1] + k2l / 2, 
                          T[i-1] + k2T / 2,
                          z[i-1] + k2z / 2)
            k3l = dt * lf(w[i-1] + k2w / 2,
                          q[i-1] + k2q / 2,
                          l[i-1] + k2l / 2, 
                          T[i-1] + k2T / 2,
                          z[i-1] + k2z / 2)
            # K4
            k4w = dt * wf(T[i-1] + k3T,
                          z[i-1] + k3z,
                          l[i-1] + k3l)
            k4T = dt * Tf(w[i-1] + k3w,
                          q[i-1] + k3q,
                          l[i-1] + k3l,
                          T[i-1] + k3T,
                          z[i-1] + k3z)
            k4z = dt * zf(w[i-1] + k3w)
            k4q = dt * qf(w[i-1] + k3w,
                          q[i-1] + k3q,
                          l[i-1] + k3l,
                          T[i-1] + k3T,
                          z[i-1] + k3z)
            k4l = dt * lf(w[i-1] + k3w,
                          q[i-1] + k3q,
                          l[i-1] + k3l,
                          T[i-1] + k3T,
                          z[i-1] + k3z)
            # Update
            w[i] = w[i-1] + 1. / 6. * (k1w + 2.0 * (k2w + k3w) + k4w)
            T[i] = T[i-1] + 1. / 6. * (k1T + 2.0 * (k2T + k3T) + k4T)
            z[i] = z[i-1] + 1. / 6. * (k1z + 2.0 * (k2z + k3z) + k4z)
            q[i] = q[i-1] + 1. / 6. * (k1q + 2.0 * (k2q + k3q) + k4q)
            l[i] = l[i-1] + 1. / 6. * (k1l + 2.0 * (k2l + k3l) + k4l)
            
        self.storage = (T, w, z, q, l)
        return T, w, z, q, l
            
if __name__ == "__main__":
    sounding = snd.Sounding(None, None)
    sounding.from_lapse_rate(trial_lapse_rate, 0, 20e3, 10000)
    
    parcel = CloudParcel(T0 = 301., q0=0.0)
    
    T, w, z, q, l = parcel.run(1.0, 6000, sounding)

    #p = np.full(T.size, 1e5)
    #for i in range(1, T.size):
    #    p[i] = p[0] * np.exp(np.trapz(-gval / Rair / T[:i], z[:i]))
    plt.plot(z)
    plt.show()
    #plt.plot(q+l)
    #plt.ylim([0.,0.0201])
    #plt.show()
    #plt.plot(0.5*w**2+gval*z+cpa*T+q*cpv*T-(Rair+q*Rv)*T)
    #plt.show()
    #plt.plot(0.5*w**2+gval*z+cpa*T+q*cpv*T-(Rair+q*Rv)*T*(p-p[0])/p)
    #plt.plot(0.5*w**2)
    #plt.plot(gval*z)
    #plt.plot(cpa*T)
    #plt.plot(q*cpv*T)
    #plt.plot(-(Rair+q*Rv)*T*(p-p[0])/p)
    #plt.legend(['total','kin','pot','temp','water','druk'])
