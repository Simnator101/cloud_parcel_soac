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
#    elif 1500.0 <= z < 3000.0:
#        return 2e-3
    elif 1500.0 <= z < 10000.0:
        return -6.5e-3
    else:
        return 5e-3


class CloudParcel(object):
    def __init__(self, T0=300.0, z0=0.0, w0=0.0, q0=0.0, mix_len=0.0, method='RK4'):
        self.__t0 = T0
        self.__w0 = w0
        self.__z0 = z0
        self.__q0 = q0
        self.__mu = mix_len
        self.induced_mass = 0.5
        self.storage = None
        self.dt = None
        self.surf_p = 1e5
        self.method = method
        
    @staticmethod
    def tanh_vapour_flow(wv, wl, wmax, dt):
        v = -np.tanh(15.0 * (wv / wmax - 1.0)) * 0.2 * dt
        if v < 0.0:
            return wv * v
        return wl * v
        
        
    def run(self, dt, NT, environ, flux_func=None):
        assert(environ is not None)
        assert(NT > 0)
        assert(dt > 0.0)
        #print(self.__t0, self.__w0, self.__z0)
        
        T = np.full(NT, self.__t0)
        w = np.full(NT, self.__w0)
        z = np.full(NT, self.__z0)
        q = np.full(NT, self.__q0)
        l = np.zeros(NT)
        p = environ.pressure_profile()
        acc = gval / (1 + self.induced_mass)
        
        self.dt = dt
        self.surf_p = p[0]
                
        def wf(T, z, l, w):
            mu_eval = self.__mu
            if type(mu_eval) is not float:
               mu_eval = mu_eval(z)            
            
            return acc * ((T - environ.sample(z)) / environ.sample(z) - l) -\
                          mu_eval /(1. + self.induced_mass) * abs(w) * w
        
        # Different Flux functions
        def flux(wv, wl, T, z):
            wmax = snd.saturation_pressure(T) / np.interp(z, environ.height, p) * Rair / Rv
            if flux_func is None:
                fl = 0.0
                if wv/wmax > 1.01:
                    fl = wmax - wv
                elif wv/wmax < 0.99:
                    fl = min(wmax - wv, wl)
                return fl#*(1-np.exp(-dt/self.__Kmix))
            
            assert callable(flux_func), str(flux_func) + " is not a function"
            return flux_func(wv, wl, wmax, dt)
        
        
        def Tf(w, wv, wl, T, z):
            mu_eval = self.__mu
            if type(mu_eval) is not float:
               mu_eval = mu_eval(z) 
            return -gval / cpa * w - Lv / cpa * flux(wv, wl, T, z) - mu_eval *\
                    (T - environ.sample(z)) * abs(w)
        
        def zf(w):
            return w
        
        def qf(w, wv, wl, T, z):
            mu_eval = self.__mu
            if type(mu_eval) is not float:
               mu_eval = mu_eval(z)
            return flux(wv, wl, T, z) - mu_eval * (wv - environ.sample_q(z)) *  abs(w)
        
        def lf(w, wv, wl, T, z):
            mu_eval = self.__mu
            if type(mu_eval) is not float:
               mu_eval = mu_eval(z)
            return -flux(wv, wl, T, z) - mu_eval * wl * abs(w)
        
        if self.method == 'RK4':
            for i in range(1, NT):
                # K1
                k1w = dt * wf(T[i-1], 
                              z[i-1], 
                              l[i-1],
                              w[i-1])
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
                              l[i-1] + k1l / 2,
                              w[i-1] + k1w / 2)
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
                              l[i-1] + k2l / 2,
                              w[i-1] + k2w / 2)
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
                              l[i-1] + k3l,
                              w[i-1] + k3w)
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

        elif self.method == 'Euler':
            for i in range(1, NT):
                w[i] = w[i-1] + dt* wf(T[i-1], z[i-1], l[i-1], w[i-1])
                T[i] = T[i-1] + dt* Tf(w[i-1], q[i-1], l[i-1], T[i-1], z[i-1])
                z[i] = z[i-1] + dt* zf(w[i-1])
                q[i] = q[i-1] + dt* qf(w[i-1], q[i-1], l[i-1], T[i-1], z[i-1])
                l[i] = l[i-1] + dt* lf(w[i-1], q[i-1], l[i-1], T[i-1], z[i-1])
                        
        elif self.method == 'Matsuno':
            for i in range(1, NT):
                w_pred = w[i-1] + dt* wf(T[i-1], z[i-1], l[i-1], w[i-1])
                T_pred = T[i-1] + dt* Tf(w[i-1], q[i-1], l[i-1], T[i-1], z[i-1])
                z_pred = z[i-1] + dt* zf(w[i-1])
                q_pred = q[i-1] + dt* qf(w[i-1], q[i-1], l[i-1], T[i-1], z[i-1])
                l_pred = l[i-1] + dt* lf(w[i-1], q[i-1], l[i-1], T[i-1], z[i-1])
                w[i] = w[i-1] + dt* wf(T_pred, z_pred, l_pred, w_pred)
                T[i] = T[i-1] + dt* Tf(w_pred, q_pred, l_pred, T_pred, z_pred)
                z[i] = z[i-1] + dt* zf(w_pred)
                q[i] = q[i-1] + dt* qf(w_pred, q_pred, l_pred, T_pred, z_pred)
                l[i] = l[i-1] + dt* lf(w_pred, q_pred, l_pred, T_pred, z_pred)                
        else:
            raise(ValueError('Input a valid method'))
                
        self.storage = np.array([T, w, z, q, l])
        self.storage = np.vstack((self.storage, self.pressure))
        return self.storage
    
    @property
    def pressure(self):
        assert self.storage is not None, "No profile is available"
        p0 = self.surf_p
        pvals = np.full(len(self.storage[0]), p0)
        for i in range(1, pvals.size):
            pvals[i] = p0 * np.exp(-gval / Rair * np.trapz(1. / self.storage[0][:i],
                                                               self.storage[2][:i]))
        
        return pvals
    
    @property
    def static_energy(self):
        return self.storage[0] * cpa + gval * self.storage[2]
    
    @property
    def static_moist_energy(self):
        return self.static_energy + self.storage[3] * Lv
    
    @property
    def LCL(self):
        assert self.storage is not None, "No profile is available"
        Td = snd.dew_point_temperature(self.storage[3][0], self._surp)      
        return 122.0 * (T - Td)
    
    @property
    def one_cycle(self):
        assert self.storage is not None, "No profile is available"
        index = np.argwhere(np.diff(self.storage[2]) < 0.0)[0,0]
        return self.storage.T[:index].T
            
    def EL(self, environ):
        assert self.storage is not None, "No profile is available"
        T, p = self.one_cycle[np.array([0,5])]
        Tsnd = environ.temperature
        psnd = environ.pressure_profile()
        
        # Interpolate sounding space into profile space
        
        print(T.size, Tsnd.size)
        Tsnd_map = np.empty(T.size)
        for i, pv in enumerate(p):
            ilow = np.argwhere(psnd >= pv)[-1,0]
            iupp = np.argwhere(psnd <= pv)[0,0]
            
            if np.isclose(psnd[iupp], psnd[ilow]):
                Tsnd_map[i] = Tsnd[ilow]
            else:
                invf = 1. + (np.log(psnd[iupp]) - np.log(pv)) / (np.log(pv) - np.log(psnd[ilow]))
                Tsnd_map[i] = Tsnd[iupp] ** (1. / invf) * Tsnd[ilow] ** (1 - 1. / invf)
        return (T - Tsnd_map) / Tsnd_map
        
        
    
# TEST PLOTS
def test_energy_budget(parcel):
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(9, 6))
    t = np.arange(len(parcel.storage[0])) * parcel.dt
    
    ax[0].plot(t, parcel.static_energy * 1e-3)
    ax[0].grid(True)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Energy (kJ/kg)")
    ax[0].set_title("Static Energy")
    ax[0].set_ylim(parcel.static_energy.min() * 1e-3 - 10,
                   parcel.static_energy.max() * 1e-3 + 10)
    
    ax[1].plot(t, parcel.static_moist_energy * 1e-3)
    ax[1].grid(True)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_title("Static Moist Energy")
    plt.show()
    del t
    
def test_water_budget(parcel):
    f, ax = plt.subplots()
    t = np.arange(len(parcel.storage[0])) * parcel.dt
    
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Water Concentration (g/kg)")
    ax.set_title("Water Stores")
    
    ax.plot(t, parcel.storage[3] * 1e3, label='Water Vapour')
    ax.plot(t, parcel.storage[4] * 1e3, label='Water Liquid')
    ax.plot(t, np.sum(parcel.storage[3:], axis=0) * 1e3, label='Water Budget')
    
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()
    del t
            
if __name__ == "__main__":
    sounding = snd.Sounding(None, None)
    sounding.from_lapse_rate(trial_lapse_rate, 0, 20e3, 10000)
    #sounding.from_wyoming_httpd(snd.SoundingRegion.NORTH_AMERICA, 72201, "01102019")
    
    parcel = CloudParcel(T0 = sounding.surface_temperature + 1.,
                         q0 = 0.02,
                         mix_len=0.0, w0=0.0, method='Matsuno')
    
    T, w, z, q, l, p = parcel.run(0.5, 6000, sounding, flux_func=CloudParcel.tanh_vapour_flow)


    plt.plot(np.arange(6000) * 0.5, z)
    plt.show()
    

    f, ax = sounding.SkewT_logP(show=False)
    ax.plot(T, p * 1e-2)
    plt.show()
    
    test_energy_budget(parcel)
#    test_water_budget(parcel)
    
    
#    plt.plot(q+l)
#    plt.ylim([0.,0.0201])
#    plt.show()
#    plt.plot(0.5*w**2+gval*z+cpa*T+q*cpv*T-(Rair+q*Rv)*T)
#    plt.show()
#    plt.plot(0.5*w**2+gval*z+cpa*T+q*cpv*T-(Rair+q*Rv)*T*(p-p[0])/p)
#    plt.plot(0.5*w**2)
#    plt.plot(gval*z)
#    plt.plot(cpa*T)
#    plt.plot(q*cpv*T)
#    plt.plot(-(Rair+q*Rv)*T*(p-p[0])/p)
#    plt.legend(['total','kin','pot','temp','water','druk'])
