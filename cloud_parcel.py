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
from datetime import datetime


# Constants
gval = 9.81                     # m / s^2
Rideal = 8.31446261815324	    # J / k / mol
Rair = 287.058                  # J / k / kg
Rv = 461.5                      # J / k / kg
cpa = 1004.0                    # J / kg / K
cpv = 716.0                     # J / kg / K
Lv = 2.5e6                      # J / kg
hydrometeor_r = 4e-3            # 1 / s
water_std_density = 997.0       # kg / m ^ 3

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
        self.__iecmwf = [None, None, None]
                
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
        
        def precipitation(wl):
            if not environ.wyoming_info.valid or not hasattr(environ, 'rain'):
                return 0.0
            """
            pwat = environ.wyoming_info.precipitable_water_mm * 1e-3
            slat, slon = (environ.wyoming_info.lat,
                          environ.wyoming_info.lon % 360.0)
            
            tdelta = environ.wyoming_info.observation_time - datetime(1900, 1, 1) 
            
            if self.__iecmwf[0] is None:
                ctim = tdelta.total_seconds() / 60 / 60 
                self.__iecmwf[0] = np.argmin(np.abs(environ.ecmwf_dimensions[0] - ctim))
                self.__iecmwf[1] = np.argmin(np.abs(environ.ecmwf_dimensions[1] - slat))
                self.__iecmwf[2] = np.argmin(np.abs(environ.ecmwf_dimensions[2] - slon))
            
            itim, ilat, ilon = self.__iecmwf
            dset = environ.rain[itim,
                                ilat-1:ilat+1,
                                ilon-1:ilon+1]
            cp = dset.sum() + dset[1].sum() + dset[:,1].sum() + dset[1,1]
            cp /= 16.

            return (dt/24/3600) * cp / pwat * wl  
            """
            return wl * hydrometeor_r
        
        
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
            if not isinstance(mu_eval, (int, float)):
               mu_eval = mu_eval(z)
            return flux(wv, wl, T, z) - mu_eval * (wv - environ.sample_q(z)) *  abs(w)
        
        def lf(w, wv, wl, T, z):
            mu_eval = self.__mu
            if not isinstance(mu_eval, (int, float)):
               mu_eval = mu_eval(z)
            return -flux(wv, wl, T, z) - precipitation(wl) - mu_eval * wl * abs(w)
        
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
        
        
        self.precip_rate = np.array([precipitation(lv) for lv in l])
        self.storage = {'T' : T,
                        'w' : w,
                        'z' : z,
                        'q' : q,
                        'l' : l}
        self.storage['p'] = self.pressure
        #self.storage = np.vstack((self.storage, self.pressure))
        del self.__iecmwf
        return self.storage
    
    @property
    def pressure(self):
        assert self.storage is not None, "No profile is available"
        p0 = self.surf_p
        pvals = np.full(len(self.storage['T']), p0)
        for i in range(1, pvals.size):
            pvals[i] = p0 * np.exp(-gval / Rair * np.trapz(1. / self.storage['T'][:i],
                                                               self.storage['z'][:i]))
        
        return pvals
    
    @property
    def virtual_temperature(self):
        return self.storage['T'] * (1. + 0.608 * self.storage['q'] - self.storage['l'])
    
    @property
    def static_energy(self):
        return self.storage['T'] * cpa + gval * self.storage['z']
    
    @property
    def static_moist_energy(self):
        return self.static_energy + self.storage['q'] * Lv
    
    def _buoyancy(self, environ):
        assert self.storage is not None, "No profile is available"
        # We only take one whole cycle
        index = np.argwhere(np.diff(self.storage['z']) < 0.0)[0,0]
        Te = np.empty(index)
        pe = environ.pressure_profile()
        
        for i,pv in enumerate(self.pressure[:index]):
            ilow = np.argwhere(pe >= pv)[-1,0]
            iupp = np.argwhere(pe <= pv)[0,0]
            Te[i] = environ.temperature[ilow]
            
            if not np.isclose(pe[iupp], pe[ilow]):
                invf = 1. + (np.log(pe[iupp]) - np.log(pv)) / (np.log(pv) - np.log(pe[ilow]))
                Te[i] = environ.temperature[iupp] ** (1. / invf) * environ.temperature[ilow] ** (1 - 1. / invf)
        B = (self.storage['T'][:index] - Te) / Te - self.storage['l'][:index]
        return B
        
        
    def CAPE_CIN(self, environ):
        B = self._buoyancy(environ)
        Cp, Cn = (0.0, 0.0)
        
        for i in range(1, B.size):
            dz = self.storage['z'][i] - self.storage['z'][i-1]
            Bint = (B[i-1] + B[i]) / 2.0
            if Bint >= 0.0:
                Cp += Bint * dz
            else:
                Cn += Bint * dz
            
        return (gval / (1.0 + self.induced_mass) * Cp,
                gval / (1.0 + self.induced_mass) * Cn)
    
    @property
    def precipitation_total(self):
        if not hasattr(self, "precip_rate"):
            return 0.0
        #index = np.argwhere(np.diff(self.storage['z']) < 0.0)[0,0]
        air_d = self.pressure / Rair / self.virtual_temperature
        layer_thck = np.abs(np.gradient(self.storage['z']))
        precip_a = np.trapz(self.precip_rate * air_d * layer_thck, dx=self.dt)
        
        peak_i = np.argmax(self.precip_rate)
        max_p = self.precip_rate[peak_i]
        right_i = np.argwhere(self.precip_rate / max_p <= 0.05).flatten()
        right_i = list(filter(lambda i : i > peak_i, right_i))[0]
        t_ascent = self.dt * right_i
        
        return precip_a / water_std_density * 86400. / t_ascent
        
    
    @property
    def LCL(self):
        assert self.storage is not None, "No profile is available"
        Td = snd.dew_point_temperature(self.storage['q'][0], self.surf_p)      
        return 122.0 * (self.storage['T'] - Td)
    
    @property
    def one_cycle(self):
        assert self.storage is not None, "No profile is available"
        index = np.argwhere(np.diff(self.storage['z']) < 0.0)[0,0]
        temp = np.array(list(self.storage.values())).T[:index].T
        out = dict()
        ind = 0
        
        for key,_ in self.storage.items():
            out[key] = temp[ind]
            ind += 1    
        
        return out
    
    def Tv_deficit(self, environ):
        assert self.storage is not None, "No profile is available"
        Td = self.virtual_temperature
        pe = environ.pressure_profile()
        
        
        for i,pv in enumerate(self.pressure):
            ilow = np.argwhere(pe >= pv)[-1,0]
            iupp = np.argwhere(pe <= pv)[0,0]
            Te = environ.temperature[ilow]
            qe = environ.humidity[ilow]
            if not np.isclose(pe[iupp], pe[ilow]):
                invf = 1. + (np.log(pe[iupp]) - np.log(pv)) / (np.log(pv) - np.log(pe[ilow]))
                Te = environ.temperature[iupp] ** (1. / invf) * environ.temperature[ilow] ** (1 - 1. / invf)
                qe = environ.humidity[iupp] ** (1. / invf) * environ.humidity[ilow] ** (1 - 1. / invf)
            Td[i] -= Te * (1. + 0.608 * qe)
        return Td
            
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
    t = np.arange(len(parcel.storage['T'])) * parcel.dt
    
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
    t = np.arange(len(parcel.storage['T'])) * parcel.dt
    
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Water Concentration (g/kg)")
    ax.set_title("Water Stores")
    
    ax.plot(t, parcel.storage['q'] * 1e3, label='Water Vapour')
    ax.plot(t, parcel.storage['l'] * 1e3, label='Water Liquid')
    ax.plot(t, (parcel.storage['q'] + parcel.storage['l']) * 1e3,
            label='Water Budget')
    
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()
    del t

    
def test_model_stability(dt=0.5):
    from scipy.signal import find_peaks
    
    sounding = snd.Sounding(None, None)
    sounding.from_lapse_rate(trial_lapse_rate, 0, 20e3, 10000)
    methods = ["Euler", "Matsuno", "RK4"]
    parcels = [CloudParcel(T0 = sounding.surface_temperature + 1.,
                         q0 = 0.02,
                         mix_len=0.0, w0=0.0, method=method) 
               for method in methods]   
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(9,6))
    for i, mstr, parcel in zip(range(3), methods, parcels):
        T, w, z, q, l, pre, p = parcel.run(.5, 10000, sounding, flux_func=CloudParcel.tanh_vapour_flow)
        zpeaki = find_peaks(z)[0]
        zpeaks = z[zpeaki] / z[zpeaki[0]]
        
        zmin = -z + z[zpeaki[0]]
        zthroughi = find_peaks(zmin)[0]
        
        zthroughs = zmin[zthroughi] / zmin[zthroughi[0]]
        
        ax[0].plot(zpeaki * dt, zpeaks, color='C%i' % i)
        ax[1].plot(zthroughi * dt, zthroughs, color='C%i' % i, label=mstr)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel("Time (s)")
    ax[1].set_xlabel("Time (s)")
    ax[0].set_ylabel("Relative Peak/Through Height")
    ax[0].set_xlim(500, 4000)
    ax[1].set_xlim(500, 4000)
    
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()
    
    
def mixing_func(z):
    starting_radius = 1000
    return 0.2 / (0.1 * z + starting_radius)
    
            
if __name__ == "__main__":
    sounding = snd.Sounding(None, None)
    #sounding.from_lapse_rate(trial_lapse_rate, 0, 20e3, 10000)
    sounding.from_wyoming_httpd(snd.SoundingRegion.NORTH_AMERICA, 72210, "10102018",utc12=False)
    sounding.attach_ecmwf_field("./oct_2018_cp.nc", {'cp': 'rain'})
    
    parcel = CloudParcel(T0 = sounding.surface_temperature + 3,
                         q0 = sounding.surface_humidity,
                         mix_len=1./10e3, w0=0.0, method='RK4')
    
    parcel.run(0.5, 6000, sounding, flux_func=CloudParcel.tanh_vapour_flow)


    plt.plot(np.arange(6000) * 0.5, parcel.storage['z'])
    plt.grid()
    plt.show()
    

    f, ax = sounding.SkewT_logP(show=False)
    ax.plot(parcel.storage['T'], parcel.pressure * 1e-2)
    plt.show()
    
    # Test Sections
#    test_energy_budget(parcel)
#    test_model_stability()
#    test_water_budget(parcel)
#    plt.plot(np.arange(6000) * 0.5,parcel.precip_rate)
#    plt.grid()
#    plt.show()
#    plt.plot(np.arange(6000) * 0.5,q+l)
#    plt.ylim([0,1.1*sounding.surface_humidity])
#    plt.grid()
#    plt.show()
    
#    plt.plot(np.arange(6000) * 0.5, parcel.virtual_temperature)
    plt.show()
