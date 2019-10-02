#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:43:28 2019

@author: simon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter, MultipleLocator
import matplotlib.transforms as mtrans
from datetime import date, datetime
from enum import Enum
import requests
import re
import os.path
from scipy.integrate import odeint

import skewt_logp

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

def relative_humidity(qs, p, T):
    air_d = p / T / Rair
    return qs * air_d * Rv * T / saturation_pressure(T) * 100.0

def moist_adiabatic_lapse_rate(T, p):
    qs = saturation_pressure(T) / p * Rair / Rv
    eps = Rair / Rv
    frac = (Rair * T + Lv * qs) / (cpa + (Lv * Lv * qs * eps) / Rair / T / T)
    return frac / p

class SoundingRegion(Enum):
    NORTH_AMERICA = "naconf"
    SOUTH_AMERICA = "samer"
    SOUTH_PACIFIC = "pac"
    NEW_ZEELAND = "nz"
    ANTARTICA = "ant"
    ARCTIC = "np"
    EUROPE = "europe"
    AFRICA = "africa"
    SOUTHEAST_ASIA = "seasia"
    MIDEAST = "mideast"
    
class WyomingSoundingInfo(object):
    def __init__(self, sfields=None):
        self.id = None
        self.id_num = 0
        self.observation_time = date(1900, 1, 1)
        self.lat = 0.0
        self.lon = 0.0
        self.z0 = 0.0
        self.showalter_index = 0.0
        self.lifted_index = 0.0
        self.virtual_LIFT = 0.0
        self.SWEAT_index = 0.0
        self.K_index = 0.0
        self.cross_totals_index = 0.0
        self.vertical_totals_index = 0.0
        self.totals_totals_index = 0.0
        self.CAPE = 0.0
        self.virtual_CAPE = 0.0
        self.CIN = 0.0
        self.virtual_CIN = 0.0
        self.bulk_Richardson_number = 0.0
        self.bulk_Richardson_number_capv = 0.0
        self.LCL_K = 0.0
        self.LCL_hPa = 0.0
        self.mean_PBL_ptemp = 0.0
        self.mean_PBL_mixing_r = 0.0
        self.half_atmos_thickness = 0.0
        self.precipitable_water_mm = 0.0
        
        if sfields is not None:
            for s in sfields:
                sid, sval = s.split(': ')
                sid = sid.strip()
                if sid == "Station identifier":
                    self.id = sval
                elif sid == "Station number":
                    self.id_num = int(sval)
                elif sid == "Observation time":
                    self.observation_time = datetime.strptime(sval, "%y%m%d/%H%M")
                elif sid == "Station latitude":
                    self.lat = float(sval)
                elif sid == "Station longitude":
                    self.lon = float(sval)
                elif sid == "Station elevation":
                    self.z0 = float(sval)
                elif sid == "Showalter index":
                    self.showalter_index = float(sval)
                elif sid == "Lifted index":
                    self.lifted_index = float(sval)
                elif sid == "LIFT computed using virtual temperature":
                    self.virtual_LIFT = float(sval)
                elif sid == "SWEAT index":
                    self.SWEAT_index = float(sval)
                elif sid == "K index":
                    self.K_index = float(sval)
                elif sid == "Cross totals index":
                    self.cross_totals_index = float(sval)
                elif sid == "Vertical totals index":
                    self.vertical_totals_index = float(sval)
                elif sid == "Totals totals index":
                    self.totals_totals_index = float(sval)
                elif sid == "Convective Available Potential Energy":
                    self.CAPE = float(sval)
                elif sid == "CAPE using virtual temperature":
                    self.virtual_CAPE = float(sval)
                elif sid == "Convective Inhibition":
                    self.CIN = float(sval)
                elif sid == "CINS using virtual temperature":
                    self.virtual_CIN = float(sval)
                elif sid == "Bulk Richardson Number":
                    self.bulk_Richardson_number = float(sval)
                elif sid == "Bulk Richardson Number using CAPV":
                    self.bulk_Richardson_number_capv = float(sval)
                elif sid == "Temp [K] of the Lifted Condensation Level":
                    self.LCL_K = float(sval)
                elif sid == "Pres [hPa] of the Lifted Condensation Level":
                    self.LCL_hPa = float(sval)
                elif sid == "Mean mixed layer potential temperature":
                    self.mean_PBL_ptemp = float(sval)
                elif sid == "Mean mixed layer mixing ratio":
                    self.mean_PBL_mixing_r = float(sval)
                elif sid == "1000 hPa to 500 hPa thickness":
                    self.half_atmos_thickness = float(sval)
                elif sid == "Precipitable water [mm] for entire sounding":
                    self.precipitable_water_mm = float(sval)
                
                
        
    @property
    def valid(self):
        return True if self.id is not None else False
        
 
class Sounding(object):
    def __init__(self, z, T, qhum=None):
        if (z is None):
            assert(z == T)
            assert(z == qhum)
        self.__z = z
        self.__p = None
        self.__T = T
        self.__Td = None
        self.__q = qhum 
        self.wyoming_info = WyomingSoundingInfo()
        
    def from_lapse_rate(self, g_rate, zb, zt, nsamps, T0=300.0, q_f=lambda z: 1e-6):
        self.__z = np.linspace(zb, zt, nsamps)
        self.__T = np.full(nsamps, T0)
        self.__q = np.zeros(nsamps)
        dz = np.diff(self.__z)[0]
        for i, zv in enumerate(self.__z):
            self.__T[i] = self.__T[i-1] + g_rate(zv) * dz
            self.__q[i] = q_f(zv)
        self.__p = self.pressure_profile()
        self.__Td = np.empty(nsamps)
        for i in range(nsamps):
            RH = relative_humidity(self.__q[i], self.__p[i], self.__T[i])
            self.__Td[i] = self.dew_point_NOAA(self.__T[i], RH)
            
    def from_wyoming_file(self, file):
        assert os.path.isfile(file), "The sounding file %s cannot be found" % file
        txt = None
        with open(file, "r") as f:
            txt = f.read()
        self.from_wyoming_text(txt)
            
    def from_wyoming_text(self, txt):
        lines = txt.split('\n')
        data_start, data_mid, data_end = (-1, -1, -1)
        for i, ln in enumerate(lines):
            if not ln and data_start > 0 and data_end < 0:
                data_end = i
            elif ln and ln == ''.join(['-'] * len(ln)):
                data_start = i + 1
        assert data_start > 0, "Invalid Wyoming sounding: no data found"
        
        # Find mid point
        for i, ln in enumerate(lines[data_start:data_end]):
            if ln[0].isalpha():
                data_mid = i + data_start
                break
        str_fields = lines[data_start:data_mid]
        id_fields = lines[data_mid+1:data_end]
        
        # Grab Sounding Info
        rec_field = None
        for ln in str_fields:
            record = list(map(float, filter(None, re.split('\s+', ln))))
            if len(record) != 11:
                continue
            rec_field = record if rec_field is None else np.vstack((rec_field, record))
        self.__p = rec_field.T[0] * 1e2     # to Pa
        self.__z = rec_field.T[1]           # in metres
        self.__T = rec_field.T[2] + 273.15  # in Kelvin
        self.__Td = rec_field.T[3] + 273.15 # in Kelvin
        self.__q = rec_field.T[5] * 1e-3    # In kg/kg

        self.wyoming_info = WyomingSoundingInfo(id_fields)
        
        
            
    def from_wyoming_httpd(self, region, station_id, when, utc12=False):
        # Parse region
        assert type(region) is type(SoundingRegion.NEW_ZEELAND)
        region_str = "region=%s" % region.value
        
        # Parse date
        utc_val = 12 if utc12 else 0
        if type(when) is str:
            when = datetime.strptime(when, "%d%m%Y")
        date_str = "YEAR=%i&MONTH=%02i&FROM=%02i%02i&TO=%02i%02i" %\
                   (when.year, when.month, when.day, utc_val, when.day, utc_val)
        
        # Parse Station Id
        station_str = "STNM=%i" % station_id
        
        # HTTPD Request
        url = "&".join([region_str, "TYPE=TEXT%3ALIST", date_str, station_str])
        url = "http://weather.uwyo.edu/cgi-bin/sounding?" + url
        rres = requests.get(url)
        assert rres.status_code == 200, "URL %s cannot be read" % url
        txt = re.sub('<[^<]+?>', '', rres.text)
        self.from_wyoming_text(txt.replace("\\n", "\n"))
                
        
    def sample(self, z):
        #assert(self.__z.max() + 10.0 >= z and self.__z.min() - 10.0 <= z)
        return np.interp(z, self.__z, self.__T)
    
            
    def add_points(self, z, T, q=1e-6):
        z = np.array(z, dtype=np.float)
        T = np.array(T, dtype=np.float)
        self.__q = np.r_[self.__q, q] if self.__q is not None else q
        self.__T = np.r_[self.__T, T] if self.__T is not None else T
        self.__z = np.r_[self.__z, z] if self.__z is not None else z
        self.__p = self.pressure_profile()
        Td = self.dew_point_NOAA(T, relative_humidity(q, self.__p[-1], T))
        self.__Td = np.r_[self.__Td, Td] if self.__Td is not None else Td
            
    def pressure_profile(self, p0=1e5):
        # Return Sounding Profile
        if self.__p is not None:
            return self.__p
        
        # Return calculated profile
        assert(self.__z is not None and self.__T is not None)
        p = np.empty(self.__z.size)
        p[0] = p0
        for i in range(1, p.size):
            p[i] = p0 * np.exp(np.trapz(-gval / Rair / self.__T[:i], self.__z[:i]))
        return p
    
    @staticmethod
    def dew_point_NOAA(T, RH):
        T -= 273.15
        b = 18.678
        c = 257.14
        d = 234.5
    
        ps = np.exp((b - T / d) * T / (c + T))
        if np.isclose(RH, 0.0):
            return 0.0
        g = np.log(RH / 100.0 * ps)
        return c * g / (b - g) + 273.15
            
       
    @property
    def height(self):
        return self.__z
    
    @property
    def RH(self):
        loopd = zip((self.__q, self.__p, self.__T))
        RH = [relative_humidity(qv, pv, Tv) for qv, pv, Tv in loopd]
        return np.array(RH)
        
        
    def plot(self, show=True):
        difTm = np.abs(self.__T - self.__T[0]).max() + 10
        f, ax = plt.subplots()
        ax.plot(self.__T, self.__z * 1e-3)
        ax.plot(self.__Td, self.__z * 1e-3)
        ax.grid(True)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Height (km)")
        ax.set_xlim(self.__T[0] - difTm, self.__T[0] + difTm)
        if show:
            plt.show()
        return f, ax
        
    def SkewT_logP(self, show=True, isotherms=False, dry_adiabats=True, moist_adiabats=True, sat_mix=True):     
        difTm = np.abs(self.__T - self.__T[0]).max() + 10
        f = plt.figure(figsize=(6.6,6.6))
        ax = f.add_subplot(111, projection='skewx')

        
        ax.grid(True)
        trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
        
        pticks = np.linspace(100, 1000, 10)
        
    
        if isotherms:
            for T in np.arange(10, 410, 10):
                ztop = T / gval * cpa * (1 - (100. / 1050.) ** (Rair / cpa))
                ax.semilogy([T, T + gval / cpa * ztop], [100., 1050.], 'b--', alpha=0.3)
                
        if dry_adiabats:
            isot = np.arange(220, 510, 10, dtype=np.float)
            p = np.linspace(pticks.min(), pticks.max() + 50, 100)
            for th in isot:
                Tvals = th * np.power(p / 1050., Rair / cpa)
                ax.semilogy(Tvals, p, 'g-', alpha=0.3)
                
        if moist_adiabats:
            p = np.linspace(pticks.min(), pticks.max() + 50, 100)[::-1] * 1e2
            for T in np.arange(10, 410, 10):
                res = odeint(moist_adiabatic_lapse_rate, T, p)
                ax.semilogy(res, p * 1e-2, 'g--', alpha=0.3)
                
                
        if sat_mix:
            # these ratios are plotted in g/kg
            srats = np.array([0.1, 0.2, 0.4, 0.6, 1., 2., 4., 6, 10., 15.])
            srats = np.r_[srats, np.arange(20, 50, 10, dtype=np.float)] * 1e-3
            coords = np.array([211.25, 217, 223.5, 228.5, 235, 245, 255.75,
                               262.75, 273, 281.5, 288.5, 298.75, 306.5])
            
            Tr = np.linspace(100, 320, 100)
            for qcoord, qs in zip(coords, srats):
                es = saturation_pressure(Tr)
                plevs = es / qs * Rair / Rv                
                ln, = ax.semilogy(Tr, plevs * 1e-2, '--', color='brown', alpha=0.3)
                
                labelLine(ln, qcoord, label='%.1f' % (qs * 1e3), color='k')                
                
        # Plot Data
        ax.semilogy(self.__T, self.__p * 1e-2, color='C3', label='Temperature (K)')
        ax.semilogy(self.__Td, self.__p * 1e-2, color='C2', label='Dew point temperature (K)')
        
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        
                
        if self.wyoming_info.valid:
            ax.text(0.13, 0.89, "%i %s %s" % (self.wyoming_info.id_num,
                                              self.wyoming_info.id,
                                              self.wyoming_info.observation_time.strftime("%HZ %d %b %Y")),
                    transform=f.transFigure,  weight='bold')
    
        
        # Labels
        ax.set_yticks(pticks)
        ax.set_ylim(1050, 100.0)
    
        yinds = np.array([np.argmin(np.abs(self.__p * 1e-2 - pv))
                          for pv in pticks[1::2]], dtype=np.int)
        for pv, zv in zip(1e-2 * self.__p[yinds], self.__z[yinds]):
            ax.text(0.01, pv, "%.0f m" % zv, transform=trans, verticalalignment='center')
        
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlim(220, 320)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (hPa)")
            
        if show:
            plt.show()
        return f, ax
        
        
if __name__ == "__main__":
    def trial_lapse_rate(z):
        if 0.0 <= z < 100.0:
            return -20.0e-3
        elif 100.0 <= z < 800.0:
            return -9.8e-3
        else:
            return 10e-3
    
    # Trial Lapse rate
    snd = Sounding(None, None)
    snd.from_lapse_rate(trial_lapse_rate, 0, 5e3, 1000)
    snd.plot()
    snd.SkewT_logP()
    # Pressure function
    #plt.plot(snd.pressure_profile(), snd.height)
    
    # Plot Wyoming Sounding
    # Grab North American Sounding over Florida Key West from 1 Oct 2019 00 UTC
    snd.from_wyoming_httpd(SoundingRegion.NORTH_AMERICA, 72201, "01102019")
    snd.SkewT_logP()