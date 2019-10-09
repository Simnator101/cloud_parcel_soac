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
from netCDF4 import Dataset

from skewt_logp import *

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

def dew_point_temperature(q, p):
    T0 = 273.16
    es0 = 610.78
    eta = Rair / Rv
    return T0 / (1 - np.log((p / es0 * q / eta) ** (Rv * T0 / Lv)))

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
        self.eq_level = 0.0
        self.virtual_eq_level = 0.0
        self.LFC = 0.0
        self.virtual_LFC = 0.0
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
                elif sid == "Equilibrum Level":
                    self.eq_level = float(sval)
                elif sid == "Equilibrum Level using virtual temperature":
                    self.virtual_eq_level = float(sval)
                elif sid == "Level of Free Convection":
                    self.LFC = float(sval)
                elif sid == "LFCT using virtual temperature":
                    self.virtual_LFC = float(sval)
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
    
    def __str__(self):
        if not self.valid:
            return None
        stro = """SLAT %.2f $^\\circ$\nSLON %.2f $^\\circ$\nSELV %.2f m\n
SHOW %.2f\nLIFT %.2f\nLFTV %.2f\nSWET %.2f\nKINX %.2f\nCTOT %.2f\nVTOT %.2f\n
TOTL %.2f\nCAPE %.2f J/kg\nCAPV %.2f J/kg\nCINS %.2f J/kg\nCINV %.2f J/kg\n
EQLV %.2f hPa\nEQTV %.2f hPa\nLFCT %.2f hPa\nLFCV %.2f hPa\nBRCH %.2f\n
BRCV %.2f\nLCLT %.2f K\nLCLP %.2f hPa\nMLTH %.2f K\nMLMR %.2f g/kg\n
THCK %.2f m\nPWAT %.2f mm\n
               """
        return stro % (self.lat, self.lon, self.z0, self.showalter_index, self.lifted_index,
                       self.virtual_LIFT, self.SWEAT_index, self.K_index, self.cross_totals_index,
                       self.vertical_totals_index, self.totals_totals_index, self.CAPE,
                       self.virtual_CAPE, self.CIN, self.virtual_CIN, self.eq_level, 
                       self.virtual_eq_level, self.LFC, self.virtual_LFC, self.bulk_Richardson_number,
                       self.bulk_Richardson_number_capv, self.LCL_K, self.LCL_hPa,
                       self.mean_PBL_ptemp, self.mean_PBL_mixing_r,
                       self.half_atmos_thickness, self.precipitable_water_mm)
    
    @property
    def field_str(self):
        return self.__str__()
        
 
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
            
            
    @staticmethod
    def _decode_level(LEV, D, E, mode='AA'):
        p, z = np.nan, np.nan
        
        levid = int(LEV[:2])
        if levid != 88 and mode == 'CC':
            p = float(levid) * 100
            z = float(LEV[2:])
            z = z * 10 + 1e4 if levid >= 70 else z * 10 + 2e4
        elif levid == 99 and mode == 'AA':
            p = float(LEV[2:]) * 1e2
            p += 1e5 if LEV[2] == '0' else 0.0
            z = 0.0
        elif levid == 88 and mode == 'CC':
            p = float(LEV[2:]) * 10
            z = np.nan
        elif levid == 88 and mode == 'AA':
            p = float(LEV[2:]) * 1e2
            z = np.nan
        elif levid == 00:
            p = 1e5   
            z = float(LEV[2:])
        elif levid == 92:
            p = 9.25e4  
            z = float(LEV[2:])
        elif levid == 85:
            p = 8.5e4
            z = float(LEV[2:]) + 1e3
        elif levid == 85:
            p = 8.5e4
            z = float(LEV[2:]) + 1e3
        elif levid == 70:
            p = 7.0e4
            z = float(LEV[2:])
            z += 2e3 if z > 500.0 else 3e3
        elif 30 <= levid <= 50 and mode == 'AA':
            p = levid * 1e3
            z = float(LEV[2:]) * 10 
        elif 15 <= levid <= 25 and mode == 'AA':
            p = levid * 1e3
            z = float(LEV[2:]) * 10 + 1e4
        elif (levid % 11) == 0 and mode == 'BB':
            p = float(LEV[2:]) * 1e2
        elif (levid % 11) == 0 and mode == 'DD':
            p = float(LEV[2:]) * 10
        else:
            p = float(LEV[:2]) * 1e3
            z = float(LEV[2:])
            
        
        
        T = float(D[:3]) / 10.0 if not '/' in D[:3] else np.nan
        T = T if (int(D[2]) % 2) == 0 else -T
        DD = float(D[3:]) if not '/' in D[3:] else np.nan
        Td = T - (DD / 10.0 if DD <= 50. else DD - 50.0)
        
        
        Wd = float(E[:3]) if not '/' in E[:3] else np.nan
        Ws = float(E[3:]) if not '/' in E[3:] else np.nan
        
        
        Ws = Ws if np.mod(Wd, 5.0) == 0.0 else Ws + 100.0
        return np.array([p,z,T,Td,Wd,Ws])
            
    def _TTAA_decode(self, val, settings):
        # First Entry is the time stamp
        if int(val[0][:2]) > 50:
            settings['day'] = int(val[0][:2]) - 50
            settings['use_knots'] = True
        else:
            settings['day'] = int(val[0][:2])
            settings['use_knots'] = False
        settings['utc'] = int(val[0][2:4])
        # Second Entry is ID
        settings['id'] = int(val[1])
        # Third Entry is the surface data
        assert int(val[2][:2]) == 99
        settings['data'][0] = self._decode_level(val[2], val[3], val[4])
        
        # The Following entries are the most important levels
        j = 1
        for i in range(5, 38, 3):
            settings['data'][j] = self._decode_level(val[i], val[i+1], val[i+2])
            j += 1
          
        # Residuals
        settings['trp_data'] = None
        settings['max_wnd'] = None
        settings['shr'] = None
        settings['SHOW'] = 0.0
        settings['mean_wnd'] = None
        j = 38
        while (j + 1 < len(val)):
            if val[j][:2] == '88':
                settings['trp_data'] = self._decode_level(val[j], val[j+1],
                                                          val[j+2])
                j += 3
            elif val[j][:2] == '77' or val[j][:2] == '66':
                settings['max_wnd'] = np.empty(3)
                settings['max_wnd'][0] = float(val[j][2:]) * 1e2
                settings['max_wnd'][1] = float(val[j+1][:2])
                settings['max_wnd'][2] = float(val[j+1][2:]) 
                settings['max_wnd'][2] += 1e2 if np.mod(settings['max_wnd'][1], 5) != 0.0 else 0.0
                j += 2
            elif val[j][0] == '4':
                settings['shr'] = np.array([float(val[j][1:3]), float(val[j][3:])])
                j += 1
            elif val[j] == '51515' and val[j+1] == '10164':
                lift = float(val[j+2][1:])
                lift = (lift - 50) / 10 if lift > 50 else lift / 10.0
                settings['LIFT'] = -lift if (int(val[j+2][1:]) % 2) != 0 else lift
                j += 3
            elif val[j] == '10194':
                settings['mean_wnd'] = np.empty(4)
                settings['mean_wnd'][0] = float(val[j+1][:2])
                settings['mean_wnd'][1] = float(val[j+1][2:]) 
                settings['mean_wnd'][2] = float(val[j+2][:2])
                settings['mean_wnd'][3] = float(val[j+2][2:]) 
                settings['mean_wnd'][1] += 1e2 if np.mod(settings['mean_wnd'][0], 5) != 0.0 else 0.0
                settings['mean_wnd'][3] += 1e2 if np.mod(settings['mean_wnd'][2], 5) != 0.0 else 0.0
                j += 3
            else:
                j += 1
        
        return settings
    
    def _TTBB_decode(self, val, settings, mode='BB'):
        # We assume with the following simple check that starting entries check
        # out correctly against data in TTAA
        assert int(val[1]) == settings['id'], "Invalid sounding segment"
        
        # end_i = -1
        ccount = 11
        slevels = settings.get('slevels')
        for i in range(4, len(val), 2):
            # end_i = i
            if int(val[i][:2]) != ccount:
                break;
            slevel = self._decode_level(val[i], val[i+1], '////', mode)
            slevels = np.vstack((slevels, slevel)) if slevels is not None else slevel
            ccount = (ccount % 99) + 11
        #slevels = np.vstack((settings['data'], slevels))
        #sind = np.argsort(slevels.T[0])[::-1]
        settings['slevels'] = slevels
        return settings
    
    def _PPBB_decode(self, val, settings):
        assert int(val[1]) == settings['id'], "Invalid sounding segment"
        
        ind = 2
        plevels = settings.get('sig_wnd')
        while ind + 1 < len(val) and val[ind][0] == '9':
            heights = np.full(3, np.nan)
            for j in range(3):
                heights[j] = np.nan if val[ind][j+2] =='/' else float(val[ind][j+2]) * 1e3
            heights += float(val[ind][1]) * 1e4
            heights *= 0.3048
            
            entries = np.full((3, 3), np.nan)
            ind += 1
            
            for j in range(3):
                if np.isnan(heights[j]):
                    continue
                entries[j][0] = heights[j]
                entries[j][1] = float(val[ind + j][:3]) if not '/' in val[ind + j][:3] else np.nan
                entries[j][2] = float(val[ind + j][3:]) if not '/' in val[ind + j][3:] else np.nan
            plevels = entries if plevels is None else np.vstack((plevels, entries))
            ind += 3 - np.sum(np.isnan(heights))
        settings['sig_wnd'] = plevels
        return settings
    
    def _TTCC_decode(self, val, settings):
        # Mandatory < 100 hPa data
        assert int(val[1]) == settings['id'], "Invalid sounding segment"
        ind = 2
        data = settings['data']
        
        while ind + 1 < len(val) and (int(val[ind][:2]) % 10) == 0:
            data = np.vstack((data, self._decode_level(val[ind], val[ind+1], val[ind+2], 'CC')))
            ind += 3
        settings['data'] = data
        
        # We ignore the rest of the data fields here; we are not that much
        # interested in the < 100 hPa levels for the cloud model
        return settings
    
    def _TTDD_decode(self, val, settings):
        # Significant Pressure Levels < 100 hPa
        return self._TTBB_decode(val, settings, mode='DD')
    
            
    def from_generic_sounding(self, txt):
        print("WARNING beter to use the Wyoming module; it collects more data")
        cmds = list(filter(None, re.split('\\s+', txt.replace('\n', ' '))))
        assert cmds[0] == "TTAA", "Generic sounding file must start with TTAA"
        
        # Settings
        settings = dict(data=np.empty((12, 6)))
        
        
        cmd_collected = []
        current_handler = self._TTAA_decode
        for cmd in cmds[1:]:
            if cmd == 'TTBB':
                settings = current_handler(cmd_collected, settings)
                current_handler = self._TTBB_decode
                cmd_collected = []
            elif cmd == 'PPBB':
                settings = current_handler(cmd_collected, settings)
                current_handler = self._PPBB_decode
                cmd_collected = []
            elif cmd == 'TTCC':
                settings = current_handler(cmd_collected, settings)
                current_handler = self._TTCC_decode
                cmd_collected = []
            elif cmd == 'TTDD':
                settings = current_handler(cmd_collected, settings)
                current_handler = self._TTDD_decode
                cmd_collected = []
            elif cmd == 'PPDD':
                settings = current_handler(cmd_collected, settings)
                # PPDD is the same as PPBB
                current_handler = self._PPBB_decode
                cmd_collected = []
            else:
                cmd_collected.append(cmd)
            
        settings = current_handler(cmd_collected, settings)
        # Shuffle in the significant pressure levels        
        data = np.vstack((settings['data'], settings['slevels']))
        data = data[np.argsort(data.T[0])[::-1]]
        data = data[~np.isnan(data.T[0])]
        
        # Reconstruct the geopotential height
        data[:,1] = np.interp(np.flip(data[:,0]),
                             np.flip(settings['data'][:,0]),
                             np.flip(settings['data'][:,1]))
        data[:,1] = np.flip(data[:,1])


        # Upload data
        self.__p = data[:,0]
        self.__z = data[:,1]
        self.__T = data[:,2] + 273.15 
        self.__Td = data[:,3] + 273.15   
        
        self.wyoming_info = WyomingSoundingInfo()
            
    def from_wyoming_file(self, file, raw=False):
        assert os.path.isfile(file), "The sounding file %s cannot be found" % file
        txt = None
        with open(file, "r") as f:
            txt = f.read()
        self.from_wyoming_text(txt, raw=raw)
            
    def from_wyoming_text(self, txt, raw=False):
        if raw:
            # Generic sounding file
            self.from_generic_sounding(txt)
        else:
            # Wyoming sounding human readable sounding file
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
        
        
            
    def from_wyoming_httpd(self, region, station_id, when, utc12=False, raw=False):
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
        url = None
        if raw:
            url = "&".join([region_str, "TYPE=TEXT%3ARAW", date_str, station_str])
        else:
            url = "&".join([region_str, "TYPE=TEXT%3ALIST", date_str, station_str])
        url = "http://weather.uwyo.edu/cgi-bin/sounding?" + url
        rres = requests.get(url)
        assert rres.status_code == 200, "URL %s cannot be read" % url
        txt = re.sub('<[^<]+?>', '', rres.text)
        self.from_wyoming_text(txt.replace("\\n", "\n"), raw=raw)
        
    def attach_ecmwf_field(self, file_n, fields):
        assert type(fields) is dict, "Fields must be a ECMWF to internal map"
        data = Dataset(file_n, mode='r')
        if not hasattr(self, "ecmwf_dimensions"):
            setattr(self, 'ecmwf_dimensions', (data['time'][:],
                                               data['latitude'][:],
                                               data['longitude'][:]))
        for key, field in fields.items():
            setattr(self, field, data[key][:])
        data.close()
                
        
    def sample(self, z):
        #assert(self.__z.max() + 10.0 >= z and self.__z.min() - 10.0 <= z)
        return np.interp(z, self.__z, self.__T)
    
    def sample_q(self, z):
        return np.interp(z, self.__z, self.__q)
    
            
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
    def temperature(self):
        return self.__T
       
    @property
    def height(self):
        return self.__z
    
    @property
    def RH(self):
        loopd = zip((self.__q, self.__p, self.__T))
        RH = [relative_humidity(qv, pv, Tv) for qv, pv, Tv in loopd]
        return np.array(RH)
        
    @property
    def surface_humidity(self):
        return self.__q[0]
    
    @property
    def surface_temperature(self):
        return self.__T[0]
    
    @property
    def static_energy(self):
        assert self.__T is not None
        return cpa * self.__T + gval * self.__z + Lv * self.__q
    
    @property
    def static_moist_energy(self):
        assert self.__T is not None
        return cpa * self.__T + gval * self.__z + Lv * self.__q
        
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
        
    def SkewT_logP(self, show=True, isotherms=False, dry_adiabats=True,
                   moist_adiabats=True, sat_mix=True, info_plot=True,
                   save_file=None):     
        difTm = np.abs(self.__T - self.__T[0]).max() + 10
        f = plt.figure(figsize=(7,7))
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
        ax.plot(self.__T, self.__p * 1e-2, color='C3', label='Temperature (K)')
        ax.plot(self.__Td, self.__p * 1e-2, color='C2', label='Dew point temperature (K)')
        
        atxt = [None, None]
        if self.wyoming_info.valid:
            atxt[0] = ax.text(0.13, 0.89, "%i %s %s" % (self.wyoming_info.id_num,
                                                        self.wyoming_info.id,
                                                        self.wyoming_info.observation_time.strftime("%HZ %d %b %Y")),
                    transform=f.transFigure,  weight='bold')
        if self.wyoming_info.valid and info_plot:
            atxt[1] = ax.text(-0.2, 0.5, str(self.wyoming_info),
                               transform=f.transFigure,
                               va='center', ha='left')
    
        
        # Labels
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
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
            
        if save_file and atxt:
            f.savefig(save_file, dpi=300, bbox_extra_artists=atxt, bbox_inches='tight')
        elif save_file:
            f.savefig(save_file, dpi=300, bbox_inches='tight')
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
    f, ax = snd.SkewT_logP(save_file='./KEY_01102019_sounding.png')
