#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:11:21 2019

@author: simon
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import matplotlib.animation as manimation
from netCDF4 import Dataset

from advection_mod import adv2p
import sounding as snd


# constants
gval = 9.81                     # m / s^2
Rideal = 8.31446261815324	    # J / k / mol
Rair = 287.058                  # J / k / kg
Rv = 461.5                      # J / k / kg
cpa = 1004.0                    # J / kg / K
cpv = 716.0                     # J / kg / K
Lv = 2.5e6                      # J / kg
Li = 2.834e6                    # J / kg
wdens = 997.0                   # kg / m ^ 3
kinvis_air = 1.5e-5

# Test Switches
test_collision = False
test_condensation = False
test_model = True

 

def droplet_mass(r):
    """ Returns droplet mass input is in metres output is kg
    """
    return 4. / 3. * np.pi * r * r * r * wdens
    

def prdist(N, a, b, mode='linexp'):
    """ Returns the droplet bin centre distribution in m
    
    Keyword Arguments:
        N -- Amount of bins
        a -- linear distance between bins in micrometres
        b -- Either exponential or mass dependence factor for bin distance
        mode -- type of bin grid; masmult, linexp or masdist
    """
    r = None
    
    if mode == 'massmult':
        r = np.full(N, 1e-6)
        mp = droplet_mass(r[0])
        for i in range(1, N):
            mp *= pow(2, 1. / b)
            r[i] = pow(3. / 4. * mp / np.pi / wdens, 1. / 3.)
    elif mode == 'linexp':
        fnc = lambda i : i * a + pow(10., i * b)
        r = np.array(list(map(fnc, np.arange(N)))) * 1e-6
    elif mode == 'massdist':
        r = np.zeros(N)
        mi = droplet_mass(1e-6)
        for i in range(N):
            rm = pow(3. / 4. * mi / np.pi / wdens, 1. / 3.)      
            r[i] = i * a * 1e-6 + rm
            mi *= pow(2, 1. / b)
    else:
        raise ValueError("Invalid mode to prdist " + str(mode) + " is invalid")
    
    return r

def binedges(rc):
    """ Returns the edge loctions of the bins in metres
    """
    redges = np.empty(len(rc))
    for i in range(redges.size - 1):
        redges[i] = 0.5 * (rc[i+1] + rc[i])
    redges[-1] = rc[-1] + (rc[-1] - redges[-2])
    
    ledge = max(1e-8, rc[0] - (redges[0] - rc[0]))
    ledges = np.r_[ledge, redges[:-1]]
    edges = map(lambda v: v, zip(ledges, redges))
    
    return np.array(list(edges)).flatten()

def binwidth(rc):
    """ Returns the bin widths in metres. Requires bin centres as input
    """
    edges = binedges(rc).reshape((len(rc), 2))
    return np.array(list(map(lambda v: v[1] - v[0], edges)))
    

def activition_CCN(S, Smax, CCN_C=(120.0, 0.4)):
    """ Return the number of activated CCN nuclei and the new max supersaturated state
    
    Parameters
    ----------    
    S : float
        Current super saturation
    Smax: float
        the highest found super saturation at previous times
    CCN_C: 2 element tuple
        CCN properties tuple of 2 containing fraction in mg^-1 and constant
    """
    assert isinstance(CCN_C, (tuple, list, np.ndarray)) and len(CCN_C) == 2
    if S <= Smax:
        return Smax, 0.0
    dN = CCN_C[0] * pow(100.0 * S, CCN_C[1]) -\
         CCN_C[0] * pow(100.0 * Smax, CCN_C[1])
    return S, dN * 1e6

def vdtherm(rc):
    """ Calculate droplet therminal velocity in metres / s
    Parameters
    ----------
    rc: float
        the radius of the droplet in metres
    
    Info
    ----
    Taken from Simmel et al. (2002)
    """
    a, b = 4.5795e5, 2. / 3.
    md = droplet_mass(rc) * 1e3        # Must be in grams
    tmp = 2e6 * rc
    if 134.43 < tmp <= 1511.64:
        a, b = 4.962e3, 1. / 3.
    elif 1511.64 <= tmp < 3477.84:
        a, b = 1.732e3, 1. / 6.
    elif tmp >= 3477.84:
        a, b = 9.17e2, 0.0
    return a * pow(md, b) * 1e-2

def ventilation_coef(rc):
    """ Calculate the ventilation coeficient
    """
    vt = vdtherm(rc)
    Nsh = 0.71
    Nre = 2.0 * rc * vt / kinvis_air
    fac = pow(Nsh, 1. / 3.) * pow(Nre, 0.5)
    if fac < 1.4:
        return 1.00 + 0.108 * fac * fac
    return 0.78 + 0.308 * fac

def _av_kernel(kernel, i, j):
    if i == j:
        return kernel[i,j]
    
    jm = max(j-1,0)
    im = max(i-1,0)
    jp = min(j+1,kernel.shape[0] - 1)
    ip = min(i+1,kernel.shape[1] - 1)
    av = 0.125 * (kernel[i,jm] + kernel[im,j] + kernel[ip,j] + kernel[i,jp])
    av += 0.5 * kernel[i,j]
    
    return av

def Long_colkernel(rc):
    """
    """
    K = np.empty((len(rc), len(rc)))
    vdt = np.array(list(map(vdtherm, rc)))
    for i in range(len(rc)):
        for j in range(len(rc)):
            mfac = np.pi * (rc[i] + rc[j]) ** 2 * abs(vdt[i] - vdt[j])
            Ecol = 1.0
            if rc[i] < 50.0:
                ry = rc[i] * 1e-4
                rx = rc[j] * 1e-4
                Ecol = max(4.5e4 * ry * ry * (1 - 3e-4 / rx), 1e-3)
            K[i,j] = Ecol * mfac
    return K
    #return np.pad(K, ((0,1), (0,1)), 'constant', constant_values=0.0)
    
def Golovin_colkernel(rc, b=1.5e3):
    """ Returns Golovin collection kernel 
    Keyword Arguments:
        rc -- droplet radii in micrometres
        b -- rate factor in cm^3/g/s
    Definition:
        for droplet r_y with r_x K(r_x, r_y) = b (m(r_y) + m(r_x))
    """
    K = np.empty((len(rc), len(rc)))
    for i in range(len(rc)):
        for j in range(len(rc)):
            mi = droplet_mass(rc[i])
            mj = droplet_mass(rc[j])
            K[i,j] = b * (mi + mj) * 1e-3
    return K

    
# Helper functions
def conv_n_to_g(n, r):
    tmp = list(map(lambda rv : 3.0 * droplet_mass(rv) ** 2, r))
    return n * np.array(tmp)

def conv_g_to_n(g, r):
    tmp = list(map(lambda rv : 3.0 * droplet_mass(rv) ** 2, r))
    return g / tmp

def conv_sd_to_g(sd, r):
    be = binedges(r).reshape((r.size, 2))
    dy = np.array(list(map(lambda v : np.log(v[1] / v[0]), be)))
    dr = binwidth(r)
    return sd / dy * dr * np.array([droplet_mass(rv) for rv in r])

def conv_g_to_sd(g, r):
    be = binedges(r).reshape((r.size, 2))
    dy = np.array(list(map(lambda v : np.log(v[1] / v[0]), be)))
    dr = binwidth(r)
    return g * dy / dr / np.array([droplet_mass(rv) for rv in r])

def initial_g_dist(r, r0=1e-5, L0=1e-3):
    mm = droplet_mass(r0)
    md = np.array(list(map(droplet_mass, r)))
    m0 = L0 / mm / mm
    return 3.0 * md * md * m0 * np.exp(-md / mm)
    
def calc_bott_courant(r):
    N = len(r)
    dlnr = np.log(r[1:] / r[:-1])
    dlnr = np.r_[dlnr, dlnr[-1]]
    
    
    c = np.zeros((N, N))
    ima = np.zeros((N,N), dtype=np.int)
    mcr = np.array([droplet_mass(rv) for rv in r])
    
    for i in range(N):
        for j in range(i,N):
            m0 = mcr[i] + mcr[j]
            for k in range(j, N):
                kk = -1
                if mcr[k] >= m0 and mcr[k-1] < m0:
                    if c[i,j] <= 1. - 1e-8:
                        kk = k - 1
                        c[i,j] = np.log(m0 / mcr[k-1]) / 3.0 / dlnr[i]
                    else:
                        kk = k
                        c[i,j] = 0.0
                    ima[i,j] = min(N-2,kk)
                    break
            c[j,i] = c[i,j]
            ima[j,i] = ima[i,j]
    return c, ima


def collision_ql(r, gij, ck, c, ima, dt):
    """ Returns the mass distribution of water droplets after collisione has 
    been applied to them. Method was taken from Bott (1997)
    """
    N = len(r)
    i0,il = -1,N
    gmin = 1e-60
    mcr = list(map(droplet_mass, r))
    
    # Check if cell i has got anything of value
    for i in np.arange(N - 1):
        i0 = i
        if gij[i] > gmin:
            break;
    # Check if cell j has got anything of value 
    for j in np.arange(N - 2, -1, -1):
        il = j
        if gij[j] > gmin: 
            break
    
    for i in np.arange(i0, il+1):
        for j in np.arange(i, il+1):
            k = ima[i,j]
            kp = k + 1
            m0 = ck[i,j] * gij[i] * gij[j]
            m0 = min(m0, gij[i] * mcr[j])
            m0 = m0 if j == k else min(m0, gij[j] * mcr[i])
            
            gsi = m0 / mcr[j]
            gsj = m0 / mcr[i]
            gsk = gsi + gsj
            gij[i] -= gsi
            gij[j] -= gsj
            gk = gij[k] + gsk
            
            if gk > gmin:
                m1 = np.log(gij[kp] / gk + 1e-60)
                flux = gsk / m1 * (np.exp(0.5 * m1) - np.exp(m1 * (0.5 - c[i,j])))
                flux = min(flux, gk)
                gij[k]  = gk - flux
                gij[kp] = gij[kp] + flux
    
    return gij

def smooth_kernel(kernel_type, r, dt):
    assert callable(kernel_type)
    assert isinstance(r, (list, tuple, np.ndarray))
    N = len(r)
    kern = kernel_type(r)
    ck = np.empty(kern.shape)
    dlnr = np.log(r[1:] / r[:-1])
    dlnr = np.r_[dlnr, dlnr[-1]]
    
    # Apply Smoothing
    for i in range(N):
        im = max(i-1,0)
        ip = min(i+1,N-1)
        for j in range(N):
            jm = max(j-0,0)
            jp = min(j+1,N-1)
            res = (kern[i,jm] + kern[im,j] + kern[ip,j] + kern[i,jp]) * 0.125
            res += .5 * kern[i,j]
            ck[i,j] = res
            if i == j:
                ck[i,j] *= 0.5
    del kern
    # Scale by dy and dr
    for i in range(N):
        ck[i,:] *= dt * dlnr[i]
    
    return ck
    
    
def run_collision_sim(g, r, NT, dt, kernel_type=Golovin_colkernel):
    assert callable(kernel_type)
    assert dt > 0.0
    assert NT > 0
    assert isinstance(r, (list, tuple, np.ndarray))
    
    c, ima = calc_bott_courant(r)
    ck = smooth_kernel(kernel_type, r, dt)
    
    for ni in range(NT):
        g = collision_ql(r, g, ck, c, ima, dt)
    return g

class CCNParcel(object):
    def __init__(self, T0=300.0, q0=0.0, w0=0.0, z0=0.0, mix_len=np.inf,
                 p0=1e5, bin_params=(80, 0.5, 2.0, 'massmult'),
                 ql_kernel=Golovin_colkernel):
        assert callable(ql_kernel), "Kernel must be an evaluable object"
        
        self.T  = np.array([T0])
        self.q  = np.array([q0])
        self.w  = np.array([w0])
        self.z  = np.array([z0])
        self.fl = np.array([np.zeros(bin_params[0])])
        self.__p0 = p0
        self.__dt = 0.0
                
        self.rbin = prdist(*bin_params)
        self.mbin = np.array([droplet_mass(r) for r in self.rbin])
        self.qlkern_fnc = ql_kernel
        
        self.entrainment = 0.0 if mix_len < 1.0 else 1. / mix_len
        self.induced_mass = 0.5
        
    @property
    def reduced_gravity(self):
        return gval / (1. + self.induced_mass)
    
    @property
    def p(self):
        pv = np.full(self.T.size, self.__p0)
        for i in range(1, pv.size):
            trpv = np.trapz(1. / self.T[:i], self.z[:i])
            pv[i] = self.__p0 * np.exp(-gval / Rair * trpv)
        return pv
    
    @p.setter
    def p(self, p0):
        assert p0 > 0.0, "Surface pressure must be non-zero"
        self.__p0 = p0
        
    def run(self, dt, NT, environ):
        assert isinstance(environ, snd.Sounding)
        assert NT > 0
        assert dt > 0.0
        
        # Create space for distributions
        self.T  = np.full(NT, self.T[-1])
        self.q  = np.full(NT, self.q[-1])
        self.w  = np.full(NT, self.w[-1])
        self.z  = np.full(NT, self.z[-1])
        self.fl = np.full((NT, self.fl.size), self.fl[-1])
        
        # Internal temp variables
        bwidth = -1
        smax = 0.0
        acc = self.reduced_gravity
        went = self.entrainment / (1. + self.induced_mass)
        pe = environ.pressure_profile()
        ze = environ.height
        dr = binwidth(self.rbin)
        
        # Kernel Calculation for Collision/Coalescence
        qlkern = smooth_kernel(self.qlkern_fnc, self.rbin, dt)
        qlc, qlima = calc_bott_courant(self.rbin)
        
        # Condensation/Evaporation (ql)
        vent_ql = np.array(list(map(ventilation_coef, self.rbin)))
        
        self.__dt = dt
                
        # Run simulation
        for i in range(1, NT):
            # Grab state variable
            Te = environ.sample(self.z[i-1])
            qe = environ.sample_q(self.z[i-1])
            wl = np.trapz(self.mbin * self.fl[i-1], self.rbin)
            pf = np.interp(self.z[i-1], ze, pe)
            qs = snd.saturation_mixr(self.T[i-1], pf)
            S = self.q[i-1] / qs - 1.0
            
            # Momentum Equation(s)
            dw_dt  = acc * ((self.T[i-1] - Te) / Te - wl) 
            dw_dt -= went * abs(self.w[i-1]) * self.w[i-1]
            
            # Cloud Microphysics (Droplet Spectrum)
            tmp_fl = self.fl[i-1] 
            ## CCN
            smax, dNCNN = activition_CCN(S,smax)
            tmp_fl[0] = tmp_fl[0] + dNCNN / dr[0]
            
            
            ## Collision/Coalesence
            tmp_fl = conv_sd_to_g(tmp_fl, self.rbin)
            tmp_fl = collision_ql(self.rbin, tmp_fl, qlkern, qlc, qlima, dt)
            tmp_fl = conv_g_to_sd(tmp_fl, self.rbin)
            
            ## Condensation
            drl_dt = vent_ql * 1e-10 * S / self.rbin
            cr_ql = drl_dt * dt / dr
            tmp_fl2 = adv2p(np.copy(tmp_fl), cr_ql)      
            eps1 = np.trapz(self.mbin * (tmp_fl2 - tmp_fl) / dt, self.rbin)
            
            ## Entrainment of liquid water
            tmp_fl2 = tmp_fl2 - self.entrainment*tmp_fl2*abs(self.w[i-1])*dt
            
            # Thermodynamic Equations
            dT_dt  = -gval / cpa * self.w[i-1] + Lv / cpa * eps1
            dT_dt -= self.entrainment * (self.T[i-1] - Te) * abs(self.w[i-1])
            dq_dt = -eps1 - self.entrainment*(self.q[i-1]-qe)*abs(self.w[i-1])
            
            # Update
            self.w[i] = self.w[i-1] + dw_dt * dt
            self.fl[i] = tmp_fl2
            self.T[i] = self.T[i-1] + dT_dt * dt
            self.q[i] = self.q[i-1] + dq_dt * dt
            self.z[i] = self.z[i-1] + self.w[i-1] * dt
            
            prog = (i + 1.) / float(NT)
            nbwidth = int(50 * prog)
            if nbwidth > bwidth:
                pbar = ['#'] * nbwidth
                nbar = [' '] * (50 - nbwidth)
                ipr = int(prog * 100.0)
                bar = '\r|' + ''.join(pbar) + ''.join(nbar) + '| %03i%%' % ipr
                sys.stdout.write(bar)
                sys.stdout.flush()
                bwidth = nbwidth
                
        sys.stdout.write("\nDone!\n")
        return self.T
    
    def write_to_netCDF(self, file_name):
        assert isinstance(file_name, str), str(file_name)+" must be file name."
        data = Dataset(file_name, mode='w')
        
        data.createDimension("time", None)
        data.createDimension("rbin", self.rbin.size)
        
        vtim = data.createVariable("time", "f4", ("time",))
        vrbn = data.createVariable("rbin", "f8", ("rbin",))
        vT = data.createVariable("T", "f4", ("time",))
        vq = data.createVariable("q", "f4", ("time",))
        vw = data.createVariable("w", "f4", ("time",))
        vz = data.createVariable("z", "f4", ("time",))
        vfl = data.createVariable("fl", "f4", ("time","rbin"))
        
        vtim.units = 'seconds'
        vrbn.units = 'meters'
        vT.units = 'Kelvin'
        vq.units = 'kg kg**-1'
        vw.units = 'm s**-1'
        vz.units = 'meters'
        vfl.units = 'kg m**-3'
            
        data['time'][:] = np.arange(self.T.size) * self.__dt
        data['rbin'][:] = self.rbin
        data['T'][:] = self.T
        data['q'][:] = self.q
        data['w'][:] = self.w
        data['z'][:] = self.z
        data['fl'][:] = conv_sd_to_g(self.fl, self.rbin)
        
        data.close()
            
    
        


# Plot Distributions
f,ax = plt.subplots()
ax.semilogx(prdist(69,  0.25, 0.055)  * 1e6, [1] * 69, 'x')
ax.semilogx(prdist(120, 0.125, 0.032) * 1e6, [2] * 120, 'x')
ax.semilogx(prdist(200, 0.075, 0.019) * 1e6, [3] * 200, 'x')
ax.semilogx(prdist(300, 0.05, 0.0125) * 1e6, [4] * 300, 'x')

ax.semilogx(prdist(40,  1.0, 1.0, mode='massdist') * 1e6, [6] * 40, 'x')
ax.semilogx(prdist(80, 0.5, 2.0, mode='massdist') * 1e6, [7] * 80, 'x')
ax.semilogx(prdist(160, 0.25, 4.0, mode='massdist') * 1e6, [8] * 160, 'x')
ax.semilogx(prdist(320, 0.125, 8.0, mode='massdist') * 1e6, [9] * 320, 'x')

ax.text(1000., 8.4, "N=320") 
ax.text(1000., 7.4, "N=160")
ax.text(1000., 6.4, "N=80")
ax.text(1000., 5.4, "N=40")

ax.text(1000., 3.4, "N=300")
ax.text(1000., 2.4, "N=200")
ax.text(1000., 1.4, "N=120")
ax.text(1000., 0.4, "N=69")


ax.set_xlabel("Radius ($\\mu$m)")
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([])
ax.grid()
ax.set_ylim([0, 10])
plt.show()

# Plot Collision Kernel
R = prdist(80, 0.5, 2, mode='massdist')
XV, YV = np.meshgrid(R * 1e6, R * 1e6) 
f, ax = plt.subplots()
cb = ax.contourf(XV, YV, Golovin_colkernel(R) * 1e6,
                 levels=[0., 10, 50, 100, 300, 500, 1000,1500], extend='max')
ax.set_yscale('log')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel("$r_x$ ($\\mu$m)")
ax.set_ylabel("$r_y$ ($\\mu$m)")
cax = plt.colorbar(cb)
cax.set_label("cm$^3$/s")
plt.show()

if test_model:
    def trial_lapse_rate(z):
        Tv = 0.0
        Td = 0.0
        if 0.0 <= z < 1500.0:
            Tv = -10.0e-3
            Td = 1e-4
        elif 1500.0 <= z < 10000.0:
            Tv = -6.5e-3
            Td = -8e-3
        else:
            Tv = 1e-3
            Td = -9.8e-4
        return {'T' : Tv, 'Td' : Td}
    sounding = snd.Sounding(None, None)
    sounding.from_lapse_rate(trial_lapse_rate, 0, 2e4, 10000)
    
    # Setup sim
    parcel = CCNParcel(T0=sounding.surface_temperature + 1.0,
                       q0=0.01, mix_len=np.inf)
    parcel.run(0.02, 42500, sounding)
    
    f, ax = sounding.SkewT_logP(show=False)
    ax.plot(parcel.T, parcel.p * 1e-2)
    plt.show()

# Test  Collision Growth Module
if test_collision:
    R = prdist(80, 0.0, 2.0, mode='massmult')
    rm = 1e-5 # In metres
    Lc = 1e-3 # Cloud liquid content in kg / m^-3
    gd = initial_g_dist(R, rm, Lc)
    
    f, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Initial Distribution
    ax[0].semilogx(R * 1e6, gd * 1e3, 'C1')
    ax[1].loglog(R * 1e6, gd * 1e3, 'C1')
    maxi = np.argmax(gd)
    ax[0].text((R * 1e6)[maxi], (gd * 1e3)[maxi], '0')
    ax[1].text((R * 1e6)[maxi], (gd * 1e3)[maxi], '0')
    
    
    # +30 min distribution
    gd = run_collision_sim(gd, R, 1800, 1.0)
    ax[0].semilogx(R * 1e6, gd * 1e3, 'C2')
    ax[1].loglog(R * 1e6, gd * 1e3, 'C2')
    maxi = np.argmax(gd)
    ax[0].text((R * 1e6)[maxi], (gd * 1e3)[maxi], '30')
    ax[1].text((R * 1e6)[maxi], (gd * 1e3)[maxi], '30')
    
    # +60 min distribution
    gd = run_collision_sim(gd, R, 1800, 1.0)
    ax[0].semilogx(R * 1e6, gd * 1e3, 'C3')
    ax[1].loglog(R * 1e6, gd * 1e3, 'C3')
    maxi = np.argmax(gd)
    ax[0].text((R * 1e6)[maxi], (gd * 1e3)[maxi], '60')
    ax[1].text((R * 1e6)[maxi], (gd * 1e3)[maxi], '60')
    
    
    
    ax[0].set_xlabel("r ($\\mu$m)")
    ax[0].set_ylabel("g(g m$^{-3}$)")
    ax[0].grid(True)
    ax[0].xaxis.set_major_formatter(ScalarFormatter())
    ax[0].set_xlim(R.min() * 1e6, R.max() * 1e6)
    
    ax[1].grid(True)
    ax[1].set_xlabel("r ($\\mu$m)")
    ax[1].set_ylabel("g(g m$^{-3}$)")
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].set_xlim(R.min() * 1e6, R.max() * 1e6)
    ax[1].set_ylim(1e-10, 5)
    f.savefig("./water_droplet_dist_growth.png", dpi=300)
    plt.show()

if test_condensation:
    def satmix_r(T,p):
        es = 611.2 * np.exp(Lv / Rv / 273.16  * (1 - 273.16 / T))
        return es / p * Rair / Rv
    
    # Create Initial vapour distribution
    R = prdist(80, 0.5, 2.0, mode='massmult')
    rm = 1e-5 # In metres
    Lc = 1e-3 # Cloud liquid content in kg / m^-3
    ql = 0.020 # Water Vapour in the cloud
    T,p = (295.0, 1e5) # Environmental Temperature and pressure
    dt = 0.02
    nt = 500
    
    ql = np.full(nt * 2, ql)
    mcr = np.array(list(map(droplet_mass, R)))
    sd = conv_g_to_sd(initial_g_dist(R, rm, Lc), R)
    sd = np.full((nt * 2, R.size), sd)
    dr = binwidth(R)
    vent_c = np.array([ventilation_coef(rv) for rv in R])
    
    # Run simulation
    for i in range(1, 2 * nt):
        if i == nt:
            T = 298.15 # At this temperature 0.02 kg/kg is the sat. mix. ratio
        
        dr_dt = vent_c * 1e-10 * (ql[i-1] / satmix_r(T,p) - 1.0) / R
        courn = dr_dt * dt / dr
        
        sd[i] = adv2p(np.copy(sd[i-1]), courn)
        cond_t = (sd[i] - sd[i-1]) / dt
        Cr = np.trapz(mcr * cond_t, R)
        ql[i] = ql[i-1] - Cr * dt
    
    f,ax = plt.subplots()
    ax.semilogx(R * 1e6, sd[0], 'k--', alpha=0.7)
    ln, = ax.semilogx(R * 1e6, sd[0])
    txt = ax.text(0.13, 0.85, "$t=%.1f$ s" % 0.0, transform=f.transFigure)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid()
    ax.set_xlabel("r ($\\mu$m)")
    ax.set_ylabel("$\\phi$")
    ax.set_xlim(R.min() * 1e6, R.max() * 1e6)
    ax.set_ylim(0, 8e13)
        
    def update_frame(i, data):
        ln.set_ydata(data[i])
        txt.set_text("$t=%.1f$ s" % (i * dt))
        return ln,txt
    
    # Create Animation    
    wtype = manimation.writers['ffmpeg']
    writer = wtype(fps=24, bitrate=1800)
    
    ani = manimation.FuncAnimation(f, update_frame,
                                   frames=np.arange(0, 2*nt, 5),
                                   interval=24, blit=True, fargs=(sd,))
    ani.save("./anims/ql_evap_condens_test.mp4", writer=writer, dpi=250)
    plt.close(f)
    
    
    
    # Create Water vapour content plot
    f,ax = plt.subplots()
    ax.plot(np.arange(2 * nt) * dt, ql * 1e3)
    ax.plot([nt * dt] * 2, [min(ql) * 1e3, max(ql) * 1e3], 'k--')
    ax.grid()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$q_v$ (g/kg)")
    ax.set_xlim(0, 2 * nt * dt)
    f.savefig("./images/ql_evap_condens_test.png", dpi=300)
    plt.show()