# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:38:52 2019

@author: Simon
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['adv1', 'adv2p', 'adv4p']

def top_hat(x, c=1.0, xmin=-.25, xmax=.25):
    if xmin <= x <= xmax:
        return c
    return 0.0


def adv1(f, c, cyclic=False):
    """ Upstream advection equation solver for a single time step
    
    Parameters
    ----------
    f : array_like
        the 1D data field which is advected
    c : array_like
        the courant numbers at the grid points of f forcing the advection
    cyclic : scalar, optional
        Whether the domain on which f exists is cyclic
    Returns
    -------
    f : np.ndarray
        Advected field f at time step f^n+1
    """
    assert len(f) == len(c)
    N = len(f)
    fm = np.zeros(N)
    fp = np.zeros(N)
    
    for i in range(N-1):
        fm[i] = -min(0.0, c[i]) * f[i+1]
        fp[i] =  max(0.0, c[i]) * f[i]
      
    if cyclic:
        fm[-1] = -min(0.0, c[-1]) * f[0]
        fp[-1] =  max(0.0, c[-1]) * f[-1]
     
    for i in range(0 if cyclic else 1,
                   N if cyclic else N-1):
        f[i] = f[i] - fm[i-1] + fp[i-1] + fm[i]- fp[i]

    return f

def adv2p(f, c, cyclic=False):
    """ Second order definite positive upstream advection solver.
    
    Method for solving an 1D advection problem. Method was taken from Bott
    (1988).
    
    Parameters
    ----------
    f : array_like
        the 1D data field which is advected
    c : array_like
        the courant numbers at the grid points of f forcing the advection
    cyclic : scalar, optional
        Whether the domain on which f exists is cyclic
    Returns
    -------
    f : np.ndarray
        Advected field f at time step f^n+1
    """
    assert len(f) == len(c)
    if cyclic:
        f = np.pad(f, (1,1), 'constant', constant_values=(f[-1], f[0]))
        c = np.pad(c, (1,1), 'constant', constant_values=(c[-1], c[0]))
    
    N = len(c)
    
    fp = np.zeros(N)
    fm = np.zeros(N)
    w = np.zeros(N)
    
    # Left most value
    cr = max(c[0], 0.0)
    fp[0] = min(f[0], cr * (f[0] + (1 - cr) * (f[1] - f[0]) * .5 ))
    w[0] = 1.0
    for i in range(1, N-1):
        a0 = (26 * f[i] - f[i+1] - f[i-1]) / 24.
        a1 = (f[i+1] - f[i-1]) / 16.
        a2 = (f[i+1] - 2. * f[i] + f[i-1]) / 48.
        
        cl = -min(0., c[i-1])
        x1 = 1 - 2 * cl
        x2 = x1 * x1
        fm[i-1] = max(0., a0 * cl - a1 * (1 - x2) + a2 * (1. - x1 * x2))
        
        cr = max(0.0, c[i])
        x1 = 1. - 2. * cr
        x2 = x1 * x1
        fp[i] = max(0., a0 * cr + a1 * (1 - x2) + a2 * (1. - x1 * x2))
        
        w[i] = f[i] / max(fm[i-1] + fp[i] + 1e-20, a0 + 2 * a2)
        
    # Right most value
    cl = -min(0.0, c[-2])
    fm[-2] = min(f[-1], cl * (f[-1] - (1. - cl) * (f[-1] - f[-2]) * .5))
    w[-1] = 1.
    
    for i in range(1, N-1):
        f[i] = f[i] - (fm[i-1] + fp[i]) * w[i] + fm[i]*w[i+1] + fp[i-1]*w[i-1]
        
    if cyclic:
        c = c[1:-1]
        return f[1:-1]
    return f

def adv4p(f, c, cyclic=False):
    """ Fourth order definite positive upstream advection solver.
    
    Method for solving an 1D advection problem. Method was taken from Bott
    (1988).
    
    Parameters
    ----------
    f : array_like
        the 1D data field which is advected
    c : array_like
        the courant numbers at the grid points of f forcing the advection
    cyclic : scalar, optional
        Whether the domain on which f exists is cyclic
    Returns
    -------
    f : np.ndarray
        Advected field f at time step f^n+1
    """
    assert len(f) == len(c)
    if cyclic:
        f = np.pad(f, (1,1), 'constant', constant_values=(f[-1], f[0]))
        c = np.pad(c, (1,1), 'constant', constant_values=(c[-1], c[0]))
    
    N = len(c)    
    fp = np.zeros(N)
    fm = np.zeros(N)
    w = np.zeros(N)
    
    # Two left entries
    cr = max(0.0, c[0])
    fp[0] = min(f[0], cr * (f[0] + (1. - cr) * (f[1] - f[0]) * .5))
    w[0] = 1.0
    
    a0 = (26 * f[1] - f[2] - f[0]) / 24.
    a1 = (f[2] - f[0]) / 16.
    a2 = (f[2] + f[0] - 2 * f[1]) / 48.
    cl = -min(0.0, c[0])
    x1 = 1. - 2. * cl
    x2 = x1 * x1
    fm[0] = max(0., a0 * cl - a1 * (1 - x2) + a2 * (1 - x1 * x2))
    cr = max(0., c[1])
    x1 = 1. - 2. * cr
    x2 = x1 * x1
    fp[1] = max(0., a0 * cr + a1 * (1. - x2) + a2 * (1 - x1 * x2))
    w[1] = f[1] / max(fm[0] + fp[1] + 1e-20, a0 + 2. * a2)
    
    # Centre
    for i in range(2, N-2):
        a0 = (9*(f[i+2]+f[i-2]) - 116*(f[i+1]+f[i-1]) + 2134*f[i]) / 1920.
        a1 = (-5*(f[i+2]-f[i-2]) + 34*(f[i+1]-f[i-1])) / 384.
        a2 = (-f[i+2]+12.*(f[i+1]+f[i-1])-22.*f[i]-f[i-2]) / 384.
        a3 = (f[i+2]-2*(f[i+1]-f[i-1])-f[i-2]) / 768.
        a4 = (f[i+2]-4*(f[i+1]+f[i-1])+6.*f[i]+f[i-2]) / 3840.
        
        cl = -min(0.0, c[i-1])
        x1 = 1. - 2. * cl
        x2 = x1 * x1
        x3 = x1 * x2
        fm[i-1] = max(0., a0 * cl - a1 * (1 - x2) + a2 * (1. - x3) \
                        - a3 * (1. - x1 * x3) + a4 * (1. - x2 * x3))
        
        cr = max(0., c[i])
        x1 = 1. - 2. * cr
        x2 = x1 * x1
        x3 = x1 * x2
        fp[i] = max(0., a0 * cr + a1 * (1 - x2) + a2 * (1. - x3) \
                      + a3 * (1. - x1 * x3) + a4 * (1. - x2 * x3))
        
        w[i] = f[i] / max(fm[i-1] + fp[i] + 1e-20, f[i])
        
    # Right Edge
    a0 = (26 * f[-2] - f[-1] - f[-3]) / 24.
    a1 = (f[-1] - f[-3]) / 16.
    a2 = (f[-1] + f[-3] - 2 * f[-2]) / 48.
    cl = -min(0.0, c[-3])
    x1 = 1. - 2. * cl
    x2 = x1 * x1
    fm[-3] = max(0., a0 * cl - a1 * (1. - x2) + a2 * (1. - x1 * x2))
    cr = max(0.0, c[-2])
    x1 = 1. - 2. * cr
    x2 = x1 * x1
    fp[-2] = max(0., a0 * cr + a1 * (1. - x2) + a2 * (1. - x1 * x2)) 
    w[-2] = f[-2] / max(fm[-3] + fp[-2] + 1e-20, a0 + 2. * a2)
    
    cl = -max(0., c[-1])
    fm[-1] = min(f[-1], cl * (f[-1] - (1. - cl) * (f[-1] - f[-2]) * .5))
    w[-1] = 1.0
    
    for i in range(1, N-1):
        f[i] = f[i] - (fm[i-1]+fp[i])*w[i] + fm[i]*w[i+1] + fp[i-1]*w[i-1]
        
    if cyclic:
        c = c[1:-1]
        return f[1:-1]
    return f
    
# Test section
if __name__ == '__main__':
    dt = 0.1
    nt = 1200
    x = np.linspace(-1, 1, 1000)
    y = np.array([top_hat(xv, xmin=-.25, xmax=.25) for xv in x])
    ya = np.copy(y)
    v =  np.full(x.size, 2.0 / (dt * nt))
    # Calculate dx as the width of the cell
    dx = np.empty(x.size)
    for i in range(1, x.size-1):
        dx[i] = 0.5 * (x[i+1] - x[i]) + 0.5 * (x[i] - x[i-1])
    # Cyclic space
    dx[0]  = dx[1]
    dx[-1] = dx[-2]
    c = v * dt / dx
    
    for i in range(nt):
        ya = adv4p(ya, c, True)
        
    # Fourier Analysis
    fy, fya = (2 / y.size * np.abs(np.fft.fft(y)),
               2 / y.size * np.abs(np.fft.fft(ya)))
    frq = np.linspace(0.0, 0.5 / dx[1], y.size // 2)
    
    
    f, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].plot(x, y, 'C0--')
    ax[0].plot(x, ya, 'C1')
    ax[0].text(0.13, 0.84, "MSE : %.1f" % np.sum((y -ya) ** 2),
               transform=f.transFigure)
    ax[1].plot(frq[:y.size//2], fy[:y.size//2], 'C0--')
    ax[1].plot(frq[:y.size//2], fya[:y.size//2], 'C1')
    
    ax[0].grid(True)
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Concentration")
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(0, 1.2)
    ax[1].grid(True)
    ax[1].set_xlabel("1/dx")
    ax[1].set_ylabel("Spectral Power")
    ax[1].set_xlim(0, 20)
    plt.show()
