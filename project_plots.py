#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:41:34 2019

@author: simon
"""

import matplotlib.pyplot as plt
import numpy as np


# Vapour flow
RH = np.linspace(0.9, 1.1, 100)
block_flw = np.zeros(RH.size)
for Rv, i in enumerate(RH):
    if Rv > 1.05:
        block_flw[i] = 1.0 - R


f, ax = plt.subplots()
ax.plot(RH * 100.0, -0.2 * np.tanh(30.0 * (RH - 1.0)), label='tanh')
ax.grid(True)
ax.set_xlabel("Relative Humidity (%)")
ax.set_ylabel("Flow speed (1/s)")
ax.text(0.13, 0.9, "$\\Delta t$: 1.0 s", transform=f.transFigure)
f.savefig("./images/tanh_vapour.png", dpi=300)
plt.show()