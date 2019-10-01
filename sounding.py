#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:43:28 2019

@author: simon
"""

import numpy as np

class Sounding(object):
    def __init__(self, p, T):
        if (p is None):
            assert(p == T)
        self.__p = p
        self.__T = T
            
    def add_points(self, p, T):
        p = np.array(p, dtype=np.float)
        T = np.array(T, dtype=np.float)
        self.__T = np.r_[self.__T, T] if self.__T is not None else T
        self.__p = np.r_[self.__p, p] if self.__p is not None else p
        