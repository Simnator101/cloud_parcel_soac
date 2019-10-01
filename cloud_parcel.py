#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:40:54 2019

Cloud parcel module class
"""

import numpy as np
import sounding

class CloudParcel(object):
    def __init__(self, T0):
        self.__temp = T0