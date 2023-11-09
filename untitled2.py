# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:09:09 2023

@author: kgavahi
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np






ds1 = xr.Dataset({"a": ("x", [10, 20, 30, np.nan])}, {"x": [1, 2, 3, 4]})

ds2 = xr.Dataset({"b": ("x", [np.nan, 30, 40, 50])}, {"x": [2, 3, 4, 5]})

xr.merge([ds1, ds2])
