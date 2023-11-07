# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:09:09 2023

@author: kgavahi
"""

import xarray as xr
import matplotlib.pyplot as plt


grbs = xr.open_dataset('gfs_4_20110905_0600_000.grb2', engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'})


print(grbs)


