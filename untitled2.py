# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:09:09 2023

@author: kgavahi
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np





x = xr.Dataset(
    {
        "temperature": ("station", 20 * np.random.rand(4)),
        "pressure": ("station", 500 * np.random.rand(4)),
    },
    coords={"station": ["boston", "nyc", "seattle", "denver"]},
)

new_index = ["boston", "austin", "seattle", "lincoln"]
y = x.reindex({"station": new_index})