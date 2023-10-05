# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:06:12 2023

@author: kgavahi
"""

#test timing
import numpy as np
import shapefile
from matplotlib.path import Path



x = np.arange(10*7).reshape(10, 7)


mask = ((x>15) & (x<19)) | ((x>22) & (x<26)) |  ((x>29) & (x<33))
mask[1, 3] = True
mask[3, 1] = True
mask[5, 3] = True
mask[3, 5] = True

print(index[mask.flatten()])





x[:, ~nan_cols][~nan_rows] = 1000