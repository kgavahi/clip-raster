# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:02:31 2023

@author: kgavahi
"""

from clipraster import ClipRaster
import h5py  
import numpy as np
import time
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
''' -------------Dataset Specific------------------- '''
#Load your original 25km xarray dataset
da_mask = h5py.File(f'AMSR_U2_L3_DailySnow_B02_20230330.he5','r')
cell_size = 0.3
lat = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lat'))
lon = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lon'))
swe = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernDaily'))
''' ------------------------------------------------ '''

'''
s=time.time()        
r1 = ClipRaster(swe, lat, lon, .3)
print(time.time()-s)
#mask = r1.mask_shp('Export_Output_2.shp')

s=time.time()   
mean = r1.get_mean('Export_Output_2.shp')
print(time.time()-s)

s=time.time()   
mean = r1.clip('Export_Output_2.shp')
print(time.time()-s)'''


# da = xr.open_dataset('us_ssmv11034tS__T0001TTNATS2003100105HP001.nc')
# swe = np.array(da.Band1)
# lat = np.array(da.lat)
# lon = np.array(da.lon)


## TODO: name should change, instantiate with ClipRaster is wierd
r1 = ClipRaster(swe, lat, lon, 0.3)




s=time.time()   
r1_cliped = r1.clip('hysets_06469400.shp', drop=True, scale_factor=1)


print('mean=', r1.get_mean('hysets_06469400.shp', scale_factor=1))

print("time:", time.time()-s)


mask = r1.mask_shp('hysets_06469400.shp')

np.savetxt('r1_cliped_slow.csv', np.flip(np.flip(r1_cliped), axis=1), delimiter=',')


aa

np.savetxt('r1_cliped_slow.csv', np.flip(np.flip(r1_cliped), axis=1), delimiter=',')


plt.imshow(r1_cliped)
plt.colorbar()

# =============================================================================
# x = np.arange(10*10*3).reshape(3,10,10)
# 
# m = np.arange(10*10).reshape(10,10)
# mask = m>5
# 
# #t = x[mask[..., None]]
# 
# 
# xx = mask[None, ...]
# b = np.dstack([mask]*3)
# b = np.tile(mask,(3, 1,1))
# 
# t = x[:, mask]
# 
# da = xr.DataArray(
#     x,
# 
# )
# 
# da_m = xr.DataArray(
#     mask,
# 
# )
# =============================================================================







