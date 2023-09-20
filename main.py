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
''' -------------Dataset Specific------------------- '''
# Load your original 25km xarray dataset
#da_mask = h5py.File(f'AMSR_U2_L3_DailySnow_B02_20230330.he5','r')
#cell_size = 0.3
#lat = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lat'))
#lon = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lon'))
#swe = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernDaily'))
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


da = xr.open_dataset('us_ssmv11034tS__T0001TTNATS2017112005HP001.nc')
swe = np.array(da.Band1)
lat = np.array(da.lat)
lon = np.array(da.lon)

r1 = ClipRaster(swe, lat, lon, 0.015)

s=time.time()   
mean = r1.clip('hysets_01135300.shp')
print(mean)
print(time.time()-s)
