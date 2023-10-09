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



mod = xr.open_dataset('MOD09A1.A2003001.h10v05.006.2015153105208.hdf')
data = np.array(mod.sur_refl_b01)
df = pd.read_csv('h10v05_raster_to_p.txt')
tile = 'h10v05.npy'
lon = np.array(df.POINT_X).reshape(2400, 2400)
lat = np.array(df.POINT_Y).reshape(2400, 2400)




## TODO: name should change, instantiate with ClipRaster is wierd
r1 = ClipRaster(data, lat, lon, 0.005)



s=time.time()   
for i in range(1):
    r1_cliped = r1.clip('shpfiles/ACF_basin.shp', drop=True, scale_factor=1)
print("time:", time.time()-s)

print('mean=', r1.get_mean('shpfiles/ACF_basin.shp', scale_factor=1))


plt.imshow(r1_cliped)

mask = r1.mask_shp('hysets_06469400.shp')

np.savetxt('r1_cliped_slow.csv', np.flip(np.flip(r1_cliped), axis=1), delimiter=',')










