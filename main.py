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
from mpl_toolkits.basemap import Basemap


np.random.seed(10)
time_axis = 0


# mod = xr.open_dataset('MOD09A1.A2003001.h10v05.006.2015153105208.hdf', engine='netcdf4')
# print(mod.sur_refl_b01)
# print(np.array(mod.sur_refl_b01).shape)

# data = np.array(mod.sur_refl_b01)
# df = pd.read_csv('h10v05_raster_to_p.txt')
# tile = 'h10v05.npy'
# lon = np.array(df.POINT_X).reshape(2400, 2400)
# lat = np.array(df.POINT_Y).reshape(2400, 2400)
# cell_size = 0.005
# #print(data)
# data3d = np.dstack([data]*3)
# #data3d = np.random.rand(2400, 2400, 3)
# data3d = np.moveaxis(data3d, -1, time_axis)





# nldas = xr.open_dataset('NLDAS_FORA0125_H.A20000101.0000.002.grb.SUB.nc4', engine='netcdf4')
# data = np.array(nldas.TMP)[0, 0]
# lat = np.array(nldas.lat)
# lon = np.array(nldas.lon)
# cell_size = 0.125
# data3d = np.dstack([data]*3)
# #data3d = np.random.rand(224, 464, 3)
# data3d = np.moveaxis(data3d, -1, time_axis)




da_mask = h5py.File(f'AMSR_U2_L3_DailySnow_B02_20230330.he5','r')
cell_size = 0.3
lat = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lat'))
lon = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lon'))
data = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernDaily'))
lon = np.where(lon>10000, 0, lon)
lat = np.where(lat>10000, 0, lat)

data3d = np.dstack([data]*3)
data3d = np.random.rand(721, 721, 3)
data3d = np.moveaxis(data3d, -1, time_axis)


shp_path = 'shpfiles/for_amsr.shp'
## TODO: name should change, instantiate with ClipRaster is wierd
r1 = ClipRaster(data3d, lat, lon, cell_size)

scale_factor = 1

s=time.time()   
for i in range(1):
    r1_cliped, lat, lon = r1.clip3d(shp_path, time_axis, drop=True, scale_factor=scale_factor)
print("time:", (time.time()-s))

# print('mean=', r1.get_mean3d('shpfiles/small_basin.shp', time_axis, scale_factor=scale_factor))

# r2 = ClipRaster(data3d[0, :, :], lat, lon, 0.005)
# print('mean=', r2.get_mean2d('shpfiles/small_basin.shp', scale_factor=scale_factor))


#plt.imshow(r1_cliped[1, :, :])
#plt.colorbar()

#lon[lon < 0] += 360
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=50.22, urcrnrlat =50.589,
            llcrnrlon=-103.286, urcrnrlon =-102.351)   

shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
							   linewidth=1,color='r')             
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(np.floor(np.min(lat)), np.ceil(np.max(lat)), .25),
                labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(np.floor(np.min(lon)), np.ceil(np.max(lon)), .25),
                labels=[0, 0, 0, 1])

m.pcolormesh(lon, lat, r1_cliped[0, :, :], latlon=True)



















