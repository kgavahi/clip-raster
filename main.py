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
import shapefile

np.random.seed(10)
time_axis = 2

shp_path = 'shpfiles/ACF_basin.shp'
shp = shapefile.Reader(shp_path)

# Get the polygon vertices of the basin
tupVerts = shp.shapes()[0].points
tupVerts_np = np.array(tupVerts)
up = np.max(tupVerts_np[:, 1])
down = np.min(tupVerts_np[:, 1])
left = np.min(tupVerts_np[:, 0])
right = np.max(tupVerts_np[:, 0])




# ds2011_2014 = xr.open_mfdataset('precip.V1.0.*.nc', concat_dim='time', combine='nested')
# data = np.array(ds2011_2014.to_array())
# lat = np.array(ds2011_2014.lat)
# lon = np.array(ds2011_2014.lon)-360
# cell_size = 0.25


# r_cpc = ClipRaster(data, lat, lon, cell_size)
# r_mean = r_cpc.get_mean(shp_path, scale_factor=1)


# weights, landmask = r_cpc.mask_shp(shp_path, scale_factor=100)


# dataplot = ds2011_2014.where(landmask)

# xr_mean = np.array(dataplot.mean(dim=('lat', 'lon')).to_array())


# ds2011_2014_w = ds2011_2014 * weights

# wxr_mean = np.array(ds2011_2014_w.sum(dim=('lat', 'lon')).to_array())



# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-.1, urcrnrlat =up+.1,
#             llcrnrlon=left-.1, urcrnrlon =right+.1)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# #pcolormesh = m.pcolormesh(lon, lat, data[0, 1, :], latlon=True, cmap='terrain_r')
# pcolormesh = m.pcolormesh(lon, lat, dataplot.precip[1], latlon=True, cmap='terrain_r')






# mod = xr.open_dataset('MOD09A1.A2003001.h10v05.006.2015153105208.hdf', engine='netcdf4')

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



nldas = xr.open_dataset('NLDAS_FORA0125_H.A20000101.0000.002.grb.SUB.nc4', engine='netcdf4')
data = np.array(nldas.to_array())
lat = np.array(nldas.lat)
lon = np.array(nldas.lon)
cell_size = 0.125
data3d = np.dstack([data]*3)
#data3d = np.random.rand(224, 464, 3)
data3d = np.moveaxis(data3d, -1, time_axis)


sf=100

r_nldas = ClipRaster(data, lat, lon, cell_size)
r_mean = r_nldas.get_mean(shp_path, scale_factor=sf).ravel()
weights, landmask = r_nldas.mask_shp(shp_path, scale_factor=sf)


dataplot = nldas.where(landmask)

xr_mean = np.array(dataplot.mean(dim=('lat', 'lon')).to_array()).ravel()


nldas_w = nldas * weights

wxr_mean = np.array(nldas_w.sum(dim=('lat', 'lon')).to_array()).ravel()


m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=down-.1, urcrnrlat =up+.1,
            llcrnrlon=left-.1, urcrnrlon =right+.1)    

shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
							   linewidth=1,color='r')             


#pcolormesh = m.pcolormesh(lon, lat, data[0, 1, :], latlon=True, cmap='terrain_r')
pcolormesh = m.pcolormesh(lon, lat, dataplot.TMP[0, 0], latlon=True, cmap='terrain_r')

aa


data5d = np.array(nldas.to_array())


r_nldas = ClipRaster(data3d, lat, lon, cell_size)

r1_cliped, lat_cliped, lon_cliped = r_nldas.clip(shp_path, drop=True, scale_factor=1)

M = r_nldas.get_mean(shp_path, scale_factor=1)
print(M.shape)




m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=24.523100, urcrnrlat =49.384366,
            llcrnrlon=-124.763083, urcrnrlon =-66.949894)   

shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
							   linewidth=1,color='r')             


pcolormesh = m.pcolormesh(lon, lat, r1_cliped[0, 0, 0, :, :, 0, 0], latlon=True, cmap='terrain_r')


size = 0
try:
    m.scatter(lon, lat, s=size)
except:
    lon, lat = np.meshgrid(lon, lat)
    m.scatter(lon, lat, s=size)


fig = plt.gcf()

fig.colorbar(pcolormesh)




aa





# da_mask = h5py.File(f'AMSR_U2_L3_DailySnow_B02_20230330.he5','r')
# cell_size = 0.3
# lat = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lat'))
# lon = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lon'))
# data = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernDaily'),dtype=np.float32)
# lon = np.where(lon>10000, 0, lon)
# lat = np.where(lat>10000, 0, lat)

# data3d = np.dstack([data]*3)
# data3d = np.random.rand(721, 721, 3)
# data3d = np.moveaxis(data3d, -1, time_axis)



# dataset = h5py.File(f'SMAP_L3_SM_P_20150502_R18290_001.h5','r')
# cell_size = 0.3
# name_am = '/Soil_Moisture_Retrieval_Data_AM/soil_moisture'
# SM_am = dataset['Soil_Moisture_Retrieval_Data_AM/soil_moisture'][:]
# data = np.where(SM_am==-9999.0,np.nan,SM_am)
# lat_am = dataset['Soil_Moisture_Retrieval_Data_AM/latitude'][:]
# lat = np.where(lat_am==-9999.0,np.nan,lat_am)
# lon_am = dataset['Soil_Moisture_Retrieval_Data_AM/longitude'][:]
# lon = np.where(lon_am==-9999.0,np.nan,lon_am)
    
# data3d = np.dstack([data]*3)
# #data3d = np.random.rand(721, 721, 3)
# data3d = np.moveaxis(data3d, -1, time_axis)





#d_lat = np.abs(lat[1:, :] - lat[:-1, :])
#d_lon = np.abs(lon[:, 1:] - lon[:, :-1])



## TODO: name should change, instantiate with ClipRaster is wierd
r1 = ClipRaster(data3d, lat, lon, cell_size)

scale_factor = 1

s=time.time()   
for i in range(1):
    r1_cliped, lat_cliped, lon_cliped = r1.clip3d(shp_path, time_axis, drop=True, scale_factor=scale_factor)
print("time:", (time.time()-s))




weights, mask = r1.mask_shp(shp_path, scale_factor=scale_factor)

r2 = ClipRaster(data3d[0, :, :], lat, lon, cell_size)
s=time.time()   
for i in range(1):
    r1_cliped, lat_cliped, lon_cliped = r2.clip2d(shp_path, drop=True, scale_factor=scale_factor)
print("time:", (time.time()-s))



print('mean=', r1.get_mean3d(shp_path, time_axis, scale_factor=scale_factor))

# r2 = ClipRaster(data3d[0, :, :], lat, lon, 0.005)
# print('mean=', r2.get_mean2d('shpfiles/small_basin.shp', scale_factor=scale_factor))


#plt.imshow(r1_cliped[1, :, :])
#plt.colorbar()

#lon[lon < 0] += 360
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=down-.1, urcrnrlat =up+.1,
            llcrnrlon=left-.1, urcrnrlon =right+.1)   

shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
							   linewidth=1,color='r')             
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(down, up, .25),
                labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(left, right, .25),
                labels=[0, 0, 0, 1])

pcolormesh = m.pcolormesh(lon, lat, data, latlon=True, cmap='terrain_r', vmin=0, vmax=1)


size = .001
try:
    m.scatter(lon, lat, s=size)
except:
    lon, lat = np.meshgrid(lon, lat)
    m.scatter(lon, lat, s=size)


fig = plt.gcf()

fig.colorbar(pcolormesh)

















