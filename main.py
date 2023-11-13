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
import re
import pyproj
from pyproj import Transformer
import os
import shutil
import requests
#import pygrib


def dl_dataset(url):
    #url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2022/01/3B-DAY.MS.MRG.3IMERG.20220101-S000000-E235959.V07.nc4'
    saveName = url.split('/')[-1].strip()
    #urllib.request.urlretrieve(url, saveName)
    
    
    
    pathNetrc = os.path.join(os.path.expanduser("~"),'.netrc')
    if os.path.exists(pathNetrc):
        os.remove(pathNetrc)
        
    netrcFile = ['machine urs.earthdata.nasa.gov','login ' + 'kgavahi','password '+'491Newyork']
    with open('.netrc', 'w') as f:
        for item in netrcFile:
            f.write("%s\n" % item)
        
    shutil.copy('.netrc',os.path.expanduser("~"))
    
    
    with requests.get(url.strip(), stream=True) as response:
        if response.status_code != 200:
            print("Verify that your username and password are correct")
        else:
            response.raw.decode_content = True
            content = response.raw
            with open(saveName, 'wb') as d:
                while True:
                    chunk = content.read(16 * 1024)
                    if not chunk:
                        break
                    d.write(chunk)
            print('Downloaded file: {}'.format(saveName))

def FTranspose(lon, lat):
   
    if lat.ndim == 1:

        x, y = np.meshgrid(lon, lat)

    if lat.ndim == 2:
        x = lon
        y = lat
        
    xf, yf = x.flatten(), y.flatten()

    # TODO here can be more optimization
    #points = np.vstack((xf,yf)).T
    #points = np.transpose((xf, yf))
    points = np.column_stack((xf,yf))    
    
    return points
def mod_lat_lon(mod):
    fattrs = mod.attrs
    gridmeta = fattrs["StructMetadata.0"]
    
    ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                              (?P<upper_left_x>[+-]?\d+\.\d+)
                              ,
                              (?P<upper_left_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    
    match = ul_regex.search(gridmeta)
    x0 = float(match.group('upper_left_x'))
    y0 = float(match.group('upper_left_y'))
    
    lr_regex = re.compile(r'''LowerRightMtrs=\(
                              (?P<lower_right_x>[+-]?\d+\.\d+)
                              ,
                              (?P<lower_right_y>[+-]?\d+\.\d+)
                              \)''', re.VERBOSE)
    match = lr_regex.search(gridmeta)
    x1 = float(match.group('lower_right_x'))
    y1 = float(match.group('lower_right_y'))
    
    nx, ny = data[0].shape
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    xv, yv = np.meshgrid(x, y)
    
    # sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    # wgs84 = pyproj.Proj("+init=EPSG:4326") 
    # lon, lat= pyproj.transform(sinu, wgs84, xv, yv)    
    
    transformer = Transformer.from_crs("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext", 
                                       "+init=EPSG:4326")
    
    lon, lat= transformer.transform(xv, yv)
    
    
    
    
    return lat, lon

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


#grbs = xr.open_dataset('gfs_4_20110905_0600_000.grb2', engine='cfgrib')




# grbs = pygrib.open('gfs_4_20110905_0600_000.grb2')
# cell_size = 0.5
# grbs.seek(0)
# for grb in grbs:
#     print(grb)

# selected_grb = grbs.select(name='Precipitable water')[0]

# data, lat, lon = selected_grb.data()

# sf=1

# r_gfs = ClipRaster(data, lat, lon, cell_size)
# r_mean = r_gfs.get_mean(shp_path, scale_factor=sf)

# weights, landmask = r_gfs.mask_shp(shp_path, scale_factor=sf)




# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-.1, urcrnrlat =up+.1,
#             llcrnrlon=left-.1, urcrnrlon =right+.1)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# pcolormesh = m.pcolormesh(lon, lat, data, latlon=True)

# fig = plt.gcf()

# fig.colorbar(pcolormesh)


# import os
# import urllib.request
# import requests
# import shutil
# url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2022/01/3B-DAY.MS.MRG.3IMERG.20220101-S000000-E235959.V07.nc4'
# saveName = url.split('/')[-1].strip()
# #urllib.request.urlretrieve(url, saveName)



# pathNetrc = os.path.join(os.path.expanduser("~"),'.netrc')
# if os.path.exists(pathNetrc):
#     os.remove(pathNetrc)
    
# netrcFile = ['machine urs.earthdata.nasa.gov','login ' + 'kgavahi','password '+'491Newyork']
# with open('.netrc', 'w') as f:
#     for item in netrcFile:
#         f.write("%s\n" % item)
    
# shutil.copy('.netrc',os.path.expanduser("~"))


# with requests.get(url.strip(), stream=True) as response:
#     if response.status_code != 200:
#         print("Verify that your username and password are correct")
#     else:
#         response.raw.decode_content = True
#         content = response.raw
#         with open(saveName, 'wb') as d:
#             while True:
#                 chunk = content.read(16 * 1024)
#                 if not chunk:
#                     break
#                 d.write(chunk)
#         print('Downloaded file: {}'.format(saveName))

url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/2022/01/3B-DAY.MS.MRG.3IMERG.20220101-S000000-E235959.V07.nc4'
url = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.002/2023/001/NLDAS_FORA0125_H.A20230101.0000.002.grb'

dl_dataset(url)

os.system("wget --load-cookies C:\.urs_cookies --save-cookies C:\.urs_cookies --auth-no-challenge=on -P 22 --keep-session-cookies --content-disposition -i links.txt")
..

aa

daymet = xr.open_dataset('daymet_v4_daily_na_tmax_2011.nc')
chirps = xr.open_dataset('chirps-v2.0.2011.days_p05.nc')
imerg = xr.open_dataset('3B-DAY.MS.MRG.3IMERG.20220101-S000000-E235959.V07.nc4')
ds_nldas = xr.open_mfdataset('NLDAS/*.nc4', concat_dim='time', combine='nested')




chirps = chirps.isel(longitude=(chirps.longitude >= ds_nldas.lon.min()) & (chirps.longitude <= ds_nldas.lon.max()),
                          latitude=(chirps.latitude >= ds_nldas.lat.min()) & (chirps.latitude <= ds_nldas.lat.max()),
                          )


nldas_down = ds_nldas.interp(lat = chirps.latitude, 
                             lon = chirps.longitude, 
                             method='nearest')

imerg_down = imerg.interp(lat = chirps.latitude, 
                             lon = chirps.longitude, 
                             method='nearest')


chirps_up = chirps.interp(latitude = ds_nldas.lat, 
                             longitude = ds_nldas.lon, 
                             method='linear')



print(chirps)
lon = np.array(chirps_up.longitude)
lat = np.array(chirps_up.latitude)
tmax = np.array(chirps_up.precip[0])


m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=down+5, urcrnrlat =up+5,
            llcrnrlon=left, urcrnrlon =right)    

shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
							   linewidth=1,color='r')             



pcolormesh = m.pcolormesh(imerg.lon, imerg.lat, imerg.precipitation[0].T, 
                          latlon=True, vmin=0, vmax=100)

# pcolormesh = m.pcolormesh(imerg_down.lon, imerg_down.lat, imerg_down.precipitation[0].T, 
#                           latlon=True, vmin=0, vmax=100)


# pcolormesh = m.pcolormesh(np.array(daymet.lon), np.array(daymet.lat), daymet.tmax[0], 
#                           latlon=True, cmap='terrain_r', vmin=285, vmax=300)


# pcolormesh = m.pcolormesh(lon, lat, tmax, 
#                           latlon=True)

# pcolormesh = m.pcolormesh(lon, lat, tmax, 
#                           latlon=True)

# pcolormesh = m.pcolormesh(ds_nldas.lon, ds_nldas.lat, ds_nldas.TMP[0,0], 
#                           latlon=True, cmap='terrain_r', vmin=285, vmax=300)

# pcolormesh = m.pcolormesh(nldas_down.lon, nldas_down.lat, nldas_down.TMP[0,0], 
#                           latlon=True, cmap='terrain_r', vmin=285, vmax=300)

fig = plt.gcf()

fig.colorbar(pcolormesh)




# import urllib.request
# for yr in range(2011,2012): # note that in python, the end range is not inclusive. So, in this case data for 2015 is not downloaded.
#     url = f'https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/2129/daymet_v4_daily_na_tmax_{yr}.nc'
#     savename = url.split('/')[-1]
#     urllib.request.urlretrieve(url,savename)
    
aa
# daymet = xr.open_dataset('daymet_v4_daily_na_tmax_2011.nc')

# lon = np.array(daymet.lon)
# lat = np.array(daymet.lat)
# tmax = np.array(daymet.tmax[0])


# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-50, urcrnrlat =up+50,
#             llcrnrlon=left-50, urcrnrlon =right+50)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# pcolormesh = m.pcolormesh(lon, lat, tmax, 
#                           latlon=True, cmap='terrain_r')

aa

# ds_nldas = xr.open_mfdataset('NLDAS/*.nc4', concat_dim='time', combine='nested')

# time_range = pd.date_range('2000-01-01T00:00:00.000000000', '2000-01-01T06:00:00.000000000', freq='H')

# y = ds_nldas.reindex({"time": time_range})

# print(y)


aa

# ds2011_2014 = xr.open_mfdataset('precip.V1.0.*.nc', concat_dim='time', combine='nested')
# ds2011_2014['lon'] = ds2011_2014['lon']-360

# nldas = xr.open_dataset('NLDAS_FORA0125_H.A20000101.0000.002.grb.SUB.nc4', engine='netcdf4')
# data = np.array(nldas.to_array())
# lat = np.array(nldas.lat)
# lon = np.array(nldas.lon)


# ds2011_2014_down = ds2011_2014.interp(lat = lat, lon = lon, method='nearest')

# mrg = xr.merge([nldas, ds2011_2014_down])




# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-.1, urcrnrlat =up+.1,
#             llcrnrlon=left-.1, urcrnrlon =right+.1)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# pcolormesh = m.pcolormesh(mrg.lon, mrg.lat, mrg.APCP[1], 
#                           latlon=True, cmap='terrain_r')


aa


# ds2011_2014 = xr.open_mfdataset('precip.V1.0.*.nc', concat_dim='time', combine='nested')
# ds2011_2014['lon'] = ds2011_2014['lon']-360
# data_cpc = np.array(ds2011_2014.to_array())[0, 1].flatten()

# x, y = np.meshgrid(ds2011_2014.lon, ds2011_2014.lat)



# mod = xr.open_dataset('MOD16A2.A2023297.h11v05.061.2023313003400.hdf', engine='netcdf4')
# data = np.array(mod.to_array())
# lat, lon = mod_lat_lon(mod)



# #da_down = da.interp(y = lat, x = lon, method='nearest')

# from scipy.spatial import KDTree


# points = FTranspose(lon, lat)

# points_product = FTranspose(x, y)

# kdtree = KDTree(points_product)
# d, arg_dd = kdtree.query(points)


# datacpc_over_mod = data_cpc[arg_dd].reshape(2400, 2400)

# mod = mod.assign(cpc=(['YDim:MOD_Grid_500m_Surface_Reflectance',
#                        'XDim:MOD_Grid_500m_Surface_Reflectance'], datacpc_over_mod))

# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-30, urcrnrlat =up+30,
#             llcrnrlon=left-30, urcrnrlon =right+30)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             

# pcolormesh = m.pcolormesh(lon, lat, mod.cpc, latlon=True)
# #pcolormesh = m.pcolormesh(x, y, ds2011_2014.precip[1], latlon=True)
# #pcolormesh = m.pcolormesh(ds2011_2014.lon, ds2011_2014.lat, ds2011_2014.precip[1], latlon=True)




aa




data = np.array(ds2011_2014.to_array())
lat = np.array(ds2011_2014.lat)
lon = np.array(ds2011_2014.lon)-360
cell_size = 0.25

sf = 1
r_cpc = ClipRaster(data, lat, lon, cell_size)
r_mean = r_cpc.get_mean(shp_path, scale_factor=sf)


weights, landmask = r_cpc.mask_shp(shp_path, scale_factor=sf)

ds2011_2014 = ds2011_2014.assign(landmask=(['lat','lon'], landmask))
ds2011_2014 = ds2011_2014.assign(weights=(['lat','lon'], weights))



dataplot = ds2011_2014.where(ds2011_2014.landmask, drop=True)

dataplot.mean(dim=('lat', 'lon')).precip.cumsum().plot()
################################################################
sf = 10
r_cpc = ClipRaster(data, lat, lon, cell_size)
r_mean = r_cpc.get_mean(shp_path, scale_factor=sf)


weights, landmask = r_cpc.mask_shp(shp_path, scale_factor=sf)

dataplot = ds2011_2014 * weights

dataplot.sum(dim=('lat', 'lon')).precip.cumsum().plot()


# import geopandas
# from shapely.geometry import mapping
# geodf = geopandas.read_file(shp_path)
# dataplot = ds2011_2014.where(geodf.geometry.apply(mapping))

print(dataplot)


xr_mean = np.array(dataplot.mean(dim=('lat', 'lon')).to_array())


ds2011_2014_w = ds2011_2014 * weights

wxr_mean = np.array(ds2011_2014_w.sum(dim=('lat', 'lon')).to_array())



m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=down-.1, urcrnrlat =up+.1,
            llcrnrlon=left-.1, urcrnrlon =right+.1)    

shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
							   linewidth=1,color='r')             


#pcolormesh = m.pcolormesh(lon, lat, data[0, 1, :], latlon=True, cmap='terrain_r')
pcolormesh = m.pcolormesh(dataplot.lon, dataplot.lat, dataplot.precip[1], latlon=True, cmap='terrain_r')

aa




# mod = xr.open_dataset('MOD09A1.A2003001.h10v05.006.2015153105208.hdf', engine='netcdf4')

# data = np.array(mod.to_array())

# lat, lon = mod_lat_lon(mod)
# cell_size = 0.005
# #print(data)
# data3d = np.dstack([data]*3)
# #data3d = np.random.rand(2400, 2400, 3)
# data3d = np.moveaxis(data3d, -1, time_axis)


# sf=2

# r_mod = ClipRaster(data, lat, lon, cell_size)
# r_mean = r_mod.get_mean(shp_path, scale_factor=sf)



# weights, landmask = r_mod.mask_shp(shp_path, scale_factor=sf)
# dataplot = mod.where(landmask)
# xr_mean = np.array(dataplot.mean(dim=('YDim:MOD_Grid_500m_Surface_Reflectance'
#                                       , 'XDim:MOD_Grid_500m_Surface_Reflectance')).to_array())



# mod_w = mod * weights

# wxr_mean = np.array(mod_w.sum(dim=('YDim:MOD_Grid_500m_Surface_Reflectance'
#                                       , 'XDim:MOD_Grid_500m_Surface_Reflectance')).to_array()).ravel()

# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-.1, urcrnrlat =up+.1,
#             llcrnrlon=left-.1, urcrnrlon =right+.1)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# pcolormesh = m.pcolormesh(lon, lat, dataplot.sur_refl_b01, latlon=True)

aa

# nldas = xr.open_dataset('NLDAS_FORA0125_H.A20000101.0000.002.grb.SUB.nc4', engine='netcdf4')
# data = np.array(nldas.to_array())
# lat = np.array(nldas.lat)
# lon = np.array(nldas.lon)
# cell_size = 0.125
# data3d = np.dstack([data]*3)
# #data3d = np.random.rand(224, 464, 3)
# data3d = np.moveaxis(data3d, -1, time_axis)


# sf=100

# r_nldas = ClipRaster(data, lat, lon, cell_size)
# r_mean = r_nldas.get_mean(shp_path, scale_factor=sf).ravel()
# weights, landmask = r_nldas.mask_shp(shp_path, scale_factor=sf)


# dataplot = nldas.where(landmask)

# xr_mean = np.array(dataplot.mean(dim=('lat', 'lon')).to_array()).ravel()


# nldas_w = nldas * weights

# wxr_mean = np.array(nldas_w.sum(dim=('lat', 'lon')).to_array()).ravel()


# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=down-.1, urcrnrlat =up+.1,
#             llcrnrlon=left-.1, urcrnrlon =right+.1)    

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# pcolormesh = m.pcolormesh(lon, lat, dataplot.TMP[0, 0], latlon=True, cmap='terrain_r')

# aa


# data5d = np.array(nldas.to_array())


# r_nldas = ClipRaster(data3d, lat, lon, cell_size)

# r1_cliped, lat_cliped, lon_cliped = r_nldas.clip(shp_path, drop=True, scale_factor=1)

# M = r_nldas.get_mean(shp_path, scale_factor=1)
# print(M.shape)




# m = Basemap(projection='cyl', resolution='l',
#             llcrnrlat=24.523100, urcrnrlat =49.384366,
#             llcrnrlon=-124.763083, urcrnrlon =-66.949894)   

# shp_info = m.readshapefile(shp_path[:-4],'for_amsr',drawbounds=True,
# 							   linewidth=1,color='r')             


# pcolormesh = m.pcolormesh(lon, lat, r1_cliped[0, 0, 0, :, :, 0, 0], latlon=True, cmap='terrain_r')


# size = 0
# try:
#     m.scatter(lon, lat, s=size)
# except:
#     lon, lat = np.meshgrid(lon, lat)
#     m.scatter(lon, lat, s=size)


# fig = plt.gcf()

# fig.colorbar(pcolormesh)




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

















