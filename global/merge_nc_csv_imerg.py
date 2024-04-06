import pandas as pd
import xarray as xr
import time
import glob


#s = time.time()
#df = pd.read_csv('/mh1/kgavahi/Paper4/test.csv')
#print(time.time()-s, 'done reading csv file')



#########################
s=time.time()
selected_columns = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'PRCP']
df = pd.concat((pd.read_csv(file, usecols=selected_columns) 
                for file in glob.glob('stations/*.csv')), 
                ignore_index=True)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[(df['DATE'] >= '2001-01-01') & (df['DATE'] <= '2002-01-01')]
print(time.time()-s, 'done reading csv file')
#############################
s = time.time()
df_dd = df.drop_duplicates(subset=['STATION'])
lat_st = df_dd.LATITUDE
lon_st = df_dd.LONGITUDE
stations = df_dd.STATION
tgt_lat = xr.DataArray(lat_st, dims="STATION", coords=dict(STATION=stations))
tgt_lon = xr.DataArray(lon_st, dims="STATION", coords=dict(STATION=stations))
print(time.time()-s, 'done drop_duplicates')
################################
print('''----------------------------IMERG---------------------------------''')
s = time.time()
da = xr.open_mfdataset('IMERG/'
                       '3B-DAY.MS.MRG.3IMERG.*-S000000-E235959.V06.nc4')
da = da.HQprecipitation
datetimeindex = da.indexes['time'].to_datetimeindex()
da['time'] = datetimeindex
print(time.time()-s, 'done reading prdt files')



s = time.time()
da = da.sel(lon=tgt_lon, lat=tgt_lat, method="nearest")
print(time.time()-s, 'done sel')

s = time.time()
df_prdt = da.to_dataframe()
df_prdt = df_prdt.reset_index().rename(columns={'time':'DATE',
                                                    #'lat':'lat_imerg',
                                                    #'lon':'lon_imerg',
                                                    'HQprecipitation':'imerg'})
df_prdt.drop(columns=['lat', 'lon'], inplace=True)
print(time.time()-s, 'done to_dataframe')



s = time.time()
df2 = df.merge(df_prdt, on=['DATE', 'STATION'], how='outer')
print(time.time()-s, 'done merge')
print('''----------------------------CMORPH---------------------------------''')
s = time.time()
da = xr.open_mfdataset('CMORPH/'
                       'CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_*.nc')
da.coords['lon'] = (da.coords['lon'] + 180) % 360 - 180
#da = da.sortby(da.lon)



da = da.cmorph
da = da.where((da>=0) & (da<100000))
da[0].plot(cmap='gist_ncar_r')
print(time.time()-s, 'done reading prdt files')
aa
s = time.time()
da = da.sel(lon=tgt_lon, lat=tgt_lat, method="nearest")
print(time.time()-s, 'done sel')

s = time.time()
df_prdt = da.to_dataframe()
df_prdt = df_prdt.reset_index().rename(columns={'time':'DATE',
                                                    #'lat':'lat_cmorph',
                                                    #'lon':'lon_cmorph',
                                                    'cmorph':'cmorph'})
df_prdt.drop(columns=['lat', 'lon'], inplace=True)
print(time.time()-s, 'done to_dataframe')


s = time.time()
df2 = df2.merge(df_prdt, on=['DATE', 'STATION'], how='outer')
print(time.time()-s, 'done merge')



s = time.time()
df2.to_csv('test.csv')
print(time.time()-s, 'done to_csv') 

a
print('---------------------------------------------------------------------------')

#import numpy as np
da = xr.open_mfdataset('PERSIANN/'
                       'CDR_2022-04-17030747pm_*.nc')

da = da.where((da>=0) & (da<100000))
print(da.encoding)
#data = np.array(da.precip[0])

aa


import numpy.ma as ma
stat = pd.DataFrame()
products = ['imerg', 'cmorph']

for product in products:
    # Calculate RMSE for each station
    stat[f'rmse_{product}'] = df2.groupby(['STATION', 'LATITUDE', 'LONGITUDE']).apply(lambda x: \
                        np.sqrt(((x['PRCP'] - x[product]) ** 2).mean()))


    stat[f'corr_{product}'] = df2.groupby(['STATION', 'LATITUDE', 'LONGITUDE']).apply(lambda x: \
                        ma.corrcoef(ma.masked_invalid(x['PRCP']), ma.masked_invalid(x[product]))[0, 1])


    
#stat.to_csv('stat.csv')

stat = pd.read_csv('stat.csv')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

down = -89
up = 89
left = -180
right = 180
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=down-.1, urcrnrlat =up+.1,
            llcrnrlon=left-.1, urcrnrlon =right+.1)

m.drawcoastlines(linewidth=0.5)

lat = stat['LATITUDE']
lon = stat['LONGITUDE']

pcolormesh = m.scatter(lon, lat, c=stat['corr_persiann'], cmap='rainbow_r', s=.1, vmin=0)
fig = plt.gcf()

fig.colorbar(pcolormesh)









