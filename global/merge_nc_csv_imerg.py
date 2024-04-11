import pandas as pd
import xarray as xr
import time
import glob
import numpy as np


start_date = '2001-01-01'
end_date = '2022-01-01'
#########################
s=time.time()
selected_columns = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'PRCP']
df = pd.concat((pd.read_csv(file, usecols=selected_columns) 
                for file in glob.glob('stations/*.csv')), 
                ignore_index=True)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
print(time.time()-s, 'done reading csv file')

df = df.dropna()

L = len(pd.date_range(start_date, end_date))
acceptable_size = int(L*0.2)

# Calculate station lengths
station_lengths = df.groupby('STATION').size()
print('station_lengths:', len(station_lengths))
'''
# Find stations with length less than acceptable_size
stations_to_remove = station_lengths[station_lengths < acceptable_size].index

# Filter the original dataframe to remove stations with length less than 7500
df = df[~df['STATION'].isin(stations_to_remove)]

# Calculate station lengths
station_lengths = df.groupby('STATION').size()
print('station_lengths after filter:', len(station_lengths))'''


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
                       '3B-DAY.MS.MRG.3IMERG.*-S000000-E235959.V06.nc4', data_vars=['HQprecipitation'])
da = da.HQprecipitation
da = da.where((da>=0) & (da<100000))
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
                       '*.nc')


da = da.cmorph
daCMORPH = da.where((da>=0) & (da<100000))

print(time.time()-s, 'done reading prdt files')


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
print('''----------------------------CHIRPS---------------------------------''')
# s = time.time()
# da = xr.open_mfdataset('CHIRPS/'
#                        'chirps-v2.0.*.days_p05.nc')
# da = da.precip
# daCHIRPS = da.where((da>=0) & (da<100000))
# print(time.time()-s, 'done reading prdt files')

# s = time.time()
# da = da.sel(longitude=tgt_lon, latitude=tgt_lat, method="nearest")
# print(time.time()-s, 'done sel')

# s = time.time()
# df_prdt = da.to_dataframe()
# df_prdt = df_prdt.reset_index().rename(columns={'time':'DATE',
#                                                     #'lat':'lat_cmorph',
#                                                     #'lon':'lon_cmorph',
#                                                     'precip':'chirps'})
# df_prdt.drop(columns=['latitude', 'longitude'], inplace=True)
# print(time.time()-s, 'done to_dataframe')


# s = time.time()
# df2 = df2.merge(df_prdt, on=['DATE', 'STATION'], how='outer')
# print(time.time()-s, 'done merge')
# print('''-------------------------PERSIANN-CDR------------------------------''')
# s = time.time()
# da = xr.open_mfdataset('PERSIANN/'
#                        'CDR_2022-04-17030747pm_*.nc')
# da = da.precip
# daPERSIANN = da.where((da>=0) & (da<100000))
# print(time.time()-s, 'done reading prdt files')


# s = time.time()
# da = da.sel(lon=tgt_lon, lat=tgt_lat, method="nearest")
# print(time.time()-s, 'done sel')

# s = time.time()
# df_prdt = da.to_dataframe()
# df_prdt = df_prdt.reset_index().rename(columns={'datetime':'DATE',
#                                                     #'lat':'lat_cmorph',
#                                                     #'lon':'lon_cmorph',
#                                                     'precip':'persiann'})
# df_prdt.drop(columns=['lat', 'lon'], inplace=True)
# print(time.time()-s, 'done to_dataframe')


# s = time.time()
# df2 = df2.merge(df_prdt, on=['DATE', 'STATION'], how='outer')
# print(time.time()-s, 'done merge')
print('''------------------------------CPC----------------------------------''')
s = time.time()
da = xr.open_mfdataset('CPC/'
                        'precip.V1.0.*.nc')
da.coords['lon'] = (da.coords['lon'] + 180) % 360 - 180
da = da.sortby(da.lon)
da = da.precip
daCPC = da.where((da>=0) & (da<100000))
print(time.time()-s, 'done reading prdt files')


s = time.time()
da = da.sel(lon=tgt_lon, lat=tgt_lat, method="nearest")
print(time.time()-s, 'done sel')

s = time.time()
df_prdt = da.to_dataframe()
df_prdt = df_prdt.reset_index().rename(columns={'time':'DATE',
                                                    #'lat':'lat_cmorph',
                                                    #'lon':'lon_cmorph',
                                                    'precip':'cpc'})
df_prdt.drop(columns=['lat', 'lon'], inplace=True)
print(time.time()-s, 'done to_dataframe')


s = time.time()
df2 = df2.merge(df_prdt, on=['DATE', 'STATION'], how='outer')
print(time.time()-s, 'done merge')
print('''-------------------------------------------------------------------''')

df2=df2.dropna()
s = time.time()
df2.to_csv('merge.csv')
print(time.time()-s, 'done to_csv') 

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

pcolormesh = m.scatter(lon, lat, c=stat['corr_imerg'], cmap='rainbow_r', s=.1, vmin=0)
fig = plt.gcf()

#fig.colorbar(pcolormesh)









