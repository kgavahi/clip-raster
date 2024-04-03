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
df = df[(df['DATE'] >= '2001-01-01') & (df['DATE'] <= '2022-01-01')]
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
da = da.cmorph
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



s = time.time()
df2.to_csv('test.csv')
print(time.time()-s, 'done to_csv') 








