import os
import datetime
import requests
import shutil
import xarray as xr



class DataPreprocess:
    
    
    
    
    
    
    
    def __init__(self, date: str, user: str, password: str):
        
        
        # date format should be %Y%m%d (20230101)
        self.date = date
        self.user = user
        self.password = password
        
    
    def dl_nldas(self, path: str):
        
        # determine the day of year
        fmt = '%Y%m%d'
        dt = datetime.datetime.strptime(self.date, fmt)
        numday = dt.timetuple().tm_yday
        
        # get the 24 urls, one for each hour
        urls = [(f'https://hydro1.gesdisc.eosdis.nasa.gov/'
                f'daac-bin/OTF/HTTP_services.cgi?FILENAME=%2'
                f'Fdata%2FNLDAS%2FNLDAS_FORA0125_H.002%2F'
                f'{self.date[:4]}%2F{numday:03d}%2FNLDAS_FORA0125_'
                f'H.A{self.date}.{h:02d}00.002.grb&FORMAT=bmM0Lw'
                f'&BBOX=25%2C-125%2C53%2C-67&LABEL=NLDAS_FORA0125'
                f'_H.A{self.date}.{h:02d}00.002.grb.SUB.nc4&SHORTNAME'
                f'=NLDAS_FORA0125_H&SERVICE=L34RS_LDAS&VERSION=1'
                f'.02&DATASET_VERSION=002') for h in range(24)]
        
        # let's save the urls in a text file so that we 
        # could download them with a single wget command
        file_path = os.path.join(path, "urls.txt")
        if os.path.exists(file_path): os.remove(file_path)
        with open(file_path, 'w') as fp:
            fp.write('\n'.join(urls))        
        
            
        # download the files
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path}\
                          --content-disposition -i {file_path}')
        
        
    def dl_chirps(self, path: str):
        #TODO: which chirps product??
        url = ('https://data.chc.ucsb.edu/products/CHIRPS-2.0/'
               'global_daily/netcdf/p05/by_month/chirps-v2.0.'
               f'{self.date[:4]}.{self.date[4:6]}.days_p05.nc')
        
        
        fileName = url.split('/')[-1].strip()
        
        # download the url
        print(f'downloading {fileName} ...')
        os.system(f'wget -P {path} --content-disposition {url}')
        
        
    def dl_cmorph(self, path: str):
        
        url = (f'https://www.ncei.noaa.gov/data/cmorph-high'
               f'-resolution-global-precipitation-estimates/'
               f'access/daily/0.25deg/{self.date[:4]}/{self.date[4:6]}'
               f'/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_{self.date}.nc')
        

        fileName = url.split('/')[-1].strip()
        
        # download the url
        print(f'downloading {fileName} ...')
        os.system(f'wget -P {path} --content-disposition {url}')
      
        
        
        
        
        
#dp = DataPreprocess('20230401', 'kgavahi', '491Newyork')
#dp.dl_chirps('chirps')

chirps = xr.open_dataset('chirps/chirps-v2.0.2023.04.days_p05.nc')

daymet = xr.open_dataset('daymet_v4_daily_na_swe_2011.nc')

import numpy as np


up= 49.384366
down=24.523100
right=-66.949894
left=-124.763083

chirps = chirps.isel(longitude=(chirps.longitude >= left) & (chirps.longitude <= right),
                          latitude=(chirps.latitude >= down) & (chirps.latitude <= up),
                          )

up= 1407298.913147
down=-1503823.977287
right=2258121.111016
left=-2361365.578107

daymet = daymet.isel(x=(daymet.x >= left) & (daymet.x <= right),
                          y=(daymet.y >= down) & (daymet.y <= up),
                          )

lat_daymet = np.array(daymet.lat)
lon_daymet = np.array(daymet.lon)

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

points = FTranspose(lon_daymet, lat_daymet)

points_product = FTranspose(chirps.longitude, chirps.latitude)

from scipy.spatial import KDTree
kdtree = KDTree(points_product)
d, arg_dd = kdtree.query(points)


# ch_p = np.array(chirps.precip[0]).flatten()

# weights = np.zeros(ch_p.shape)

# unique, counts = np.unique(arg_dd, return_counts=True)

# weights = weights.flatten()
# print(weights.shape)

# weights =+ np.array(daymet.swe[0]).flatten()[unique]

# weights = weights.reshape(chirps.precip[0].shape)


from mpl_toolkits.basemap import Basemap
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=45.172096, urcrnrlat =45.3,
            llcrnrlon=-110.292881, urcrnrlon =-110) 

# pcolormesh = m.pcolormesh(daymet.lon, daymet.lat, daymet.swe[0], 
#                           latlon=True, cmap='jet', vmin=0, vmax=500) 

# pcolormesh = m.pcolormesh(chirps.longitude, chirps.latitude,
#                           chirps.precip[0], 
#                           latlon=True, cmap='jet', vmin=0, vmax=10) 

pcolormesh = m.pcolormesh(daymet.lon, daymet.lat,
                          arg_dd.reshape(2911, 4620), 
                          latlon=True, cmap='jet')      


import matplotlib.pyplot as plt
fig = plt.gcf()

fig.colorbar(pcolormesh)        
# m.drawparallels(np.arange(-90, 90, 50),
#                 labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180, 180, 50),
#                 labels=[0, 0, 0, 1])
        
        

