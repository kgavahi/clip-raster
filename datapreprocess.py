import os
import datetime
import requests
import shutil
import xarray as xr
import pandas as pd
import time
from scipy.spatial import KDTree
import urllib.request
from bs4 import BeautifulSoup

class DataPreprocess:
    
    
    
    
    
    
    
    def __init__(self, user=None, password=None):
        
        
        self.user = user
        self.password = password
        
    
    def dl_nldas(self, path=None, start_date=None, end_date=None):
        
        if end_date==None:
            date_range = pd.date_range(start=start_date, 
                                       end=start_date, 
                                       freq='D')
        else:
            date_range = pd.date_range(start=start_date, 
                                       end=end_date, 
                                       freq='D')
        
        # let's save the urls in a text file to 
        # download them with a single wget command
        txt_path = os.path.join(path, "urls.txt")
        if os.path.exists(txt_path): os.remove(txt_path)
        
        for date in date_range:
            numday = date.timetuple().tm_yday
            date_str = str(date)[:10].replace('-', '') 
       
        
            # get the 24 urls, one for each hour
            urls = [(f'https://hydro1.gesdisc.eosdis.nasa.gov/'
                    f'daac-bin/OTF/HTTP_services.cgi?FILENAME=%2'
                    f'Fdata%2FNLDAS%2FNLDAS_FORA0125_H.002%2F'
                    f'{date_str[:4]}%2F{numday:03d}%2FNLDAS_FORA0125_'
                    f'H.A{date_str}.{h:02d}00.002.grb&FORMAT=bmM0Lw'
                    f'&BBOX=25%2C-125%2C53%2C-67&LABEL=NLDAS_FORA0125'
                    f'_H.A{date_str}.{h:02d}00.002.grb.SUB.nc4&SHORTNAME'
                    f'=NLDAS_FORA0125_H&SERVICE=L34RS_LDAS&VERSION=1'
                    f'.02&DATASET_VERSION=002') for h in range(24)]
        

            
            with open(txt_path, 'a') as fp:
                fp.write('\n'.join(urls))        
        
        
        # download the files
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path}\
                          --content-disposition -i {txt_path}')
        
        
    def dl_chirps(self, path=None, start_date=None, end_date=None):
        
        if end_date==None:
            date_range = pd.date_range(start=start_date, 
                                       end=start_date, 
                                       freq='D')
        else:
            date_range = pd.date_range(start=start_date, 
                                       end=end_date, 
                                       freq='D')        
        
        date_str = [str(date)[:10].replace('-', '') for date in date_range]

        
        #TODO: which chirps product??
        urls = [('https://data.chc.ucsb.edu/products/CHIRPS-2.0/'
               'global_daily/netcdf/p05/by_month/chirps-v2.0.'
               f'{date[:4]}.{date[4:6]}.days_p05.nc')
                for date in date_str]
               
        # let's save the urls in a text file to 
        # download them with a single wget command
        txt_path = os.path.join(path, "urls.txt")
        if os.path.exists(txt_path): os.remove(txt_path)
        
        with open(txt_path, 'a') as fp:
            fp.write('\n'.join(set(urls)))        
        
        
        # download the urls
        os.system(f'wget -P {path} --content-disposition -i {txt_path}')
        
        
    def dl_cmorph(self, path=None, start_date=None, end_date=None):

        if end_date==None:
            date_range = pd.date_range(start=start_date, 
                                       end=start_date, 
                                       freq='D')
        else:
            date_range = pd.date_range(start=start_date, 
                                       end=end_date, 
                                       freq='D')        
        
        date_str = [str(date)[:10].replace('-', '') for date in date_range]

        
        urls = [(f'https://www.ncei.noaa.gov/data/cmorph-high'
               f'-resolution-global-precipitation-estimates/'
               f'access/daily/0.25deg/{date[:4]}/{date[4:6]}'
               f'/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_{date}.nc') 
               for date in date_str]
        
        # let's save the urls in a text file to 
        # download them with a single wget command
        txt_path = os.path.join(path, "urls.txt")
        if os.path.exists(txt_path): os.remove(txt_path)
        
        with open(txt_path, 'a') as fp:
            fp.write('\n'.join(set(urls))) 
        
        # download the urls
        os.system(f'wget -P {path} --content-disposition -i {txt_path}')
      

    def dl_imerg(self, path=None, start_date=None, end_date=None, product=None, version='07'):

        if end_date==None:
            date_range = pd.date_range(start=start_date, 
                                       end=start_date, 
                                       freq='D')
        else:
            date_range = pd.date_range(start=start_date, 
                                       end=end_date, 
                                       freq='D')        
        
        date_str = [str(date)[:10].replace('-', '') for date in date_range]      
        
        
        s=time.time()
        page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                    f'{product}.{version}/{date[:4]}/{date[4:6]}/')
                    for date in date_str])
        urls = []
        for page_url in page_urls:
            
            uf = urllib.request.urlopen(page_url)
            html = uf.read()
            soup = BeautifulSoup(html, "lxml")
            link_list = set([link.get('href') for link in soup.find_all('a') 
                         if link.get('href').endswith('nc4')])
            
            filtered_links = [link for link in link_list if any(date in link for date in date_str)]
            
            for link in filtered_links:
                urls.append(page_url+link)
        
        print(urls)
        print(time.time()-s)
        a

        urls = [(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                 f'{product}.{version}/{date[:4]}/{date[4:6]}/3B-DAY.MS.MRG.3IMERG'
                 f'.{date}-S000000-E235959.V{version}.nc4') 
               for date in date_str]        

        # let's save the urls in a text file to 
        # download them with a single wget command
        txt_path = os.path.join(path, "urls.txt")
        if os.path.exists(txt_path): os.remove(txt_path)
        
        with open(txt_path, 'a') as fp:
            fp.write('\n'.join(set(urls))) 
            
        # download the files
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path}\
                          --content-disposition -i {txt_path}')            
            
            
        
dp = DataPreprocess(user='kgavahi', password='491Newyork')
dp.dl_imerg(path='chirps', start_date='20010101', end_date='20030101',
            product='GPM_3IMERGDE', version='06')
# import urllib.request
# url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.06/2010/01/'
# uf = urllib.request.urlopen(url)
# html = uf.read()
# from bs4 import BeautifulSoup
# soup = BeautifulSoup(html, "lxml")
# for link in soup.find_all('a'):
#     print(link.get('href'))

aa
chirps = xr.open_dataset('chirps/chirps-v2.0.2023.04.days_p05.nc')

daymet = xr.open_dataset('daymet_v4_daily_na_tmax_2011.nc')

import numpy as np




# up= 785366.110912
# down=767749.595638
# right=-1094376.75696
# left=-1119311.59835

up= 1407298.913147
down=-1503823.977287
right=2258121.111016
left=-2361365.578107

# up= 805366.110912
# down=707749.595638
# right=-914376.75696
# left=-1019311.59835

daymet = daymet.isel(x=(daymet.x >= left) & (daymet.x <= right),
                          y=(daymet.y >= down) & (daymet.y <= up),
                          )

# daymet = daymet.coarsen(x=5, boundary='pad').mean()\
#         .coarsen(y=5, boundary='pad').mean()

lat_daymet = np.array(daymet.lat)
lon_daymet = np.array(daymet.lon)

up= lat_daymet.max()
down=lat_daymet.min()
right=lon_daymet.max()
left=lon_daymet.min()

chirps = chirps.isel(longitude=(chirps.longitude >= left) & (chirps.longitude <= right),
                          latitude=(chirps.latitude >= down) & (chirps.latitude <= up),
                          )

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
s = time.time()
points = FTranspose(lon_daymet, lat_daymet)

points_product = FTranspose(chirps.longitude, chirps.latitude)


kdtree = KDTree(points_product)
d, arg_dd = kdtree.query(points)


daymet_f = np.array(daymet.tmax[0]).flatten()
chirps_f = np.array(chirps.precip[0]).flatten()

# daymet_coarse = np.empty(len(chirps_f))

# for i in range(len(chirps_f)):
#     print(i, len(chirps_f))
#     daymet_coarse[i] = np.nanmean(daymet_f[np.where(arg_dd==i)])
# daymet_coarse = daymet_coarse.reshape(chirps.precip[0].shape)


##########




df1 = np.column_stack((arg_dd, daymet_f))
df2 = pd.DataFrame(df1, columns=['group', 'daymet'])
df3 = df2.groupby('group').mean().reindex(np.arange(len(chirps_f)))
daymet_coarse = np.array(df3).reshape(chirps.precip[0].shape)
print(time.time() - s)
##############
from scipy.interpolate import griddata

vs = np.moveaxis(daymet.tmax[:20].values, 0, -1).reshape(4620*2911, 20)

s = time.time()
daymet_coarse = griddata(
    #values=daymet.tmax[0].values.ravel(),
    values=vs,
    points=np.stack((daymet['lon'], daymet['lat']), axis=-1).reshape((-1, 2)),
    xi=np.stack(np.meshgrid(chirps.longitude, chirps.latitude), axis=-1).reshape((-1, 2)),
method='linear').reshape(493, 1412, 20)
print(time.time() - s)






from mpl_toolkits.basemap import Basemap
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=lat_daymet.min(), urcrnrlat =lat_daymet.max(),
            llcrnrlon=lon_daymet.min(), urcrnrlon =lon_daymet.max()) 


pcolormesh = m.pcolormesh(chirps.longitude, chirps.latitude,
                          daymet_coarse[:, :, 0], 
                          latlon=True, cmap='jet')


# pcolormesh = m.pcolormesh(daymet.lon, daymet.lat, daymet.tmax[0], 
#                           latlon=True, cmap='jet') 

# np.random.seed(0)
# pcolormesh = m.pcolormesh(chirps.longitude, chirps.latitude,
#                           chirps.precip[0]+np.random.rand(4,8), 
#                           latlon=True, cmap='jet') 

# pcolormesh = m.pcolormesh(daymet.lon, daymet.lat,
#                           arg_dd.reshape(18, 25), 
#                           latlon=True, cmap='jet',
#                           vmin=0, vmax=30)      


import matplotlib.pyplot as plt
fig = plt.gcf()

fig.colorbar(pcolormesh)        
# m.drawparallels(np.arange(-90, 90, 50),
#                 labels=[1, 0, 0, 0])
# m.drawmeridians(np.arange(-180, 180, 50),
#                 labels=[0, 0, 0, 1])
        
        

