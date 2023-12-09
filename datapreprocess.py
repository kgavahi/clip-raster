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
import numpy as np
from urllib.parse import urljoin, urlparse

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
        
        urls = []
        for date in date_range:
            numday = date.timetuple().tm_yday
            date_str = str(date)[:10].replace('-', '') 
       
            for h in range(24):
                # get the 24 urls, one for each hour
                urls_h = (f'https://hydro1.gesdisc.eosdis.nasa.gov/'
                        f'daac-bin/OTF/HTTP_services.cgi?FILENAME=%2'
                        f'Fdata%2FNLDAS%2FNLDAS_FORA0125_H.002%2F'
                        f'{date_str[:4]}%2F{numday:03d}%2FNLDAS_FORA0125_'
                        f'H.A{date_str}.{h:02d}00.002.grb&FORMAT=bmM0Lw'
                        f'&BBOX=25%2C-125%2C53%2C-67&LABEL=NLDAS_FORA0125'
                        f'_H.A{date_str}.{h:02d}00.002.grb.SUB.nc4&SHORTNAME'
                        f'=NLDAS_FORA0125_H&SERVICE=L34RS_LDAS&VERSION=1'
                        f'.02&DATASET_VERSION=002')
        
                urls.append(urls_h)
        

            
        with open(txt_path, 'a') as fp:
            fp.write('\n'.join(urls)) 

                

        
        # download the files
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path}\
                          --content-disposition -i {txt_path}')
        
    def dl_gldas(self, path=None, start_date=None, end_date=None):
        
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
        
        urls = []
        for date in date_range:
            numday = date.timetuple().tm_yday
            date_str = str(date)[:10].replace('-', '') 
       
            for h in range(3, 25, 3):
                if h==24: h=0
                # get the 8 urls, one for each 3hour
                urls_h = (f'https://hydro1.gesdisc.eosdis.nasa.gov/data/'
                          f'GLDAS/GLDAS_NOAH025_3H.2.1/{date_str[:4]}/'
                          f'{numday:03d}/GLDAS_NOAH025_3H.A{date_str}'
                          f'.{h:02d}00.021.nc4')
        
                urls.append(urls_h)

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
      

    def dl_gpmL3(self, path=None, product=None,
                 start_date=None, end_date=None):
        """
        

        Parameters
        ----------
        path : TYPE, optional
            DESCRIPTION. The default is None.
        start_date : TYPE, optional
            DESCRIPTION. The default is None.
        end_date : TYPE, optional
            DESCRIPTION. The default is None.
        product : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        extns = ['nc', 'HDF', 'xml']
        
        
        url = f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/{product}/'
        
        
        file_name=[]
        while not any(extn in file_name for extn in extns):
            url = get_next_link(url)
            file_name = url.split('.')[-1]
      
        
      
        url_split = url.split('/')
        url_s = str(url.split('/')[-1])
        first_date_availa = url_s.split('.')[4][:8]
        
        if pd.to_datetime(first_date_availa) > pd.to_datetime(start_date):
            print(f'dates before {first_date_availa} are not available to download')
            start_date = first_date_availa
            
        
        if end_date==None:
            date_range = pd.date_range(start=start_date, 
                                       end=start_date, 
                                       freq='D')
        else:
            date_range = pd.date_range(start=start_date, 
                                       end=end_date, 
                                       freq='D')        
        
        date_str = [str(date)[:10].replace('-', '') for date in date_range]
        numday = [date.timetuple().tm_yday for date in date_range]
        
        



        
        

        if len(url_split)==9 and len(url_split[-2])==2:
  
            page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                    f'{product}/{date[:4]}/{date[4:6]}/')
                    for date in date_str])

        if len(url_split)==9 and len(url_split[-2])==3:
            page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                f'{product}/{date[:4]}/{numday:03d}/')
                for date, numday in zip(date_str, numday)])

            
        
        if len(url.split('/'))==8:
            page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                        f'{product}/{date[:4]}/')
                        for date in date_str])             

        if len(url.split('/'))==7:
            page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                        f'{product}/')
                        for date in date_str])         
        
        
        '''
        uf = urllib.request.urlopen('https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/', timeout=20)
        html = uf.read()        
        soup = BeautifulSoup(html, 'html.parser')        
        all_prdts = soup.find_all('a', href=True)[6:-4]
        
        for prdt in all_prdts:
            prdt = prdt.get('href')
            #print(prdt)

            #prdt = 'GPM_3GPROFF19SSMIS.07/'
            if prdt == 'GPM_3GPROFF19SSMIS.07/': continue
            if prdt == 'GPM_3GPROFF19SSMIS_DAY.07/': continue
            if prdt == 'GPM_3GPROFMETOPAMHS.07/': continue
            if prdt == 'GPM_3GPROFMETOPAMHS_DAY.07/': continue
            if prdt == 'GPM_3GPROFNOAA18MHS.07/': continue
            if prdt == 'GPM_3GPROFNOAA18MHS_DAY.07/': continue
            url = f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/{prdt}/'
            extns = ['nc', 'HDF', 'xml']
            
            file_name=[]
            while not any(extn in file_name for extn in extns):
    
                url = get_next_link(url)
                #print(url)
                file_name = url.split('.')[-1]
                #print('file_name', file_name)
            
            #print(url.split('/'))
            url_s = str(url.split('/')[-1])
            print(url_s.split('.')[4][:8])
            
        aa
        n1 = get_next_link(prdt_page)
        print(n1)
        n2 = get_next_link(n1)
        print(n2)
        n3 = get_next_link(n2)
        print(n3)
        
        aa
        
        
        
        
        s=time.time()
        crawl(prdt_page, max_depth=4, visited=set())
        print('time:', time.time()-s)
        aa
        
        uf = urllib.request.urlopen(prdt_page, timeout=20)
        html = uf.read()        
        soup = BeautifulSoup(html, 'html.parser')
        # Find all links on the page
        link_list = set([link.get('href') for link in soup.find_all('a')])

        filtered_links = [link for link in link_list if 
                          any(date[:4] in link for date in date_str)]
        
        if not filtered_links:
            print(f'No date available to download for {product}')
        


        url_page = urljoin(prdt_page, filtered_links[0])
        uf = urllib.request.urlopen(url_page, timeout=20)
        html = uf.read()        
        soup = BeautifulSoup(html, 'html.parser')
        # Find all links on the page
        link_list = set([link.get('href') for link in soup.find_all('a')])
        for link in link_list:
            print(link)
        
        
        aa
        
        

        
        
        try:
            page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                    f'{product}/{date[:4]}/{date[4:6]}/')
                    for date in date_str]) 
            uf = [urllib.request.urlopen(p) for p in page_urls]

        except:
            try:
                page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                    f'{product}/{date[:4]}/{numday:03d}/')
                    for date, numday in zip(date_str, numday)]) 
                uf = [urllib.request.urlopen(p) for p in page_urls]
            except:
                try:
                    page_urls = set([(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/'
                                f'{product}/{date[:4]}/')
                                for date in date_str])  
                    uf = [urllib.request.urlopen(p) for p in page_urls]
                except:
                    print('DateError: Some dates are not available for this product')
                    print('Check the following link for available dates:')
                    print(f'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/{product}')
                    return
        '''
        
        
        urls = []
        page_urls = sorted(page_urls)
        for page_url in page_urls:
            try:
                uf = urllib.request.urlopen(page_url, timeout=120)
            except urllib.error.HTTPError as http_err:
                if http_err.code == 404:
                    #print(f'The requested URL {page_url} was not found.')
                    #print((f'check the https://gpm1.gesdisc.eosdis.nasa.gov'
                    #       f'/data/GPM_L3/{product} for available dates.'))
                    continue
                else:
                    print(f'HTTP error occurred: {http_err}')    
            #print(f'preparing the urls in {page_url}')
            html = uf.read()
            soup = BeautifulSoup(html, "lxml")
            link_list = set([link.get('href') for link in soup.find_all('a')])
            
            filtered_links = [link for link in link_list if 
                              any(date in link for date in date_str)]

            for link in filtered_links:
                urls.append(page_url+link)
        
       

        # let's save the urls in a text file to 
        # download them with a single wget command
        txt_path = os.path.join(path, "urls.txt")
        if os.path.exists(txt_path): os.remove(txt_path)
        
        with open(txt_path, 'a') as fp:
            fp.write('\n'.join(set(urls[:1]))) 
            
        # download the files
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path}\
                          --content-disposition -i {txt_path}')            
            

    def dl_modis(self, path=None, product=None,
                 start_date=None, end_date=None, tiles=None):            
        
        if tiles=='conus':
            # 14 tiles that cover the CONUS
            tiles = ["h08v04","h08v05","h08v06",
                            "h09v04","h09v05","h09v06",
                             "h10v04","h10v05","h10v06",
                             "h11v04","h11v05","h12v04",
                             "h12v05","h13v04"]       
        
        satellite_mapping = {
            'MCD': 'MOTA',
            'MYD': 'MOLA',
            'MOD': 'MOLT'
        }
        
        satellite = satellite_mapping.get(product[:3], 'unknown')
        
        if end_date==None:
            date_range = pd.date_range(start=start_date, 
                                       end=start_date, 
                                       freq='D')
        else:
            date_range = pd.date_range(start=start_date, 
                                       end=end_date, 
                                       freq='D')        
        
        date_str = [str(date)[:10].replace('-', '') for date in date_range]

        page_url = f'https://e4ftl01.cr.usgs.gov/{satellite}/{product}/'
        
        uf = urllib.request.urlopen(page_url, timeout=120)
        html = uf.read()
        soup = BeautifulSoup(html, "lxml")
        link_list = set([link.get('href') for link in soup.find_all('a')])

        filtered_links = [link for link in link_list if 
                          any(date in link.replace('.', '') for date in date_str)]

        page_urls = [page_url+link for link in filtered_links]
        

        urls = []
        for page_url in page_urls:
            
            uf = urllib.request.urlopen(page_url, timeout=120)
            html = uf.read()
            soup = BeautifulSoup(html, "lxml")
            link_list = set([link.get('href') for link in soup.find_all('a')])
            
            filtered_links = [link for link in link_list if 
                              any(tile in link for tile in tiles)]
            
            for link in filtered_links:
                urls.append(page_url+link)
        
        
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

def get_next_link(prdt_page):
    uf = urllib.request.urlopen(prdt_page, timeout=120)
    html = uf.read()        
    soup = BeautifulSoup(html, 'html.parser')        
    
    # Find the link with the text "Parent Directory"
    parent_dir_link = soup.find('a', string='Parent Directory')
    
    
    next_page = parent_dir_link.find_next('a')

    next_link = urljoin(prdt_page, next_page.get("href"))  

    return next_link    
              
             

# dp = DataPreprocess(user='kgavahi', password='491Newyork')
# dp.dl_gldas(path='chirps',
#             start_date='19990101', end_date='20210102',
#             )
        

# dp = DataPreprocess(user='kgavahi', password='491Newyork')
# dp.dl_modis(path='chirps', product='MYD14.061', 
#             start_date='20210101', end_date='20210117',
#             tiles='conus')

uf = urllib.request.urlopen('https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/', timeout=20)
html = uf.read()        
soup = BeautifulSoup(html, 'html.parser')        
all_prdts = [prdt.get('href') for prdt in soup.find_all('a', href=True)[6:-4]]
all_prdts = set(all_prdts)
all_prdts = sorted(all_prdts)
print(len(all_prdts))
c=1
S=time.time()
sl = 300
for prdt in all_prdts:
    #prdt = prdt.get('href')
    #print(prdt)
    time.sleep(sl)

    #prdt = 'GPM_3GPROFF16SSMIS_DAY_CLIM.07'
    if prdt == 'GPM_3GPROFF19SSMIS.07/': continue
    if prdt == 'GPM_3GPROFF19SSMIS_DAY.07/': continue
    if prdt == 'GPM_3GPROFMETOPAMHS.07/': continue
    if prdt == 'GPM_3GPROFMETOPAMHS_DAY.07/': continue
    if prdt == 'GPM_3GPROFNOAA18MHS.07/': continue
    if prdt == 'GPM_3GPROFNOAA18MHS_DAY.07/': continue
    print(f'----------------{c}---{prdt}------start-----------------------------')
    s=time.time()
    dp = DataPreprocess(user='kgavahi', password='491Newyork')
    dp.dl_gpmL3(path='chirps', product=prdt, 
                start_date='19000101', end_date='20240101')
    print('time:', time.time()-s)
    print('total time:', time.time()-S-sl*c)
    print(f'----------------{c}---{prdt}------finished-----------------------------')
    c+=1
    
aa

da = xr.open_mfdataset('chirps/3B-DAY.*.V07.nc4')
da = da.precipitation
#datetimeindex = da.indexes['time'].to_datetimeindex()
#da['time'] = datetimeindex
import glob


#########################
s=time.time()
selected_columns = ['STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'PRCP']
df = pd.concat((pd.read_csv(file, usecols=selected_columns) 
                for file in glob.glob('st/*.csv')), 
                ignore_index=True)
print(time.time()-s)
#############################




df['DATE'] = pd.to_datetime(df['DATE'])
df = df[(df['DATE'] >= '2000-01-01') & (df['DATE'] <= '2002-01-01')]



df_dd = df.drop_duplicates(subset=['STATION'])



lat_st = df_dd.LATITUDE
lon_st = df_dd.LONGITUDE
stations = df_dd.STATION




tgt_lat = xr.DataArray(lat_st, dims="STATION", coords=dict(STATION=stations))
tgt_lon = xr.DataArray(lon_st, dims="STATION", coords=dict(STATION=stations))
da = da.sel(lon=tgt_lon, lat=tgt_lat, method="nearest")
df_imerg = da.to_dataframe()
df_imerg = df_imerg.reset_index().rename(columns={"time":"DATE"})





df2 = df.merge(df_imerg, on=['DATE', 'STATION'], how='outer')

print(df2)
 
aa

import dask.dataframe as dd
df = dd.read_csv('st/*.csv', assume_missing=True)
print(df.groupby('LATITUDE').PRCP.mean().compute())


df.to_parquet('st/test.parquet', engine='pyarrow')

aa

# files = os.listdir('st')
# files = sorted(files)
# s = time.time()
# for file in files:
#     df = pd.read_csv(f'st/{file}')
#     df['DATE'] = pd.to_datetime(df['DATE'])
#     lat_st = df.LATITUDE[0]
#     lon_st = df.LONGITUDE[0]
    
#     da_st = da.sel(lon=lon_st, lat=lat_st, method="nearest")
    
#     df_imerg = da_st.to_dataframe().reset_index().rename(columns={"time":"DATE"})
    
#     print(df_imerg)
    
#     df = df.merge(df_imerg, on='DATE', how='outer')
    
#     df.to_csv('test.csv')
#  print(time.time()-s)   


files = os.listdir('st')
files = sorted(files)
lat_st = []
lon_st = []
s = time.time()
for file in files:
    df = pd.read_csv(f'st/{file}')
    df['DATE'] = pd.to_datetime(df['DATE'])
    lat_st.append(df.LATITUDE[0])
    lon_st.append(df.LONGITUDE[0])
   

    
tgt_lat = xr.DataArray(lat_st, dims="points")
tgt_lon = xr.DataArray(lon_st, dims="points")
da = da.sel(lon=tgt_lon, lat=tgt_lat, method="nearest")
df_imerg = da.to_dataframe()




s = time.time()
for i, file in enumerate(files):
    df = pd.read_csv(f'st/{file}')
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    #df_imerg_st = df_imerg.loc[df_imerg.index.get_level_values('points') == i]
    #df_imerg_st = df_imerg_st.reset_index().rename(columns={"time":"DATE"})
    
    #df = df.merge(df_imerg_st, on='DATE', how='outer')
    
    #df.to_csv('test.csv')




print(time.time()-s) 


# tgt_x = xr.DataArray(np.linspace(0, 4, num=10), dims="points")
# tgt_y = xr.DataArray(np.linspace(0, 4, num=10), dims="points")
# da = da.sel(lon=tgt_x, lat=tgt_y, method="nearest")

# print(da.precipitation.to_dataframe())
    

aa

page_url = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD43A1.061/'
uf = urllib.request.urlopen(page_url, timeout=20)
html = uf.read()
soup = BeautifulSoup(html, "lxml")
link_list = set([link.get('href') for link in soup.find_all('a')])

filtered_links = [link for link in link_list if 
                  datetime.strptime(link[:-1], '%Y.%m.%d')]


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
        
        

