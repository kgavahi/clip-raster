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
        
    
    def dl_nldas(self, path_nldas: str):
        
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
        file_path = os.path.join(path_nldas, "urls.txt")
        if os.path.exists(file_path): os.remove(file_path)
        with open(file_path, 'w') as fp:
            fp.write('\n'.join(urls))        
        
            
        # download the files
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path_nldas}\
                          --content-disposition -i {file_path}')
        
        
    def dl_chirps(self, path_chirps: str):
        
        url = ('https://data.chc.ucsb.edu/products/CHIRPS-2.0/'
               'global_daily/netcdf/p05/by_month/chirps-v2.0.'
               f'{self.date[:4]}.{self.date[4:6]}.days_p05.nc')
        
        #url = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/netcdf/p05/chirps-v2.0.2023.days_p05.nc'
        
        fileName = url.split('/')[-1].strip()
        
        # download the url
        print(f'downloading {fileName} ...')
        os.system(f'wget --load-cookies .urs_cookies --save-cookies \
                  .urs_cookies --keep-session-cookies --user={self.user}\
                      --password={self.password} -P {path_chirps}\
                          --content-disposition {url}')
        
        
        
        
        
        

      
        
        
        
        
        
dp = DataPreprocess('20230901', 'kgavahi1', '491Newyork')
dp.dl_chirps('chirps')

#da = xr.open_dataset('chirps/chirps-v2.0.2023.days_p05.nc')
        
        
        
        
        
        

