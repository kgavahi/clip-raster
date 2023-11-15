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
        
        # Create the .netrc file for authentication
        pathNetrc = os.path.join(os.path.expanduser("~"),'.netrc')
        if os.path.exists(pathNetrc):
            os.remove(pathNetrc)
            
        netrcFile = ['machine urs.earthdata.nasa.gov',
                      'login ' + self.user,
                      'password '+self.password]
        
        with open('.netrc', 'w') as f:
            for item in netrcFile:
                f.write("%s\n" % item)
            
        shutil.copy('.netrc', os.path.expanduser("~"))        
        
    
    def dl_nldas(self, path_nldas: str):
        
        # determine the day of year
        fmt = '%Y%m%d'
        dt = datetime.datetime.strptime(self.date, fmt)
        numday = dt.timetuple().tm_yday
        
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
        
        url = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/netcdf/p05/chirps-v2.0.2023.days_p05.nc'
        
        saveName = url.split('/')[-1].strip()
        
        file_path = os.path.join(path_chirps, saveName)

        
        with requests.get(url.strip(), stream=True) as response:
            if response.status_code != 200:
                print("Verify that your username and password are correct")
            else:
                response.raw.decode_content = True
                content = response.raw
                with open(file_path, 'wb') as d:
                    while True:
                        chunk = content.read(1024 * 1024)
                        if not chunk:
                            break
                        d.write(chunk)
                print('Downloaded file: {}'.format(saveName))        
        
        
        
        
        
dp = DataPreprocess('20230901', 'kgavahi1', '491Newyork')
dp.dl_chirps('chirps')

da = xr.open_dataset('chirps/chirps-v2.0.2023.days_p05.nc')
        
        
        
        
        
        

