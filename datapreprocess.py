
import datetime






class DataPreprocess:
    
    
    
    
    
    
    
    def __init__(self, date: str):
        
        
        # date format should be %Y%m%d (20230101)
        self.date = date
        
        
    
    def dl_nldas(self, path: str):
        
        # determine the day of year
        fmt = '%Y%m%d'
        dt = datetime.datetime.strptime(self.date, fmt)
        numday = dt.timetuple().tm_yday
        
        urls = [f'https://hydro1.gesdisc.eosdis.nasa.gov/\
                daac-bin/OTF/HTTP_services.cgi?FILENAME=%2\
                Fdata%2FNLDAS%2FNLDAS_FORA0125_H.002%2F\
                {self.date[:4]}%2F{numday:03d}%2FNLDAS_FORA0125_\
                H.A{self.date}.{h:02d}00.002.grb&FORMAT=bmM0Lw\
                &BBOX=25%2C-125%2C53%2C-67&LABEL=NLDAS_FORA0125\
                _H.A{self.date}.{h:02d}00.002.grb.SUB.nc4&SHORTNAME\
                =NLDAS_FORA0125_H&SERVICE=L34RS_LDAS&VERSION=1\
                .02&DATASET_VERSION=002' for h in range(24)]
        
        
        
        
        
        
        
        
        
        
        
        
        
        

