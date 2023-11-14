year = 2023
numday = 1
date = '20230101'
h = 0

url = f'https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?\
FILENAME=%2Fdata%2FNLDAS%2FNLDAS_FORA0125_H.002%2F{year}%2F{numday:03d}\
%2FNLDAS_FORA0125_H.A{date}.{h:02d}00.002.grb&FORMAT=bmM0Lw&BBOX=25%2C\
-125%2C53%2C-67&LABEL=NLDAS_FORA0125_H.A{date}.{h:02d}00.002.grb.SUB.nc4\
&SHORTNAME=NLDAS_FORA0125_H&SERVICE=L34RS_LDAS&VERSION=1.02&DATASET_VERSION=002'

print(f'wget --load-cookies .urs_cookies --save-cookies .urs_cookies\
 --keep-session-cookies --user=kgavahi --ask-password --content-disposition {url}')


import os

os.system(f'wget --load-cookies .urs_cookies --save-cookies .urs_cookies\
 --keep-session-cookies --user=kgavahi --password=491Newyork --content-disposition "{url}"')