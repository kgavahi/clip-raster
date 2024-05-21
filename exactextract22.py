# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:53:52 2024

@author: pmoghaddasi
"""
from exactextract import exact_extract
import xarray as xr
import geopandas as gpd
from pyproj import CRS
import rioxarray
import time

# Load the raster dataset from the NetCDF file
nc_path = "daymet_v4_daily_na_swe_2011.nc"
rast = xr.open_dataset(nc_path)

rast = rast['swe'][:10]

# Open the shapefile using geopandas
# =============================================================================
# shapefile = "D:\\Fatemeh\\camels-20240418T1048Z\\basin_set_full_res\\HCDN_nhru_final_671.shp"
# 
# =============================================================================
output_shapefile = "HCDN_nhru_final_671_daymet.shp"
# =============================================================================
# 
# gdf = gpd.read_file(shapefile)
# 
# daymet_crs = CRS.from_proj4(
#     "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
#     "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
# )
# 
# 
# gdf_daymet = gdf.to_crs(daymet_crs)
# 
# shapefile = gdf_daymet.to_file(output_shapefile)
# 
# =============================================================================

s=time.time()
# Perform the exact extraction
stats = exact_extract(rast, output_shapefile, ['mean'])
tpassed = time.time()-s
print(tpassed)

# print(stats)
