import pandas as pd
import numpy as np
import xarray as xr
import time
import clipraster as cr



# sr = time.time()
# da = xr.open_mfdataset(
#     'nc_files/us_ssmv11034tS__T0001TTNATS200401*05HP001.nc',
#     concat_dim="time",
#     combine="nested",
# )
# print('open_mfdataset:', (time.time()-sr), 'sec')

da = xr.open_dataset('snodas_200401.nc')

sr = time.time()
da_np = np.array(da.Band1)
print('np.array', (time.time()-sr), 'sec')
print(da_np.shape)



shp_path = 'hysets_basin_shapes.shp'



lat = np.array(da.lat)
lon = np.array(da.lon)
data = np.zeros([10, 10])


r_da = cr.open_raster(data, lat, lon)



sr = time.time()
weights, landmasks = r_da.mask_shp(shp_path, exact = True, scale_factor=10, crs = da.crs.spatial_ref)
print('cr time:', (time.time()-sr), 'sec')

da = da.drop_vars('crs')


arr = np.zeros([len(da.time), len(weights)])

s=time.time()
c=0
for w, l in zip(weights, landmasks):
    
    print(w.shape)
    sr = time.time()
    da_m = da_np[:, :, np.any(l, axis=0)]
    da_m = da_m[:, np.any(l, axis=1), :]
    da_wnp = w * da_m
    
    arr[:, c] = np.nansum(da_wnp, axis=(1,2))/ np.nansum(w)
    
    print('np way:', (time.time()-sr), 'sec')
    
    print(arr[:, c])
    c+=1
    
    
    sr = time.time()
    da_new = da.assign(landmask=(['lat','lon'], l))
    print('assign:', (time.time()-sr), 'sec')
    
    sr = time.time()
    da_new = da_new.where(da_new.landmask, drop=True)
    da_new = da_new.drop_vars('landmask')
    da_new = da_new.assign(w=(['lat','lon'], w))
    print('da.where:', (time.time()-sr), 'sec')
    
    
    da_new.Band1[30].plot()
    
    sr = time.time()
    da_w = da_new * w 
    
    #da_sum = da_w.sum(dim=('y', 'x')) / (np.sum(weights))
    da_sum = da_w.sum(dim=('lat', 'lon')) / np.nansum(w)
    print('da_w da_sum:', (time.time()-sr), 'sec')
    
    
    
    
    
    # sr = time.time()
    # '''
    # try:
    #     df[f'{c}'] = np.array(da_sum.Band1)
    # except:
    #     df = da_sum.Band1.to_dataframe(name='1')  '''
    # arr[:, c] = da_sum.to_array()
    # print('to_dataframe:', (time.time()-sr), 'sec')
    
    # c+=1
    
    # #print(arr)
    
#print(arr[:, 0])    
print('total:', (time.time()-s), 'sec')    
    