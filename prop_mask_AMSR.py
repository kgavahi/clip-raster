import xarray as xr
import numpy as np
from matplotlib.path import Path
import shapefile
import time
import h5py  


    
    
''' -------------Dataset Specific------------------- '''
# Load your original 25km xarray dataset
da_mask = h5py.File(f'AMSR_U2_L3_DailySnow_B02_20230330.he5','r')
cell_size = 0.3
lat = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lat'))
lon = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lon'))
swe = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernDaily'))
''' ------------------------------------------------ '''



# Define the shapefile
shp_filename = 'shpfiles/hysets_basin_shapes.shp'
shp = shapefile.Reader(shp_filename)

for basin_id in range(4620):

    # Define the mask filename
    name = shp.records()[basin_id][0]
    
    #if not name == 'hysets_02OB033':
    #    continue
    #print(basin_id)
    print(basin_id, name)

    
    # Get the polygon vertices of the basin
    tupVerts = shp.shapes()[basin_id].points
    
    # Get the boundries of the basin
    tupVerts_np = np.array(tupVerts)
    up = np.max(tupVerts_np[:, 1]) + cell_size
    down = np.min(tupVerts_np[:, 1]) - cell_size
    left = np.min(tupVerts_np[:, 0]) - cell_size
    right = np.max(tupVerts_np[:, 0]) + cell_size
    

        
    # Define the scale factor for downscaling (25km to 1km)
    scale_factor =  400 # This assumes a 25x downscaling (25km to 1km)

    # Create new coordinates for the 1km grid
    new_lon = np.arange(left, right, 1/scale_factor)
    new_lat = np.arange(down, up, 1/scale_factor)
    
    # Create empty arrays to hold the downscaled data
    downscaled_data = np.zeros((new_lat.size, new_lon.size))
    
    # Create a new xarray dataset with downscaled data
    downscaled_dataset = xr.Dataset(
        {
            'swe_downscaled': (('lat', 'lon'), downscaled_data)
        },
        coords={
            'lat': new_lat,
            'lon': new_lon
        }
    )
    

    # Create a mask for the basin
    x, y = np.meshgrid(new_lon, new_lat)
    xf, yf = x.flatten(), y.flatten()
    points = np.vstack((xf,yf)).T 
    p = Path(tupVerts) # make a polygon
    grid = p.contains_points(points)
    mask = grid.reshape(x.shape[0],x.shape[1]) # now you have a mask with points inside a polygon    
    
    
    mask_true = np.where(mask)
    
    
    mask_original = np.zeros((721, 721))
    for i, j in zip(mask_true[0], mask_true[1]):
           
        #abs_lat = np.abs(new_lat[i] - lat)
        #arg_lat = np.argmin(abs_lat)
        
        #abs_lon = np.abs(new_lon[j] - lon)
        #arg_lon = np.argmin(abs_lon)
        
        d = np.sqrt((new_lat[i] - lat)**2 + (new_lon[j] - lon)**2)
        arg_d = np.where(d==np.nanmin(d))
        arg_lat = arg_d[0][0]
        arg_lon = arg_d[1][0]
        

       
        
        mask_original[arg_lat, arg_lon] += 1
        
    
    mask_original /= np.sum(mask_original)
    
    
    np.save(f'mask_ASMR/mask_{name}', mask_original)
    

    
