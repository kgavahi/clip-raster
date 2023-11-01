# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:21:47 2023

@author: kgavahi
"""



import h5py  
import numpy as np
import time
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shapefile
from matplotlib.path import Path
from inpoly import inpoly2
from numba import jit
def FTranspose(x, y):
    xf, yf = x.flatten(), y.flatten()

    # TODO here can be more optimization
    #points = np.vstack((xf,yf)).T
    #points = np.transpose((xf, yf))
    points = np.column_stack((xf,yf))    
    
    return points
def mask_with_vert_points(tupVerts, lat, lon, mode='inpoly'):

    if lat.ndim == 1:

        x, y = np.meshgrid(lon, lat)

    if lat.ndim == 2:
        x = lon
        y = lat

    if mode == 'matplotlib':
        # Create a mask for the shapefile
        points = FTranspose(x, y)
        p = Path(tupVerts)  # make a polygon
        grid = p.contains_points(points)
        # now you have a mask with points inside a polygon
        mask = grid.reshape(x.shape[0], x.shape[1])

    if mode == 'inpoly':

        
        points = FTranspose(x, y)

        # use inpoly which lightning fast
        isin, ison = inpoly2(points, tupVerts)
        #mask = isin.reshape(x.shape[0], x.shape[1])

    return isin

np.random.seed(10)
time_axis = 0
scale_factor = 100

shp_path = 'shpfiles/for_amsr.shp'
da_mask = h5py.File(f'AMSR_U2_L3_DailySnow_B02_20230330.he5','r')
cell_size = 0.3
lat = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lat'))
lon = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/lon'))
data = np.array(da_mask.get('HDFEOS/GRIDS/Northern Hemisphere/Data Fields/SWE_NorthernDaily'))
data3d = np.dstack([data]*3)
#data3d = np.random.rand(224, 464, 3)
data3d = np.moveaxis(data3d, -1, time_axis)

shp = shapefile.Reader(shp_path)

# Get the polygon vertices of the basin
tupVerts = shp.shapes()[0].points
tupVerts_np = np.array(tupVerts)
up = np.max(tupVerts_np[:, 1])
down = np.min(tupVerts_np[:, 1])
left = np.min(tupVerts_np[:, 0])
right = np.max(tupVerts_np[:, 0])

# Create new coordinates for the downscaled grid
new_lon = np.arange(left, right, cell_size/scale_factor)
new_lat = np.arange(down, up, cell_size/scale_factor)

# Create a mask for the shapefile
mask = mask_with_vert_points(tupVerts, new_lat, new_lon)

mask_true = np.where(mask)



from numpy import random
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances_argmin


# def closest_node(node, nodes):
#     closest_index = distance.cdist(node, nodes)
#     return closest_index

# a = random.randint(1000, size=(50000, 2))

# some_pt = random.randint(1000, size=(100, 2))




# dd = pairwise_distances_argmin(some_pt, a)
from scipy.spatial import KDTree


s_total = time.time()
x, y = np.meshgrid(new_lon, new_lat)
points = FTranspose(x, y)[mask]

points_amsr = FTranspose(lon, lat)
points_amsr = np.where(points_amsr>10000000, 10000, points_amsr)

s_arg_dd = time.time()
arg_dd = pairwise_distances_argmin(points, points_amsr)
print('pairwise_distances_argmin', time.time()-s_arg_dd)

s_kd = time.time()
kdtree = KDTree(points_amsr)
d, i = kdtree.query(points)
print('kd', time.time()-s_kd)

unique, counts = np.unique(arg_dd, return_counts=True)




mask_original = np.zeros([721, 721]).flatten()

mask_original[unique] =+ counts
mask_original /= np.sum(mask_original)

mask_original = mask_original.reshape([721, 721])
print(time.time()-s_total)









