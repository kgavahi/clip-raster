# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:06:12 2023

@author: kgavahi
"""

#test timing
import numpy as np
import shapefile
from matplotlib.path import Path




shp_path = 'shpfiles/ACF_basin.shp'

shp = shapefile.Reader(shp_path)
tupVerts = shp.shapes()[0].points

# Create a mask for the shapefile
xf, yf = x.flatten(), y.flatten()
points = np.vstack((xf,yf)).T 
p = Path(tupVerts) # make a polygon
grid = p.contains_points(points)
mask = grid.reshape(x.shape[0],x.shape[1])