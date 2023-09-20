# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:49:03 2023

@author: kgavahi
"""
import numpy as np
import shapefile
from matplotlib.path import Path
class ClipRaster:
    """Clip raster class for clipping 2D raster files using a shapefile. 
    The clip raster class can also calculate weighted average over the shapefile
    based on the areas covered by each cell.
    
    Parameters
    ---------
    
    raster : nparray
        2D raster that needs to be clipped using a shapefile.
    lat : nparray
        2D or 1D array representing latitudes of cells in the raster.
    lon: nparray
        2D or 1D array representing longitudes of cells in the raster.
    shp_path: str
        path to the shapefile
    cell_size: float
        spatial resolution of the raster file.
    scale_factor: int
        higher scale factor will result in higher computational cost but also
        higher accuracy.
    
    """
    def __init__(self, raster, lat, lon, cell_size):
        
        assert isinstance(raster, np.ndarray), "raster is not a numpy array"
        assert isinstance(lat, np.ndarray), "lat is not a numpy array"
        assert isinstance(lon, np.ndarray), "lon is not a numpy array"
        assert raster.ndim == 2, "raster must be a 2D array"
        
        self.raster = raster
        self.lat = lat
        self.lon = lon
        self.cell_size = cell_size
        self.shape = raster.shape
        
    def clip(self, shp_path: str):
        """
        

        Parameters
        ----------
        shp_path : str
            path to the shapefile..

        Returns
        -------
        raster_cropped : nparray
            clipped raster using the shapefile with nan values assigned to out
            of shapefile values.

        """
        shp = shapefile.Reader(shp_path)
        
        # Get the polygon vertices of the basin
        tupVerts = shp.shapes()[0].points

        # Create a mask for the shapefile
        
        if self.lat.ndim==1:
            x, y = np.meshgrid(self.lon, self.lat)
            xf, yf = x.flatten(), y.flatten()
        if self.lat.ndim==2:    
            x = self.lon
            y = self.lat
            xf, yf = x.flatten(), y.flatten()
            
        points = np.vstack((xf,yf)).T 
        p = Path(tupVerts) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(x.shape[0],x.shape[1]) # now you have a mask with points inside a polygon
        
        raster_clipped = np.where(mask, self.raster, np.nan)
        
        
        nan_cols = np.all(~mask, axis=0)
        nan_rows = np.all(~mask, axis=1)

        
        raster_cropped = raster_clipped[:, ~nan_cols][~nan_rows]
        
        return raster_cropped                  
        
    def mask_shp(self, shp_path: str, scale_factor=25):
        """
        
        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : TYPE, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. The default is 25.

        Returns
        -------
        mask_original : nparray
            a mask array in which cells outside the shapefile are zero and the
            inside cells are the percentage of the area covered by the shapefile.

        """
        shp = shapefile.Reader(shp_path)
        
        # Get the polygon vertices of the basin
        tupVerts = shp.shapes()[0].points
        
        # Get the boundries of the basin
        tupVerts_np = np.array(tupVerts)
        up = np.max(tupVerts_np[:, 1]) + self.cell_size
        down = np.min(tupVerts_np[:, 1]) - self.cell_size
        left = np.min(tupVerts_np[:, 0]) - self.cell_size
        right = np.max(tupVerts_np[:, 0]) + self.cell_size

        # Create new coordinates for the downscaled grid
        new_lon = np.arange(left, right, 1/scale_factor)
        new_lat = np.arange(down, up, 1/scale_factor)
        
        # Create a mask for the shapefile
        x, y = np.meshgrid(new_lon, new_lat)
        xf, yf = x.flatten(), y.flatten()
        points = np.vstack((xf,yf)).T 
        p = Path(tupVerts) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(x.shape[0],x.shape[1]) # now you have a mask with points inside a polygon

        mask_true = np.where(mask)
               
        mask_original = np.zeros(self.shape)
        for i, j in zip(mask_true[0], mask_true[1]):
              
            
            if self.lat.ndim==1:
                abs_lat = np.abs(new_lat[i] - self.lat)
                arg_lat = np.argmin(abs_lat)
                
                abs_lon = np.abs(new_lon[j] - self.lon)
                arg_lon = np.argmin(abs_lon)
            
            if self.lat.ndim==2:
                d = np.sqrt((new_lat[i] - self.lat)**2 + (new_lon[j] - self.lon)**2)
                arg_d = np.where(d==np.nanmin(d))
                arg_lat = arg_d[0][0]
                arg_lon = arg_d[1][0]
            
            
            
            mask_original[arg_lat, arg_lon] += 1
            
        
        mask_original /= np.sum(mask_original)
        
        return mask_original
    
    def get_mean(self, shp_path: str, scale_factor=25):
        """
        

        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : TYPE, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. The default is 25.

        Returns
        -------
        float
            weighted average of cell values over the shapefile.

        """
        
        
        mask = self.mask_shp(shp_path, scale_factor)
        
        return np.nansum(mask * self.raster)
        
        
        
        
        
        
        
        
        
        
