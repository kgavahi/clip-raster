import numpy as np
import shapefile
from matplotlib.path import Path
from inpoly import inpoly2
from numba import jit
from sklearn.metrics import pairwise_distances_argmin
from scipy.spatial import KDTree

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
        # TODO add Multidimensional >3D
        #assert raster.ndim == 2, "raster must be a 2D array"

        self.raster = raster
        self.lat = lat
        self.lon = lon
        self.cell_size = cell_size
        
        
        # dimensions along lat and lon
        if lat.ndim == 1:
    
            self.shape = (len(lat), len(lon))

        if lat.ndim == 2:
    
            self.shape = (lat.shape[0], lat.shape[1])

    def mask_shp(self, shp_path: str, scale_factor=1):
        assert scale_factor >= 1, "scale_factor is less than one"
        """
        
        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.

        Returns
        -------
        weights : nparray
            a mask array in which cells outside the shapefile are zero and the
            inside cells are the percentage of the area covered by the shapefile.

        """
        # TODO: assert that the shapefile file has only one shapefile in it.
        shp = shapefile.Reader(shp_path)

        # Get the polygon vertices of the basin
        tupVerts = shp.shapes()[0].points

        if scale_factor == 1:

            # Create a mask for the shapefile
            mask = mask_with_vert_points(tupVerts, self.lat, self.lon)
            mask = mask.reshape(self.shape)

            weights = mask / np.sum(mask)

        else:
            # Get the boundries of the basin
            tupVerts_np = np.array(tupVerts)
            up = np.max(tupVerts_np[:, 1])
            down = np.min(tupVerts_np[:, 1])
            left = np.min(tupVerts_np[:, 0])
            right = np.max(tupVerts_np[:, 0])
            
                
            # Create new coordinates for the downscaled grid
            new_lon = np.arange(left, right, self.cell_size/scale_factor)
            new_lat = np.arange(down, up, self.cell_size/scale_factor)
            
            
            # Create a mask for the shapefile
            mask_new = mask_with_vert_points(tupVerts, new_lat, new_lon)
            
            weights = np.zeros(self.shape)
            
            weights = CalW2(mask_new, 
                                  self.lat, self.lon, 
                                  new_lat, new_lon, 
                                  self.shape)
            
            mask = weights > 0          
            

            # # Create a mask for the shapefile
            # mask = mask_with_vert_points(tupVerts, new_lat, new_lon)
            # mask = mask.reshape(new_lat.shape[0], new_lon.shape[0])

            # mask_true = np.where(mask)

            # mask_original = np.zeros(self.shape)
            
            # mask_original = CalW(mask_original, mask_true, 
            #                       self.lat, self.lon, new_lat, new_lon)
            



        return weights, mask

    def clip(self, shp_path: str, scale_factor=1, drop=True):
        #TODO: also return lat and lons for plotting purposes
        """


        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. If you want to include cells that are covered by 
            even smaller areas of the shapefile, increase the scale_factor.
            The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.
        drop : bool, optional
            if true the margins will be removed to get a smaller raster after clip.
            if False, the size will not change but outside values will be nan.

        Returns
        -------
        raster_cropped : nparray
            clipped raster using the shapefile with nan values assigned to out
            of shapefile values.

        """
        weights, mask = self.mask_shp(shp_path, scale_factor)
        
        lat_dim = 0
        lon_dim = 1
        
        dims = np.arange(len(self.raster.shape))
        
        dims = tuple(dims[~((dims==lat_dim) | (dims==lon_dim))])



        
        mask_ex = np.broadcast_to(np.expand_dims(mask, dims), self.raster.shape)
        
        # dd = self.raster[:, :, :, mask, :, :]
        # print(self.raster.shape)
        # print(dd.shape)
        # print(mask.shape)
        

    
        

        raster_cropped = np.where(mask_ex, self.raster, np.nan)
        

        
        
        
        if self.lat.ndim == 1:

            x, y = np.meshgrid(self.lon, self.lat)

        if self.lat.ndim == 2:
            x = self.lon
            y = self.lat
        
        lat_cropped = np.where(mask, y, np.nan)
        lon_cropped = np.where(mask, x, np.nan)
        #TODO: how to add drop=True???
        


        return raster_cropped, lat_cropped, lon_cropped
    
    def get_mean(self, shp_path: str, scale_factor=1):
        """


        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.

        Returns
        -------
        float
            weighted average of cell values over the shapefile.

        """

        weights, mask = self.mask_shp(shp_path, scale_factor)
        
        # Define the axis mapping for time_axis
        lat_dim = 3
        lon_dim = 4   

        dims = np.arange(len(self.raster.shape))
        
        dims = tuple(dims[~((dims==lat_dim) | (dims==lon_dim))])



        
        weights = np.broadcast_to(np.expand_dims(weights, dims), self.raster.shape)
        
        #weights = np.expand_dims(weights, axis=(lat_dim, lon_dim))
        

        return np.nansum(weights * self.raster, axis=(lat_dim, lon_dim))
    
    def clip2d(self, shp_path: str, scale_factor=1, drop=True):
        #TODO: also return lat and lons for plotting purposes
        """


        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. If you want to include cells that are covered by 
            even smaller areas of the shapefile, increase the scale_factor.
            The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.
        drop : bool, optional
            if true the margins will be removed to get a smaller raster after clip.
            if False, the size will not change but outside values will be nan.

        Returns
        -------
        raster_cropped : nparray
            clipped raster using the shapefile with nan values assigned to out
            of shapefile values.

        """
        weights, mask = self.mask_shp(shp_path, scale_factor)


        raster_cropped = np.where(mask, self.raster, np.nan)
        lat_cropped = self.lat
        lon_cropped = self.lon
        
        
        if drop:

            nan_cols = np.all(~mask, axis=0)
            nan_rows = np.all(~mask, axis=1)
            raster_cropped = raster_cropped[:, ~nan_cols][~nan_rows]

            if self.lat.ndim == 2:
                lat_cropped = self.lat[:, ~nan_cols][~nan_rows]
                lon_cropped = self.lon[:, ~nan_cols][~nan_rows]

            if self.lat.ndim == 1:
                lat_cropped = self.lat[~nan_rows]
                lon_cropped = self.lon[~nan_cols]

        return raster_cropped, lat_cropped, lon_cropped

    def clip3d(self, shp_path: str, time_axis: int, scale_factor=1, drop=True):
        """


        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        time_axis : int
            the axis that represents time.  
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. If you want to include cells that are covered by 
            even smaller areas of the shapefile, increase the scale_factor.
            The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.
        drop : bool, optional
            if true the margins will be removed to get a smaller raster after clip.
            if False, the size will not change but outside values will be nan.

        Returns
        -------
        raster_cropped : nparray
            clipped raster using the shapefile with nan values assigned to out
            of shapefile values.

        """
        weights, mask = self.mask_shp(shp_path, scale_factor)



        mask_ex = np.expand_dims(mask, axis=time_axis)
        raster_cropped = np.where(mask_ex, self.raster, np.nan)
        lat_cropped = self.lat
        lon_cropped = self.lon
        
        if drop:
            nan_cols = np.all(~mask, axis=0)
            nan_rows = np.all(~mask, axis=1)
            raster_cropped = raster_cropped[:, :, ~nan_cols][:, ~nan_rows] if time_axis == 0 else \
                            raster_cropped[:, :, ~nan_cols][~nan_rows] if time_axis == 1 else \
                            raster_cropped[:, ~nan_cols, :][~nan_rows]        
                            
            if self.lat.ndim == 2:
                lat_cropped = self.lat[:, ~nan_cols][~nan_rows]
                lon_cropped = self.lon[:, ~nan_cols][~nan_rows]

            if self.lat.ndim == 1:
                lat_cropped = self.lat[~nan_rows]
                lon_cropped = self.lon[~nan_cols]

        return raster_cropped, lat_cropped, lon_cropped

    def get_mean2d(self, shp_path: str, scale_factor=1):
        """


        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.

        Returns
        -------
        float
            weighted average of cell values over the shapefile.

        """

        weights, mask = self.mask_shp(shp_path, scale_factor)
        

        return np.nansum(weights * self.raster)

    def get_mean3d(self, shp_path: str, time_axis: int, scale_factor=1):
        """


        Parameters
        ----------
        shp_path : str
            path to the shapefile.
        scale_factor : int, optional
            higher scale factor will result in higher computational cost but also
            higher accuracy. The default is 1 which means no downscaling and
            the center of the pixel must be inside the polygon to be added to 
            the mask.

        Returns
        -------
        float
            weighted average of cell values over the shapefile.

        """

        weights, mask = self.mask_shp(shp_path, scale_factor)
        
        # Define the axis mapping for time_axis
        axis_map = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        
        weights = np.expand_dims(weights, axis=time_axis)
        

        return np.nansum(weights * self.raster, axis=axis_map[time_axis])


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def CalW(mask_original, mask_true, lat, lon, new_lat, new_lon):
    for i, j in zip(mask_true[0], mask_true[1]):
        #print(i)
        if lat.ndim == 1:
            abs_lat = np.abs(new_lat[i] - lat)
            arg_lat = np.argmin(abs_lat)
    
            abs_lon = np.abs(new_lon[j] - lon)
            arg_lon = np.argmin(abs_lon)
    
        if lat.ndim == 2:
            d = ((new_lat[i] - lat) **
                        2 + (new_lon[j] - lon)**2)**0.5
            arg_d = np.where(d == np.nanmin(d))
            arg_lat = arg_d[0][0]
            arg_lon = arg_d[1][0]
    
        mask_original[arg_lat, arg_lon] += 1
    
    mask_original /= np.sum(mask_original)   
    
    return mask_original


def CalW2(mask, lat, lon, new_lat, new_lon, shape):
    
    
    weights = np.zeros(shape)
    
    x, y = np.meshgrid(new_lon, new_lat)
    points = FTranspose(x, y)[mask]

    points_product = FTranspose(lon, lat)
    
    #arg_dd = pairwise_distances_argmin(points, points_product)
    
    kdtree = KDTree(points_product)
    d, arg_dd = kdtree.query(points)
    
    unique, counts = np.unique(arg_dd, return_counts=True)
    
    weights = weights.flatten()
    
    weights[unique] =+ counts
    
    weights /= np.sum(weights)
    
    weights = weights.reshape(shape)
    
    return weights


def FTranspose(lon, lat):
   
    if lat.ndim == 1:

        x, y = np.meshgrid(lon, lat)

    if lat.ndim == 2:
        x = lon
        y = lat
        
    xf, yf = x.flatten(), y.flatten()

    # TODO here can be more optimization
    #points = np.vstack((xf,yf)).T
    #points = np.transpose((xf, yf))
    points = np.column_stack((xf,yf))    
    
    return points


def mask_with_vert_points(tupVerts, lat, lon, mode='inpoly'):


    # if mode == 'matplotlib':
    #     # Create a mask for the shapefile
    #     points = FTranspose(lon, lat)
    #     p = Path(tupVerts)  # make a polygon
    #     grid = p.contains_points(points)
    #     # now you have a mask with points inside a polygon
    #     mask = grid.reshape(x.shape[0], x.shape[1])

    if mode == 'inpoly':

        
        points = FTranspose(lon, lat)

        # use inpoly which lightning fast
        isin, ison = inpoly2(points, tupVerts)
        #mask = isin.reshape(x.shape[0], x.shape[1])

    return isin
