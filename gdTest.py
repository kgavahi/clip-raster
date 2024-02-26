

from osgeo import gdal



src = gdal.Open('wrfout_d01_2012-01-01_00%3A00%3A00')
ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()


print(xres, yres)

print(src.GetGeoTransform())

