# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:00:50 2020

@author: 
    original version (Matlab):  Jerome Gourrion - OceanScope - jerome.gourrion@ocean-scope.com
    python version:             Marine Gallian  - OceanScope - marine.gallian@ocean-scope.com
"""
import scipy.io as spio
from Tools_MinMax_ISEA import lonlat2indexISEA, val2index #from function_ISEA 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import PolyCollection
import matplotlib.colorbar as cbar


""" Define path and data to compute """
# ISEA file
DIR_ISEA_AUXfiles = 'AUX_files/'
ISEA_type = '4H6'

# load lat lon from netcdf file
# caution: the lon/lat vectors must be declared as numpy arrays. Example:
#    lon = np.array([146.347])
#    lat = np.array([13.4111])
data = xr.open_dataset('./AUX_files/CO_DMQCGL01_20190416_PR_PF.nc')
lon = data.LONGITUDE.data
lat = data.LATITUDE.data


""" run  """
pres = data.PRES.data
psal = data.PSAL.data
temp = data.TEMP.data
n_prof = data.dims['N_PROF']

# load min max file
minmax_psal = xr.open_dataset('./REFERENCE_FILES/PSAL_MIN_MAX.nc')
minmax_temp = xr.open_dataset('./REFERENCE_FILES/TEMP_MIN_MAX.nc')
minmax_grid = xr.open_dataset('./REFERENCE_FILES/GRID_MIN_MAX.nc')

# loading matlab structure
print('Loading ISEA-' + ISEA_type + ' info file ...')
info_DGG = spio.loadmat(DIR_ISEA_AUXfiles+'info_DGG'+ISEA_type+'.mat')

# Get ISEA ID 
hgrid_id = lonlat2indexISEA(lon,lat,info_DGG,ISEA_type)-1;

# Initialize
layer_id = np.empty((pres.shape))*np.nan
psal_min = np.empty((pres.shape))*np.nan
psal_max = np.empty((pres.shape))*np.nan
temp_min = np.empty((pres.shape))*np.nan
temp_max = np.empty((pres.shape))*np.nan

for i in range(0,n_prof):
    # Compute depth index
    layer_id[i,:] = val2index(pres[i,:],minmax_psal.depth.data)
    nonan = ~np.isnan(layer_id[i,:])
    # retrieve Min/Max values from reference files
    psal_min[i,np.where(nonan)[0]] = minmax_psal.psal_min.data[hgrid_id[i],layer_id[i,nonan].astype(int)]
    psal_max[i,np.where(nonan)[0]] = minmax_psal.psal_max.data[hgrid_id[i],layer_id[i,nonan].astype(int)]
    temp_min[i,np.where(nonan)[0]] = minmax_temp.temp_min.data[hgrid_id[i],layer_id[i,nonan].astype(int)]
    temp_max[i,np.where(nonan)[0]] = minmax_temp.temp_max.data[hgrid_id[i],layer_id[i,nonan].astype(int)]


# example 1 : plot min max for a profile
k = 12   # profile number to be displayed
plt.figure(1,figsize = (9, 5))
plt.subplot(121)
plt.plot(psal[k,:],-pres[k,:],'k',label='psal',)
plt.plot(psal_min[k,:],-pres[k,:],'b')
plt.plot(psal_max[k,:],-pres[k,:],'b')
plt.legend()
plt.subplot(122)
plt.plot(temp[k,:],-pres[k,:],'r',label='temp',)
plt.plot(temp_min[k,:],-pres[k,:],'b')
plt.plot(temp_max[k,:],-pres[k,:],'b')
plt.legend()

# example 2 : plot global map of PSAL minimum
imap=0 # level to be displayed
lonv = info_DGG['vertices'][0,0]['lon']
latv = info_DGG['vertices'][0,0]['lat']
poly = np.swapaxes(np.array([lonv,latv]), 0, 2)
Pmin=minmax_psal.psal_min[:,imap].data # plotting PSAL min
normal = plt.Normalize(32,38) #Min Max value 
cmap = plt.cm.jet(normal(Pmin))
fig1 = plt.figure(2,figsize = (9, 5))
plt.axis('off')
plt.title('PSAL min\n imap: ' + str(imap))
ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', linewidth=0.5,edgecolor='black', facecolor='grey'))
coll = PolyCollection(poly, facecolor=cmap, edgecolors=cmap)
ax1.add_collection(coll)
ax1 = fig1.add_axes([0.12, 0.10, 0.78, 0.06])
cbar.ColorbarBase(ax1, cmap=plt.cm.jet,norm=normal,orientation='horizontal')
plt.show()
