# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:08:25 2020

@author: marine gallian - OceanScope
"""
# import
import numpy as np
import matplotlib.path as mpltPath
import time

def lonlat2indexISEA(lon,lat,info_DGG4H,ISEA_type):
    # Add auxiliary information: list ISEA points per box
    if (ISEA_type == '4H7') | (ISEA_type == '4H6'):
       N0 = 10
    elif (ISEA_type == '4H5') | (ISEA_type == '4H4'):
       N0 = 20
    elif ISEA_type == '4H3':
       N0 = 30
    
    # Filter lat lon values
    latlon_filt = abs(lon)>999 
    if np.any(latlon_filt):
        lon[latlon_filt] = np.nan
        lat[latlon_filt] = np.nan
        del latlon_filt
    
    # Set lon between [-180 and 180]
    lon[lon>=180] = lon[lon>=180] - 360
    
    # Prepare box 
    ilon = np.floor((lon+180)/N0)+1
    ilat = np.floor((lat+90)/N0)+1
    condit=ilat>180/N0
    if sum(condit)>0:
          ilat[condit] = 180/N0
    ibox = np.ravel_multi_index(np.array([ilat,ilon]).astype(int),(int(180/N0),int(360/N0)),order='F',mode='wrap')
    
    kk = ~np.isnan(ibox)
    indexISEA = find_ISEA_box(lon[kk],lat[kk],np.vstack((ilat,ilon)).transpose(),ibox,info_DGG4H)

    return indexISEA

def find_ISEA_box(lon,lat,ilatlon,ibox,info_DGG4H):
    # initialise
    indexISEA = np.zeros(shape=lon.shape)
    iboxocc  = np.unique(ibox)
    for i in range(len(iboxocc)):
        list_ind = np.array(np.where(ibox == iboxocc[i]))[0]
        ll = len(list_ind)
        jlatlon = ilatlon[list_ind[0]].astype(int)    
        listISEAbox = info_DGG4H['list_ISEApts_in_boxes'][jlatlon[0]-1][jlatlon[1]-1][0][0].astype(int)
            
        # get lon lat
        clon = info_DGG4H['lon'][0][listISEAbox-1]
        clat = info_DGG4H['lat'][0][listISEAbox-1]
        lontt = lon[list_ind]
        lattt = lat[list_ind]

        # compute distance 
        dd = distance(lattt, lontt, np.repeat(clat[:,None],ll,axis=1),np.repeat(clon[:,None],ll,axis=1))
      
        isort = np.argsort(dd,axis=0)[0:3]
        list_bis=np.unique(listISEAbox[isort].flatten()-1) #ici
        
        for j in list_bis:

            #get lon lat
            clon = info_DGG4H['lon'][0][j]
            clat = info_DGG4H['lat'][0][j]
            # get vertices
            vlat0 = info_DGG4H['vertices']['lat'][0][0][:,j]
            vlon0 = info_DGG4H['vertices']['lon'][0][0][:,j]

            if vlon0.max() - vlon0.min()>100:
                vlon0[vlon0>0] = vlon0[vlon0>0] - 360
                
            ii = ~np.isnan(vlon0+vlat0)
            vlon0 = vlon0[ii]
            vlat0 = vlat0[ii]
    
            if abs(clat)!=90:
                vc1 = vlon0; vc2 = vlat0
                c1 = lontt; c2 = lattt
                if np.any(c1>max(vc1)):
                    c1[c1>max(vc1)] = c1[c1>max(vc1)]-360                    
                if np.any(c1<min(vc1)) :
                    c1[c1<min(vc1)] = c1[c1<min(vc1)]+360
                path = mpltPath.Path(np.vstack((vc1,vc2)).transpose())
                inside = path.contains_points(np.vstack((c1,c2)).transpose())
                indexISEA[list_ind[inside]] = j+1
                       
            else:
                ii = np.argsort(vlon0[0:-1])
                vc2 = vlat0[ii]
                vc1 = vlon0[ii]
                    
                vc1 = np.concatenate(([vc1[-1]-360], vc1, [vc1[0]+360], [vc1[0]+360],[vc1[-1]-360],[vc1[-1]-360]))
                vc2 = np.concatenate(([vc2[-1]], vc2, [vc2[0]], [clat+np.sign(clat)],[clat+np.sign(clat)],[vc2[-1]]));

                c1 = lontt
                c2 = lattt
                if np.any(c1>max(vc1)) :
                    c1[c1>max(vc1)] = c1[c1>max(vc1)]-360                    
                if np.any(c1<min(vc1)) != False:
                    c1[c1<min(vc1)] = c1[c1<min(vc1)]+360        
                path = mpltPath.Path(np.vstack((vc1,vc2)).transpose())
                inside = path.contains_points(np.vstack((c1,c2)).transpose())
                indexISEA[list_ind[inside]] = j+1                    
                    
    return indexISEA.astype(int)
        
def distance(lat1,lon1,lat2,lon2):
   lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
   a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2
   rng = 2 * np.arctan(np.sqrt(a)/np.sqrt(1 - a))
   return np.degrees(rng)


def val2index(VALs_orig,val_tab):
    nd = len(val_tab);
    VALs_size = VALs_orig.shape[0]; 
    VALs_m = np.ones((nd-1,VALs_size))*VALs_orig
    valm = val_tab[0:-1]; 
    valp = val_tab[1:]; 
    valm_m = np.transpose(np.ones((VALs_size,nd-1))*valm)
    valp_m = np.transpose(np.ones((VALs_size,nd-1))*valp)
    ii = np.transpose(np.ones((VALs_size,nd-1))*np.arange(0,nd-1))            
    is_ok = (VALs_m > valm_m) & (VALs_m <= valp_m); 
    index = sum(ii*is_ok); 
    VALs_out = (VALs_orig <= valm[0]) | (VALs_orig > valp[-1]); 
    index[VALs_out] = 'nan'; 
    index[np.isnan(VALs_orig)] = 'nan' 
    index = np.reshape(index,VALs_size); 
    return index