####################
# Combine npz file for each level into 1 nc file
# Make sure all the anomaly data has the time unit yyyymmddhh 
# and that longitude is 0-360
# 2022.11.1
# Mu-Ting Chien
#######################
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import os

DIR = '/home/disk/eos4/muting/'
dir_in = DIR+'KW/output_data/RA_EAPEG_EKEG/3hr/'
dir_in0 = DIR+'data/combine_reanalysis/era5/regrid_2.5deg_3hr/'
vname  = list(['T','u','v','w','q','Q','F','gph'])
vname_long = list(['temperature','zonal wind','meridioinal wind','pressure vertical velocity (omega)','specific humidity',\
                  'diabatic heating','momentum forcing','geopotential height'])

unit = list(['K','m/s','m/s','Pa/s','g/kg','K/day','m^2/s^2','m'])
g = 9.8
#tmin = 19980101
#tmax = 20131231

for v in range(7,8):#np.size(vname)):
    
    print(vname[v])
    if v==5 or v==6:
        dir_in2 = dir_in+vname[v]+'ano_plev/'
    else:
        dir_in2 = dir_in0+vname[v]+'ano_plev/'

    # load dimension (use the sample nc file)
    if v==5:
        file_in = dir_in+'Q1_1000_100_byT_15SN_3hr.nc'
    elif v==6:
        file_in = dir_in+'F_eddy_momentum_forcing_3hr.nc'
    else:
        file_in = dir_in0+'T_ano_3hr.nc'
        
    data = Dataset( file_in, "r", format="NETCDF4")
    lat  = data.variables['lat'][:]
    lon  = data.variables['lon'][:]
    plev = data.variables['plev'][:]
    
    if v==5:
        time = data.variables['time'][7:-1-6]   
    else:
        time = data.variables['time'][:]
    del data
    
    if v == 7: # only 15S-15N for gph, 30S-30N for other variables
        imin = np.argwhere(lat==-15).squeeze()
        imax = np.argwhere(lat==15).squeeze()
        lat = lat[imin:imax+1]
        #print(lat)
    
    nlat = np.size(lat)
    nlev = np.size(plev)
    nlon = np.size(lon)
    nt   = np.size(time)
    ano = np.empty([nt,nlev,nlat,nlon])
    
    # Load data
    for ilev in range(0,nlev):
        plev_str = str(int(plev[ilev]))
        print(plev_str)
        file = dir_in2+vname[v]+'_ano_3hr_'+plev_str+'hPa.npz'
        data = np.load(file)
        if v==5 or v==6:
            ano[:,ilev,:,:] = data[vname[v]+'_ano']
        else:
            ano[:,ilev,:,:] = data['w_ano']
        del data
        
    if v == 7:
        # Change unit to m
        ano = ano/g
        
        # reverse latitude dimension, original one is 15~-15
        ano = ano[:,:,::-1,:] #(-15~15)
        
        # change longitude dimension to 0~360, original one is -180~180, 
        nlon_mid = int(nlon/2)
        ano_new    = np.empty([nt, nlev, nlat, nlon])
        ano_new[:,:,:,nlon_mid:] = ano[:,:,:,:nlon_mid]
        ano_new[:,:,:,:nlon_mid] = ano[:,:,:,nlon_mid:]
    
    ################################
    # Save data
    file_out = dir_in0+vname[v]+'_ano_3hr.nc'
    ncout = Dataset(file_out, 'w', format='NETCDF4')
    # define axis size
    ncout.createDimension('time', nt)
    ncout.createDimension('lat',  nlat)
    ncout.createDimension('lon',  nlon)
    ncout.createDimension('plev', nlev)
    # create time axis
    time2 = ncout.createVariable('time', dtype('double').char, ('time'))
    time2.long_name = 'time'
    time2.units = 'YYYYMMDDHH'
    time2.calendar = 'standard'
    time2.axis = 'T'
    # create latitude axis
    lat2 = ncout.createVariable('lat', dtype('double').char, ('lat'))
    lat2.standard_name = 'lat'
    lat2.long_name = 'latitude'
    lat2.units = 'degrees_north'
    lat2.axis = 'Y'
    # create longitude axis
    lon2 = ncout.createVariable('lon', dtype('double').char, ('lon'))
    lon2.standard_name = 'lon'
    lon2.long_name = 'longitude'
    lon2.units = 'degrees_east'
    lon2.axis = 'X'
    # create level axis
    lev2 = ncout.createVariable('plev', dtype('double').char, ('plev'))
    lev2.standard_name = 'plev'
    lev2.long_name = 'pressure'
    lev2.units = 'hPa'
    lev2.axis = 'Pa'
    # create variables
    V1out = ncout.createVariable(vname[v]+'_ano', dtype('double').char, ('time','plev', 'lat', 'lon'))
    V1out.long_name = 'ERA5 '+vname_long[v]+' anomaly (remove diurnal cycle, seasonal cycle, and linear trend)'
    V1out.units = unit[v]

    # copy variable
    time2[:] = time[:]
    lon2[:]  = lon[:]
    lat2[:]  = lat[:]
    lev2[:]  = plev[:]
    if v!=7:
        V1out[:] = ano[:]
    else:
        V1out[:] = ano_new[:]
    print('finish saving data '+vname[v])
