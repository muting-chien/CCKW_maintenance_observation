####################
# This code is to calculate the anomaly for the 3-hourly cwv data from era5
# Remove diurnal cycle
# Remove ann_cycle
# 2023.12.28
# Mu-Ting Chien
#####################
import sys
sys.path.append('/home/disk/eos4/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import signal
import os 

DIR = '/home/disk/eos4/muting/'
dir_in = DIR+'data/combine_reanalysis/era5/regrid_2.5deg_3hr/'
latmax = 15 
plot_test_fig = 1
save_data = 1

vname    = list(['cwv','pr','LH'])
vname_nc = list(['tcwv','tp','slhf'])
vname_long = list(['column water vapor','precipitation','surface latent heat flux'])
unit = list(['g/kg','mm/day','W/m^2'])
nv = np.size(vname)

for v in range(2,3):#nv):
    print(vname[v])
    dir_out = dir_in
    os.makedirs(dir_out,exist_ok=True)
    
    figdir = DIR+'KW/figure/RA_EAPEG_EKEG/test_anomaly/'+vname[v]+'/'
    os.makedirs(figdir,exist_ok=True)
    #print(dir_out)
    ###################
    # load era5 data
    file_in = dir_in+'regrid_'+vname[v]+'.nc' # trange=1998-2013
    data = Dataset( file_in, "r", format="NETCDF4")
    lat  = data.variables['latitude'][:]
    lon  = data.variables['longitude'][:]
    nlat = np.size(lat)
    nlon = np.size(lon)

    # trimm narrower tropical band for space-time spectrum calculation
    latmin = -latmax
    dmin = np.abs(lat-latmin)
    dmax = np.abs(lat-latmax)
    imax = np.argwhere(dmin==np.min(dmin)).squeeze()
    imin = np.argwhere(dmax==np.min(dmax)).squeeze()
    lat_tropics = lat[imin:imax+1]
    nlat_tropics = np.size(lat_tropics)
    #print(lat_tropics)
    
    time = data.variables['time'][:]
    nt = np.size(time)
    
    V = data.variables[vname_nc[v]][:,imin:imax+1,:][:,::-1,:] 
    
    if v == 1:
        V = V*24*1000 #(change m/hr to mm/day)
    elif v == 2:
        V = V/3600 #(J/m^2 over 1hr convert to w/m^2)
    
    lat_tropics = lat_tropics[::-1]
    print(lat_tropics)
    #del data

    #################
    # Remove diurnal, seasonal cycle (use daily mean to obtain seasonal cycle), and linear trend
    dt = 8 #how many data per day
    nday = int(nt/dt)
    V_reshape = V.reshape(nday, dt, nlat_tropics, nlon)
    diurnal_cyc = np.tile( np.nanmean(V_reshape,0).squeeze(),(nday,1,1,1))
    del V_reshape
    diurnal_cyc_flat = np.reshape(diurnal_cyc,(nday*dt, nlat_tropics, nlon))
    del diurnal_cyc
    V_ano = V-diurnal_cyc_flat #(nday*dt, nlat_tropics, nlon)
    Vref = V[:,1,1]
    dcyc_ref = diurnal_cyc_flat[:,1,1]
    del V, diurnal_cyc_flat

    if plot_test_fig == 1:
        # Plot removing diurnal cycle
        t = np.arange(0,dt*10) #10 days
        plt.plot(t,Vref[t],'k-o')
        plt.plot(t,dcyc_ref[t],'b-o')
        plt.legend(['raw','diurnal cycle'])
        plt.title(vname[v])
        plt.xlabel('hours')
        plt.savefig(figdir+vname[v]+'_timeseries_diurnal_cyc.png')
        #plt.show()
        plt.close()

    ####################
    # Remove annual cycle
    V_ano_final, cyc_final = MJO.remove_anncycle_3d( signal.detrend(V_ano,0), time, lat_tropics, lon, 1/8 )
    del V_ano

    if plot_test_fig == 1:
        # Plot removing annual cycle
        ts = np.arange(0,365*8*2)
        plt.subplot(2,1,1)
        plt.plot(ts, Vref[ts], 'k')
        plt.plot(ts, cyc_final[ts,1,1]+dcyc_ref[ts], 'g')
        plt.legend([vname[v]+' raw','diurnal+seasonal cyc'])
        plt.subplot(2,1,2)
        plt.plot(ts, V_ano_final[ts,1,1], 'r')
        plt.legend(['ano'])
        plt.savefig(figdir+vname[v]+'_timeseries_seasonal_cyc_ano.png')
        #plt.show()
        plt.close()

    del Vref, cyc_final

    # Rearrange longitude
    nlon_mid = int(nlon/2)
    V_ano_final2    = np.empty([nt, nlat_tropics, nlon])
    V_ano_final2[:,:,nlon_mid:] = V_ano_final[:,:,:nlon_mid]
    V_ano_final2[:,:,:nlon_mid] = V_ano_final[:,:,nlon_mid:]
    
    # Load time data
    file_in = dir_in+'T_ano_3hr.nc'
    data = Dataset( file_in, "r", format="NETCDF4")
    time = data.variables['time'][:]
    
    #################
    # Save data
    #############
    if save_data == 1:
        file_out = dir_out+vname[v]+'_ano_3hr.nc'
        ncout = Dataset(file_out, 'w', format='NETCDF4')
        # define axis size
        ncout.createDimension('time', nt)
        ncout.createDimension('lat',  nlat_tropics)
        ncout.createDimension('lon',  nlon)
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
        # create variables
        V1out = ncout.createVariable(vname[v]+'_ano', dtype('double').char, ('time', 'lat', 'lon'))
        V1out.long_name = 'ERA5 '+vname_long[v]+' anomaly (remove diurnal cycle, seasonal cycle, and linear trend)'
        V1out.units = unit[v]

        # copy variable
        time2[:] = time[:]
        lon2[:] = lon[:]
        lat2[:] = lat_tropics[:]
        V1out[:] = V_ano_final2[:]

        del V_ano_final, V_ano_final2

        print('finish saving anomaly')