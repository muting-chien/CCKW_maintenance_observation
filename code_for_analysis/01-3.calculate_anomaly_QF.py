####################
# This code is to calculate the anomaly for the 3-hourly Q, F data 
# Remove diurnal cycle
# Remove ann_cycle
# 2022.11.11
# Mu-Ting Chien
#####################
import sys
sys.path.append('/home/disk/eos9/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy import signal
import os 
import xarray as xr

DIR = '/home/disk/eos9/muting/'
figdir = DIR+'KW/figure/RA_EAPEG_EKEG/test_anomaly/'
dir_in = DIR+'KW/output_data/RA_EAPEG_EKEG/3hr/'#Q location
plot_test_fig = 1
save_data = 1

vname    = list(['Q','F'])
vname_nc = list(['Q1','F'])
vname_long = list(['diabatic heating','momentum flux convergence'])
unit = list(['K/day','m^2/s^2'])
nv = np.size(vname)
file_in = list(['Q1_1000_100_byT_15SN_3hr.nc','F_eddy_momentum_forcing_3hr.nc'])

for v in range(1,2):
    print(vname[v])
    dir_out = dir_in+vname[v]+'ano_plev/'
    os.makedirs(dir_out,exist_ok=True)
    ###################
    # load Q, F data
    data = Dataset( dir_in+file_in[v], "r", format="NETCDF4")
    lat  = data.variables['lat'][:]
    lon  = data.variables['lon'][:]
    plev = data.variables['plev'][:]
    if v==0:
        time = data.variables['time'][7:-1-6]
    elif v==1:
        time = data.variables['time'][:]
    '''
    elif v==1:
        ds = xr.open_mfdataset(dir_in+file_in[v])
        lat = ds['lat']
        lon = ds['lon']
        plev = ds['plev']
        time = ds['time'] 
    '''
    
    nlat = np.size(lat)
    nlon = np.size(lon)
    nlev = np.size(plev)
    nt = np.size(time)
    
    for ilev in range(0,nlev):
        plev_str = str(int(plev[ilev]))
        print(plev_str)
        if ilev!=0:
            data = Dataset( dir_in+file_in[v], "r", format="NETCDF4")
        if v == 0:
            V = data.variables[vname_nc[v]][7:-1-6,ilev,:,:]
        elif v == 1:
            V = data.variables[vname_nc[v]][:,ilev,:,:]
            
        #elif v == 1:
        #    V = ds[vname_nc[v]][:,ilev,:,:]
        del data
        print('finish loading data')

        #################
        # Remove diurnal, seasonal cycle (use daily mean to obtain seasonal cycle), and linear trend
        dt = 8 #how many data per day
        nday = int(nt/dt)
        V_reshape = V.reshape(nday, dt, nlat, nlon)
        diurnal_cyc = np.tile( np.nanmean(V_reshape,0).squeeze(),(nday,1,1,1))
        del V_reshape
        diurnal_cyc_flat = np.reshape(diurnal_cyc,(nday*dt, nlat, nlon))
        del diurnal_cyc
        V_ano = V-diurnal_cyc_flat #(nday*dt, nlat, nlon)
        Vref = V[:,1,1]
        dcyc_ref = diurnal_cyc_flat[:,1,1]
        del V, diurnal_cyc_flat
        print('finish removing dirunal cyc')

        if plot_test_fig == 1:
            # Plot removing diurnal cycle
            t = np.arange(0,dt*10) #10 days
            plt.plot(t,Vref[t],'k-o')
            plt.plot(t,dcyc_ref[t],'b-o')
            plt.legend(['raw','diurnal cycle'])
            plt.title(vname[v]+' '+plev_str+'hPa')
            plt.xlabel('hours')
            plt.savefig(figdir+vname[v]+'_timeseries_diurnal_cyc_'+plev_str+'.png')
            #plt.show()
            plt.close()

        ####################
        # Remove annual cycle
        V_ano_final, cyc_final = MJO.remove_anncycle_3d( signal.detrend(V_ano,0), time, lat, lon, 1/8 )
        #print(np.shape(V_ano_final))
        #print(nt, nlat, nlon)
        del V_ano
        print('finish removing seasonal cycle')

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
            plt.suptitle(plev_str+'hPa')
            plt.savefig(figdir+vname[v]+'_timeseries_seasonal_cyc_ano_'+plev_str+'.png')
            #plt.show()
            plt.close()

        del Vref, cyc_final

        #################
        # Save data
        #############
        if save_data == 1:
            if v == 0:
                np.savez(dir_out+vname[v]+'_ano_3hr_'+plev_str+'hPa.npz',Q_ano=V_ano_final)
            elif v == 1:
                np.savez(dir_out+vname[v]+'_ano_3hr_'+plev_str+'hPa.npz',F_ano=V_ano_final)
            
            '''
            #file_out = dir_in+'T_ano_2.5SN_3hr.nc'
            file_out = dir_out+vname[v]+'_ano_3hr_'+plev_str+'hPa.nc'
            ncout = Dataset(file_out, 'w', format='NETCDF4')
            # define axis size
            ncout.createDimension('time', nt)
            ncout.createDimension('lat',  nlat)
            ncout.createDimension('lon',  nlon)
            #ncout.createDimension('plev', nlev)
            # create time axis
            time2 = ncout.createVariable('time', dtype('double').char, ('time'))
            time2.long_name = 'time'
            time2.units = 'hours since 1900-1-1 00:00:00'
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
            #lev2 = ncout.createVariable('plev', dtype('double').char, ('plev'))
            #lev2.standard_name = 'plev'
            #lev2.long_name = 'pressure'
            #lev2.units = 'hPa'
            #lev2.axis = 'Pa'
            # create variables
            V1out = ncout.createVariable(vname[v]+'_ano', dtype('double').char, ('time', 'lat', 'lon'))
            V1out.long_name = 'ERA5 '+vname_long[v]+' anomaly (remove diurnal cycle, seasonal cycle, and linear trend)'
            V1out.units = unit[v]

            # copy variable
            time2[:] = time[:]
            print(time)
            lon2[:] = lon[:]
            print(lon)
            lat2[:] = lat[:]
            print(lat)
            #lev2[:] = plev[:]
            print(V_ano_final[:5,0,0])
            V1out[:] = V_ano_final[:]
            '''
            del V_ano_final

            print('finish saving anomaly')