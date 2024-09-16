####################
# This code is to calculate the anomaly for the 3-hourly Tb data
# This is the .py version (there is another .ipynb version, where you can see the figures)
# Remove diurnal cycle
# Remove ann_cycle
# 2022.10.10
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

DIR = '/home/disk/eos9/muting/'
dir_in = '/home/disk/eos9/muting/data/Tb/'
figdir = DIR+'KW/figure/mesoscale_CCEW/'
latmax = 27.5 # Do not use 30/ns because it contains nan
plot_test_fig = 0
save_data = 1

###################
# load Tb data
file_in = dir_in+'Tb_1983_2013_30SN_3hr.nc'
data = Dataset( file_in, "r", format="NETCDF4")
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
nlat = np.size(lat)
nlon = np.size(lon)

# trimm narrower tropical band for space-time spectrum calculation
latmin = -latmax
dmin = np.abs(lat-latmin)
dmax = np.abs(lat-latmax)
imin = np.argwhere(dmin==np.min(dmin)).squeeze()
imax = np.argwhere(dmax==np.min(dmax)).squeeze()
lat_tropics = lat[imin:imax+1]
nlat_tropics = np.size(lat_tropics)

time = data.variables['time'][:]
nt = np.size(time)
V = data.variables['brtmp'][:,imin:imax+1,:] # K
del data

#################
# Remove diurnal, seasonal cycle (use daily mean to obtain seasonal cycle), and linear trend
dt = 8 #how many data per day
nday = int(nt/dt)
V_reshape = V.reshape(nday, dt, nlat_tropics, nlon)
diurnal_cyc = np.tile( np.nanmean(V_reshape,0).squeeze(),(nday,1,1,1))
diurnal_cyc_flat = np.reshape(diurnal_cyc,(nday*dt, nlat_tropics, nlon))
V_ano = V-diurnal_cyc_flat #(nday*dt, nlat_tropics, nlon)

if plot_test_fig == 1:
    # Plot removing diurnal cycle
    t = np.arange(0,dt*10) #10 days
    plt.plot(t,V[t,0,0],'k-o')
    plt.plot(t,diurnal_cyc_flat[t,0,0],'b-o')
    plt.legend(['raw','diurnal cycle',])
    plt.xlabel('hours')
    plt.show()

####################
# Remove annual cycle
V_ano_final, cyc_final = MJO.remove_anncycle_3d( signal.detrend(V_ano,0), time, lat_tropics, lon, 1/8 )

if plot_test_fig == 1:
    # Plot removing annual cycle
    ts = np.arange(0,365*8*2)
    plt.subplot(2,1,1)
    plt.plot(ts, V[ts,1,1], 'k')
    plt.plot(ts, cyc_final[ts,1,1]+diurnal_cyc_flat[ts,1,1], 'g')
    plt.legend(['raw','diurnal+seasonal cyc'])
    plt.subplot(2,1,2)
    plt.plot(ts, V_ano_final[ts,1,1], 'r')
    plt.legend(['ano'])
    plt.show()
    
#################
# Save data
#############
if save_data == 1:
    file_out = dir_in+'Tb_ano_1983_2013_27.5SN_3hr.nc'
    ncout = Dataset(file_out, 'w', format='NETCDF4')
    # define axis size
    ncout.createDimension('time',nt)
    ncout.createDimension('lat', nlat_tropics)
    ncout.createDimension('lon', nlon)
    # create time axis
    time2 = ncout.createVariable('time', dtype('double').char, ('time',))
    time2.long_name = 'time'
    time2.units = 'days since 1980-01-01 00:00:00'
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
    V1out = ncout.createVariable('Tb_ano', dtype('double').char, ('time', 'lat', 'lon'))
    V1out.long_name = 'Brightness temperature anomaly (remove diurnal cycle, seasonal cycle, and linear trend)'
    V1out.units = 'K'

    # copy variable
    time2[:] = time[:]
    lon2[:] = lon[:]
    lat2[:] = lat_tropics[:]
    V1out[:] = V_ano_final[:]

    print('finish saving anomaly')