#######################################
# Reproduce_Cheng_2022_*.py
# Goal: Reproduce Fig. 9 in Cheng et al. 2022 
# (EKE generation in wavenumber-frequency space-->Cross spectrum of u and F)
# The motivation for reproducing this figure is to make sure my calculation of EKEG is correct.
# 
# For this code, the specific goal is to calculate ua nd F
# 2022.9.22
# Mu-Ting
############################
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import os
import xarray as xr
from datetime import datetime

DIR = '/home/disk/eos9/muting/'
dir_out = DIR+'KW/'
dir_in = DIR+'data/combine_reanalysis/era5/regrid_2.5deg_3hr/'
dir_out_data = dir_out+'output_data/RA_EAPEG_EKEG/3hr/'
os.makedirs(dir_out_data,exist_ok=True)
#tmin = 19980101
#tmax = 20131231

################################
# load ERA5 and calculate anomaly for u, v, omega
###############################
# Load u
ds = xr.open_mfdataset(dir_in+'u_ano_3hr.nc')
plev = ds['plev']
lat  = ds['lat']
lon  = ds['lon']
time = ds['time']
plev = plev.values
lat  = lat.values
lon  = lon.values
time = time.values
'''
data   = Dataset(DIR_IN+'u_ano_3hr.nc','r',format='NETCDF4')
plev   = data.variables['plev'][:]
lat    = data.variables['lat'][:]
lon    = data.variables['lon'][:]
time   = data.variables['time'][:]
ua  = data.variables['u_ano'][:,:,:,:] #(time,plev,lat,lon)
del data
'''
nt   = np.size(time)
nlev = np.size(plev)
nlat = np.size(lat)
nlon = np.size(lon)

# Make sure plev in Pa
if np.max(plev)<=900*100:
    plev = plev*100
#print(plev)
F = np.empty([nt,nlev-2,nlat-2,nlon])
for it in range(0,nt):
    print(it)
    ds = xr.open_mfdataset(dir_in+'u_ano_3hr.nc')
    ua = ds['u_ano'][it,:,:,:] #(time,plev,lat,lon)
    ua = ua.values
    #
    print('finish loading data u')

    ##################################
    # Calculate F
    # F'=du'/dt + u'du/dx + v'du/dy + w'du/dp + udu'/dx + vdu'/dy + wdu'/dp -fv' + dphi'/dx
    # I used the below equation to calculate F=-( d(u'u')/dx + d(u'v')/dy + d(u'w')/dp)
    ##############################
    dx_temp = 2*2.5*111*1000*np.cos(lat[1:-1]*2*np.pi/360) #(m)
    dx = np.tile(dx_temp,(nlev-2,nlon,1))
    del dx_temp
    dx = np.transpose(dx,(0,2,1))
    duaua_dx = np.empty([nlev-2,nlat-2,nlon])
    duaua_dx[:,:,1:-1]   = ( ua[1:-1,1:-1,2:]**2-ua[1:-1,1:-1,:nlon-2]**2)
    duaua_dx[:,:,0]      = ( ua[1:-1,1:-1,1]**2-ua[1:-1,1:-1,-1]**2)
    duaua_dx[:,:,nlon-1] = ( ua[1:-1,1:-1,0]**2-ua[1:-1,1:-1,nlon-2]**2)
    duaua_dx = duaua_dx/dx #(nt,nlev-2,nlat-2,nlon)
    del dx
    F[it,:,:,:] = F[it,:,:,:]-duaua_dx 
    del duaua_dx
    print('d(ua*ua)/dx')

    ####################################
    # Load v
    ds = xr.open_mfdataset(dir_in+'v_ano_3hr.nc')
    va = ds['v_ano'][it,:,:,:] #(time,plev,lat,lon)
    va = va.values
    #
    dy = (lat[2]-lat[0])*111*1000 
    duava_dy = (ua[1:-1,2:,:]*va[1:-1,2:,:]\
                -ua[1:-1,:nlat-2,:]*va[1:-1,:nlat-2,:])/dy#np.empty([nt-2,nlev-2,nlat-2,nlon])
    del va
    F[it,:,:,:] = F[it,:,:,:] - duava_dy
    del duava_dy
    print('d(ua*va)/dy')

    ##########################################
    # Load w
    ds = xr.open_mfdataset(dir_in+'w_ano_3hr.nc')
    wa = ds['w_ano'][it,:,:,:] #(time,plev,lat,lon)
    wa = wa.values
    #
    duawa_dp = ua[2:,1:-1,:]*wa[2:,1:-1,:]\
                -ua[:nlev-2,1:-1,:]*wa[:nlev-2,1:-1,:] #([nt-2,nlev-2,nlat-2,nlon])
    dp = plev[2:]-plev[:nlev-2] #np.empty([nlev-2])
    for ip in range(0,nlev-2):
        duawa_dp[ip,:,:] = duawa_dp[ip,:,:]/dp[ip]
    F[it,:,:,:] = F[it,:,:,:] - duawa_dp
    
    del wa
    del duawa_dp
    print('d(ua*wa)/dp')

#######################
# Save F, not saving ua, but ua=ua[1:-1,1:-1,1:-1,:]
################
filename = dir_out_data+'F_eddy_momentum_forcing_3hr.nc'  
ncout = Dataset(filename, 'w', format='NETCDF4')

# define axis size
ncout.createDimension('plev', nlev-2)
ncout.createDimension('lat', nlat-2)
ncout.createDimension('lon', nlon)
ncout.createDimension('time', nt)

# create plev axis
plev2 = ncout.createVariable('plev', dtype('double').char, ('plev'))
plev2.standard_name = 'plev'
plev2.long_name = 'pressure level'    
plev2.units = 'hPa'
plev2.axis = 'Y'
# create loc axis
lat2 = ncout.createVariable('lat', dtype('double').char, ('lat'))
lat2.long_name = 'latitude'
lat2.units = 'deg'
lat2.axis = 'lat'
# create loc axis
lon2 = ncout.createVariable('lon', dtype('double').char, ('lon'))
lon2.long_name = 'longitude'
lon2.units = 'deg'
lon2.axis = 'lon'
# create time axis
time2 = ncout.createVariable('time', dtype('double').char, ('time'))
time2.long_name = 'time'
time2.units = 'yyyymmddhh'
time2.calendar = 'standard'
time2.axis = 'T'

F_out = ncout.createVariable('F', dtype('double').char, ('time','plev','lat','lon'))
F_out.long_name = 'eddy momentum forcing = -d(ua*ua)/dx -d(ua*va)/dy - d(ua*wa)/dp '
F_out.units = 'm/s^2'    
    
F_out[:] = F[:] 
lat2[:] = lat[1:-1]
lon2[:] = lon[:]
plev2[:] = plev[1:-1]
time2[:] = time[:]
print('finish saving')