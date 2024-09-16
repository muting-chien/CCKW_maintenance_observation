###################################
# Calculate Q1 (output)
# Input: T, u, v, w, geopotential (4D fields)
# but now we use Q1 = [ dT/dt + udT/dx + vdT/dy + wdT/dp - alpha*w/Cp], alpha = P/(Rd*T)
# instead of Q1 = 1/Cp*[ dDSE/dt + udDSE/dx + vdDSE/dy + wdDSE/Dp ]
#
# !!!! Caution: This code takes a lot of time to run!
# Remember to make sure ths input data has the correct unit
# Remember to check the end product of Q1, make sure the magnitude makes sense
# It is easy to get crazy numbers if you use the input variables are not in the correct units. 
# 2022.11.1
#####################################
'''
import module
'''
import sys
sys.path.append('/home/disk/eos4/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os

'''
Input data information
'''
####################
# Caution: Change this if diff data
dir_out = '/home/disk/eos4/muting/KW/'
dir_in  = '/home/disk/eos4/muting/data/combine_reanalysis/era5/regrid_2.5deg_3hr/'
dir_out_data = dir_out+'output_data/RA_EAPEG_EKEG/3hr/'#Save Q
time_nc = dir_in+'u_ano_3hr.nc'
# include time from 1980-2018

## ERA5 data
model_name = 'regrid_'
# 1. File name
T_nc = model_name+'T.nc'
u_nc = model_name+'u.nc'
v_nc = model_name+'v.nc'
w_nc = model_name+'w.nc'
# 2. variable name
T_vname = 't'
u_vname = 'u'
v_vname = 'v'
w_vname = 'w'
# 3. variable nan value
#nanvalue = 10**14 # Check this!!!!!!!!!!!!!
#big_or_small = 1
isnan = 0 #There is no nan in ERA5. If there is nan in your data, specify isnan=1.

# 4. Specify the domain and time range
latS = -17.5 #so that the end product of Q is 15S-15N (1 additional grid at the boundary), change this if your data resolution is not 2.5 deg. If your data resolution is 1 deg. latS = -16, LagN = 16.
latN = 17.5 
PlevL = 100000 #1000 hPa, this code automatically change the pressure level from hPa to Pa. So, indicate the lowest and highest pressure level in Pa here.
PlevH = 10000 #100 hPa
t_min = 1998010100
t_max = 2013123121
Fs = 8 # sampling rate in time (3hourly)
dt = 86400/Fs # temporal resolution of the data in seconds (our data is daily, 86400s), change this if the data is not daily


test_val = 1# Whether you want to output the value of each budget term to make sure it makes sense, choose 1 if you want


# Constants:
Cp = 1004.07  # J*kg^-1*K^-1
Rd = 287.05   # J*kg^-1*K^-1
g = 9.8       # m/s^2
s2d = 86400 # 1 day = 86400s
re = 6371*1000 #earth radius (m)


'''
Q1 Calculation (from dDSE/dt)
'''

# load data: dimension (era5)
data2 = Dataset(time_nc, "r", format="NETCDF4")
time = data2.variables['time'][:]
lon  = data2.variables['lon'][:]
lat  = data2.variables['lat'][:]
plev = data2.variables['plev'][:]
del data2
it_min = np.squeeze(np.argwhere(time==t_min))
it_max = np.squeeze(np.argwhere(time==t_max))
time = time[it_min:it_max+1]  
nt   = np.size(time)


print(np.size(time))

if np.max(plev)<1200:
    plev = plev*100 # plev(Pa)
    print('already changed pressure unit') 

# Grid Limits
ilat_min = np.squeeze(np.argwhere(lat==latS))
ilat_max = np.squeeze(np.argwhere(lat==latN))
lat = lat[ilat_min:ilat_max+1]

nlev = np.size(plev)
nlat = np.size(lat)
nlon = np.size(lon)
tmax = nt 

# load data
data = Dataset(dir_in+T_nc, "r", format="NETCDF4")
T = data.variables[T_vname][0:tmax,:,ilat_min:ilat_max+1,:] 
del data
if isnan == 1:
    T = MJO.filled_to_nan(T,nanvalue,big_or_small) #-->no nan in era5, so this step is unnecessary
    
#**********************************************
#           Calculate Q1  
# Q1 = d(T)/d(t) + udT/dx  + vdT/dy + wdT/dp -alpha*w/Cp  ]
#       [0]           [1]       [2]   [3]      [4]
# d means partial, w means omega
#*****************TERM [0]********************
# partial(T)/partial(t)
dTdt = np.ones([nt-2,nlev,nlat,nlon]) #(time-2,lev-1,lat,lon)

for ii in range(0,nlev): #lev
    T_temp = T[:,ii,:,:]
    for tt in range(1,nt-1):
        dT_time = T_temp[tt+1,:,:] - T_temp[tt-1,:,:]
        dTdt[tt-1,ii,:,:] = dT_time/(2*dt)
    del T_temp,dT_time
dTdt = (dTdt[:,:-1,1:-1,:]+dTdt[:,1:,1:-1,:])/2
if test_val == 1:
    print('dTdt=')
    print( np.mean(np.mean(np.mean(dTdt,3),2),0) )
Q = 0
Q = Q+dTdt
del dTdt
print('finish term 0: dT/dt')
#*****************TERM [1]********************
# <u*dT/dx>
data = Dataset(dir_in+u_nc, "r", format="NETCDF4")
U  = data.variables[u_vname][0:tmax,:,ilat_min:ilat_max+1,:] 
del data
if isnan == 1:
    U = MJO.filled_to_nan(U,nanvalue,big_or_small) #remove nan

dx = (lon[3]-lon[1])*np.cos(lat*2*np.pi/360)*111000 # convert into meters
dTdx = np.ones([nlev,nlat,nlon])
udTdx = np.ones([nt,nlev,nlat,nlon])
for tt in range(0,nt): #try: faster to have this loop or not, this can be removed too
    T_temp = T[tt,:,:,:]
    for ii in range(0,nlon):
        if ii == 0:
             dTdx[:,:,ii] = T_temp[:,:,1] - T_temp[:,:,143]
        elif ii == 143:
             dTdx[:,:,ii] = T_temp[:,:,0] - T_temp[:,:,142]
        else:
             dTdx[:,:,ii] = T_temp[:,:,ii+1] - T_temp[:,:,ii-1]

    for ilat in range(0,nlat):
        dTdx[:,ilat,:] = dTdx[:,ilat,:]/dx[ilat]
    udTdx[tt,:,:,:] = dTdx*U[tt,:,:,:]
    
del dTdx, U
udTdx = ( udTdx[1:-1,:-1,1:-1,:]+udTdx[1:-1,1:,1:-1,:] )/2
Q = Q+udTdx
if test_val == 1:
    print('udTdx=')
    print( np.mean(np.mean(np.mean(udTdx,3),2),0) )
del udTdx
print('finish term 1: u*dT/dx')   
#*****************TERM [2]********************
# <v*dT/dy>
data = Dataset(dir_in+v_nc, "r", format="NETCDF4")
V = data.variables[v_vname][0:tmax,:,ilat_min:ilat_max+1,:] 
del data
if isnan == 1:
    V = MJO.filled_to_nan(V,nanvalue,big_or_small) #remove nan

dy = (lat[2]-lat[0])*111000 # convert into meters
dTdy = np.ones([nlev,nlat-2,nlon])
vdTdy = np.ones([nt,nlev,nlat-2,nlon])
for tt in range(0,nt): #try: faster to have this loop or not, this can be removed too
    T_temp = T[tt,:,:,:]
    for ii in range(1,nlat-1):
         dTdy[:,ii-1,:] = T_temp[:,ii+1,:] - T_temp[:,ii-1,:]
    vdTdy[tt,:,:,:] = dTdy*V[tt,:,1:-1,:]/dy
    del T_temp

del dTdy, V
vdTdy = ( vdTdy[1:-1,:-1,:,:]+vdTdy[1:-1,1:,:,:] )/2
if test_val == 1:
    print('vdTdy=')
    print( np.mean(np.mean(np.mean(vdTdy,3),2),0) )
Q = Q+vdTdy
del vdTdy
print('finish term 2: v*dT/dy')

#*****************TERM [3]********************
# Use mid-point instead of center difference
# <omega*dT/dp>
data = Dataset(dir_in+w_nc, "r", format="NETCDF4") #change varname for w
W = data.variables[w_vname][0:tmax,:,ilat_min:ilat_max+1,:] 
if isnan == 1:
    W = MJO.filled_to_nan(W,nanvalue,big_or_small) #remove nan
del data

dP = plev[1:]-plev[:nlev-1]
dTdP = np.ones([nt,nlev-1,nlat-2,nlon])
Wmid = ( W[:,:-1,1:-1,:] + W[:,1:,1:-1,:] )/2
del W
for ii in range(0,nlev-1): #Takes the midpoint of each pressure level
    dTdP[:,ii,:,:] = ( T[:,ii+1,1:-1,:] - T[:,ii,1:-1,:])/dP[ii] 

wdTdP = dTdP[1:-1,:,:,:]*Wmid[1:-1,:,:,:]
if test_val == 1:
    print('wdTdP=')
    print( np.mean(np.mean(np.mean(wdTdP,3),2),0) )
Q = Q+wdTdP
del wdTdP
print('finish term 3: w*dT/dP')

#*****************TERM [4]********************
# Use mid-point instead of center difference
# <1/Cp*alpha*omega>

#Takes the midpoint of each pressure level
Tmid = ( T[:,:-1,1:-1,:]+T[:,1:,1:-1,:] )/2 #([nt,nlev-1,nlat-2,nlon])
del T

plev_mid = 1/2*(plev[:-1]+plev[1:])
amid = np.ones([nt,nlev-1,nlat-2,nlon]) #alpha
for ii in range(0,nlev-1):
    amid[:,ii,:,:] = Rd*Tmid[:,ii,:,:]/plev_mid[ii] #alpha
aw = -1/Cp*amid*Wmid
aw = aw[1:-1,:,:,:]
Q = Q+aw
# Test value makes sense:
if test_val == 1:
    print('-alpha*w/Cp=')
    print( np.mean(np.mean(np.mean(aw,3),2),0) )
del amid, Tmid, Wmid
print('finish term 4: alpha*omega/Cp')
    
#*****************Sum of [1],[2],[3],[4]*****************
#Q1 = dTdt+udTdx+vdTdy+wdTdP+aw


#*****************Save output******************************
output = dir_out_data+'Q1_1000_100_byT_15SN_3hr.nc'  
ncout = Dataset(output, 'w', format='NETCDF4')
lat = lat[1:-1]
nlat = np.size(lat)
time = time[0:tmax]
time = time[1:-1]
nlev = np.size(plev_mid)

# define axis size
ncout.createDimension('time', nt-2)  
ncout.createDimension('lat', nlat)
ncout.createDimension('lon', nlon)
ncout.createDimension('plev', nlev)

# create time axis
time2 = ncout.createVariable('time', dtype('double').char, ('time'))
time2.long_name = 'time'
time2.units = 'yyyymmdd'
time2.calendar = 'standard'
time2.axis = 'T'

# create lev axis
lev2 = ncout.createVariable('plev', dtype('double').char, ('plev'))
lev2.long_name = 'pressure'
lev2.units = 'hPa'
lev2.calendar = 'standard'
lev2.axis = 'P'

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

# create variable array
Q1out = ncout.createVariable('Q1', dtype('double').char, ('time','plev', 'lat', 'lon'))
Q1out.long_name = 'Q1 (from Lorenz 1955) = dT/dt + udT/dx + vdT/dy + omega*dT/dp -alpha*omega/Cp'
Q1out.units = 'K/day'  
    
# copy axis from original dataset
time2[:] = time[:]
lon2[:] = lon[:]
lat2[:] = lat[:]
lev2[:] = plev_mid[:]/100
Q1out[:] = Q[:]*s2d
    
print('finish saving Q1') 


