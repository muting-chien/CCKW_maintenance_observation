####################
# This code is to save post process 3D variables from anomaly (T, u, q, F, Q, gph)
# Post processing include: meridional projection and remove 30-day low pass variability
# 2023.12.29
# Mu-Ting Chien
#####################
import numpy as np
from netCDF4 import Dataset
from numpy import dtype
import sys
sys.path.append('/home/disk/eos4/muting/function/python/')
import KW_diagnostics as KW
import mjo_mean_state_diagnostics as MJO
import os

DIR = '/home/disk/eos4/muting/'
dir_in = DIR+'data/combine_reanalysis/era5/regrid_2.5deg_3hr/' # for T, u, q, gph, w
dir_out = DIR+'KW/'
dir_in2 = dir_out+'output_data/RA_EAPEG_EKEG/3hr/' # for Q and F
dir_out2 = dir_out+'output_data/RA_EAPEG_EKEG/3hr/final/'
os.makedirs(dir_out2, exist_ok=True)
lowpass_txt = 'remove 30-day low-pass filtered field'
vname = list(['gph','T','u','w','q','F','Q'])


for v in range(0,np.size(vname)):
    print(vname[v])
    
    if v!=5 and v!=6:
        ##############################################
        # Load 3D anomaly (gph, T)
        file   = dir_in+vname[v]+'_ano_3hr.nc'
        data   = Dataset(file, 'r', format='NETCDF4')
        V_ano  = data.variables[vname[v]+'_ano'][8:-8, 1:-1, :, :]# time selection is to match other 3D data (Q)

        plev   = data.variables['plev'][1:-1]
        time   = data.variables['time'][8:-8]
    elif v == 5:
        # load momentum forcing F
        file = dir_in2+vname[v]+'_ano_3hr.nc'
        data = Dataset(file, 'r', format='NETCDF4')
        V_ano = data.variables[vname[v]+'_ano'][8:-8, :, :, :]
    elif v == 6:
        # load diabatic heating Q
        file = dir_in2+vname[v]+'_ano_3hr.nc'
        data = Dataset(file, 'r', format='NETCDF4')
        V_ano = data.variables[vname[v]+'_ano'][:, :, :, :]
        V_ano = (V_ano[:,1:,:,:]+V_ano[:,:-1,:,:])/2
        
    lat    = data.variables['lat'][:]
    lon    = data.variables['lon'][:]
        
    # Meridional projection
    if v!=5:
        V_proj = KW.KW_meridional_projection(V_ano, lat, tropics_or_midlat=0) # except F, use tropical filter
    else:
        V_proj = KW.KW_meridional_projection(V_ano, lat, tropics_or_midlat=1) # F, use extratropical filter
    del V_ano
    
    # Remove low-pass filtered variability (30-day low pass)
    if v == 0:
        gph_proj = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj)
    elif v == 1:
        T_proj   = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj, T_proj=T_proj)
    elif v == 2:
        u_proj   = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj, T_proj=T_proj, u_proj=u_proj)
    elif v == 3:
        w_proj   = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj, T_proj=T_proj, u_proj=u_proj,\
                w_proj=w_proj)
    elif v == 4:
        q_proj   = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj, T_proj=T_proj, u_proj=u_proj,\
                w_proj=w_proj, q_proj=q_proj)
    elif v == 5:
        F_proj   = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj, T_proj=T_proj, u_proj=u_proj,\
                w_proj=w_proj, q_proj=q_proj, F_proj=F_proj)
    elif v == 6:
        Q_proj   = MJO.remove_lowpass_from_3hr_data(V_proj)
        np.savez(dir_out2+'merdional_proj_3d_remove_lowpass.npz',gph_proj=gph_proj, T_proj=T_proj, u_proj=u_proj,\
                w_proj=w_proj, q_proj=q_proj, F_proj=F_proj, Q_proj=Q_proj, time=time, lon=lon, plev=plev)

    nt   = np.size(time)
    nlon = np.size(lon)
    nlev = np.size(plev)


#################################################
# Save projected variables
output = dir_out2+'merdional_proj_3d_remove_lowpass.nc'
ncout = Dataset(output, 'w', format='NETCDF4')

# define axis size
ncout.createDimension('time',  nt)
ncout.createDimension('lon',   nlon)
ncout.createDimension('plev',  nlev)

# create time axis
time2 = ncout.createVariable('time', dtype('double').char, ('time'))
time2.standard_name = 'time'
time2.long_name = 'time'    
time2.units = 'yyyymmddhh'
time2.axis = 't'        
# create loc axis
lon2 = ncout.createVariable('lon', dtype('double').char, ('lon'))
lon2.long_name = 'longitude'
lon2.units = 'deg'
lon2.axis = 'lon'
# create lev axis
lev2 = ncout.createVariable('plev', dtype('double').char, ('plev'))
lev2.long_name = 'pressure level'
lev2.units = 'hPa'
lev2.axis = 'p'

# create output variables
z_out = ncout.createVariable('gph_proj', dtype('double').char, ('time', 'plev','lon'))
z_out.long_name = 'meridionally projected geopoential height, '+lowpass_txt
z_out.units = 'm'  
z_out[:] = gph_proj[:]
del gph_proj  

T_out = ncout.createVariable('T_proj', dtype('double').char, ('time', 'plev','lon'))
T_out.long_name = 'meridionally projected tropical temperature, '+lowpass_txt
T_out.units = 'K'
T_out[:] = T_proj[:]
del T_proj

u_out = ncout.createVariable('u_proj', dtype('double').char, ('time', 'plev','lon'))
u_out.long_name = 'meridionally projected tropical zonal wind, '+lowpass_txt
u_out.units = 'm/s'  
u_out[:] = u_proj[:]
del u_proj

w_out = ncout.createVariable('w_proj', dtype('double').char, ('time', 'plev','lon'))
w_out.long_name = 'meridionally projected tropical vertical velocity, '+lowpass_txt
w_out.units = 'Pa/s'  
w_out[:] = w_proj[:]
del w_proj

q_out = ncout.createVariable('q_proj', dtype('double').char, ('time', 'plev','lon'))
q_out.long_name = 'meridionally projected tropical specific humidity, '+lowpass_txt
q_out.units = 'kg/kg'  
q_out[:] = q_proj[:]
del q_proj 

F_out = ncout.createVariable('F_proj', dtype('double').char, ('time', 'plev','lon'))
F_out.long_name = 'meridionally projected extratropical eddy momentum flux convergence, '+lowpass_txt
F_out.units = 'm2/s2' 
F_out[:] = F_proj[:]
del F_proj

Q_out = ncout.createVariable('Q_proj', dtype('double').char, ('time', 'plev','lon'))
Q_out.long_name = 'meridionally projected tropical diabatic heating Q, '+lowpass_txt
Q_out.units = 'K/day'  
Q_out[:] = Q_proj[:]
del Q_proj

lev2[:] = plev[:]
lon2[:] = lon[:]
time2[:] = time[:]
print('finish saving data')