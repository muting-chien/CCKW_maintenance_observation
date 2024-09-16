####################
# This code is to save post process 2D variables from anomaly (LH, precip)
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
dir_in = DIR+'data/combine_reanalysis/era5/regrid_2.5deg_3hr/'
dir_out = DIR+'KW/'
dir_out2 = dir_out+'output_data/RA_EAPEG_EKEG/3hr/final/'
os.makedirs(dir_out2, exist_ok=True)
lowpass_txt = 'remove 30-day low-pass filtered field'

##############################################
# Load 2D anomaly (LH, pr)
file   = dir_in+'LH_ano_3hr.nc'
data   = Dataset(file, 'r', format='NETCDF4')
LH_ano = data.variables['LH_ano'][8:-8,:,:]# time selection is to match 3D data
lat    = data.variables['lat'][:]
lon    = data.variables['lon'][:]
time   = data.variables['time'][8:-8]
LH_ano = LH_ano*(-1) # make positive anomaly: more evaporation

file   = dir_in+'pr_ano_3hr.nc'
data   = Dataset(file, 'r', format='NETCDF4')
pr_ano = data.variables['pr_ano'][8:-8,:,:]

# Meridional projection
LH_proj = KW.KW_meridional_projection(LH_ano, lat, tropics_or_midlat=0)
pr_proj = KW.KW_meridional_projection(pr_ano, lat, tropics_or_midlat=0)

# Remove low-pass filtered variability (30-day low pass)
LH_proj = MJO.remove_lowpass_from_3hr_data(LH_proj)
pr_proj = MJO.remove_lowpass_from_3hr_data(pr_proj)

nt = np.size(time)
nlon = np.size(lon)

#################################################
# Save projected variables
output = dir_out2+'merdional_proj_LHpr_remove_lowpass.nc'
ncout = Dataset(output, 'w', format='NETCDF4')

# define axis size
ncout.createDimension('time',  nt)
ncout.createDimension('lon',   nlon)

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

LH_out = ncout.createVariable('LH_proj', dtype('double').char, ('time', 'lon'))
LH_out.long_name = 'meridionally projected surface latent heat flux (positive: more evaportation), '+lowpass_txt
LH_out.units = 'W/m^2'
LH_out[:] = LH_proj[:]
del LH_proj

pr_out = ncout.createVariable('pr_proj', dtype('double').char, ('time', 'lon'))
pr_out.long_name = 'meridionally projected precipitation, '+lowpass_txt
pr_out.units = 'mm/day'
pr_out[:] = pr_proj[:]
del pr_proj

lon2[:]  = lon[:]
time2[:] = time[:]
print('finish saving data')