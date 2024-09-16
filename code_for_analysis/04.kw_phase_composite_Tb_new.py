###############
# Assign the wave phase for each lat-lon grid for each time
# Based on Tb data
# 2022.10.11
# Mu-Ting Chien
####################
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/disk/eos9/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import scipy.stats as stat
from matplotlib.cm import get_cmap
import os

dir_out = '/home/disk/eos9/muting/mesoscale_CCEW/'

trange = '1983_2013'
PI = '\u03C0'
Nstd = 1 #only consider cckw with np.abs(Tb)>Nstd*std
plot_fig = 1
save_data = 1

ifr = 1 # or 0 (freq_min[ifr]~freq_max[ifr], wnum_min[ifr]~wnum_max[ifr])
#              ifrq=0-->Full Wheeler-Kiladis band;
#              ifreq=1-->Same as Nakamura and Takayabu 2022
freqname = list(['','_NT2022'])
freqname_s = list(['','NT2022/'])
freqname_long = list(['Kiladis 2009','Nakamura and Takayabu 2022, 4-6 day, zonal wnum 6'])
figdir = dir_out+'figure/'+freqname_s[ifr]
os.makedirs(figdir,exist_ok=True)

###############
# Step1: Load CCKW-filtered Tb data (KW band depends on ifr=0 or ifr=1)
#######################
file_in = dir_out+'output_data/Tb_15NS_CCKW_filt_'+trange+'_3hr'+freqname[ifr]+'.nc'
data  = Dataset( file_in, 'r', format='NETCDF4')
Tb_kw = data.variables['Tb_kw'][:]
time  = data.variables['time'][:]
lat   = data.variables['lat'][:]
lon   = data.variables['lon'][:]
nt    = np.size(time)
nlat  = np.size(lat)
nlon  = np.size(lon)
Tb_std = np.nanstd(Tb_kw)
print(Tb_std)

################
# Step2: Find local maximum and minimum and the nearest peak for each instant
##################
#  Added: 2022.10.11
#  Only consider phase if the nearest local maximum or minimum is greater than Tb_std*Nstd
# Find index of the local maxmium and minimum
local_min_max_id = np.empty([nt,nlat,nlon])
local_min_max_id[:] = np.nan
local_min_max_id[1:-1,:,:] = np.where( (Tb_kw[1:-1,:,:]<=Tb_kw[2:,:,:]) & (Tb_kw[1:-1,:,:]<=Tb_kw[0:-2,:,:]), 1, np.nan ) #local min
local_min_max_id[1:-1,:,:] = np.where( (Tb_kw[1:-1,:,:]>=Tb_kw[2:,:,:]) &(Tb_kw[1:-1,:,:]>=Tb_kw[0:-2,:,:]), -1, local_min_max_id[1:-1,:,:] ) #local max

#***********************
# For each instant, assign the nearest peak of the same sign
Tb_peak = np.empty([nt,nlat,nlon])
Tb_peak[:] = np.nan

const = np.array([1,-1])
for ilat in range(0,nlat):
    print(ilat)
    for ilon in range(0,nlon):
        
        zero_idx = np.argwhere(Tb_kw[:-1,ilat,ilon]*Tb_kw[1:,ilat,ilon]<=0).squeeze()
        peak_idx = np.argwhere(np.abs(local_min_max_id[:, ilat, ilon])==1).squeeze() #local min/max

        # for every local min/max, find the nearest zero or local min
        nmax = np.size(peak_idx)
        #id_r = np.empty([nmax],dtype='int') #right
        #id_l = np.empty([nmax],dtype='int') #left
        for i in range(0,nmax): 
            # Only select strong peak, do not consider weak peak
            if (np.abs(Tb_kw[peak_idx[i],ilat,ilon])<Tb_std*Nstd): 
                continue
            dpeak_zero = zero_idx-peak_idx[i]
            dpeak_zero_p = np.where(dpeak_zero>=0, dpeak_zero, np.nan) #positive
            dpeak_zero_n = np.where(dpeak_zero<0, dpeak_zero, np.nan) #negative
            
            # Find out the right boundary:
            tmp = np.argwhere( (dpeak_zero>=0) & (dpeak_zero==np.nanmin(dpeak_zero_p)) ).squeeze()
            if np.size(tmp)==0: #this means tmp is empty
                id_r = peak_idx[i]
            else:
                id_tmp = zero_idx[tmp]
                if i!=nmax-1:
                    if id_tmp>peak_idx[i+1]:
                        id_r = peak_idx[i+1]-1
                    else:
                        id_r = id_tmp
                else:
                    id_r = id_tmp
                
            # Find out the left boundary
            tmp = np.argwhere( (dpeak_zero<0) & (dpeak_zero==np.nanmax(dpeak_zero_n)) ).squeeze()
            if np.size(tmp)==0: #this means temp is empty
                id_l = peak_idx[i]
            else:
                id_tmp = zero_idx[tmp]+1
                if i!=0:
                    if id_tmp<peak_idx[i-1]:
                        id_l = peak_idx[i-1]+1
                    else:
                        id_l = id_tmp
                else:
                    id_l = id_tmp

            Tb_peak[ int(id_l):int(id_r+1),   ilat, ilon] = Tb_kw[peak_idx[i], ilat, ilon]

# Find whether this is the enhanced or decaying phase (enhanced phase:0, decaying phase:1)
enh_dec = np.empty([nt,nlat,nlon])
enh_dec[:] = np.nan
# Enhanced phase: Tb value is "smaller" than the previous day and "larger" than the next day
enh_dec[1:-1,:,:] = np.where( ((Tb_kw[1:-1,:,:]<Tb_kw[0:-2,:,:]) & (Tb_kw[1:-1,:,:]>Tb_kw[2:,:,:]) & (np.isnan(Tb_peak[1:-1,:,:])==0)), 0, np.nan  ) 
# Decaying phase: Tb value is "larger" than the previous day and "smaller" than the next day
enh_dec[1:-1,:,:] = np.where( ((Tb_kw[1:-1,:,:]>Tb_kw[0:-2,:,:]) & (Tb_kw[1:-1,:,:]<Tb_kw[2:,:,:]) & (np.isnan(Tb_peak[1:-1,:,:])==0)), 1, enh_dec[1:-1,:,:]  ) 

# For each instant, determine whether it is the enhanced and decaying phase
enh = Tb_peak*(1-enh_dec)
dec = Tb_peak*enh_dec
enh = np.where(enh==0,np.nan,enh)
dec = np.where(dec==0,np.nan,dec)

################
# Test Step2: Check Tb_kw timeseries and Tb_peak
#################
t = np.arange(2100,2300)
zero = np.zeros([np.size(t)])
if plot_fig == 1:
    fig_name = 'KW_Tb_time_evolution_sample_enh_dec_'+trange+'.png'
    fig = plt.subplots(1,1,figsize=(3.2, 2.4),dpi=600)
    plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.16)
    plt.rcParams.update({'font.size': 7})
    plt.plot(t, Tb_kw[t,0,0],'b-o',markersize=4)
    plt.plot(t, Tb_peak[t,0,0],'g-o',markersize=4)
    plt.plot(t, enh[t,0,0],'m-o',markersize=4)
    plt.plot(t, dec[t,0,0],'y-o',markersize=4)
    plt.legend(['Tb kw','strong peak','growing','decaying'])
    plt.plot(t, zero, 'k--')
    plt.xlabel('time (days since 1980101)')
    plt.ylabel('KW-filtered Tb (K)')
    plt.savefig(figdir+fig_name,format='png', dpi=600)
    #plt.show()
    plt.close()

############################
# Step 3: Calculate wave phase
###########################
# Normalize the Tb value with the closest peak Tb_kw values for the active and inactive phase.
Tb_norm = Tb_kw/np.abs(Tb_peak)
# Calculate the phase
phase = np.arcsin(Tb_norm)
phase = np.where( np.isnan(Tb_peak)==0, phase, np.nan)
phase_corr  = np.where( ((enh_dec==1) & (Tb_peak<=0)), -np.pi-phase, phase) # dec + Tbpeak <0: (-pi~-pi/2), new_theta = -pi-theta
phase_corr2 = np.where( ((enh_dec==1) & (Tb_peak>=0)), np.pi-phase,  phase_corr) # dec + Tbpeak >0: (pi/2~pi), new_theta = pi-theta
print('=============================================')
print('Should be pi and -pi')
print(np.nanmax(phase_corr2), np.nanmin(phase_corr2)) 

######################
# Test step3: Check wave_phase assign is correct
######################
kw_active = np.where(phase==-np.pi/2,Tb_kw,np.nan)
kw_inactive = np.where(phase==np.pi/2,Tb_kw,np.nan)

t = np.arange(2100,2300)
zero = np.zeros([np.size(t)])
half_p = 1/2*np.pi*np.ones([np.size(t)])
half_n = -1/2*np.pi*np.ones([np.size(t)])
#
if plot_fig == 1:
    fig_name = 'KW_Tb_time_evolution_sample_'+trange+'.png'
    fig = plt.subplots(1,1,figsize=(3.2, 3.2),dpi=600)
    plt.subplots_adjust(left=0.2,right=0.99,top=0.95,bottom=0.13,hspace=0.24)
    plt.rcParams.update({'font.size': 7})
    plt.subplot(2,1,1)
    plt.plot(t,Tb_kw[t,0,0],'b-o',markersize=2)
    plt.plot(t,kw_active[t,0,0],'mo',markersize=2)
    plt.plot(t,kw_inactive[t,0,0],'go',markersize=2)
    plt.plot(t,zero,'k:')
    plt.ylabel('KW Tb (K)')
    plt.legend(['KW Tb','active','inactive'])
    plt.subplot(2,1,2)
    plt.plot(t,phase_corr2[t,0,0],'c-o',markersize=2)
    plt.plot(t,zero,'k:')
    plt.plot(t,half_p,'k:')
    plt.plot(t,half_n,'k:')
    plt.xlabel('time (days since 1980101)')
    plt.ylabel('KW-filtered Tb (K)')
    plt.yticks(np.arange(-1,1.5,0.5)*np.pi,('-'+PI,'-1/2'+PI,'0','1/2'+PI,PI))
    plt.ylim([-np.pi,np.pi])
    plt.savefig(figdir+fig_name,format='png', dpi=600)
    #plt.show()
    plt.close()

###########
# Regenerate time based on yyyymmdd
time_nc = '/home/disk/eos9/muting/from_muting_laptop/combined_eof/input_data/merra2_u850_u200_small.nc'
data = Dataset(time_nc, "r", format="NETCDF4")
time_ref = data.variables['time'][:] #19800101-20181231
del data
for it in range(0,np.size(time_ref)):
    time_ref[it] = int(time_ref[it])
t_min  = time_ref[int(time[0])]
t_max  = time_ref[int(np.floor(time[-1]))]
it_min = np.argwhere(time_ref==t_min).squeeze()
it_max = np.argwhere(time_ref==t_max).squeeze()
time_ymd  = time_ref[it_min:it_max+1]
time_ymd2 = np.tile( time_ymd, (8,1) )
time_ymd2 = np.transpose( time_ymd2, (1,0) )
time_ymd2 = np.reshape( time_ymd2, 8*np.size(time_ymd)) #use time_ymd2 for seasonal selection (yyyymmdd)
#
##########
# Count the convective active points for each longitude for each month
monid_12 = list(['1','2','3','4','5','6','7','8','9','10','11','12'])
active_day = np.empty([12,nlat,nlon])
active_day[:] = np.nan
for ss in range(0,np.size(monid_12)):
    MonID = monid_12[ss]
    phase_mon, time_sea = MJO.Seasonal_Selection(phase_corr2,time_ymd2,MonID,0)
    for ilon in range(0,nlon):
        for ilat in range(0,nlat):
            active_day[ss,ilat,ilon] = np.sum(phase_mon[:,ilat,ilon]==-np.pi/2)
#
ilat_eq = np.argwhere(lat==0).squeeze()
#
# Plot statistics of KW active days: fig1 (lon_Tb)
if plot_fig == 1:
    fig_name = 'KW_active_day_lon_Tb_'+trange+'.png'
    fig = plt.subplots(1,1,figsize=(3.2, 2.4),dpi=600)
    plt.subplots_adjust(left=0.17,right=0.99,top=0.9,bottom=0.2)#,hspace=0.24)
    plt.rcParams.update({'font.size': 7})
    plt.plot(lon,active_day[0,ilat_eq,:],'b')
    plt.plot(lon,active_day[3,ilat_eq,:],'g')
    plt.plot(lon,active_day[6,ilat_eq,:],'r')
    plt.plot(lon,active_day[9,ilat_eq,:],'orange')
    plt.grid(True, linestyle='-.',color='grey',linewidth=0.1)
    plt.xticks(np.arange(0,360,60))
    plt.xlim([0,360])
    plt.ylabel('# of days')
    plt.title('Most active phase for KW at 0N')
    plt.xlabel('Longitude')
    plt.legend(['Jan','Apr','July','Oct'])
    plt.savefig(figdir+fig_name,format='png', dpi=600)
    #plt.show()
    plt.close()
    #################
    # Plot statistics of KW active days: fig2 (lon_month) (Same as Fig.2 in NT 2021)
    fig_name = 'KW_active_day_lon_mon_Tb_'+trange+'.png'
    fig = plt.subplots(1,1,figsize=(3.2, 2.4),dpi=600)
    plt.subplots_adjust(left=0.15,right=0.97,top=0.9,bottom=0.1)#,hspace=0.24)
    plt.rcParams.update({'font.size': 7})
    clev = np.arange(0,180,5)
    mon = np.arange(1,13)
    lon2d, mon2d = np.meshgrid(lon,mon)
    cf = plt.contourf(lon2d, mon2d, active_day[:,ilat_eq,:],\
            levels=clev,cmap="RdYlBu_r",zorder=2,alpha=0.99,extend='max')
    plt.gca().invert_yaxis()
    cb1 = plt.colorbar(cf,orientation = 'horizontal', pad = 0.25, shrink=1, aspect=30)
    plt.yticks(np.arange(1,13,1))
    plt.xticks(np.arange(0,360,60))
    plt.title('Days of the most active phase for KW at 0N')
    plt.ylabel('Month')
    plt.xlabel('Longitude')
    plt.savefig(figdir+fig_name,format='png', dpi=600)
    #plt.show()
    plt.close()

####################
# Step4: Composite other variables based on the phase. 
##################
# Decide how to assign the wave phase (what is the optimal bin size)
pi = np.pi
dph = 1/16*pi 
mybin = np.arange(-pi, pi+dph*2, dph)-dph/2 #play around this 1/8
bin_center = mybin[:-1] + dph/2

# Make sure there is no nan
phase_flat = np.ndarray.flatten(phase_corr2)
phase_flat_nonan = np.delete(phase_flat, np.argwhere(np.isnan(phase_flat)==1) )
ndata_nonan = np.size(phase_flat_nonan)

# Count how many points for each bin. 
N_    = stat.binned_statistic(phase_flat_nonan, phase_flat_nonan, statistic='count', bins=mybin)
N = N_.statistic
#print(N)

# Plot how many data for each phase in a KW cycle
if plot_fig == 1:
    fig_name = 'KW_data_count_Tb_'+trange+'.png'
    bin_simple = np.arange(-pi,pi+1/4*pi,1/4*pi)
    fig = plt.subplots(1,1,figsize=(3.2, 2),dpi=600)
    plt.subplots_adjust(left=0.2,right=0.99,top=0.9,bottom=0.2)#,hspace=0.24)
    plt.rcParams.update({'font.size': 7})
    plt.plot(bin_center,N,'m-o',markersize=5)
    plt.xticks(bin_simple,('-'+PI,'-3/4'+PI,'-1/2'+PI,'-1/4'+PI,'0','1/4'+PI,'1/2'+PI,'3/4'+PI,PI))
    plt.legend(['Number of data'])
    plt.ylabel('#')
    plt.xlabel('KW Phase')
    plt.savefig(figdir+fig_name,format='png', dpi=600)
    #plt.show()
    plt.close()

###############
# PLot evolution of Tb in a KW cycle
temp = np.ndarray.flatten(Tb_kw)
Tb_kw_nonan = np.delete(temp, np.argwhere(np.isnan(phase_flat)==1) )
Mean_        = stat.binned_statistic(phase_flat_nonan, Tb_kw_nonan, statistic='mean', bins=mybin)
Tb_kw_new    = Mean_.statistic
zero = np.zeros([np.size(bin_center)])
#
if plot_fig == 1:
    fig_name = 'KW_Tb_1cyc_'+trange+'.png'
    fig = plt.subplots(1,1,figsize=(3.2, 2),dpi=600)
    plt.subplots_adjust(left=0.2,right=0.99,top=0.95,bottom=0.2)
    bin_simple = np.arange(-pi,pi+1/4*pi,1/4*pi)
    plt.plot(bin_center,Tb_kw_new,'b-o',markersize=5)
    plt.plot(bin_center,zero,'k--')
    plt.rcParams.update({'font.size': 7})
    plt.xticks(bin_simple,('-'+PI,'-3/4'+PI,'-1/2'+PI,'-1/4'+PI,'0','1/4'+PI,'1/2'+PI,'3/4'+PI,PI))
    plt.legend(['Brightness temperature Tb'])
    plt.xlabel('KW Phase')
    plt.ylabel('KW Tb (K)')
    plt.savefig(figdir+fig_name,format='png', dpi=600)
    #plt.show()
    plt.close()

##################
# Step 5: Save phase data (phase_corr2, active days)
###########################
if save_data == 1:
    file_out = dir_out+'output_data/phase_CCKW_filt_Tb_'+trange+freqname[ifr]+'.nc'
    ncout = Dataset(file_out, 'w', format='NETCDF4')
    # define axis size
    ncout.createDimension('time',nt)
    ncout.createDimension('lat', nlat)
    ncout.createDimension('lon', nlon)
    ncout.createDimension('month',12)
    # create time axis
    time2 = ncout.createVariable('time', dtype('double').char, ('time',))
    time2.long_name = 'time'
    time2.units = 'days since 19800101'
    time2.calendar = 'standard'
    time2.axis = 'T'
    # create month axis
    mon2 = ncout.createVariable('month', dtype('double').char, ('month',))
    mon2.long_name = 'month'
    mon2.units = 'NA'
    mon2.calendar = 'standard'
    mon2.axis = 'mon'
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
    V1out = ncout.createVariable('phase_kw', dtype('double').char, ('time', 'lat', 'lon'))
    V1out.long_name = 'phase-CCKW filtered ('+freqname_long[ifr]+')-Tb from CLAUS, -pi~pi'
    V1out.units = 'radius'
    
    # create variables
    V2out = ncout.createVariable('active_days', dtype('double').char, ('month', 'lat', 'lon'))
    V2out.long_name = '# of days with the most active convective phase of CCKW'
    V2out.units = 'days'
    
    # create variables
    tout = ncout.createVariable('time_ymd', dtype('double').char, ('time'))
    tout.long_name = 'time in yyymmdd'
    tout.units = 'YYYYMMDD'

    # copy variable
    mon2[:] = np.arange(1,13)
    time2[:] = time[:]
    tout[:] = time_ymd2[:]
    lon2[:] = lon[:]
    lat2[:] = lat[:]
    V1out[:] = phase_corr2[:]
    V2out[:] = active_day[:]