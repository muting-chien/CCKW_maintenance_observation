###################
# Calculate CCKW-filtered brightness temperature
# Input: brightness time series from CLAUS (Granted by Juliana Dias)
# Output: precip anomaly timeseries, CCKW-filtered precip timeseries,\
#         CCKW-filtered precip std
###################
# 2022.10.11
import sys
sys.path.append('/home/disk/eos9/muting/function/python/')
import mjo_mean_state_diagnostics as MJO
import numpy as np
from numpy import dtype
from netCDF4 import Dataset
import scipy.signal as signal

# Constants:
g = 9.8       # m/s^2
pi = np.pi
spi = '\u03C0'
re = 6371*1000 #earth radius (m)


# for KW band (
ifr = 1 # or 0 (freq_min[ifr]~freq_max[ifr], wnum_min[ifr]~wnum_max[ifr])
#              ifrq=0-->Full Wheeler-Kiladis band;
#              ifreq=1-->Same as Nakamura and Takayabu 2022
freq_min = np.array([1/20,  1/6])
freq_max = np.array([1/2.5, 1/4])
wnum_min = np.array([1,    5.5])
wnum_max = np.array([14,   6.5])
hmin     = np.array([8,    0]) #m
hmax     = np.array([90,   10000])# #m
freqname = list(['','_NT2022'])
freqname_long = list(['Kiladis 2009','Nakamura and Takayabu 2022, 4-6 day, zonal wnum 6'])

Fs_t = 8 # 3hourly, Fs_t = 1 if daily. Fs_t represents how many grids per day.
Fs_lon = 1/2.5 #Fs_lon represents how many grids per longitude.

#######################
save_data = 1
trange = '1983_2013'
    
dir_out = '/home/disk/eos9/muting/mesoscale_CCEW/'
for im in range(0,1):   

    # Load Tb from Claus
    dir_in = '/home/disk/eos9/muting/data/Tb/'
    file_in = dir_in+'Tb_ano_1983_2013_27.5SN_3hr.nc'
    data = Dataset( file_in, "r", format="NETCDF4")

    time = data.variables['time'][:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    tmin = time[0]
    tmax = time[-1]
    itmin = np.argwhere(time==tmin).squeeze()
    itmax = np.argwhere(time==tmax).squeeze()
    imin = np.argwhere(lat==-15).squeeze()
    imax = np.argwhere(lat==15).squeeze()

    time = time[itmin:itmax+1]
    lat = lat[imin:imax+1]

    nt = np.size(time)
    nlon = np.size(lon)
    nlat = np.size(lat)
    nspace = nlat*nlon

    # CCKW-filtering
    vname = list(['Tb_ano'])
    Va = data.variables[vname[0]][itmin:itmax+1,imin:imax+1,:]
            
    # FFT
    V1 = signal.detrend(Va,axis=0)#*HANN 
    del Va
    FFT_V1 = np.zeros([nt,nlat,nlon],dtype=complex)
    for ilat in range(0,nlat):
        FFT_V1[:,ilat,:] = np.fft.fft2(V1[:,ilat,:])#/nlon   
            
    for nf in range(ifr,ifr+1): # choice of frequency band
        V1_shift = np.fft.fftshift( np.fft.fftshift(FFT_V1,axes=2),axes=0 )  
        V1_shift2 = np.zeros([nt,nlat,nlon],dtype=complex)
        # freq filter
        freq = np.arange(-nt/2,nt/2)*Fs_t/nt #(1/day)
        freq_1 = freq_min[nf]#1/20 
        freq_2 = freq_max[nf]#1/2.5 
        ifreq_1 = np.abs(freq-freq_1).argmin() 
        ifreq_2 = np.abs(freq-freq_2).argmin()
        ifreq_1_neg = np.abs(freq-(-freq_1)).argmin() #not used
        ifreq_2_neg = np.abs(freq-(-freq_2)).argmin() #not used 

        # zwnum filter
        zwnum = np.arange(-nlon/2,nlon/2)*Fs_lon/nlon*360 #zonal wavenum
        wnum_1 = wnum_min[nf]#1
        wnum_2 = wnum_max[nf]#15
        iwnum_1 = np.abs(zwnum-wnum_1).argmin() #not used
        iwnum_2 = np.abs(zwnum-wnum_2).argmin() #not used
        iwnum_1_neg = np.abs(zwnum+wnum_1).argmin() #negative here means positive in reality
        iwnum_2_neg = np.abs(zwnum+wnum_2).argmin()                                     

        # Equivalent depth filter
        for ilat in range(0,nlat):
            for ifreq in range(ifreq_1,ifreq_2+1):
                for iwnum in range(iwnum_2_neg,iwnum_1_neg+1):
                    f = freq[ifreq]/86400
                    k = zwnum[iwnum]/(2*6371*1000*np.pi)
                    C = abs(f/k) #phase speed c = f(1/s)/k(1/m) 
                    he = C**2/g
                    if he<=hmax[ifr] and he>=hmin[ifr]:
                        V1_shift2[ifreq, ilat, iwnum]  = V1_shift[ifreq,ilat,iwnum]  

            for ifreq in range(ifreq_2_neg,ifreq_1_neg+1):
                for iwnum in range(iwnum_1,iwnum_2+1):
                    f = freq[ifreq]/86400
                    k = zwnum[iwnum]/(2*6371*1000*np.pi)
                    C = abs(f/k) #phase speed c = f(1/s)/k(1/m) 
                    he = C**2/g
                    if he<=hmax[ifr] and he>=hmin[ifr]:
                        V1_shift2[ifreq, ilat, iwnum]  = V1_shift[ifreq,ilat,iwnum] 
        V1_shift2 = np.fft.ifftshift( np.fft.ifftshift(V1_shift2,axes=2),axes=0 )
        pr_kw = np.zeros([nt,nlat,nlon])
        for ilat in range(0,nlat):
            pr_kw[:,ilat,:] = np.fft.ifft2(V1_shift2[:,ilat,:])
        
print('finish kw filtering')

if save_data == 1:
    ################################
    # Save KW pr data
    file_out = dir_out+'output_data/Tb_15NS_CCKW_filt_'+trange+'_3hr'+freqname[ifr]+'.nc'
    ncout = Dataset(file_out, 'w', format='NETCDF4')
    # define axis size
    ncout.createDimension('time',nt)
    ncout.createDimension('lat', nlat)
    ncout.createDimension('lon', nlon)
    # create time axis
    time2 = ncout.createVariable('time', dtype('double').char, ('time',))
    time2.long_name = 'time'
    time2.units = 'yyyymmdd'
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
    V1out = ncout.createVariable('Tb_kw', dtype('double').char, ('time', 'lat', 'lon'))
    V1out.long_name = 'CCKW-filtered ('+freqname_long[ifr]+') Brightness temperature anomaly'
    V1out.units = 'K'
    
    # copy variable
    time2[:] = time[:]
    lon2[:] = lon[:]
    lat2[:] = lat[:]
    V1out[:] = pr_kw[:]

    print('finish saving KW filtered data')
