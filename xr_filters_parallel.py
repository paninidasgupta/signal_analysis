import numpy as np
import scipy as sc
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

###################################### Bandpass filtering using FFT ###############################################

def filter_signal_scipy(signal,fs,highcut,lowcut,keep_mean_no=0):
    filter_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        raise ValueError('There is NaN in the signal')
    else:
        hf=1./highcut
        lf=1./lowcut

        temp_fft = np.fft.fft(signal)

        fftfreq = np.fft.fftfreq(len(signal),fs) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
        i1=(np.abs(fftfreq) >= lf) & (np.abs(fftfreq) <= hf)  
        inv_fft=np.zeros(temp_fft.size,dtype=complex)
        inv_fft[i1]=temp_fft[i1]
        if keep_mean_no:
            inv_fft[0]=temp_fft[0]
        filter_signal= np.real(np.fft.ifft(inv_fft))
    
    return filter_signal

###################################### Bandpass filtering using butterwith ###############################################


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
#     print(1/low,1/high)
    b, a = butter(order, [low, high], btype='band')
    bandpass = lfilter(b, a, data)
    return bandpass

################################################ xarray serial bandpass ##########################################
def xr_bandpass(AA,varname,filter_type,fs,highcut,lowcut,opt=1):
    
    AA1 = AA.copy()
    if filter_type=='butter':
        BB =np.apply_along_axis(butter_bandpass_filter,axis=0,arr=AA1.values,lowcut=1/lowcut,highcut=1/highcut,fs=fs)
    elif filter_type=='fft':
        BB =np.apply_along_axis(filter_signal_scipy,axis=0,arr=AA1.values,highcut=highcut,lowcut=lowcut,fs=fs,keep_mean_no=0)
    else:
        raise ValueError('Please enter correct filter name')
    if opt:
        mse_vert_filt = xr.Dataset({varname: (('time', 'lat','lon'), BB)}, coords={'time': AA1.time,'lat': AA1.lat,'lon': AA1.lon})
    else:
        mse_vert_filt = xr.Dataset({varname: (('time','level', 'lat','lon'), BB)}, coords={'time': AA1.time,'level':AA1.level,'lat': AA1.lat,'lon': AA1.lon})

   
    return mse_vert_filt## return xarray

################################################ xarray parallel bandpass ##########################################
def xr_bandpass_par(AA,varname,filter_type,fs,highcut,lowcut,processes,opt=1):
    
    AA1 = AA.copy()
    if filter_type=='butter':
        BB =parallel_apply_along_axis(butter_bandpass_filter,axis=0,arr=AA1.values,processes=processes,
                                lowcut=1/lowcut,highcut=1/highcut,fs=fs)
    elif filter_type=='fft':
        BB =parallel_apply_along_axis(filter_signal_scipy,axis=0,arr=AA1.values,processes=processes,
                                highcut=highcut,lowcut=lowcut,fs=fs,keep_mean_no=0)
    else:
        raise ValueError('Please enter correct filter name')
    if opt:
        mse_vert_filt = xr.Dataset({varname: (('time', 'lat','lon'), BB)}, coords={'time': AA1.time,'lat': AA1.lat,'lon': AA1.lon})
    else:
        mse_vert_filt = xr.Dataset({varname: (('time','level', 'lat','lon'), BB)}, coords={'time': AA1.time,'level':AA1.level,'lat': AA1.lat,'lon': AA1.lon})

   
    return mse_vert_filt## return xarray


##################################################  lowpass Filtering using FFT ###########################################

def lowpass_scipy_fft(signal,sample_freq,time_period,keep_mean):
    lowpass_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        raise ValueError('There is NaN in the signal')
    else:
        hf = 1./time_period

        temp_fft = np.fft.fft(signal)

        fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
        i1 = np.abs(fftfreq) >= hf  
        
        temp_fft[i1] = 0
        if not(keep_mean):
            temp_fft[0]=0
        lowpass_signal= np.real_if_close(np.fft.ifft(temp_fft))
    
    return lowpass_signal

#################################################   xarray serial lowpass ###################################
def xr_lowpass(AA,varname,filter_type,lf,opt=1):
    
    AA1 = AA.copy()
    if filter_type=='butter':
        BB =np.apply_along_axis(lowpass_scipy_butter,axis=0,arr = AA1.values,lt=10,wn=3)
    elif filter_type=='fft':
        BB =np.apply_along_axis(lowpass_scipy_fft,axis=0,arr = AA1.values,sample_freq=1,time_period=lf,keep_mean=1)
    else:
        raise ValueError('Please enter correct filter name')
    
    if opt:
        LP = xr.Dataset({varname: (('time', 'lat','lon'), BB)}, coords={'time': AA1.time,'lat': AA1.lat,'lon': AA1.lon})
    else:
        LP = xr.Dataset({varname: (('time','level', 'lat','lon'), BB)}, coords={'time': AA1.time,'level':AA1.level,'lat': AA1.lat,'lon': AA1.lon})

        
    return LP ## return xarray

#################################################   xarray parallel lowpass ###################################

def xr_lowpass_par(AA,varname,filter_type,lf,processes,opt=1):
    
    AA1 = AA.copy()
    if filter_type=='butter':
        BB = parallel_apply_along_axis(lowpass_scipy_butter,axis=0,arr = AA1.values,process=processes,lt=10,wn=3)
    elif filter_type=='fft':
        BB = parallel_apply_along_axis(lowpass_scipy_fft,axis=0,arr = AA1.values,processes=processes,
                                       sample_freq=1,time_period=lf,keep_mean=1)
    else:
        raise ValueError('Please enter correct filter name')
    
    if opt:
        LP = xr.Dataset({varname: (('time', 'lat','lon'), BB)}, coords={'time': AA1.time,'lat': AA1.lat,'lon': AA1.lon})
    else:
        LP = xr.Dataset({varname: (('time','level', 'lat','lon'), BB)}, coords={'time': AA1.time,'level':AA1.level,'lat': AA1.lat,'lon': AA1.lon})

        
    return LP


########33333333333333333333333333########### vertical integration   #########################3333333333333333333###################
def vert_integration_pres(field):
    mb_to_pa = 100
    g = 9.81
    integrated = -(1/g)*field.integrate("level")*mb_to_pa
    return integrated
###################################################################################################################################


# def parallel_apply_along_axis(func1d,axis, arr, processes,*args, **kwargs):
    
#     """
#     Like numpy.apply_along_axis(), but takes advantage of multiple cores.
#   Adapted from `here <https://stackoverflow.com/questions/45526700/
#     easy-parallelization-of-numpy-apply-along-axis
#     """        

#     if axis==0:           
#         effective_axis  =   1  
#     else: 
#         effective_axis  =   0    
        
#    ## effective_axis for splittiing into chunks 
#     chunks         =      np.array_split(arr, processes,effective_axis)          ## chunks
#    ##print(np.shape[i] for i in chunks)

#     pool   = multiprocessing.Pool(processes=processes)                           ## embarassing parallel
#     func1d  = func1d                                                             ## use of partial for passing other kwargs
#     axis   = axis
#     func   = partial(apply_on_chunck,func1d, axis,*args,**kwargs)
#     individual_results = pool.map(func, chunks)
#     pool.close()
#     pool.join()

#     return np.concatenate(individual_results,effective_axis)

# def apply_on_chunck(func1d,axis,arr,*args,**kwargs):                                   ## function applied on chunks seprately
# #     print(arr.shape,axis)
#     return np.apply_along_axis(func1d,axis,arr,*args,**kwargs) 

#######################################################################################################################
############################################ parrallel computation ####################################################
#######################################################################################################################

def parallel_apply_along_axis(func1d,axis, arr, processes,*args, **kwargs):
    
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple cores.
    Adapted from `here <https://stackoverflow.com/questions/45526700/
    easy-parallelization-of-numpy-apply-along-axis
    """        

    dims            =      np.arange(arr.ndim)[np.arange(arr.ndim)!=axis]        ## finding the dim  
    dim_len         =      np.array([np.size(arr,i) for i in dims])              ## which should be splitted into chunks 
    effective_axis  =      dims[dim_len==np.max(dim_len)][0]                     ## effective_axis for splittiing into chunks 
    chunks         =      np.array_split(arr, processes,effective_axis)          ## chunks
#     print(np.shape[i] for i in chunks)
    pool   = multiprocessing.Pool(processes=processes)                           ## embarassing parallel
    func1d  = func1d                                                             ## use of partial for passing other kwargs
    axis   = axis
    func   = partial(apply_on_chunck,func1d, axis,*args,**kwargs)
    individual_results = pool.map(func, chunks)
    pool.close()
    pool.join()

    return np.concatenate(individual_results,effective_axis)

def apply_on_chunck(func1d,axis,arr,*args,**kwargs):                                   ## function applied on chunks seprately
#     print(arr.shape,axis)
    return np.apply_along_axis(func1d,axis,arr,*args,**kwargs)                         ## use of numpy along axis
