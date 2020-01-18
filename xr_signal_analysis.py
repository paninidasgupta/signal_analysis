import signal_smooth_filter as smf

def xr_lowpass(file,varname,filter_type):
    AA = xr.open_dataset(file)
    AA1 = AA[varname]
    if filter_type=='butter':
        BB =np.apply_along_axis(smf.lowpass_scipy_butter,axis=0,arr = AA1.values,lt=10,wn=3)
    elif filter_type=='fft':
        BB =np.apply_along_axis(smf.lowpass_scipy_fft,axis=0,arr = AA1.values,sample_freq=1,time_period=10,keep_mean=1)
    else:
        raise ValueError('Please enter correct filter name')
    
    AA[varname].values = BB*1
    return AA ## return xarray
