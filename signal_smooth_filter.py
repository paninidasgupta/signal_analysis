import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from scipy import signal
import math    

### spectra codes are based on Eric J Oliver MATLAB codes ######
#****************************** power spectrum *************************##
def spec(sig,sample_freq,window_type,alpha,**kwargs):
    ##https://www1.udel.edu/biology/rosewc/kaap686/notes/windowing.html
    ## Environmental data analysis with MATLAB-William Menke & Jashua Menke

    ##************************ signal must have even length.************##
    ## window=0 'no window'
    ## window=1 'Hamming window'
    ## window=2 'Hanning window'
    
    N=len(sig)                          #     no of points N
    Dt=sample_freq*1                       # sample frequency /rate
    T=N*Dt;                    
    fmax=1/(2*Dt);                         #     Nyquist (maximum) frequency,
    Df =fmax/(N/2);                        #     frequency interval,
    Nf=N/2+1;                              #     number of non-negative frequencies,#     frequency vector , #f=Df*[0:N/2,-N/2þ1:-1]’; 
    sig=sig-np.mean(sig)
    
    if window_type==1:
        w1=0.54-0.46*np.cos(2*np.pi*np.arange(N)/(N-1)) ## hamming window weight
    elif window_type==2:
        w1=0.5-0.5*np.cos(2*np.pi*np.arange(N)/(N-1)) ## hanning window weight
    else:
        w1=1
    
    signal1=w1*sig
    temp_fft = Dt*sc.fftpack.fft(signal1)
    fftfreq = np.fft.fftfreq(N,Dt)         # daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
    temp_psd = temp_fft*np.conj(temp_fft)*(2/T)
    f = fftfreq[(fftfreq >= 0) |(fftfreq==-fmax)]
    f[-1]=-1*f[-1]
    p = temp_psd[(fftfreq >= 0) | (fftfreq==-fmax)]
    
    #Null Hypothesis: time series is uncorrelated random noise
    sd2est=np.std(sig);                   #   variance of time series,
    ff=np.sum(w1*w1)/N;                        #   power in window function, 
    c = (ff*sd2est)/(2*Nf*Df);               #   scaling constant,  
    cl95 = c*chi2.ppf(1.0-alpha, df=2);           #   95% confidence level,
    
    if len(kwargs)!=0:
        for item,values in kwargs.items():
            ax=values
            ax.plot(1/f,p)
            ax.set_xscale('log')
            ax.axhline(y=cl95,color='r', linestyle='--')
            ax.set_xlabel('Time Period (year)')
            ax.set_ylabel('PSD (dB)')
            ax.grid()

    return p,f,cl95
#****************************** cross spectrum *************************##

def xspec(x,y,sample_freq,window_type,**kwargs):
    ## https://ecjoliver.weebly.com/code.html
    ## http://www2.ocean.washington.edu/flowmow/processing/currents/coh/billtest/cohtest.html
    x=x-np.mean(x)
    y=y-np.mean(y)
    N=len(x)
    dt=sample_freq
    T=dt*N
    fmax=1/(2*Dt);
    if N != len(y): 
        raise ValueError('Data sets x  and y of different lengths: %i , %i!'%(n, len(y)))
        
    if window_type==1: 
        w1=np.hamming(N) 
    elif window_type==2:
        w1=np.hanning(N)
    else:
        w1=1
  
    fx = dt*sc.fftpack.fft(w1*x)
    fy = dt*sc.fftpack.fft(w1*y)
    fftfreq = np.fft.fftfreq(N,sample_freq)### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
    i1=(fftfreq >= 0) |(fftfreq==-fmax)
    f= fftfreq[i1]
    f[-1]=-1*f[-1]
    Pxy= np.zeros((len(f),),dtype=complex)
    Pxy = fx[i1]*np.conj(fy[i1])*(2/T)
    if len(kwargs)!=0: 
        for item,values in kwargs.items():
            ax=values
            ax.plot(1/f,np.real(Pxy))
            ax.set_xscale('log')
            ax.set_xlabel('Time Period (year)')
            ax.set_ylabel('Cospectra(dB)')
            ax.grid()

    return Pxy,f
#****************************** coherence squared *************************##

def spectra_pan(x,y,sample_freq,window_type,nwindow,alpha):
    ## https://dsp.stackexchange.com/questions/10012/magnitude-squared-coherence-calculation-inconsistence
    ## https://dsp.stackexchange.com/questions/16558/statistical-significance-of-coherence-values
    #      f    - frequencies
    # %   Pxx  - power spectrum of x
    # %   Pyy  - power spectrum of y
    # %   Pxy  - cross-spectrum of x and y
    # %   Cxy  - coherence (squared) between x and y
    # %   C    - 5% significance level for coherence
    # %   Phxy - phase spectrum between x and y
    # %   cxy  - cospectrum between x and y
    # %   qxy  - quadspectrum between x and y
    
    n = len(x)
    dt=sample_freq
    if n != len(y): 
        raise ValueError('Data sets x  and y of different lengths: %i , %i!'%(n, len(y)))
        f = []; Pxx = []; Pyy = []; Pxy = []; Cxy = []; C1 = [];C2=[]; Phxy = []; cxy = []; qxy = []; Lh = [];
        return f,Pxx,Pyy,Pxy,Cxy,C1,C2,Phxy,cxy,qxy
    
   
    n1 = 2**math.ceil(np.log2(len(x)))             ## number of points for fft
    k=int(n1/nwindow)                              ## No of windows
    
    if window_type==1:
        w1=np.hamming(k)
    elif window_type==2:
        w1=np.hanning(k)
    else:
        w1=1
        
    x1=np.zeros(k*nwindow)
    y1=np.zeros(k*nwindow)
    x1[0:len(x)]=x[:]
    y1[0:len(y)]=y[:]

    x1      = np.reshape(x1,(nwindow,k)).T
    y1      = np.reshape(y1,(nwindow,k)).T
    fftfreq = np.fft.fftfreq(k,dt)
    nqf     = int(k/2+1)
    ssx    = np.zeros(x1.shape,dtype=complex)
    ssy    = np.zeros(x1.shape,dtype=complex)
    ssxy   = np.zeros(x1.shape,dtype=complex)
    for i in np.arange(nwindow):
        ssx[:,i] = 2*(dt**2/k)*sc.fftpack.fft(w1*x1[:,i])*np.conj(sc.fftpack.fft(w1*x1[:,i]))
        ssy[:,i] = 2*(dt**2/k)*sc.fftpack.fft(w1*y1[:,i])*np.conj(sc.fftpack.fft(w1*y1[:,i]))
        ssxy[:,i] = 2*(dt**2/k)*sc.fftpack.fft(w1*x1[:,i])*np.conj(sc.fftpack.fft(w1*y1[:,i]))
    
    Pxx=1/k*np.mean(ssx[0:nqf,:],axis=1)
    Pyy=1/k*np.mean(ssy[0:nqf,:],axis=1)
    Pxy=1/k*np.mean(ssxy[0:nqf,:],axis=1)
    
    f=fftfreq[:nqf]
    f[nqf-1]=-1*f[nqf-1]    
    #% scale to have integral(Pxx) = variance(x)
    Pxx=1/n*Pxx
    Pyy=1/n*Pyy
    Pxy=1/n*Pxy
    
    #% set co- and quad-spectra
    cxy = Pxy.real
    qxy = -Pxy.imag
    #% generate phase spectra and make points where Cxy < C NaNs
    Phxy = np.arctan2( qxy , cxy )*180/np.pi
    #%ii = find( Cxy < C ); Phxy(ii) = NaN*ones(size(ii));
    
    #% generate coherence and significance limit
    Cxy = np.abs(Pxy)**2/ (Pxx * Pyy);
    df = 2*nwindow #https://dsp.stackexchange.com/questions/16558/statistical-significance-of-coherence-values

    C1 = 1 - alpha**(2/(df-2));   #     alpha = 0.05
   
    #% this is a very good approximation to:
    F = sc.stats.f.ppf(q=1-alpha, dfn=2, dfd=df-2)
    C2 = F/( 0.5*(df-2) + F);

    
    
    return f,Pxx,Pyy,Pxy,Cxy,C1,C2,Phxy,cxy,qxy



#****************************** coherence square SCIPY and MATPLOTLIB ******************

def spectra_scipy_matplotlib(x,y,sample_freq,window_type,nwindow,alpha):
    ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
    ## https://in.mathworks.com/help/signal/ref/mscohere.html
    ## https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.cohere.html
    #      f1,f2    - scipy and mlab frequencies
    # %   Cxy1,Cxy2  - scipy and mlab coherence (not squared) between x and y
    #        k -       nperseg/NFFT/each window length 
    
    n = len(x)
    dt=sample_freq
    if n != len(y): 
        raise ValueError('Data sets x  and y of different lengths: %i , %i!'%(n, len(y)))
        f = []; Pxx = []; Pyy = []; Pxy = []; Cxy = []; C1 = [];C2=[]; Phxy = []; cxy = []; qxy = []; Lh = [];
        return f1,f2,Cxy1,Cxy2,C1,C2
    
   
    n1 = 2**math.ceil(np.log2(len(x)))             ## number of points for fft
    k=int(n1/nwindow)                              ## No of windows
    
    if window_type==1:
        w1=np.hamming(k)
        
    elif window_type==2:
        w1=np.hanning(k)
        
    else:
        w1=1
        
    x=x-np.mean(x)
    y=y-np.mean(y)
    
    f1, Cxy1 = signal.coherence(x,y,fs=dt,window=w1,nperseg=k,noverlap=0)
    Cxy2,f2  = plt.mlab.cohere(x,y,window= w1, NFFT=k, noverlap=0, Fs=dt)
    
    #% generate coherence and significance limit
    df = 2*nwindow #https://dsp.stackexchange.com/questions/16558/statistical-significance-of-coherence-values

    C1 = 1 - alpha**(2/(df-2));   #     alpha = 0.05
   
    #% this is a very good approximation to:
    F = sc.stats.f.ppf(q=1-alpha, dfn=2, dfd=df-2)
    C2 = F/( 0.5*(df-2) + F);

    
    
    return f1,f2,Cxy1,Cxy2,C1,C2,k
#****************************** filter signal *************************##


def filter_signal_scipy(signal,sample_freq,ltime_period,htime_period,keep_mean):
    filter_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        raise ValueError('There is NaN in the signal')
    else:
        hf=1./ltime_period
        lf=1./htime_period

        temp_fft = sc.fftpack.fft(signal)

        fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
        i1=(np.abs(fftfreq) >= lf) & (np.abs(fftfreq) <= hf)  
        inv_fft=np.zeros(temp_fft.size,dtype=complex)
        inv_fft[i1]=temp_fft[i1]
        if keep_mean:
            inv_fft[0]=temp_fft[0]
        filter_signal= np.real_if_close(sc.fftpack.ifft(inv_fft))
    
    return filter_signal


#****************************   Low Pass Filter ********************************************** ##

def lowpass_scipy(signal,sample_freq,time_period,keep_mean):
    
    lowpass_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        raise ValueError('There is NaN in the signal')
    else:
        hf = 1./time_period

        temp_fft = sc.fftpack.fft(signal)

        fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
          
        i1 = np.abs(fftfreq) >= hf  
        
        temp_fft[i1] = 0
        if not(keep_mean):
            temp_fft[0]=0
        lowpass_signal= np.real_if_close(sc.fftpack.ifft(temp_fft))
    
    return lowpass_signal

#****************************** smoothing a signal with a flat window *************************##

def smooth_pan(timeseries,window_length):
    smoothed_sig          =     np.zeros(timeseries.size)
    i=int((window_length+1)/2-1)
    h=int(np.floor(window_length/2))
    print(i,h)
    smoothed_sig=np.empty((timeseries.shape))
    smoothed_sig[:] = numpy.nan
    while len(timeseries)-(i-h)>=window_length:
        smoothed_sig[i]=np.mean(timeseries[i-h:i+h+1])
        i=i+1
    return smoothed_sig

#****************************** smoothing a signal with a choice of windows *************************##


def smooth(x,window_len=11,window='hanning'):
    ##https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

