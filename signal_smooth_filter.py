import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
### spectra codes are based on Eric J Oliver MATLAB codes ######
#****************************** power spectrum *************************##
def spec(signal,sample_freq,window_hann,*args):
    
    N=len(signal)                          #     no of points N
    Dt=sample_freq*1                       # sample frequency /rate
    T=N*Dt;                    
    fmax=1/(2*Dt);                         #     Nyquist (maximum) frequency,
    Df =fmax/(N/2);                        #     frequency interval,
    Nf=N/2+1;                              #     number of non-negative frequencies,
   
    #     frequency vector , 
    #f=Df*[0:N/2,-N/2þ1:-1]’;   
    
    if window_hann:
        signal=signal-np.mean(signal)
        w1=0.54-0.46*np.cos(2*np.pi*np.arange(N)/(N-1)) ## hann window weight
        signal1=w1*signal  
    else:
        signal1=signal*1
        
    temp_fft = Dt*sc.fftpack.fft(signal1)
    fftfreq = np.fft.fftfreq(N,Dt)         # daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
    temp_psd = temp_fft*np.conj(temp_fft)*(2/T)
    w = fftfreq[(fftfreq >= 0) |(fftfreq==-0.5)]
    w[-1]=-1*w[-1]
    p = temp_psd[(fftfreq >= 0) | (fftfreq==-0.5)]
    
    #Null Hypothesis: time series is uncorrelated random noise
    sd2est=np.std(signal);                   #   variance of time series,
    ff=np.sum(w*w)/N;                        #   power in window function, 
    c = (ff*sd2est)/(2*Nf*Df);               #   scaling constant,  
    cl95 = c*chi2.ppf(0.95, df=2);           #   95% confidence level,
    
    if len(args)!=0:
        ax=args[0]
        ax.plot(1/w,p)
        ax.set_xscale('log')
        ax.axhline(y=cl95,color='r', linestyle='--')
        ax.set_xlabel('Time Period (year)')
        ax.set_ylabel('PSD (dB)')
        ax.grid()

    return p,w
#****************************** cross spectrum *************************##

def xspec(x,y,sample_freq,*args):
    fx = sc.fftpack.fft(x)
    fy = sc.fftpack.fft(y)
    fftfreq = np.fft.fftfreq(len(x),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
    w = fftfreq[(fftfreq >= 0) |(fftfreq==-0.5)]
    w[-1]=-1*w[-1]
    Pxy= np.zeros((len(x),),dtype=complex)
    Pxy = fx[(fftfreq >= 0) |(fftfreq==-0.5)]*np.conj(fy[(fftfreq >= 0) |(fftfreq==-0.5)])*(2/len(x))
    if len(args)!=0:
        ax=args[0]
        i = fftfreq > 0
        ax.plot(1/w,np.real(Pxy))
        ax.set_xscale('log')
        ax.set_xlabel('Time Period (year)')
        ax.set_ylabel('PSD (dB)')
        ax.grid()

    return Pxy,w
#****************************** coherence squared *************************##
def spectra(*args):
    #      f    - frequencies
    # %   Pxx  - power spectrum of x
    # %   Pyy  - power spectrum of y
    # %   Pxy  - cross-spectrum of x and y
    # %   Cxy  - coherence (not squared) between x and y
    # %   C    - 5% significance level for coherence
    # %   Phxy - phase spectrum between x and y
    # %   cxy  - cospectrum between x and y
    # %   qxy  - quadspectrum between x and y
    
    nargs=len(args)
    x = args[0]
    n = len(x)

    #% generate power spectra
    Pxx,w1 = spec(x,1,0)
    Pxx=1/n*Pxx
    f=w1*1
    #% if one argument
    if nargs == 1:
        Pyy = []; Pxy = []; Cxy = []; C1 = []; C2=[];Phxy = []; cxy = []; qxy = []; Lh = [];
        return f,Pxx,Pyy,Pxy,Cxy,C1,C2,Phxy,cxy,qxy
    #% if two argument    
    y = args[1]
    
    if n != len(y): 
        raise ValueError('Data sets x  and y of different lengths: %i , %i!'%(n, len(y)))
        f = []; Pxx = []; Pyy = []; Pxy = []; Cxy = []; C1 = [];C2=[]; Phxy = []; cxy = []; qxy = []; Lh = [];
        return f,Pxx,Pyy,Pxy,Cxy,C1,C2,Phxy,cxy,qxy
    
    Pyy,w2     = spec(y,1,0)
    Pxy,w3     = xspec(x,y,1)
    
    #% scale to have integral(Pxx) = variance(x)
   
    Pyy=1/n*Pyy
    Pxy=1/n*Pxy
    
    #% set co- and quad-spectra
    cxy = Pxy.real
    qxy = -Pxy.imag

    #% generate coherence and significance limit
    Cxy = np.abs(Pxy)**2/ Pxx * Pyy;
    df = 2*n #https://dsp.stackexchange.com/questions/16558/statistical-significance-of-coherence-values
    alpha = 0.05
    C1 = 1 - alpha**(2/(df-2));
   
    #% this is a very good approximation to:
    F = sc.stats.f.ppf(q=1-alpha, dfn=2, dfd=df-2)
    C2 = F/( 0.5*(df-2) + F);

    #% generate phase spectra and make points where Cxy < C NaNs
    Phxy = np.arctan2( qxy , cxy )*180/np.pi
    #%ii = find( Cxy < C ); Phxy(ii) = NaN*ones(size(ii));
    
    return f,Pxx,Pyy,Pxy,Cxy,C1,C2,Phxy,cxy,qxy

#****************************** filter signal *************************##


def filter_signal_scipy(signal,sample_freq,ltime_period,htime_period,opt_show_psd,keep_mean):
    filter_signal=np.zeros(signal.shape)
    if any(np.isnan(signal)):
        filter_signal=signal*1
    else:
        hf=1./ltime_period
        lf=1./htime_period

        temp_fft = sc.fftpack.fft(signal)
        temp_psd = np.abs(temp_fft) ** 2

        fftfreq = np.fft.fftfreq(len(signal),sample_freq) ### daily data it is 1./365 ## monthly data 1./12 ## yearly data=1
        i = fftfreq > 0

        if opt_show_psd:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot( (1/fftfreq[i]),temp_psd[i])
            ax.set_xscale('log')
            ax.set_xlabel('Time Period (year)')
            ax.set_ylabel('PSD (dB)')
            ax.grid()
            
        i1=(np.abs(fftfreq) >= lf) & (np.abs(fftfreq) <= hf)  
        inv_fft=np.zeros(temp_fft.size,dtype=complex)
        inv_fft[i1]=temp_fft[i1]
        if keep_mean:
            inv_fft[0]=temp_fft[0]
        filter_signal= np.real_if_close(sc.fftpack.ifft(inv_fft))
    
    return filter_signal

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

