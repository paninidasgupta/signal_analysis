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

def my_autocorr(x, t=1,plot=True,label=''):
    ''' compute correlation at each lag with list comprehension
        t is the lag 
    '''
    ac = [np.corrcoef(np.array([x[0:len(x)-i], x[i:len(x)]])) for i in np.arange(t)]
    acvsm = np.asarray(ac)[:,0,1]
    
    #Plot results
    if plot:
        plt.plot(np.arange(N/2),acvsm[:int(N/2)]/acvsm[0],marker='x',linestyle='--',label = label)
        plt.xlabel('Lag [months]')
        plt.ylabel('Autocorrelation C(t)')
        plt.legend()
    return acvsm

def autocorrelation_v1(x,label,plot=True):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))   # validation
    result = r/(variance*(np.arange(n, 0, -1))) # unbiased estimate?
    #Plot results
    if plot:
        plt.plot(np.arange(N/2),result[:int(N/2)]/result[0],marker='x',linestyle='--',label = label)
        plt.xlabel('Lag [months]')
        plt.ylabel('Autocorrelation C(t)')
        plt.legend()
    return result

def red_noise_significance(ts,tau):
    """Adopted from https://nbviewer.org/url/atmos.spyndle.net/notebooks/Courses/ATMS552/Nino2.ipynb"""
    tshat = np.fft.fft(ts)
    R = 1/tau # decorrelation rate (inverse months)
    N = len(ts)
    vartot = np.sum(np.abs(tshat/N)**2)
    M = np.fft.fftshift(np.arange(-N/2, N/2)) # in matlab: M = [0:N/2 - 1 -N/2:-1]
    omM = 2*np.pi*M/N
    dom = 2*np.pi/N
    ps_red = (vartot/np.pi)*dom*R/(R**2 + omM**2)
    ff= M[:int(N/2)]/N
    return ff, 2*N*ps_red[:int(N/2)]