
import numpy as np
import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt

def equivalent_noise_bandwidth(window):
    #Returns the Equivalent Noise BandWidth (ENBW)
    return len(window) * np.sum(window**2) / np.sum(window)**2

def get_enbw_windows():
    #Return ENBW for all the following windows as a dataframe
    window_names = ['boxcar','barthann','bartlett','blackman','blackmanharris','bohman','cosine','exponential','flattop','hamming','hann','nuttall','parzen','triang']
    
    df = pd.DataFrame(columns=['Window','ENBW (bins)','ENBW correction (dB)'])
    for window_name in window_names:
        method_name = window_name
        func_to_run = getattr(signal, method_name) #map window names to window functions in scipy package
        L = 16384 #Number of points in the output window
        window = func_to_run(L) #call the functions
        
        enbw = equivalent_noise_bandwidth(window) #compute ENBW

        plt.figure()
        plt.plot(np.abs(np.fft.fft(window)))
        plt.show()
        
        print({'Window': window_name.title(),'ENBW (bins)':round(enbw,3),'ENBW correction (dB)': round(10*np.log10(enbw),3)})
                       
    return df

get_enbw_windows() 