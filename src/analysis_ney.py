"""
Frequency analysis of EMA (experimental modal analysis)  
--------
@poletts
Last update May 31st, 2022
"""

import os 

import numpy as np
import matplotlib.pyplot as plt
import objects


def read_txt_file(file_name: str) -> object:
    """
    read txt file

    return:
    ------
        dt : int
            time step [s]
        data : float array
            time series 
    """
    t, data = np.loadtxt(fname = file_name, delimiter='\t', skiprows=4, unpack=True)
    dt = 1e-9*np.mean(np.diff(t))

    return dt, data

if __name__ == '__main__':
    path = "xx"
    file_list = os.listdir(path)
    data = []
    n=0
    # for file in [file_list[8]]:
    for file in file_list:
        print(file)
        dt, single_data = read_txt_file(path + '/' + file)
        data.append(single_data)
        m=single_data.size
        n+=1
        del single_data
    
    ts = objects.TimeSeries('mv', dt, np.reshape(data,(n,m)).T)
    ts.plot()

    spec = objects.Spectrum(ts, df=50, window='hamming', overlap=0.67)
    print(spec)
    spec.plot()
