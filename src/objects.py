
import matplotlib.pyplot as plt
import numpy as np


class TimeSeries(object):
    """
    Time series object

    Instances:
    -----------
        unity : str
            unity of measured signal
        dt : float
            timestep [s]
        data : float
            amplitude values of measured signal [according to unity]


    Methods:
    --------
        fs : int
            sampling frequency [Hz]
        plot : None
            time plot 
        samples : int
            number of samples per channel 
        channels : int
            number of channels
        time : 1D array
            time array [s]

    """
    def __init__(self, unity:str, dt:float, data:float) -> None:
        super().__init__()
        self.unity = unity
        self.dt = dt
        self.data = data

    def fs(self) -> int:
        """sample rate [Hz]"""
        return int(np.round(1/self.dt))
    
    def samples(self) -> int:
        """number of samples per channel"""
        return self.data.shape[0]

    def channels(self) -> int:
        """number of channels"""
        if self.data.ndim > 1:
            return self.data.shape[1]
        else:
            return 1

    def time(self):
        """time vector [s]"""
        return np.arange(0, self.samples()*self.dt, self.dt)

    def trim(self, time_start: float, time_end: float):
        time = self.time()
        idx = np.logical_and((time >= time_start),(time <= time_end))
        if self.data.ndim > 1:
            self.data = self.data[idx,:]
        else:
            self.data = self.data[idx]

    def __str__(self):
        txt = ('signal in ' + self.unity + '\n') + \
              (str(self.samples()) + ' samples per channel\n') + \
              ('Acquisition rate: ' + str(self.fs()) + ' Hz\n') + \
              (str(self.channels()) + ' channels')
        return txt

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(1e3*self.time(),self.data)
        ax.set(xlabel="time [ms]", ylabel=self.unity)
        # ax.set_xlim(0, 50)
        # ax.set_ylim(-5000, 5000)
        ax.grid()
        plt.show()


class Spectrum(object):
    """
    Frequency series object
    
    Instance:
    --------
        df : float
            frequency resolution (span)
        window : str
            kernel window
        overlap : float
            window overlap
        f : float
            frequnecy vector
        z : float
            amplitude vector

    Methods:
    --------
        fft : z
            return the fourier transform of the input signal
    """
    def __init__(self, time: object, df: float, window:str, overlap:float) -> None:
        self.dt = time.dt
        self.unity = time.unity
        self.fs = time.fs()
        self.channels = time.channels()
        
        if overlap >= 1:
            raise RuntimeError('Unexpected overlap time')
        else:
            self.overlap = overlap
        self.window = window     

        if df==0 or 1/(df * self.dt)>time.samples():
            self.samples = time.samples()
            self.df = np.ceil(1/(self.dt * self.samples))
        else:
            self.df = df
            self.samples = int(1/(df * self.dt))

        self.n_spectrums = int(1 + np.floor((time.samples()-self.samples)/(self.samples * (1-self.overlap))))

        self.f = np.arange(start=0, stop=self.fs/2, step=self.df)
        self.z = self.transform(time.data)

    def __str__(self) -> str:
        txt = ('window size ' + str(self.samples) +'\n') +\
              ('frequency span ' + str(self.df) +' Hz \n') +\
              ('number of spectrum per channel ' + str(self.n_spectrums) +' \n')
        return txt

    def freq(self):
        f = np.fft.fftfreq(self.samples, d=self.dt)
        (neg,pos) = np.split(f,2)
        return neg

    def transform(self, time_data):
        """
        Performs Fourier transform of time series object

        Return:
        -------
            z : np. array
                array with spectrums of the 
        """
        if self.window == 'hanning':
            w = 2 * np.hanning(self.samples)
        elif self.window == 'hamming':
            w = 1.85 * np.hamming(self.samples)
        elif self.window == 'blackman':
            w = 2.8 * np.blackman(self.samples)
        elif self.window == 'bartlett':
            w = 2.8 * np.bartlett(self.samples)
        elif self.window == 'kaiser':
            b = 14
            w = 2.49 * np.kaiser(self.samples, b)
        else:
            w = np.ones(self.samples)
        
        z = np.empty(len(self.f))
        for n in range(self.n_spectrums):
            for ch in range(self.channels):
                start = int(np.round(n * (1-self.overlap) * self.samples))
                end = int(start + self.samples)
                if self.channels > 1:
                    x = time_data[start:end,ch]
                else:
                    x = time_data[start:end]
                # fft computation
                ydft = np.fft.fft(x * w)/self.samples
                # (neg,pos) = np.split(ydft,2)
                # z = np.column_stack((z, 2*neg))
                z = np.column_stack((z, 2*ydft[:len(self.f)]))
                del start, end, x
        return z

    def power(self):
        # pwr = np.multiply(np.conjugate(self.z),self.z).real
        pwr =self.mag()**2
        self.unity = self.unity+'^2'
        return pwr

    def mag(self):
        "magnitude of frequency spectrum"
        mag = np.abs(self.z)
        return np.nanmean(mag, axis=1)

    def psd(self):
        "power spectral density"
        self.unity = self.unity+'/Hz'
        # ToDo: look at esd (energy spectral density)
        return np.multiply(self.power(),self.f)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.f, self.mag())
        ax.set(xlabel="Frequency [Hz]", ylabel=self.unity)
        ax.set_yscale('log')
        # ax.set_xlim(0, 20)
        # ax.set_xticks(np.arange(0,21,2))
        ax.grid()
        plt.show()

    def save(self, file_name:str):
        "saving spectrum to txt data"
        np.savetxt(file_name+'.txt',np.column_stack((self.f,self.mag())),delimiter='\t')
        print('file '+file_name+' sucessfully saved')


def dummy_signal(dt, waves):
    duration = 5
    rng = np.random.default_rng()
    t = np.arange(0.0, duration, dt)
    y = np.zeros(t.shape)
    for i in range(waves):
        frequency = rng.integers(low=5, high=50)
        amplitude = rng.integers(low=1, high=10)
        print('frequency %d and amplitude %d' % (frequency, amplitude))
        y += amplitude * np.sin(frequency * 2 * np.pi * t)

    return y


if __name__ == '__main__':
    fs = 500
    dt = 1/fs
    
    data = dummy_signal(dt, 3)
    # plt.plot(data)
    ts = TimeSeries('V', dt, data)
    ts.plot()
    
    spec = Spectrum(ts, df = 0.5, window='', overlap=0.67)
    print(spec)
    spec.plot()

