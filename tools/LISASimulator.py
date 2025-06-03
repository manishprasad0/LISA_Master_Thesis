# Expected LISA Instrumental Noise from the Test Masses and the Optical Metrology Subsystem (OMS)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import welch
import scipy as sp
from lisatools.sensitivity  import SensitivityMatrix, A1TDISens,E1TDISens, T1TDISens, AET1SensitivityMatrix, get_sensitivity
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray


def signal_time_to_freq_domain(signals, dt, winow_length_denominotor=4.5):
    fs = 1/dt
    win_length = int(len(signals[0]) / winow_length_denominotor)
    window = hann(win_length)

    fout = []
    pxxout = []
    for signal in signals:
        f, pxx = welch(signal, window=window, noverlap=0, nfft=None, fs=fs, return_onesided=True)
        fout.append(f)
        pxxout.append(pxx)
    fout = np.array(fout)
    pxxout = np.array(pxxout)

    return fout, pxxout

class LISASimulator:
    def __init__(self, Tobs, dt, wave_gen, waveform_kwargs=None):
        self.dt = dt
        self.N = int(Tobs / dt)
        self.Tobs = self.N * dt
        self.freq = np.fft.rfftfreq(self.N, self.dt)
        if self.freq[0] == 0:
            self.freq[0] = self.freq[1]

        # Waveform
        self.parameters = None
        self.modes = None
        self.wave_gen = wave_gen
        self.waveform_kwargs = waveform_kwargs
        self.num_bin = None     # Number of binaries in the simulation

        # Data containers
        self.noise_t = None
        self.signal_f = None
        self.signal_t = None
        self.injected_t = None
        self.sens_mat = None

        # Plotting Labels
        self.plot_labels = ["A", "E", "T"]

    def generate_noise(self, seed=None, include_sens_kwargs=False):
        """
        Generate noise in the time domain for A, E, and T channels of LISA.
        Parameters:
        - seed: Random seed for reproducibility.
        - include_sens_kwargs: If True, include stochastic parameters in sensitivity matrix.
        """

        if seed is not None:
            np.random.seed(seed)

        if include_sens_kwargs:
            sens_kwargs = dict(stochastic_params=(self.Tobs,))
            sens_mat = AET1SensitivityMatrix(self.freq, **sens_kwargs)
        else:
            sens_mat = AET1SensitivityMatrix(self.freq)
        
        self.sens_mat = sens_mat
        noises = []
        for sens_fn in sens_mat.sens_mat:
            noise = np.fft.irfft(np.random.normal(0.0, np.sqrt(sens_fn))
                                +1j * np.random.normal(0.0, np.sqrt(sens_fn))
                                ) /np.sqrt(self.dt*4/self.N)
            noises.append(noise) 
        noises = np.array(noises)

        self.noise_t = noises
        self.time = np.arange(len(self.noise_t[0])) * self.dt

    def generate_waveform(self, parameters, modes, waveform_kwargs):
        self.parameters = parameters
        self.modes = modes
        self.num_bin = parameters.ndim
        self.signal_f = self.wave_gen(*parameters, freqs=self.freq, modes=modes, **waveform_kwargs)

        signal_t = np.fft.irfft(self.signal_f, axis=-1)

        if self.num_bin == 1:
            self.signal_t = signal_t
        else:
            self.signal_t = signal_t.sum(axis=0)

    def inject_signal(self):
        if self.noise_t is None or self.signal_f is None:
            raise ValueError("Generate both noise and signal in frequency domain first.")

        self.injected_t = self.noise_t + self.signal_t
        
    def snr_from_lisatools(self):
        snr_list = []
        for signal in self.signal_f:
            data = DataResidualArray(signal, f_arr=self.freq)
            analysis = AnalysisContainer(data_res_arr=data, sens_mat=self.sens_mat)
            snr_list.append(analysis.snr())
        print("SNRs for each injected binary:", snr_list)
        return snr_list

    def __call__(
        self,
        seed=None,
        parameters=None,
        modes=None,
        waveform_kwargs=None, 
        include_sens_kwargs=False,
        include_T_channel=True
        ):
        """
        Main method to generate noise, waveform, and inject the signal.
        Parameters:
        - seed: Random seed for reproducibility.
        - parameters: Parameters for the waveform generation.
        - modes: Modes for the waveform generation.
        - waveform_kwargs: Additional keyword arguments for the waveform generation.
        - include_sens_kwargs: If True, include stochastic parameters in sensitivity matrix.
        - include_T_channel: If True, include T channel in the injected signal.
        Returns:
        - injected_t: Signal injected into the noise in time domain.
        """

        self.generate_noise(seed=seed, include_sens_kwargs=include_sens_kwargs)
        self.generate_waveform(parameters=parameters, modes=modes, waveform_kwargs=waveform_kwargs)
        self.inject_signal()

        if not include_T_channel:
            return self.injected_t[:2] # Exclude T channel if not needed
        else:
            return self.injected_t
    
    # ---------------------- Plotting Methods ---------------------- #
    def plot_waveform_frequency(self):
        for i in range((self.num_bin)):
            for j, let in enumerate(self.plot_labels):
                plt.loglog(self.freq, np.abs(self.signal_f[i][j]), label=let + f" bin {i}")
        plt.legend()
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"$\tilde{h}(f)$ (Hz$^{-1/2}$)")
        plt.show()

    def plot_time_series(self, object, channel=0):
        if self.injected_t is None:
            raise ValueError("Run inject_signal() first.")
        plt.plot(self.time, self.object[channel])
        plt.xlabel("Time [s]")
        plt.ylabel("Strain")
        plt.title("Injected Signal in Time Domain")
        plt.grid(True)
        plt.show()

    def plot_time_series_near_merger(self, channel=0, binary_index=0):
        if self.injected_t is None:
            raise ValueError("Run inject_signal() first.")
        
        plt.plot(self.time, self.injected_t[channel])
        
        if self.num_bin == 1:            
            plt.xlim(self.parameters[-1] - 60*60*2, self.parameters[-1] + 60*60*1)  # self.parameters[-1] = t_ref. 2 hours before and 1 hour after merger
            plt.axvline(self.parameters[-1], color='k', linestyle='--', label='t_ref')

        else:
            plt.xlim(self.parameters[-1][binary_index] - 60*60*2, self.parameters[-1][binary_index] + 60*60*1)
            plt.axvline(self.parameters[-1][binary_index], color='k', linestyle='--', label='t_ref')
        
        plt.xlabel("Time [s]")
        plt.ylabel("Strain")
        plt.title("Time Domain Signal Near Merger")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_time_frequency(self, channel=0, max_freq = 0.1, min_freq = 1e-4):
        if self.injected_t is None:
            raise ValueError("Run inject_signal() first.")

        f, t, Zxx = sp.signal.stft(self.injected_t[channel], fs=1/self.dt, nperseg=15000)
        max_freq_idx = np.searchsorted(f, max_freq)
        min_freq_idx = np.searchsorted(f, min_freq)

        plt.figure()
        plt.pcolormesh(t, f[min_freq_idx:max_freq_idx], np.abs(Zxx[min_freq_idx:max_freq_idx]), vmin=0, 
                    vmax= np.max(np.abs(Zxx[min_freq_idx:max_freq_idx]))/10)
        plt.yscale('log')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()

    def plot_spectrogram(
        self,
        max_frequency = 0.1,
        min_frequency = 0.0001):

        f_mesh, t_mesh, sig_Z = sp.signal.stft(self.injected_t, 1/self.dt, nperseg=50000/self.dt)
        max_frequency_index = np.searchsorted(f_mesh, max_frequency)
        min_frequency_index = np.searchsorted(f_mesh, min_frequency)

        plt.figure()
        plt.pcolormesh(t_mesh, f_mesh[min_frequency_index:max_frequency_index], np.log(np.abs(sig_Z[min_frequency_index:max_frequency_index])), shading='gouraud')
        plt.colorbar()
        plt.xlabel('time (s)')
        plt.ylabel('frequency (Hz)')
        plt.yscale('log')

    def plot_frequency_domain(self, num_channels):
        if self.injected_t is None:
            raise ValueError("Run inject_signal() first.")
        
        fout, pxxout = signal_time_to_freq_domain(self.injected_t, self.dt)
        plt.figure()
        for i in range(num_channels):
            plt.loglog(fout[i], np.sqrt(pxxout[i]), label=self.plot_labels[i] + ' Channel Signal')
            plt.loglog(self.sens_mat.frequency_arr, np.sqrt(self.sens_mat.sens_mat[i]), label = self.plot_labels[i] + ' Noise Model')
            
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'ASD [strain$/\sqrt{\mathrm{Hz}}$]')
        plt.grid(True, which='both')
        plt.legend(loc = 'lower left')
        plt.show()