# Expected LISA Instrumental Noise from the Test Masses and the Optical Metrology Subsystem (OMS)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import welch
import scipy as sp
from lisatools.sensitivity  import SensitivityMatrix, A1TDISens,E1TDISens, T1TDISens, AET1SensitivityMatrix


def noise_time_to_freq_domain(noises, dt, winow_length_denominotor=4.5):
    fs = 1/dt
    win_length = int(len(noises[0]) / winow_length_denominotor)
    window = hann(win_length)

    fout = []
    pxxout = []
    for noise in noises:
        f, pxx = welch(noise, window=window, noverlap=0, nfft=None, fs=fs, return_onesided=True)
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

    def plot_time_series(self, channel=0):
        if self.injected_t is None:
            raise ValueError("Run inject_signal() first.")
        plt.plot(self.time, self.injected_t[channel])
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

    def plot_frequency_series(self):
        if self.injected_f is None:
            raise ValueError("Run inject_signal() first.")
        plt.loglog(self.freq, np.abs(self.injected_f))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("Injected Signal in Frequency Domain")
        plt.grid(True)
        plt.show()

    def plot_time_frequency(self):
        if self.injected_t is None:
            raise ValueError("Run inject_signal() first.")
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(self.injected_t, fs=1/self.dt)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-20), shading='auto', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.title('Spectrogram (Time-Frequency Plot)')
        plt.colorbar(label='Power [dB]')
        plt.yscale('log')
        plt.ylim([min(self.freq), max(self.freq)])
        plt.show()