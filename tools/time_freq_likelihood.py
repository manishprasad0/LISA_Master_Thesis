import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import welch
import scipy as sp
from lisatools.sensitivity  import SensitivityMatrix, AET1SensitivityMatrix, get_sensitivity, AE1SensitivityMatrix
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from typing import Any, Tuple, Optional, List


# RIGHT NOW MAKING THIS ONLY FOR A AND E CHANNELS, NOT T CHANNEL
# class TimeFreqLikelihood is like the AnalysisContainer class in lisatools, but it is specifically designed for time-frequency likelihood calculations.
class TimeFreqLikelihood:
    def __init__(self, data_t, wave_gen, nperseg = 15000, dt_full=5.0):
        self.data_t = data_t
        self.wave_gen = wave_gen
        self.dt_full = dt_full  # This is the time step for the full data, not the STFT
        self.time_before_merger = None
        
        # All the following quantities are for after the STFT is calculated
        self.dt = None
        self.nperseg = nperseg  # default value, can be changed later
        self.f = None
        self.t = None
        self.Zxx_data_A = None
        self.Zxx_data_E = None
        self.sens_mat_new = None
        self.dd = None

    def pre_merger(self, time_before_merger, t_ref, t_array):
        self.time_before_merger = time_before_merger

        cutoff_time = t_ref - time_before_merger
        cutoff_index = np.searchsorted(t_array, cutoff_time)
        self.cutoff_index = cutoff_index

        data_t_truncated = self.data_t[:, :cutoff_index]
        self.data_t = data_t_truncated
    
    def get_stft_of_data(self, include_sens_kwargs=False):
        """
        Calculate the Short-Time Fourier Transform (STFT) of the data and set up the frequency and time arrays.
        Parameters:
        - dt: Time step for the data, default is 5.0 seconds. Different from self.dt which is the time step for the STFT.
        - include_sens_kwargs: If True, include sensitivity matrix parameters in the sensitivity matrix calculation.
        """
        
        f, t, Zxx_data_A = sp.signal.stft(self.data_t[0], fs=1/self.dt_full, nperseg=self.nperseg)
        f, t, Zxx_data_E = sp.signal.stft(self.data_t[1], fs=1/self.dt_full, nperseg=self.nperseg)
        
        self.f = f
        self.df = f[1] - f[0]  # frequency bin width
        self.f[0] = self.f[1]  # set the first frequency to the second frequency to avoid division by zero
        
        self.t = t
        self.dt = t[1] - t[0]

        self.Zxx_data_A = Zxx_data_A
        self.Zxx_data_E = Zxx_data_E

        if include_sens_kwargs:
            sens_kwargs = dict(stochastic_params=(self.Tobs,))
            self.sens_mat_new = AE1SensitivityMatrix(self.f, **sens_kwargs).sens_mat
        else:
            self.sens_mat_new = AE1SensitivityMatrix(self.f).sens_mat

        self.dd = self.get_dd()

    def get_dd(self):
        # Calculate the inner product for A and E channels
        inner_product_A = np.abs(self.Zxx_data_A)**2 / self.sens_mat_new[0][:, np.newaxis]
        inner_product_E = np.abs(self.Zxx_data_E)**2 / self.sens_mat_new[1][:, np.newaxis]

        return 4 * self.df * np.sum(inner_product_A.real + inner_product_E.real)

    def get_hh(self, Zxx_A, Zxx_E):
        inner_product_A = np.abs(Zxx_A)**2 / self.sens_mat_new[0][:, np.newaxis]
        inner_product_E = np.abs(Zxx_E)**2 / self.sens_mat_new[1][:, np.newaxis]
        
        return 4 * self.df * np.sum(inner_product_A + inner_product_E)
    
    def get_dh(self, Zxx_A, Zxx_E):
        inner_product_A = (Zxx_A * np.conj(self.Zxx_data_A)) / self.sens_mat_new[0][:, np.newaxis] 
        inner_product_E = (Zxx_E * np.conj(self.Zxx_data_E)) / self.sens_mat_new[1][:, np.newaxis]

        return 4 * self.df * np.sum(inner_product_A.real + inner_product_E.real)
    
    def calculate_time_frequency_likelihood(
        self,
        *parameters: Any,
        waveform_kwargs: Optional[dict] = {},
        **kwargs: dict,
    ):
        parameters
        template_f = self.wave_gen(*parameters, **waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel
        template_t = np.fft.irfft(template_f, axis=-1)

        if self.time_before_merger is not None:
            # Truncate the template to the same length as the data
            template_t = template_t[:, :self.cutoff_index]

        Zxx_temp_A = sp.signal.stft(template_t[0], fs=1/self.dt, nperseg=self.nperseg)[2]
        Zxx_temp_E = sp.signal.stft(template_t[1], fs=1/self.dt, nperseg=self.nperseg)[2]

        # Calculate the inner product for A and E channels
        hh = self.get_hh(Zxx_temp_A, Zxx_temp_E)
        dh = self.get_dh(Zxx_temp_A, Zxx_temp_E)
        return (dh - hh / 2.0 - self.dd / 2.0) * self.dt

    def plot_spectrogram(self, max_freq = 0.1, min_freq = 1e-4):

        max_freq_idx = np.searchsorted(self.f, max_freq)
        min_freq_idx = np.searchsorted(self.f, min_freq)

        plt.figure()
        plt.pcolormesh(self.t, self.f[min_freq_idx:max_freq_idx], np.abs(self.Zxx_data_A[min_freq_idx:max_freq_idx]), vmin=0, 
                    vmax= np.max(np.abs(self.Zxx_data_A[min_freq_idx:max_freq_idx]))/10)
        plt.yscale('log')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()