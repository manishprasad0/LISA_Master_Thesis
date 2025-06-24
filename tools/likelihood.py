import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from scipy.signal import welch
import scipy as sp
from lisatools.sensitivity  import SensitivityMatrix, AET1SensitivityMatrix, get_sensitivity, AE1SensitivityMatrix
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray

# get_hh calculates the inner product of the variable 'signal' for the AET1SensitivityMatrix sensitivity matrix 'sens_mat' in the frequency domain.
# for the LISASimulator class, h (or the variable 'signal') is the injected signal in the simulated LISA data.
# for the LogLikelihood class, h (or the variable 'signal') is the template signal in the frequency domain.

def get_hh(data, sens_mat, df):
    """
    Calculate the squared norm of the signal in the frequency domain, weighted by the sensitivity matrix.
    Parameters:
    - signal: The signal in the frequency domain
    - sens_mat: lisatools.sensitivity.AET1SensitivityMatrix object.
    - df: Frequency bin width.
    - exclude_T_channel: If True, exclude the T channel from the calculation.
    Returns:
    - hh: The squared norm of the signal, weighted by the sensitivity matrix.
    """
    hh = np.sum(np.abs(data)**2 / sens_mat.sens_mat)
        
    return (hh * 4.0 * df)

def get_dh(data, template, sens_mat, df):
    """
    Calculate the inner product of the data with the template in the frequency domain, weighted by the sensitivity matrix.
    Parameters:
    - data: The GW data in the frequency domain
    - template: A template signal in the frequency domain
    - sens_mat: lisatools.sensitivity.AET1SensitivityMatrix object.
    - df: Frequency bin width.
    - exclude_T_channel: If True, exclude the T channel from the calculation.
    Returns:
    - dh: The inner product of the data with the template, weighted by the sensitivity matrix.
    """
    dh = np.sum(data * np.conj(template) / sens_mat.sens_mat)
        
    return (np.real(dh) * 4.0 * df)

def template_snr(data, template, sens_mat, df):
    """
    Calculate the signal-to-noise ratio (SNR) of a template against the data in the frequency domain.
    Parameters:
    - data: The GW data in the frequency domain
    - template: A template signal in the frequency domain
    - sens_mat: lisatools.sensitivity.AET1SensitivityMatrix object.
    - df: Frequency bin width.
    - exclude_T_channel: If True, exclude the T channel from the calculation.
    Returns:
    - snr: The SNR of the template against the data.
    """
    hh = get_hh(template, sens_mat, df)
    dh = get_dh(data, template, sens_mat, df)
    
    return dh / np.sqrt(hh)


def template_snr_lisatools(data, template, sens_mat, freq):
    template = DataResidualArray(template, f_arr=freq)
    data = DataResidualArray(data, f_arr=freq)
    analysis = AnalysisContainer(data_res_arr=data, sens_mat=sens_mat)
    return analysis.template_inner_product(template=template)


def SNR_optimal_lisatools(self):
        SNR = []
        for signal in self.signal_f:
            data = DataResidualArray(signal, f_arr=self.freq)
            analysis = AnalysisContainer(data_res_arr=data, sens_mat=self.sens_mat)
            SNR.append(analysis.snr())
        SNR = np.array(SNR)
        return SNR

def inner_product_time_frequency(signal_1, signal_2, sens_mat, df, nperseg):
    """
    Calculate the inner product of two signals in the time-frequency domain, weighted by the sensitivity matrix.
    Parameters:
    - signal_1: The first signal in the time domain
    - signal_2: The second signal in the time domain
    - sens_mat: lisatools.sensitivity object.
    - df: Frequency bin width.
    Returns:
    - ip: The inner product of the two signals, weighted by the sensitivity matrix.
    """

    f_1, t_1, Z_1 = sp.signal.stft(signal_1, fs=df, nperseg=nperseg*df)
    f_2, t_2, Z_2 = sp.signal.stft(signal_2, fs=df, nperseg=nperseg*df)



from typing import Any, Tuple, Optional, List
class TimeFreqSNR:
    def __init__(self, data_t, wave_gen, cutoff_index, nperseg = 15000, dt_full=5.0, pre_merger=False):
        self.data_t = data_t
        self.wave_gen = wave_gen
        self.dt_full = dt_full  # This is the time step for the full data, not the STFT
        self.time_before_merger = None
        self.cutoff_index = cutoff_index  # Index to truncate the data before merger
        self.pre_merger = pre_merger  # Flag to indicate if the data is pre-merger

        # All the following quantities are for after the STFT is calculated
        self.dt = None
        self.nperseg = nperseg  # default value, can be changed later
        self.f = None
        self.t = None
        self.Zxx_data_A = None
        self.Zxx_data_E = None
        self.sens_mat_new = None
        self.dd = None
    
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

    def get_hh(self, Zxx_A, Zxx_E):
        inner_product_A = np.abs(Zxx_A)**2 / self.sens_mat_new[0][:, np.newaxis]
        inner_product_E = np.abs(Zxx_E)**2 / self.sens_mat_new[1][:, np.newaxis]
        
        return 4 * self.df * np.sum(inner_product_A + inner_product_E)
    
    def get_dh(self, Zxx_A, Zxx_E):
        inner_product_A = (Zxx_A * np.conj(self.Zxx_data_A)) / self.sens_mat_new[0][:, np.newaxis] 
        inner_product_E = (Zxx_E * np.conj(self.Zxx_data_E)) / self.sens_mat_new[1][:, np.newaxis]

        return 4 * self.df * np.sum(inner_product_A.real + inner_product_E.real)
    
    def calculate_time_frequency_SNR(
        self,
        *parameters: Any,
        waveform_kwargs: Optional[dict] = {},
        **kwargs: dict,
    ):
        template_f = self.wave_gen(*parameters,  **waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel
        template_t = np.fft.irfft(template_f, axis=-1)

        if self.pre_merger:
            # Truncate the template to the same length as the data
            template_t = template_t[:, :self.cutoff_index]

        Zxx_temp_A = sp.signal.stft(template_t[0], fs=1/self.dt, nperseg=self.nperseg)[2]
        Zxx_temp_E = sp.signal.stft(template_t[1], fs=1/self.dt, nperseg=self.nperseg)[2]

        # Calculate the inner product for A and E channels
        hh = self.get_hh(Zxx_temp_A, Zxx_temp_E)
        dh = self.get_dh(Zxx_temp_A, Zxx_temp_E)
        amplitude = dh/hh

        return  dh / np.sqrt(hh), amplitude