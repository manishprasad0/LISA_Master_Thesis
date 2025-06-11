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

    