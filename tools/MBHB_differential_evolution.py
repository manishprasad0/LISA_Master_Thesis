from bbhx.utils.constants import *
from bbhx.utils.transform import *
import numpy as np
import scipy as sp
import xarray as xr
import time
import os
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt

from lisatools.sensitivity  import SensitivityMatrix, AET1SensitivityMatrix, get_sensitivity, AE1SensitivityMatrix
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from typing import Any, Tuple, Optional, List

# variable names:

# parameters_10 = [mT,  q, a1, a2,                   phi,        cos(i), lambda, sin(beta), psi, t_ref]
# parameters_11 = [mT,  q, a1, a2, dL/(PC_SI * 1e6), phi,        cos(i), lambda, sin(beta), psi, t_ref]
# parameters_all= [m1, m2, a1, a2, dL              , phi, f_ref,     i , lambda,     beta , psi, t_ref] = parameters_bbhx
# results.x is same as parameters_10 i.e. transformed parameters without distance and f_ref


def transform_to_bbhx_parameters(x_11: np.ndarray) -> np.ndarray:
    """
    Transform parameters to BBHX format. 
    - x: array of 11 parameters: [mT, q, a1, a2, dL/(PC_SI * 1e6), phi, cos(i), lambda, sin(beta), psi, t_ref]
    Returns 
    - all_parameters: an array of 12 parameters (input for bbhx): [m1, m2, a1, a2, dL, phi, f_ref, i, lambda, beta, psi, t_ref]
    """

    all_parameters = np.zeros(12)
    mT = x_11[0]
    q  = x_11[1]
    all_parameters[0] = mT / (1 + q)
    all_parameters[1] = mT * q / (1 + q)
    all_parameters[2] = x_11[2]
    all_parameters[3] = x_11[3]
    all_parameters[4] = x_11[4] * PC_SI * 1e6
    all_parameters[5] = x_11[5]
    all_parameters[6] = 0.0
    all_parameters[7] = np.arccos(x_11[6])
    all_parameters[8] = x_11[7]
    all_parameters[9] = np.arcsin(x_11[8])
    all_parameters[10] = x_11[9]
    all_parameters[11] = x_11[10]
    return all_parameters


class MBHB_finder:
    def __init__(self, data_t, wave_gen, cutoff_index, waveform_kwargs, boundaries, nperseg = 15000, dt_full=5.0, pre_merger=False):
        self.data_t = data_t
        self.wave_gen = wave_gen
        self.dt_full = dt_full  # This is the time step for the full data, not the STFT
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

        # waveform kwargs
        self.waveform_kwargs = waveform_kwargs
        self.boundaries = boundaries  # boundaries for the 11 parameters including distance as a dictionary.
        # Other stuff
#self.parameters_sample = ['TotalMass', 'MassRatio', 'Spin1', 'Spin2', 'Distance', 'Phase', 'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'CoalescenceTime']
    
    def get_stft_of_data(self, include_sens_kwargs=False):
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

    def calculate_time_frequency_SNR_without_distance(
        self,
        parameters_10: np.ndarray,
    ) -> float:
        """ Calculate the time-frequency SNR without distance parameter.
        - parameters_10: [mT,  q, a1, a2, phi, cos(i), lambda, sin(beta), psi, t_ref]

        The reason is that the function optimized by differential evolution needs to only have the parameters that are optimized,
        Since distance does not affect the SNR, it is set to a constant value.
        f_ref = 0.0 is added to the parameters in the function transform_to_bbhx_parameters to be compatible with the waveform generator. 
        
        Returns:
        - -SNR: the negative SNR value, as we want to minimize the SNR.
        """

        # Add distance parameter to the parameters_10 and set it to the middle of the range
        parameters_11 = np.zeros(len(parameters_10) + 1)  # +2 for f_ref and distance
        parameters_11[:4] = parameters_10[:4]                                                                                           # mT, q, a1, a2
        parameters_11[4] = self.boundaries['Distance'][0] + 0.5 * (self.boundaries['Distance'][1] - self.boundaries['Distance'][0])     # dL = mid point of the distance prior
        parameters_11[5:] = parameters_10[4:]                                                                                           # phase, phi, cos(i), lambda, sin(beta), psi, t_ref

        # Transform to BBHX parameters
        parameters_bbhx = transform_to_bbhx_parameters(parameters_11)

        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel
        template_t = np.fft.irfft(template_f, axis=-1)

        # Truncate the template to the same length as the data i.e. pre-merger
        if self.pre_merger:
            template_t = template_t[:, :self.cutoff_index]

        # Calculate the STFT of the template
        Zxx_temp_A = sp.signal.stft(template_t[0], fs=1/self.dt, nperseg=self.nperseg)[2]
        Zxx_temp_E = sp.signal.stft(template_t[1], fs=1/self.dt, nperseg=self.nperseg)[2]

        # Calculate the inner products for A and E channels
        hh = self.get_hh(Zxx_temp_A, Zxx_temp_E)
        dh = self.get_dh(Zxx_temp_A, Zxx_temp_E)

        return - dh / np.sqrt(hh) # Return the negative SNR value, as we want to minimize the SNR

    def calculate_amplitude(self, parameters_11: Any):

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_to_bbhx_parameters(parameters_11)
        
        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        template_f = template_f[:2]
        template_t = np.fft.irfft(template_f, axis=-1)

        # Truncate the template to the same length as the data i.e. pre-merger
        if self.pre_merger:
            template_t = template_t[:, :self.cutoff_index]
        
        # Calculate the STFT of the template and calculate the inner products
        Zxx_temp_A = sp.signal.stft(template_t[0], fs=1/self.dt, nperseg=self.nperseg)[2]
        Zxx_temp_E = sp.signal.stft(template_t[1], fs=1/self.dt, nperseg=self.nperseg)[2]
        hh = self.get_hh(Zxx_temp_A, Zxx_temp_E)
        dh = self.get_dh(Zxx_temp_A, Zxx_temp_E)

        # Calculate the amplitude factor = <d|h> / <h|h> 
        amplitude = dh/hh
        
        return amplitude
    
    def calculate_time_frequency_SNR_with_distance(
        self,
        parameters: Any,
        ):
        """ Calculate the time-frequency SNR with distance parameter.
        - parameters: [mT,  q, a1, a2, dL/(PC_SI * 1e6), phi, cos(i), lambda, sin(beta), psi, t_ref]
        """

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_to_bbhx_parameters(parameters) 

        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel
        template_t = np.fft.irfft(template_f, axis=-1)

        # Truncate the template to the same length as the data i.e. pre-merger
        if self.pre_merger:
            template_t = template_t[:, :self.cutoff_index]

        # Calculate the STFT of the template
        Zxx_temp_A = sp.signal.stft(template_t[0], fs=1/self.dt, nperseg=self.nperseg)[2]
        Zxx_temp_E = sp.signal.stft(template_t[1], fs=1/self.dt, nperseg=self.nperseg)[2]

        # Calculate the inner product for A and E channels
        hh = self.get_hh(Zxx_temp_A, Zxx_temp_E)
        dh = self.get_dh(Zxx_temp_A, Zxx_temp_E)

        # Return the normal/positive SNR value as we are not optimizing this function
        return dh / np.sqrt(hh) 

    def find_MBHB(self, strategy='best1exp', popsize=10, tol=1e-8, maxiter=500, recombination=1, mutation=(0.5, 1), polish=True, disp=False, workers=-1):

        # make an array of the boundaries from the boundaries dictionary
        boundaries_array = np.array(list(self.boundaries.values()))
        
        # remove the distance parameter from the boundaries array as it is set to a constant value. Distance is later calculated from the amplitude.
        boundaries_without_distance = np.delete(boundaries_array, 4, axis=0)

        # draw an initial guess from the boundaries without distance
        initial_guess_without_distance = np.random.uniform(low=boundaries_without_distance[:, 0], high=boundaries_without_distance[:, 1])

        time_start = time.time()
        SNR = self.calculate_time_frequency_SNR_without_distance(initial_guess_without_distance)
        print('time SNR ',np.round(time.time() - time_start,2))
        print('initial guess', SNR)
        time_start = time.time()

        results = sp.optimize.differential_evolution(self.calculate_time_frequency_SNR_without_distance,            # The function only takes 10 parameters (all except dL & f_ref)
                                                                bounds=boundaries_without_distance,                 # Bounds for the 10 parameters (all except dL & f_ref)
                                                                x0=initial_guess_without_distance,                  # Initial guess for the 10 parameters (all except dL & f_ref) 
                                                                strategy=strategy,                                  # Strategy for the differential evolution
                                                                popsize=popsize,                                    # Population size
                                                                tol=tol,                                            # Tolerance for convergence
                                                                maxiter=maxiter,                                    # Maximum number of iterations
                                                                recombination=recombination,                        # Recombination factor
                                                                mutation=mutation,                                  # Mutation factor
                                                                polish=polish,                                      # Whether to polish the result
                                                                disp=disp,                                          # Whether to display the progress
                                                                workers=workers,                                    # Number of workers for parallel processing. -1 = use all cores
                                                                )
        
        print('time DE ',np.round(time.time() - time_start,2))
        print(results)

        function_evaluations = results.nfev
        found_parameters_10 = results.x

        # Add the distance to the found parameters. Distance is set to the middle of the range.
        found_parameters_11 = np.zeros(len(found_parameters_10)+1)
        found_parameters_11[:4] = found_parameters_10[:4]
        found_parameters_11[4] = self.boundaries['Distance'][0] + 0.5 * (self.boundaries['Distance'][1] - self.boundaries['Distance'][0])
        found_parameters_11[5:] = found_parameters_10[4:]
    
        # Calculate the amplitude factor and normalize the distance parameter to get the true distance MLE
        amplitude_factor = self.calculate_amplitude(found_parameters_11)
        found_parameters_11[4] /= amplitude_factor

        print(found_parameters_11, self.calculate_time_frequency_SNR_with_distance(found_parameters_11))
        return found_parameters_11, self.calculate_time_frequency_SNR_with_distance(found_parameters_11), function_evaluations

"""
def find_MBHB_in_data(start_time, end_time):
    boundaries = {}
    boundaries['Spin1'] = [-1, 1]
    boundaries['Spin2'] = [-1, 1]
    boundaries['Distance'] = [500*PC_SI*1e6, 1e6*PC_SI*1e6]
    boundaries['Phase'] = [-np.pi, np.pi]
    boundaries['Inclination'] = [-1, 1]
    boundaries['EclipticLongitude'] = [0, 2*np.pi]
    boundaries['EclipticLatitude'] = [-1, 1]
    boundaries['Polarization'] = [0, np.pi]
    boundaries['CoalescenceTime'] = [start_time, end_time]  
    boundaries['TotalMass'] = [1e5, 1e6]
    boundaries['MassRatio'] = [1, 10]
"""