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


def transform_to_bbhx_parameters(x: np.ndarray) -> np.ndarray:
    """
    Transform parameters to BBHX format. 
    - x: array of 11 parameters: [mT, q, a1, a2, dL, phi, cos(i), lambda, sin(beta), psi, t_ref]
    Returns 
    - all_parameters: an array of 12 parameters (input for bbhx): [m1, m2, a1, a2, dL, phi, f_ref, i, lambda, beta, psi, t_ref]
    """

    all_parameters = np.zeros(12)
    mT = x[0]
    q = x[1]
    all_parameters[0] = mT / (1 + q)
    all_parameters[1] = mT * q / (1 + q)
    all_parameters[2] = x[2]
    all_parameters[3] = x[3]
    all_parameters[4] = x[4] * PC_SI * 1e6
    all_parameters[5] = x[5]
    all_parameters[6] = 0.0
    all_parameters[7] = np.arccos(x[6])
    all_parameters[8] = x[7]
    all_parameters[9] = np.arcsin(x[8])
    all_parameters[10] = x[9]
    all_parameters[11] = x[10]
    return all_parameters


class MBHB_finder:
    def __init__(self, data_t, wave_gen, cutoff_index, waveform_kwargs, boundaries, nperseg = 15000, dt_full=5.0, pre_merger=False):
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

        # waveform kwargs
        self.waveform_kwargs = waveform_kwargs
        self.boundaries = boundaries  # boundaries for the parameters
        # Other stuff
        self.parameters_sample = ['TotalMass', 'MassRatio', 'Spin1', 'Spin2', 'Distance', 'Phase', 'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'CoalescenceTime']
    
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
    
    def calculate_time_frequency_SNR(
        self,
        parameters: Any,
    ):
        template_f = self.wave_gen(*parameters, **self.waveform_kwargs)[0]
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
        #amplitude = dh/hh

        return - dh / np.sqrt(hh)
    

    def calculate_time_frequency_SNR_without_distance(
        self,
        parameters: Any,
    ):
        """ Calculate the time-frequency SNR without distance parameter.
        - parameters: 10 parameters. f_ref = 0.0 added inside the function and distance is set to the middle of the range.
        The reason is that the function optimized by differential evolution needs to only have the parameters that are optimized,
        Since distance does not affect the SNR, it is set to a constant value.
        f_ref = 0.0 is added to the parameters to be compatible with the waveform generator. 
        Returns:
        - -SNR: the negative SNR value, as we want to minimize the SNR.
        """
        params = np.zeros(len(parameters) + 2)  # +2 for f_ref and distance
        params[:4] = parameters[:4]                                                                                             # mT, q, a1, a2
        params[4] = self.boundaries['Distance'][0] + 0.5 * (self.boundaries['Distance'][1] - self.boundaries['Distance'][0])    # dL
        params[5] = parameters[4]                                                                                               # phase                   
        params[6] = 0.0                                                                                                         # f_ref is set to 0.0
        params[7:] = parameters[5:]                                                                                             # phi, cos(i), lambda, sin(beta), psi, t_ref

        parameters_bbhx = transform_to_bbhx_parameters(params)

        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
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
        #amplitude = dh/hh

        return - dh / np.sqrt(hh) # Return the negative SNR value, as we want to minimize the SNR
    

    def calculate_amplitude(self, *parameters: Any):
        template_f = self.wave_gen(*parameters, **self.waveform_kwargs)[0]
        template_f = template_f[:2]
        template_t = np.fft.irfft(template_f, axis=-1)
        if self.pre_merger:
            template_t = template_t[:, :self.cutoff_index]
        Zxx_temp_A = sp.signal.stft(template_t[0], fs=1/self.dt, nperseg=self.nperseg)[2]
        Zxx_temp_E = sp.signal.stft(template_t[1], fs=1/self.dt, nperseg=self.nperseg)[2]
        hh = self.get_hh(Zxx_temp_A, Zxx_temp_E)
        dh = self.get_dh(Zxx_temp_A, Zxx_temp_E)
        amplitude = dh/hh
        return amplitude

    def find_MBHB(self):
        initial_guess01 = np.array([np.random.rand(len(self.boundaries))])
        initial_guess = transform_to_parameters(initial_guess01, self.boundaries, self.parameters_sample)
        initial_guess01 = transform_to_01(initial_guess, self.boundaries, self.parameters_sample)
        bounds = []
        for i in range(len(self.boundaries)):
            bounds.append((0,1))
        
        initial_guess01_without_distance = np.append(initial_guess01[:,:4],initial_guess01[:,5:])

        time_start = time.time()
        SNR = self.calculate_time_frequency_SNR_without_distance(initial_guess01_without_distance)
        print('time SNR ',np.round(time.time() - time_start,2))
        print('initial guess', SNR)
        time_start = time.time()

        result01 = sp.optimize.differential_evolution(self.calculate_time_frequency_SNR_without_distance, 
                                                      bounds=bounds, 
                                                      disp=True, 
                                                      strategy='best1exp', 
                                                      popsize=10,
                                                      tol= 1e-8, 
                                                      maxiter=500, 
                                                      recombination= 1, 
                                                      mutation=(0.5,1), 
                                                      x0=initial_guess01_without_distance, 
                                                      polish=True)
        
        print('time DE ',np.round(time.time() - time_start,2))
        print(result01)
        function_evaluations = result01.nfev
        found_parameters01 = result01.x
        found_params01_with_distance = np.zeros(len(found_parameters01)+1)
        found_params01_with_distance[:4] = found_parameters01[:4]
        found_params01_with_distance[4] = 0.5
        found_params01_with_distance[5:] = found_parameters01[4:]
        found_parameters01 = found_params01_with_distance
        found_parameters = transform_to_parameters(found_parameters01, self.boundaries, self.parameters_sample)
        #found_parameters[10],  found_parameters[7], found_parameters[8], found_parameters[9] = ConvertLframeParamsToSSBframe(found_parameters[10], found_parameters[7], found_parameters[8], found_parameters[9], 0.0)
        amplitude_factor = self.calculate_amplitude(found_parameters)
        found_parameters[4] /= amplitude_factor
        print(found_parameters, self.calculate_time_frequency_SNR(found_parameters))
        # print(params_in, self.SNR(params_in), self.loglikelihood_ratio(params_in))
        return found_parameters, self.calculate_time_frequency_SNR(found_parameters), function_evaluations


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