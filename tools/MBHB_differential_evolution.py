from bbhx.utils.constants import *
import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult
import xarray as xr
import time
import os
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt

from lisatools.sensitivity  import SensitivityMatrix, AET1SensitivityMatrix, get_sensitivity, AE1SensitivityMatrix
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.datacontainer import DataResidualArray
from typing import Any, Tuple, Optional, List, Dict

# variable names:

# parameters_10 = [mT,  q, a1, a2,                   phi,        cos(i), lambda, sin(beta), psi, t_ref]
# parameters_11 = [mT,  q, a1, a2, dL/(PC_SI * 1e6), phi,        cos(i), lambda, sin(beta), psi, t_ref]
# parameters_all= [m1, m2, a1, a2, dL              , phi, f_ref,     i , lambda,     beta , psi, t_ref] = parameters_bbhx
# results.x is same as parameters_10 i.e. transformed parameters without distance and f_ref

def transform_parameters_to_bbhx(x_11: np.ndarray, cutoff_time=None) -> np.ndarray:
    """
    Transform parameters to BBHX format. 
    - x: array of 11 parameters: [mT, q, a1, a2, dL/(PC_SI * 1e6), phi, cos(i), lambda, sin(beta), psi, t_ref]
    Returns 
    - all_parameters: an array of 12 parameters (input for bbhx): [m1, m2, a1, a2, dL, phi, f_ref, i, lambda, beta, psi, t_ref]
    """

    all_parameters = np.zeros(12)
    mT = np.exp(x_11[0])
    q  = x_11[1]
    all_parameters[0] = mT / (1 + q)
    all_parameters[1] = mT * q / (1 + q)
    all_parameters[2] = x_11[2]
    all_parameters[3] = x_11[3]
    all_parameters[4] = x_11[4] * PC_SI * 1e9
    all_parameters[5] = x_11[5]
    all_parameters[6] = 0.0
    all_parameters[7] = np.arccos(x_11[6])
    all_parameters[8] = x_11[7]
    all_parameters[9] = np.arcsin(x_11[8])
    all_parameters[10] = x_11[9]
    all_parameters[11] = x_11[10] + cutoff_time if cutoff_time is not None else x_11[10]  # Add cutoff_time to t_ref if provided, otherwise keep it as is
    return all_parameters

def transform_bbhx_to_parameters(x: np.ndarray, cutoff_time=None) -> np.ndarray:
    """ Transform parameters to a format that can be used for analysis.
    - x: array of 12 true parameters: [m1, m2, a1, a2, dL, phi, f_ref, i, lambda, beta, psi, t_ref]
    Returns
    - all_parameters: an array of 11 parameters: [mT, q, a1, a2, dL/(PC_SI * 1e6), phi, cos(i), lambda, sin(beta), psi, t_ref]
    """
    all_parameters = np.zeros(11)
    all_parameters[0] = np.log(x[0] + x[1])
    all_parameters[1] = x[1] / x[0]
    all_parameters[2] = x[2]
    all_parameters[3] = x[3]
    all_parameters[4] = x[4] / (PC_SI * 1e9)
    all_parameters[5] = x[5]
    all_parameters[6] = np.cos(x[7]) # neglecting f_ref as it is not used in the analysis by jumping directly to x[7]
    all_parameters[7] = x[8]
    all_parameters[8] = np.sin(x[9])
    all_parameters[9] = x[10]
    all_parameters[10] = x[11] - cutoff_time if cutoff_time is not None else x[11]  # Subtract cutoff_time from t_ref if provided, otherwise keep it as is
    return all_parameters

class MBHB_finder_time_frequency:
    def __init__(self, data_t, wave_gen, waveform_kwargs, boundaries, cutoff_time, nperseg = 15000, dt_full=5.0, pre_merger=False, cutoff_index=None, true_parameters=None):
        self.data_t = data_t
        self.wave_gen = wave_gen
        self.dt_full = dt_full                  # This is the time step for the full data, not the STFT
        self.cutoff_index = cutoff_index        # Index to truncate the data before merger
        self.cutoff_time = cutoff_time         # Time until when data is observed
        self.pre_merger = pre_merger            # Flag to indicate if the data is pre-merger

        # All the following quantities are for after the STFT is calculated
        self.dt = None
        self.nperseg = nperseg
        self.f = None
        self.t = None
        self.Zxx_data_A = None
        self.Zxx_data_E = None
        self.sens_mat_new = None
        self.dd = None

        # waveform kwargs
        self.waveform_kwargs = waveform_kwargs
        self.boundaries = boundaries                            # boundaries for the 11 parameters including distance as a dictionary.
        self.parameter_names = list(self.boundaries.keys())     # List of parameter names from the boundaries dictionary

        # Initialize found parameters and true parameters
        self.true_parameters = true_parameters

        # Others
        self.history = []

    def __str__(self):
        if self.true_parameters is None:
            raise ValueError("True parameters are not set. Please provide true parameters to the MBHB_finder instance.")

        from io import StringIO
        buffer = StringIO()

        param_names = list(self.boundaries.keys())
        transformed_true = transform_bbhx_to_parameters(self.true_parameters, self.cutoff_time)

        # Use multiple columns for multiple runs
        if hasattr(self, "found_parameters_11_all"):
            num_runs = self.found_parameters_11_all.shape[0]

            # Header
            header = f"{'Index':<5} {'Parameter':<25} {'Lower Bound':<15}"
            for run in range(num_runs):
                header += f"{f'Found {run+1}':<20}"
            header += f"{'True':<20} {'Upper Bound':<15} {'Status':<10}"
            buffer.write(header + "\n")
            buffer.write('-' * len(header) + '\n')

            for i, param in enumerate(param_names):
                lower, upper = self.boundaries[param]
                true = transformed_true[i]

                if param == 'Distance':
                    status = 'variable'
                elif param in self.fixed_parameters:
                    status = 'fixed'
                else:
                    status = 'variable'

                row = f"{i:<5} {param:<25} {lower:<15.6g}"
                for run in range(num_runs):
                    found = self.found_parameters_11_all[run, i]
                    row += f"{found:<20.6g}"
                row += f"{true:<20.6g} {upper:<15.6g} {status:<10}"
                buffer.write(row + "\n")

            # SNRs
            buffer.write("\nSNRs for each run:\n")
            for i, snr in enumerate(self.SNR_all):
                buffer.write(f"Run {i+1:<2}: SNR = {snr:.8f}\n")

        return buffer.getvalue()


    def callback(self, intermediate_result: OptimizeResult):
        """
        Callback function to store the history of the optimization process.
        """
        self.history.append({
            'xk': intermediate_result.x.copy(),
            'fun': intermediate_result.fun,
            'nit': intermediate_result.nit,
            'message': intermediate_result.message,
            'success': intermediate_result.success,
            'convergence': intermediate_result.convergence
        })


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

    def calculate_amplitude(self, parameters_11: Any):

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_parameters_to_bbhx(parameters_11, self.cutoff_time)
        
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
        - parameters: [mT,  q, a1, a2, dL/(PC_SI * 1e6), phi, cos(i), lambda, sin(beta), psi, t_ref-cutoff_time]
        """

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_parameters_to_bbhx(parameters, self.cutoff_time) 

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

        # Return the normal positive SNR value as we are not optimizing this function
        return dh / np.sqrt(hh) 
    
    def calculate_time_frequency_SNR_without_distance(
        self,
        variable_parameters: np.ndarray,
        fixed_parameters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """ Calculate the time-frequency SNR without distance parameter.
        - variable_parameters: the parameters that need to be optimized with differential evolution
        - fixed_parameters: dictionary of fixed parameters. Always include {'Distance': 1e6} as a fixed parameter

        For the most general case, have variable_parameters = parameters_10: [mT,  q, a1, a2, phi, cos(i), lambda, sin(beta), psi, t_ref]
        Since distance does not affect the SNR, it is set to a constant value.
        f_ref = 0.0 is added to the parameters in the function transform_parameters_to_bbhx to be compatible with the waveform generator. 
        
        Returns:
        - -SNR: the negative SNR value, as we want to minimize the SNR.
        """
        
        # Combine fixed and variable parameters into a single array of 11 parameters
        parameters_11 = []
        variable_parameter_index = 0
        for name in self.parameter_names:
            if name in fixed_parameters:
                parameters_11.append(fixed_parameters[name])
            else:
                parameters_11.append(variable_parameters[variable_parameter_index])
                variable_parameter_index += 1
        parameters_11 = np.array(parameters_11)
        
        # Transform to BBHX parameters
        parameters_bbhx = transform_parameters_to_bbhx(parameters_11, self.cutoff_time)

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

        # Return the negative SNR value, as we want to minimize the SNR
        return - dh / np.sqrt(hh)

    def find_MBHB(self, 
                  number_of_searches= 1,
                  differential_evolution_kwargs = None, 
                  fixed_parameters: Optional[Dict[str, Any]] = None):

        if fixed_parameters is None:
            fixed_parameters = {list(self.boundaries.items())[4][0] : list(self.boundaries.items())[4][1][0] + 0.5 * (list(self.boundaries.items())[4][1][1] - list(self.boundaries.items())[4][1][0])}


        self.fixed_parameters = fixed_parameters

        variable_parameter_names = [name for name in self.parameter_names if name not in fixed_parameters]
        bounds = np.array([self.boundaries[name] for name in variable_parameter_names])
        
        found_parameters_11_all = []
        SNR_all = []
        results_all = []
        parameters_history = []

        for search_index in range(number_of_searches):

            self.history = [] 
            
            initial_guess_without_distance = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])

            #time_start = time.time()
            #SNR = self.calculate_time_frequency_SNR_without_distance(variable_parameters=initial_guess_without_distance, fixed_parameters=fixed_parameters)
            #print('time SNR ',np.round(time.time() - time_start,2))
            #print('initial guess', SNR)
            #time_start = time.time()

            results = sp.optimize.differential_evolution(self.calculate_time_frequency_SNR_without_distance,    # The function only takes 10 parameters (all except dL & f_ref)
                                                        bounds=bounds,                                          # Bounds for the 10 parameters (all except dL & f_ref)
                                                        x0=initial_guess_without_distance,                      # Initial guess for the 10 parameters (all except dL & f_ref) 
                                                        args=(fixed_parameters,),
                                                        **differential_evolution_kwargs,                        # Additional keyword arguments for the differential evolution algorithm
                                                        callback=self.callback,   # <--- here
                                                        )
            results_all.append(results)
            
            # Store the history of the optimization process
            parameters_history.append([step['xk'] for step in self.history])

            # Extract the optimized parameters from the results and combine them with the fixed parameters
            found_parameters = {}                                                                                # This will hold the complete set of parameters, both fixed and optimized
            variable_parameter_index = 0                                                                         # Index to track where we are in results.x (the optimized free parameters)

            for name in self.parameter_names:
                if name in fixed_parameters:
                    # If this parameter was fixed, take its value directly from the fixed_parameters dictionary
                    found_parameters[name] = fixed_parameters[name]
                else:
                    # If this parameter was optimized by differential evolution, take its value from results.x
                    found_parameters[name] = results.x[variable_parameter_index]
                    variable_parameter_index += 1                                                                # Move to the next optimized parameter

            found_parameters_11 = np.array([found_parameters[name] for name in self.parameter_names])
            
            # Calculate the amplitude factor and normalize the distance parameter to get the true distance MLE
            amplitude_factor = self.calculate_amplitude(found_parameters_11)
            found_parameters_11[4] /= amplitude_factor

            found_parameters_11_all.append(found_parameters_11)
            SNR_all.append(self.calculate_time_frequency_SNR_with_distance(found_parameters_11))                 # function calculate_time_frequency_SNR_with_distance does not multiply the SNR by -1

        found_parameters_11_all = np.array(found_parameters_11_all)
        SNR_all = np.array(SNR_all)
        results_all = np.array(results_all)
        parameters_history = np.array(parameters_history)

        # Find the maximum SNR and the corresponding parameters. Taking argmax because the function calculate_time_frequency_SNR_with_distance does not multiply the SNR by -1
        max_index = np.argmax(SNR_all)
        found_parameters_11_max = found_parameters_11_all[max_index]
        SNR_max = SNR_all[max_index]
        results_max = results_all[max_index]
        parameters_history_max = parameters_history[max_index]

        # Store the found parameters and SNR values
        self.found_parameters_11_all = found_parameters_11_all
        self.SNR_all = SNR_all

        self.found_parameters_11_max = found_parameters_11_max
        self.SNR_max = SNR_max

        return found_parameters_11_max, SNR_max, results_max, parameters_history_max
    

class MBHB_finder_frequency_domain:
    def __init__(self, data_f, wave_gen, waveform_kwargs, boundaries, true_parameters=None):
        self.data_f = data_f
        self.wave_gen = wave_gen

        # All the following quantities are for after the STFT is calculated
        self.dt = None
        self.f = None
        self.t = None
        self.Zxx_data_A = None
        self.Zxx_data_E = None
        self.sens_mat_new = None
        self.dd = None

        # waveform kwargs
        self.waveform_kwargs = waveform_kwargs
        self.boundaries = boundaries                            # boundaries for the 11 parameters including distance as a dictionary.
        self.parameter_names = list(self.boundaries.keys())     # List of parameter names from the boundaries dictionary

        # Initialize found parameters and true parameters
        self.true_parameters = true_parameters

    def __str__(self):
        if self.true_parameters is None:
            raise ValueError("True parameters are not set. Please provide true parameters to the MBHB_finder instance.")

        from io import StringIO
        buffer = StringIO()

        param_names = list(self.boundaries.keys())
        transformed_true = transform_bbhx_to_parameters(self.true_parameters)

        # Use multiple columns for multiple runs
        if hasattr(self, "found_parameters_11_all"):
            num_runs = self.found_parameters_11_all.shape[0]

            # Header
            header = f"{'Index':<5} {'Parameter':<25} {'Lower Bound':<15}"
            for run in range(num_runs):
                header += f"{f'Found {run+1}':<20}"
            header += f"{'True':<20} {'Upper Bound':<15} {'Status':<10}"
            buffer.write(header + "\n")
            buffer.write('-' * len(header) + '\n')

            for i, param in enumerate(param_names):
                lower, upper = self.boundaries[param]
                true = transformed_true[i]

                if param == 'Distance':
                    status = 'variable'
                elif param in self.fixed_parameters:
                    status = 'fixed'
                else:
                    status = 'variable'

                row = f"{i:<5} {param:<25} {lower:<15.6g}"
                for run in range(num_runs):
                    found = self.found_parameters_11_all[run, i]
                    row += f"{found:<20.6g}"
                row += f"{true:<20.6g} {upper:<15.6g} {status:<10}"
                buffer.write(row + "\n")

            # SNRs
            buffer.write("\nSNRs for each run:\n")
            for i, snr in enumerate(self.SNR_all):
                buffer.write(f"Run {i+1:<2}: SNR = {snr:.8f}\n")

        return buffer.getvalue()



    def prepare_data(self, f_array, df, include_sens_kwargs=False):
        self.f = f_array
        self.df = df

        self.Zxx_data_A = self.data_f[0]
        self.Zxx_data_E = self.data_f[1]

        if include_sens_kwargs:
            sens_kwargs = dict(stochastic_params=(self.Tobs,))
            self.sens_mat_new = AE1SensitivityMatrix(self.f, **sens_kwargs).sens_mat
        else:
            self.sens_mat_new = AE1SensitivityMatrix(self.f).sens_mat

    def get_hh(self, Zxx_A, Zxx_E):
        inner_product_A = np.abs(Zxx_A)**2 / self.sens_mat_new[0]
        inner_product_E = np.abs(Zxx_E)**2 / self.sens_mat_new[1]

        return 4 * self.df * np.sum(inner_product_A + inner_product_E)
    
    def get_dh(self, Zxx_A, Zxx_E):
        inner_product_A = (Zxx_A * np.conj(self.Zxx_data_A)) / self.sens_mat_new[0]
        inner_product_E = (Zxx_E * np.conj(self.Zxx_data_E)) / self.sens_mat_new[1]

        return 4 * self.df * np.sum(inner_product_A.real + inner_product_E.real)

    def calculate_amplitude(self, parameters_11: Any):

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_parameters_to_bbhx(parameters_11)
        
        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        
        hh = self.get_hh(template_f[0], template_f[1])
        dh = self.get_dh(template_f[0], template_f[1])

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
        parameters_bbhx = transform_parameters_to_bbhx(parameters) 

        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
 
        # Calculate the inner product for A and E channels
        hh = self.get_hh(template_f[0], template_f[1])
        dh = self.get_dh(template_f[0], template_f[1])

        # Return the normal positive SNR value as we are not optimizing this function
        return dh / np.sqrt(hh) 
    
    def calculate_time_frequency_SNR_without_distance(
        self,
        variable_parameters: np.ndarray,
        fixed_parameters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """ Calculate the time-frequency SNR without distance parameter.
        - variable_parameters: the parameters that need to be optimized with differential evolution
        - fixed_parameters: dictionary of fixed parameters. Always include {'Distance': 1e6} as a fixed parameter

        For the most general case, have variable_parameters = parameters_10: [mT,  q, a1, a2, phi, cos(i), lambda, sin(beta), psi, t_ref]
        Since distance does not affect the SNR, it is set to a constant value.
        f_ref = 0.0 is added to the parameters in the function transform_parameters_to_bbhx to be compatible with the waveform generator. 
        
        Returns:
        - -SNR: the negative SNR value, as we want to minimize the SNR.
        """
        if fixed_parameters is None:
            fixed_parameters = {list(self.boundaries.items())[4][0] : list(self.boundaries.items())[4][1][0] + 0.5 * (list(self.boundaries.items())[4][1][1] - list(self.boundaries.items())[4][1][0])}

        # Combine fixed and variable parameters into a single array of 11 parameters
        parameters_11 = []
        variable_parameter_index = 0
        for name in self.parameter_names:
            if name in fixed_parameters:
                parameters_11.append(fixed_parameters[name])
            else:
                parameters_11.append(variable_parameters[variable_parameter_index])
                variable_parameter_index += 1
        parameters_11 = np.array(parameters_11)

        # Transform to BBHX parameters
        parameters_bbhx = transform_parameters_to_bbhx(parameters_11)

        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        
        # Calculate the inner products for A and E channels
        hh = self.get_hh(template_f[0], template_f[1])
        dh = self.get_dh(template_f[0], template_f[1])

        # Return the negative SNR value, as we want to minimize the SNR
        return - dh / np.sqrt(hh)

    def find_MBHB(self, 
                  number_of_searches= 1,
                  differential_evolution_kwargs = None, 
                  fixed_parameters: Optional[Dict[str, Any]] = None):

        if fixed_parameters is None:
            raise ValueError("Distance must be included as a fixed parameter. Please provide a dictionary with 'Distance' as a key and its value.")

        self.fixed_parameters = fixed_parameters

        variable_parameter_names = [name for name in self.parameter_names if name not in fixed_parameters]
        bounds = np.array([self.boundaries[name] for name in variable_parameter_names])
        
        found_parameters_11_all = []
        SNR_all = []
        results_all = []

        for search_index in range(number_of_searches):

            initial_guess_without_distance = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])

            time_start = time.time()
            SNR = self.calculate_time_frequency_SNR_without_distance(variable_parameters=initial_guess_without_distance, fixed_parameters=fixed_parameters)
            print('time SNR ',np.round(time.time() - time_start,2))
            print('initial guess', SNR)
            time_start = time.time()

            results = sp.optimize.differential_evolution(self.calculate_time_frequency_SNR_without_distance,    # The function only takes 10 parameters (all except dL & f_ref)
                                                        bounds=bounds,                                          # Bounds for the 10 parameters (all except dL & f_ref)
                                                        x0=initial_guess_without_distance,                      # Initial guess for the 10 parameters (all except dL & f_ref) 
                                                        args=(fixed_parameters,),
                                                        **differential_evolution_kwargs,                        # Additional keyword arguments for the differential evolution algorithm
                                                        )
            results_all.append(results)

            # Extract the optimized parameters from the results and combine them with the fixed parameters
            found_parameters = {}                                                                                # This will hold the complete set of parameters, both fixed and optimized
            variable_parameter_index = 0                                                                         # Index to track where we are in results.x (the optimized free parameters)

            for name in self.parameter_names:
                if name in fixed_parameters:
                    # If this parameter was fixed, take its value directly from the fixed_parameters dictionary
                    found_parameters[name] = fixed_parameters[name]
                else:
                    # If this parameter was optimized by differential evolution, take its value from results.x
                    found_parameters[name] = results.x[variable_parameter_index]
                    variable_parameter_index += 1                                                                # Move to the next optimized parameter

            found_parameters_11 = np.array([found_parameters[name] for name in self.parameter_names])
            
            # Calculate the amplitude factor and normalize the distance parameter to get the true distance MLE
            amplitude_factor = self.calculate_amplitude(found_parameters_11)
            found_parameters_11[4] /= amplitude_factor

            found_parameters_11_all.append(found_parameters_11)
            SNR_all.append(self.calculate_time_frequency_SNR_with_distance(found_parameters_11))                 # function calculate_time_frequency_SNR_with_distance does not multiply the SNR by -1

        found_parameters_11_all = np.array(found_parameters_11_all)
        SNR_all = np.array(SNR_all)

        # Find the maximum SNR and the corresponding parameters. Taking argmax because the function calculate_time_frequency_SNR_with_distance does not multiply the SNR by -1
        max_index = np.argmax(SNR_all)
        found_parameters_11_max = found_parameters_11_all[max_index]
        SNR_max = SNR_all[max_index]

        # Store the found parameters and SNR values
        self.found_parameters_11_all = found_parameters_11_all
        self.SNR_all = SNR_all

        self.found_parameters_11_max = found_parameters_11_max
        self.SNR_max = SNR_max

        return found_parameters_11_max, SNR_max, results_all




class MBHB_finder_lisatools:
    def __init__(self, data_f, sens_mat, wave_gen, waveform_kwargs, f_array, boundaries, dt_full=5.0, pre_merger=False, cutoff_index=None, true_parameters=None):
        self.data_f = data_f
        self.sens_mat = sens_mat
        self.wave_gen = wave_gen
        self.dt_full = dt_full                  # This is the time step for the full data, not the STFT
        self.cutoff_index = cutoff_index        # Index to truncate the data before merger
        self.pre_merger = pre_merger            # Flag to indicate if the data is pre-merger
        self.f_array = f_array
        # All the following quantities are for after the STFT is calculated
        

        # waveform kwargs
        self.waveform_kwargs = waveform_kwargs
        self.boundaries = boundaries                            # boundaries for the 11 parameters including distance as a dictionary.
        self.parameter_names = list(self.boundaries.keys())     # List of parameter names from the boundaries dictionary

        # Initialize found parameters and true parameters
        self.true_parameters = true_parameters

    def __str__(self):
        if self.true_parameters is None:
            raise ValueError("True parameters are not set. Please provide true parameters to the MBHB_finder instance.")

        from io import StringIO
        buffer = StringIO()

        param_names = list(self.boundaries.keys())
        transformed_true = transform_bbhx_to_parameters(self.true_parameters)

        # Use multiple columns for multiple runs
        if hasattr(self, "found_parameters_11_all"):
            num_runs = self.found_parameters_11_all.shape[0]

            # Header
            header = f"{'Index':<5} {'Parameter':<25} {'Lower Bound':<15}"
            for run in range(num_runs):
                header += f"{f'Found {run+1}':<20}"
            header += f"{'True':<20} {'Upper Bound':<15} {'Status':<10}"
            buffer.write(header + "\n")
            buffer.write('-' * len(header) + '\n')

            for i, param in enumerate(param_names):
                lower, upper = self.boundaries[param]
                true = transformed_true[i]

                if param == 'Distance':
                    status = 'variable'
                elif param in self.fixed_parameters:
                    status = 'fixed'
                else:
                    status = 'variable'

                row = f"{i:<5} {param:<25} {lower:<15.6g}"
                for run in range(num_runs):
                    found = self.found_parameters_11_all[run, i]
                    row += f"{found:<20.6g}"
                row += f"{true:<20.6g} {upper:<15.6g} {status:<10}"
                buffer.write(row + "\n")

            # SNRs
            buffer.write("\nSNRs for each run:\n")
            for i, snr in enumerate(self.SNR_all):
                buffer.write(f"Run {i+1:<2}: SNR = {snr:.8f}\n")

        return buffer.getvalue()



    def get_analysis_container_of_data(self):
        data_res_arr = DataResidualArray(self.data_f, f_arr=self.f_array)
        analysis = AnalysisContainer(data_res_arr, self.sens_mat)
        self.analysis = analysis

        print('SNR of the data:', analysis.snr())

    def calculate_amplitude(self, parameters_11: Any):

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_parameters_to_bbhx(parameters_11)
        
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel
        
        template = DataResidualArray(template_f, f_arr=self.f_array)
        
        # opt = sqrt(hh), det = dh
        opt_snr, det_snr = self.analysis.template_snr(template=template)

        # Return dh/hh = dh/[sqrt(hh)**2]. opt_snr = sqrt(hh) and det_snr = dh/sqrt(hh), so we return det_snr/opt_snr
        return det_snr/opt_snr
    
    def calculate_lisatools_SNR_with_distance(
        self,
        parameters: Any,
        ):
        """ Calculate the time-frequency SNR with distance parameter.
        - parameters: [mT,  q, a1, a2, dL/(PC_SI * 1e6), phi, cos(i), lambda, sin(beta), psi, t_ref]
        """

        # Transform the parameters to BBHX format by adding f_ref = 0.0
        parameters_bbhx = transform_parameters_to_bbhx(parameters) 

        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel
        
        template = DataResidualArray(template_f, f_arr=self.f_array)
        
        # Calculate the STFT of the template
        opt_snr, det_snr = self.analysis.template_snr(template=template)

        # Return the negative SNR value, as we want to minimize the SNR
        return det_snr
    
    def calculate_lisatools_SNR_without_distance(
        self,
        variable_parameters: np.ndarray,
        fixed_parameters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """ Calculate the time-frequency SNR without distance parameter.
        - variable_parameters: the parameters that need to be optimized with differential evolution
        - fixed_parameters: dictionary of fixed parameters. Always include {'Distance': 1e6} as a fixed parameter

        For the most general case, have variable_parameters = parameters_10: [mT,  q, a1, a2, phi, cos(i), lambda, sin(beta), psi, t_ref]
        Since distance does not affect the SNR, it is set to a constant value.
        f_ref = 0.0 is added to the parameters in the function transform_parameters_to_bbhx to be compatible with the waveform generator. 
        
        Returns:
        - -SNR: the negative SNR value, as we want to minimize the SNR.
        """
        if fixed_parameters is None:
            fixed_parameters = {list(self.boundaries.items())[4][0] : list(self.boundaries.items())[4][1][0] + 0.5 * (list(self.boundaries.items())[4][1][1] - list(self.boundaries.items())[4][1][0])}

        # Combine fixed and variable parameters into a single array of 11 parameters
        parameters_11 = []
        variable_parameter_index = 0
        for name in self.parameter_names:
            if name in fixed_parameters:
                parameters_11.append(fixed_parameters[name])
            else:
                parameters_11.append(variable_parameters[variable_parameter_index])
                variable_parameter_index += 1
        parameters_11 = np.array(parameters_11)
        
        # Transform to BBHX parameters
        parameters_bbhx = transform_parameters_to_bbhx(parameters_11)

        # Generate the waveform template with parameters_bbhx and remove the T channel
        template_f = self.wave_gen(*parameters_bbhx, **self.waveform_kwargs)[0]
        template_f = template_f[:2] # remove T channel

        template = DataResidualArray(template_f, f_arr=self.f_array)

        # Calculate the STFT of the template
        opt_snr, det_snr = self.analysis.template_snr(template=template)

        # Return the negative SNR value, as we want to minimize the SNR
        return - det_snr

    def find_MBHB(self, 
                  number_of_searches= 1,
                  differential_evolution_kwargs = None, 
                  fixed_parameters: Optional[Dict[str, Any]] = None):

        if fixed_parameters is None:
            raise ValueError("Distance must be included as a fixed parameter. Please provide a dictionary with 'Distance' as a key and its value.")

        self.fixed_parameters = fixed_parameters

        variable_parameter_names = [name for name in self.parameter_names if name not in fixed_parameters]
        bounds = np.array([self.boundaries[name] for name in variable_parameter_names])
        
        found_parameters_11_all = []
        SNR_all = []
        results_all = []

        for search_index in range(number_of_searches):

            initial_guess_without_distance = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])

            time_start = time.time()
            SNR = self.calculate_lisatools_SNR_without_distance(variable_parameters=initial_guess_without_distance, fixed_parameters=fixed_parameters)
            print('time SNR ',np.round(time.time() - time_start,2))
            print('initial guess', SNR)
            time_start = time.time()

            results = sp.optimize.differential_evolution(self.calculate_lisatools_SNR_without_distance,    # The function only takes 10 parameters (all except dL & f_ref)
                                                        bounds=bounds,                                          # Bounds for the 10 parameters (all except dL & f_ref)
                                                        x0=initial_guess_without_distance,                      # Initial guess for the 10 parameters (all except dL & f_ref) 
                                                        args=(fixed_parameters,),
                                                        **differential_evolution_kwargs,                        # Additional keyword arguments for the differential evolution algorithm
                                                        )
            results_all.append(results)

            # Extract the optimized parameters from the results and combine them with the fixed parameters
            found_parameters = {}                                                                                # This will hold the complete set of parameters, both fixed and optimized
            variable_parameter_index = 0                                                                         # Index to track where we are in results.x (the optimized free parameters)

            for name in self.parameter_names:
                if name in fixed_parameters:
                    # If this parameter was fixed, take its value directly from the fixed_parameters dictionary
                    found_parameters[name] = fixed_parameters[name]
                else:
                    # If this parameter was optimized by differential evolution, take its value from results.x
                    found_parameters[name] = results.x[variable_parameter_index]
                    variable_parameter_index += 1                                                                # Move to the next optimized parameter

            found_parameters_11 = np.array([found_parameters[name] for name in self.parameter_names])
            
            # Calculate the amplitude factor and normalize the distance parameter to get the true distance MLE
            amplitude_factor = self.calculate_amplitude(found_parameters_11)
            found_parameters_11[4] /= amplitude_factor

            found_parameters_11_all.append(found_parameters_11)
            SNR_all.append(self.calculate_lisatools_SNR_with_distance(found_parameters_11))                 # function calculate_time_frequency_SNR_with_distance does not multiply the SNR by -1

        found_parameters_11_all = np.array(found_parameters_11_all)
        SNR_all = np.array(SNR_all)

        # Find the maximum SNR and the corresponding parameters. Taking argmax because the function calculate_time_frequency_SNR_with_distance does not multiply the SNR by -1
        max_index = np.argmax(SNR_all)
        found_parameters_11_max = found_parameters_11_all[max_index]
        SNR_max = SNR_all[max_index]

        # Store the found parameters and SNR values
        self.found_parameters_11_all = found_parameters_11_all
        self.SNR_all = SNR_all

        self.found_parameters_11_max = found_parameters_11_max
        self.SNR_max = SNR_max

        return found_parameters_11_max, SNR_max, results_all
    



"""
    def calculate_time_frequency_SNR_without_distance(
        # Add distance parameter to the parameters_10 and set it to the middle of the range
        parameters_11 = np.zeros(len(parameters_10) + 1)  # +1 for f_ref and distance
        parameters_11[:4] = parameters_10[:4]                                                                                           # mT, q, a1, a2
        parameters_11[4] = self.boundaries['Distance'][0] + 0.5 * (self.boundaries['Distance'][1] - self.boundaries['Distance'][0])     # dL = mid point of the distance prior
        parameters_11[5:] = parameters_10[4:]                                                                                           # phase, phi, cos(i), lambda, sin(beta), psi, t_ref

        
    def find_MBHB():
    
        # make an array of the boundaries from the boundaries dictionary
        boundaries_array  = np.array(list(self.boundaries.values()))
        
        # remove the distance parameter from the boundaries array as it is set to a constant value. Distance is later calculated from the amplitude.
        boundaries_without_distance = np.delete(boundaries_array, 4, axis=0)

        # draw an initial guess from the boundaries without distance
        initial_guess_without_distance = np.random.uniform(low=boundaries_without_distance[:, 0], high=boundaries_without_distance[:, 1])

        if close_initial_guess:
            initial_guess_without_distance = self.initial_guess(variable_parameter_names, bounds, closeness_factor=0.1)
        else:
            initial_guess_without_distance = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])
"""