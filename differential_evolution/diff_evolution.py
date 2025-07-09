import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

import numpy as np
import multiprocessing as mp
import time
from datetime import datetime

# LISA modules
from lisatools.utils.constants import *
from lisatools.sensitivity  import AE1SensitivityMatrix, AET1SensitivityMatrix
from bbhx.waveformbuild import BBHWaveformFD

# My modules
from tools.LISASimulator import LISASimulator
from tools.likelihood import get_dh, get_hh, TimeFreqSNR
from tools.MBHB_differential_evolution import MBHB_finder_time_frequency, transform_bbhx_to_parameters, transform_parameters_to_bbhx
from tools.save_and_load_DE import save_de_results

import psutil

mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / (1024 ** 3):.2f} GB")
print(f"Available RAM: {mem.available / (1024 ** 3):.2f} GB")
print(f"Used RAM: {mem.used / (1024 ** 3):.2f} GB")
print(f"RAM Usage: {mem.percent}%")
print("Number of CPU cores:", mp.cpu_count())

def main():
    # Set up multiprocessing
    # mp.set_start_method('fork', force=True)
    
    # Simulation parameters
    Tobs = YRSID_SI/12
    dt = 5.
    include_T_channel = False # Set to True if you want to include the T channel in the simulation, otherwise only A and E channels will be included.

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
    sim = LISASimulator(Tobs=Tobs, dt=dt, wave_gen=wave_gen, include_T_channel=include_T_channel)

    m1 = 3e5
    m2 = 1.5e5
    a1 = 0.2
    a2 = 0.4
    dist = 20 * PC_SI * 1e9
    phi_ref = np.pi/2
    f_ref = 0.0
    inc = np.pi/3
    lam = np.pi/1.
    beta = np.pi/4.
    psi = np.pi/4.
    t_ref = 0.95 * Tobs
    parameters = np.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref])
    modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    waveform_kwargs = dict(length=1024, direct=False, fill=True, squeeze=False, modes=modes)

    data_t, data_f, f_array, t_array, sens_mat = sim(seed = 42, parameters=parameters, waveform_kwargs=waveform_kwargs)
    waveform_kwargs.update(freqs=f_array)

    # Pre-merger settings
    hours_before_merger = 20
    time_before_merger = hours_before_merger*60*60
    cutoff_time = t_ref - time_before_merger
    max_time = t_ref + (24 - hours_before_merger)*60*60

    def pre_merger(gravitational_wave_data_t, time_before_merger, t_ref, t_array):
            cutoff_time = t_ref - time_before_merger
            cutoff_index = np.searchsorted(t_array, cutoff_time)
            data_t_truncated = gravitational_wave_data_t[:, :cutoff_index]
            return data_t_truncated, cutoff_index

    data_t_truncated,   cutoff_index =  pre_merger(data_t, time_before_merger, t_ref, t_array)

    # Differential Evolution Analysis
    boundaries = {}
    boundaries['Total_Mass'] = [np.log(1e5), np.log(1e6)]   
    boundaries['Mass_Ratio'] = [0.05, 0.999999]
    boundaries['Spin1'] = [-1, 1]
    boundaries['Spin2'] = [-1, 1]
    boundaries['Distance'] = [1, 50] # in GPc i.e. dL / (PC_SI * 1e9)
    boundaries['Phase'] = [0.0, 2 * np.pi]
    boundaries['cos(Inclination)'] = [-1, 1]
    boundaries['Ecliptic_Longitude'] = [0, 2*np.pi]
    boundaries['sin(Ecliptic_Latitude)'] = [-1, 1]
    boundaries['Polarization'] = [0, np.pi]
    boundaries['Coalescence_Time'] = [0, max_time - cutoff_time]

    number_of_searches = 1
    nperseg = 5000

    differential_evolution_kwargs = {
        'strategy': 'best1bin',
        'popsize': 15,
        'tol': 1e-8,
        'maxiter': 500,
        'recombination': 0.9,
        'mutation': (0.4, 0.8),
        'polish': False,
        'disp': True,
        'workers': 16,
        'updating': 'deferred',
        'init': 'latinhypercube',
    } 

    analysis = TimeFreqSNR(
        data_t_truncated,
        wave_gen=wave_gen,
        nperseg=nperseg,
        dt_full=dt,
        cutoff_index=cutoff_index,
        pre_merger=True
    )
    analysis.get_stft_of_data()
    true_snr, amplitude = analysis.calculate_time_frequency_SNR(*parameters, waveform_kwargs=waveform_kwargs)

    # For full signal, use data_t =  sim.signal_t[0] , set pre_merger=False, and comment   cutoff_index = cutoff_index
    # For pre-merger,  use data_t =  data_t_truncated, set pre_merger=True , and uncomment cutoff_index = cutoff_index
    DifferentialEvolution_time_frequency = MBHB_finder_time_frequency(
        data_t = data_t_truncated,
        wave_gen= wave_gen,
        waveform_kwargs=waveform_kwargs,
        boundaries=boundaries,
        nperseg=nperseg,
        dt_full= dt,
        pre_merger=True,
        cutoff_index=cutoff_index,
        cutoff_time=cutoff_time,
        true_parameters=parameters,
    )

    start_time = time.time()

    DifferentialEvolution_time_frequency.get_stft_of_data()
    
    print("Starting differential evolution search...")
    found_parameters_tf, found_snr_found_tf, results_tf, parameters_history_tf = DifferentialEvolution_time_frequency.find_MBHB(number_of_searches=number_of_searches, 
                                                                                                                                differential_evolution_kwargs=differential_evolution_kwargs,)
    
    end_time = time.time()
    print(f"Differential evolution search completed in {end_time - start_time:.2f} seconds.")

    save_de_results(
        found_parameters_tf,
        found_snr_found_tf,
        true_snr,
        results_tf,
        parameters_history_tf,
        folder_name="differential_evolution/differential_evolution_results",
        filename_prefix="tf_run"
    )
    
    print("Finished at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    main()