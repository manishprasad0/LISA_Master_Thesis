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
from tools.save_and_load_DE import save_de_results, load_de_results

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
    Tobs = 2*(YRSID_SI/12)
    dt = 5.
    include_T_channel = False # Set to True if you want to include the T channel in the simulation, otherwise only A and E channels will be included.

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False))
    sim = LISASimulator(Tobs=Tobs, dt=dt, wave_gen=wave_gen, include_T_channel=include_T_channel)

    m1 = 3e5
    m2 = 1.5e5
    a1 = 0.753
    a2 = 0.621
    dist = 10 * PC_SI * 1e9
    phi_ref = 0.0 #np.pi/2
    f_ref = 0.0
    inc = 1.224
    lam = 3.509
    beta = 1.292
    psi = 0
    t_ref = 0.95 * Tobs
    parameters = np.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref])
    modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    waveform_kwargs = dict(length=1024, direct=False, fill=True, squeeze=False, modes=modes)

    data_t, data_f, f_array, t_array, sens_mat = sim(seed = 42, parameters=parameters, waveform_kwargs=waveform_kwargs)
    waveform_kwargs.update(freqs=f_array)

    # Pre-merger settings
    hours_before_merger = 10
    time_before_merger = hours_before_merger*60*60
    cutoff_time = t_ref - time_before_merger
    width_of_tref_prior = 20
    max_time = t_ref + (width_of_tref_prior - hours_before_merger)*60*60

    def pre_merger(gravitational_wave_data_t, time_before_merger, t_ref, t_array):
            cutoff_time = t_ref - time_before_merger
            cutoff_index = np.searchsorted(t_array, cutoff_time)
            data_t_truncated = gravitational_wave_data_t[:, :cutoff_index]
            return data_t_truncated, cutoff_index

    data_t_truncated, cutoff_index =  pre_merger(data_t, time_before_merger, t_ref, t_array)
    #signal_t_truncated, cutoff_index =  pre_merger(sim.signal_t[0], time_before_merger, t_ref, t_array)

    # Differential Evolution Analysis
    boundaries = {}
    boundaries['Total_Mass'] = [np.log(1e5), np.log(1e6)]   
    boundaries['Mass_Ratio'] = [0.05, 0.99]
    boundaries['Spin1'] = [-1, 1]
    boundaries['Spin2'] = [-1, 1]   
    boundaries['Distance'] = [1, 50] # in GPc i.e. dL / (PC_SI * 1e9)
    boundaries['Phase'] = [0, 2*np.pi]
    boundaries['cos(Inclination)'] = [-1, 1]
    boundaries['Ecliptic_Longitude'] = [0, 2*np.pi]
    boundaries['sin(Ecliptic_Latitude)'] = [-1, 1]
    boundaries['Polarization'] = [0, np.pi]
    boundaries['Coalescence_Time'] = [0, (max_time - cutoff_time)/(60*60*width_of_tref_prior)]    # Prior of 24 hours

    number_of_searches = 1
    nperseg = 5000

    # make population around transformed_true_tref
    transformed_true = transform_bbhx_to_parameters(parameters, cutoff_time)
    transformed_t_ref = transformed_true[-1]
    
    population = []
    for i in range(popsize * dim):
        candidate = []
        for j, (low, high) in enumerate(bounds):
            span = high - low
            if j < 9:
                # first 9 parameters: uniform within bounds
                val = np.random.uniform(low, high)
            else:
                # last parameter: normal perturbation around x0[-1]
                val = transformed_t_ref + np.random.normal(scale=perturb_scale * span)
                val = np.clip(val, low, high)
            candidate.append(val)
        population.append(candidate)
    population = np.array(population)

    differential_evolution_kwargs = {
        'init': population,            # great for initial run, space-filling
        'polish': True,             # yes, lets L-BFGS-B finish up
        'disp': True,               # monitor progress
        'strategy': 'rand1exp',     # good default; 'best1exp' can converge faster but risks premature convergence
        'popsize': 15,              # decent; you could try 20 if evaluations are cheap, more diversity
        'tol': 1e-6,                # loosen a bit; 1e-8 is *very* strict and often wastes iterations
        'maxiter': 1000,            # give it more room for global exploration
        'recombination': 0.7,       # lower than 1.0 usually helps maintain diversity
        'mutation': (0.7, 1.5),     # broader range → larger jumps for exploration
        'workers': -1,              # parallelism, good
        'updating': 'deferred',     # efficient with multiple workers   
    }
    
    fixed_parameters = {
        #'Total_Mass': np.log(m1 + m2),
        #'Mass_Ratio': m2 / m1,
        #'Spin1': a1,
        #'Spin2': a2,
        'Distance': boundaries['Distance'][0] + 0.5 * (boundaries['Distance'][1] - boundaries['Distance'][0]), # Always include distance in fixed parameters
        #'Phase': phi_ref,
        #'cos(Inclination)': np.cos(inc),
        #'Ecliptic_Longitude': lam,
        #'sin(Ecliptic_Latitude)': np.sin(beta),
        #'Polarization': psi,
        #'Coalescence_Time': t_ref-cutoff_time-60*60,
    }

    analysis = TimeFreqSNR(
        data_t = data_t_truncated,
        wave_gen=wave_gen,
        nperseg=nperseg,
        dt_full=dt,
        cutoff_index=cutoff_index,
        pre_merger=True
    )
    analysis.get_stft_of_data()
    true_snr, amplitude = analysis.calculate_time_frequency_SNR(*parameters, waveform_kwargs=waveform_kwargs)
    new_distance = dist /  amplitude
    print((new_distance - dist)/(PC_SI*1e9) , (new_distance-dist)/dist)
    print( "True distance       = ",  dist/(PC_SI*1e9), "Gpc")
    print( "Dist from Amplitude = ",  new_distance/(PC_SI*1e9), "Gpc")
    print( "SNR calculated      = ",  true_snr)
    print(f"True SNR: {true_snr}")
    
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

    DifferentialEvolution_time_frequency.get_stft_of_data()
    
    start_time = time.time()
    print("Starting differential evolution search...")
    
    found_parameters_tf, found_snr_found_tf, results_tf, parameters_history_tf = DifferentialEvolution_time_frequency.find_MBHB(number_of_searches=number_of_searches, 
                                                                                                                                differential_evolution_kwargs=differential_evolution_kwargs,
                                                                                                                                fixed_parameters=fixed_parameters,)
    
    end_time = time.time()
    print(f"Differential evolution search completed in {end_time - start_time:.2f} seconds.")
    
    print(DifferentialEvolution_time_frequency)
    
    found_tref = transform_parameters_to_bbhx(found_parameters_tf, cutoff_time=cutoff_time)[-1]
    print(found_tref, t_ref, found_trhef - t_ref)

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