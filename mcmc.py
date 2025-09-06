import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import time
from datetime import datetime
from scipy.stats import truncnorm


# LISA modules
from lisatools.utils.constants import *
from lisatools.sensitivity  import AE1SensitivityMatrix
from bbhx.waveformbuild import BBHWaveformFD

# Immports for generating simulated LISA data
import noise_generation as noise_generation
from tools.LISASimulator import LISASimulator
from tools.likelihood import get_dh, get_hh
import tools.likelihood as likelihood

# Imports for MCMC
from tools.save_and_load_DE import load_de_results
from tools.time_freq_likelihood import TimeFreqLikelihood
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from scipy.stats import truncnorm

# Imports for plotting
from chainconsumer import Chain, ChainConsumer, make_sample, Truth
import pandas as pd

gpu = False
#if gpu:
#    import cupy as cp
#else:
    #import numpy as cp

#import psutil
#mem = psutil.virtual_memory()
#print(f"Total RAM: {mem.total / (1024 ** 3):.2f} GB")
#print(f"Available RAM: {mem.available / (1024 ** 3):.2f} GB")
#print(f"Used RAM: {mem.used / (1024 ** 3):.2f} GB")
#print(f"RAM Usage: {mem.percent}%")
#print("Number of CPU cores:", mp.cpu_count())

def main():
    Tobs = 1.2*YRSID_SI/12
    dt = 5.
    include_T_channel = False

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), use_gpu=gpu)
    sim = LISASimulator(Tobs=Tobs, dt=dt, wave_gen=wave_gen, include_T_channel=include_T_channel)

    m1 = 3e5
    m2 = 1.5e5
    a1 = 0.753
    a2 = 0.621
    dist = 10 * PC_SI * 1e9
    phi_ref = 0.0 #np.pi/2
    f_ref = 0.0
    inc = 0.224
    lam = 60*(np.pi/180)
    beta = 20*(np.pi/180)
    psi = 0
    t_ref = Tobs - (24*60*60)
    parameters = np.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref])
    modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    waveform_kwargs = dict(length=1024, direct=False, fill=True, squeeze=False, modes=modes)

    data_t, data_f, f_array, t_array, sens_mat = sim(seed = 42, parameters=parameters, waveform_kwargs=waveform_kwargs)
    waveform_kwargs.update(freqs=f_array)

    print(sim.SNR_optimal()[0])

    # Pre-merger settings
    hours_before_merger = 10
    time_before_merger = hours_before_merger*60*60
    cutoff_time = t_ref - time_before_merger
    width_of_tref_prior = 20
    max_time = t_ref + (width_of_tref_prior - hours_before_merger)*60*60
    nperseg = 1414

    def pre_merger(gravitational_wave_data_t, time_before_merger, t_ref, t_array):
            cutoff_time = t_ref - time_before_merger
            cutoff_index = np.searchsorted(t_array, cutoff_time)
            data_t_truncated = gravitational_wave_data_t[:, :cutoff_index]
            return data_t_truncated, cutoff_index

    data_t_truncated, cutoff_index =  pre_merger(data_t, time_before_merger, t_ref, t_array)

    # MCMC
    analysis = TimeFreqLikelihood(data_t=data_t, wave_gen=wave_gen, nperseg=nperseg)
    analysis.pre_merger(time_before_merger=time_before_merger, t_ref=t_ref, t_array=t_array)
    analysis.get_stft_of_data()
    print("Best likelihood (true parameters)= ", analysis.calculate_time_frequency_likelihood(*parameters, waveform_kwargs=waveform_kwargs))

    # MCMC parameters
    nwalkers = 24
    ntemps = 4
    ndims = 11
    nleaves_max = 1
    nsteps = 1000

    param_labels = [
        r"$M_T \, [\mathrm{M_\odot}]$",
        r"$q$",
        r"$a_1$",
        r"$a_2$",
        r"$d_L \, [\mathrm{Gpc}]$",
        r"$\phi_{\mathrm{ref}}$",
        r"$\cos(\iota)$",
        r"$\lambda$",
        r"$\sin(\beta)$",
        r"$\psi$",
        r"$t_{\mathrm{c}} \, [\mathrm{hrs}]$"
    ]

    # Results from Differential Evolution
    def DE_to_MCMC_params(found_parameters_DE, cutoff_time):
        mT, q, a1, a2, dist_Gpc, phi_ref, cos_inc, lam, sin_beta, psi, t_ref_01 = found_parameters_DE
        mT_exp = np.exp(mT)
        dist_Mpc = dist_Gpc 
        phi_ref = phi_ref % (2*np.pi)
        time_to_coalescence = t_ref_01 * 24
        return np.array([mT_exp, q, a1, a2, dist_Mpc, phi_ref, cos_inc, lam, sin_beta, psi, time_to_coalescence])
        
    def likelihood(x, freqs, TimeFreqLikelihood_object, cutoff_time):
        all_parameters = np.zeros(12)
        mT = x[0]
        q = x[1]
        all_parameters[0] = mT / (1 + q)
        all_parameters[1] = mT * q / (1 + q)
        all_parameters[2] = x[2]
        all_parameters[3] = x[3]
        all_parameters[4] = x[4] * PC_SI * 1e9
        all_parameters[5] = x[5]
        all_parameters[6] = f_ref
        all_parameters[7] = np.arccos(x[6])
        all_parameters[8] = x[7]
        all_parameters[9] = np.arcsin(x[8])
        all_parameters[10] = x[9]
        all_parameters[11] = x[10] + cutoff_time

        ll = TimeFreqLikelihood_object.calculate_time_frequency_likelihood(
            *all_parameters,
            waveform_kwargs=dict(
            length=1024, 
            direct=False,
            fill=True,
            squeeze=False,
            freqs=freqs,
            modes=modes
            )
        )
        return ll
    
    priors = {"mbh": ProbDistContainer({
        0 : uniform_dist(1e5, 1e6),                  # mT = m1 + m2
        1 : uniform_dist(0.01, 0.99),                # q = m2/m1
        2 : uniform_dist(-1, +1),                    # a1
        3 : uniform_dist(-1, +1),                    # a2
        4 : uniform_dist(1, 50),                     # dist in Mpc
        5 : uniform_dist(0, 2*np.pi),                # phi_ref
        6 : uniform_dist(-1, 1),                     # cos(inc)
        7 : uniform_dist(0.0, 2 * np.pi),            # lam
        8 : uniform_dist(-1.0, 1.0),                 # sin(beta)
        9 : uniform_dist(0.0, np.pi),                # psi
        10: uniform_dist(0, width_of_tref_prior),    # t_ref
    })}

    periodic = {"mbh": {5: 2 * np.pi,   # phi_ref
                        7: 2 * np.pi,   # lam
                        9: np.pi  }}    # psi
    
    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        likelihood,
        priors,
        args=(f_array, analysis, cutoff_time),
        branch_names=["mbh"],
        tempering_kwargs=dict(ntemps = ntemps),
        nleaves_max=dict(mbh = nleaves_max),
        periodic=periodic
    )

    injection_parameters = np.array([m1+m2, m2/m1, a1, a2, dist / (PC_SI * 1e9), phi_ref, np.cos(inc), lam, np.sin(beta), psi, (t_ref-cutoff_time)/(60*60)])

    x0 = load_de_results(filepath='differential_evolution/differential_evolution_results/tf_run_20250902_183628.npz')
    found_parameters_DE = DE_to_MCMC_params(x0['found_parameters'], cutoff_time=cutoff_time)
    found_SNR = x0['found_snr']
    print("Parameters from Differential Evolution: ", found_parameters_DE)
    print("SNR from Differential Evolution: ", found_SNR)

    starting_points = np.zeros(shape=[ntemps, nwalkers, nleaves_max, found_parameters_DE.shape[0]])

    # Perturb the injection parameters to create starting points for the walkers
    perturb_frac = 0.001

    non_periodic_params = [0, 1, 2, 3, 4, 6, 7, 8, 10]  # indices of non-periodic parameters

    # For non-periodic parameters, draw from a truncated normal distribution around the injection parameters
    for i in non_periodic_params:
        low, high = priors["mbh"].priors[i][1].min_val, priors["mbh"].priors[i][1].max_val
        mu = found_parameters_DE[i]
        sigma = perturb_frac * (high - low)
        a, b = (low - mu) / sigma, (high - mu) / sigma
        starting_points[:, :, :, i] = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=(ntemps, nwalkers, nleaves_max))

    # For periodic parameters, draw from a normal distribution around the injection parameters and then wrap around the period
    for i, period in periodic["mbh"].items():
        sigma = perturb_frac * period
        perturbed = found_parameters_DE[i] + sigma * np.random.randn(ntemps, nwalkers, nleaves_max)
        starting_points[:, :, :, i] = np.mod(perturbed, period)

    starting_state = State({"mbh": starting_points})
    
    start_time = time.time()
    print("Starting MCMC")
    sampler.run_mcmc(starting_state, nsteps=nsteps, progress=True)
    end_time = time.time()
    print(f"MCMC with {nsteps} steps took {end_time - start_time:.2f} seconds")
    
    mcmc_results = sampler.get_chain()["mbh"]




    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mcmc_results_folder = "mcmc_results"
    folder_name = f"{mcmc_results_folder}/run_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    fig.savefig(f"{folder_name}/example_plot.png", dpi=300)




    np.save(f"{folder_name}/chain.npy", mcmc_results)

    samples = mcmc_results[:, 0].reshape(-1, 11)
    df = pd.DataFrame(samples, columns=param_labels)

    # Full Corner Plot
    c = ChainConsumer() 
    c.add_chain(Chain(samples=df, name="MCMC Results"))
    c.add_truth(Truth(location = dict(zip(param_labels[:11], injection_parameters[:11]))))
    fig_corner_plot = c.plotter.plot()
    fig_corner_plot.savefig(f"{folder_name}/mcmc_corner_plot.png", dpi=300)

    # Corner Plot for lam, beta, tc
    new_df = df[[param_labels[7], param_labels[8], param_labels[10]]]
    new_injection_parameters = injection_parameters[[7, 8, 10]]

    new_c = ChainConsumer() 
    new_c.add_chain(Chain(samples=new_df, name="MCMC Results"))
    new_c.add_truth(Truth(location={param_labels[7]: new_injection_parameters[0],
                                    param_labels[8]: new_injection_parameters[1],
                                    param_labels[10]: new_injection_parameters[2]}))
    fig_corner_plot_lam_beta_tc = new_c.plotter.plot()
    fig_corner_plot_lam_beta_tc.savefig(f"{folder_name}/mcmc_corner_plot_lam_beta_tc.png", dpi=300)

    
    # Trace plots
    plotting_walkers_range = nwalkers
    colors = cm.viridis(np.linspace(0, 1, plotting_walkers_range))

    fig_walkers, ax_walkers = plt.subplots(ndims, 1, sharex=True, figsize=(10, 2.5 * ndims))
    fig_walkers.subplots_adjust(hspace=0.3)

    for i in range(ndims):
        ax_walkers[i].axhline(injection_parameters[i], color='red', linestyle='--', linewidth=1.2, label="True value")
        for walk in range(plotting_walkers_range):
            ax_walkers[i].plot(mcmc_results[:, 0, walk, :, i].flatten(), color=colors[walk], alpha=0.6, linewidth=0.8)
        #ax[i].axhline(starting_points[0, walk, :, i], color='blue', linestyle='--', linewidth=1.2, label="Starting point of the last walker")

        ax_walkers[i].set_ylabel(param_labels[i], fontsize=12)
        #ax[i].legend(loc='upper right', fontsize=10)

    ax_walkers[-1].set_xlabel("Number of steps", fontsize=12)

    for axis in ax_walkers:
        axis.set_xlim(0, nsteps-1)
    fig_walkers.savefig(f"{folder_name}/mcmc_trace_plots.png", dpi=300)

    log_like_samples = sampler.get_log_like() 

    fig_log_like, ax_log_like = plt.subplots(figsize=(10, 6))
    for i in range(nwalkers):
        ax_log_like.plot(np.arange(0, log_like_samples.shape[0]), log_like_samples[:,0,i], label=f"Walker {i+1}", color=colors[i], alpha=0.6, linewidth=0.8)

    ax_log_like.axhline(analysis.calculate_time_frequency_likelihood(*parameters, waveform_kwargs=waveform_kwargs)  , 
                color='red', linestyle='--', linewidth=1.2, label="Likelihood of Injection Parameters")

    ax_log_like.set_title("Log-likelihood samples")
    ax_log_like.set_xlabel("Number of steps")
    ax_log_like.set_ylabel("Log-likelihood")
    ax_log_like.set_xlim(0, nsteps-1)
    fig_log_like.savefig(f"{folder_name}/mcmc_log_likelihood.png", dpi=300)

    end_time_total = time.time()
    print(f"Finished plotting in {end_time_total - start_time:.2f} seconds")

if __name__ == "__main__":
    main()