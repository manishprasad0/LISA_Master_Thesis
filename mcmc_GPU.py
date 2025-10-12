import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing as mp
import scipy as sp
import time
from datetime import datetime

# LISA modules
from lisatools.utils.constants import *
from lisatools.sensitivity  import AE1SensitivityMatrix
from bbhx.waveformbuild import BBHWaveformFD

# Immports for generating simulated LISA data
import noise_generation as noise_generation
from tools.LISASimulator_GPU import LISASimulator
from tools.likelihood_GPU import get_dh, get_hh, TimeFreqSNR
import tools.likelihood_GPU as likelihood

# Imports for MCMC
from tools.save_and_load_DE import load_de_results
from tools.time_freq_likelihood_GPU import TimeFreqLikelihood
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from scipy.stats import truncnorm

# Imports for plotting
from chainconsumer import Chain, ChainConsumer, make_sample, Truth
import pandas as pd


def main():
    # Tobs = 1.2*(YRSID_SI/12)
    Tobs = 1.5*(YRSID_SI/12)
    dt = 5.
    include_T_channel = False

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), force_backend="cuda12x",)
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
    print("The SNR of the signal is", sim.SNR_optimal()[0])

    # Pre-merger settings
    # hours_before_merger = 10
    
    hours_before_merger = 14
    time_before_merger = hours_before_merger*60*60
    cutoff_time = t_ref - time_before_merger
    width_of_tref_prior = 20
    max_time = t_ref + (width_of_tref_prior - hours_before_merger)*60*60
    nperseg = 1414

    # TimeFreqLikelihood Object
    analysis = TimeFreqLikelihood(data_t=data_t, wave_gen=wave_gen, nperseg=nperseg)
    analysis.pre_merger(time_before_merger=time_before_merger, t_ref=t_ref, t_array=t_array)
    analysis.get_stft_of_data()
    print("Likelihood with the true parameters = ", analysis.calculate_time_frequency_likelihood(*parameters, waveform_kwargs=waveform_kwargs))

    # MCMC parameters
    nwalkers = 24
    ntemps = 4
    ndims = 11
    nleaves_max = 1
    nsteps = 2000

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
        r"$t_{\mathrm{c}} \, [\mathrm{s}]$"
    ]

    def likelihood(x, freqs, TimeFreqLikelihood_object):
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
        all_parameters[11] = x[10]

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
        4 : uniform_dist(1, 1000),                     # dist in Gpc
        5 : uniform_dist(0, 2*np.pi),                # phi_ref
        6 : uniform_dist(-1, 1),                     # cos(inc)
        7 : uniform_dist(0.0, 2 * np.pi),            # lam
        8 : uniform_dist(-1.0, 1.0),                 # sin(beta)
        9 : uniform_dist(0.0, np.pi),                # psi
        10: uniform_dist(cutoff_time, max_time),    # t_ref
    })}

    periodic = {"mbh": {5: 2 * np.pi,   # phi_ref
                        7: 2 * np.pi,   # lam
                        9: np.pi  }}    # psi
    
    sampler = EnsembleSampler(
        nwalkers,
        ndims,
        likelihood,
        priors,
        args=(f_array, analysis),
        branch_names=["mbh"],
        tempering_kwargs=dict(ntemps = ntemps),
        nleaves_max=dict(mbh = nleaves_max),
        periodic=periodic
    )

    injection_parameters = np.array([m1+m2, m2/m1, a1, a2, dist / (PC_SI * 1e9), phi_ref, np.cos(inc), lam, np.sin(beta), psi, t_ref])

    # Results from Differential Evolution
    def DE_to_MCMC_params(found_parameters_DE, cutoff_time):
        mT, q, a1, a2, dist_Gpc, phi_ref, cos_inc, lam, sin_beta, psi, t_ref_01 = found_parameters_DE
        mT_exp = np.exp(mT)
        dist_Mpc = dist_Gpc 
        phi_ref = phi_ref % (2*np.pi)
        time_to_coalescence = t_ref_01 * (60*60*24) + cutoff_time
        return np.array([mT_exp, q, a1, a2, dist_Mpc, phi_ref, cos_inc, lam, sin_beta, psi, time_to_coalescence])
        
        
    # x0 = np.array([13.0165, 0.501283, 0.650153, 0.871754, 10.0336, 2.27851, 0.97478,  1.04851, 0.337238, 1.95062, 0.416537])
    
    x0 = np.array([13.0442, 0.480049, 0.820059, 0.262087, 9.93484, 4.12773, 0.971511, 0.843206, 0.390985, 2.12432, 0.577044])
    found_parameters_DE = DE_to_MCMC_params(x0, cutoff_time=cutoff_time)

    starting_points = np.zeros(shape=[ntemps, nwalkers, nleaves_max, found_parameters_DE.shape[0]])

    # Perturb the injection parameters to create starting points for the walkers
    perturb_frac = 0.02

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
    print("MCMC results shape:", mcmc_results.shape)


    np.save(f"mcmc_results_GPU/original_inputs/mcmc_14hours_2000_002.npy", mcmc_results)
    print(f"MCMC results saved to mcmc_results_GPU/original_inputs/mcmc_14hours_2000_002.npy")


    log_like_samples = sampler.get_log_like()
    np.save(f"mcmc_results_GPU/original_inputs/mcmc_loglike_14hours_2000_002.npy", log_like_samples)
    print(f"Log-likelihood samples saved to mcmc_results_GPU/original_inputs/mcmc_loglike_14hours_2000_002.npy")


if __name__ == "__main__":
    main()