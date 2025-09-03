import numpy as np
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'  # For sharper figures, but it takes more time
import scipy as sp
from copy import deepcopy 

from lisatools.utils.constants import *
from lisatools.sensitivity  import AE1SensitivityMatrix
from bbhx.waveformbuild import BBHWaveformFD

# Imports for generating simulated LISA data
from tools.LISASimulator import LISASimulator
from tools.likelihood import get_dh, get_hh
import tools.likelihood as likelihood

# Imports for MCMC
from tools.time_freq_likelihood import TimeFreqLikelihood
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State

# Imports for plotting
from chainconsumer import Chain, ChainConsumer, make_sample, Truth
import pandas as pd



def main():

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
    beta = 0.292
    psi = 0
    t_ref = 0.95 * Tobs
    parameters = np.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref])
    modes = [(2,2)]#, (2,1), (3,3), (3,2), (4,4), (4,3)]
    waveform_kwargs = dict(length=1024, direct=False, fill=True, squeeze=False, modes=modes)

    data_t, data_f, f_array, t_array, sens_mat = sim(seed = 42, parameters=parameters, waveform_kwargs=waveform_kwargs)
    waveform_kwargs.update(freqs=f_array)

    analysis = TimeFreqLikelihood(data_t=data_t, wave_gen=wave_gen, nperseg=nperseg)
    analysis.pre_merger(time_before_merger=time_before_merger, t_ref=t_ref, t_array=t_array)
    analysis.get_stft_of_data()

    # Pre-merger settings
    hours_before_merger = 20
    time_before_merger = hours_before_merger*60*60
    cutoff_time = t_ref - time_before_merger
    max_time = t_ref + (24 - hours_before_merger)*60*60

    nperseg = 5000

    # MCMC parameters
    nwalkers = 24
    ntemps = 4
    ndims = 11
    nleaves_max = 1
    nsteps = 100
    perturb_frac = 0.02

    param_labels = [
        r"$M_T$",
        r"$q$",
        r"$a_1$",
        r"$a_2$",
        r"$d_L \, [\mathrm{Mpc}]$",
        r"$\phi_{\mathrm{ref}}$",
        r"$\cos(\iota)$",
        r"$\lambda$",
        r"$\sin(\beta)$",
        r"$\psi$",
        r"$t_{\mathrm{ref}}$"
    ]

    def likelihood(x, freqs, TimeFreqLikelihood_object, modes):
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
        1 : uniform_dist(0.05, 0.99),            # q = m2/m1
        2 : uniform_dist(-1, +1),  # a1
        3 : uniform_dist(-1, +1),  # a2
        4 : uniform_dist(1e3, 50e3),                 # dist in Mpc
        5 : uniform_dist(0, 2*np.pi),            # phi_ref
        6 : uniform_dist(-1, 1),                     # cos(inc)
        7 : uniform_dist(0.0, 2 * np.pi),            # lam
        8 : uniform_dist(-1.0, 1.0),                 # sin(beta)
        9 : uniform_dist(0.0, np.pi),                # psi
        10: uniform_dist(cutoff_time, max_time),     # t_ref
    })}

    periodic = {"mbh": {5: 2 * np.pi,   # phi_ref
                        7: 2 * np.pi,   # lam
                        9: np.pi  }}    # psi

    analysis = TimeFreqLikelihood(data_t=data_t, wave_gen=wave_gen, nperseg=nperseg)
    analysis.pre_merger(time_before_merger=time_before_merger, t_ref=t_ref, t_array=t_array)

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

    from scipy.stats import truncnorm
    from tools.MBHB_differential_evolution import transform_parameters_to_bbhx
    
    found_parameters_DE = np.array([np.exp(12.9675), 0.684924, 0.524065, 0.999998, (10.1203)*1000, -1.82545 % (2*np.pi), 0.345506, 3.51093, 0.292316, 1.57164, 72000+cutoff_time])
    found_SNR = 0.06530506

    injection_parameters = np.array([m1+m2, m2/m1, a1, a2, dist / (PC_SI * 1e6), phi_ref, np.cos(inc), lam, np.sin(beta), psi, t_ref])

    #injection_parameters = np.array([4.5e5, 1.5e5, 0.2, 0.4, 8e3, 2*np.pi-0.4, 0.0, np.pi/3, np.pi/1., np.pi/4., np.pi/4., 0.95 * Tobs])

    starting_points = np.zeros(shape=[ntemps, nwalkers, nleaves_max, found_parameters_DE.shape[0]])

    # Perturb the injection parameters to create starting points for the walkers
    #perturb_frac = 0.05

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

    # For t_ref (index 11): uniform draw
    #starting_points[:, :, :, 11] = np.random.uniform(low=priors["mbh"].priors[11][1].min_val, high=priors["mbh"].priors[11][1].max_val, size=(ntemps, nwalkers, nleaves_max))

    starting_state = State({"mbh": starting_points})

    sampler.run_mcmc(starting_state, nsteps=nsteps, progress=True)

    from chainconsumer import Chain, ChainConsumer, make_sample, Truth
    import pandas as pd

    mcmc_results = sampler.get_chain()["mbh"]

    samples = mcmc_results[:, 0].reshape(-1, 11)
    df = pd.DataFrame(samples, columns=param_labels)

    #df.drop(columns=[r"$f_{\mathrm{ref}}$"], inplace=True)  # Remove f_ref column if not needed
    # df.to_pickle("mcmc_samples.pkl")

    #burn_in_2 = 5000                           # Remove burn-in samples if needed
    #df_post_burnin = df.iloc[burn_in_2:]

    injection_parameters = np.array([m1+m2, m2/m1, a1, a2, dist / (PC_SI * 1e6), phi_ref, np.cos(inc), lam, np.sin(beta), psi, t_ref])
    c = ChainConsumer() 
    c.add_chain(Chain(samples=df, name="MCMC Results"))
    c.add_truth(Truth(location={r"$M_T$"                      : injection_parameters[0],
                                r"$q$"                        : injection_parameters[1], 
                                r"$a_1$"                      : injection_parameters[2],
                                r"$a_2$"                      : injection_parameters[3],
                                r"$d_L \, [\mathrm{Mpc}]$"    : injection_parameters[4],
                                r"$\phi_{\mathrm{ref}}$"      : injection_parameters[5],
                                #r"$f_{\mathrm{ref}}$"         : injection_parameters[6],
                                r"$\cos(\iota)$"              : injection_parameters[6],
                                r"$\lambda$"                  : injection_parameters[7],
                                r"$\sin(\beta)$"              : injection_parameters[8],
                                r"$\psi$"                     : injection_parameters[9],
                                r"$t_{\mathrm{ref}}$"         : injection_parameters[10]} ))
    fig = c.plotter.plot()


    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    plotting_walkers_range = nwalkers
    colors = cm.viridis(np.linspace(0, 1, plotting_walkers_range))

    fig, ax = plt.subplots(ndims, 1, sharex=True, figsize=(10, 2.5 * ndims))
    fig.subplots_adjust(hspace=0.3)


    for i in range(ndims):
        ax[i].axhline(injection_parameters[i], color='red', linestyle='--', linewidth=1.2, label="True value")
        for walk in range(plotting_walkers_range):
            ax[i].plot(mcmc_results[:, 0, walk, :, i].flatten(), color=colors[walk], alpha=0.6, linewidth=0.8)
        #ax[i].axhline(starting_points[0, walk, :, i], color='blue', linestyle='--', linewidth=1.2, label="Starting point of the last walker")

        ax[i].set_ylabel(param_labels[i], fontsize=12)
        #ax[i].legend(loc='upper right', fontsize=10)

    ax[-1].set_xlabel("Step", fontsize=12)



if __name__ == "__main__":
    main()