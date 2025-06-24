# import torch
from bbhx.utils.constants import *
from bbhx.utils.transform import *
import numpy as np
import scipy as sp
import xarray as xr
import time
import os
import pickle
from ldc.common.tools import window
from ldc.lisa.noise import get_noise_model
from ldc.common.series import TimeSeries, FrequencySeries, TDI
from lisatools_conversion import ConvertSSBframeParamsToLframe, ConvertLframeParamsToSSBframe
from copy import deepcopy
import matplotlib.pyplot as plt


import numpy as cp
####get noise
def get_noise_from_frequency_domain(tdi_fs, nperseg=15000):
    tdi_ts = xr.Dataset(dict([(k, tdi_fs[k].ts.ifft()) for k in ["A", "E", "T"]]))
    dt = tdi_ts.t.values[1] - tdi_ts.t.values[0]
    f, psdA =  sp.signal.welch(tdi_ts["A"], fs=1.0/dt, nperseg=nperseg)#, average='mean', window= 'boxcar')
    f, psdE =  sp.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=nperseg)
    # f2, psdE2 =  sp.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=len(tdi_ts["A"]), scaling='spectrum')
    f, psdT =  sp.signal.welch(tdi_ts["T"], fs=1.0/dt, nperseg=nperseg)
    return f, psdA, psdE, psdT


def transform_to_parameters(params01, boundaries, parameters_sample):
    one_dim = False
    if params01.ndim == 1:
        one_dim = True
        params01 = cp.array([params01])
    params = cp.zeros_like(params01)
    for i, parameter in enumerate(parameters_sample):
        if parameter in ['EclipticLatitude']:
            params[:,i] = cp.arcsin(params01[:,i] * (boundaries[parameter][1] - boundaries[parameter][0]) + boundaries[parameter][0])
        elif parameter in ['Inclination']:
            params[:,i] = cp.arccos(params01[:,i] * (boundaries[parameter][1] - boundaries[parameter][0]) + boundaries[parameter][0])
        elif parameter in ['TotalMass']:
            params[:,i] = 10**(params01[:,i] * (boundaries[parameter][1] - boundaries[parameter][0]) + boundaries[parameter][0])
        else:
            params[:,i] = params01[:,i] * (boundaries[parameter][1] - boundaries[parameter][0]) + boundaries[parameter][0]
    m1 = params[:,0] * params[:,1] / (1 + params[:,1])
    m2 = params[:,0] / (1 + params[:,1])
    params[:,0] = m1
    params[:,1] = m2
    if one_dim:
        params = params[0]
    return params
    

def transform_to_01(params, boundaries, parameters_sample):
    one_dim = False
    if params.ndim == 1:
        one_dim = True
        params = cp.array([params])
    params01 = cp.zeros_like(params)
    M = params[:,0] + params[:,1]
    q = params[:,0] / params[:,1]
    for i, parameter in enumerate(parameters_sample):
        if parameter in ['EclipticLatitude']:
            params01[:,i] = (cp.sin(params[:,i]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in ['Inclination']:
            params01[:,i] = (cp.cos(params[:,i]) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in ['TotalMass']:
            params01[:,i] = (cp.log10(M) - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        elif parameter in ['MassRatio']:
            params01[:,i] = (q - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
        else:
            params01[:,i] = (params[:,i] - boundaries[parameter][0]) / (boundaries[parameter][1] - boundaries[parameter][0])
    if one_dim:
        params01 = params01[0]
    return params01

def median_windows(y, window_size):
    medians = deepcopy(y)
    for i in range(int(len(y)/window_size*2)-1):
        start_index = int(i/2*window_size)
        end_index = int((i/2+1)*window_size)
        median = np.median(y[start_index:end_index])
        outliers = np.abs(medians[start_index:end_index]) > median*2
        medians[start_index:end_index][outliers] = median
    return medians

def smooth_psd(psd, f):
    smoothed = median_windows(psd, 30)
    smoothed[:40] = psd[:40]
    index_cut = np.searchsorted(f, 0.0008)  # 0.0008 for 1,2 years
    index_cut_lower = np.searchsorted(f, 3*10**-4)
    psd_fit = np.ones_like(smoothed)
    psd_fit_low = sp.signal.savgol_filter(smoothed, 10, 1)
    psd_fit_high = sp.signal.savgol_filter(smoothed, 70, 1) # 70 for 1,2 years
    psd_fit[:index_cut] = psd_fit_low[:index_cut] 
    psd_fit[index_cut:] = psd_fit_high[index_cut:] 
    psd_fit[:index_cut_lower] = smoothed[:index_cut_lower]
    return psd_fit, smoothed


def get_psd(tdi_fs, freq_new, dataset='Sangria'):
    nperseg = 15000
    frequencies, psdA, psdE, psdT = get_noise_from_frequency_domain(tdi_fs, nperseg=nperseg)

    psdA, smoothedA = smooth_psd(psdA, frequencies)
    psdE, smoothedE = smooth_psd(psdE, frequencies)
    psdT, smoothedT = smooth_psd(psdT, frequencies)

    psdA = sp.interpolate.interp1d(frequencies, psdA)(freq_new)
    psdE = sp.interpolate.interp1d(frequencies, psdE)(freq_new)
    psdT = sp.interpolate.interp1d(frequencies, psdT)(freq_new)

    if dataset == 'Radler':
        noise_model = "SciRDv1"
        Nmodel = get_noise_model(noise_model, freq_new)
        psdA = Nmodel.psd(freq=freq_new, option="A")
        psdE = Nmodel.psd(freq=freq_new, option="E")
        psdT = Nmodel.psd(freq=freq_new, option="T")

    # Nmodel = get_noise_model(noise_model, freq_new)
    # psdA = Nmodel.psd(freq=freq_new, option="A")
    # psdE = Nmodel.psd(freq=freq_new, option="E")
    # psdT = Nmodel.psd(freq=freq_new, option="T")
    if freq_new[0] == 0:
        psdA[0] = 1
        psdE[0] = 1
        psdT[0] = 1
    return psdA, psdE, psdT


class MBHB_finder():
    def __init__(self, tdi_fs, len_time, len_extended_time, freq_new, freq_extended_all, index_lower_cut, index_upper_cut, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample, noise=None, merger_outside=False, dataset='Sangria'):
        self.tdi_fs = tdi_fs
        self.len_time = len_time
        self.len_extended_time = len_extended_time
        self.freq_new = freq_new
        self.freq_new_cp = cp.array(freq_new)
        self.freq_extended_all = freq_extended_all
        self.index_lower_cut = index_lower_cut
        self.index_upper_cut = index_upper_cut
        self.boundaries = boundaries
        self.data_channels = data_channels
        self.waveform_kwargs = waveform_kwargs
        self.wave_gen = wave_gen
        self.merger_outside = merger_outside
        if noise is None:
            self.psdA, self.psdE, self.psdT = get_psd(tdi_fs, freq_new, dataset)
        else:
            self.psdA = noise[0]
            self.psdE = noise[1]
            self.psdT = noise[2]
        self.df = float(tdi_fs['f'][1]-tdi_fs['f'][0])
        self.parameters_sample = parameters_sample

    def get_wave(self, parameters, include_T=False):
        if self.merger_outside:
            wave = self.wave_gen(*parameters.T, freqs=self.freq_extended_all,**self.waveform_kwargs)[0]
            start_timer = time.time()
            A_in_extended_all = cp.fft.irfft(wave[0])
            E_in_extended_all = cp.fft.irfft(wave[1])
            print('time extended', np.round(time.time() - start_timer,2))
            start_timer = time.time()
            A_in_extended_all = A_in_extended_all[:self.len_time]
            E_in_extended_all = E_in_extended_all[:self.len_time]
            print('time extended', np.round(time.time() - start_timer,2))
            start_timer = time.time()
            A_fs_extended_all = cp.fft.rfft(A_in_extended_all)[self.index_lower_cut:self.index_upper_cut]
            E_fs_extended_all = cp.fft.rfft(E_in_extended_all)[self.index_lower_cut:self.index_upper_cut]
            print('time extended fft', np.round(time.time() - start_timer,2))
            if include_T:
                T_in_extended_all = cp.fft.irfft(wave[2], n=self.len_extended_time)[:self.len_time]
                T_fs_extended_all = cp.fft.rfft(T_in_extended_all)[self.index_lower_cut:self.index_upper_cut]
                wave = np.asarray([[A_fs_extended_all, E_fs_extended_all, T_fs_extended_all]])
            else:
                wave = np.asarray([[A_fs_extended_all, E_fs_extended_all]])
            # wave = self.wave_gen(*parameters.T, freqs=self.freq_all,**self.waveform_kwargs)[0]
            # A_in_all = cp.fft.irfft(wave[0], n=len(self.tdi_ts_segment['A']))
            # E_in_all = cp.fft.irfft(wave[1], n=len(self.tdi_ts_segment['E']))
            # A_fs_all = cp.fft.rfft(A_in_all)[self.index_lower_cut:self.index_upper_cut]
            # E_fs_all = cp.fft.rfft(E_in_all)[self.index_lower_cut:self.index_upper_cut]
        else:
            wave = self.wave_gen(*parameters.T, freqs=self.freq_new,**self.waveform_kwargs)
        return wave

    def get_dh_hh(self, parameters):
        wave = self.get_wave(parameters.T)
        # # add plot of wave
        # plt.figure()
        # plt.plot(self.freq_new, self.data_channels[0])
        # plt.plot(self.freq_new, wave[0][0])
        # # plt.plot(self.freq_new, wave2[0][0])
        # plt.show()


        hh = np.sum((np.absolute(wave[:,0])**2 + np.absolute(wave[:,1])**2)/self.psdA, axis=1)
        dh = np.sum( np.real(self.data_channels[0] * np.conjugate(wave[:,0]) + self.data_channels[1] * np.conjugate(wave[:,1]))/self.psdA , axis=1)
        # hh = np.sum((np.absolute(wave[:,0])**2 + np.absolute(wave[:,1])**2)/psdA + np.absolute(wave[:,2])**2 /psdT, axis=1)
        # dh = np.sum( np.real(data_channels[0] * np.conjugate(wave[:,0]) + data_channels[1] * np.conjugate(wave[:,1]))/psdA + np.real(data_channels[2] * np.conjugate(wave[:,2]))/psdT, axis=1)
        hh = 4.0*self.df* hh
        dh = 4.0*self.df* dh
        return dh, hh

    def loglikelihood_ratio(self, parameters):
        dh, hh = self.get_dh_hh(parameters)
        return -hh/2 + dh

    def SNR(self, parameters):
        dh, hh = self.get_dh_hh(parameters)
        return dh/np.sqrt(hh)



    def get_ll_from01(self, params01):
        params = transform_to_parameters(params01, self.boundaries, self.parameters_sample)
        # out = like.get_ll(params.T, **waveform_kwargs)[:,0]
        out = self.loglikelihood_ratio(params)
        return -out

    def get_SNR_from01(self, params01):
        params = transform_to_parameters(params01, self.boundaries, self.parameters_sample)
        # out = like.get_ll(params.T, return_extracted_snr=True, **waveform_kwargs)[:,1]
        out = self.SNR(params)
        # out = loglikelihood_lisa(params)
        return -out

        
    def get_SNR_from01_Lframe_without_distance(self, params01):
        params01_with_distance = cp.zeros(len(params01)+1)
        params01_with_distance[:4] = params01[:4]
        params01_with_distance[4] = 0.5                 # FIXING THE DISTANCE TO 0.5 Gpc FOR DIFFERENTIAL EVOLUTION.
        params01_with_distance[5:] = params01[4:]
        params = transform_to_parameters(params01_with_distance, self.boundaries, self.parameters_sample)

        params[10],  params[7], params[8], params[9] = ConvertLframeParamsToSSBframe(params[10], params[7], params[8], params[9], 0.0)

        # out = like.get_ll(params.T, return_extracted_snr=True, **waveform_kwargs)[:,1]
        out = self.SNR(params)
        # out = loglikelihood_lisa([params])
        return -out
    
    def get_SNR_from01_without_distance(self, params01):
        params01_with_distance = np.zeros(len(params01)+1)
        params01_with_distance[:4] = params01[:4]
        params01_with_distance[4] = 0.5
        params01_with_distance[5:] = params01[4:]
        params = transform_to_parameters(params01_with_distance, self.boundaries, self.parameters_sample)
        # out = like.get_ll(params.T, return_extracted_snr=True, **waveform_kwargs)[:,1]
        out = self.SNR(params)
        # out = loglikelihood_lisa([params])
        return -out

    def calculate_Amplitude(self, pGBs):
        dh, hh = self.get_dh_hh(pGBs)
        A = dh / hh
        return A


    def search_MBHB(self, tdi_fs, freq_new, boundaries):
        initial_guess01 = cp.array([cp.random.rand(len(boundaries))])
        initial_guess = transform_to_parameters(initial_guess01, boundaries, self.parameters_sample)
        initial_guess01 = transform_to_01(initial_guess, boundaries, self.parameters_sample)
        # print(initial_guess,  self.SNR(initial_guess),  self.loglikelihood_ratio(initial_guess),'initial guess')
        found_parameters01 = initial_guess01[0]
        bounds = []
        for i in range(len(boundaries)):
            bounds.append((0,1))
        initial_guess01_without_distance = cp.append(initial_guess01[:,:4],initial_guess01[:,5:])

        time_start = time.time()
        SNR = self.get_SNR_from01_Lframe_without_distance(initial_guess01_without_distance)
        print('time SNR ',np.round(time.time() - time_start,2))
        print('initial guess', SNR)
        time_start = time.time()
        # result01 = sp.optimize.differential_evolution(get_SNR_from01, bounds=bounds, disp=True, strategy='best1exp', popsize=11,tol= 1e-3 , maxiter=2000, recombination= 0.75, mutation=(0.5,1), x0=initial_guess01)
        result01 = sp.optimize.differential_evolution(self.get_SNR_from01_Lframe_without_distance, bounds=bounds[:-1], disp=True, strategy='best1exp', popsize=10,tol= 1e-8, maxiter=500, recombination= 1, mutation=(0.5,1), x0=initial_guess01_without_distance, polish=True)
        print('time DE ',np.round(time.time() - time_start,2))
        print(result01)
        function_evaluations = result01.nfev
        found_parameters01 = result01.x
        found_params01_with_distance = np.zeros(len(found_parameters01)+1)
        found_params01_with_distance[:4] = found_parameters01[:4]
        found_params01_with_distance[4] = 0.5
        found_params01_with_distance[5:] = found_parameters01[4:]
        found_parameters01 = found_params01_with_distance
        found_parameters = transform_to_parameters(found_parameters01, boundaries, self.parameters_sample)
        found_parameters[10],  found_parameters[7], found_parameters[8], found_parameters[9] = ConvertLframeParamsToSSBframe(found_parameters[10], found_parameters[7], found_parameters[8], found_parameters[9], 0.0)
        amplitude_factor = self.calculate_Amplitude(found_parameters)
        found_parameters[4] /= amplitude_factor
        print(found_parameters, self.SNR(found_parameters), self.loglikelihood_ratio(found_parameters), found_parameters[4]/PC_SI/1e9,'found')
        # print(params_in, self.SNR(params_in), self.loglikelihood_ratio(params_in))
        return found_parameters, self.SNR(found_parameters), function_evaluations

def find_MBHBs_in_segment(tdi_ts, time_segment_index, Tobs, wave_gen, freq_full, parameters_sample, padding_future=np.round(1*24*3600/10)*10*1, padding_past=np.round(1*24*3600/10)*10*7, modes=[(2,2)],
                           noise_estimate=None, number_of_MBHBs_per_segment=5, dataset='Sangria', saving_directory='', params_in=None):
    tdi_ts_input = deepcopy(tdi_ts)
    Tobs_data = tdi_ts['t'][-1]
    dt = tdi_ts['t'][1]-tdi_ts['t'][0]
    # padding = 0
    Tobs_padded = Tobs + padding_future + padding_past
    t_shift = time_segment_index*(Tobs)-padding_past
    # t_shift = t_ref - 3600 - Tobs_padded
    if t_shift + Tobs_padded > Tobs_data:
        Tobs_padded -= padding_future
    # freq_new = cp.fft.rfftfreq(target_length, d=dt)
    # index_upper_cut = int(np.searchsorted(freq_new, 0.02))
    # index_lower_cut = 0
    # freq_new = freq_new[index_lower_cut:index_upper_cut]

    if time_segment_index == 0:
        t_shift = 0
        Tobs_padded -= padding_past
    t_shift_index = int(np.round(t_shift/dt))
    ts = cp.arange(0, Tobs_padded, dt)
    target_length = len(ts)
    print(t_shift,t_shift_index, 't shift and index', target_length)
    waveform_kwargs = dict(initial_t_val= t_shift, modes=modes, direct=False, fill=True, squeeze=True, length=1024,  shift_t_limits=False)

    boundaries = {}
    boundaries['Spin1'] = [-1, 1]
    boundaries['Spin2'] = [-1, 1]
    boundaries['Distance'] = [500*PC_SI*1e6, 1e6*PC_SI*1e6]
    boundaries['Phase'] = [-cp.pi, cp.pi]
    boundaries['Inclination'] = [-1, 1]
    boundaries['EclipticLongitude'] = [0, 2*cp.pi]
    boundaries['EclipticLatitude'] = [-1, 1]
    boundaries['Polarization'] = [0, cp.pi]
    # boundaries['CoalescenceTime'] = [t_ref-500, t_ref+500]
    boundaries['CoalescenceTime'] = [t_shift+padding_past, t_shift+Tobs_padded-padding_future]  
    # boundaries['CoalescenceTime'] = [t_shift+padding+Tobs, t_shift+Tobs_padded-padding+Tobs/2]  
    # boundaries['TotalMass'] = [6, 7]
    boundaries['TotalMass'] = [5, 8]
    boundaries['MassRatio'] = [1, 10]

    tdi_ts_segment = TDI(dict([(k,TimeSeries(tdi_ts[k][t_shift_index:t_shift_index+target_length], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
    tdi_fs = TDI(dict([(k,tdi_ts_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
    index_upper_cut = int(np.searchsorted(tdi_fs.f, 0.02))
    if Tobs != Tobs_data:
        index_lower_cut = int(np.searchsorted(tdi_fs.f, 0.00015))
        # index_lower_cut = 1

    data_A_fd = tdi_fs['A'].values[index_lower_cut:index_upper_cut]
    data_E_fd = tdi_fs['E'].values[index_lower_cut:index_upper_cut]
    data_T_fd = tdi_fs['T'].values[index_lower_cut:index_upper_cut]
    data_channels = cp.array([data_A_fd, data_E_fd, data_T_fd])
    freq_new = np.array(tdi_fs.f[index_lower_cut:index_upper_cut])
    # wave = wave_gen(*params_in, freqs=freq_new, initial_t_val= t_shift, modes=modes, direct=False, fill=True, squeeze=True, length=1024,  shift_t_limits=False)[0]
    # data_A_fd_sub = tdi_fs['A'].values[index_lower_cut:index_upper_cut] - wave[0]


    MBHB_detective = MBHB_finder(tdi_fs, freq_new, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample)
    if params_in is not None: 
        print('SNR injected', float(MBHB_detective.SNR(params_in)))

    if dataset == 'Radler':
        noise_estimate_interpolated = [MBHB_detective.psdA, MBHB_detective.psdE, MBHB_detective.psdT]
    if dataset == 'Sangria':
        # tdi_ts_first_segment = TDI(dict([(k,TimeSeries(tdi_ts_input[k][:len(ts)], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
        # tdi_fs_first = TDI(dict([(k,tdi_ts_first_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
        # noise_estimate_interpolated = get_psd(tdi_fs_first, freq_new)
        noise_estimate_interpolated = []
        for i in range(3):
            noise_estimate_interpolated.append(sp.interpolate.interp1d(noise_estimate[3], noise_estimate[i])(freq_new))

    # plt.figure()
    # plt.plot(tdi_ts_segment['t'],tdi_ts_segment['A'])
    # # plt.plot(injected_signal_bbhx_tdi_td['t'],injected_signal_bbhx_tdi_td['A'])
    # # plt.plot(found_signal_bbhx_tdi_td['t'],found_signal_bbhx_tdi_td['A'])
    # plt.show()
    
    # plt.figure()
    # plt.semilogx(freq_new,(data_A_fd))
    # # plt.semilogx(freq_new,(data_A_fd_sub))
    # # plt.semilogx(freq_new,(wave[0]))
    # # plt.semilogx(freq_new,(data_A_fd_sub_2))
    # # plt.semilogx(freq_new,(injected_signal_bbhx_tdi_fd['A']))
    # # plt.semilogx(freq_new,(found_signal_bbhx_tdi_fd['A']))
    # plt.show()

    # plt.figure()
    # # plt.loglog(freq_new,noise_estimate_current[0])
    # plt.loglog(freq_new,noise_estimate_interpolated[0])
    # # plt.loglog(freq_new,noise_estimate_interpolated_current2[0])
    # plt.show()

    MBHB_detective = MBHB_finder(tdi_fs, freq_new, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample, noise_estimate_interpolated)
    if params_in is not None: 
        print('SNR injected', float(MBHB_detective.SNR(params_in)))
    print(MBHB_detective.data_channels[1][:10])

    found_parameters_list, number_of_function_evaluations = find_MBHB(tdi_fs, data_channels, freq_new, boundaries, t_shift, wave_gen, waveform_kwargs, parameters_sample, modes=modes,
                                                                       number_of_searches=5, noise_estimate=noise_estimate_interpolated, dump=True,
                                                                         number_of_MBHBs_per_segment=number_of_MBHBs_per_segment, saving_directory=saving_directory)
    for found_parameters in found_parameters_list:
        wave = wave_gen(*found_parameters, freqs=freq_full, initial_t_val= 0, modes=modes, direct=False, fill=True, squeeze=True, length=1024,  shift_t_limits=False)[0]
        found_signal_bbhx_tdi_fd = TDI(dict([(k,FrequencySeries(wave[i], fs=freq_full)) for i,k in enumerate(["A", "E", "T"])]))
        found_signal_bbhx_tdi_td = TDI(dict([(k,found_signal_bbhx_tdi_fd[k].ts.ifft()) for k in ["A", "E", "T"]]))
        for i, k in enumerate(["A", "E", "T"]):
            tdi_ts[k] -= found_signal_bbhx_tdi_td[k]

    # if found_parameters_list != []:
    #     tdi_ts_segment = TDI(dict([(k,TimeSeries(tdi_ts[k][t_shift_index:t_shift_index+target_length], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
    #     tdi_fs = TDI(dict([(k,tdi_ts_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
    #     noise_estimate = get_psd(tdi_fs, freq_new)

    #     # plt.figure()
    #     # plt.plot(tdi_ts_segment['t'],tdi_ts_segment['A'])
    #     # plt.plot(tdi_ts['t'],tdi_ts['A'])
    #     # # plt.plot(tdi_ts_input['t'],tdi_ts_input['A'])
    #     # plt.show()

    #     tdi_ts = deepcopy(tdi_ts_input)
    #     tdi_ts_segment = TDI(dict([(k,TimeSeries(tdi_ts[k][t_shift_index:t_shift_index+target_length], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
    #     tdi_fs = TDI(dict([(k,tdi_ts_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
    #     index_upper_cut = int(np.searchsorted(tdi_fs.f, 0.02))
    #     if Tobs != Tobs_data:
    #         index_lower_cut = int(np.searchsorted(tdi_fs.f, 0.00015))
    #     data_A_fd = tdi_fs['A'].values[index_lower_cut:index_upper_cut]
    #     data_E_fd = tdi_fs['E'].values[index_lower_cut:index_upper_cut]
    #     data_T_fd = tdi_fs['T'].values[index_lower_cut:index_upper_cut]
    #     data_channels = cp.array([data_A_fd, data_E_fd, data_T_fd])

    #     found_parameters_list, number_of_function_evaluations = find_MBHB(tdi_fs, data_channels, freq_new, boundaries, t_shift, waveform_kwargs, number_of_searches=5, noise_estimate=noise_estimate, dump=True)
    #     for found_parameters in found_parameters_list:
    #         wave = wave_gen(*found_parameters, freqs=freq_full, initial_t_val= 0, modes=modes, direct=False, fill=True, squeeze=True, length=1024,  shift_t_limits=False)[0]
    #         found_signal_bbhx_tdi_fd = TDI(dict([(k,FrequencySeries(wave[i], fs=freq_full)) for i,k in enumerate(["A", "E", "T"])]))
    #         found_signal_bbhx_tdi_td = TDI(dict([(k,found_signal_bbhx_tdi_fd[k].ts.ifft()) for k in ["A", "E", "T"]]))
    #         for i, k in enumerate(["A", "E", "T"]):
    #             tdi_ts[k] -= found_signal_bbhx_tdi_td[k]

    return tdi_ts, found_parameters_list, number_of_function_evaluations


def find_MBHBs_in_range(tdi_ts, wave_gen, freq_full, parameters_sample, start_time, end_time, merger_time_boundary, mass_boundary=None, mass_ratio_boundary=None, modes=[(2,2)],
                           noise_estimate=None, number_of_MBHBs_per_segment=5, dataset='Sangria', saving_directory='', params_in=None):
    tdi_ts_input = deepcopy(tdi_ts)
    Tobs_data = tdi_ts['t'][-1]
    dt = float(tdi_ts['t'][1]-tdi_ts['t'][0])
    # end_time -= 10000
    start_time_index = int(np.round(start_time/dt))
    end_time_index = int(np.round(end_time/dt))
    Tobs = end_time-start_time



    waveform_kwargs = dict(initial_t_val= start_time, modes=modes, direct=False, fill=True, squeeze=False, length=1024,  shift_t_limits=False)

    boundaries = {}
    boundaries['Spin1'] = [-1, 1]
    boundaries['Spin2'] = [-1, 1]
    boundaries['Distance'] = [500*PC_SI*1e6, 1e6*PC_SI*1e6]
    boundaries['Phase'] = [-cp.pi, cp.pi]
    boundaries['Inclination'] = [-1, 1]
    boundaries['EclipticLongitude'] = [0, 2*cp.pi]
    boundaries['EclipticLatitude'] = [-1, 1]
    boundaries['Polarization'] = [0, cp.pi]
    # boundaries['CoalescenceTime'] = [t_ref-500, t_ref+500]
    boundaries['CoalescenceTime'] = [start_time, end_time] 
    if merger_time_boundary is not None: 
        boundaries['CoalescenceTime'] = merger_time_boundary 
    boundaries['TotalMass'] = [6, 7]
    if mass_boundary is not None:
        boundaries['TotalMass'] = mass_boundary
    # boundaries['TotalMass'] = [5, 8]
    boundaries['MassRatio'] = [1, 10]
    if mass_boundary is not None:
        boundaries['MassRatio'] = mass_ratio_boundary

    print(boundaries)
    tdi_ts_segment = TDI(dict([(k,TimeSeries(tdi_ts[k][start_time_index:end_time_index], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
    len_time = len(tdi_ts_segment['A'])
    tdi_fs = TDI(dict([(k,tdi_ts_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
    index_upper_cut = int(np.searchsorted(tdi_fs.f, 0.02))
    if Tobs != Tobs_data:
        index_lower_cut = int(np.searchsorted(tdi_fs.f, 0.00015))
        # index_lower_cut = 1
    # index_lower_cut = 0
    # index_upper_cut = -1

    data_A_fd = tdi_fs['A'].values[index_lower_cut:index_upper_cut]
    data_E_fd = tdi_fs['E'].values[index_lower_cut:index_upper_cut]
    data_T_fd = tdi_fs['T'].values[index_lower_cut:index_upper_cut]
    data_A_fd_no_window = cp.fft.rfft(tdi_ts_segment['A'])[index_lower_cut:index_upper_cut]
    data_channels = cp.array([data_A_fd, data_E_fd, data_T_fd])
    freq_new = np.array(tdi_fs.f[index_lower_cut:index_upper_cut])
    freq_all = np.array(tdi_fs.f)
    # freq_new = np.array(tdi_fs.f)

    if end_time < merger_time_boundary[1]:
        merger_outside = True
    else:
        merger_outside = False

    wave_in = wave_gen(*params_in.T, freqs=freq_new,**waveform_kwargs)[0]
    A_in = cp.fft.irfft(wave_in[0], n=len(tdi_ts_segment['A']))
    A_fs = cp.fft.rfft(A_in)[index_lower_cut:index_upper_cut]
    len_time_extended = np.copy(len_time)
    freq_extended_all = np.array(tdi_fs.f)
    # merger_outside = False
    if merger_outside:
        end_time_extended = merger_time_boundary[1] + 100000
        # end_time_extended = end_time
        end_time_extended_index = int(np.round(end_time_extended/dt))
        tdi_ts_segment_extended = TDI(dict([(k,TimeSeries(tdi_ts[k][start_time_index:end_time_extended_index], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
        len_time_extended = len(tdi_ts_segment_extended['A'])
        tdi_fs = TDI(dict([(k,tdi_ts_segment_extended[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
        freq_extended = np.array(tdi_fs.f[index_lower_cut:index_upper_cut])
        wave_in = wave_gen(*params_in.T, freqs=freq_extended,**waveform_kwargs)[0]
        A_in_extended = cp.fft.irfft(wave_in[0], n=len(tdi_ts_segment_extended['A']))
        E_in_extended = cp.fft.irfft(wave_in[1], n=len(tdi_ts_segment_extended['E']))
        A_in_extended = A_in_extended[:len(tdi_ts_segment['A'])]
        E_in_extended = E_in_extended[:len(tdi_ts_segment['E'])]
        A_fs_extended = cp.fft.rfft(A_in_extended)[index_lower_cut:index_upper_cut]
        E_fs = cp.fft.rfft(E_in_extended)[index_lower_cut:index_upper_cut]

        freq_extended_all = np.array(tdi_fs.f)
        wave_in = wave_gen(*params_in.T, freqs=freq_extended_all,**waveform_kwargs)[0]
        A_in_extended_all = cp.fft.irfft(wave_in[0], n=len_time_extended)[:len_time]
        E_in_extended_all = cp.fft.irfft(wave_in[1], n=len_time_extended)[:len_time]
        A_fs_extended_all = cp.fft.rfft(A_in_extended_all)[index_lower_cut:index_upper_cut]
        E_fs_extended_all = cp.fft.rfft(E_in_extended_all)[index_lower_cut:index_upper_cut]

    # plt.figure()
    # plt.plot(freq_new, data_A_fd, label='data')
    # plt.plot(freq_new, data_A_fd_no_window, label='data no window')
    # plt.plot(freq_new, A_fs, label='injected')
    # plt.plot(freq_new, A_fs_extended, label='injected extended')
    # plt.plot(freq_new, A_fs_extended_all, '--', label='injected extended all')
    # plt.legend()
    # plt.show()




    # plt.figure()
    # plt.plot(tdi_ts_segment['t']+start_time, tdi_ts_segment['A']*dt) 
    # # plt.plot(tdi_ts_segment_before['t']+start_time, A_in_before, label='t end before merger')
    # plt.plot(tdi_ts_segment['t']+start_time, A_in, '--', label='injected')

    # # plt.plot(tdi_ts_segment['t']+start_time, A_in_extended, '.', label='injected extended')
    # # plt.plot(tdi_ts_segment['t']+start_time, A_in_extended_all1, '--', label='injected extended all')
    # plt.plot(tdi_ts_segment['t']+start_time, A_in_extended_all, '--', label='injected extended all')
    # # plt.plot(ts+t_shift, A_found, label='found')
    # # plt.xlim([cat_mbhb[s_index]['CoalescenceTime']-3600, cat_mbhb[s_index]['CoalescenceTime']+3600])
    # # plt.plot(tdi_ts['t'], tdi_ts['A']*dt) 
    # plt.legend()
    # plt.show()

    if dataset == 'Radler':
        noise_estimate_interpolated = [MBHB_detective.psdA, MBHB_detective.psdE, MBHB_detective.psdT]
    if dataset == 'Sangria':
        # tdi_ts_first_segment = TDI(dict([(k,TimeSeries(tdi_ts_input[k][:len(ts)], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
        # tdi_fs_first = TDI(dict([(k,tdi_ts_first_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
        # noise_estimate_interpolated = get_psd(tdi_fs_first, freq_new)
        noise_estimate_interpolated = []
        for i in range(3):
            noise_estimate_interpolated.append(sp.interpolate.interp1d(noise_estimate[3], noise_estimate[i])(freq_new))

    MBHB_detective = MBHB_finder(tdi_fs, len_time, len_time_extended, freq_new, freq_extended_all, index_lower_cut, index_upper_cut, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample, noise_estimate_interpolated, merger_outside=merger_outside)
    if params_in is not None:
        print('SNR injected', float(MBHB_detective.SNR(params_in)))

    found_parameters_list, number_of_function_evaluations = find_MBHB(tdi_fs, len_time, len_time_extended, freq_new, freq_extended_all, index_lower_cut, index_upper_cut, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample, modes=modes,
                                                                       number_of_searches=1, noise_estimate=noise_estimate_interpolated, dump=False,
                                                                         number_of_MBHBs_per_segment=number_of_MBHBs_per_segment, saving_directory=saving_directory, merger_outside=merger_outside)
    for found_parameters in found_parameters_list:
        wave = wave_gen(*found_parameters, freqs=freq_full, initial_t_val= 0, modes=modes, direct=False, fill=True, squeeze=True, length=1024,  shift_t_limits=False)[0]
        found_signal_bbhx_tdi_fd = TDI(dict([(k,FrequencySeries(wave[i], fs=freq_full)) for i,k in enumerate(["A", "E", "T"])]))
        found_signal_bbhx_tdi_td = TDI(dict([(k,found_signal_bbhx_tdi_fd[k].ts.ifft()) for k in ["A", "E", "T"]]))
        for i, k in enumerate(["A", "E", "T"]):
            tdi_ts[k] -= found_signal_bbhx_tdi_td[k]


    return tdi_ts, found_parameters_list, number_of_function_evaluations

def find_MBHB(tdi_fs, len_time, len_time_extended, freq_new, freq_extended_all, index_lower_cut, index_upper_cut, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample, modes=[(2,2)], noise_estimate=None, number_of_searches=5, dump=True, number_of_MBHBs_per_segment=5, saving_directory='', merger_outside=False):
    SNR_threshold = 10
    SNR = SNR_threshold +1
    number_of_MBHBs = 0
    number_of_function_evaluations = 0
    found_parameters_list = []
    while SNR > SNR_threshold and number_of_MBHBs < number_of_MBHBs_per_segment:


        #compute noise
        # if time_segment_index != 0:

            # tdi_ts_current_segment = TDI(dict([(k,TimeSeries(tdi_ts[k][t_shift_index:t_shift_index+len(ts)], dt=dt, t0=0)) for i,k in enumerate(["A", "E", "T"])]))
            # tdi_fs_current = TDI(dict([(k,tdi_ts_current_segment[k].ts.fft(win=window)) for k in ["A", "E", "T"]])) 
            # noise_estimate_current2 = MBHB_detective = get_psd(tdi_fs_current, freq_new)
        # wave = wave_gen(*found_parameters, freqs=freq_new,**waveform_kwargs)[0]
        # found_signal_bbhx_tdi_fd = TDI(dict([(k,FrequencySeries(wave[i], fs=freq_new)) for i,k in enumerate(["A", "E", "T"])]))
        # found_signal_bbhx_tdi_td = TDI(dict([(k,found_signal_bbhx_tdi_fd[k].ts.ifft()) for k in ["A", "E", "T"]]))
        # wave = wave_gen(*params_in, freqs=freq_new,**waveform_kwargs)[0]
        # injected_signal_bbhx_tdi_fd = TDI(dict([(k,FrequencySeries(wave[i], fs=freq_new)) for i,k in enumerate(["A", "E", "T"])]))
        # injected_signal_bbhx_tdi_td = TDI(dict([(k,injected_signal_bbhx_tdi_fd[k].ts.ifft()) for k in ["A", "E", "T"]]))


        MBHB_detective = MBHB_finder(tdi_fs, len_time, len_time_extended, freq_new, freq_extended_all, index_lower_cut, index_upper_cut, boundaries, data_channels, wave_gen, waveform_kwargs, parameters_sample, merger_outside=merger_outside, noise=noise_estimate)
        for i in range(number_of_searches):
            found_parameters, SNR, function_evalutations = MBHB_detective.search_MBHB(tdi_fs, freq_new, boundaries)
            number_of_function_evaluations += function_evalutations
            if i == 0:
                found_parameters_best = deepcopy(found_parameters)
                SNR_best = deepcopy(SNR)
            else:
                if SNR > SNR_best:
                    found_parameters_best = deepcopy(found_parameters)
                    SNR_best = deepcopy(SNR)
            print( i, ' SNR ', SNR, 'best SNR ', SNR_best)
            if i != 0:
                if SNR_best < SNR_threshold:
                    break
        SNR = deepcopy(SNR_best)
        found_parameters = deepcopy(found_parameters_best)
        found_parameters = np.append(found_parameters,SNR)
        if SNR > SNR_threshold:
            found_parameters_list.append(found_parameters)
            if dump:
                # saving_directory = SAVEPATH+'found_signals_'+dataset+'_HM_'+str(HM)+'_'+str(weeks)+'w_original_seed'+str(seed)+'_padding_past_'+str(int(padding_past/3600/24))+'_future'+str(int(padding_future/3600/24))+'_GB'+str(GB)
                if not os.path.exists(saving_directory):
                    os.makedirs(saving_directory)
                pickle.dump(found_parameters, open(saving_directory+'/mbhb_t'+str(int(np.round(found_parameters[10]*100)))+'found_parameters.pkl', 'wb'))
            print('found MBHB at t = ', found_parameters[10])

            wave = MBHB_detective.get_wave(found_parameters, include_T=True)[0]

            # add plot of wave
            plt.figure()
            plt.plot(MBHB_detective.freq_new, MBHB_detective.data_channels[0])
            plt.plot(MBHB_detective.freq_new, wave[0])
            # plt.plot(self.freq_new, wave2[0][0])
            plt.show()

            # wave = wave_gen(*found_parameters, freqs=freq_new, initial_t_val= waveform_kwargs['initial_t_val'], modes=modes, direct=False, fill=True, squeeze=True, length=1024,  shift_t_limits=False)[0]
            for i in range(len(data_channels)):
                data_channels[i] -= wave[i]
            number_of_MBHBs += 1
            print(found_parameters, MBHB_detective.SNR(found_parameters), MBHB_detective.loglikelihood_ratio(found_parameters), found_parameters[4]/PC_SI/1e9,'found')
    return found_parameters_list, number_of_function_evaluations
        