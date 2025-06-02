from bbhx.utils.constants import *
from bbhx.utils.transform import *
import numpy as cp
import scipy as sp
import xarray as xr
import time
import os
import pickle
from copy import deepcopy
from ldc.lisa.noise import get_noise_model
import matplotlib.pyplot as plt

def get_noise_from_frequency_domain(tdi_fs, nperseg=15000):
    tdi_ts = xr.Dataset(dict([(k, tdi_fs[k].ts.ifft()) for k in ["A", "E", "T"]]))
    dt = tdi_ts.t.values[1] - tdi_ts.t.values[0]
    f, psdA =  sp.signal.welch(tdi_ts["A"], fs=1.0/dt, nperseg=nperseg)#, average='mean', window= 'boxcar')
    f, psdE =  sp.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=nperseg)
    # f2, psdE2 =  sp.signal.welch(tdi_ts["E"], fs=1.0/dt, nperseg=len(tdi_ts["A"]), scaling='spectrum')
    f, psdT =  sp.signal.welch(tdi_ts["T"], fs=1.0/dt, nperseg=nperseg)
    return f, psdA, psdE, psdT

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

    def calculate_Amplitude(self, pGBs):
        dh, hh = self.get_dh_hh(pGBs)
        A = dh / hh
        return A