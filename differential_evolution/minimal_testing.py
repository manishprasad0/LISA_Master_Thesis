import os
import sys

import numpy as np
import multiprocessing as mp
import time
from datetime import datetime

# LISA modules
from lisatools.utils.constants import *
from lisatools.sensitivity  import AE1SensitivityMatrix, AET1SensitivityMatrix
from bbhx.waveformbuild import BBHWaveformFD

gpu = True
if gpu:
    import cupy as cp
else:
    import numpy as cp

def main():
    Tobs = YRSID_SI/12
    dt = 5.

    wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False, initial_t_val = 0.0), use_gpu=gpu)

    m1 = 3e5
    m2 = 1.5e5
    a1 = 0.2
    a2 = 0.4
    dist = 10 * PC_SI * 1e9
    phi_ref = np.pi/2
    f_ref = 0.0
    inc = np.pi/3
    lam = np.pi/1.
    beta = np.pi/4.
    psi = np.pi/4.
    t_ref = 0.95 * Tobs

    parameters = cp.array([m1, m2, a1, a2, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref])
    modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    waveform_kwargs = dict(length=1024, direct=False, fill=True, squeeze=False, modes=modes)

    N = int(int(Tobs / dt)/2)*2
    Tobs = N * dt
    freq = cp.fft.rfftfreq(N, dt)
    freq[0] = freq[1]
    
    signal = wave_gen(*parameters, freqs=freq, **waveform_kwargs) 
    print(signal)

if __name__ == "__main__":
    main()
