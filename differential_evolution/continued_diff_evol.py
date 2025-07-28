from tools.MBHB_differential_evolution import scale_parameters, unscale_parameters, scale_fixed_parameters
from tools.save_and_load_DE import save_de_results, load_de_results
import numpy as np
from lisatools.utils.constants import *


Tobs = YRSID_SI/3
hours_before_merger = 20
time_before_merger = hours_before_merger*60*60
t_ref = 0.95 * Tobs
cutoff_time = t_ref - time_before_merger
max_time = t_ref + (24 - hours_before_merger)*60*60


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

boundaries_array = np.array(list(boundaries.values()))

found_parameters = load_de_results(filepath="euler_output/tf_run_20250716_141948.npz")["found_parameters"]

found_parameters_01 = scale_parameters(found_parameters, boundaries=boundaries_array)
found_parameters_01 = np.delete(found_parameters_01, 4) 
found_parameters_01

