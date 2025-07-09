import os
import numpy as np
from datetime import datetime

def save_de_results(found_parameters_tf, found_snr_found_tf, true_snr,
                    results_tf, parameters_history_tf,
                    folder_name="differential_evolution_results",
                    filename_prefix="run"):
    """
    Save DE results as a .npz file inside a manually defined project folder.

    Parameters:
    - folder_name: relative path like 'differential_evolution/differential_evolution_results'
    """

    os.makedirs(folder_name, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.npz"
    filepath = os.path.join(folder_name, filename)

    np.savez_compressed(filepath,
        found_parameters=found_parameters_tf,
        found_snr=found_snr_found_tf,
        true_snr=true_snr,
        result_x=results_tf.x,
        result_fun=results_tf.fun,
        result_nit=results_tf.nit,
        result_success=results_tf.success,
        result_message=results_tf.message,
        parameters_history=parameters_history_tf
    )

    print(f"✅ Saved results to {filepath}")



def load_de_results(filepath):
    """
    Load DE results from a .npz file created by save_de_results().

    Parameters:
    - filepath: str, path to the .npz file

    Returns:
    - dict with keys:
        'found_parameters', 'found_snr', 'true_snr',
        'result_x', 'result_fun', 'result_nit',
        'result_success', 'result_message', 'parameters_history'
    """

    data = np.load(filepath)

    results = {
        'found_parameters': data['found_parameters'],
        'found_snr': data['found_snr'].item(),
        'true_snr': data['true_snr'].item(),
        'result_x': data['result_x'],
        'result_fun': data['result_fun'].item(),
        'result_nit': int(data['result_nit']),
        'result_success': bool(data['result_success']),
        'result_message': str(data['result_message']),
        'parameters_history': data['parameters_history']
    }

    return results
