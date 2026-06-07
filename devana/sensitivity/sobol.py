import numpy as np
import os
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
from devana.physics.frf import frf
import pandas as pd

def perform_sobol_analysis(
    main_system_parameters,
    dva_parameters_bounds,
    dva_parameter_order,
    omega_start,
    omega_end,
    omega_points,
    num_samples_list,
    target_values_dict,
    weights_dict,
    visualize=False,
    n_jobs=1
):
    """
    Perform Sobol sensitivity analysis on the singular response.
    """
    if isinstance(dva_parameters_bounds, list):
        bounds_dict = {}
        order_list = []
        for item in dva_parameters_bounds:
            name, low, up, fixed = item
            order_list.append(name)
            if not fixed: bounds_dict[name] = (low, up)
            else: bounds_dict[name] = low
        dva_parameters_bounds = bounds_dict
        if dva_parameter_order is None:
            dva_parameter_order = order_list

    fixed_parameters = {k: v for k, v in dva_parameters_bounds.items() if not isinstance(v, tuple)}
    variable_parameters = {k: v for k, v in dva_parameters_bounds.items() if isinstance(v, tuple)}

    if not variable_parameters:
        raise ValueError("No variable parameters specified for sensitivity analysis.")

    problem = {
        'num_vars': len(variable_parameters),
        'names': list(variable_parameters.keys()),
        'bounds': list(variable_parameters.values())
    }

    all_results = {'S1': [], 'ST': [], 'samples': []}
    warning_messages = []

    for N in num_samples_list:
        param_values = saltelli.sample(problem, N, calc_second_order=True)
        Y = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_frf)(
                main_system_parameters, fixed_parameters, variable_parameters, dva_parameter_order,
                omega_start, omega_end, omega_points, params,
                target_values_dict, weights_dict
            ) for params in param_values
        )
        Y = np.array(Y, dtype=np.float64)
        if not np.all(np.isfinite(Y)):
            num_nonfinite = np.sum(~np.isfinite(Y))
            msg = f"Non-finite values encountered in Y. Replacing {num_nonfinite} values with 0.0."
            warning_messages.append(msg)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
        all_results['S1'].append(Si['S1'])
        all_results['ST'].append(Si['ST'])
        all_results['samples'].append(N)

    if visualize:
        # Plotting is decoupled; user should use visualization utilities separately.
        pass

    return all_results, warning_messages

def evaluate_frf(
    main_system_parameters,
    fixed_parameters,
    variable_parameters,
    dva_parameter_order,
    omega_start,
    omega_end,
    omega_points,
    params,
    target_values_dict,
    weights_dict
):
    try:
        sampled_params = {name: val for name, val in zip(variable_parameters.keys(), params)}
        dva_parameters_combined = {**fixed_parameters, **sampled_params}
        dva_parameters_tuple = tuple(dva_parameters_combined[param] for param in dva_parameter_order)

        frf_results = frf(
            main_system_parameters=main_system_parameters,
            dva_parameters=dva_parameters_tuple,
            omega_start=omega_start,
            omega_end=omega_end,
            omega_points=omega_points,
            target_values_mass1=target_values_dict.get('mass_1', {}),
            weights_mass1=weights_dict.get('mass_1', {}),
            target_values_mass2=target_values_dict.get('mass_2', {}),
            weights_mass2=weights_dict.get('mass_2', {}),
            target_values_mass3=target_values_dict.get('mass_3', {}),
            weights_mass3=weights_dict.get('mass_3', {}),
            target_values_mass4=target_values_dict.get('mass_4', {}),
            weights_mass4=weights_dict.get('mass_4', {}),
            target_values_mass5=target_values_dict.get('mass_5', {}),
            weights_mass5=weights_dict.get('mass_5', {}),
        )
        value = frf_results.get('singular_response', 0.0)
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0

def save_results(all_results, param_names, folder_name='sobol_analysis'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    results_file = os.path.join(folder_name, 'singular_response_results.csv')
    with open(results_file, 'w') as f:
        header = 'run,sample_size,' + ','.join(param_names) + '\n'
        f.write(header)
        for run_idx, num_samples in enumerate(all_results['samples']):
            S1_values = all_results['S1'][run_idx]
            S1_values_str = ','.join(str(S1_values[param_idx]) for param_idx in range(len(S1_values)))
            f.write(f"{run_idx + 1},{num_samples},{S1_values_str}\n")
    save_sorted_sensitivity(all_results, param_names, folder_name)

def save_sorted_sensitivity(all_results, param_names, folder_name='sobol_analysis'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    last_run_idx = -1
    df = pd.DataFrame({'Parameter': param_names, 'S1': all_results['S1'][last_run_idx], 'ST': all_results['ST'][last_run_idx]})
    df.sort_values(by='S1', ascending=False).to_csv(os.path.join(folder_name, 'singular_response_sorted_S1.csv'), index=False)
    df.sort_values(by='ST', ascending=False).to_csv(os.path.join(folder_name, 'singular_response_sorted_ST.csv'), index=False)

def calculate_and_save_errors(all_results, param_names, folder_name='sobol_analysis'):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    errors_file = os.path.join(folder_name, 'singular_response_errors.csv')
    with open(errors_file, 'w') as f:
        f.write('parameter,measure,variance,std,MAD,CI_lower,CI_upper\n')
        for param_idx, param in enumerate(param_names):
            for measure in ['S1', 'ST']:
                values = np.array([all_results[measure][i][param_idx] for i in range(len(all_results[measure]))])
                variance = np.var(values, ddof=1)
                std = np.std(values, ddof=1)
                mad = np.mean(np.abs(values - np.mean(values)))
                ci_lower = np.mean(values) - 1.96 * std / np.sqrt(len(values))
                ci_upper = np.mean(values) + 1.96 * std / np.sqrt(len(values))
                f.write(f"{param},{measure},{variance},{std},{mad},{ci_lower},{ci_upper}\n")

def format_parameter_name(param):
    GREEK_LETTERS = {'beta': r'\beta', 'lambda': r'\lambda', 'mu': r'\mu', 'nu': r'\nu'}
    for greek_letter, symbol in GREEK_LETTERS.items():
        if param.startswith(greek_letter):
            index = param[len(greek_letter):]
            return f'${symbol}_{{{index}}}$'
    return param.replace("_", " ")
