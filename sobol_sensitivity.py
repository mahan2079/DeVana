# sobol_sensitivity.py

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli
from SALib.analyze import sobol
from joblib import Parallel, delayed
from FRF import frf  # Ensure FRF.py is in the same directory or properly installed
import pandas as pd

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")
plt.rc('text', usetex=True)  # Use LaTeX for rendering text in plots


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
    Perform Sobol sensitivity analysis on the singular response using the FRF module.

    Parameters:
        main_system_parameters (tuple): Parameters for the main system.
        dva_parameters_bounds (dict): Dictionary of DVA parameters with their bounds.
        dva_parameter_order (list): List specifying the order of DVA parameters.
        omega_start (float): Starting frequency (rad/s).
        omega_end (float): Ending frequency (rad/s).
        omega_points (int): Number of frequency points.
        num_samples_list (list): List of sample sizes for analysis.
        target_values_dict (dict): Dictionary containing target values for each mass.
        weights_dict (dict): Dictionary containing weights for each mass.
        visualize (bool, optional): Whether to generate visualizations. Defaults to False.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.

    Returns:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        warning_messages (list): List of warning messages encountered during analysis.
    """
    # Separate fixed and variable parameters
    fixed_parameters = {k: v for k, v in dva_parameters_bounds.items() if isinstance(v, (int, float))}
    variable_parameters = {k: v for k, v in dva_parameters_bounds.items() if isinstance(v, tuple)}

    if not variable_parameters:
        raise ValueError("No variable parameters specified for sensitivity analysis.")

    # Define the problem for SALib
    problem = {
        'num_vars': len(variable_parameters),
        'names': list(variable_parameters.keys()),
        'bounds': list(variable_parameters.values())
    }

    # Initialize the results dictionary
    all_results = {'S1': [], 'ST': [], 'samples': []}
    warning_messages = []

    for N in num_samples_list:
        print(f"\n[INFO] Running Sobol analysis with base sample size N = {N}...")
        param_values = saltelli.sample(problem, N, calc_second_order=True)

        print(f"  Evaluating singular response for {param_values.shape[0]} samples...")
        Y = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_frf)(
                main_system_parameters, fixed_parameters, variable_parameters, dva_parameter_order,
                omega_start, omega_end, omega_points, params,
                target_values_dict, weights_dict
            ) for params in param_values
        )

        Y = np.array(Y, dtype=np.float64)

        # Check for non-finite values and replace them with a default value
        if not np.all(np.isfinite(Y)):
            num_nonfinite = np.sum(~np.isfinite(Y))
            warning_message = f"[WARNING] Non-finite values encountered in Y for singular_response. Replacing {num_nonfinite} values with 0.0."
            print(warning_message)
            warning_messages.append(warning_message)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        # Perform Sobol analysis
        Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)

        # Store the results
        all_results['S1'].append(Si['S1'])
        all_results['ST'].append(Si['ST'])
        all_results['samples'].append(N)

        print(f"  Analysis for singular response completed.")

    if visualize:
        print("\n[INFO] Generating visualizations for the last run and convergence...")
        variable_param_names = list(variable_parameters.keys())
        visualize_last_run(all_results, variable_param_names)
        visualize_grouped_bar_plot_sorted_on_ST(all_results, variable_param_names)
        visualize_convergence_plots(all_results, variable_param_names)
        visualize_combined_heatmap(all_results, variable_param_names)
        visualize_comprehensive_radar_plots(all_results, variable_param_names)
        visualize_separate_radar_plots(all_results, variable_param_names)
        visualize_box_plots(all_results)
        visualize_violin_plots(all_results)
        visualize_scatter_S1_ST(all_results, variable_param_names)
        visualize_parallel_coordinates(all_results, variable_param_names)
        visualize_histograms(all_results)

    print("\n[INFO] Sobol sensitivity analysis completed.")
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
    """
    Evaluate the FRF for a given set of parameters and extract the singular response.

    Parameters:
        main_system_parameters (tuple): Parameters for the main system.
        fixed_parameters (dict): Fixed DVA parameters.
        variable_parameters (dict): Variable DVA parameters with bounds.
        dva_parameter_order (list): List specifying the order of DVA parameters.
        omega_start (float): Starting frequency (rad/s).
        omega_end (float): Ending frequency (rad/s).
        omega_points (int): Number of frequency points.
        params (numpy.ndarray): Sampled values for variable parameters.
        target_values_dict (dict): Dictionary containing target values for each mass.
        weights_dict (dict): Dictionary containing weights for each mass.

    Returns:
        float: The singular response value.
    """
    try:
        # Combine fixed and sampled parameters
        sampled_params = {name: val for name, val in zip(variable_parameters.keys(), params)}
        dva_parameters_combined = {**fixed_parameters, **sampled_params}

        # Ensure the parameters are ordered correctly
        dva_parameters_tuple = tuple(dva_parameters_combined[param] for param in dva_parameter_order)

        # Extract target values and weights for each mass
        target_values_mass1 = target_values_dict.get('mass_1', {})
        weights_mass1 = weights_dict.get('mass_1', {})
        target_values_mass2 = target_values_dict.get('mass_2', {})
        weights_mass2 = weights_dict.get('mass_2', {})
        target_values_mass3 = target_values_dict.get('mass_3', {})
        weights_mass3 = weights_dict.get('mass_3', {})
        target_values_mass4 = target_values_dict.get('mass_4', {})
        weights_mass4 = weights_dict.get('mass_4', {})
        target_values_mass5 = target_values_dict.get('mass_5', {})
        weights_mass5 = weights_dict.get('mass_5', {})

        # Run the FRF analysis
        frf_results = frf(
            main_system_parameters=main_system_parameters,
            dva_parameters=dva_parameters_tuple,
            omega_start=omega_start,
            omega_end=omega_end,
            omega_points=omega_points,
            target_values_mass1=target_values_mass1,
            weights_mass1=weights_mass1,
            target_values_mass2=target_values_mass2,
            weights_mass2=weights_mass2,
            target_values_mass3=target_values_mass3,
            weights_mass3=weights_mass3,
            target_values_mass4=target_values_mass4,
            weights_mass4=weights_mass4,
            target_values_mass5=target_values_mass5,
            weights_mass5=weights_mass5,
            plot_figure=False,    # Disable plotting during sensitivity analysis
            show_peaks=False,     # Disable peak annotations during sensitivity analysis
            show_slopes=False     # Disable slope plotting during sensitivity analysis
        )

        # Extract the singular response
        value = frf_results.get('singular_response', 0.0)
        return value if np.isfinite(value) else 0.0  # Replace non-finite values with 0.0
    except Exception as e:
        print(f"[ERROR] Exception occurred during model evaluation: {e}. Returning default value 0.0.")
        return 0.0  # Return default value to maintain sample size


def save_results(all_results, param_names, num_samples_list, folder_name='sobol_analysis'):
    """
    Save the Sobol sensitivity results to CSV files and generate sorted sensitivity CSV.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        num_samples_list (list): List of sample sizes used.
        folder_name (str): Base folder to save result files.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save raw sensitivity indices
    results_file = os.path.join(folder_name, 'singular_response_results.csv')
    print(f"\n[INFO] Saving results for singular response to {results_file}...")
    with open(results_file, 'w') as f:
        header = 'run,sample_size,' + ','.join(param_names) + '\n'
        f.write(header)
        for run_idx, num_samples in enumerate(all_results['samples']):
            S1_values = all_results['S1'][run_idx]
            S1_values_str = ','.join(str(S1_values[param_idx]) for param_idx in range(len(S1_values)))
            f.write(f"{run_idx + 1},{num_samples},{S1_values_str}\n")
    print(f"[INFO] Results saved to {results_file}.")

    # Generate sorted sensitivity CSV
    save_sorted_sensitivity(all_results, param_names, folder_name)


def save_sorted_sensitivity(all_results, param_names, folder_name='sobol_analysis'):
    """
    Save a CSV file with parameters sorted from most to least effective based on sensitivity indices.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder to save the sorted sensitivity file.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    last_run_idx = -1  # Use the last run for sorting
    S1_last_run = np.array(all_results['S1'][last_run_idx])
    ST_last_run = np.array(all_results['ST'][last_run_idx])

    # Combine S1 and ST into a DataFrame
    df = pd.DataFrame({
        'Parameter': param_names,
        'S1': S1_last_run,
        'ST': ST_last_run
    })

    # Sort by S1 descending
    df_sorted_S1 = df.sort_values(by='S1', ascending=False)
    sorted_file_S1 = os.path.join(folder_name, 'singular_response_sorted_S1.csv')
    print(f"\n[INFO] Saving sorted sensitivity indices based on S1 to {sorted_file_S1}...")
    df_sorted_S1.to_csv(sorted_file_S1, index=False)
    print(f"[INFO] Sorted S1 sensitivities saved to {sorted_file_S1}.")

    # Sort by ST descending
    df_sorted_ST = df.sort_values(by='ST', ascending=False)
    sorted_file_ST = os.path.join(folder_name, 'singular_response_sorted_ST.csv')
    print(f"[INFO] Saving sorted sensitivity indices based on ST to {sorted_file_ST}...")
    df_sorted_ST.to_csv(sorted_file_ST, index=False)
    print(f"[INFO] Sorted ST sensitivities saved to {sorted_file_ST}.")


def calculate_and_save_errors(all_results, param_names, folder_name='sobol_analysis'):
    """
    Calculate statistical errors for Sobol indices and save them to CSV files.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder to save error files.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    errors_file = os.path.join(folder_name, 'singular_response_errors.csv')
    print(f"\n[INFO] Calculating errors for singular response...")
    with open(errors_file, 'w') as f:
        f.write('parameter,measure,variance,std,MAD,CI_lower,CI_upper\n')
        for param_idx, param in enumerate(param_names):
            for measure in ['S1', 'ST']:
                try:
                    values = np.array([all_results[measure][i][param_idx]
                                       for i in range(len(all_results[measure]))])
                except IndexError as e:
                    print(f"[ERROR] IndexError encountered while calculating errors: {e}")
                    continue

                variance = np.var(values, ddof=1)  # Sample variance
                std = np.std(values, ddof=1)       # Sample standard deviation
                mad = np.mean(np.abs(values - np.mean(values)))  # Mean Absolute Deviation
                ci_lower = np.mean(values) - 1.96 * std / np.sqrt(len(values))
                ci_upper = np.mean(values) + 1.96 * std / np.sqrt(len(values))

                f.write(f"{param},{measure},{variance},{std},{mad},{ci_lower},{ci_upper}\n")
    print(f"[INFO] Errors saved to {errors_file}.")


def visualize_last_run(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize the Sobol sensitivity indices for the last run using a grouped bar plot.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Visualizing last run results for singular response...")
    last_run_idx = -1
    S1_last_run = np.array(all_results['S1'][last_run_idx])
    ST_last_run = np.array(all_results['ST'][last_run_idx])

    sorted_indices_S1 = np.argsort(S1_last_run)[::-1]
    sorted_param_names_S1 = [param_names[i] for i in sorted_indices_S1]
    S1_sorted = S1_last_run[sorted_indices_S1]
    ST_sorted = ST_last_run[sorted_indices_S1]

    # Plot grouped bar plot for S1 and ST
    plt.figure(figsize=(25, 15))
    x = np.arange(len(sorted_param_names_S1))  # label locations
    width = 0.35  # width of the bars

    plt.bar(x - width/2, S1_sorted, width, label=r'$S_1$', color='skyblue')
    plt.bar(x + width/2, ST_sorted, width, label=r'$S_T$', color='salmon')

    plt.xlabel('Parameters', fontsize=20)
    plt.ylabel('Sensitivity Index', fontsize=20)
    plt.title('First-order ($S_1$) and Total-order ($S_T$) Sensitivity Indices - Singular Response', fontsize=24)
    plt.xticks(x, [format_parameter_name(param) for param in sorted_param_names_S1], rotation=90, fontsize=12)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.savefig(os.path.join(folder_name, 'singular_response_S1_ST_grouped_bar.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved grouped bar plot to singular_response_S1_ST_grouped_bar.png.")


def visualize_grouped_bar_plot_sorted_on_ST(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize a grouped bar plot of S1 and ST sensitivity indices sorted based on ST.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating grouped bar plot sorted on ST sensitivity indices...")
    last_run_idx = -1
    S1_last_run = np.array(all_results['S1'][last_run_idx])
    ST_last_run = np.array(all_results['ST'][last_run_idx])

    # Sort parameters based on ST
    sorted_indices_ST = np.argsort(ST_last_run)[::-1]
    sorted_param_names_ST = [param_names[i] for i in sorted_indices_ST]
    S1_sorted = S1_last_run[sorted_indices_ST]
    ST_sorted = ST_last_run[sorted_indices_ST]

    plt.figure(figsize=(25, 15))
    x = np.arange(len(sorted_param_names_ST))  # label locations
    width = 0.35  # width of the bars

    plt.bar(x - width/2, S1_sorted, width, label=r'$S_1$', color='skyblue')
    plt.bar(x + width/2, ST_sorted, width, label=r'$S_T$', color='salmon')

    plt.xlabel('Parameters', fontsize=20)
    plt.ylabel('Sensitivity Index', fontsize=20)
    plt.title('First-order ($S_1$) and Total-order ($S_T$) Sensitivity Indices - Singular Response (Sorted by $S_T$)', fontsize=24)
    plt.xticks(x, [format_parameter_name(param) for param in sorted_param_names_ST], rotation=90, fontsize=12)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(folder_name, 'singular_response_S1_ST_grouped_bar_sorted_on_ST.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved grouped bar plot sorted on ST to singular_response_S1_ST_grouped_bar_sorted_on_ST.png.")


def visualize_convergence_plots(all_results, param_names, folder_name='sobol_analysis'):
    print(f"\n[INFO] Generating convergence plots as subplots for parameters...")
    os.makedirs(folder_name, exist_ok=True)

    sample_sizes = all_results['samples']
    S1_matrix = np.array(all_results['S1'])  # Shape: (num_runs, num_params)
    ST_matrix = np.array(all_results['ST'])  # Shape: (num_runs, num_params)

    plots_per_fig = 12
    total_params = len(param_names)
    num_figs = int(np.ceil(total_params / plots_per_fig))

    for fig_idx in range(num_figs):
        plt.figure(figsize=(25, 20))
        start_idx = fig_idx * plots_per_fig
        end_idx = min(start_idx + plots_per_fig, total_params)

        for subplot_idx, param_idx in enumerate(range(start_idx, end_idx)):
            param = param_names[param_idx]
            ax = plt.subplot(3, 4, subplot_idx + 1)

            S1_values = S1_matrix[:, param_idx]
            ST_values = ST_matrix[:, param_idx]

            ax.plot(sample_sizes, S1_values, marker='o', linestyle='-', color='tab:blue', label=r'$S_1$')
            ax.plot(sample_sizes, ST_values, marker='s', linestyle='-', color='tab:red', label=r'$S_T$')

            ax.set_xlabel('Sample Size', fontsize=12)
            ax.set_ylabel('Sensitivity Index', fontsize=12)
            ax.set_title(f'Convergence for {format_parameter_name(param)}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        plot_filename = f'convergence_plots_fig_{fig_idx + 1}.png'
        plt.savefig(os.path.join(folder_name, plot_filename), bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved convergence subplot figure to '{plot_filename}'.")


def visualize_combined_heatmap(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize a combined heatmap of S1 and ST sensitivity indices.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating combined heatmap for sensitivity indices...")
    last_run_idx = -1
    S1_last_run = np.array(all_results['S1'][last_run_idx])
    ST_last_run = np.array(all_results['ST'][last_run_idx])

    # Create a DataFrame for heatmap
    df = pd.DataFrame({
        'Parameter': param_names,
        'S1': S1_last_run,
        'ST': ST_last_run
    }).set_index('Parameter')

    # Sort parameters based on S1 for better visualization
    df_sorted = df.sort_values(by='S1', ascending=False)

    plt.figure(figsize=(20, max(8, len(param_names) * 0.3)))
    sns.heatmap(df_sorted, annot=True, cmap='coolwarm', cbar_kws={'label': 'Sensitivity Index'}, linewidths=.5, linecolor='gray')
    plt.title('Combined Heatmap of First-order ($S_1$) and Total-order ($S_T$) Sensitivity Indices - Singular Response', fontsize=24)
    plt.xlabel('Sensitivity Indices', fontsize=20)
    plt.ylabel('Parameters', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_combined_heatmap.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved combined heatmap to singular_response_combined_heatmap.png.")


def visualize_comprehensive_radar_plots(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize a comprehensive radar plot showing both S1 and ST for all parameters.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating comprehensive radar plot for all parameters...")
    last_run_idx = -1
    S1 = np.array(all_results['S1'][last_run_idx])
    ST = np.array(all_results['ST'][last_run_idx])

    num_vars = len(param_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Initialize the radar plot
    fig, ax = plt.subplots(figsize=(30, 30), subplot_kw=dict(polar=True))

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], [format_parameter_name(param) for param in param_names], fontsize=14)

    # Draw ylabels
    ax.set_rlabel_position(30)
    max_val = max(np.max(S1), np.max(ST)) * 1.1
    plt.yticks(np.linspace(0, max_val, 5), [f"{v:.2f}" for v in np.linspace(0, max_val, 5)], color="grey", size=12)
    plt.ylim(0, max_val)

    # Plot S1
    values_S1 = S1.tolist()
    values_S1 += values_S1[:1]
    ax.plot(angles, values_S1, linewidth=3, linestyle='solid', label=r'$S_1$')
    ax.fill(angles, values_S1, alpha=0.25, color='skyblue')

    # Plot ST
    values_ST = ST.tolist()
    values_ST += values_ST[:1]
    ax.plot(angles, values_ST, linewidth=3, linestyle='solid', label=r'$S_T$')
    ax.fill(angles, values_ST, alpha=0.25, color='salmon')

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=20)

    plt.title('Comprehensive Radar Plot of Sensitivity Indices - Singular Response', fontsize=28, y=1.05)
    plt.tight_layout()

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.savefig(os.path.join(folder_name, 'singular_response_comprehensive_radar_plot.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved comprehensive radar plot to singular_response_comprehensive_radar_plot.png.")


def visualize_separate_radar_plots(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize two separate radar plots: one for S1 and one for ST.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating separate radar plots for S1 and ST...")
    last_run_idx = -1
    S1 = np.array(all_results['S1'][last_run_idx])
    ST = np.array(all_results['ST'][last_run_idx])

    num_vars = len(param_names)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Radar plot for S1
    fig, ax = plt.subplots(figsize=(30, 30), subplot_kw=dict(polar=True))
    values_S1 = S1.tolist()
    values_S1 += values_S1[:1]
    ax.plot(angles, values_S1, linewidth=3, linestyle='solid', label=r'$S_1$')
    ax.fill(angles, values_S1, alpha=0.25, color='skyblue')
    plt.xticks(angles[:-1], [format_parameter_name(param) for param in param_names], fontsize=14)
    ax.set_rlabel_position(30)
    max_val_S1 = np.max(S1) * 1.1
    plt.yticks(np.linspace(0, max_val_S1, 5), [f"{v:.2f}" for v in np.linspace(0, max_val_S1, 5)], color="grey", size=12)
    plt.ylim(0, max_val_S1)
    plt.title('Radar Plot of First-order Sensitivity Indices ($S_1$) - Singular Response', fontsize=28, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_radar_plot_S1.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved radar plot for S1 to singular_response_radar_plot_S1.png.")

    # Radar plot for ST
    fig, ax = plt.subplots(figsize=(30, 30), subplot_kw=dict(polar=True))
    values_ST = ST.tolist()
    values_ST += values_ST[:1]
    ax.plot(angles, values_ST, linewidth=3, linestyle='solid', label=r'$S_T$')
    ax.fill(angles, values_ST, alpha=0.25, color='salmon')
    plt.xticks(angles[:-1], [format_parameter_name(param) for param in param_names], fontsize=14)
    ax.set_rlabel_position(30)
    max_val_ST = np.max(ST) * 1.1
    plt.yticks(np.linspace(0, max_val_ST, 5), [f"{v:.2f}" for v in np.linspace(0, max_val_ST, 5)], color="grey", size=12)
    plt.ylim(0, max_val_ST)
    plt.title('Radar Plot of Total-order Sensitivity Indices ($S_T$) - Singular Response', fontsize=28, y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_radar_plot_ST.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved radar plot for ST to singular_response_radar_plot_ST.png.")


def visualize_box_plots(all_results, folder_name='sobol_analysis'):
    """
    Visualize box plots for S1 and ST sensitivity indices.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating box plots for sensitivity indices...")
    # Combine all runs into a DataFrame
    data = {
        'S1': np.concatenate(all_results['S1']),
        'ST': np.concatenate(all_results['ST'])
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(16, 12))
    sns.boxplot(data=df, palette=['skyblue', 'salmon'])
    plt.xlabel('Sensitivity Indices', fontsize=22)
    plt.ylabel('Values', fontsize=22)
    plt.title('Box Plots of First-order ($S_1$) and Total-order ($S_T$) Sensitivity Indices', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_box_plots.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved box plots to singular_response_box_plots.png.")


def visualize_violin_plots(all_results, folder_name='sobol_analysis'):
    """
    Visualize violin plots for S1 and ST sensitivity indices.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating violin plots for sensitivity indices...")
    # Combine all runs into a DataFrame
    data = {
        'S1': np.concatenate(all_results['S1']),
        'ST': np.concatenate(all_results['ST'])
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(16, 12))
    sns.violinplot(data=df, palette=['skyblue', 'salmon'], inner='quartile')
    plt.xlabel('Sensitivity Indices', fontsize=22)
    plt.ylabel('Values', fontsize=22)
    plt.title('Violin Plots of First-order ($S_1$) and Total-order ($S_T$) Sensitivity Indices', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_violin_plots.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved violin plots to singular_response_violin_plots.png.")


def visualize_scatter_S1_ST(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize a scatter plot of S1 vs ST indices for all parameters.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating scatter plot of S1 vs ST...")
    last_run_idx = -1
    S1_last_run = np.array(all_results['S1'][last_run_idx])
    ST_last_run = np.array(all_results['ST'][last_run_idx])

    plt.figure(figsize=(18, 14))
    scatter = plt.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)), cmap='tab20', edgecolor='k', s=200)

    # Annotate each point with parameter name
    for i, param in enumerate(param_names):
        plt.text(S1_last_run[i] + 0.001, ST_last_run[i] + 0.001, format_parameter_name(param), fontsize=12)

    plt.xlabel(r'$S_1$', fontsize=22)
    plt.ylabel(r'$S_T$', fontsize=22)
    plt.title('Scatter Plot of First-order vs Total-order Sensitivity Indices', fontsize=26)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_S1_vs_ST_scatter.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved scatter plot of S1 vs ST to singular_response_S1_vs_ST_scatter.png.")


def visualize_parallel_coordinates(all_results, param_names, folder_name='sobol_analysis'):
    """
    Visualize parallel coordinates of sensitivity indices across sample sizes.

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        param_names (list): List of parameter names.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating parallel coordinates plot for sensitivity indices...")
    data = []
    for run_idx, num_samples in enumerate(all_results['samples']):
        row = {'Sample Size': num_samples}
        for param_idx, param in enumerate(param_names):
            row[f'S1_{param}'] = all_results['S1'][run_idx][param_idx]
            row[f'ST_{param}'] = all_results['ST'][run_idx][param_idx]
        data.append(row)

    df = pd.DataFrame(data)
    plt.figure(figsize=(25, 20))
    for param in param_names:
        plt.plot(df['Sample Size'], df[f'S1_{param}'], label=f'S1 {format_parameter_name(param)}', linestyle='-', marker='o', alpha=0.6)
        plt.plot(df['Sample Size'], df[f'ST_{param}'], label=f'ST {format_parameter_name(param)}', linestyle='--', marker='s', alpha=0.6)

    plt.xlabel('Sample Size', fontsize=22)
    plt.ylabel('Sensitivity Index', fontsize=22)
    plt.title('Parallel Coordinates of Sensitivity Indices Across Sample Sizes', fontsize=28)
    # To prevent the legend from being too large, we can omit it or use a proxy
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_parallel_coordinates.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved parallel coordinates plot to singular_response_parallel_coordinates.png.")


def visualize_histograms(all_results, folder_name='sobol_analysis'):
    """
    Visualize histograms of sensitivity indices (S1 and ST).

    Parameters:
        all_results (dict): Dictionary containing Sobol sensitivity results.
        folder_name (str): Base folder where results are saved.
    """
    print(f"\n[INFO] Generating histograms for sensitivity indices...")
    last_run_idx = -1
    S1_last_run = np.array(all_results['S1'][last_run_idx])
    ST_last_run = np.array(all_results['ST'][last_run_idx])

    # Histogram for S1
    plt.figure(figsize=(18, 12))
    sns.histplot(S1_last_run, bins=30, kde=True, color='skyblue')
    plt.xlabel(r'$S_1$', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.title('Histogram of First-order Sensitivity Indices ($S_1$)', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_S1_histogram.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved S1 histogram to singular_response_S1_histogram.png.")

    # Histogram for ST
    plt.figure(figsize=(18, 12))
    sns.histplot(ST_last_run, bins=30, kde=True, color='salmon')
    plt.xlabel(r'$S_T$', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.title('Histogram of Total-order Sensitivity Indices ($S_T$)', fontsize=26)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'singular_response_ST_histogram.png'), bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved ST histogram to singular_response_ST_histogram.png.")


def format_parameter_name(param):
    """
    Format parameter names using LaTeX symbols for plotting.

    Parameters:
        param (str): The parameter name.

    Returns:
        str: Formatted parameter name.
    """
    GREEK_LETTERS = {
        'beta': r'\beta',
        'lambda': r'\lambda',
        'mu': r'\mu',
        'nu': r'\nu'
    }
    for greek_letter, symbol in GREEK_LETTERS.items():
        if param.startswith(greek_letter):
            index = param[len(greek_letter):]  # Adjusted to capture the entire index
            return f'${symbol}_{{{index}}}$'
    return param.replace("_", " ")
