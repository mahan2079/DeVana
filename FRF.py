# FRF.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.signal import find_peaks
from adjustText import adjust_text

# --- Helper Functions ---

def calculate_slopes(peak_positions, peak_values):
    """
    Calculate slopes between pairs of peaks.

    Parameters:
        peak_positions (array-like): Positions of the peaks.
        peak_values (array-like): Values of the peaks.

    Returns:
        slopes (dict): Slopes between peak pairs.
        slope_max (float): Maximum absolute slope value.
    """
    slopes = {}
    slope_max = np.nan
    num_peaks = len(peak_positions)

    if num_peaks > 1:
        for i in range(num_peaks):
            for j in range(i + 1, num_peaks):
                # Calculate the slope between two peaks
                delta_pos = peak_positions[j] - peak_positions[i]
                if delta_pos != 0:
                    slope = (peak_values[j] - peak_values[i]) / delta_pos
                else:
                    slope = 0  # Avoid division by zero
                slopes[f'slope_{i+1}_{j+1}'] = slope
                # Keep track of the maximum absolute slope
                if np.isnan(slope_max) or abs(slope) > abs(slope_max):
                    slope_max = slope
    return slopes, slope_max

def remove_zero_mass_dofs(mass_matrix, damping_matrix, stiffness_matrix, forcing_matrix, tol=1e-8):
    """
    Remove Degrees of Freedom (DOFs) with zero or near-zero rows or columns in any of the provided matrices.

    Parameters:
        mass_matrix (ndarray): Mass matrix (NxN).
        damping_matrix (ndarray): Damping matrix (NxN).
        stiffness_matrix (ndarray): Stiffness matrix (NxN).
        forcing_matrix (ndarray): Forcing matrix/vector (N or NxM).
        tol (float): Tolerance for determining if a row or column is zero. Default is 1e-8.

    Returns:
        mass_matrix_reduced (ndarray): Reduced mass matrix.
        damping_matrix_reduced (ndarray): Reduced damping matrix.
        stiffness_matrix_reduced (ndarray): Reduced stiffness matrix.
        forcing_matrix_reduced (ndarray): Reduced forcing matrix/vector.
        active_dofs (ndarray): Boolean array indicating active DOFs.
    """
    # Initialize a list to collect DOFs to remove
    dofs_to_remove = np.zeros(mass_matrix.shape[0], dtype=bool)

    # Helper function to identify zero DOFs in a matrix
    def identify_zero_dofs(matrix, is_forcing=False):
        """
        Identify DOFs (rows/columns) that are zero or near-zero in the given matrix.

        Parameters:
            matrix (ndarray): The matrix to check.
            is_forcing (bool): If True, treats the matrix as a forcing vector/matrix.

        Returns:
            zero_dofs (ndarray): Boolean array indicating zero DOFs.
        """
        if is_forcing:
            # If forcing_matrix is 1D, treat each entry as a DOF
            if matrix.ndim == 1:
                zero_dofs = np.isclose(matrix, 0, atol=tol)
            else:
                # If forcing_matrix is 2D, check if all elements in a row are zero
                zero_dofs = np.all(np.isclose(matrix, 0, atol=tol), axis=1)
        else:
            # Check for zero rows
            zero_rows = np.all(np.isclose(matrix, 0, atol=tol), axis=1)
            # Check for zero columns
            zero_cols = np.all(np.isclose(matrix, 0, atol=tol), axis=0)
            # A DOF is zero if either its row or column is zero
            zero_dofs = zero_rows | zero_cols
        return zero_dofs

    # Identify zero DOFs in each matrix
    zero_dofs_mass = identify_zero_dofs(mass_matrix)
    zero_dofs_damping = identify_zero_dofs(damping_matrix)
    zero_dofs_stiffness = identify_zero_dofs(stiffness_matrix)
    zero_dofs_forcing = identify_zero_dofs(forcing_matrix, is_forcing=True)

    # Combine all zero DOFs across all matrices
    dofs_to_remove = zero_dofs_mass | zero_dofs_damping | zero_dofs_stiffness | zero_dofs_forcing

    # If no DOFs to remove, return original matrices and active_dofs as all True
    if not np.any(dofs_to_remove):
        active_dofs = np.ones(mass_matrix.shape[0], dtype=bool)
        return mass_matrix, damping_matrix, stiffness_matrix, forcing_matrix, active_dofs

    # Identify active DOFs (those not to be removed)
    active_dofs = ~dofs_to_remove

    # Remove DOFs
    mass_matrix_reduced = mass_matrix[active_dofs][:, active_dofs]
    damping_matrix_reduced = damping_matrix[active_dofs][:, active_dofs]
    stiffness_matrix_reduced = stiffness_matrix[active_dofs][:, active_dofs]

    # Handle forcing_matrix reduction
    if forcing_matrix.ndim == 1:
        # If forcing_matrix is a vector, simply remove the entries corresponding to removed DOFs
        forcing_matrix_reduced = forcing_matrix[active_dofs]
    elif forcing_matrix.ndim == 2:
        # If forcing_matrix is a matrix, remove the rows corresponding to removed DOFs
        forcing_matrix_reduced = forcing_matrix[active_dofs, :]
    else:
        raise ValueError("forcing_matrix must be either a 1D or 2D array.")

    return mass_matrix_reduced, damping_matrix_reduced, stiffness_matrix_reduced, forcing_matrix_reduced, active_dofs

def safe_structure(key, value, ensure_serializable=True, recursive=True, tol=1e-8):
    """
    Safely structure nested outputs for JSON-like results.

    Parameters:
        key (str): The key for the structured output.
        value (any): The value to be structured.
        ensure_serializable (bool): If True, ensures all data is JSON serializable.
        recursive (bool): If True, applies the structuring recursively to nested structures.
        tol (float): Tolerance for determining if numeric values are close to zero (used for custom handling if needed).

    Returns:
        structured_output (dict): Structured dictionary.
    """
    import collections.abc

    def serialize(obj):
        """
        Helper function to serialize objects into JSON-compatible formats.

        Parameters:
            obj (any): The object to serialize.

        Returns:
            Serialized object.
        """
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray, list, tuple, set)):
            return [serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            # For objects not serializable by default, convert to string
            return str(obj)

    def process(value):
        """
        Recursively process the value to ensure it's structured appropriately.

        Parameters:
            value (any): The value to process.

        Returns:
            Processed value.
        """
        if isinstance(value, dict):
            if recursive:
                return {str(k): process(v) for k, v in value.items()}
            else:
                return {str(k): v for k, v in value.items()}
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            if recursive:
                return [process(item) for item in value]
            else:
                try:
                    return list(value)
                except TypeError:
                    return value
        else:
            return value

    # Initial structuring based on the type of value
    if isinstance(value, dict):
        if recursive:
            structured_value = process(value)
        else:
            structured_value = {str(k): v for k, v in value.items()}
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        if recursive:
            structured_value = process(value)
        else:
            try:
                structured_value = list(value)
            except TypeError:
                structured_value = value
    else:
        structured_value = value

    # Apply serialization if required
    if ensure_serializable:
        structured_value = serialize(structured_value)

    return {str(key): structured_value}

def process_mass(a_mass, omega):
    """
    Process the mass response to extract relevant criteria.

    Parameters:
        a_mass (ndarray): Complex response of the mass across frequencies.
        omega (ndarray): Array of frequency values (rad/s).

    Returns:
        mass_output (dict): Dictionary containing processed criteria.
    """
    # Calculate the magnitude of the response
    a_mag = np.abs(a_mass)

    # Identify peaks in the magnitude response
    peaks, _ = find_peaks(a_mag)
    peak_positions = omega[peaks]
    peak_values = a_mag[peaks]

    # Calculate bandwidths between peaks
    bandwidths = {}
    for i in range(len(peak_positions)):
        for j in range(i + 1, len(peak_positions)):
            bandwidth_name = f'bandwidth_{i+1}_{j+1}'
            bandwidths[bandwidth_name] = peak_positions[j] - peak_positions[i]

    # Calculate area under the curve for magnitude
    if len(a_mag) > 0:
        area_under_curve = simpson(a_mag, x=omega)
    else:
        area_under_curve = np.nan

    # Calculate slopes between peaks
    slopes, slope_max = calculate_slopes(peak_positions, peak_values)

    # Limit to top 5 peaks if more than 5 exist
    if len(peak_positions) > 5:
        sorted_indices = np.argsort(peak_values)[-5:]
        peak_positions = peak_positions[sorted_indices]
        peak_values = peak_values[sorted_indices]
        slopes, slope_max = calculate_slopes(peak_positions, peak_values)

    # Create dictionaries for peak positions and values
    peak_positions_dict = {f'peak_position_{i+1}': pos for i, pos in enumerate(peak_positions)}
    peak_values_dict = {f'peak_value_{i+1}': val for i, val in enumerate(peak_values)}

    # Structure the output
    mass_output = {
        **safe_structure('peak_positions', peak_positions_dict),
        **safe_structure('peak_values', peak_values_dict),
        **safe_structure('bandwidths', bandwidths),
        **safe_structure('area_under_curve', area_under_curve),
        **safe_structure('slopes', slopes),
        **safe_structure('slope_max', slope_max),
        'magnitude': a_mag
    }

    return mass_output

import matplotlib.pyplot as plt

class DraggableAnnotation:
    def __init__(self, annotation):
        """
        Initialize the draggable annotation.
        
        Parameters:
            annotation (matplotlib.text.Annotation): The annotation to make draggable.
        """
        self.annotation = annotation
        self.press = None
        self.annotation.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.annotation.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.annotation.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def on_press(self, event):
        """Handle the mouse button press event."""
        if event.inaxes != self.annotation.axes:
            return
        contains, _ = self.annotation.contains(event)
        if not contains:
            return
        x0, y0 = self.annotation.get_position()
        self.press = (x0, y0, event.xdata, event.ydata)
    
    def on_motion(self, event):
        """Handle the mouse movement event."""
        if self.press is None or event.inaxes != self.annotation.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        new_x = x0 + dx
        new_y = y0 + dy
        self.annotation.set_position((new_x, new_y))
        self.annotation.figure.canvas.draw()
    
    def on_release(self, event):
        """Handle the mouse button release event."""
        self.press = None
        self.annotation.figure.canvas.draw()

def plot_mass_response(mass_label, omega, mass_data, show_peaks=False, show_slopes=False,
                       figsize=(10, 6), color=None, alpha_fill=0.2, font_size=10, font_style='normal'):
    """
    Plot the frequency response of a single mass with optional draggable annotations.
    
    Parameters:
        mass_label (str): Label of the mass (e.g., 'Primary Mass 1').
        omega (ndarray): Frequency array.
        mass_data (dict): Mass data containing magnitude and processed results.
        show_peaks (bool): Whether to annotate peaks on the plot.
        show_slopes (bool): Whether to annotate slopes between peaks.
        figsize (tuple): Size of the figure (width, height).
        color (str or tuple): Color of the plot line. If None, a default color is used.
        alpha_fill (float): Transparency level for the area under the curve.
        font_size (int): Font size for annotations.
        font_style (str): Font style for annotations (e.g., 'normal', 'italic', 'bold').
    """
    plt.figure(figsize=figsize)
    
    # Assign color if not provided
    if color is None:
        color = 'C0'  # Default matplotlib color cycle
    
    # Plot the frequency response function
    a_mag = mass_data.get('magnitude', np.zeros_like(omega))
    line, = plt.plot(omega, a_mag, label=mass_label, linewidth=2, color=color)
    
    # Highlight the area under the curve
    plt.fill_between(omega, a_mag, color=color, alpha=alpha_fill)
    
    draggable_annotations = []
    
    if show_peaks:
        # Extract peak positions and values
        peak_positions = [mass_data['peak_positions'].get(f'peak_position_{i+1}', 0) 
                          for i in range(len(mass_data.get('peak_positions', {})))]
        peak_values = [mass_data['peak_values'].get(f'peak_value_{i+1}', 0) 
                       for i in range(len(mass_data.get('peak_values', {})))]
        
        # Sort peaks by amplitude and select top 3
        sorted_peaks = sorted(zip(peak_values, peak_positions), key=lambda x: x[0], reverse=True)[:3]
        
        for i, (val, pos) in enumerate(sorted_peaks, 1):
            # Plot the peak point
            scatter = plt.scatter(pos, val, color='darkred', s=100, zorder=5, edgecolor='black')
            
            # Create and add the annotation
            annotation_text = f'Peak {i}\nFreq: {pos:.2f} rad/s\nAmp: {val:.2e}'
            annotation = plt.annotate(
                annotation_text,
                xy=(pos, val),
                xytext=(0, 20 if i % 2 == 1 else -30),
                textcoords='offset points',
                ha='center',
                va='bottom' if i % 2 == 1 else 'top',
                fontsize=font_size,
                fontstyle=font_style,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black')
            )
            
            # Make the annotation draggable
            draggable = DraggableAnnotation(annotation)
            draggable_annotations.append(draggable)
    
    # Set axis labels and title
    plt.xlabel('Frequency (rad/s)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.title(f'Frequency Response of {mass_label}', fontsize=18, weight='bold')
    
    # Add gridlines
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Configure and place legend
    plt.legend(fontsize=12, loc='upper right')
    
    # Adjust tick parameters
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_all_mass_responses(omega, mass_data_list, mass_labels, show_peaks=False, show_slopes=False,
                            figsize=(16, 10), alpha_fill=0.1, font_size=12, font_style='normal'):
    """
    Plot the frequency responses of multiple masses in a single plot with an internal, enlarged legend.
    
    Parameters:
        omega (ndarray): Frequency array.
        mass_data_list (list of dict): List containing mass responses and results.
        mass_labels (list of str): List of mass labels (e.g., ['mass_1', 'mass_2', ...]).
        show_peaks (bool): Whether to annotate peaks on the plots.
        show_slopes (bool): Whether to annotate slopes between peaks in the plots.
        figsize (tuple): Size of the figure (width, height).
        alpha_fill (float): Transparency level for the area under the curves.
        font_size (int): Font size for annotations and legend.
        font_style (str): Font style for annotations and legend (e.g., 'normal', 'italic', 'bold').
    """
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    import numpy as np

    # Create figure and axis
    fig, ax_plot = plt.subplots(figsize=figsize)
    
    # Define a color cycle using 'tab10' colormap for distinct colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(mass_labels))]
    
    # Plot the frequency responses
    for idx, (mass_label, mass_data) in enumerate(zip(mass_labels, mass_data_list)):
        color = colors[idx]
        a_mag = mass_data.get('magnitude', np.zeros_like(omega))
        ax_plot.plot(omega, a_mag, label=mass_label, linewidth=2, color=color)
        ax_plot.fill_between(omega, a_mag, color=color, alpha=alpha_fill)
        
        if show_peaks:
            # Extract peak positions and values
            peak_positions = [mass_data['peak_positions'].get(f'peak_position_{i+1}', 0)
                              for i in range(len(mass_data.get('peak_positions', {})))]
            peak_values = [mass_data['peak_values'].get(f'peak_value_{i+1}', 0)
                           for i in range(len(mass_data.get('peak_values', {})))]
            
            # Sort peaks by amplitude and select top 3
            sorted_peaks = sorted(zip(peak_values, peak_positions), key=lambda x: x[0], reverse=True)[:3]
            
            for i, (val, pos) in enumerate(sorted_peaks, 1):
                # Plot the peak point
                ax_plot.scatter(pos, val, color='darkred', s=100, zorder=5, edgecolor='black')
                
                # Determine annotation position
                offset = 20 if i % 2 == 1 else -30
                va = 'bottom' if i % 2 == 1 else 'top'
                
                # Create annotation
                annotation = ax_plot.annotate(
                    f'Peak {i}\nFreq: {pos:.3f} rad/s\nAmp: {val:.3e}',
                    xy=(pos, val),
                    xytext=(0, offset),
                    textcoords='offset points',
                    ha='center',
                    va=va,
                    fontsize=font_size,
                    fontstyle=font_style,
                    fontname='Times New Roman',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black')
                )
                
                # Make the annotation draggable
                draggable = DraggableAnnotation(annotation)
    
    # Set axis labels and title with Times New Roman font
    ax_plot.set_xlabel('Frequency (rad/s)', fontsize=14, fontname='Times New Roman')
    ax_plot.set_ylabel('Amplitude', fontsize=14, fontname='Times New Roman')
    ax_plot.set_title('Combined Frequency Responses of All Masses', fontsize=18, weight='bold', fontname='Times New Roman')
    
    # Add gridlines
    ax_plot.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Configure and place legend inside the plot area
    handles, labels = ax_plot.get_legend_handles_labels()
    
    # Determine number of columns for the legend based on the number of labels
    if len(labels) <= 3:
        ncol = 1
    elif len(labels) <= 6:
        ncol = 2
    else:
        ncol = 3
    
    legend = ax_plot.legend(handles, labels, fontsize=font_size, loc='upper right', ncol=ncol,
                            prop={'family': 'Times New Roman', 'size': font_size},
                            frameon=True, fancybox=True, shadow=True)
    
    # Optionally, increase legend font size further by iterating over text objects
    for text in legend.get_texts():
        text.set_fontsize(font_size)
        text.set_fontname('Times New Roman')
        text.set_fontstyle(font_style)
    
    # Adjust tick parameters with Times New Roman font
    ax_plot.tick_params(axis='both', which='major', labelsize=12)
    for label in (ax_plot.get_xticklabels() + ax_plot.get_yticklabels()):
        label.set_fontname('Times New Roman')
    
    # Improve layout to fit the plot neatly within the figure
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def calculate_composite_measure(mass_key, results, target_values, weights):
    """
    Calculate the composite measure for a given mass based on target values and weights.

    Parameters:
        mass_key (str): The key for the mass in the results dictionary.
        results (dict): The results dictionary containing mass data.
        target_values (dict): Target values for each criterion.
        weights (dict): Weights for each criterion.

    Returns:
        composite (float): The composite measure for the mass.
    """
    composite = 0.0
    mass_results = results.get(mass_key, {})
    if not mass_results:
        return composite

    for criterion, target in target_values.items():
        weight = weights.get(criterion, 0.0)
        if weight == 0.0:
            continue  # Skip criteria with zero weight

        actual = None
        if criterion.startswith('peak_position'):
            # Typically, peak positions might not be used directly in composite measures
            continue
        elif criterion.startswith('peak_value'):
            actual = mass_results['peak_values'].get(criterion, 0.0)
        elif criterion.startswith('bandwidth'):
            actual = mass_results['bandwidths'].get(criterion, 0.0)
        elif criterion.startswith('slope'):
            actual = mass_results['slopes'].get(criterion, 0.0)
        elif criterion == 'area_under_curve':
            actual = mass_results.get(criterion, 0.0)
        elif criterion == 'slope_max':
            actual = mass_results.get(criterion, 0.0)
        else:
            # Handle any other criteria if necessary
            actual = mass_results.get(criterion, 0.0)

        if actual is None:
            continue

        if target != 0:
            composite += weight * (actual / target)
        else:
            continue  # Skip division by zero

    return composite

def calculate_singular_response(results, target_values_dict, weights_dict):
    """
    Calculate the composite measure for all masses and the singular response.

    Parameters:
        results (dict): The results dictionary containing mass data.
        target_values_dict (dict): Dictionary containing target values for each mass.
        weights_dict (dict): Dictionary containing weights for each mass.

    Returns:
        results (dict): The updated results dictionary containing composite measures and singular response.
    """
    composite_measures = {}

    # Iterate over all masses for which we have target values and weights
    for mass_key in target_values_dict.keys():
        target_values = target_values_dict[mass_key]
        weights = weights_dict.get(mass_key, {})
        
        # Calculate composite measure for the current mass
        composite_mass = calculate_composite_measure(mass_key, results, target_values, weights)
        composite_measures[mass_key] = composite_mass

    # Calculate the singular response as the sum of all composite measures
    singular_response = sum(composite_measures.values())

    # Store the composite measures and singular response in results
    results['composite_measures'] = composite_measures
    results['singular_response'] = singular_response

    return results

# --- Main FRF Function ---

def frf(main_system_parameters, dva_parameters, omega_start, omega_end, omega_points,
        target_values_mass1, weights_mass1, target_values_mass2, weights_mass2,
        target_values_mass3, weights_mass3, target_values_mass4, weights_mass4,
        target_values_mass5, weights_mass5,
        plot_figure=False, show_peaks=False, show_slopes=False):
    """
    Calculate the Frequency Response Function (FRF) and compute a singular response based on weighted, normalized criteria.

    Parameters:
        main_system_parameters (list or tuple): Parameters for the main system.
        dva_parameters (list or tuple): Parameters for the Dynamic Vibration Absorber (DVA).
        omega_start (float): Starting frequency (rad/s).
        omega_end (float): Ending frequency (rad/s).
        omega_points (int): Number of frequency points.
        target_values_mass1 (dict): Target values for each criterion for mass_1.
        weights_mass1 (dict): Weights for each criterion for mass_1.
        target_values_mass2 (dict): Target values for each criterion for mass_2.
        weights_mass2 (dict): Weights for each criterion for mass_2.
        target_values_mass3 (dict): Target values for each criterion for mass_3.
        weights_mass3 (dict): Weights for each criterion for mass_3.
        target_values_mass4 (dict): Target values for each criterion for mass_4.
        weights_mass4 (dict): Weights for each criterion for mass_4.
        target_values_mass5 (dict): Target values for each criterion for mass_5.
        weights_mass5 (dict): Weights for each criterion for mass_5.
        plot_figure (bool): Whether to plot the frequency response.
        show_peaks (bool): Whether to annotate peaks in the plots.
        show_slopes (bool): Whether to annotate slopes between peaks in the plots.

    Returns:
        results (dict): Dictionary containing processed responses, composite measures, and the singular response.
    """
    # Unpack main system parameters
    MU, LANDA_1, LANDA_2, LANDA_3, LANDA_4, LANDA_5, NU_1, NU_2, NU_3, NU_4, NU_5, \
    A_LOW, A_UPP, F_1, F_2, OMEGA_DC, ZETA_DC = main_system_parameters

    # Unpack DVA parameters
    (
        beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8, beta_9, beta_10,
        beta_11, beta_12, beta_13, beta_14, beta_15,
        lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
        lambda_11, lambda_12, lambda_13, lambda_14, lambda_15,
        mu_1, mu_2, mu_3,
        nu_1, nu_2, nu_3, nu_4, nu_5, nu_6, nu_7, nu_8, nu_9, nu_10,
        nu_11, nu_12, nu_13, nu_14, nu_15
    ) = dva_parameters

    # Frequency range
    omega = np.linspace(omega_start, omega_end, omega_points)
    Omega = omega   # Dimensional frequency

    # --- Define Mass Matrix ---
    mass_matrix = np.array([
    [1 + beta_1 + beta_2 + beta_3, 0, -beta_1, -beta_2, -beta_3],
    [0, MU + beta_4 + beta_5 + beta_6, -beta_4, -beta_5, -beta_6],
    [-beta_1, -beta_4, mu_1 + beta_1 + beta_4 + beta_7 + beta_8 + beta_10 + beta_9, -beta_9, -beta_10],
    [-beta_2, -beta_5, -beta_9, mu_2 + beta_11 + beta_2 + beta_9 + beta_12 + beta_5 + beta_15, -beta_15],
    [-beta_3, -beta_6, -beta_10, -beta_15, mu_3 + beta_14 + beta_6 + beta_13 + beta_3 + beta_15 + beta_10]
    ])


    # --- Define Damping Matrix (without pre-multiplying) ---
    damping_matrix_raw = 2 * ZETA_DC * OMEGA_DC * np.array([
    [1 + nu_1 + nu_2 + nu_3 + NU_1 + NU_2 + NU_3, -NU_3, -nu_1, -nu_2, -nu_3],
    [-NU_3, NU_5 + NU_4 + NU_3 + nu_4 + nu_5 + nu_6, -nu_4, -nu_5, -nu_6],
    [-nu_1, -nu_4, nu_1 + nu_4 + nu_7 + nu_8 + nu_10 + nu_9, -nu_9, -nu_10],
    [-nu_2, -nu_5, -nu_9, nu_11 + nu_2 + nu_9 + nu_12 + nu_5 + nu_15, -nu_15],
    [-nu_3, -nu_6, -nu_10, -nu_15, nu_14 + nu_6 + nu_13 + nu_3 + nu_15 + nu_10]
    ])


    # --- Define Stiffness Matrix (without pre-multiplying) ---
    stiffness_matrix_raw = OMEGA_DC ** 2 * np.array([
    [1 + lambda_1 + lambda_2 + lambda_3 + LANDA_1 + LANDA_2 + LANDA_3, -LANDA_3, -lambda_1, -lambda_2, -lambda_3],
    [-LANDA_3, LANDA_5 + LANDA_4 + LANDA_3 + lambda_4 + lambda_5 + lambda_6, -lambda_4, -lambda_5, -lambda_6],
    [-lambda_1, -lambda_4, lambda_1 + lambda_4 + lambda_7 + lambda_8 + lambda_10 + lambda_9, -lambda_9, -lambda_10],
    [-lambda_2, -lambda_5, -lambda_9, lambda_11 + lambda_2 + lambda_9 + lambda_12 + lambda_5 + lambda_15, -lambda_15],
    [-lambda_3, -lambda_6, -lambda_10, -lambda_15, lambda_14 + lambda_6 + lambda_13 + lambda_3 + lambda_15 + lambda_10]
    ])


    # Define the forcing functions (frequency-dependent)
    f_1_omega = F_1 * np.exp(1j * omega)
    f_2_omega = F_2 * np.exp(1j * omega)

    # Define the external inputs (frequency-dependent)
    u_low = A_LOW * np.exp(1j * omega)
    u_upp = A_UPP * np.exp(1j * omega)

    # Construct the forcing vector (5xN, where N is the number of frequencies)
    f = np.array([
        f_1_omega + 2 * ZETA_DC * OMEGA_DC * (1j * omega * u_low + NU_2 * 1j * omega * u_upp) + OMEGA_DC ** 2 * (u_low + LANDA_2 * u_upp),
        f_2_omega + 2 * ZETA_DC * OMEGA_DC * (NU_4 * 1j * omega * u_low + NU_5 * 1j * omega * u_upp) + OMEGA_DC ** 2 * (LANDA_4 * u_low + LANDA_5 * u_upp),
        beta_7 * (-omega ** 2) * u_low + 2 * ZETA_DC * OMEGA_DC * (nu_7 * 1j * omega * u_low + nu_8 * 1j * omega * u_upp) + OMEGA_DC ** 2 * (lambda_7 * u_low + lambda_8 * u_upp) + beta_8 * (-omega ** 2) * u_upp,
        beta_11 * (-omega ** 2) * u_low + 2 * ZETA_DC * OMEGA_DC * (nu_11 * 1j * omega * u_low + nu_12 * 1j * omega * u_upp) + OMEGA_DC ** 2 * (lambda_11 * u_low + lambda_12 * u_upp) + beta_12 * (-omega ** 2) * u_upp,
        beta_13 * (-omega ** 2) * u_low + 2 * ZETA_DC * OMEGA_DC * (nu_13 * 1j * omega * u_low + nu_14 * 1j * omega * u_upp) + OMEGA_DC ** 2 * (lambda_13 * u_low + lambda_14 * u_upp) + beta_14 * (-omega ** 2) * u_upp
    ])

    # Remove zero DOFs
    mass_matrix_reduced, damping_matrix_reduced, stiffness_matrix_reduced, f_reduced, active_dofs = remove_zero_mass_dofs(
        mass_matrix, damping_matrix_raw, stiffness_matrix_raw, f
    )

    # Check if any DOFs are left after reduction
    if mass_matrix_reduced.size == 0:
        raise ValueError("All degrees of freedom have zero mass. Cannot perform analysis.")

    # Number of active DOFs
    num_dofs = mass_matrix_reduced.shape[0]

    # Initialize the response matrix 'A' (number of active DOFs x number of frequencies)
    A = np.zeros((num_dofs, len(omega)), dtype=complex)

    for i in range(len(omega)):
        Omega_i = Omega[i]
        # hh = -Omega_i^2 [M] + 2j \zeta_{dc} \Omega_i [C] + [K']
        hh = - Omega_i ** 2 * mass_matrix_reduced + 2 * ZETA_DC * Omega_i * damping_matrix_reduced + stiffness_matrix_reduced
        # Multiply hh by OMEGA_DC ** 2
        hh = OMEGA_DC ** 2 * hh
        # Solve for A
        try:
            A[:, i] = np.linalg.solve(hh, f_reduced[:, i])
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Linear algebra error at frequency index {i}: {e}")
        # Multiply A by OMEGA_DC ** 2
        A[:, i] = OMEGA_DC ** 2 * A[:, i]

    # Process results for each active mass
    results = {}
    dof_indices = np.where(active_dofs)[0]  # Indices of active DOFs in the original system

    # Map DOF indices to mass labels
    mass_labels_map = {
        0: 'mass_1',  # Main system mass 1
        1: 'mass_2',  # Main system mass 2
        2: 'mass_3',  # DVA mass 1
        3: 'mass_4',  # DVA mass 2
        4: 'mass_5'   # DVA mass 3
    }

    mass_data_list = []  # Collect data for plotting
    mass_labels_list = []  # Collect labels for plotting

    for idx, dof in enumerate(dof_indices):
        # Get the label for this mass
        mass_label = mass_labels_map.get(dof, f'mass_{dof+1}')

        # Process the mass response
        mass_results = process_mass(A[idx, :], omega)
        results[mass_label] = mass_results

        # Collect data for plotting
        mass_data_list.append(mass_results)
        mass_labels_list.append(mass_label)

    # Plotting if requested
    if plot_figure:
        # Plot individual mass responses
        for mass_label, mass_data in zip(mass_labels_list, mass_data_list):
            plot_mass_response(mass_label, omega, mass_data, show_peaks, show_slopes)
        # Plot all masses together
        plot_all_mass_responses(Omega, mass_data_list, mass_labels_list, show_peaks, show_slopes)

        # --- New Feature: Plot Without DVAs for Mass 1 and Mass 2 ---
        if show_slopes:
            # Define DVA parameters as zero
            dva_parameters_zero = [0]*len(dva_parameters)  # All DVA parameters set to zero

            # Recompute FRF with DVAs set to zero
            # To avoid recursive plotting, set plot_figure to False
            results_without_dva = frf(
                main_system_parameters,
                dva_parameters_zero,
                omega_start, omega_end, omega_points,
                target_values_mass1, weights_mass1,
                target_values_mass2, weights_mass2,
                target_values_mass3, weights_mass3,
                target_values_mass4, weights_mass4,
                target_values_mass5, weights_mass5,
                plot_figure=False,  # Avoid plotting in recursive call
                show_peaks=show_peaks,
                show_slopes=show_slopes
            )

            # Extract mass data without DVAs
            mass_data_list_without_dva = []
            mass_labels_list_without_dva = []

            for mass_label in mass_labels_list:
                # Only extract Mass 1 and Mass 2 from the without DVA results
                if mass_label in ['mass_1', 'mass_2']:
                    mass_data = results_without_dva.get(mass_label, {})
                    mass_data_list_without_dva.append(mass_data)
                    mass_labels_list_without_dva.append(mass_label)

            # Plot combined frequency responses: with and without DVAs for Mass 1 and Mass 2
            plt.figure(figsize=(16, 10))
            cmap = plt.get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(mass_labels_list))]

            for idx, (mass_label, mass_data) in enumerate(zip(mass_labels_list, mass_data_list)):
                color = colors[idx]
                # Plot with DVA
                a_mag_with_dva = mass_data.get('magnitude', np.zeros_like(omega))
                plt.plot(omega, a_mag_with_dva, label=f'{mass_label} with DVA', linewidth=2, color=color, linestyle='-')
                plt.fill_between(omega, a_mag_with_dva, color=color, alpha=0.1)

                if mass_label in ['mass_1', 'mass_2']:
                    # Plot without DVA
                    a_mag_without_dva = mass_data_list_without_dva[mass_labels_list_without_dva.index(mass_label)].get('magnitude', np.zeros_like(omega))
                    plt.plot(omega, a_mag_without_dva, label=f'{mass_label} without DVA', linewidth=2, color=color, linestyle='--')
                    plt.fill_between(omega, a_mag_without_dva, color=color, alpha=0.05)

            # Set axis labels and title
            plt.xlabel('Frequency (rad/s)', fontsize=14)
            plt.ylabel('Amplitude', fontsize=14)
            plt.title('Frequency Responses of All Masses: With and Without DVAs for Mass 1 and Mass 2', fontsize=18, weight='bold')

            # Add gridlines
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Configure and place legend
            plt.legend(fontsize=12, loc='upper right')

            # Adjust tick parameters
            plt.tick_params(axis='both', which='major', labelsize=12)

            # Improve layout
            plt.tight_layout()

            # Show the plot
            plt.show()

    # Build target values and weights dictionaries for five masses
    target_values_dict = {
        'mass_1': target_values_mass1,
        'mass_2': target_values_mass2,
        'mass_3': target_values_mass3,
        'mass_4': target_values_mass4,
        'mass_5': target_values_mass5
    }

    weights_dict = {
        'mass_1': weights_mass1,
        'mass_2': weights_mass2,
        'mass_3': weights_mass3,
        'mass_4': weights_mass4,
        'mass_5': weights_mass5
    }

    # Calculate singular response
    results = calculate_singular_response(results, target_values_dict, weights_dict)

    # Return the results
    return results
