import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def calculate_slopes(peak_positions, peak_values):
    """
    Calculate the slopes between every pair of peaks in a frequency response function.
    """
    slopes = {}
    slope_max = np.nan
    num_peaks = len(peak_positions)
    
    if num_peaks > 1:
        for i in range(num_peaks):
            for j in range(i + 1, num_peaks):
                dx = peak_positions[j] - peak_positions[i]
                slope = (peak_values[j] - peak_values[i]) / dx if dx != 0 else 0.0
                slopes[f"slope_{i+1}_{j+1}"] = slope
                if np.isnan(slope_max) or abs(slope) > abs(slope_max):
                    slope_max = slope
    
    return slopes, slope_max

# -----------------------------------------------------------------------------
# Interpolation functions
# -----------------------------------------------------------------------------

def apply_interpolation(x, y, method='cubic', num_points=1000):
    """
    Apply various interpolation methods to smooth frequency response functions.
    """
    from scipy import signal
    from scipy.interpolate import (
        interp1d, Akima1DInterpolator, PchipInterpolator, 
        BarycentricInterpolator, Rbf, UnivariateSpline,
        BSpline, splrep
    )
    
    if len(x) < 4:
        if len(x) < 2:
            return x, y
        method = 'linear'
    
    x_new = np.linspace(min(x), max(x), num_points)
    
    if method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
        f = interp1d(x, y, kind=method, bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    elif method == 'akima':
        f = Akima1DInterpolator(x, y)
        y_new = f(x_new)
    elif method == 'pchip':
        f = PchipInterpolator(x, y)
        y_new = f(x_new)
    elif method == 'barycentric':
        f = BarycentricInterpolator(x, y)
        y_new = f(x_new)
    elif method == 'rbf':
        f = Rbf(x, y, function='multiquadric')
        y_new = f(x_new)
    elif method == 'smoothing_spline':
        s = len(x) * 0.01
        f = UnivariateSpline(x, y, s=s)
        y_new = f(x_new)
    elif method == 'bspline':
        k = min(3, len(x) - 1)
        t, c, k = splrep(x, y, k=k)
        y_new = BSpline(t, c, k)(x_new)
    elif method == 'savgol':
        window_length = min(9, len(y) - 2 if len(y) % 2 == 0 else len(y) - 1)
        if window_length < 3: window_length = 3
        if window_length % 2 == 0: window_length -= 1
        poly_order = min(3, window_length - 1)
        y_smooth = signal.savgol_filter(y, window_length, poly_order)
        f = interp1d(x, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    elif method == 'moving_average':
        window_size = min(5, len(y))
        weights = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, weights, mode='valid')
        start_idx = window_size // 2
        end_idx = -(window_size // 2) if window_size > 1 else None
        x_smooth = x[start_idx:end_idx]
        if len(x_smooth) != len(y_smooth):
            half_window = (window_size - 1) // 2
            y_smooth = np.array([np.mean(y[max(0, i-half_window):min(len(y), i+half_window+1)]) 
                               for i in range(len(y))])
            x_smooth = x
        f = interp1d(x_smooth, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    elif method == 'gaussian':
        sigma = 2
        y_smooth = signal.gaussian_filter1d(y, sigma)
        f = interp1d(x, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    elif method == 'bessel':
        b, a = signal.bessel(4, 0.1, 'low')
        y_smooth = signal.filtfilt(b, a, y)
        f = interp1d(x, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    else:
        f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    
    return x_new, y_new

INTERPOLATION_METHODS = [
    'linear', 'cubic', 'quadratic', 'nearest', 'akima', 'pchip', 
    'smoothing_spline', 'bspline', 'savgol', 'moving_average', 
    'gaussian', 'bessel', 'barycentric', 'rbf'
]

def remove_zero_mass_dofs(
    mass_matrix,
    damping_matrix,
    stiffness_matrix,
    forcing_matrix,
    *,
    tol: float = 1e-8,
):
    """Remove degrees of freedom that are *truly* inactive."""
    def _zero_dofs(mat, *, is_forcing=False):
        if is_forcing:
            if mat.ndim == 1:
                return np.isclose(mat, 0, atol=tol)
            return np.all(np.isclose(mat, 0, atol=tol), axis=1)
        rows = np.all(np.isclose(mat, 0, atol=tol), axis=1)
        cols = np.all(np.isclose(mat, 0, atol=tol), axis=0)
        return rows | cols

    z_mass = _zero_dofs(mass_matrix)
    z_damp = _zero_dofs(damping_matrix)
    z_stif = _zero_dofs(stiffness_matrix)
    z_force = _zero_dofs(forcing_matrix, is_forcing=True)

    dofs_to_remove = z_mass | (z_damp & z_stif & z_force)
    active_dofs = ~dofs_to_remove

    if not np.any(dofs_to_remove):
        return mass_matrix, damping_matrix, stiffness_matrix, forcing_matrix, active_dofs

    mm = mass_matrix[active_dofs][:, active_dofs]
    cc = damping_matrix[active_dofs][:, active_dofs]
    kk = stiffness_matrix[active_dofs][:, active_dofs]

    if forcing_matrix.ndim == 1:
        ff = forcing_matrix[active_dofs]
    elif forcing_matrix.ndim == 2:
        ff = forcing_matrix[active_dofs, :]
    else:
        raise ValueError("forcing_matrix must be 1‑ or 2‑D array")

    return mm, cc, kk, ff, active_dofs

def safe_structure(key, value, ensure_serializable=True, recursive=True, tol=1e-8):
    def serialize(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray, list, tuple, set)):
            return [serialize(o) for o in obj]
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def process(val):
        if isinstance(val, dict):
            return {str(k): process(v) for k, v in val.items()} if recursive else {str(k): v for k, v in val.items()}
        if isinstance(val, (list, tuple, set, np.ndarray)):
            return [process(v) for v in val] if recursive else list(val)
        return val

    structured = process(value)
    if ensure_serializable:
        structured = serialize(structured)
    return {str(key): structured}

def find_significant_peaks(magnitude, omega, min_prominence_ratio=0.05, max_peaks=5):
    if len(magnitude) < 3: return np.array([]), np.array([])
    peaks, _ = find_peaks(magnitude)
    if len(peaks) == 0: return np.array([]), np.array([])
    prominences = peak_prominences(magnitude, peaks)[0]
    max_magnitude = np.max(magnitude)
    min_prominence = max_magnitude * min_prominence_ratio
    significant_peaks = peaks[prominences >= min_prominence]
    significant_values = magnitude[significant_peaks]
    if len(significant_peaks) > max_peaks:
        idx = np.argsort(significant_values)[-max_peaks:]
        significant_peaks = significant_peaks[idx]
        significant_values = significant_values[idx]
    return omega[significant_peaks], significant_values

def interpolate_peak_vicinity(magnitude, omega, peak_indices, vicinity_ratio=0.05):
    if len(peak_indices) == 0 or len(magnitude) < 3: return np.array([]), np.array([])
    interp_func = interp1d(omega, magnitude, kind='cubic', bounds_error=False, fill_value='extrapolate')
    omega_range = omega[-1] - omega[0]
    vicinity_width = omega_range * vicinity_ratio
    refined_peak_positions = []
    refined_peak_values = []
    for idx in peak_indices:
        peak_pos = omega[idx]
        vicinity_start = max(omega[0], peak_pos - vicinity_width/2)
        vicinity_end = min(omega[-1], peak_pos + vicinity_width/2)
        high_res_omega = np.linspace(vicinity_start, vicinity_end, 100)
        high_res_mag = interp_func(high_res_omega)
        max_idx = np.argmax(high_res_mag)
        refined_peak_positions.append(high_res_omega[max_idx])
        refined_peak_values.append(high_res_mag[max_idx])
    return np.array(refined_peak_positions), np.array(refined_peak_values)

def calculate_robust_slopes(peak_positions, peak_values, slope_threshold=0.01):
    slopes = {}
    slope_max = np.nan
    num_peaks = len(peak_positions)
    if num_peaks <= 1: return slopes, slope_max
    amplitude_range = np.max(peak_values) - np.min(peak_values)
    frequency_range = np.max(peak_positions) - np.min(peak_positions)
    if amplitude_range == 0 or frequency_range == 0: return slopes, slope_max
    for i in range(num_peaks):
        for j in range(i + 1, num_peaks):
            dx = peak_positions[j] - peak_positions[i]
            if dx == 0: continue
            slope = (peak_values[j] - peak_values[i]) / dx
            normalized_slope = slope * (frequency_range / amplitude_range)
            if abs(normalized_slope) >= slope_threshold:
                slopes[f"slope_{i+1}_{j+1}"] = slope
                if np.isnan(slope_max) or abs(slope) > abs(slope_max):
                    slope_max = slope
    return slopes, slope_max

def process_mass(a_mass, omega, user_peak_positions=None):
    a_mag = np.abs(a_mass)
    peak_positions = []
    peak_values = []
    if user_peak_positions is not None:
        user_peak_positions = np.array(user_peak_positions)
        for pos in user_peak_positions:
            idx = np.argmin(np.abs(omega - pos))
            peak_positions.append(omega[idx])
            peak_values.append(a_mag[idx])
    detected_positions, detected_values = find_significant_peaks(a_mag, omega)
    if len(detected_positions) > 0:
        peak_positions.extend(detected_positions)
        peak_values.extend(detected_values)
    peak_positions = np.array(peak_positions)
    peak_values = np.array(peak_values)
    sort_idx = np.argsort(peak_positions)
    peak_positions = peak_positions[sort_idx]
    peak_values = peak_values[sort_idx]
    if len(peak_positions) > 0:
        peak_indices = np.array([np.argmin(np.abs(omega - pos)) for pos in peak_positions])
        peak_positions, peak_values = interpolate_peak_vicinity(a_mag, omega, peak_indices)
    bandwidths = {}
    for i in range(len(peak_positions)):
        for j in range(i + 1, len(peak_positions)):
            bandwidths[f"bandwidth_{i+1}_{j+1}"] = peak_positions[j] - peak_positions[i]
    area_under_curve = simpson(a_mag, x=omega) if len(a_mag) else np.nan
    slopes, slope_max = calculate_robust_slopes(peak_positions, peak_values)
    peaks_dict = {f"peak_position_{i+1}": p for i, p in enumerate(peak_positions)}
    values_dict = {f"peak_value_{i+1}": v for i, v in enumerate(peak_values)}
    return {
        **safe_structure("peak_positions", peaks_dict),
        **safe_structure("peak_values", values_dict),
        **safe_structure("bandwidths", bandwidths),
        **safe_structure("area_under_curve", area_under_curve),
        **safe_structure("slopes", slopes),
        **safe_structure("slope_max", slope_max),
        "magnitude": a_mag,
    }

def calculate_composite_measure(mass_key, results, target_values, weights):
    composite = 0.0
    percentage_differences = {}
    mass_results = results.get(mass_key, {})
    if not mass_results: return composite, percentage_differences
    for criterion, target in target_values.items():
        w = weights.get(criterion, 0.0)
        if w == 0.0: continue
        if criterion.startswith("peak_value"): actual = mass_results["peak_values"].get(criterion, 0.0)
        elif criterion.startswith("peak_position"): actual = mass_results["peak_positions"].get(criterion, 0.0)
        elif criterion.startswith("bandwidth"): actual = mass_results["bandwidths"].get(criterion, 0.0)
        elif criterion.startswith("slope"): actual = mass_results["slopes"].get(criterion, 0.0)
        elif criterion in ("area_under_curve", "slope_max"): actual = mass_results.get(criterion, 0.0)
        else: actual = mass_results.get(criterion, 0.0)
        if target != 0:
            composite += w * (actual / target)
            percent_diff = ((actual - target) / target) * 100
            percentage_differences[criterion] = percent_diff
    return composite, percentage_differences

def calculate_singular_response(results, target_values_dict, weights_dict):
    composite_measures = {}
    percentage_differences = {}
    for m in target_values_dict:
        comp, pdiffs = calculate_composite_measure(m, results, target_values_dict[m], weights_dict.get(m, {}))
        composite_measures[m] = comp
        percentage_differences[m] = pdiffs
    results["composite_measures"] = composite_measures
    results["percentage_differences"] = percentage_differences
    results["singular_response"] = sum(composite_measures.values())
    return results

def frf(
    main_system_parameters,
    dva_parameters,
    omega_start,
    omega_end,
    omega_points,
    target_values_mass1,
    weights_mass1,
    target_values_mass2,
    weights_mass2,
    target_values_mass3,
    weights_mass3,
    target_values_mass4,
    weights_mass4,
    target_values_mass5,
    weights_mass5,
    *,
    user_peak_positions=None,
    interpolation_method='cubic',
    interpolation_points=1000,
):
    MU, LANDA_1, LANDA_2, LANDA_3, LANDA_4, LANDA_5, NU_1, NU_2, NU_3, NU_4, NU_5, A_LOW, A_UPP, F_1, F_2, OMEGA_DC, ZETA_DC = main_system_parameters
    (
        beta_1, beta_2, beta_3, beta_4, beta_5, beta_6, beta_7, beta_8, beta_9, beta_10,
        beta_11, beta_12, beta_13, beta_14, beta_15,
        lambda_1, lambda_2, lambda_3, lambda_4, lambda_5, lambda_6, lambda_7, lambda_8, lambda_9, lambda_10,
        lambda_11, lambda_12, lambda_13, lambda_14, lambda_15,
        mu_1, mu_2, mu_3,
        nu_1, nu_2, nu_3, nu_4, nu_5, nu_6, nu_7, nu_8, nu_9, nu_10,
        nu_11, nu_12, nu_13, nu_14, nu_15,
    ) = dva_parameters

    omega = np.linspace(omega_start, omega_end, omega_points)
    Omega = omega

    mass_matrix = np.array([
        [1 + beta_1 + beta_2 + beta_3, 0, -beta_1, -beta_2, -beta_3],
        [0, MU + beta_4 + beta_5 + beta_6, -beta_4, -beta_5, -beta_6],
        [-beta_1, -beta_4, mu_1 + beta_1 + beta_4 + beta_7 + beta_8 + beta_10 + beta_9, -beta_9, -beta_10],
        [-beta_2, -beta_5, -beta_9, mu_2 + beta_11 + beta_2 + beta_9 + beta_12 + beta_5 + beta_15, -beta_15],
        [-beta_3, -beta_6, -beta_10, -beta_15, mu_3 + beta_14 + beta_6 + beta_13 + beta_3 + beta_15 + beta_10],
    ])

    damping_matrix_raw = 2 * ZETA_DC * OMEGA_DC * np.array([
        [1 + nu_1 + nu_2 + nu_3 + NU_1 + NU_2 + NU_3, -NU_3, -nu_1, -nu_2, -nu_3],
        [-NU_3, NU_5 + NU_4 + NU_3 + nu_4 + nu_5 + nu_6, -nu_4, -nu_5, -nu_6],
        [-nu_1, -nu_4, nu_1 + nu_4 + nu_7 + nu_8 + nu_10 + nu_9, -nu_9, -nu_10],
        [-nu_2, -nu_5, -nu_9, nu_11 + nu_2 + nu_9 + nu_12 + nu_5 + nu_15, -nu_15],
        [-nu_3, -nu_6, -nu_10, -nu_15, nu_14 + nu_6 + nu_13 + nu_3 + nu_15 + nu_10],
    ])

    stiffness_matrix_raw = OMEGA_DC**2 * np.array([
        [1 + lambda_1 + lambda_2 + lambda_3 + LANDA_1 + LANDA_2 + LANDA_3, -LANDA_3, -lambda_1, -lambda_2, -lambda_3],
        [-LANDA_3, LANDA_5 + LANDA_4 + LANDA_3 + lambda_4 + lambda_5 + lambda_6, -lambda_4, -lambda_5, -lambda_6],
        [-lambda_1, -lambda_4, lambda_1 + lambda_4 + lambda_7 + lambda_8 + lambda_10 + lambda_9, -lambda_9, -lambda_10],
        [-lambda_2, -lambda_5, -lambda_9, lambda_11 + lambda_2 + lambda_9 + lambda_12 + lambda_5 + lambda_15, -lambda_15],
        [-lambda_3, -lambda_6, -lambda_10, -lambda_15, lambda_14 + lambda_6 + lambda_13 + lambda_3 + lambda_15 + lambda_10],
    ])

    f_1_omega = F_1 * np.exp(1j * omega)
    f_2_omega = F_2 * np.exp(1j * omega)
    u_low = A_LOW * np.exp(1j * omega)
    u_upp = A_UPP * np.exp(1j * omega)

    f = np.array([
        f_1_omega + 2 * ZETA_DC * OMEGA_DC * (1j * omega * u_low + NU_2 * 1j * omega * u_upp) + OMEGA_DC**2 * (u_low + LANDA_2 * u_upp),
        f_2_omega + 2 * ZETA_DC * OMEGA_DC * (NU_4 * 1j * omega * u_low + NU_5 * 1j * omega * u_upp) + OMEGA_DC**2 * (LANDA_4 * u_low + LANDA_5 * u_upp),
        beta_7 * (-omega**2) * u_low + 2 * ZETA_DC * OMEGA_DC * (nu_7 * 1j * omega * u_low + nu_8 * 1j * omega * u_upp) + OMEGA_DC**2 * (lambda_7 * u_low + lambda_8 * u_upp) + beta_8 * (-omega**2) * u_upp,
        beta_11 * (-omega**2) * u_low + 2 * ZETA_DC * OMEGA_DC * (nu_11 * 1j * omega * u_low + nu_12 * 1j * omega * u_upp) + OMEGA_DC**2 * (lambda_11 * u_low + lambda_12 * u_upp) + beta_12 * (-omega**2) * u_upp,
        beta_13 * (-omega**2) * u_low + 2 * ZETA_DC * OMEGA_DC * (nu_13 * 1j * omega * u_low + nu_14 * 1j * omega * u_upp) + OMEGA_DC**2 * (lambda_13 * u_low + lambda_14 * u_upp) + beta_14 * (-omega**2) * u_upp,
    ])

    mm, cc, kk, f_reduced, active = remove_zero_mass_dofs(mass_matrix, damping_matrix_raw, stiffness_matrix_raw, f)
    if mm.size == 0: raise ValueError("All degrees of freedom have zero mass.")

    n_dofs = mm.shape[0]
    A = np.zeros((n_dofs, len(omega)), dtype=complex)
    def _robust_solve(hmat, rhs):
        try: return np.linalg.solve(hmat, rhs)
        except np.linalg.LinAlgError:
            scale = np.linalg.norm(hmat, ord=np.inf)
            base_eps = (1e-12 if scale == 0 else 1e-12 * scale)
            I = np.eye(hmat.shape[0], dtype=hmat.dtype)
            for mult in (1.0, 1e1, 1e2, 1e3, 1e4):
                try: return np.linalg.solve(hmat + (base_eps * mult) * I, rhs)
                except np.linalg.LinAlgError: continue
            try: return np.linalg.pinv(hmat) @ rhs
            except Exception: return np.linalg.lstsq(hmat, rhs, rcond=None)[0]

    for i, Om in enumerate(Omega):
        hh = -Om**2 * mm + 2 * ZETA_DC * Om * cc + kk
        hh *= OMEGA_DC**2
        A[:, i] = _robust_solve(hh, f_reduced[:, i]) * OMEGA_DC**2

    results = {}
    idxs = np.where(active)[0]
    label_map = {0: "mass_1", 1: "mass_2", 2: "mass_3", 3: "mass_4", 4: "mass_5"}
    for local_idx, dof in enumerate(idxs):
        lbl = label_map.get(dof, f"mass_{dof+1}")
        mass_peaks = user_peak_positions.get(lbl, None) if user_peak_positions else None
        results[lbl] = process_mass(A[local_idx, :], omega, user_peak_positions=mass_peaks)

    target_dict = {"mass_1": target_values_mass1, "mass_2": target_values_mass2, "mass_3": target_values_mass3, "mass_4": target_values_mass4, "mass_5": target_values_mass5}
    weight_dict = {"mass_1": weights_mass1, "mass_2": weights_mass2, "mass_3": weights_mass3, "mass_4": weights_mass4, "mass_5": weights_mass5}
    results = calculate_singular_response(results, target_dict, weight_dict)
    results["interpolation_info"] = {"method": interpolation_method, "points": interpolation_points}
    return results

def perform_omega_points_sensitivity_analysis(
    main_system_parameters,
    dva_parameters,
    omega_start,
    omega_end,
    initial_points=100,
    max_points=1000000000,
    step_size=100,
    convergence_threshold=0.01,
    max_iterations=100,
    mass_of_interest="mass_1",
):
    point_values, slope_max_values, relative_changes = [], [], []
    peak_position_changes, bandwidth_changes, metrics_history = [], [], []
    empty_dict = {}
    prev_metrics, optimal_points, converged, convergence_point, iteration = None, max_points, False, None, 0
    current_points = initial_points

    def _flatten_metrics(mass_dict):
        flat = {}
        for key, val in mass_dict.items():
            if key == "magnitude": continue
            if isinstance(val, dict):
                for subk, subval in val.items():
                    if np.isscalar(subval): flat[f"{key}.{subk}"] = float(subval)
            elif np.isscalar(val): flat[key] = float(val)
            elif isinstance(val, (list, tuple, np.ndarray)):
                for i, subval in enumerate(val):
                    if np.isscalar(subval): flat[f"{key}.{i}"] = float(subval)
        return flat

    while current_points <= max_points and iteration < max_iterations:
        results = frf(
            main_system_parameters=main_system_parameters, dva_parameters=dva_parameters,
            omega_start=omega_start, omega_end=omega_end, omega_points=current_points,
            target_values_mass1=empty_dict, weights_mass1=empty_dict, target_values_mass2=empty_dict, weights_mass2=empty_dict,
            target_values_mass3=empty_dict, weights_mass3=empty_dict, target_values_mass4=empty_dict, weights_mass4=empty_dict,
            target_values_mass5=empty_dict, weights_mass5=empty_dict
        )
        if mass_of_interest in results:
            mass_data = results[mass_of_interest]
            current_metrics = _flatten_metrics(mass_data)
            point_values.append(current_points)
            slope_max_values.append(mass_data.get("slope_max", np.nan))
            metrics_history.append(current_metrics)
            if prev_metrics is not None:
                changes = {k: abs((current_metrics[k] - prev_metrics[k]) / prev_metrics[k]) for k in current_metrics.keys() & prev_metrics.keys() if prev_metrics[k] != 0 and not np.isnan(prev_metrics[k]) and not np.isnan(current_metrics[k])}
                max_change = max(changes.values(), default=np.nan)
                relative_changes.append(max_change)
                peak_position_changes.append(max([v for k, v in changes.items() if k.startswith("peak_positions")], default=np.nan))
                bandwidth_changes.append(max([v for k, v in changes.items() if k.startswith("bandwidths")], default=np.nan))
                if max_change < convergence_threshold and not converged:
                    converged = True
                    convergence_point = current_points
            else:
                relative_changes.append(np.nan)
                peak_position_changes.append(np.nan)
                bandwidth_changes.append(np.nan)
            prev_metrics = current_metrics
        current_points += step_size
        iteration += 1
    
    reached_max = (current_points - step_size) >= max_points
    optimal_points = max_points if reached_max else (point_values[-1] if point_values else initial_points)
    
    return {
        "omega_points": point_values, "max_slopes": slope_max_values, "relative_changes": relative_changes,
        "peak_position_changes": peak_position_changes, "bandwidth_changes": bandwidth_changes,
        "metrics_history": metrics_history, "optimal_points": optimal_points, "converged": converged,
        "convergence_point": convergence_point, "all_points_analyzed": reached_max,
        "requested_max_points": max_points, "highest_analyzed_point": point_values[-1] if point_values else initial_points,
        "iteration_limit_reached": iteration >= max_iterations, "step_size": step_size,
    }

def determine_optimal_omega_points(main_system_parameters, dva_parameters, omega_start, omega_end, **kwargs):
    results = perform_omega_points_sensitivity_analysis(main_system_parameters, dva_parameters, omega_start, omega_end, **kwargs)
    return results["optimal_points"]
