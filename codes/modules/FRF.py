# FRF.py
"""
Only functional change is in `remove_zero_mass_dofs`: a DOF is now removed **only
when it is inactive across *all* system matrices** (or if it truly has zero
mass).  This prevents legitimate degrees‑of‑freedom from being eliminated when,
for example, the damping matrix row/column happens to be zero while the mass or
stiffness is not.  No other behavioural changes have been introduced; the file
structure and public interfaces remain intact so the module can be dropped into
existing codebases as a drop‑in replacement.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from adjustText import adjust_text

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def calculate_slopes(peak_positions, peak_values):
    """
    Calculate the slopes between every pair of peaks in a frequency response function.
    
    Scientific Explanation:
    In frequency response analysis, peaks represent resonant frequencies where the system
    has maximum response. The slope between peaks helps characterize how quickly the
    response changes between resonances. This is important for:
    1. Understanding system dynamics
    2. Identifying modal coupling
    3. Characterizing damping effects
    
    Parameters:
    -----------
    peak_positions : array-like
        The x-coordinates (usually frequencies) where peaks occur
    peak_values : array-like
        The y-coordinates (usually amplitudes) at each peak
    
    Returns:
    --------
    slopes : dict
        Dictionary containing slopes between each pair of peaks
        Keys are formatted as "slope_1_2" for slope between peaks 1 and 2
    slope_max : float
        The maximum absolute slope found between any pair of peaks
    """
    # Initialize empty dictionary to store slopes
    slopes = {}
    
    # Initialize maximum slope as NaN (Not a Number)
    # This is a special value that represents undefined or missing data
    slope_max = np.nan
    
    # Get total number of peaks found
    num_peaks = len(peak_positions)
    
    # Only calculate slopes if we have at least 2 peaks
    if num_peaks > 1:
        # Loop through each peak
        for i in range(num_peaks):
            # For each peak, compare with all subsequent peaks
            # This ensures we don't calculate the same slope twice
            for j in range(i + 1, num_peaks):
                # Calculate horizontal distance between peaks
                dx = peak_positions[j] - peak_positions[i]
                
                # Calculate slope using rise/run formula
                # If dx is 0 (peaks at same x-position), slope is 0
                slope = (peak_values[j] - peak_values[i]) / dx if dx != 0 else 0.0
                
                # Store slope in dictionary with descriptive key
                # Keys are 1-based for user-friendliness (e.g., "slope_1_2")
                slopes[f"slope_{i+1}_{j+1}"] = slope
                
                # Update maximum slope if current slope is larger
                # abs() gives absolute value, so we compare magnitudes
                if np.isnan(slope_max) or abs(slope) > abs(slope_max):
                    slope_max = slope
    
    # Return both the dictionary of slopes and the maximum slope
    return slopes, slope_max

# -----------------------------------------------------------------------------
# Interpolation functions
# -----------------------------------------------------------------------------

def apply_interpolation(x, y, method='cubic', num_points=1000):
    """
    Apply various interpolation methods to smooth frequency response functions.
    
    Parameters:
    -----------
    x : array-like
        The x-coordinates (usually frequencies)
    y : array-like
        The y-coordinates (usually amplitudes)
    method : str
        Interpolation method to use
    num_points : int
        Number of points in the interpolated output
    
    Returns:
    --------
    x_new : ndarray
        New x coordinates (evenly spaced)
    y_new : ndarray
        Interpolated y values
    """
    from scipy import signal
    from scipy.interpolate import (
        interp1d, Akima1DInterpolator, PchipInterpolator, 
        BarycentricInterpolator, Rbf, UnivariateSpline,
        BSpline, splrep
    )
    
    # Handle case with too few points
    if len(x) < 4:
        if len(x) < 2:
            return x, y
        # Use linear interpolation if we only have 2-3 points
        method = 'linear'
    
    x_new = np.linspace(min(x), max(x), num_points)
    
    # Simple interpolation methods
    if method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']:
        f = interp1d(x, y, kind=method, bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    
    # Advanced interpolation methods
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
        # Smoothing parameter s controls the tradeoff between closeness and smoothness
        s = len(x) * 0.01  # This can be adjusted
        f = UnivariateSpline(x, y, s=s)
        y_new = f(x_new)
        
    elif method == 'bspline':
        # Degree of the spline
        k = min(3, len(x) - 1)
        t, c, k = splrep(x, y, k=k)
        y_new = BSpline(t, c, k)(x_new)
        
    # Smoothing methods (not strictly interpolation)
    elif method == 'savgol':
        # Use existing points, apply filter, then interpolate to new points
        window_length = min(9, len(y) - 2 if len(y) % 2 == 0 else len(y) - 1)
        if window_length < 3:
            window_length = 3
        if window_length % 2 == 0:
            window_length -= 1  # Ensure odd window length
        poly_order = min(3, window_length - 1)
        y_smooth = signal.savgol_filter(y, window_length, poly_order)
        f = interp1d(x, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
        
    elif method == 'moving_average':
        # Apply moving average then interpolate
        window_size = min(5, len(y))
        weights = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, weights, mode='valid')
        # For convolution, the output size is reduced, so adjust x accordingly
        start_idx = window_size // 2
        end_idx = -(window_size // 2) if window_size > 1 else None
        x_smooth = x[start_idx:end_idx]
        if len(x_smooth) != len(y_smooth):
            # If lengths don't match, use another approach
            half_window = (window_size - 1) // 2
            y_smooth = np.array([np.mean(y[max(0, i-half_window):min(len(y), i+half_window+1)]) 
                               for i in range(len(y))])
            x_smooth = x
        f = interp1d(x_smooth, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
        
    elif method == 'gaussian':
        # Gaussian filter then interpolate
        sigma = 2
        y_smooth = signal.gaussian_filter1d(y, sigma)
        f = interp1d(x, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
        
    elif method == 'bessel':
        # Bessel filter then interpolate
        b, a = signal.bessel(4, 0.1, 'low')
        y_smooth = signal.filtfilt(b, a, y)
        f = interp1d(x, y_smooth, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    
    else:
        # Default to cubic interpolation if method not recognized
        f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        y_new = f(x_new)
    
    return x_new, y_new

# List of available interpolation methods
INTERPOLATION_METHODS = [
    'linear',         # Linear segments
    'cubic',          # Cubic spline (default)
    'quadratic',      # Quadratic interpolation
    'nearest',        # Nearest neighbor interpolation
    'akima',          # Akima interpolation (reduced oscillations)
    'pchip',          # Piecewise Cubic Hermite Interpolating Polynomial
    'smoothing_spline', # Smoothing spline with automatic parameter
    'bspline',        # B-spline interpolation
    'savgol',         # Savitzky-Golay filter (smoothing)
    'moving_average', # Moving average smoothing
    'gaussian',       # Gaussian filter smoothing
    'bessel',         # Bessel filter (good for frequency data)
    'barycentric',    # Barycentric interpolation
    'rbf'             # Radial basis function interpolation
]

# -----------------------------------------------------------------------------
# *** FIXED *** selective DOF elimination
# -----------------------------------------------------------------------------

def remove_zero_mass_dofs(
    mass_matrix,
    damping_matrix,
    stiffness_matrix,
    forcing_matrix,
    *,
    tol: float = 1e-8,
):
    """Remove degrees of freedom that are *truly* inactive.

    A DOF is eliminated if **either**
    1. Its corresponding row *and* column are effectively zero in the *mass* matrix
       (no inertia).
    2. It is simultaneously zero in *all* of the other system matrices
       (damping, stiffness, *and* forcing).

    This criterion is much stricter than the previous implementation, which
    removed a DOF when it was zero in *any* matrix, and was therefore prone to
    deleting valid DOFs whenever, for instance, the damping matrix contained a
    zero row/column.
    """

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

    # Strict criterion (see docstring):
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

# -----------------------------------------------------------------------------
# Peak detection and slope calculation improvements
# -----------------------------------------------------------------------------

def safe_structure(key, value, ensure_serializable=True, recursive=True, tol=1e-8):
    import collections.abc

    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
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
    """
    Find significant peaks in the frequency response using prominence filtering.
    
    Parameters:
    -----------
    magnitude : ndarray
        Magnitude of the frequency response
    omega : ndarray
        Frequency values
    min_prominence_ratio : float
        Minimum prominence as a ratio of the maximum magnitude
    max_peaks : int
        Maximum number of peaks to return
        
    Returns:
    --------
    peak_positions : ndarray
        Frequencies of significant peaks
    peak_values : ndarray
        Magnitude values at the peaks
    """
    if len(magnitude) < 3:
        return np.array([]), np.array([])
    
    # Find all peaks
    peaks, properties = find_peaks(magnitude)
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    # Calculate prominence for each peak
    prominences = peak_prominences(magnitude, peaks)[0]
    
    # Filter peaks based on prominence
    max_magnitude = np.max(magnitude)
    min_prominence = max_magnitude * min_prominence_ratio
    
    significant_peaks = peaks[prominences >= min_prominence]
    significant_values = magnitude[significant_peaks]
    
    # If we have too many peaks, keep only the highest ones
    if len(significant_peaks) > max_peaks:
        idx = np.argsort(significant_values)[-max_peaks:]
        significant_peaks = significant_peaks[idx]
        significant_values = significant_values[idx]
    
    # Return peak positions in frequency domain
    return omega[significant_peaks], significant_values

def interpolate_peak_vicinity(magnitude, omega, peak_indices, vicinity_ratio=0.05):
    """
    Creates a higher resolution interpolation around peak areas for more accurate slope calculations.
    
    Parameters:
    -----------
    magnitude : ndarray
        Magnitude of the frequency response
    omega : ndarray
        Frequency values
    peak_indices : ndarray
        Indices of detected peaks
    vicinity_ratio : float
        Ratio of the total frequency range to consider around each peak
        
    Returns:
    --------
    refined_peak_positions : ndarray
        Refined frequency values at peaks
    refined_peak_values : ndarray
        Refined magnitude values at peaks
    """
    if len(peak_indices) == 0 or len(magnitude) < 3:
        return np.array([]), np.array([])
    
    # Create interpolation function
    interp_func = interp1d(omega, magnitude, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Frequency range
    omega_range = omega[-1] - omega[0]
    vicinity_width = omega_range * vicinity_ratio
    
    refined_peak_positions = []
    refined_peak_values = []
    
    for idx in peak_indices:
        peak_pos = omega[idx]
        
        # Create high-resolution samples around the peak
        vicinity_start = max(omega[0], peak_pos - vicinity_width/2)
        vicinity_end = min(omega[-1], peak_pos + vicinity_width/2)
        
        # Create 100 points around the peak for high-resolution analysis
        high_res_omega = np.linspace(vicinity_start, vicinity_end, 100)
        high_res_mag = interp_func(high_res_omega)
        
        # Find the maximum in the high-resolution data
        max_idx = np.argmax(high_res_mag)
        refined_peak_positions.append(high_res_omega[max_idx])
        refined_peak_values.append(high_res_mag[max_idx])
    
    return np.array(refined_peak_positions), np.array(refined_peak_values)

def calculate_robust_slopes(peak_positions, peak_values, slope_threshold=0.01):
    """
    Calculate slopes between peaks with improved robustness.
    Filters out insignificant slopes based on threshold.
    
    Parameters:
    -----------
    peak_positions : ndarray
        Frequencies of peaks
    peak_values : ndarray
        Magnitude values at peaks
    slope_threshold : float
        Minimum normalized slope magnitude to consider significant
        
    Returns:
    --------
    slopes : dict
        Dictionary of significant slopes
    slope_max : float
        Maximum significant slope
    """
    slopes = {}
    slope_max = np.nan
    num_peaks = len(peak_positions)
    
    if num_peaks <= 1:
        return slopes, slope_max
    
    # Calculate normalization factor to make slopes comparable
    amplitude_range = np.max(peak_values) - np.min(peak_values)
    frequency_range = np.max(peak_positions) - np.min(peak_positions)
    
    if amplitude_range == 0 or frequency_range == 0:
        return slopes, slope_max
    
    # Calculate normalized slopes
    for i in range(num_peaks):
        for j in range(i + 1, num_peaks):
            dx = peak_positions[j] - peak_positions[i]
            if dx == 0:
                continue
                
            slope = (peak_values[j] - peak_values[i]) / dx
            
            # Normalize the slope for comparison
            normalized_slope = slope * (frequency_range / amplitude_range)
            
            # Only keep significant slopes
            if abs(normalized_slope) >= slope_threshold:
                slopes[f"slope_{i+1}_{j+1}"] = slope
                if np.isnan(slope_max) or abs(slope) > abs(slope_max):
                    slope_max = slope
    
    return slopes, slope_max

def process_mass(a_mass, omega, user_peak_positions=None):
    """
    Process mass response data to calculate peaks, slopes, bandwidths and area.
    
    Parameters:
    -----------
    a_mass : ndarray
        Complex frequency response data
    omega : ndarray
        Frequency values
    user_peak_positions : list or ndarray, optional
        User-specified peak positions (frequencies) to consider
        
    Returns:
    --------
    dict
        Dictionary containing processed results including peaks, slopes, bandwidths and area
    """
    a_mag = np.abs(a_mass)
    
    # Initialize peak data
    peak_positions = []
    peak_values = []
    
    # Handle user-specified peaks
    if user_peak_positions is not None:
        user_peak_positions = np.array(user_peak_positions)
        # Find the closest actual frequency points to user-specified positions
        for pos in user_peak_positions:
            idx = np.argmin(np.abs(omega - pos))
            peak_positions.append(omega[idx])
            peak_values.append(a_mag[idx])
    
    # Find additional significant peaks with prominence filtering
    detected_positions, detected_values = find_significant_peaks(a_mag, omega)
    
    # Combine user-specified and detected peaks
    if len(detected_positions) > 0:
        peak_positions.extend(detected_positions)
        peak_values.extend(detected_values)
    
    # Convert to numpy arrays
    peak_positions = np.array(peak_positions)
    peak_values = np.array(peak_values)
    
    # Sort peaks by frequency
    sort_idx = np.argsort(peak_positions)
    peak_positions = peak_positions[sort_idx]
    peak_values = peak_values[sort_idx]
    
    # If we found peaks, refine their positions with interpolation
    if len(peak_positions) > 0:
        # First convert peak positions back to indices in the original array
        peak_indices = np.array([np.argmin(np.abs(omega - pos)) for pos in peak_positions])
        
        # Then refine with interpolation
        peak_positions, peak_values = interpolate_peak_vicinity(a_mag, omega, peak_indices)
    
    # Calculate bandwidths
    bandwidths = {}
    for i in range(len(peak_positions)):
        for j in range(i + 1, len(peak_positions)):
            bandwidths[f"bandwidth_{i+1}_{j+1}"] = peak_positions[j] - peak_positions[i]

    # Calculate area under curve
    area_under_curve = simpson(a_mag, x=omega) if len(a_mag) else np.nan
    
    # Calculate robust slopes
    slopes, slope_max = calculate_robust_slopes(peak_positions, peak_values)

    # Format peak data for return
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

# -----------------------------------------------------------------------------
# Remaining plotting utilities (unchanged)
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt

class DraggableAnnotation:
    def __init__(self, annotation):
        self.annotation = annotation
        self.press = None
        self.annotation.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.annotation.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.annotation.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def _on_press(self, event):
        if event.inaxes != self.annotation.axes:
            return
        contains, _ = self.annotation.contains(event)
        if not contains:
            return
        x0, y0 = self.annotation.get_position()
        self.press = (x0, y0, event.xdata, event.ydata)

    def _on_motion(self, event):
        if self.press is None or event.inaxes != self.annotation.axes:
            return
        x0, y0, xpress, ypress = self.press
        self.annotation.set_position((x0 + event.xdata - xpress, y0 + event.ydata - ypress))
        self.annotation.figure.canvas.draw()

    def _on_release(self, _event):
        self.press = None
        self.annotation.figure.canvas.draw()


def plot_mass_response(
    mass_label,
    omega,
    mass_data,
    *,
    show_peaks=False,
    show_slopes=False,
    figsize=(10, 6),
    color=None,
    alpha_fill=0.2,
    font_size=10,
    font_style="normal",
    interpolation_method='cubic',
    interpolation_points=1000,
):
    plt.figure(figsize=figsize)
    color = color or "C0"

    a_mag = mass_data.get("magnitude", np.zeros_like(omega))
    
    # Apply interpolation if requested
    if interpolation_method != 'none' and len(omega) > 1:
        omega_smooth, a_mag_smooth = apply_interpolation(
            omega, a_mag, 
            method=interpolation_method,
            num_points=interpolation_points
        )
        plt.plot(omega_smooth, a_mag_smooth, label=mass_label, linewidth=2, color=color)
        plt.fill_between(omega_smooth, a_mag_smooth, color=color, alpha=alpha_fill)
        
        # Also plot the original points with small markers for reference
        plt.plot(omega, a_mag, 'o', markersize=2, color=color, alpha=0.5)
    else:
        # No interpolation, just plot the original data
        plt.plot(omega, a_mag, label=mass_label, linewidth=2, color=color)
        plt.fill_between(omega, a_mag, color=color, alpha=alpha_fill)

    draggable_annotations = []

    if show_peaks:
        n_peaks = len(mass_data.get("peak_positions", {}))
        positions = [mass_data["peak_positions"].get(f"peak_position_{i+1}", 0) for i in range(n_peaks)]
        values = [mass_data["peak_values"].get(f"peak_value_{i+1}", 0) for i in range(n_peaks)]
        for i, (val, pos) in enumerate(sorted(zip(values, positions), key=lambda x: x[0], reverse=True)[:3], 1):
            s = plt.scatter(pos, val, color="darkred", s=100, zorder=5, edgecolor="black")
            ann = plt.annotate(
                f"Peak {i}\nFreq: {pos:.2f} rad/s\nAmp: {val:.2e}",
                xy=(pos, val),
                xytext=(0, 20 if i % 2 else -30),
                textcoords="offset points",
                ha="center",
                va="bottom" if i % 2 else "top",
                fontsize=font_size,
                fontstyle=font_style,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"),
            )
            draggable_annotations.append(DraggableAnnotation(ann))

    plt.xlabel("Frequency (rad/s)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)
    plt.title(f"Frequency Response of {mass_label}", fontsize=18, weight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_all_mass_responses(
    omega,
    mass_data_list,
    mass_labels,
    *,
    show_peaks=False,
    show_slopes=False,
    figsize=(16, 10),
    alpha_fill=0.1,
    font_size=12,
    font_style="normal",
    interpolation_method='cubic',
    interpolation_points=1000,
):
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(mass_labels))]

    for idx, (lbl, data) in enumerate(zip(mass_labels, mass_data_list)):
        color = colors[idx]
        a_mag = data.get("magnitude", np.zeros_like(omega))
        
        # Apply interpolation if requested
        if interpolation_method != 'none' and len(omega) > 1:
            omega_smooth, a_mag_smooth = apply_interpolation(
                omega, a_mag, 
                method=interpolation_method,
                num_points=interpolation_points
            )
            ax.plot(omega_smooth, a_mag_smooth, label=lbl, linewidth=2, color=color)
            ax.fill_between(omega_smooth, a_mag_smooth, color=color, alpha=alpha_fill)
            
            # Plot original points with small markers
            ax.plot(omega, a_mag, 'o', markersize=2, color=color, alpha=0.3)
        else:
            # No interpolation, plot original data
            ax.plot(omega, a_mag, label=lbl, linewidth=2, color=color)
            ax.fill_between(omega, a_mag, color=color, alpha=alpha_fill)

        if show_peaks:
            n_peaks = len(data.get("peak_positions", {}))
            pos = [data["peak_positions"].get(f"peak_position_{i+1}", 0) for i in range(n_peaks)]
            val = [data["peak_values"].get(f"peak_value_{i+1}", 0) for i in range(n_peaks)]
            for i, (v, p) in enumerate(sorted(zip(val, pos), key=lambda x: x[0], reverse=True)[:3], 1):
                ax.scatter(p, v, color="darkred", s=100, zorder=5, edgecolor="black")
                ann = ax.annotate(
                    f"Peak {i}\nFreq: {p:.3f} rad/s\nAmp: {v:.3e}",
                    xy=(p, v),
                    xytext=(0, 20 if i % 2 else -30),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if i % 2 else "top",
                    fontsize=font_size,
                    fontstyle=font_style,
                    fontname="Times New Roman",
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="black"),
                )
                DraggableAnnotation(ann)

    ax.set_xlabel("Frequency (rad/s)", fontsize=14, fontname="Times New Roman")
    ax.set_ylabel("Amplitude", fontsize=14, fontname="Times New Roman")
    ax.set_title("Combined Frequency Responses of All Masses", fontsize=18, weight="bold", fontname="Times New Roman")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ncol = 1 if len(mass_labels) <= 3 else 2 if len(mass_labels) <= 6 else 3
    lgd = ax.legend(fontsize=font_size, loc="upper right", ncol=ncol, frameon=True, fancybox=True, shadow=True,
                    prop={"family": "Times New Roman", "size": font_size})
    for txt in lgd.get_texts():
        txt.set_fontname("Times New Roman")
        txt.set_fontstyle(font_style)

    ax.tick_params(axis="both", which="major", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Composite / singular response utilities (unchanged)
# -----------------------------------------------------------------------------

def calculate_composite_measure(mass_key, results, target_values, weights):
    composite = 0.0
    percentage_differences = {}
    mass_results = results.get(mass_key, {})
    if not mass_results:
        return composite, percentage_differences

    for criterion, target in target_values.items():
        w = weights.get(criterion, 0.0)
        if w == 0.0:
            continue
        if criterion.startswith("peak_value"):
            actual = mass_results["peak_values"].get(criterion, 0.0)
        elif criterion.startswith("peak_position"):
            actual = mass_results["peak_positions"].get(criterion, 0.0)
        elif criterion.startswith("bandwidth"):
            actual = mass_results["bandwidths"].get(criterion, 0.0)
        elif criterion.startswith("slope"):
            actual = mass_results["slopes"].get(criterion, 0.0)
        elif criterion in ("area_under_curve", "slope_max"):
            actual = mass_results.get(criterion, 0.0)
        else:
            actual = mass_results.get(criterion, 0.0)
        
        # Calculate composite measure using ratio
        if target != 0:
            composite += w * (actual / target)
            
            # Calculate percentage difference for non-zero targets
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

# -----------------------------------------------------------------------------
# Main FRF routine (unchanged apart from the dependency on new DOF function)
# -----------------------------------------------------------------------------

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
    plot_figure=False,
    show_peaks=False,
    show_slopes=False,
    user_peak_positions=None,  # User-specified peaks
    interpolation_method='cubic',  # Added interpolation method
    interpolation_points=1000,  # Added interpolation points
):
    """
    Calculate frequency response functions for the system.
    
    Parameters:
    -----------
    main_system_parameters : list or array
        Main system parameters
    dva_parameters : list or array
        DVA parameters
    omega_start : float
        Starting frequency
    omega_end : float
        Ending frequency
    omega_points : int
        Number of frequency points
    target_values_mass1-5 : dict
        Target values for each mass
    weights_mass1-5 : dict
        Weights for each mass
    plot_figure : bool
        Whether to plot the results
    show_peaks : bool
        Whether to show peaks in plots
    show_slopes : bool
        Whether to show slopes in plots
    user_peak_positions : dict, optional
        Dictionary of user-specified peak positions for each mass
        Format: {"mass_1": [freq1, freq2, ...], "mass_2": [freq1, freq2, ...], ...}
    interpolation_method : str
        Interpolation method to use for plotting
        Options: 'none', 'linear', 'cubic', 'quadratic', 'akima', etc.
    interpolation_points : int
        Number of points to use in the interpolated curve
    """
    # ---------------------------------------------------------------------
    # Unpack parameters (unchanged)…
    # ---------------------------------------------------------------------
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

    # Mass matrix … (unchanged)
    mass_matrix = np.array([
        [1 + beta_1 + beta_2 + beta_3, 0, -beta_1, -beta_2, -beta_3],
        [0, MU + beta_4 + beta_5 + beta_6, -beta_4, -beta_5, -beta_6],
        [-beta_1, -beta_4, mu_1 + beta_1 + beta_4 + beta_7 + beta_8 + beta_10 + beta_9, -beta_9, -beta_10],
        [-beta_2, -beta_5, -beta_9, mu_2 + beta_11 + beta_2 + beta_9 + beta_12 + beta_5 + beta_15, -beta_15],
        [-beta_3, -beta_6, -beta_10, -beta_15, mu_3 + beta_14 + beta_6 + beta_13 + beta_3 + beta_15 + beta_10],
    ])

    # Damping & stiffness matrices (unchanged)…
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

    # Forcing vector (unchanged)…
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

    # *** uses new removal function ***
    mm, cc, kk, f_reduced, active = remove_zero_mass_dofs(mass_matrix, damping_matrix_raw, stiffness_matrix_raw, f)
    if mm.size == 0:
        raise ValueError("All degrees of freedom have zero mass. Cannot perform analysis.")

    n_dofs = mm.shape[0]
    A = np.zeros((n_dofs, len(omega)), dtype=complex)
    for i, Om in enumerate(Omega):
        hh = -Om**2 * mm + 2 * ZETA_DC * Om * cc + kk
        hh *= OMEGA_DC**2
        try:
            A[:, i] = np.linalg.solve(hh, f_reduced[:, i]) * OMEGA_DC**2
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError(f"Linear algebra error at frequency index {i}: {err}")

    results = {}
    idxs = np.where(active)[0]
    label_map = {0: "mass_1", 1: "mass_2", 2: "mass_3", 3: "mass_4", 4: "mass_5"}
    mass_data_list, mass_labels_list = [], []

    for local_idx, dof in enumerate(idxs):
        lbl = label_map.get(dof, f"mass_{dof+1}")
        # Get user-specified peaks for this mass if provided
        mass_peaks = user_peak_positions.get(lbl, None) if user_peak_positions else None
        mass_res = process_mass(A[local_idx, :], omega, user_peak_positions=mass_peaks)
        results[lbl] = mass_res
        mass_data_list.append(mass_res)
        mass_labels_list.append(lbl)

    if plot_figure:
        for lbl, data in zip(mass_labels_list, mass_data_list):
            plot_mass_response(
                lbl, omega, data, 
                show_peaks=show_peaks, 
                show_slopes=show_slopes,
                interpolation_method=interpolation_method,
                interpolation_points=interpolation_points
            )
        plot_all_mass_responses(
            omega, mass_data_list, mass_labels_list, 
            show_peaks=show_peaks, 
            show_slopes=show_slopes,
            interpolation_method=interpolation_method,
            interpolation_points=interpolation_points
        )

    target_dict = {
        "mass_1": target_values_mass1,
        "mass_2": target_values_mass2,
        "mass_3": target_values_mass3,
        "mass_4": target_values_mass4,
        "mass_5": target_values_mass5,
    }
    weight_dict = {
        "mass_1": weights_mass1,
        "mass_2": weights_mass2,
        "mass_3": weights_mass3,
        "mass_4": weights_mass4,
        "mass_5": weights_mass5,
    }

    results = calculate_singular_response(results, target_dict, weight_dict)
    
    # Add interpolation method information to results
    results["interpolation_info"] = {
        "method": interpolation_method,
        "points": interpolation_points
    }
    
    # Remove all print statements and just return the results
    return results

# -----------------------------------------------------------------------------
# Omega points sensitivity analysis
# -----------------------------------------------------------------------------

def perform_omega_points_sensitivity_analysis(
    main_system_parameters,
    dva_parameters,
    omega_start,
    omega_end,
    initial_points=100,
    max_points=1000000000,  # Allow very large values (default 10^9)
    step_size=100,
    convergence_threshold=0.01,
    max_iterations=100,  # Increased iterations to handle larger ranges
    mass_of_interest="mass_1",
    plot_results=True,
):
    """
    Perform sensitivity analysis on the number of omega points to determine
    the optimal value for stable slope calculations.
    
    Parameters:
    -----------
    main_system_parameters : list or array
        Main system parameters for FRF calculation
    dva_parameters : list or array
        DVA parameters for FRF calculation
    omega_start : float
        Starting frequency for analysis
    omega_end : float
        Ending frequency for analysis
    initial_points : int
        Initial number of omega points to start with
    max_points : int
        Maximum number of omega points to try
    step_size : int
        Increment in omega points between iterations
    convergence_threshold : float
        Threshold for determining convergence.  Convergence is evaluated using the
        maximum relative change across all available metrics (peak positions,
        peak values, bandwidths, slopes, area under curve, etc.)
    max_iterations : int
        Maximum number of iterations to run
    mass_of_interest : str
        Which mass to focus on for the analysis
    plot_results : bool
        Whether to create and return the figure (no longer shows directly)
        
    Returns:
    --------
    dict
        Dictionary containing convergence results, optimal points, and figure object if plot_results=True
    """
    # Initialize empty lists to store results
    point_values = []
    slope_max_values = []
    # Maximum relative change across metrics for each iteration
    relative_changes = []
    # Store the full metric dictionary for every iteration
    metrics_history = []
    
    # Empty target and weight dictionaries for minimal FRF calculation
    empty_dict = {}
    
    # Initialize variables for tracking convergence
    prev_metrics = None
    optimal_points = max_points  # Default to maximum if convergence is not reached
    converged = False
    convergence_point = None  # Track where convergence was first detected
    iteration = 0
    
    # Start with the initial number of points
    current_points = initial_points
    
    # Always use the user-specified step size, no auto-adjustment
    original_step_size = step_size
    
    # Ensure dva_parameters has the correct length (48 values)
    # The expected order is: 15 betas, 15 lambdas, 3 mus, 15 nus
    if len(dva_parameters) != 48:
        # If the length is not 48, try to handle common cases
        if isinstance(dva_parameters, (list, tuple)):
            # If it's a list or tuple, extend it with zeros to 48 elements
            padded_dva_params = list(dva_parameters)
            padded_dva_params.extend([0] * (48 - len(padded_dva_params)))
            dva_parameters = padded_dva_params[:48]  # Truncate if too long
        else:
            # If not a list/tuple, create a default array of zeros
            dva_parameters = [0] * 48
            print("Warning: dva_parameters not in expected format. Using zeros.")
    
    def _flatten_metrics(mass_dict):
        """Flatten nested metric dictionaries into a single level"""
        flat = {}
        for key, val in mass_dict.items():
            if key == "magnitude":
                continue
            if isinstance(val, dict):
                for subk, subval in val.items():
                    if np.isscalar(subval):
                        flat[f"{key}.{subk}"] = float(subval)
            elif np.isscalar(val):
                flat[key] = float(val)
            elif isinstance(val, (list, tuple, np.ndarray)):
                for i, subval in enumerate(val):
                    if np.isscalar(subval):
                        flat[f"{key}.{i}"] = float(subval)
        return flat

    while current_points <= max_points and iteration < max_iterations:
        # Calculate FRF with current number of omega points
        results = frf(
            main_system_parameters=main_system_parameters,
            dva_parameters=dva_parameters,
            omega_start=omega_start,
            omega_end=omega_end,
            omega_points=current_points,
            target_values_mass1=empty_dict,
            weights_mass1=empty_dict,
            target_values_mass2=empty_dict,
            weights_mass2=empty_dict,
            target_values_mass3=empty_dict,
            weights_mass3=empty_dict,
            target_values_mass4=empty_dict,
            weights_mass4=empty_dict,
            target_values_mass5=empty_dict,
            weights_mass5=empty_dict,
            plot_figure=False,
            user_peak_positions=None,
            interpolation_method='cubic',
            interpolation_points=1000
        )
        
        # Extract the maximum slope for the mass of interest
        if mass_of_interest in results:
            mass_data = results[mass_of_interest]
            slope_max = mass_data.get("slope_max", np.nan)
            current_metrics = _flatten_metrics(mass_data)

            # Store current values
            point_values.append(current_points)
            slope_max_values.append(slope_max)
            metrics_history.append(current_metrics)

            if prev_metrics is not None:
                changes = {}
                for key in current_metrics.keys() & prev_metrics.keys():
                    prev_val = prev_metrics[key]
                    curr_val = current_metrics[key]
                    if prev_val != 0 and not np.isnan(prev_val) and not np.isnan(curr_val):
                        changes[key] = abs((curr_val - prev_val) / prev_val)

                max_change = max(changes.values(), default=np.nan)
                relative_changes.append(max_change)

                if max_change < convergence_threshold and not converged:
                    converged = True
                    convergence_point = current_points
            else:
                relative_changes.append(np.nan)

            prev_metrics = current_metrics
            
        # Increment the number of points for next iteration
        current_points += step_size
        iteration += 1
    
    # Determine if we actually reached max_points or hit the iteration limit
    reached_max = (current_points - step_size) >= max_points
    
    # Set optimal_points to either max_points (if we reached it) or the last point we actually calculated
    if reached_max:
        optimal_points = max_points
    else:
        optimal_points = point_values[-1] if point_values else initial_points
    
    # Prepare the results dictionary
    convergence_results = {
        "omega_points": point_values,
        "max_slopes": slope_max_values,
        "relative_changes": relative_changes,
        "metrics_history": metrics_history,
        "optimal_points": optimal_points,
        "converged": converged,
        "convergence_point": convergence_point,  # Track where convergence was first detected
        "all_points_analyzed": reached_max,  # True only if we actually reached max_points
        "requested_max_points": max_points,  # Store what the user actually requested
        "highest_analyzed_point": point_values[-1] if point_values else initial_points,  # Highest point actually analyzed
        "iteration_limit_reached": iteration >= max_iterations,  # Did we hit the iteration limit?
        "step_size": step_size,  # User-specified step size (not adjusted)
    }
    
    # Create the figure if requested but DON'T show it directly
    # Instead, return it for the main thread to display
    if plot_results and len(point_values) > 1:
        fig = plt.figure(figsize=(10, 10))
        
        # Plot max slope vs. number of points
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(point_values, slope_max_values, 'o-', linewidth=2)
        ax1.axvline(x=optimal_points, color='r', linestyle='--', label=f'Optimal: {optimal_points} points')
        ax1.set_ylabel('Maximum Slope', fontsize=12)
        ax1.set_title('Convergence of Maximum Slope with Increasing Omega Points', fontsize=14, weight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Plot maximum relative change across metrics
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        if len(relative_changes) > 1:
            ax2.semilogy(point_values[1:], relative_changes[1:], 'o-', linewidth=2)
            ax2.axhline(y=convergence_threshold, color='r', linestyle='--',
                        label=f'Threshold: {convergence_threshold}')
        ax2.set_xlabel('Number of Omega Points', fontsize=12)
        ax2.set_ylabel('Max Relative Change', fontsize=12)
        ax2.set_title('Maximum Relative Change Across Metrics', fontsize=14, weight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Add the figure to the results but don't show it
        fig.tight_layout()
        convergence_results["figure"] = fig
    
    return convergence_results

# Function to run the sensitivity analysis and update the optimal omega points
def determine_optimal_omega_points(
    main_system_parameters,
    dva_parameters,
    omega_start,
    omega_end,
    **kwargs
):
    """
    Convenience function to determine the optimal number of omega points and
    return that value for use in subsequent FRF calculations.
    
    Parameters:
    -----------
    main_system_parameters : list or array
        Main system parameters for FRF calculation
    dva_parameters : list or array
        DVA parameters for FRF calculation
    omega_start : float
        Starting frequency for analysis
    omega_end : float
        Ending frequency for analysis
    **kwargs : dict
        Additional parameters to pass to perform_omega_points_sensitivity_analysis
        
    Returns:
    --------
    int
        Optimal number of omega points for stable slope calculations
    """
    # Run the sensitivity analysis
    results = perform_omega_points_sensitivity_analysis(
        main_system_parameters=main_system_parameters,
        dva_parameters=dva_parameters,
        omega_start=omega_start,
        omega_end=omega_end,
        **kwargs
    )
    
    # Return the optimal number of points
    return results["optimal_points"]
