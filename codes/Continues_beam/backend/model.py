import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Literal, Callable, Dict


ControlQuantity = Literal['displacement', 'velocity', 'acceleration']


@dataclass
class TargetSpecification:
    """
    Defines control targets or constraints at specific points or regions.

    Attributes
    - quantity: 'displacement' | 'velocity' | 'acceleration'
    - locations: list of x positions in [0, L] where quantity is measured
    - weights: same length as locations; weights in objective
    - target_values: desired values at locations (same length)
    - inequality: optional tuple (lower_bounds, upper_bounds) arrays same length
    """
    quantity: ControlQuantity
    locations: List[float]
    weights: List[float]
    target_values: List[float]
    inequality: Tuple[List[float], List[float]] | None = None


class BeamModel:
    """
    Simple Euler–Bernoulli beam model with Rayleigh damping baseline plus
    optional ground springs and viscous dampers at discrete points.

    This model uses a standard finite-difference discretization to compute
    modal properties and frequency-response to harmonic base excitation.
    The simulation functions are kept intentionally lightweight and stable.
    """

    def __init__(
        self,
        length: float,
        width: float,
        thickness: float,
        youngs_modulus: float,
        density: float,
        num_elements: int = 40,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
    ) -> None:
        self.L = float(length)
        self.b = float(width)
        self.h = float(thickness)
        self.E = float(youngs_modulus)
        self.rho = float(density)
        self.N = int(num_elements)
        self.alpha = float(rayleigh_alpha)
        self.beta = float(rayleigh_beta)

        # Derived geometric properties
        self.A = self.b * self.h
        self.I = self.b * self.h**3 / 12.0

        # Discretization grid
        self.x = np.linspace(0.0, self.L, self.N + 1)

        # Pre-build baseline mass and stiffness matrices (clamped-free default)
        self.M, self.K = self._assemble_beam_matrices()

    def _assemble_beam_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble mass and stiffness using a finite difference Euler–Bernoulli model.
        Boundary: clamped at x=0, free at x=L.

        Returns
        (M, K) with size (n_dof, n_dof).
        """
        n = self.N + 1
        dx = self.L / self.N

        # Fourth-derivative finite difference stiffness operator
        D4 = np.zeros((n, n))
        for i in range(n):
            for j in range(-2, 3):
                jj = i + j
                if 0 <= jj < n:
                    if j == 0:
                        D4[i, jj] += 6.0
                    elif abs(j) == 1:
                        D4[i, jj] += -4.0
                    elif abs(j) == 2:
                        D4[i, jj] += 1.0
        D4 /= dx**4

        # Apply clamped boundary at x=0: w=0, w'=0 via large penalty
        large = 1e12
        bc = np.zeros((n, n))
        bc[0, 0] = large  # displacement clamp
        if n > 1:
            bc[1, 1] = large  # approximate slope clamp

        K = self.E * self.I * D4 + bc

        # Lumped mass approximation (consistent mass could be used too)
        m_line = self.rho * self.A
        M = np.eye(n) * m_line * dx

        return M, K

    def _build_damping_matrix(self, c_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Build viscous damping matrix with Rayleigh part plus point dampers.
        c_points: list of (x_location, c_value)
        """
        n = self.N + 1
        C = self.alpha * self.M + self.beta * self.K
        if c_points:
            for xloc, cval in c_points:
                idx = int(round(np.clip(xloc / self.L * self.N, 0, self.N)))
                C[idx, idx] += float(max(0.0, cval))
        return C

    def _augment_stiffness_with_springs(self, k_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Add ground-connected linear springs at discrete grid points.
        k_points: list of (x_location, k_value)
        """
        K = self.K.copy()
        if k_points:
            for xloc, kval in k_points:
                idx = int(round(np.clip(xloc / self.L * self.N, 0, self.N)))
                K[idx, idx] += float(max(0.0, kval))
        return K

    def frequency_response(
        self,
        omega: np.ndarray,
        k_points: List[Tuple[float, float]] | None,
        c_points: List[Tuple[float, float]] | None,
        force: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute steady-state frequency response for external nodal force vector F(ω).
        If force is None, use unit force at free end.
        Returns dict with keys: 'x', 'omega', 'W' (complex displacement shape per ω)
        W has shape (n_nodes, n_omega)
        """
        n = self.N + 1
        K_aug = self._augment_stiffness_with_springs(k_points or [])
        C = self._build_damping_matrix(c_points or [])

        if force is None:
            def force(_omega: np.ndarray) -> np.ndarray:
                f = np.zeros((n, _omega.size), dtype=float)
                f[-1, :] = 1.0  # tip unit force
                return f

        F = force(omega)
        W = np.zeros((n, omega.size), dtype=complex)
        for i, w in enumerate(omega):
            A = -w**2 * self.M + 1j * w * C + K_aug
            try:
                W[:, i] = np.linalg.solve(A, F[:, i])
            except np.linalg.LinAlgError:
                W[:, i] = 0.0
        return {'x': self.x.copy(), 'omega': omega.copy(), 'W': W}

    def derive_quantity(self, resp: Dict[str, np.ndarray], quantity: ControlQuantity) -> np.ndarray:
        """
        Convert displacement FRF to desired quantity FRF at all nodes.
        """
        W = resp['W']
        omega = resp['omega']
        if quantity == 'displacement':
            return W
        if quantity == 'velocity':
            return 1j * omega[None, :] * W
        if quantity == 'acceleration':
            return -(omega[None, :]**2) * W
        raise ValueError('Unknown quantity')

    def objective_from_targets(
        self,
        k_points: List[Tuple[float, float]],
        c_points: List[Tuple[float, float]],
        targets: List[TargetSpecification],
        omega: np.ndarray,
    ) -> float:
        """
        Compute scalar objective as weighted squared error to target values across
        all provided TargetSpecifications, averaged across frequency grid.
        Inequality bounds (if provided) are penalized with hinge loss.
        """
        resp = self.frequency_response(omega, k_points, c_points)
        total = 0.0
        for spec in targets:
            Q = self.derive_quantity(resp, spec.quantity)  # (n_nodes, n_w)
            idxs = [int(round(np.clip(x / self.L * self.N, 0, self.N))) for x in spec.locations]
            # Use magnitude FRF
            mag = np.abs(Q[idxs, :])  # (n_pts, n_w)
            desired = np.array(spec.target_values, dtype=float)[:, None]
            wts = np.array(spec.weights, dtype=float)[:, None]

            err = wts * (mag - desired)
            mse = np.mean(err**2)
            total += mse

            if spec.inequality is not None:
                lo, hi = spec.inequality
                lo = np.array(lo, dtype=float)[:, None]
                hi = np.array(hi, dtype=float)[:, None]
                # hinge penalties
                pen_lo = np.mean(np.maximum(0.0, lo - mag))
                pen_hi = np.mean(np.maximum(0.0, mag - hi))
                total += 10.0 * (pen_lo + pen_hi)

        return float(total)



