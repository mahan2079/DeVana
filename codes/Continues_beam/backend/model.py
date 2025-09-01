from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import numpy as np


ControlQuantity = Literal["displacement", "velocity", "acceleration"]


@dataclass
class TargetSpecification:
    """
    Control target or restriction at selected points/regions.

    - quantity: one of 'displacement' | 'velocity' | 'acceleration'
    - locations: x positions in [0, L] where quantity is evaluated
    - weights: per-location weights (broadcasts if length 1)
    - target_values: desired magnitude values (broadcasts if length 1)
    - inequality: optional (lower_bounds, upper_bounds); each can be None or list

    Notes
    - Magnitudes are evaluated on the frequency response averaged across the
      requested frequency grid (provided to objective_from_targets).
    - Inequality bounds contribute hinge penalties when violated.
    """

    quantity: ControlQuantity
    locations: List[float]
    weights: List[float]
    target_values: List[float]
    inequality: Tuple[List[float] | None, List[float] | None] | None = None


@dataclass
class LayerSpec:
    thickness: float
    E: float
    rho: float


class BeamModel:
    """
    Euler-Bernoulli beam (clamped-free) with optional discrete ground springs and dampers.

    - Discretization: 2-noded Hermite beam FEM (w, theta per node).
    - Cross-section: single layer or composite layers; computes equivalent EI and mass/length.
    - Damping: Rayleigh baseline (alpha*M + beta*K) plus point viscous dampers at nodes (w DOF).
    - External excitation: nodal force F(omega); default is unit vertical force at free end w-DOF.
    """

    def __init__(
        self,
        length: float,
        width: float,
        thickness: float | None = None,
        youngs_modulus: float | None = None,
        density: float | None = None,
        num_elements: int = 40,
        rayleigh_alpha: float = 0.0,
        rayleigh_beta: float = 0.0,
        layers: List[LayerSpec] | List[Dict[str, float]] | None = None,
    ) -> None:
        self.L = float(length)
        self.b = float(width)
        self.h = float(thickness) if thickness is not None else 0.0
        self.E = float(youngs_modulus) if youngs_modulus is not None else 0.0
        self.rho = float(density) if density is not None else 0.0
        self.N = int(num_elements)
        self.alpha = float(rayleigh_alpha)
        self.beta = float(rayleigh_beta)

        # Layers handling
        self.layers: List[LayerSpec] | None = None
        if layers:
            self.layers = [
                LayerSpec(float(l.get("thickness", 0.0)), float(l.get("E", 0.0)), float(l.get("rho", 0.0)))
                if isinstance(l, dict) else l  # type: ignore[arg-type]
                for l in layers
            ]
        # Section properties
        self.A, self.EI, self.m_line = self._compute_section_properties()

        # Grid (nodes)
        self.x_nodes = np.linspace(0.0, self.L, self.N + 1)

        # Assemble FEM M, K (2 DOFs/node)
        self.M, self.K = self._assemble_beam_fem()

    # ------------------------------ Assembly ---------------------------------
    def _compute_section_properties(self) -> Tuple[float, float, float]:
        """Return (A, EI, m_line) for single/composite section.
        For composite, use transformed section method with constant width b.
        Coordinates: y=0 at bottom; layers stacked upward.
        """
        if self.layers and len(self.layers) > 0:
            # neutral axis using E-weighted area
            y0 = 0.0
            num = 0.0
            den = 0.0
            y_cursor = 0.0
            for Lr in self.layers:
                t = Lr.thickness
                A_i = self.b * t
                y_c = y_cursor + 0.5 * t
                num += Lr.E * A_i * y_c
                den += Lr.E * A_i
                y_cursor += t
            y_bar = num / den if den != 0.0 else 0.0
            # EI and mass/length
            EI = 0.0
            A = 0.0
            m_line = 0.0
            y_cursor = 0.0
            for Lr in self.layers:
                t = Lr.thickness
                A_i = self.b * t
                I_c = self.b * t**3 / 12.0
                y_c = y_cursor + 0.5 * t
                EI += Lr.E * (I_c + A_i * (y_c - y_bar) ** 2)
                A += A_i
                m_line += Lr.rho * A_i
                y_cursor += t
            return A, EI, m_line
        else:
            # Fallback to single-layer rectangular section
            A = self.b * self.h
            I = self.b * self.h**3 / 12.0
            EI = self.E * I
            m_line = self.rho * A
            return A, EI, m_line

    def _assemble_beam_fem(self) -> Tuple[np.ndarray, np.ndarray]:
        n_nodes = self.N + 1
        ndof = 2 * n_nodes  # w, theta per node
        Le = self.L / self.N
        EI = self.EI
        rhoA = self.m_line

        K = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))

        # Element stiffness/mass (Euler-Bernoulli)
        kfac = EI / (Le**3)
        Ke = kfac * np.array([
            [12.0, 6.0 * Le, -12.0, 6.0 * Le],
            [6.0 * Le, 4.0 * Le**2, -6.0 * Le, 2.0 * Le**2],
            [-12.0, -6.0 * Le, 12.0, -6.0 * Le],
            [6.0 * Le, 2.0 * Le**2, -6.0 * Le, 4.0 * Le**2],
        ])

        mfac = rhoA * Le / 420.0
        Me = mfac * np.array([
            [156.0, 22.0 * Le, 54.0, -13.0 * Le],
            [22.0 * Le, 4.0 * Le**2, 13.0 * Le, -3.0 * Le**2],
            [54.0, 13.0 * Le, 156.0, -22.0 * Le],
            [-13.0 * Le, -3.0 * Le**2, -22.0 * Le, 4.0 * Le**2],
        ])

        for e in range(self.N):
            n1 = e
            n2 = e + 1
            dof = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
            for i in range(4):
                for j in range(4):
                    K[dof[i], dof[j]] += Ke[i, j]
                    M[dof[i], dof[j]] += Me[i, j]

        # Clamped at x=0: w0=0, theta0=0 via penalty on K
        large = 1e18
        K[0, 0] += large
        K[1, 1] += large
        return M, K

    def _index_from_x(self, xloc: float) -> int:
        return int(round(np.clip(xloc / self.L * self.N, 0, self.N)))

    def _augment_stiffness(self, k_points: List[Tuple[float, float]]) -> np.ndarray:
        """Add ground springs on translation DOF at nearest node."""
        K = self.K.copy()
        for xloc, kval in (k_points or []):
            idx = self._index_from_x(xloc)
            dof_w = 2 * idx
            K[dof_w, dof_w] += float(max(0.0, kval))
        return K

    def _build_damping(self, c_points: List[Tuple[float, float]]) -> np.ndarray:
        C = self.alpha * self.M + self.beta * self.K
        for xloc, cval in (c_points or []):
            idx = self._index_from_x(xloc)
            dof_w = 2 * idx
            C[dof_w, dof_w] += float(max(0.0, cval))
        return C

    # ------------------------------ FRF --------------------------------------
    def frequency_response(
        self,
        omega: np.ndarray,
        k_points: List[Tuple[float, float]] | None = None,
        c_points: List[Tuple[float, float]] | None = None,
        force: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Return {'x','omega','W'} where W is complex nodal displacement FRF [n_nodes, n_w]."""
        n_nodes = self.N + 1
        ndof = 2 * n_nodes
        K = self._augment_stiffness(k_points or [])
        C = self._build_damping(c_points or [])

        if force is None:
            def force(om: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
                F = np.zeros((ndof, om.size), dtype=float)
                # unit force at free end w DOF
                F[2 * (n_nodes - 1), :] = 1.0
                return F

        F = force(omega)
        if F.shape[0] != ndof:
            raise ValueError("Force vector dimension mismatch for FEM DOFs")
        W_nodes = np.zeros((n_nodes, omega.size), dtype=complex)
        for i, w in enumerate(omega):
            A = -w**2 * self.M + 1j * w * C + K
            try:
                d = np.linalg.solve(A, F[:, i])  # full DOF vector
                W_nodes[:, i] = d[0::2]
            except np.linalg.LinAlgError:
                W_nodes[:, i] = 0.0
        return {"x": self.x_nodes.copy(), "omega": omega.copy(), "W": W_nodes}

    def derive_quantity(self, resp: Dict[str, np.ndarray], quantity: ControlQuantity) -> np.ndarray:
        W = resp["W"]
        omega = resp["omega"]
        if quantity == "displacement":
            return W
        if quantity == "velocity":
            return 1j * omega[None, :] * W
        if quantity == "acceleration":
            return -(omega[None, :] ** 2) * W
        raise ValueError("Unknown quantity")

    # ---------------------------- Objective ----------------------------------
    def _broadcast_to_length(self, arr: List[float] | None, n: int, fill: float) -> np.ndarray:
        if arr is None:
            return np.full((n, 1), fill, dtype=float)
        a = np.asarray(arr, dtype=float)
        if a.size == 1:
            a = np.full((n,), float(a[0]))
        if a.size != n:
            raise ValueError("Array length mismatch with number of locations")
        return a[:, None]

    def objective_from_targets(
        self,
        k_points: List[Tuple[float, float]],
        c_points: List[Tuple[float, float]],
        targets: List[TargetSpecification],
        omega: np.ndarray,
        penalty_weight: float = 10.0,
        force: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> float:
        """
        Weighted squared error to targets, averaged over frequency, plus hinge
        penalties for inequality bounds.
        """
        resp = self.frequency_response(omega, k_points, c_points, force=force)
        total = 0.0
        for spec in targets:
            Q = self.derive_quantity(resp, spec.quantity)  # [n_nodes, n_w]
            idxs = [self._index_from_x(x) for x in spec.locations]
            mag = np.abs(Q[idxs, :])  # [n_pts, n_w]

            desired = self._broadcast_to_length(spec.target_values, len(idxs), 0.0)
            weights = self._broadcast_to_length(spec.weights, len(idxs), 1.0)

            err = weights * (mag - desired)
            mse = float(np.mean(err**2))
            total += mse

            if spec.inequality is not None:
                lo_list, hi_list = spec.inequality
                lo = self._broadcast_to_length(lo_list, len(idxs), -np.inf)
                hi = self._broadcast_to_length(hi_list, len(idxs), +np.inf)
                pen_lo = np.maximum(0.0, lo - mag)
                pen_hi = np.maximum(0.0, mag - hi)
                total += penalty_weight * float(np.mean(pen_lo + pen_hi))

        return float(total)
