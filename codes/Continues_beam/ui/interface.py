from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt, QObject, QThread
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QTabWidget,
    QFrame,
)

from ..backend.model import BeamModel, TargetSpecification
from ..backend.optimizers import (
    optimize_values_at_locations,
    optimize_placement_and_values,
    Bounds,
)


def _parse_positions(text: str, L: float) -> List[float]:
    """
    Parse CSV of numbers and simple ranges into positions in [0, L].
    Supported tokens:
      - x (float)
      - a-b:n  -> n points linearly spaced from a to b inclusive
    """
    text = (text or "").strip()
    if not text:
        return []
    xs: List[float] = []
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok and '-' in tok:
            # a-b:n
            try:
                span, nstr = tok.split(':', 1)
                a_str, b_str = span.split('-', 1)
                a = float(a_str)
                b = float(b_str)
                n = int(nstr)
                xs.extend(np.linspace(a, b, n).tolist())
                continue
            except Exception:
                pass
        # plain float
        try:
            xs.append(float(tok))
        except ValueError:
            continue
    # clamp to [0,L]
    xs = [float(max(0.0, min(L, v))) for v in xs]
    return xs


def _parse_array(text: str, default: float, n: int) -> List[float]:
    text = (text or "").strip()
    if not text:
        return [default] * n
    toks = [t for t in text.split(',') if t.strip()]
    try:
        arr = [float(t.strip()) for t in toks]
    except ValueError:
        arr = [default] * n
    if len(arr) == 1:
        return arr * n
    return arr


class BeamOptimizationInterface(QWidget):
    analysis_completed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._theme = 'Dark'

        self._build_ui()

    def set_theme(self, theme: str):
        self._theme = theme or 'Dark'
        # Minimal theming via palette/stylesheet toggles if desired later

    # ----------------------------- UI ----------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Model tab
        self.model_tab = QWidget()
        tabs.addTab(self.model_tab, "Model")
        self._build_model_tab()

        # Targets tab
        self.targets_tab = QWidget()
        tabs.addTab(self.targets_tab, "Targets")
        self._build_targets_tab()

        # Loading tab
        self.loading_tab = QWidget()
        tabs.addTab(self.loading_tab, "Loading")
        self._build_loading_tab()

        # Optimization tab
        self.opt_tab = QWidget()
        tabs.addTab(self.opt_tab, "Optimization")
        self._build_optimization_tab()

        # Results tab
        self.results_tab = QWidget()
        tabs.addTab(self.results_tab, "Results")
        self._build_results_tab()

    def _build_model_tab(self):
        layout = QHBoxLayout(self.model_tab)

        # Geometry and material
        geom_box = QGroupBox("Beam Geometry & Material")
        geom_form = QFormLayout(geom_box)
        self.L_in = QDoubleSpinBox(); self.L_in.setRange(0.01, 1000.0); self.L_in.setValue(1.0); self.L_in.setSuffix(" m"); self.L_in.setDecimals(3)
        self.b_in = QDoubleSpinBox(); self.b_in.setRange(1e-4, 10.0); self.b_in.setValue(0.05); self.b_in.setSuffix(" m"); self.b_in.setDecimals(4)
        self.h_in = QDoubleSpinBox(); self.h_in.setRange(1e-6, 1.0); self.h_in.setValue(0.01); self.h_in.setSuffix(" m"); self.h_in.setDecimals(6)
        self.E_in = QDoubleSpinBox(); self.E_in.setRange(1e6, 1e13); self.E_in.setValue(210e9); self.E_in.setDecimals(0)
        self.rho_in = QDoubleSpinBox(); self.rho_in.setRange(10, 5e4); self.rho_in.setValue(7800); self.rho_in.setDecimals(0); self.rho_in.setSuffix(" kg/m^3")
        self.N_in = QSpinBox(); self.N_in.setRange(5, 400); self.N_in.setValue(40)
        self.alpha_in = QDoubleSpinBox(); self.alpha_in.setRange(0.0, 100.0); self.alpha_in.setDecimals(6); self.alpha_in.setValue(0.0)
        self.beta_in = QDoubleSpinBox(); self.beta_in.setRange(0.0, 100.0); self.beta_in.setDecimals(6); self.beta_in.setValue(0.0)

        # Tooltips
        self.L_in.setToolTip("Beam length L in meters (domain [0, L]).")
        self.b_in.setToolTip("Beam width b in meters (cross-section width).")
        self.h_in.setToolTip("Beam thickness h in meters (cross-section height).")
        self.E_in.setToolTip("Young's modulus E (Pa) of the beam material.")
        self.rho_in.setToolTip("Mass density ρ (kg/m^3) of the beam material.")
        self.N_in.setToolTip("Number of finite-difference elements (N). More gives finer accuracy but slower.")
        self.alpha_in.setToolTip("Rayleigh damping mass coefficient α (C = αM + βK).")
        self.beta_in.setToolTip("Rayleigh damping stiffness coefficient β (C = αM + βK).")

        geom_form.addRow("Length L:", self.L_in)
        geom_form.addRow("Width b:", self.b_in)
        geom_form.addRow("Thickness h:", self.h_in)
        geom_form.addRow("Young's E:", self.E_in)
        geom_form.addRow("Density ρ:", self.rho_in)
        geom_form.addRow("Elements N:", self.N_in)
        geom_form.addRow("Rayleigh α:", self.alpha_in)
        geom_form.addRow("Rayleigh β:", self.beta_in)

        layout.addWidget(geom_box)

        # Composite layers (optional)
        layers_box = QGroupBox("Composite Layers (optional)")
        layers_box.setToolTip("Define layered cross-section. If any layers exist, they override single-layer thickness/E/ρ.")
        layers_v = QVBoxLayout(layers_box)

        header = QLabel("Thickness [m]    E [Pa]    ρ [kg/m³]")
        header.setStyleSheet("color: #666;")
        layers_v.addWidget(header)

        self.layers_rows: List[QWidget] = []
        self.layers_container = QVBoxLayout()
        layers_v.addLayout(self.layers_container)

        btn_row = QHBoxLayout()
        self.add_layer_btn = QPushButton("Add Layer")
        self.add_layer_btn.setToolTip("Add a new layer row with thickness/E/ρ.")
        self.add_layer_btn.clicked.connect(self._add_layer_row)
        self.clear_layers_btn = QPushButton("Clear Layers")
        self.clear_layers_btn.setToolTip("Remove all layer rows (reverts to single-layer model).")
        self.clear_layers_btn.clicked.connect(self._clear_layers)
        btn_row.addWidget(self.add_layer_btn)
        btn_row.addWidget(self.clear_layers_btn)
        btn_row.addStretch(1)
        layers_v.addLayout(btn_row)

        layout.addWidget(layers_box)

        # Frequency settings
        freq_box = QGroupBox("Frequency Settings")
        freq_form = QFormLayout(freq_box)
        self.fmin_in = QDoubleSpinBox(); self.fmin_in.setRange(0.0, 1e5); self.fmin_in.setValue(0.0); self.fmin_in.setSuffix(" Hz")
        self.fmax_in = QDoubleSpinBox(); self.fmax_in.setRange(0.1, 1e6); self.fmax_in.setValue(500.0); self.fmax_in.setSuffix(" Hz")
        self.nw_in = QSpinBox(); self.nw_in.setRange(5, 4000); self.nw_in.setValue(300)
        self.fmin_in.setToolTip("Minimum frequency of evaluation band (Hz).")
        self.fmax_in.setToolTip("Maximum frequency of evaluation band (Hz).")
        self.nw_in.setToolTip("Number of frequency points for FRF averaging (higher = slower).")
        freq_form.addRow("f min:", self.fmin_in)
        freq_form.addRow("f max:", self.fmax_in)
        freq_form.addRow("points:", self.nw_in)
        layout.addWidget(freq_box)

        # Initialize with one layer matching single-layer inputs (for convenience)
        self._add_layer_row(default_from_single=True)

        # Cross-section preview
        preview_box = QGroupBox("Cross-Section Preview")
        pv_layout = QVBoxLayout(preview_box)
        self.cross_view = _CrossSectionView()
        pv_layout.addWidget(self.cross_view)
        layout.addWidget(preview_box)
        self._update_cross_preview()
        # update preview when single-layer fields change
        self.h_in.valueChanged.connect(self._update_cross_preview)
        self.b_in.valueChanged.connect(self._update_cross_preview)
        self.E_in.valueChanged.connect(self._update_cross_preview)
        self.rho_in.valueChanged.connect(self._update_cross_preview)

    def _build_targets_tab(self):
        layout = QVBoxLayout(self.targets_tab)

        # Quantity selection
        q_box = QGroupBox("Control Quantity")
        q_form = QFormLayout(q_box)
        self.quantity_in = QComboBox(); self.quantity_in.addItems(["displacement", "velocity", "acceleration"])
        self.quantity_in.setToolTip("Select the controlled response quantity to meet targets and bounds.")
        q_form.addRow("Quantity:", self.quantity_in)
        layout.addWidget(q_box)

        # Points/regions definition
        loc_box = QGroupBox("Points / Regions")
        loc_form = QFormLayout(loc_box)
        self.locs_in = QLineEdit(); self.locs_in.setPlaceholderText("e.g. 0.8, 1.0 or 0.6-1.0:5"); self.locs_in.setToolTip("Locations in meters along the beam where targets/constraints apply. Use CSV or a-b:n format for ranges.")
        self.weights_in = QLineEdit(); self.weights_in.setPlaceholderText("e.g. 1,1 or single value to broadcast"); self.weights_in.setToolTip("Per-location weights for the objective (single value broadcasts).")
        self.targets_in = QLineEdit(); self.targets_in.setPlaceholderText("target magnitudes (broadcast ok)"); self.targets_in.setToolTip("Desired magnitude for the selected quantity at each location (single value broadcasts).")
        self.lower_in = QLineEdit(); self.lower_in.setPlaceholderText("optional lower bounds (broadcast ok)"); self.lower_in.setToolTip("Optional lower bound per location; violations are penalized.")
        self.upper_in = QLineEdit(); self.upper_in.setPlaceholderText("optional upper bounds (broadcast ok)"); self.upper_in.setToolTip("Optional upper bound per location; violations are penalized.")
        loc_form.addRow("Locations:", self.locs_in)
        loc_form.addRow("Weights:", self.weights_in)
        loc_form.addRow("Target values:", self.targets_in)
        loc_form.addRow("Lower bounds:", self.lower_in)
        loc_form.addRow("Upper bounds:", self.upper_in)
        layout.addWidget(loc_box)

    def _build_optimization_tab(self):
        layout = QVBoxLayout(self.opt_tab)

        # Mode selection
        mode_box = QGroupBox("Optimization Mode")
        mode_form = QFormLayout(mode_box)
        self.mode_in = QComboBox(); self.mode_in.addItems(["Values at specified locations", "Placement + values"])
        self.mode_in.setToolTip("Choose whether to only optimize k/c values at your specified locations or to also optimize the placements.")
        mode_form.addRow("Mode:", self.mode_in)
        layout.addWidget(mode_box)

        # Mode 1: locations
        m1_box = QGroupBox("Mode 1 Inputs (fixed locations)")
        m1_form = QFormLayout(m1_box)
        self.spring_locs_in = QLineEdit(); self.spring_locs_in.setPlaceholderText("spring x-locs CSV or ranges"); self.spring_locs_in.setToolTip("Fixed spring locations for Mode 1. Use CSV or ranges (a-b:n).")
        self.damper_locs_in = QLineEdit(); self.damper_locs_in.setPlaceholderText("damper x-locs CSV or ranges"); self.damper_locs_in.setToolTip("Fixed damper locations for Mode 1. Use CSV or ranges (a-b:n).")
        m1_form.addRow("Spring locations:", self.spring_locs_in)
        m1_form.addRow("Damper locations:", self.damper_locs_in)
        layout.addWidget(m1_box)

        # Mode 2: counts
        m2_box = QGroupBox("Mode 2 Inputs (optimize placement + values)")
        m2_form = QFormLayout(m2_box)
        self.num_springs_in = QSpinBox(); self.num_springs_in.setRange(0, 50); self.num_springs_in.setValue(1); self.num_springs_in.setToolTip("Number of springs to place (Mode 2).")
        self.num_dampers_in = QSpinBox(); self.num_dampers_in.setRange(0, 50); self.num_dampers_in.setValue(1); self.num_dampers_in.setToolTip("Number of dampers to place (Mode 2).")
        self.min_sep_in = QDoubleSpinBox(); self.min_sep_in.setRange(0.0, 1e3); self.min_sep_in.setValue(0.0); self.min_sep_in.setSuffix(" m"); self.min_sep_in.setToolTip("Minimum spacing enforced between adjacent devices (Mode 2).")
        m2_form.addRow("Num springs:", self.num_springs_in)
        m2_form.addRow("Num dampers:", self.num_dampers_in)
        m2_form.addRow("Min separation:", self.min_sep_in)
        layout.addWidget(m2_box)

        # Bounds
        b_box = QGroupBox("Bounds")
        b_form = QFormLayout(b_box)
        self.kmin_in = QDoubleSpinBox(); self.kmin_in.setRange(0.0, 1e12); self.kmin_in.setDecimals(3); self.kmin_in.setValue(0.0); self.kmin_in.setToolTip("Lower bound for spring stiffness k (N/m).")
        self.kmax_in = QDoubleSpinBox(); self.kmax_in.setRange(0.0, 1e12); self.kmax_in.setDecimals(3); self.kmax_in.setValue(1e7); self.kmax_in.setToolTip("Upper bound for spring stiffness k (N/m).")
        self.cmin_in = QDoubleSpinBox(); self.cmin_in.setRange(0.0, 1e12); self.cmin_in.setDecimals(3); self.cmin_in.setValue(0.0); self.cmin_in.setToolTip("Lower bound for damper coefficient c (N·s/m).")
        self.cmax_in = QDoubleSpinBox(); self.cmax_in.setRange(0.0, 1e12); self.cmax_in.setDecimals(3); self.cmax_in.setValue(1e5); self.cmax_in.setToolTip("Upper bound for damper coefficient c (N·s/m).")
        b_form.addRow("k min:", self.kmin_in)
        b_form.addRow("k max:", self.kmax_in)
        b_form.addRow("c min:", self.cmin_in)
        b_form.addRow("c max:", self.cmax_in)
        layout.addWidget(b_box)

        # Run button
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Optimization")
        self.run_btn.setToolTip("Start optimization in a background thread without freezing the UI.")
        self.run_btn.clicked.connect(self._on_run)
        run_row.addStretch(1)
        run_row.addWidget(self.run_btn)
        layout.addLayout(run_row)

    def _build_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        self.results_view = QTextEdit(); self.results_view.setReadOnly(True)
        self.results_view.setToolTip("Optimization logs and final recommended spring/damper placements and values.")
        layout.addWidget(self.results_view)

    # ----------------------------- Actions -----------------------------------
    def _collect_model(self) -> BeamModel:
        # Gather layers if any rows exist
        layers = []
        for row in self.layers_rows:
            t = row.findChild(QDoubleSpinBox, "t")
            E = row.findChild(QDoubleSpinBox, "E")
            rho = row.findChild(QDoubleSpinBox, "rho")
            if t is not None and E is not None and rho is not None:
                if t.value() > 0.0:
                    layers.append({"thickness": t.value(), "E": E.value(), "rho": rho.value()})

        use_layers = len(layers) > 0

        return BeamModel(
            length=self.L_in.value(),
            width=self.b_in.value(),
            thickness=None if use_layers else self.h_in.value(),
            youngs_modulus=None if use_layers else self.E_in.value(),
            density=None if use_layers else self.rho_in.value(),
            num_elements=self.N_in.value(),
            rayleigh_alpha=self.alpha_in.value(),
            rayleigh_beta=self.beta_in.value(),
            layers=layers if use_layers else None,
        )

    # --------------------------- Layers helpers -------------------------------
    def _add_layer_row(self, default_from_single: bool = False):
        row = QFrame()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        t = QDoubleSpinBox(objectName="t"); t.setRange(1e-6, 1.0); t.setDecimals(6); t.setSuffix(" m"); t.setToolTip("Layer thickness (m)")
        E = QDoubleSpinBox(objectName="E"); E.setRange(1e6, 1e13); E.setDecimals(0); E.setToolTip("Layer Young's modulus (Pa)")
        rho = QDoubleSpinBox(objectName="rho"); rho.setRange(10, 5e4); rho.setDecimals(0); rho.setSuffix(" kg/m^3"); rho.setToolTip("Layer density (kg/m^3)")
        if default_from_single:
            t.setValue(self.h_in.value())
            E.setValue(self.E_in.value())
            rho.setValue(self.rho_in.value())
        else:
            t.setValue(0.005)
            E.setValue(70e9)
            rho.setValue(2700)
        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove this layer")

        def _remove():
            self.layers_container.removeWidget(row)
            row.deleteLater()
            try:
                self.layers_rows.remove(row)
            except ValueError:
                pass

        remove_btn.clicked.connect(_remove)

        hl.addWidget(t)
        hl.addWidget(E)
        hl.addWidget(rho)
        hl.addWidget(remove_btn)
        self.layers_container.addWidget(row)
        self.layers_rows.append(row)
        # Update preview on change
        t.valueChanged.connect(self._update_cross_preview)
        E.valueChanged.connect(self._update_cross_preview)
        rho.valueChanged.connect(self._update_cross_preview)

    def _clear_layers(self):
        for row in list(self.layers_rows):
            self.layers_container.removeWidget(row)
            row.deleteLater()
        self.layers_rows.clear()
        self._update_cross_preview()

    def _update_cross_preview(self):
        # Build layer list for preview or fallback to single layer
        layers = []
        for row in self.layers_rows:
            t = row.findChild(QDoubleSpinBox, "t")
            E = row.findChild(QDoubleSpinBox, "E")
            rho = row.findChild(QDoubleSpinBox, "rho")
            if t and E and rho and t.value() > 0:
                layers.append((t.value(), E.value(), rho.value()))
        if layers:
            self.cross_view.set_layers(self.b_in.value(), layers)
        else:
            self.cross_view.set_layers(self.b_in.value(), [(self.h_in.value(), self.E_in.value(), self.rho_in.value())])
        self.cross_view.update()

    # ----------------------------- Loading UI --------------------------------
    def _build_loading_tab(self):
        layout = QVBoxLayout(self.loading_tab)

        # Load kind
        kind_box = QGroupBox("Load Definition")
        kind_form = QFormLayout(kind_box)
        self.load_type_in = QComboBox(); self.load_type_in.addItems(["Point", "Distributed region"])
        self.load_type_in.setToolTip("Choose load type: point force (N) or distributed line load (N/m) over a region.")

        # Point controls
        self.point_x_in = QDoubleSpinBox(); self.point_x_in.setRange(0.0, 1e6); self.point_x_in.setDecimals(6); self.point_x_in.setSuffix(" m"); self.point_x_in.setToolTip("Point load location along the beam (m).")
        self.point_amp_in = QDoubleSpinBox(); self.point_amp_in.setRange(-1e9, 1e9); self.point_amp_in.setDecimals(3); self.point_amp_in.setToolTip("Point load amplitude (N) scale for the chosen profile.")

        # Region controls
        self.reg_x1_in = QDoubleSpinBox(); self.reg_x1_in.setRange(0.0, 1e9); self.reg_x1_in.setDecimals(6); self.reg_x1_in.setSuffix(" m"); self.reg_x1_in.setToolTip("Region start (m).")
        self.reg_x2_in = QDoubleSpinBox(); self.reg_x2_in.setRange(0.0, 1e9); self.reg_x2_in.setDecimals(6); self.reg_x2_in.setSuffix(" m"); self.reg_x2_in.setToolTip("Region end (m).")
        self.reg_q_in = QDoubleSpinBox(); self.reg_q_in.setRange(-1e9, 1e9); self.reg_q_in.setDecimals(3); self.reg_q_in.setToolTip("Uniform distributed load intensity (N/m) scale for the chosen profile.")

        # Profile controls
        self.profile_in = QComboBox(); self.profile_in.addItems(["Impulse (broadband)", "Harmonic (single f)", "Nth harmonic of base f"])
        self.profile_in.setToolTip("Time profile converted to frequency-domain amplitude: impulse = flat; harmonic = narrowband; nth harmonic = narrowband at n*f_base.")
        self.f0_in = QDoubleSpinBox(); self.f0_in.setRange(0.0, 1e6); self.f0_in.setDecimals(3); self.f0_in.setSuffix(" Hz"); self.f0_in.setToolTip("Harmonic frequency f0 (Hz).")
        self.n_in = QSpinBox(); self.n_in.setRange(1, 128); self.n_in.setToolTip("Harmonic index n when using Nth harmonic of base frequency.")
        self.fbase_in = QDoubleSpinBox(); self.fbase_in.setRange(0.0, 1e6); self.fbase_in.setDecimals(3); self.fbase_in.setSuffix(" Hz"); self.fbase_in.setToolTip("Base frequency f_base (Hz) for n-th harmonic.")
        self.band_in = QDoubleSpinBox(); self.band_in.setRange(0.1, 1e6); self.band_in.setDecimals(3); self.band_in.setValue(5.0); self.band_in.setSuffix(" Hz"); self.band_in.setToolTip("Harmonic bandwidth (Hz) for Gaussian shaping around target frequency.")

        kind_form.addRow("Load type:", self.load_type_in)
        kind_form.addRow("Point x:", self.point_x_in)
        kind_form.addRow("Point amplitude (N):", self.point_amp_in)
        kind_form.addRow("Region x1:", self.reg_x1_in)
        kind_form.addRow("Region x2:", self.reg_x2_in)
        kind_form.addRow("Region q (N/m):", self.reg_q_in)
        kind_form.addRow("Profile:", self.profile_in)
        kind_form.addRow("f0 (Hz):", self.f0_in)
        kind_form.addRow("n:", self.n_in)
        kind_form.addRow("f_base (Hz):", self.fbase_in)
        kind_form.addRow("Bandwidth (Hz):", self.band_in)
        layout.addWidget(kind_box)

        # Defaults
        self.point_x_in.setValue(1.0)
        self.point_amp_in.setValue(1.0)
        self.reg_x1_in.setValue(0.2); self.reg_x2_in.setValue(0.8); self.reg_q_in.setValue(100.0)
        self.f0_in.setValue(100.0); self.fbase_in.setValue(25.0); self.n_in.setValue(4)

    def _build_force_func(self, model: BeamModel):
        # build spectral amplitude A(omega)
        profile = self.profile_in.currentText()
        f0 = self.f0_in.value()
        fbase = self.fbase_in.value()
        n = self.n_in.value()
        band = max(1e-6, self.band_in.value())

        def amp(omega: np.ndarray) -> np.ndarray:
            # returns shape [n_w]
            if profile.startswith("Impulse"):
                return np.ones_like(omega, dtype=float)
            if profile.startswith("Harmonic (single"):
                w0 = 2 * np.pi * f0
                sigma = 2 * np.pi * band
                return np.exp(-0.5 * ((omega - w0) / sigma) ** 2)
            # Nth harmonic
            w0 = 2 * np.pi * (fbase * n)
            sigma = 2 * np.pi * band
            return np.exp(-0.5 * ((omega - w0) / sigma) ** 2)

        # build spatial load shape as nodal vector builder
        load_type = self.load_type_in.currentText()

        n_nodes = model.N + 1
        ndof = 2 * n_nodes
        Le = model.L / model.N

        if load_type.startswith("Point"):
            x0 = min(max(0.0, self.point_x_in.value()), model.L)
            idx = int(round(x0 / model.L * model.N))
            dof_w = 2 * idx
            A = self.point_amp_in.value()

            def F(omega: np.ndarray) -> np.ndarray:
                a = amp(omega) * A
                Fw = np.zeros((ndof, omega.size), dtype=float)
                Fw[dof_w, :] = a
                return Fw

            return F
        else:
            x1 = self.reg_x1_in.value(); x2 = self.reg_x2_in.value()
            if x2 < x1:
                x1, x2 = x2, x1
            x1 = max(0.0, min(model.L, x1)); x2 = max(0.0, min(model.L, x2))
            q0 = self.reg_q_in.value()

            # precompute element overlap factors
            overlaps: List[Tuple[int, float]] = []
            for e in range(model.N):
                xe1 = e * Le
                xe2 = (e + 1) * Le
                ol = max(0.0, min(x2, xe2) - max(x1, xe1))
                if ol > 0:
                    overlaps.append((e, ol / Le))

            # element consistent nodal for uniform q over full element
            Fe_full = np.array([0.5, Le / 12.0, 0.5, -Le / 12.0])  # multiplied by q*Le

            def F(omega: np.ndarray) -> np.ndarray:
                a = amp(omega) * q0
                Fw = np.zeros((ndof, omega.size), dtype=float)
                for e, phi in overlaps:
                    dof = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
                    # total nodal contributions for each frequency
                    contrib = (a * (phi * Le))[None, :] * Fe_full[:, None]
                    Fw[dof[0], :] += contrib[0, :]
                    Fw[dof[1], :] += contrib[1, :]
                    Fw[dof[2], :] += contrib[2, :]
                    Fw[dof[3], :] += contrib[3, :]
                return Fw

            return F

    def _collect_targets(self, L: float) -> List[TargetSpecification]:
        locs = _parse_positions(self.locs_in.text(), L)
        if not locs:
            return []
        n = len(locs)
        weights = _parse_array(self.weights_in.text(), default=1.0, n=n)
        targets = _parse_array(self.targets_in.text(), default=0.0, n=n)
        lowers_txt = (self.lower_in.text() or '').strip()
        uppers_txt = (self.upper_in.text() or '').strip()
        lo = _parse_array(lowers_txt, default=float('nan'), n=n) if lowers_txt else None
        hi = _parse_array(uppers_txt, default=float('nan'), n=n) if uppers_txt else None
        # replace NaNs with None-equivalent by removing array if all NaN
        lo_arr = None if (lo is None or all(np.isnan(lo))) else [0.0 if np.isnan(v) else float(v) for v in lo]  # type: ignore
        hi_arr = None if (hi is None or all(np.isnan(hi))) else [float('inf') if np.isnan(v) else float(v) for v in hi]  # type: ignore

        ineq = None
        if lo_arr is not None or hi_arr is not None:
            # fill missing side with None
            ineq = (lo_arr, hi_arr)  # type: ignore

        return [
            TargetSpecification(
                quantity=self.quantity_in.currentText(),
                locations=locs,
                weights=weights,
                target_values=targets,
                inequality=ineq,
            )
        ]

    def _omega_grid(self) -> np.ndarray:
        fmin = self.fmin_in.value()
        fmax = self.fmax_in.value()
        nf = self.nw_in.value()
        freqs = np.linspace(fmin, fmax, nf)
        omega = 2 * np.pi * freqs
        return omega

    def _on_run(self):
        # Asynchronous run to keep UI responsive
        model = self._collect_model()
        targets = self._collect_targets(model.L)
        if not targets:
            self.results_view.setPlainText("Please define at least one location/region in Targets tab.")
            return
        omega = self._omega_grid()
        force_fn = self._build_force_func(model)
        bnds = Bounds(
            k_min=self.kmin_in.value(),
            k_max=self.kmax_in.value(),
            c_min=self.cmin_in.value(),
            c_max=self.cmax_in.value(),
        )

        mode = self.mode_in.currentText()
        args = {
            "model": model,
            "targets": targets,
            "omega": omega,
            "bounds": bnds,
            "force": force_fn,
        }
        if mode == "Values at specified locations":
            args.update({
                "spring_locations": _parse_positions(self.spring_locs_in.text(), model.L),
                "damper_locations": _parse_positions(self.damper_locs_in.text(), model.L),
                "mode": 1,
            })
        else:
            args.update({
                "num_springs": self.num_springs_in.value(),
                "num_dampers": self.num_dampers_in.value(),
                "min_separation": self.min_sep_in.value() or None,
                "mode": 2,
            })

        self.results_view.setPlainText("Running optimization... This may take a moment.")
        self.run_btn.setEnabled(False)

        # Start worker thread
        self._thread = QThread(self)
        self._worker = _OptimizationWorker(args)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        # Ensure cleanup
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.failed.connect(self._thread.quit)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(lambda: self.run_btn.setEnabled(True))
        self._thread.start()

    def _on_worker_finished(self, res: dict):
        text = []
        text.append("Optimization completed.")
        text.append(f"Best objective: {res['best_objective']:.6g}")
        text.append("")
        text.append("Springs (x [m], k [N/m]):")
        for x, k in res["k_points"]:
            text.append(f"- x={x:.4g}, k={k:.4g}")
        text.append("")
        text.append("Dampers (x [m], c [N·s/m]):")
        for x, c in res["c_points"]:
            text.append(f"- x={x:.4g}, c={c:.4g}")
        text.append("")
        text.append(f"Iterations tracked: {len(res.get('history', []))}")
        self.results_view.setPlainText("\n".join(text))

        payload = {
            "k_points": res["k_points"],
            "c_points": res["c_points"],
            "objective": res["best_objective"],
        }
        self.analysis_completed.emit(payload)

    def _on_worker_failed(self, msg: str):
        self.results_view.setPlainText(f"Error: {msg}")


def create_beam_optimization_interface() -> BeamOptimizationInterface:
    return BeamOptimizationInterface()


class _OptimizationWorker(QObject):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, args: dict):
        super().__init__()
        self.args = args

    def run(self):
        try:
            if self.args.get("mode") == 1:
                res = optimize_values_at_locations(
                    model=self.args["model"],
                    spring_locations=self.args.get("spring_locations", []),
                    damper_locations=self.args.get("damper_locations", []),
                    targets=self.args["targets"],
                    omega=self.args["omega"],
                    bounds=self.args["bounds"],
                    force=self.args.get("force"),
                )
            else:
                res = optimize_placement_and_values(
                    model=self.args["model"],
                    num_springs=self.args.get("num_springs", 0),
                    num_dampers=self.args.get("num_dampers", 0),
                    targets=self.args["targets"],
                    omega=self.args["omega"],
                    bounds=self.args["bounds"],
                    min_separation=self.args.get("min_separation"),
                    force=self.args.get("force"),
                )
            self.finished.emit(res)
        except Exception as e:
            self.failed.emit(str(e))
class _CrossSectionView(QWidget):
    def __init__(self):
        super().__init__()
        self._layers: List[Tuple[float, float, float]] = []  # (t,E,rho)
        self._b: float = 0.05
        self.setMinimumHeight(140)

    def set_layers(self, b: float, layers: List[Tuple[float, float, float]]):
        self._b = max(1e-6, float(b))
        self._layers = [(float(t), float(E), float(rho)) for (t, E, rho) in layers]
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
        from PyQt5.QtCore import QRectF
        p = QPainter(self)
        p.setRenderHint(p.Antialiasing)
        w = self.width(); h = self.height()
        p.fillRect(0, 0, w, h, QColor(30, 30, 30, 30))

        if not self._layers:
            p.end(); return
        total_t = sum(max(0.0, t) for t, _, _ in self._layers)
        if total_t <= 0:
            p.end(); return

        # margin
        m = 10
        draw_w = w - 2 * m
        draw_h = h - 2 * m
        # scale thickness
        y = h - m
        # color palette by layer index
        for i, (t, E, rho) in enumerate(self._layers):
            if t <= 0:
                continue
            frac_h = (t / total_t) * draw_h
            y0 = y - frac_h
            hue = int((i * 60) % 360)
            color = QColor.fromHsv(hue, 160, 220, 160)  # semi-transparent
            p.setPen(QPen(QColor(30, 30, 30, 220)))
            p.setBrush(QBrush(color))
            rect = QRectF(m, y0, draw_w, frac_h)
            p.drawRect(rect)
            # label
            p.setPen(QPen(QColor(20, 20, 20)))
            label = f"t={t:.3g}m, E={E/1e9:.3g}GPa, ρ={rho:.0f}"
            p.drawText(rect.adjusted(4, 0, -4, -2), 0, label)
            y = y0
        p.end()
