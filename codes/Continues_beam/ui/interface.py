from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

from ..backend.model import BeamModel, TargetSpecification
from ..backend.optimizers import optimize_values_only, optimize_placement_and_values


class BeamOptimizationInterface(QWidget):
    analysis_completed = pyqtSignal(dict)

    def __init__(self, parent=None, theme: str = 'Dark'):
        super().__init__(parent)
        self.theme = theme
        self._init_state()
        self._build_ui()

    def set_theme(self, theme: str):
        self.theme = theme
        # minimal theme handling; defer to host app

    def _init_state(self):
        # Defaults
        self.length = 1.0
        self.width = 0.05
        self.thickness = 0.01
        self.E = 210e9
        self.rho = 7800
        self.num_elements = 60
        self.alpha = 0.0
        self.beta = 0.0

        self.quantity = 'displacement'
        self.omega_start = 1.0
        self.omega_stop = 500.0
        self.omega_points = 300

        self.candidate_locations = [0.25, 0.5, 0.75]
        self.num_springs = 1
        self.num_dampers = 1

        self.targets = [TargetSpecification(
            quantity='displacement',
            locations=[1.0],
            weights=[1.0],
            target_values=[0.0],
            inequality=( [0.0], [1e-3] ),
        )]

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_model_tab(), "Beam Model")
        tabs.addTab(self._build_targets_tab(), "Targets")
        tabs.addTab(self._build_optimize_values_tab(), "Optimize Values @ Locations")
        tabs.addTab(self._build_optimize_placement_tab(), "Optimize Placement + Values")
        layout.addWidget(tabs)

        # Results area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

    def _build_model_tab(self):
        w = QWidget()
        form = QFormLayout(w)

        self.length_spin = QDoubleSpinBox(); self.length_spin.setRange(0.1, 100.0); self.length_spin.setValue(self.length)
        self.width_spin = QDoubleSpinBox(); self.width_spin.setDecimals(4); self.width_spin.setRange(0.001, 1.0); self.width_spin.setValue(self.width)
        self.thick_spin = QDoubleSpinBox(); self.thick_spin.setDecimals(4); self.thick_spin.setRange(0.001, 1.0); self.thick_spin.setValue(self.thickness)
        self.E_spin = QDoubleSpinBox(); self.E_spin.setRange(1e6, 1e13); self.E_spin.setValue(self.E); self.E_spin.setDecimals(0)
        self.rho_spin = QDoubleSpinBox(); self.rho_spin.setRange(100, 30000); self.rho_spin.setValue(self.rho); self.rho_spin.setDecimals(0)
        self.N_spin = QSpinBox(); self.N_spin.setRange(10, 400); self.N_spin.setValue(self.num_elements)
        self.alpha_spin = QDoubleSpinBox(); self.alpha_spin.setRange(0.0, 100.0); self.alpha_spin.setValue(self.alpha)
        self.beta_spin = QDoubleSpinBox(); self.beta_spin.setRange(0.0, 100.0); self.beta_spin.setValue(self.beta)
        self.q_combo = QComboBox(); self.q_combo.addItems(['displacement','velocity','acceleration']); self.q_combo.setCurrentText(self.quantity)
        self.om_a = QDoubleSpinBox(); self.om_a.setRange(0.1, 1e4); self.om_a.setValue(self.omega_start)
        self.om_b = QDoubleSpinBox(); self.om_b.setRange(0.1, 1e4); self.om_b.setValue(self.omega_stop)
        self.om_n = QSpinBox(); self.om_n.setRange(10, 5000); self.om_n.setValue(self.omega_points)

        form.addRow("Length L [m]", self.length_spin)
        form.addRow("Width b [m]", self.width_spin)
        form.addRow("Thickness h [m]", self.thick_spin)
        form.addRow("Young's modulus E [Pa]", self.E_spin)
        form.addRow("Density rho [kg/m^3]", self.rho_spin)
        form.addRow("Elements N", self.N_spin)
        form.addRow("Rayleigh alpha", self.alpha_spin)
        form.addRow("Rayleigh beta", self.beta_spin)
        form.addRow("Control quantity", self.q_combo)
        form.addRow("Omega start [rad/s]", self.om_a)
        form.addRow("Omega stop [rad/s]", self.om_b)
        form.addRow("Omega points", self.om_n)
        return w

    def _build_targets_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        # Multi-point targets table
        g = QGroupBox("Targets (points/regions)")
        gl = QVBoxLayout(g)
        self.targets_table = QTableWidget(0, 5)
        self.targets_table.setHorizontalHeaderLabels(["x [m]", "weight", "target |G|", "lower |G|", "upper |G|"])
        self.targets_table.horizontalHeader().setStretchLastSection(True)
        self.targets_table.verticalHeader().setVisible(False)
        gl.addWidget(self.targets_table)

        btn_row = QWidget(); hb = QHBoxLayout(btn_row); hb.setContentsMargins(0,0,0,0)
        add_btn = QPushButton("Add Target")
        rem_btn = QPushButton("Remove Selected")
        add_btn.clicked.connect(self._add_target_row)
        rem_btn.clicked.connect(self._remove_selected_target_rows)
        hb.addWidget(add_btn); hb.addWidget(rem_btn); hb.addStretch(1)
        gl.addWidget(btn_row)

        v.addWidget(g)

        # Seed with one default row
        self._add_target_row(default=(1.0, 1.0, 0.0, 0.0, 1e-3))

        v.addStretch(1)
        return w

    def _build_optimize_values_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        self.cand_edit = QLineEdit("0.25, 0.5, 0.75")
        self.nk = QSpinBox(); self.nk.setRange(0, 10); self.nk.setValue(self.num_springs)
        self.nc = QSpinBox(); self.nc.setRange(0, 10); self.nc.setValue(self.num_dampers)
        self.kmin = QDoubleSpinBox(); self.kmin.setRange(0.0, 1e9); self.kmin.setValue(0.0)
        self.kmax = QDoubleSpinBox(); self.kmax.setRange(0.0, 1e9); self.kmax.setValue(1e7)
        self.cmin = QDoubleSpinBox(); self.cmin.setRange(0.0, 1e9); self.cmin.setValue(0.0)
        self.cmax = QDoubleSpinBox(); self.cmax.setRange(0.0, 1e9); self.cmax.setValue(1e5)

        self.run_values_btn = QPushButton("Run Values-Only Optimization")
        self.run_values_btn.clicked.connect(self._run_values_only)

        f.addRow("Candidate x list [m]", self.cand_edit)
        f.addRow("# Springs", self.nk)
        f.addRow("# Dampers", self.nc)
        f.addRow("k bounds [min, max]", self._row(self.kmin, self.kmax))
        f.addRow("c bounds [min, max]", self._row(self.cmin, self.cmax))
        f.addRow(self.run_values_btn)
        return w

    def _build_optimize_placement_tab(self):
        w = QWidget()
        f = QFormLayout(w)
        self.nk2 = QSpinBox(); self.nk2.setRange(0, 10); self.nk2.setValue(self.num_springs)
        self.nc2 = QSpinBox(); self.nc2.setRange(0, 10); self.nc2.setValue(self.num_dampers)
        self.kmin2 = QDoubleSpinBox(); self.kmin2.setRange(0.0, 1e9); self.kmin2.setValue(0.0)
        self.kmax2 = QDoubleSpinBox(); self.kmax2.setRange(0.0, 1e9); self.kmax2.setValue(1e7)
        self.cmin2 = QDoubleSpinBox(); self.cmin2.setRange(0.0, 1e9); self.cmin2.setValue(0.0)
        self.cmax2 = QDoubleSpinBox(); self.cmax2.setRange(0.0, 1e9); self.cmax2.setValue(1e5)

        self.run_place_btn = QPushButton("Run Placement + Values Optimization")
        self.run_place_btn.clicked.connect(self._run_place_values)

        f.addRow("# Springs", self.nk2)
        f.addRow("# Dampers", self.nc2)
        f.addRow("k bounds [min, max]", self._row(self.kmin2, self.kmax2))
        f.addRow("c bounds [min, max]", self._row(self.cmin2, self.cmax2))
        f.addRow(self.run_place_btn)
        return w

    def _row(self, *widgets):
        cont = QWidget(); h = QHBoxLayout(cont); h.setContentsMargins(0,0,0,0)
        for w in widgets:
            h.addWidget(w)
        return cont

    def _build_model(self) -> BeamModel:
        return BeamModel(
            length=self.length_spin.value(),
            width=self.width_spin.value(),
            thickness=self.thick_spin.value(),
            youngs_modulus=self.E_spin.value(),
            density=self.rho_spin.value(),
            num_elements=self.N_spin.value(),
            rayleigh_alpha=self.alpha_spin.value(),
            rayleigh_beta=self.beta_spin.value(),
        )

    def _build_targets(self) -> list:
        q = self.q_combo.currentText()
        locs, wts, vals, los, his = [], [], [], [], []
        for r in range(self.targets_table.rowCount()):
            try:
                x = float(self.targets_table.item(r, 0).text())
                w = float(self.targets_table.item(r, 1).text())
                v = float(self.targets_table.item(r, 2).text())
                lo = float(self.targets_table.item(r, 3).text())
                hi = float(self.targets_table.item(r, 4).text())
            except Exception:
                continue
            locs.append(x); wts.append(w); vals.append(v); los.append(lo); his.append(hi)
        if not locs:
            # Fallback default
            locs, wts, vals, los, his = [1.0], [1.0], [0.0], [0.0], [1e-3]
        return [TargetSpecification(quantity=q, locations=locs, weights=wts, target_values=vals, inequality=(los, his))]

    def _add_target_row(self, default=None):
        r = self.targets_table.rowCount()
        self.targets_table.insertRow(r)
        if default is None:
            default = (1.0, 1.0, 0.0, 0.0, 1e-3)
        vals = [str(default[0]), str(default[1]), str(default[2]), str(default[3]), str(default[4])]
        for c, s in enumerate(vals):
            item = QTableWidgetItem(s)
            item.setTextAlignment(Qt.AlignCenter)
            self.targets_table.setItem(r, c, item)

    def _remove_selected_target_rows(self):
        rows = sorted({i.row() for i in self.targets_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.targets_table.removeRow(r)

    def _omega(self) -> np.ndarray:
        return np.linspace(self.om_a.value(), self.om_b.value(), int(self.om_n.value()))

    def _run_values_only(self):
        try:
            model = self._build_model()
            omega = self._omega()
            cand = [float(s.strip()) for s in self.cand_edit.text().split(',') if s.strip()]
            result = optimize_values_only(
                model=model,
                candidate_locations=cand,
                num_springs=int(self.nk.value()),
                num_dampers=int(self.nc.value()),
                targets=self._build_targets(),
                omega=omega,
                k_bounds=(self.kmin.value(), self.kmax.value()),
                c_bounds=(self.cmin.value(), self.cmax.value()),
            )
            self._report_result(result, model, omega)
        except Exception as e:
            QMessageBox.critical(self, "Optimization Error", str(e))

    def _run_place_values(self):
        try:
            model = self._build_model()
            omega = self._omega()
            result = optimize_placement_and_values(
                model=model,
                num_springs=int(self.nk2.value()),
                num_dampers=int(self.nc2.value()),
                targets=self._build_targets(),
                omega=omega,
                k_bounds=(self.kmin2.value(), self.kmax2.value()),
                c_bounds=(self.cmin2.value(), self.cmax2.value()),
            )
            self._report_result(result, model, omega)
        except Exception as e:
            QMessageBox.critical(self, "Optimization Error", str(e))

    def _report_result(self, result: dict, model: BeamModel, omega: np.ndarray):
        # Basic text report
        lines = ["Optimization complete:"]
        lines.append(f"Best objective: {result['best_objective']:.6e}")
        lines.append("Springs (x, k):")
        for x, k in result['k_points']:
            lines.append(f"  x={x:.4f} m  k={k:.3e} N/m")
        lines.append("Dampers (x, c):")
        for x, c in result['c_points']:
            lines.append(f"  x={x:.4f} m  c={c:.3e} N·s/m")
        self.results_text.setPlainText("\n".join(lines))

        # Emit results for host app if needed
        payload = dict(result)
        payload['omega'] = omega
        self.analysis_completed.emit(payload)


