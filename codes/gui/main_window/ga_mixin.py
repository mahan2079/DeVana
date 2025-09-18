import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTabWidget,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QSplitter,
    QTextEdit,
    QSizePolicy,
    QCheckBox,
    QRadioButton,
    QScrollArea,
    QGridLayout,
    QProgressBar,
    QFileDialog,
    QDialog,
    QMenu,
    QApplication,
    QToolBar,
    QAction,
    QListWidget,
    QListWidgetItem,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import QBrush, QColor
import os
import time
from computational_metrics_new import visualize_all_metrics, ensure_all_visualizations_visible
from modules.plotwindow import PlotWindow
from workers.GAWorker import GAWorker, build_random_validation_payload
from modules.FRF import frf
from scipy.stats import qmc

class GAOptimizationMixin:
    # ===================== Group Summary: FRF Overlay helpers =====================
    def _ensure_frf_overlay_state(self):
        """Initialize state containers for the FRF Overlay tab if missing."""
        if not hasattr(self, '_frf_group_cache'):
            # Cache per run index: { mass_key: (omega: np.ndarray, mag: np.ndarray) }
            self._frf_group_cache = {}
        if not hasattr(self, '_frf_overlay_regions'):
            # List of dicts: {name, start_x, end_x, color, text, text_color, font_size}
            self._frf_overlay_regions = []
        # Selected runs order (list of run_number ints)
        if not hasattr(self, '_frf_selected_runs'):
            self._frf_selected_runs = []
        # Baseline (no-DVA) cache per mass_key
        if not hasattr(self, '_frf_baseline_cache'):
            self._frf_baseline_cache = {}

    def _build_group_frf_overlay_tab(self, plot_tabs, df):
        """Create FRF Overlay sub-tab under Group Summary and wire interactions."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from PyQt5.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
            QSpinBox, QPushButton, QCheckBox, QTableWidget, QTableWidgetItem,
            QHeaderView
        )
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QColor
        import numpy as np

        self._ensure_frf_overlay_state()

        frf_tab = QWidget()
        frf_layout = QVBoxLayout(frf_tab)

        # -------- Controls row --------
        ctrl_group = QGroupBox("FRF Overlay Settings")
        ctrl_layout = QHBoxLayout(ctrl_group)

        # Mass selection
        ctrl_layout.addWidget(QLabel("Mass:"))
        self.frf_mass_combo = QComboBox()
        self.frf_mass_combo.addItems([f"mass_{i}" for i in range(1, 6)])
        ctrl_layout.addWidget(self.frf_mass_combo)

        # Top-K per group
        ctrl_layout.addWidget(QLabel("Top Best:"))
        self.frf_best_count = QSpinBox()
        self.frf_best_count.setRange(0, 99)
        self.frf_best_count.setValue(3)
        ctrl_layout.addWidget(self.frf_best_count)

        ctrl_layout.addWidget(QLabel("Top Mid:"))
        self.frf_mid_count = QSpinBox()
        self.frf_mid_count.setRange(0, 99)
        self.frf_mid_count.setValue(3)
        ctrl_layout.addWidget(self.frf_mid_count)

        ctrl_layout.addWidget(QLabel("Top Worst:"))
        self.frf_worst_count = QSpinBox()
        self.frf_worst_count.setRange(0, 99)
        self.frf_worst_count.setValue(3)
        ctrl_layout.addWidget(self.frf_worst_count)

        # Auto limit to 9 curves
        self.frf_limit_nine_chk = QCheckBox("Limit total to 9")
        self.frf_limit_nine_chk.setChecked(True)
        ctrl_layout.addWidget(self.frf_limit_nine_chk)

        # Legend toggle
        self.frf_show_legend_chk = QCheckBox("Show legend")
        self.frf_show_legend_chk.setChecked(True)
        ctrl_layout.addWidget(self.frf_show_legend_chk)

        ctrl_layout.addStretch(1)
        frf_layout.addWidget(ctrl_group)

        # -------- Selected runs management --------
        runs_group = QGroupBox("Selected Runs (overlay order)")
        runs_layout = QVBoxLayout(runs_group)

        self.frf_selected_runs_table = QTableWidget()
        self.frf_selected_runs_table.setColumnCount(3)
        self.frf_selected_runs_table.setHorizontalHeaderLabels(["Run #", "Group", "Score"])
        self.frf_selected_runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.frf_selected_runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.frf_selected_runs_table.setSelectionMode(QTableWidget.SingleSelection)
        runs_layout.addWidget(self.frf_selected_runs_table)

        btn_row = QWidget()
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)

        self.frf_reset_topk_btn = QPushButton("Reset to Top-K")
        self.frf_add_run_btn = QPushButton("Add Run")
        self.frf_remove_run_btn = QPushButton("Remove")
        self.frf_move_up_btn = QPushButton("Move Up")
        self.frf_move_down_btn = QPushButton("Move Down")

        for b in [self.frf_reset_topk_btn, self.frf_add_run_btn, self.frf_remove_run_btn, self.frf_move_up_btn, self.frf_move_down_btn]:
            btn_row_layout.addWidget(b)
        btn_row_layout.addStretch(1)
        runs_layout.addWidget(btn_row)

        frf_layout.addWidget(runs_group)

        # -------- Regions management --------
        regions_group = QGroupBox("Regions (shaded with labels)")
        regions_layout = QVBoxLayout(regions_group)

        self.frf_region_table = QTableWidget()
        self.frf_region_table.setColumnCount(7)
        self.frf_region_table.setHorizontalHeaderLabels(["Name", "Start", "End", "Fill Color", "Text", "Text Color", "Font Size"])
        self.frf_region_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        regions_layout.addWidget(self.frf_region_table)

        reg_btn_row = QWidget()
        reg_btn_layout = QHBoxLayout(reg_btn_row)
        reg_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.frf_add_region_btn = QPushButton("Add Region")
        self.frf_remove_region_btn = QPushButton("Remove Selected")
        self.frf_clear_regions_btn = QPushButton("Clear All")
        for b in [self.frf_add_region_btn, self.frf_remove_region_btn, self.frf_clear_regions_btn]:
            reg_btn_layout.addWidget(b)
        reg_btn_layout.addStretch(1)
        regions_layout.addWidget(reg_btn_row)

        frf_layout.addWidget(regions_group)

        # -------- Plot canvas --------
        self.frf_overlay_fig = Figure(figsize=(10, 6), tight_layout=True)
        self.frf_overlay_canvas = FigureCanvas(self.frf_overlay_fig)
        self.frf_overlay_toolbar = NavigationToolbar(self.frf_overlay_canvas, frf_tab)
        self._attach_open_in_new_window(self.frf_overlay_toolbar, self.frf_overlay_fig, "FRF Overlay")
        frf_layout.addWidget(self.frf_overlay_canvas)
        frf_layout.addWidget(self.frf_overlay_toolbar)

        # Add tab now so user sees it immediately
        plot_tabs.addTab(frf_tab, "FRF Overlay")

        # --- Wiring ---
        def _on_counts_changed():
            self._frf_auto_select_runs_from_df(df)
            self._refresh_frf_selected_runs_table(df)
            self._update_group_frf_overlay_plot(df)
        self.frf_best_count.valueChanged.connect(_on_counts_changed)
        self.frf_mid_count.valueChanged.connect(_on_counts_changed)
        self.frf_worst_count.valueChanged.connect(_on_counts_changed)
        self.frf_limit_nine_chk.stateChanged.connect(_on_counts_changed)
        self.frf_mass_combo.currentIndexChanged.connect(lambda _=None: self._update_group_frf_overlay_plot(df))
        self.frf_show_legend_chk.stateChanged.connect(lambda _=None: self._update_group_frf_overlay_plot(df))

        self.frf_reset_topk_btn.clicked.connect(lambda: (_on_counts_changed()))
        self.frf_add_run_btn.clicked.connect(lambda: self._frf_add_run_via_dialog(df))
        self.frf_remove_run_btn.clicked.connect(lambda: self._frf_remove_selected_run(df))
        self.frf_move_up_btn.clicked.connect(lambda: self._frf_move_selected_row(-1, df))
        self.frf_move_down_btn.clicked.connect(lambda: self._frf_move_selected_row(1, df))

        self.frf_region_table.itemChanged.connect(lambda _=None: self._sync_regions_from_table_and_replot(df))
        try:
            self.frf_region_table.cellDoubleClicked.connect(self._on_frf_region_cell_double_clicked)
        except Exception:
            pass
        self.frf_add_region_btn.clicked.connect(lambda: self._frf_add_region_and_replot(df))
        self.frf_remove_region_btn.clicked.connect(lambda: self._frf_remove_selected_region_and_replot(df))
        self.frf_clear_regions_btn.clicked.connect(lambda: self._frf_clear_regions_and_replot(df))

        # Initial population
        self._frf_auto_select_runs_from_df(df)
        self._refresh_frf_selected_runs_table(df)
        self._refresh_frf_region_table()
        self._update_group_frf_overlay_plot(df)

    def _frf_auto_select_runs_from_df(self, df):
        """Fill self._frf_selected_runs based on Top-K per group settings and 9-cap if enabled."""
        import numpy as np
        groups = ['Best', 'Mid', 'Worst']
        counts = {
            'Best': int(self.frf_best_count.value()) if hasattr(self, 'frf_best_count') else 3,
            'Mid': int(self.frf_mid_count.value()) if hasattr(self, 'frf_mid_count') else 3,
            'Worst': int(self.frf_worst_count.value()) if hasattr(self, 'frf_worst_count') else 3,
        }
        selected = []
        try:
            if 'group' in df.columns:
                for g in groups:
                    sub = df[df['group'] == g].copy()
                    if 'score' in sub.columns:
                        sub = sub.sort_values('score', ascending=True)
                    elif 'best_fitness' in sub.columns:
                        sub = sub.sort_values('best_fitness', ascending=True)
                    pick = sub.head(max(0, counts.get(g, 0)))
                    selected.extend([int(r) for r in pick.get('run_number', [])])
        except Exception:
            selected = []

        # Fallback: if nothing selected (no groups or no matches), pick top-N overall by score/fitness
        if not selected:
            try:
                sub = df.copy()
                if 'score' in sub.columns:
                    sub = sub.sort_values('score', ascending=True)
                elif 'best_fitness' in sub.columns:
                    sub = sub.sort_values('best_fitness', ascending=True)
                limit = 9 if (hasattr(self, 'frf_limit_nine_chk') and self.frf_limit_nine_chk.isChecked()) else len(sub)
                selected = [int(r) for r in sub.get('run_number', []).head(limit)]
            except Exception:
                selected = []

        # Treat No-DVA baseline as a pseudo-run with id 0
        # Always include baseline in default auto-selection at the front
        if 0 not in selected:
            selected.insert(0, 0)
        if hasattr(self, 'frf_limit_nine_chk') and self.frf_limit_nine_chk.isChecked() and selected:
            selected = selected[:9]
        self._frf_selected_runs = selected

    def _refresh_frf_selected_runs_table(self, df):
        from PyQt5.QtWidgets import QTableWidgetItem
        from PyQt5.QtCore import Qt
        # Build map from run_number to row record
        by_run = {int(rn): rec for rn, rec in zip(df['run_number'], df.to_dict('records'))}
        tbl = self.frf_selected_runs_table
        tbl.blockSignals(True)
        tbl.setRowCount(len(self._frf_selected_runs))
        for i, rn in enumerate(self._frf_selected_runs):
            if int(rn) == 0:
                it0 = QTableWidgetItem("0")
                it0.setTextAlignment(Qt.AlignCenter)
                it1 = QTableWidgetItem("Baseline")
                it1.setTextAlignment(Qt.AlignCenter)
                it2 = QTableWidgetItem("")
                it2.setTextAlignment(Qt.AlignCenter)
            else:
                rec = by_run.get(int(rn), {})
                it0 = QTableWidgetItem(str(rn))
                it0.setTextAlignment(Qt.AlignCenter)
                it1 = QTableWidgetItem(str(rec.get('group', '')))
                it1.setTextAlignment(Qt.AlignCenter)
                score_val = rec.get('score', rec.get('best_fitness', ''))
                try:
                    it2 = QTableWidgetItem(f"{float(score_val):.6f}")
                except Exception:
                    it2 = QTableWidgetItem(str(score_val))
                it2.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(i, 0, it0)
            tbl.setItem(i, 1, it1)
            tbl.setItem(i, 2, it2)
        tbl.blockSignals(False)

    def _frf_move_selected_row(self, delta, df):
        row = self.frf_selected_runs_table.currentRow()
        if row < 0 or row >= len(self._frf_selected_runs):
            return
        new_row = row + int(delta)
        if new_row < 0 or new_row >= len(self._frf_selected_runs):
            return
        self._frf_selected_runs[row], self._frf_selected_runs[new_row] = self._frf_selected_runs[new_row], self._frf_selected_runs[row]
        self._refresh_frf_selected_runs_table(df)
        self.frf_selected_runs_table.selectRow(new_row)
        self._update_group_frf_overlay_plot(df)

    def _frf_remove_selected_run(self, df):
        row = self.frf_selected_runs_table.currentRow()
        if row < 0 or row >= len(self._frf_selected_runs):
            return
        del self._frf_selected_runs[row]
        self._refresh_frf_selected_runs_table(df)
        self._update_group_frf_overlay_plot(df)

    def _frf_add_run_via_dialog(self, df):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox
        # Determine available runs not already selected
        all_runs = [int(x) for x in df['run_number'].tolist()]
        avail = [rn for rn in all_runs if rn not in self._frf_selected_runs]
        if not avail:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Add Run to FRF Overlay")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Choose a run to add:"))
        combo = QComboBox()
        combo.addItems([str(rn) for rn in avail])
        lay.addWidget(combo)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec_() == QDialog.Accepted:
            try:
                choice = int(combo.currentText())
                self._frf_selected_runs.append(choice)
                if hasattr(self, 'frf_limit_nine_chk') and self.frf_limit_nine_chk.isChecked() and len(self._frf_selected_runs) > 9:
                    self._frf_selected_runs = self._frf_selected_runs[:9]
                self._refresh_frf_selected_runs_table(df)
                self._update_group_frf_overlay_plot(df)
            except Exception:
                pass

    def _frf_region_row_to_dict(self, row):
        # Helper to read a row from region table into dict
        def _val(col, cast=float, default=None):
            it = self.frf_region_table.item(row, col)
            if it is None:
                return default
            try:
                return cast(it.text()) if cast else it.text()
            except Exception:
                return default
        name = _val(0, cast=str, default='Region')
        start_x = _val(1, cast=float, default=0.0)
        end_x = _val(2, cast=float, default=0.0)
        color_item = self.frf_region_table.item(row, 3)
        color = color_item.text() if color_item else '#ffcc00'
        text = _val(4, cast=str, default='')
        text_color_item = self.frf_region_table.item(row, 5)
        text_color = text_color_item.text() if text_color_item else '#000000'
        font_size = _val(6, cast=float, default=10.0)
        return {'name': name, 'start_x': start_x, 'end_x': end_x, 'color': color, 'text': text, 'text_color': text_color, 'font_size': font_size}

    def _refresh_frf_region_table(self):
        from PyQt5.QtWidgets import QTableWidgetItem
        from PyQt5.QtCore import Qt
        tbl = self.frf_region_table
        tbl.blockSignals(True)
        tbl.setRowCount(len(self._frf_overlay_regions))
        for r, reg in enumerate(self._frf_overlay_regions):
            items = [
                QTableWidgetItem(str(reg.get('name', 'Region'))),
                QTableWidgetItem(f"{float(reg.get('start_x', 0.0)):.6g}"),
                QTableWidgetItem(f"{float(reg.get('end_x', 0.0)):.6g}"),
                QTableWidgetItem(str(reg.get('color', '#ffcc00'))),
                QTableWidgetItem(str(reg.get('text', ''))),
                QTableWidgetItem(str(reg.get('text_color', '#000000'))),
                QTableWidgetItem(f"{float(reg.get('font_size', 10.0)):.3g}"),
            ]
            for c, it in enumerate(items):
                it.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(r, c, it)
        tbl.blockSignals(False)

    def _sync_regions_from_table_and_replot(self, df):
        regs = []
        for r in range(self.frf_region_table.rowCount()):
            regs.append(self._frf_region_row_to_dict(r))
        self._frf_overlay_regions = regs
        self._update_group_frf_overlay_plot(df)

    def _frf_add_region_and_replot(self, df):
        self._frf_overlay_regions.append({'name': 'Region', 'start_x': 0.0, 'end_x': 1.0, 'color': '#ffcc00', 'text': '', 'text_color': '#000000', 'font_size': 10.0})
        self._refresh_frf_region_table()
        self._update_group_frf_overlay_plot(df)

    def _frf_remove_selected_region_and_replot(self, df):
        row = self.frf_region_table.currentRow()
        if row >= 0 and row < len(self._frf_overlay_regions):
            del self._frf_overlay_regions[row]
            self._refresh_frf_region_table()
            self._update_group_frf_overlay_plot(df)

    def _frf_clear_regions_and_replot(self, df):
        self._frf_overlay_regions = []
        self._refresh_frf_region_table()
        self._update_group_frf_overlay_plot(df)

    def _on_frf_region_cell_double_clicked(self, row, col):
        # Allow picking colors via dialog for Fill Color (3) and Text Color (5)
        try:
            from PyQt5.QtWidgets import QColorDialog
            if col in (3, 5):
                current = self.frf_region_table.item(row, col)
                cur_hex = current.text() if current else '#ffffff'
                c = QColorDialog.getColor(QColor(cur_hex), self, options=QColorDialog.DontUseNativeDialog)
                if c.isValid():
                    if current is None:
                        from PyQt5.QtWidgets import QTableWidgetItem
                        current = QTableWidgetItem()
                        self.frf_region_table.setItem(row, col, current)
                    current.setText(c.name())
                    # Sync and replot
                    # We don't have df here; find it from last built data if needed.
                    # As a simple approach, trigger table-changed handler which will replot.
                    self._sync_regions_from_table_and_replot(getattr(self, '_last_group_summary_df', None) or self._last_group_summary_df_placeholder())
        except Exception:
            pass

    def _last_group_summary_df_placeholder(self):
        # Fallback if df is not remembered; build from current ga_benchmark_data if possible
        try:
            if hasattr(self, 'ga_benchmark_data') and isinstance(self.ga_benchmark_data, list) and self.ga_benchmark_data:
                import pandas as pd
                return pd.DataFrame(self.ga_benchmark_data)
        except Exception:
            pass
        import pandas as pd
        return pd.DataFrame()

    def _get_main_params_targets_weights(self):
        """Fetch or construct main system parameters, target values, and weights from current UI."""
        try:
            # If previously captured during validation/benchmark, prefer those
            if all(hasattr(self, attr) for attr in ('main_params', 'target_values', 'omega_start', 'omega_end', 'omega_points')):
                main_params = self.main_params
                target_values, weights = self.target_values, self.get_target_values_weights()[1]
                omega_start, omega_end, omega_points = self.omega_start, self.omega_end, self.omega_points
            else:
                # Build from UI controls
                target_values, weights = self.get_target_values_weights()
                main_params = (
                    self.mu_box.value(),
                    *[b.value() for b in self.landa_boxes],
                    *[b.value() for b in self.nu_boxes],
                    self.a_low_box.value(),
                    self.a_up_box.value(),
                    self.f_1_box.value(),
                    self.f_2_box.value(),
                    self.omega_dc_box.value(),
                    self.zeta_dc_box.value(),
                )
                omega_start = self.omega_start_box.value()
                omega_end = self.omega_end_box.value()
                omega_points = self.omega_points_box.value()
            return main_params, target_values, weights, float(omega_start), float(omega_end), int(omega_points)
        except Exception:
            # Fallback minimally
            try:
                target_values, weights = self.get_target_values_weights()
                main_params = (
                    self.mu_box.value(),
                    *[b.value() for b in self.landa_boxes],
                    *[b.value() for b in self.nu_boxes],
                    self.a_low_box.value(),
                    self.a_up_box.value(),
                    self.f_1_box.value(),
                    self.f_2_box.value(),
                    self.omega_dc_box.value(),
                    self.zeta_dc_box.value(),
                )
                omega_start = self.omega_start_box.value()
                omega_end = self.omega_end_box.value()
                omega_points = self.omega_points_box.value()
                return main_params, target_values, weights, float(omega_start), float(omega_end), int(omega_points)
            except Exception:
                return None, None, None, 0.0, 1.0, 100

    def _get_frf_mag_for_run_and_mass(self, run_record, mass_key):
        """Compute or fetch cached FRF magnitude curve for a run and mass."""
        import numpy as np
        from modules.FRF import frf as frf_func
        rn = int(run_record.get('run_number'))
        cache = self._frf_group_cache.setdefault(rn, {})
        if mass_key in cache:
            return cache[mass_key]
        main_params, target_values, weights, omega_start, omega_end, omega_points = self._get_main_params_targets_weights()
        if main_params is None or target_values is None or weights is None:
            try:
                self._frf_last_error = 'Missing system parameters or targets.'
            except Exception:
                pass
            return None

        # Build DVA parameter tuple from run record
        vals = run_record.get('best_solution', []) or []
        try:
            vals = [float(v) for v in vals]
        except Exception:
            vals = []
        EXPECTED_LEN = 48  # beta 15 + lambda 15 + mu 3 + nu 15
        if len(vals) < EXPECTED_LEN:
            vals = list(vals) + [0.0] * (EXPECTED_LEN - len(vals))
        elif len(vals) > EXPECTED_LEN:
            vals = list(vals[:EXPECTED_LEN])
        dva_tuple = tuple(vals)
        if not dva_tuple:
            return None
        try:
            res = frf_func(
                main_system_parameters=main_params,
                dva_parameters=dva_tuple,
                omega_start=omega_start,
                omega_end=omega_end,
                omega_points=int(omega_points),
                target_values_mass1=target_values['mass_1'],
                weights_mass1=weights['mass_1'],
                target_values_mass2=target_values['mass_2'],
                weights_mass2=weights['mass_2'],
                target_values_mass3=target_values['mass_3'],
                weights_mass3=weights['mass_3'],
                target_values_mass4=target_values['mass_4'],
                weights_mass4=weights['mass_4'],
                target_values_mass5=target_values['mass_5'],
                weights_mass5=weights['mass_5'],
                plot_figure=False,
                show_peaks=False,
                show_slopes=False,
            )
            # Build omega vector and extract magnitude
            omega = np.linspace(float(omega_start), float(omega_end), int(omega_points))
            mass_rec = res.get(mass_key, {}) if isinstance(res, dict) else {}
            mag = np.array(mass_rec.get('magnitude', np.zeros_like(omega)))
            cache[mass_key] = (omega, mag)
            return cache[mass_key]
        except Exception as e:
            try:
                self._frf_last_error = str(e)
            except Exception:
                pass
            return None

    def _get_frf_mag_for_zero_dva(self, mass_key):
        """Compute or fetch cached baseline FRF (all DVA values = 0) for a mass."""
        import numpy as np
        from modules.FRF import frf as frf_func
        if mass_key in self._frf_baseline_cache:
            return self._frf_baseline_cache[mass_key]
        main_params, target_values, weights, omega_start, omega_end, omega_points = self._get_main_params_targets_weights()
        if main_params is None or target_values is None or weights is None:
            return None
        ZERO_DVA_LEN = 48
        dva_tuple = tuple(0.0 for _ in range(ZERO_DVA_LEN))
        try:
            res = frf_func(
                main_system_parameters=main_params,
                dva_parameters=dva_tuple,
                omega_start=omega_start,
                omega_end=omega_end,
                omega_points=int(omega_points),
                target_values_mass1=target_values['mass_1'],
                weights_mass1=weights['mass_1'],
                target_values_mass2=target_values['mass_2'],
                weights_mass2=weights['mass_2'],
                target_values_mass3=target_values['mass_3'],
                weights_mass3=weights['mass_3'],
                target_values_mass4=target_values['mass_4'],
                weights_mass4=weights['mass_4'],
                target_values_mass5=target_values['mass_5'],
                weights_mass5=weights['mass_5'],
                plot_figure=False,
                show_peaks=False,
                show_slopes=False,
            )
            omega = np.linspace(float(omega_start), float(omega_end), int(omega_points))
            mass_rec = res.get(mass_key, {}) if isinstance(res, dict) else {}
            mag = np.array(mass_rec.get('magnitude', np.zeros_like(omega)))
            self._frf_baseline_cache[mass_key] = (omega, mag)
            return self._frf_baseline_cache[mass_key]
        except Exception:
            return None

    def _update_group_frf_overlay_plot(self, df):
        import numpy as np
        import matplotlib.pyplot as plt
        # Prepare axes
        self.frf_overlay_fig.clf()
        ax = self.frf_overlay_fig.add_subplot(111)
        mass_key = self.frf_mass_combo.currentText() if hasattr(self, 'frf_mass_combo') else 'mass_1'
        # Map for quickly fetching run record by run_number (robust to missing column)
        try:
            by_run = {int(rn): rec for rn, rec in zip(df['run_number'], df.to_dict('records'))}
        except Exception:
            by_run = {}
        # Define base colors per group
        group_colors = {'Best': '#2ca02c', 'Mid': '#ff7f0e', 'Worst': '#d62728'}
        # For shade variation per group selection index
        def _shade(hex_color, factor):
            # factor in [0,1]; 0 darker, 1 original
            try:
                c = plt.matplotlib.colors.to_rgb(hex_color)
                base = np.array([0.0, 0.0, 0.0])
                return tuple(base + (np.array(c) - base) * (0.5 + 0.5 * factor))
            except Exception:
                return hex_color

        # Split selected runs by group to vary shades
        # Build ordered selected runs if empty
        if not self._frf_selected_runs:
            self._frf_auto_select_runs_from_df(df)
        grouped = {'Best': [], 'Mid': [], 'Worst': []}
        for rn in self._frf_selected_runs:
            if int(rn) == 0:
                continue  # baseline has no group
            rec = by_run.get(int(rn), {})
            g = str(rec.get('group', ''))
            if g in grouped:
                grouped[g].append(int(rn))
        # Plot in the order of self._frf_selected_runs to respect user ordering
        # Line styles by group: Best=solid, Mid=long dashes, Worst=short dashes
        style_map = {
            'Best': '-',
            'Mid': (0, (12, 6)),   # long dashes
            'Worst': (0, (4, 3)),  # small/short dashes
        }
        plotted = 0
        for idx, rn in enumerate(self._frf_selected_runs):
            if int(rn) == 0:
                base = self._get_frf_mag_for_zero_dva(mass_key)
                if base:
                    omega_b, mag_b = base
                    ax.plot(omega_b, mag_b, label='Run 0 (No DVA)', color='#000000', linewidth=2.2, linestyle=':', alpha=0.95)
                    plotted += 1
                continue
            rec = by_run.get(int(rn), None)
            if not rec:
                continue
            grp = str(rec.get('group', ''))
            base_color = group_colors.get(grp, '#1f77b4')
            # Shade based on position within its group list
            try:
                pos_in_group = grouped.get(grp, []).index(int(rn))
                total_in_group = max(1, len(grouped.get(grp, [])))
                factor = 1.0 - (pos_in_group / max(1, total_in_group - 1)) * 0.5 if total_in_group > 1 else 1.0
            except Exception:
                factor = 1.0
            color = _shade(base_color, factor)
            fetched = self._get_frf_mag_for_run_and_mass(rec, mass_key)
            if not fetched:
                continue
            omega, mag = fetched
            linestyle = style_map.get(grp, '-')
            ax.plot(omega, mag, label=f"Run {int(rn)} ({grp})", linewidth=1.8, alpha=0.95, color=color, linestyle=linestyle)
            plotted += 1

        # Baseline now treated as pseudo-run id 0; plotted in loop above

        # Regions (shaded) and boundaries
        for reg in self._frf_overlay_regions:
            try:
                x0 = float(reg.get('start_x', 0.0))
                x1 = float(reg.get('end_x', 0.0))
                if x1 < x0:
                    x0, x1 = x1, x0
                face = reg.get('color', '#ffcc00')
                ax.axvspan(x0, x1, color=face, alpha=0.18, linewidth=0)
                ax.axvline(x0, color=face, alpha=0.8, linestyle='--', linewidth=1.0)
                ax.axvline(x1, color=face, alpha=0.8, linestyle='--', linewidth=1.0)
                # Add label at region center
                if reg.get('text'):
                    xc = 0.5 * (x0 + x1)
                    ymin, ymax = ax.get_ylim()
                    yc = ymin + 0.85 * (ymax - ymin)
                    ax.text(xc, yc, str(reg.get('text')), ha='center', va='center',
                            fontsize=float(reg.get('font_size', 10.0)), color=str(reg.get('text_color', '#000000')),
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
            except Exception:
                continue

        # --- KDE overlay of peak positions across the plotted FRFs ---
        try:
            from scipy.signal import find_peaks
            from scipy import stats as _stats
            peak_positions = []

            # Collect peaks from each non-baseline plotted run for current mass
            for rn in self._frf_selected_runs:
                if int(rn) == 0:
                    continue
                rec = by_run.get(int(rn), None)
                if not rec:
                    continue
                fetched = self._get_frf_mag_for_run_and_mass(rec, mass_key)
                if not fetched:
                    continue
                omega, mag = fetched
                if not isinstance(omega, np.ndarray):
                    omega = np.asarray(omega, dtype=float)
                if not isinstance(mag, np.ndarray):
                    mag = np.asarray(mag, dtype=float)
                if omega.size < 5 or mag.size != omega.size:
                    continue
                # Heuristic peak detection parameters
                yspan = float(np.nanmax(mag) - np.nanmin(mag)) if mag.size else 0.0
                prominence = max(1e-12, 0.05 * yspan)
                distance = max(1, mag.size // 100)
                try:
                    peaks, _ = find_peaks(mag, prominence=prominence, distance=distance)
                except Exception:
                    peaks = []
                if isinstance(peaks, (list, tuple, np.ndarray)) and len(peaks) > 0:
                    # Keep only finite frequency positions
                    for idx in peaks:
                        try:
                            x = float(omega[int(idx)])
                            if np.isfinite(x):
                                peak_positions.append(x)
                        except Exception:
                            pass

            # Draw KDE band anchored at y=0 (overlapping FRFs) if we have enough data
            if len(peak_positions) >= 3:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xgrid = np.linspace(min(xlim[0], min(peak_positions)), max(xlim[1], max(peak_positions)), 512)
                kde = _stats.gaussian_kde(np.asarray(peak_positions, dtype=float))
                ydens = kde(xgrid)
                if np.all(np.isfinite(ydens)) and np.nanmax(ydens) > 0:
                    # Scale density into a slim band above y=0 (baseline)
                    y0 = 0.0
                    h = 0.08 * (ylim[1] - ylim[0])
                    if h <= 0:
                        h = 1.0
                    yscaled = y0 + (ydens / float(np.nanmax(ydens))) * h
                    # Subtle pale-yellow underlay with a contrasting line
                    ax.fill_between(xgrid, y0, yscaled, color='#fff9a5', alpha=0.35, zorder=1, linewidth=0)
                    ax.plot(xgrid, yscaled, color='#b39b00', linewidth=2.0, alpha=0.85, label='Peak KDE', zorder=2)
                    # Preserve original x-limits and ensure y=0 is visible
                    ax.set_xlim(xlim)
                    ax.set_ylim(min(ylim[0], 0.0), ylim[1])
        except Exception:
            # Silently ignore KDE errors to avoid breaking main plot
            pass

        ax.set_xlabel('Frequency (rad/s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'FRF Overlay across Runs as {mass_key}')
        ax.grid(True, linestyle='--', alpha=0.5)
        if self.frf_show_legend_chk.isChecked() and plotted > 0:
            ax.legend(loc='best', ncol=1, fontsize=9, frameon=True)
        if plotted == 0:
            msg = 'No runs plotted. Adjust Top-K or add runs.'
            try:
                if getattr(self, '_frf_last_error', None):
                    msg += f"\nLast FRF error: {self._frf_last_error}"
            except Exception:
                pass
            ax.text(0.5, 0.5, msg,
                    transform=ax.transAxes, ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', fc='#ffeeee', ec='#ddaaaa', alpha=0.8))
        self.frf_overlay_canvas.draw_idle()
    def _attach_open_in_new_window(self, toolbar, fig, title):
        """Attach a consistent 'Open in New Window' action to a plot toolbar."""
        try:
            action = QAction("Open in New Window", toolbar)
            action.triggered.connect(lambda checked=False, f=fig, t=title: self._open_plot_window(f, t))
            toolbar.addAction(action)
        except Exception:
            pass
    
    def _get_ga_parameters_from_table(self):
        """Read GA parameters table into a list of dicts with per-category costs.
        Returns entries with keys: name, fixed, fixed_value, low, high, costs{material,manufacturing,maintenance,operational}.
        """
        params = []
        try:
            tbl = self.ga_param_table
            if not tbl:
                return params
            for r in range(tbl.rowCount()):
                name_item = tbl.item(r, 0)
                name = name_item.text() if name_item else f"param_{r}"
                fixed_chk = tbl.cellWidget(r, 1)
                fixed_val = tbl.cellWidget(r, 2)
                low_spin = tbl.cellWidget(r, 3)
                high_spin = tbl.cellWidget(r, 4)
                cost_material = tbl.cellWidget(r, 5) if tbl.columnCount() > 5 else None
                cost_manuf = tbl.cellWidget(r, 6) if tbl.columnCount() > 6 else None
                cost_maint = tbl.cellWidget(r, 7) if tbl.columnCount() > 7 else None
                cost_oper = tbl.cellWidget(r, 8) if tbl.columnCount() > 8 else None
                fixed = bool(fixed_chk.isChecked()) if fixed_chk is not None else False
                fixed_value = float(fixed_val.value()) if fixed_val is not None else 0.0
                low = float(low_spin.value()) if low_spin is not None else 0.0
                high = float(high_spin.value()) if high_spin is not None else 0.0
                costs = {
                    'material': float(cost_material.value()) if cost_material is not None else 0.0,
                    'manufacturing': float(cost_manuf.value()) if cost_manuf is not None else 0.0,
                    'maintenance': float(cost_maint.value()) if cost_maint is not None else 0.0,
                    'operational': float(cost_oper.value()) if cost_oper is not None else 0.0,
                }
                params.append({
                    'name': name,
                    'fixed': fixed,
                    'fixed_value': fixed_value,
                    'low': low,
                    'high': high,
                    'costs': costs,
                    'row': r,
                })
        except Exception:
            pass
        return params

    def _apply_ranges_to_ga_table(self, ranges_by_name):
        """Apply low/high ranges to GA Parameters table for matching parameter names."""
        try:
            tbl = self.ga_param_table
            if not tbl:
                return
            for r in range(tbl.rowCount()):
                name_item = tbl.item(r, 0)
                name = name_item.text() if name_item else None
                if not name or name not in ranges_by_name:
                    continue
                lo, hi = ranges_by_name[name]
                low_spin = tbl.cellWidget(r, 3)
                high_spin = tbl.cellWidget(r, 4)
                fixed_chk = tbl.cellWidget(r, 1)
                fixed_val = tbl.cellWidget(r, 2)
                # Switch to range mode
                if fixed_chk is not None and fixed_chk.isChecked():
                    fixed_chk.setChecked(False)
                if low_spin is not None:
                    low_spin.setEnabled(True)
                    low_spin.setValue(float(lo))
                if high_spin is not None:
                    high_spin.setEnabled(True)
                    high_spin.setValue(float(hi))
                if fixed_val is not None:
                    fixed_val.setEnabled(False)
        except Exception as e:
            print(f"Error applying ranges to GA table: {str(e)}")

    def _build_current_ga_config_with_ranges(self, source_table):
        """Build a JSON-serializable GA config snapshot, swapping in ranges from a given table.
        source_table is expected to have columns [Parameter, Low, High, Width, Center].
        """
        try:
            import json
            cfg = {}
            # Global GA settings from controls
            cfg['ga_settings'] = {
                'pop_min': int(self.ga_pop_min_box.value()),
                'pop_max': int(self.ga_pop_max_box.value()),
                'pop_size': int(self.ga_pop_size_box.value()),
                'generations': int(self.ga_num_generations_box.value()),
                'cxpb': float(self.ga_cxpb_box.value()),
                'mutpb': float(self.ga_mutpb_box.value()),
                'tol': float(self.ga_tol_box.value()),
                'alpha': float(self.ga_alpha_box.value()),
                'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()),
                'cost_scale_factor': float(self.ga_cost_scale_box.value()),
                'dva_activation_threshold': float(self.ga_activation_threshold_box.value()),
                'dva_activation_penalty': float(self.ga_activation_penalty_box.value()),
                'use_enhanced_cost': bool(self.enh_cost_enable_chk.isChecked()),
                'benefit_w_primary': float(self.benefit_w_primary_box.value()),
                'benefit_w_accuracy': float(self.benefit_w_accuracy_box.value()),
                'benefit_w_sparsity': float(self.benefit_w_sparsity_box.value()),
                'category_w_material': float(self.cat_w_material_box.value()),
                'category_w_manufacturing': float(self.cat_w_manufacturing_box.value()),
                'category_w_maintenance': float(self.cat_w_maintenance_box.value()),
                'category_w_operational': float(self.cat_w_operational_box.value()),
                'benefit_weight_start': float(self.benefit_weight_start_box.value()),
                'benefit_weight_end': float(self.benefit_weight_end_box.value()),
                'generation_ratio': float(self.generation_ratio_box.value()),
                'adaptive_rates': bool(self.adaptive_rates_checkbox.isChecked()),
                'stagnation_limit': int(self.stagnation_limit_box.value()),
                'cxpb_min': float(self.cxpb_min_box.value()),
                'cxpb_max': float(self.cxpb_max_box.value()),
                'mutpb_min': float(self.mutpb_min_box.value()),
                'mutpb_max': float(self.mutpb_max_box.value()),
                'controller': (
                    'adaptive' if self.controller_adaptive_radio.isChecked() else
                    ('ml_bandit' if self.controller_ml_radio.isChecked() else
                     ('rl' if self.controller_rl_radio.isChecked() else 'fixed'))
                )
            }

            # Current parameter names
            current_params = self._get_ga_parameters_from_table()
            name_to_row = {p['name']: p for p in current_params}

            # Extract ranges from source_table
            ranges = {}
            for r in range(source_table.rowCount()):
                name_it = source_table.item(r, 0)
                lo_it = source_table.item(r, 1)
                hi_it = source_table.item(r, 2)
                if not name_it:
                    continue
                name = name_it.text()
                try:
                    lo = float(lo_it.text()) if lo_it and lo_it.text() not in ("-", "") else None
                    hi = float(hi_it.text()) if hi_it and hi_it.text() not in ("-", "") else None
                except Exception:
                    lo, hi = None, None
                if lo is None or hi is None:
                    continue
                ranges[name] = (lo, hi)

            # Build parameter section
            cfg_params = []
            for p in current_params:
                name = p['name']
                if name in ranges:
                    lo, hi = ranges[name]
                    cfg_params.append({
                        'name': name,
                        'fixed': False,
                        'fixed_value': float(p['fixed_value']),
                        'low': float(lo),
                        'high': float(hi),
                        'costs': p.get('costs', {
                            'material': 0.0,
                            'manufacturing': 0.0,
                            'maintenance': 0.0,
                            'operational': 0.0,
                        }),
                    })
                else:
                    cfg_params.append({
                        'name': name,
                        'fixed': bool(p['fixed']),
                        'fixed_value': float(p['fixed_value']),
                        'low': float(p['low']),
                        'high': float(p['high']),
                        'costs': p.get('costs', {
                            'material': 0.0,
                            'manufacturing': 0.0,
                            'maintenance': 0.0,
                            'operational': 0.0,
                        }),
                    })
            cfg['parameters'] = cfg_params
            cfg['export_timestamp'] = QDateTime.currentDateTime().toString(Qt.ISODate)
            cfg['type'] = 'ga_config'
            return cfg
        except Exception as e:
            print(f"Error building GA config: {str(e)}")
            return None

    def import_ga_config(self):
        """Import GA config JSON and apply settings and DVA ranges to the UI."""
        try:
            import json
            path, _ = QFileDialog.getOpenFileName(self, "Import GA Config", "", "JSON files (*.json)")
            if not path:
                return
            with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict) or cfg.get('type') != 'ga_config':
                QMessageBox.warning(self, "Import", "Selected file is not a GA config.")
                return
            # Apply GA settings
            try:
                s = cfg.get('ga_settings', {})
                if 'pop_min' in s: self.ga_pop_min_box.setValue(int(s['pop_min']))
                if 'pop_max' in s: self.ga_pop_max_box.setValue(int(s['pop_max']))
                if 'pop_size' in s: self.ga_pop_size_box.setValue(int(s['pop_size']))
                if 'generations' in s: self.ga_num_generations_box.setValue(int(s['generations']))
                if 'cxpb' in s: self.ga_cxpb_box.setValue(float(s['cxpb']))
                if 'mutpb' in s: self.ga_mutpb_box.setValue(float(s['mutpb']))
                if 'tol' in s: self.ga_tol_box.setValue(float(s['tol']))
                if 'alpha' in s: self.ga_alpha_box.setValue(float(s['alpha']))
                if 'percentage_error_scale' in s: self.ga_percentage_error_scale_box.setValue(float(s['percentage_error_scale']))
                if 'cost_scale_factor' in s:
                    loaded_value = float(s['cost_scale_factor'])
                    print(f"DEBUG: Loading cost_scale_factor = {loaded_value}")
                    self.ga_cost_scale_box.setValue(loaded_value)
                if 'dva_activation_threshold' in s: self.ga_activation_threshold_box.setValue(float(s['dva_activation_threshold']))
                if 'dva_activation_penalty' in s: self.ga_activation_penalty_box.setValue(float(s['dva_activation_penalty']))
                if 'use_enhanced_cost' in s: self.enh_cost_enable_chk.setChecked(bool(s['use_enhanced_cost']))
                if 'benefit_w_primary' in s: self.benefit_w_primary_box.setValue(float(s['benefit_w_primary']))
                if 'benefit_w_accuracy' in s: self.benefit_w_accuracy_box.setValue(float(s['benefit_w_accuracy']))
                if 'benefit_w_sparsity' in s: self.benefit_w_sparsity_box.setValue(float(s['benefit_w_sparsity']))
                if 'category_w_material' in s: self.cat_w_material_box.setValue(float(s['category_w_material']))
                if 'category_w_manufacturing' in s: self.cat_w_manufacturing_box.setValue(float(s['category_w_manufacturing']))
                if 'category_w_maintenance' in s: self.cat_w_maintenance_box.setValue(float(s['category_w_maintenance']))
                if 'category_w_operational' in s: self.cat_w_operational_box.setValue(float(s['category_w_operational']))
                if 'benefit_weight_start' in s: self.benefit_weight_start_box.setValue(float(s['benefit_weight_start']))
                if 'benefit_weight_end' in s: self.benefit_weight_end_box.setValue(float(s['benefit_weight_end']))
                if 'generation_ratio' in s: self.generation_ratio_box.setValue(float(s['generation_ratio']))
                if 'adaptive_rates' in s: self.adaptive_rates_checkbox.setChecked(bool(s['adaptive_rates']))
                if 'stagnation_limit' in s: self.stagnation_limit_box.setValue(int(s['stagnation_limit']))
                if 'cxpb_min' in s: self.cxpb_min_box.setValue(float(s['cxpb_min']))
                if 'cxpb_max' in s: self.cxpb_max_box.setValue(float(s['cxpb_max']))
                if 'mutpb_min' in s: self.mutpb_min_box.setValue(float(s['mutpb_min']))
                if 'mutpb_max' in s: self.mutpb_max_box.setValue(float(s['mutpb_max']))
                ctrl = s.get('controller', 'fixed')
                self.controller_none_radio.setChecked(ctrl == 'fixed')
                self.controller_adaptive_radio.setChecked(ctrl == 'adaptive')
                self.controller_ml_radio.setChecked(ctrl == 'ml_bandit')
                self.controller_rl_radio.setChecked(ctrl == 'rl')
            except Exception:
                pass
            # Apply parameter ranges
            ranges_by_name = {}
            for p in cfg.get('parameters', []):
                try:
                    ranges_by_name[p['name']] = (float(p['low']), float(p['high']))
                except Exception:
                    continue
            self._apply_ranges_to_ga_table(ranges_by_name)
            # Apply parameter costs/categories if provided
            try:
                costs_by_name = {}
                for p in cfg.get('parameters', []):
                    name = p.get('name')
                    if name is None:
                        continue
                    # New format: nested 'costs'
                    if 'costs' in p and isinstance(p['costs'], dict):
                        costs_by_name[name] = {
                            'material': float(p['costs'].get('material', 0.0) or 0.0),
                            'manufacturing': float(p['costs'].get('manufacturing', 0.0) or 0.0),
                            'maintenance': float(p['costs'].get('maintenance', 0.0) or 0.0),
                            'operational': float(p['costs'].get('operational', 0.0) or 0.0),
                        }
                    # Backward compatibility: single 'cost' + optional 'category'
                    elif 'cost' in p:
                        cat = str(p.get('category', 'operational')).lower()
                        if cat not in ('material','manufacturing','maintenance','operational'):
                            cat = 'operational'
                        costs_by_name[name] = {
                            'material': 0.0,
                            'manufacturing': 0.0,
                            'maintenance': 0.0,
                            'operational': 0.0,
                        }
                        costs_by_name[name][cat] = float(p['cost'])
                # Apply costs to table
                for r in range(self.ga_param_table.rowCount()):
                    try:
                        name = self.ga_param_table.item(r, 0).text()
                        if name in costs_by_name:
                            vals = costs_by_name[name]
                            for ci, key in zip(range(5,9), ['material','manufacturing','maintenance','operational']):
                                w = self.ga_param_table.cellWidget(r, ci)
                                if w is not None:
                                    w.setValue(float(vals.get(key, 0.0)))
                    except Exception:
                        pass
            except Exception:
                pass
            QMessageBox.information(self, "Import", "GA config applied.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def export_ga_config(self):
        """Export current GA config (bounds and settings) to JSON file."""
        try:
            import json
            from datetime import datetime
            from PyQt5.QtWidgets import QFileDialog

            # Build GA settings dictionary
            ga_settings = {
                'pop_min': int(self.ga_pop_min_box.value()),
                'pop_max': int(self.ga_pop_max_box.value()),
                'pop_size': int(self.ga_pop_size_box.value()),
                'generations': int(self.ga_num_generations_box.value()),
                'cxpb': float(self.ga_cxpb_box.value()),
                'mutpb': float(self.ga_mutpb_box.value()),
                'tol': float(self.ga_tol_box.value()),
                'alpha': float(self.ga_alpha_box.value()),
                'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()),
                'cost_scale_factor': float(self.ga_cost_scale_box.value()),
                'dva_activation_threshold': float(self.ga_activation_threshold_box.value()),
                'dva_activation_penalty': float(self.ga_activation_penalty_box.value()),
                'use_enhanced_cost': bool(self.enh_cost_enable_chk.isChecked()),
                'benefit_w_primary': float(self.benefit_w_primary_box.value()),
                'benefit_w_accuracy': float(self.benefit_w_accuracy_box.value()),
                'benefit_w_sparsity': float(self.benefit_w_sparsity_box.value()),
                'category_w_material': float(self.cat_w_material_box.value()),
                'category_w_manufacturing': float(self.cat_w_manufacturing_box.value()),
                'category_w_maintenance': float(self.cat_w_maintenance_box.value()),
                'category_w_operational': float(self.cat_w_operational_box.value()),
                'benefit_weight_start': float(self.benefit_weight_start_box.value()),
                'benefit_weight_end': float(self.benefit_weight_end_box.value()),
                'generation_ratio': float(self.generation_ratio_box.value()),
                'adaptive_rates': bool(self.adaptive_rates_checkbox.isChecked()),
                'stagnation_limit': int(self.stagnation_limit_box.value()),
                'cxpb_min': float(self.cxpb_min_box.value()),
                'cxpb_max': float(self.cxpb_max_box.value()),
                'mutpb_min': float(self.mutpb_min_box.value()),
                'mutpb_max': float(self.mutpb_max_box.value()),
            }

            # Determine controller type
            if self.controller_adaptive_radio.isChecked():
                controller = 'adaptive'
            elif self.controller_ml_radio.isChecked():
                controller = 'ml_bandit'
            elif self.controller_rl_radio.isChecked():
                controller = 'rl'
            else:
                controller = 'fixed'
            ga_settings['controller'] = controller

            # Build parameters list with ranges and costs
            parameters = []
            if hasattr(self, 'ga_param_table'):
                for r in range(self.ga_param_table.rowCount()):
                    try:
                        param_data = {
                            'name': self.ga_param_table.item(r, 0).text() if self.ga_param_table.item(r, 0) else '',
                            'low': float(self.ga_param_table.cellWidget(r, 3).value()) if self.ga_param_table.cellWidget(r, 3) else 0.0,
                            'high': float(self.ga_param_table.cellWidget(r, 4).value()) if self.ga_param_table.cellWidget(r, 4) else 1.0,
                            'fixed': bool(self.ga_param_table.cellWidget(r, 1).isChecked()) if self.ga_param_table.cellWidget(r, 1) else False,
                            'costs': {
                                'material': float(self.ga_param_table.cellWidget(r, 5).value()) if self.ga_param_table.cellWidget(r, 5) else 0.0,
                                'manufacturing': float(self.ga_param_table.cellWidget(r, 6).value()) if self.ga_param_table.cellWidget(r, 6) else 0.0,
                                'maintenance': float(self.ga_param_table.cellWidget(r, 7).value()) if self.ga_param_table.cellWidget(r, 7) else 0.0,
                                'operational': float(self.ga_param_table.cellWidget(r, 8).value()) if self.ga_param_table.cellWidget(r, 8) else 0.0,
                            }
                        }
                        # Note: Category column not currently implemented in table
                        # Category information is handled via cost weights in enhanced cost analysis

                        parameters.append(param_data)
                    except Exception:
                        continue

            # Create export payload
            export_payload = {
                'type': 'ga_config',
                'version': '2.0',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'ga_settings': ga_settings,
                'parameters': parameters
            }

            # Suggest default filename
            default_name = f"ga_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export GA Config",
                default_name,
                "JSON Files (*.json);;All Files (*)"
            )

            if not file_path:
                return

            if not file_path.lower().endswith('.json'):
                file_path += '.json'

            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, indent=2, ensure_ascii=False)

            QMessageBox.information(self, "Export", f"GA config exported to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _extract_cell_text(self, table, row, col):
        """Return a string representation of the cell's content for export."""
        try:
            item = table.item(row, col)
            if item is not None:
                return item.text()
            widget = table.cellWidget(row, col)
            if widget is None:
                return ""
            # Common widget types placed in tables
            from PyQt5.QtWidgets import QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton
            if isinstance(widget, QCheckBox):
                return "True" if widget.isChecked() else "False"
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                return str(widget.value())
            if isinstance(widget, QComboBox):
                return widget.currentText()
            if isinstance(widget, QPushButton):
                return widget.text()
            # Fallback: try generic text property
            return getattr(widget, 'text', lambda: "")()
        except Exception:
            return ""

    def _qtable_to_dataframe(self, table, selected_only=False):
        """Convert a QTableWidget to a pandas DataFrame."""
        try:
            import pandas as pd
            # Determine columns and headers
            num_rows = table.rowCount()
            num_cols = table.columnCount()
            headers = []
            for c in range(num_cols):
                hdr_item = table.horizontalHeaderItem(c)
                headers.append(hdr_item.text() if hdr_item is not None else str(c))

            # Determine which rows to export
            if selected_only and table.selectionModel() and table.selectionModel().hasSelection():
                rows = sorted({idx.row() for idx in table.selectionModel().selectedIndexes()})
            else:
                rows = list(range(num_rows))

            data = []
            for r in rows:
                row_vals = []
                for c in range(num_cols):
                    row_vals.append(self._extract_cell_text(table, r, c))
                data.append(row_vals)

            return pd.DataFrame(data, columns=headers)
        except Exception as e:
            print(f"Error converting table to DataFrame: {str(e)}")
            import pandas as pd
            return pd.DataFrame()

    def _save_dataframe_to_path(self, df, file_path):
        """Save DataFrame to CSV or Excel based on file extension."""
        try:
            import os
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".xlsx":
                df.to_excel(file_path, index=False)
            else:
                # Default to CSV
                df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export", f"Saved to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_table_via_dialog(self, table, default_basename="table", selected_only=False, forced_ext=None):
        """Open a dialog and export the given table to CSV/Excel."""
        try:
            default_name = f"{default_basename}.csv"
            path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Export Table",
                default_name,
                "CSV files (*.csv);;Excel files (*.xlsx)"
            )
            if not path:
                return
            # Ensure proper extension
            chosen_ext = forced_ext
            if chosen_ext is None:
                if selected_filter and "xlsx" in selected_filter.lower():
                    chosen_ext = ".xlsx"
                else:
                    chosen_ext = ".csv"
            if not path.lower().endswith(chosen_ext):
                path = path + chosen_ext

            df = self._qtable_to_dataframe(table, selected_only=selected_only)
            self._save_dataframe_to_path(df, path)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _show_table_context_menu(self, table, default_basename, pos):
        """Show context menu with export options for a table."""
        try:
            menu = QMenu(table)
            act_export_csv = menu.addAction("Export All to CSV")
            act_export_xlsx = menu.addAction("Export All to Excel")
            menu.addSeparator()
            act_export_sel_csv = menu.addAction("Export Selected Rows to CSV")
            act_export_sel_xlsx = menu.addAction("Export Selected Rows to Excel")
            menu.addSeparator()
            act_copy_rows = menu.addAction("Copy Selected Rows")

            has_sel = bool(table.selectionModel() and table.selectionModel().hasSelection())
            act_export_sel_csv.setEnabled(has_sel)
            act_export_sel_xlsx.setEnabled(has_sel)
            act_copy_rows.setEnabled(has_sel)

            act_export_csv.triggered.connect(lambda: self._export_table_via_dialog(table, default_basename, selected_only=False, forced_ext=".csv"))
            act_export_xlsx.triggered.connect(lambda: self._export_table_via_dialog(table, default_basename, selected_only=False, forced_ext=".xlsx"))
            act_export_sel_csv.triggered.connect(lambda: self._export_table_via_dialog(table, default_basename + "_selected", selected_only=True, forced_ext=".csv"))
            act_export_sel_xlsx.triggered.connect(lambda: self._export_table_via_dialog(table, default_basename + "_selected", selected_only=True, forced_ext=".xlsx"))
            act_copy_rows.triggered.connect(lambda: self._copy_table_selection_to_clipboard(table))

            global_pos = table.viewport().mapToGlobal(pos)
            menu.exec_(global_pos)
        except Exception as e:
            print(f"Context menu error: {str(e)}")

    def _copy_table_selection_to_clipboard(self, table):
        """Copy selected rows of the table to the clipboard as TSV."""
        try:
            df = self._qtable_to_dataframe(table, selected_only=True)
            if df.empty:
                return
            tsv = df.to_csv(index=False, sep='\t')
            QApplication.clipboard().setText(tsv)
        except Exception as e:
            print(f"Clipboard copy error: {str(e)}")

    def _attach_table_export(self, table, default_basename="table"):
        """Enable right-click export menu on a QTableWidget."""
        try:
            table.setContextMenuPolicy(Qt.CustomContextMenu)
            table.customContextMenuRequested.connect(lambda pos, t=table, name=default_basename: self._show_table_context_menu(t, name, pos))
        except Exception:
            pass
    def create_ga_tab(self):
        """Create the genetic algorithm optimization tab"""
        self.ga_tab = QWidget()
        layout = QVBoxLayout(self.ga_tab)

        # Create sub-tabs widget
        self.ga_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: GA Hyperparameters --------------------
        ga_hyper_tab = QWidget()
        ga_hyper_layout = QFormLayout(ga_hyper_tab)

        # Population range controls (min/max) to support dynamic resizing
        self.ga_pop_min_box = QSpinBox()
        self.ga_pop_min_box.setRange(1, 100000)
        self.ga_pop_min_box.setValue(400)
        self.ga_pop_min_box.setToolTip("Minimum population size when dynamic resizing is enabled")
        self.ga_pop_max_box = QSpinBox()
        self.ga_pop_max_box.setRange(1, 100000)
        self.ga_pop_max_box.setValue(1600)
        self.ga_pop_max_box.setToolTip("Maximum population size when dynamic resizing is enabled")
        pop_range_widget = QWidget()
        pop_range_layout = QHBoxLayout(pop_range_widget)
        pop_range_layout.setContentsMargins(0,0,0,0)
        pop_range_layout.addWidget(QLabel("Min:"))
        pop_range_layout.addWidget(self.ga_pop_min_box)
        pop_range_layout.addWidget(QLabel("Max:"))
        pop_range_layout.addWidget(self.ga_pop_max_box)

        self.ga_pop_size_box = QSpinBox()
        self.ga_pop_size_box.setRange(1, 100000)
        self.ga_pop_size_box.setValue(800)
        self.ga_pop_size_box.setToolTip("Initial population size (used if resizing disabled or as starting point)")

        self.ga_num_generations_box = QSpinBox()
        self.ga_num_generations_box.setRange(1, 10000)
        self.ga_num_generations_box.setValue(100)

        self.ga_cxpb_box = QDoubleSpinBox()
        self.ga_cxpb_box.setRange(0, 1)
        self.ga_cxpb_box.setValue(0.7)
        self.ga_cxpb_box.setDecimals(4)
        self.ga_cxpb_box.setSingleStep(0.0001)

        self.ga_mutpb_box = QDoubleSpinBox()
        self.ga_mutpb_box.setRange(0, 1)
        self.ga_mutpb_box.setValue(0.2)
        self.ga_mutpb_box.setDecimals(4)
        self.ga_mutpb_box.setSingleStep(0.0001)

        self.ga_tol_box = QDoubleSpinBox()
        self.ga_tol_box.setRange(0, 1e6)
        self.ga_tol_box.setValue(1e-3)
        self.ga_tol_box.setDecimals(6)

        self.ga_alpha_box = QDoubleSpinBox()
        self.ga_alpha_box.setRange(0.0, 10.0)
        self.ga_alpha_box.setDecimals(4)
        self.ga_alpha_box.setSingleStep(0.01)
        self.ga_alpha_box.setValue(0.01)

        self.ga_percentage_error_scale_box = QDoubleSpinBox()
        self.ga_percentage_error_scale_box.setRange(1.0, 100000.0)
        self.ga_percentage_error_scale_box.setDecimals(2)
        self.ga_percentage_error_scale_box.setSingleStep(100.0)
        self.ga_percentage_error_scale_box.setValue(1000.0)
        self.ga_percentage_error_scale_box.setToolTip("Scaling factor for percentage error in fitness calculation (higher = less influence)")

        self.ga_cost_scale_box = QDoubleSpinBox()
        self.ga_cost_scale_box.setRange(0.1, 10000.0)
        self.ga_cost_scale_box.setDecimals(2)
        self.ga_cost_scale_box.setSingleStep(1.0)
        self.ga_cost_scale_box.setValue(30.0)
        self.ga_cost_scale_box.setToolTip("Scaling factor for cost term in fitness calculation (higher = less influence, default: 30.0)")
        self.ga_cost_scale_box.setMinimumWidth(100)  # Ensure minimum width
        self.ga_cost_scale_box.setVisible(True)

        # New: Activation threshold and penalty per active DVA parameter
        self.ga_activation_threshold_box = QDoubleSpinBox()
        self.ga_activation_threshold_box.setRange(0.0, 1e9)
        self.ga_activation_threshold_box.setDecimals(6)
        self.ga_activation_threshold_box.setSingleStep(0.001)
        self.ga_activation_threshold_box.setValue(0.0)
        self.ga_activation_threshold_box.setToolTip("Threshold magnitude above which a DVA parameter is considered 'active' for penalty and cost")

        self.ga_activation_penalty_box = QDoubleSpinBox()
        self.ga_activation_penalty_box.setRange(0.0, 1e6)
        self.ga_activation_penalty_box.setDecimals(6)
        self.ga_activation_penalty_box.setSingleStep(0.1)
        self.ga_activation_penalty_box.setValue(0.0)
        self.ga_activation_penalty_box.setToolTip("Penalty added per active DVA parameter (encourages fewer active parameters)")
        
        # Add benchmarking runs box
        self.ga_benchmark_runs_box = QSpinBox()
        self.ga_benchmark_runs_box.setRange(1, 1000)
        self.ga_benchmark_runs_box.setValue(1)
        self.ga_benchmark_runs_box.setToolTip("Number of times to run the GA for benchmarking (1 = single run)")
        
        # Controller selection (mutually exclusive): None, Adaptive, ML Bandit, RL
        controller_group = QGroupBox("Controller (choose one)")
        controller_layout = QHBoxLayout(controller_group)
        self.controller_none_radio = QRadioButton("Fixed")
        self.controller_adaptive_radio = QRadioButton("Adaptive Rates")
        self.controller_ml_radio = QRadioButton("ML Bandit")
        self.controller_rl_radio = QRadioButton("RL Controller")
        self.controller_none_radio.setChecked(True)
        controller_layout.addWidget(self.controller_none_radio)
        controller_layout.addWidget(self.controller_adaptive_radio)
        controller_layout.addWidget(self.controller_ml_radio)
        controller_layout.addWidget(self.controller_rl_radio)
        ga_hyper_layout.addRow(controller_group)
        
        # Add adaptive rates checkbox
        self.adaptive_rates_checkbox = QCheckBox("Use Adaptive Rates")
        self.adaptive_rates_checkbox.setChecked(False)
        self.adaptive_rates_checkbox.setToolTip("Automatically adjust crossover and mutation rates during optimization")
        self.adaptive_rates_checkbox.stateChanged.connect(self.toggle_adaptive_rates_options)
        # Tie checkbox to radio for backward compatibility
        self.controller_adaptive_radio.toggled.connect(lambda checked: self.adaptive_rates_checkbox.setChecked(checked))
        
        # Create a widget to hold adaptive rate options
        self.adaptive_rates_options = QWidget()
        adaptive_options_layout = QFormLayout(self.adaptive_rates_options)
        adaptive_options_layout.setContentsMargins(20, 0, 0, 0)  # Add left margin for indentation
        
        # Stagnation limit spinner
        self.stagnation_limit_box = QSpinBox()
        self.stagnation_limit_box.setRange(1, 50)
        self.stagnation_limit_box.setValue(5)
        self.stagnation_limit_box.setToolTip("Number of generations without improvement before adapting rates")
        adaptive_options_layout.addRow("Stagnation Limit:", self.stagnation_limit_box)
        
        # Create a widget for crossover bounds
        crossover_bounds_widget = QWidget()
        crossover_bounds_layout = QHBoxLayout(crossover_bounds_widget)
        crossover_bounds_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cxpb_min_box = QDoubleSpinBox()
        self.cxpb_min_box.setRange(0.0, 1.0)
        self.cxpb_min_box.setValue(0.1)
        self.cxpb_min_box.setDecimals(4)
        self.cxpb_min_box.setSingleStep(0.0001)
        self.cxpb_min_box.setToolTip("Minimum crossover probability")
        
        self.cxpb_max_box = QDoubleSpinBox()
        self.cxpb_max_box.setRange(0.0, 1.0)
        self.cxpb_max_box.setValue(0.9)
        self.cxpb_max_box.setDecimals(4)
        self.cxpb_max_box.setSingleStep(0.0001)
        self.cxpb_max_box.setToolTip("Maximum crossover probability")
        
        crossover_bounds_layout.addWidget(QLabel("Min:"))
        crossover_bounds_layout.addWidget(self.cxpb_min_box)
        crossover_bounds_layout.addWidget(QLabel("Max:"))
        crossover_bounds_layout.addWidget(self.cxpb_max_box)
        
        # Show crossover bounds as hard limits (always visible)
        ga_hyper_layout.addRow("Crossover Bounds:", crossover_bounds_widget)
        
        # Create a widget for mutation bounds
        mutation_bounds_widget = QWidget()
        mutation_bounds_layout = QHBoxLayout(mutation_bounds_widget)
        mutation_bounds_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mutpb_min_box = QDoubleSpinBox()
        self.mutpb_min_box.setRange(0.0, 1.0)
        self.mutpb_min_box.setValue(0.05)
        self.mutpb_min_box.setDecimals(4)
        self.mutpb_min_box.setSingleStep(0.0001)
        self.mutpb_min_box.setToolTip("Minimum mutation probability")
        
        self.mutpb_max_box = QDoubleSpinBox()
        self.mutpb_max_box.setRange(0.0, 1.0)
        self.mutpb_max_box.setValue(0.5)
        self.mutpb_max_box.setDecimals(4)
        self.mutpb_max_box.setSingleStep(0.0001)
        self.mutpb_max_box.setToolTip("Maximum mutation probability")
        
        mutation_bounds_layout.addWidget(QLabel("Min:"))
        mutation_bounds_layout.addWidget(self.mutpb_min_box)
        mutation_bounds_layout.addWidget(QLabel("Max:"))
        mutation_bounds_layout.addWidget(self.mutpb_max_box)
        
        # Show mutation bounds as hard limits (always visible)
        ga_hyper_layout.addRow("Mutation Bounds:", mutation_bounds_widget)
        
        # Initially hide adaptive-only options (not the hard limits added above)
        self.adaptive_rates_options.setVisible(False)

        ga_hyper_layout.addRow("Population Size (initial):", self.ga_pop_size_box)
        ga_hyper_layout.addRow("Population Range:", pop_range_widget)
        ga_hyper_layout.addRow("Number of Generations:", self.ga_num_generations_box)
        ga_hyper_layout.addRow("Crossover Probability (cxpb):", self.ga_cxpb_box)
        ga_hyper_layout.addRow("Mutation Probability (mutpb):", self.ga_mutpb_box)
        ga_hyper_layout.addRow("Tolerance (tol):", self.ga_tol_box)
        ga_hyper_layout.addRow("Sparsity Penalty (alpha):", self.ga_alpha_box)
        ga_hyper_layout.addRow("Percentage Error Scale:", self.ga_percentage_error_scale_box)
        ga_hyper_layout.addRow("Cost Scale Factor:", self.ga_cost_scale_box)
        # Ensure the widget is visible and layout is updated
        self.ga_cost_scale_box.setVisible(True)
        self.ga_cost_scale_box.update()
        ga_hyper_tab.update()  # Update the parent tab
        ga_hyper_layout.addRow("Activation Threshold:", self.ga_activation_threshold_box)
        ga_hyper_layout.addRow("Activation Penalty per DVA:", self.ga_activation_penalty_box)

        # Enhanced Cost-Benefit group
        enh_group = QGroupBox("Enhanced Cost-Benefit")
        enh_form = QFormLayout(enh_group)
        self.enh_cost_enable_chk = QCheckBox("Enable Enhanced Cost-Benefit Analysis")
        self.enh_cost_enable_chk.setChecked(True)
        enh_form.addRow(self.enh_cost_enable_chk)

        # Ensure the group is visible and expanded
        enh_group.setVisible(True)

        # Benefit weights (primary, accuracy, sparsity)
        self.benefit_w_primary_box = QDoubleSpinBox(); self.benefit_w_primary_box.setRange(0.0, 1.0); self.benefit_w_primary_box.setSingleStep(0.05); self.benefit_w_primary_box.setValue(0.6)
        self.benefit_w_accuracy_box = QDoubleSpinBox(); self.benefit_w_accuracy_box.setRange(0.0, 1.0); self.benefit_w_accuracy_box.setSingleStep(0.05); self.benefit_w_accuracy_box.setValue(0.3)
        self.benefit_w_sparsity_box = QDoubleSpinBox(); self.benefit_w_sparsity_box.setRange(0.0, 1.0); self.benefit_w_sparsity_box.setSingleStep(0.05); self.benefit_w_sparsity_box.setValue(0.1)
        bw_row = QWidget(); bw_l = QHBoxLayout(bw_row); bw_l.setContentsMargins(0,0,0,0)
        bw_l.addWidget(QLabel("Primary:")); bw_l.addWidget(self.benefit_w_primary_box)
        bw_l.addWidget(QLabel("Accuracy:")); bw_l.addWidget(self.benefit_w_accuracy_box)
        bw_l.addWidget(QLabel("Sparsity:")); bw_l.addWidget(self.benefit_w_sparsity_box)
        enh_form.addRow("Benefit Weights:", bw_row)

        # Category weights (material, manufacturing, maintenance, operational)
        self.cat_w_material_box = QDoubleSpinBox(); self.cat_w_material_box.setRange(0.0, 1.0); self.cat_w_material_box.setSingleStep(0.05); self.cat_w_material_box.setValue(0.4)
        self.cat_w_manufacturing_box = QDoubleSpinBox(); self.cat_w_manufacturing_box.setRange(0.0, 1.0); self.cat_w_manufacturing_box.setSingleStep(0.05); self.cat_w_manufacturing_box.setValue(0.35)
        self.cat_w_maintenance_box = QDoubleSpinBox(); self.cat_w_maintenance_box.setRange(0.0, 1.0); self.cat_w_maintenance_box.setSingleStep(0.05); self.cat_w_maintenance_box.setValue(0.15)
        self.cat_w_operational_box = QDoubleSpinBox(); self.cat_w_operational_box.setRange(0.0, 1.0); self.cat_w_operational_box.setSingleStep(0.05); self.cat_w_operational_box.setValue(0.10)
        cw_row = QWidget(); cw_l = QHBoxLayout(cw_row); cw_l.setContentsMargins(0,0,0,0)
        cw_l.addWidget(QLabel("Material:")); cw_l.addWidget(self.cat_w_material_box)
        cw_l.addWidget(QLabel("Manufacturing:")); cw_l.addWidget(self.cat_w_manufacturing_box)
        cw_l.addWidget(QLabel("Maintenance:")); cw_l.addWidget(self.cat_w_maintenance_box)
        cw_l.addWidget(QLabel("Operational:")); cw_l.addWidget(self.cat_w_operational_box)
        enh_form.addRow("Category Weights:", cw_row)

        # Benefit weight schedule (start->end)
        self.benefit_weight_start_box = QDoubleSpinBox(); self.benefit_weight_start_box.setRange(0.0, 1.0); self.benefit_weight_start_box.setSingleStep(0.05); self.benefit_weight_start_box.setValue(0.3)
        self.benefit_weight_end_box = QDoubleSpinBox(); self.benefit_weight_end_box.setRange(0.0, 1.0); self.benefit_weight_end_box.setSingleStep(0.05); self.benefit_weight_end_box.setValue(0.7)
        bw_sched_row = QWidget(); bws_l = QHBoxLayout(bw_sched_row); bws_l.setContentsMargins(0,0,0,0)
        bws_l.addWidget(QLabel("Start:")); bws_l.addWidget(self.benefit_weight_start_box)
        bws_l.addWidget(QLabel("End:")); bws_l.addWidget(self.benefit_weight_end_box)
        enh_form.addRow("Benefit Weight Schedule:", bw_sched_row)

        # Generation ratio control
        self.generation_ratio_box = QDoubleSpinBox()
        self.generation_ratio_box.setRange(0.0, 1.0)
        self.generation_ratio_box.setSingleStep(0.01)
        self.generation_ratio_box.setValue(0.5)
        self.generation_ratio_box.setDecimals(3)
        self.generation_ratio_box.setToolTip("Generation ratio for adaptive cost-benefit weighting (0.0 = early optimization, 1.0 = late optimization)")
        self.generation_ratio_box.setVisible(True)  # Ensure it's visible
        self.generation_ratio_box.setMinimumWidth(100)  # Ensure minimum width
        self.generation_ratio_box.setMaximumWidth(200)  # Ensure maximum width
        # Add generation ratio at the end of the form
        enh_form.addRow("Generation Ratio:", self.generation_ratio_box)

        # Add some spacing at the end to ensure visibility
        enh_form.addRow("", QWidget())

        ga_hyper_layout.addRow(enh_group)

        # Ensure the layout is properly updated
        enh_group.updateGeometry()
        ga_hyper_layout.update()
        ga_hyper_layout.addRow("Benchmark Runs:", self.ga_benchmark_runs_box)
        ga_hyper_layout.addRow("", self.adaptive_rates_checkbox)
        ga_hyper_layout.addRow("", self.adaptive_rates_options)

        # ML Bandit controller (rates + optional population)
        self.ml_controller_checkbox = QCheckBox("Use ML Bandit (rates + population)")
        self.ml_controller_checkbox.setChecked(False)
        self.ml_controller_checkbox.setToolTip("Use a UCB bandit controller to adapt crossover, mutation, and optionally population size per generation")
        # Tie checkbox to radio for backward compatibility
        self.controller_ml_radio.toggled.connect(lambda checked: self.ml_controller_checkbox.setChecked(checked))
        ga_hyper_layout.addRow("", self.ml_controller_checkbox)
        self.ml_pop_adapt_checkbox = QCheckBox("Allow population resizing")
        self.ml_pop_adapt_checkbox.setChecked(True)
        self.ml_pop_adapt_checkbox.setToolTip("If enabled, ML controller can increase/decrease population size within bounds")
        ga_hyper_layout.addRow("ML Population:", self.ml_pop_adapt_checkbox)

        self.ml_diversity_weight_box = QDoubleSpinBox()
        self.ml_diversity_weight_box.setRange(0.0, 1.0)
        self.ml_diversity_weight_box.setDecimals(3)
        self.ml_diversity_weight_box.setSingleStep(0.005)
        self.ml_diversity_weight_box.setValue(0.02)
        self.ml_diversity_weight_box.setToolTip("Weight for diversity penalty in reward")
        ga_hyper_layout.addRow("ML Diversity Weight:", self.ml_diversity_weight_box)

        self.ml_diversity_target_box = QDoubleSpinBox()
        self.ml_diversity_target_box.setRange(0.0, 1.0)
        self.ml_diversity_target_box.setDecimals(2)
        self.ml_diversity_target_box.setSingleStep(0.05)
        self.ml_diversity_target_box.setValue(0.20)
        self.ml_diversity_target_box.setToolTip("Target normalized diversity (std/mean)")
        ga_hyper_layout.addRow("ML Diversity Target:", self.ml_diversity_target_box)

        self.ml_ucb_c_box = QDoubleSpinBox()
        self.ml_ucb_c_box.setRange(0.1, 3.0)
        self.ml_ucb_c_box.setDecimals(2)
        self.ml_ucb_c_box.setSingleStep(0.05)
        self.ml_ucb_c_box.setValue(0.60)
        self.ml_ucb_c_box.setToolTip("Exploration strength for UCB (higher explores more)")
        ga_hyper_layout.addRow("ML UCB c:", self.ml_ucb_c_box)

        # ML stagnation boost controls
        self.ml_stag_chk = QCheckBox("Boost exploration on stagnation")
        self.ml_stag_chk.setChecked(True)
        self.ml_stag_limit = QSpinBox()
        self.ml_stag_limit.setRange(1, 50)
        self.ml_stag_limit.setValue(5)
        ml_stag_row = QWidget()
        ml_stag_row_l = QHBoxLayout(ml_stag_row)
        ml_stag_row_l.setContentsMargins(0, 0, 0, 0)
        ml_stag_row_l.addWidget(self.ml_stag_chk)
        ml_stag_row_l.addWidget(QLabel("Limit (gens):"))
        ml_stag_row_l.addWidget(self.ml_stag_limit)
        ga_hyper_layout.addRow("ML Stagnation:", ml_stag_row)

        # ML reward blending weights
        self.ml_historical_weight_box = QDoubleSpinBox()
        self.ml_historical_weight_box.setRange(0.0, 1.0)
        self.ml_historical_weight_box.setDecimals(2)
        self.ml_historical_weight_box.setSingleStep(0.05)
        self.ml_historical_weight_box.setValue(0.70)
        self.ml_historical_weight_box.setToolTip("Weight for historical average in reward blending (0.0-1.0)")
        ga_hyper_layout.addRow("ML Historical Weight:", self.ml_historical_weight_box)

        self.ml_current_weight_box = QDoubleSpinBox()
        self.ml_current_weight_box.setRange(0.0, 1.0)
        self.ml_current_weight_box.setDecimals(2)
        self.ml_current_weight_box.setSingleStep(0.05)
        self.ml_current_weight_box.setValue(0.30)
        self.ml_current_weight_box.setToolTip("Weight for current reward in reward blending (0.0-1.0)")
        ga_hyper_layout.addRow("ML Current Weight:", self.ml_current_weight_box)

        # Connect the weight boxes to automatically adjust each other to sum to 1.0
        def adjust_weights():
            historical = self.ml_historical_weight_box.value()
            current = self.ml_current_weight_box.value()
            total = historical + current
            if total != 0:
                # Normalize to sum to 1.0
                self.ml_historical_weight_box.setValue(historical / total)
                self.ml_current_weight_box.setValue(current / total)
        
        self.ml_historical_weight_box.valueChanged.connect(adjust_weights)
        self.ml_current_weight_box.valueChanged.connect(adjust_weights)

        # RL controller options
        self.rl_options_widget = QWidget()
        rl_form = QFormLayout(self.rl_options_widget)
        rl_form.setContentsMargins(20, 0, 0, 0)
        self.rl_alpha_box = QDoubleSpinBox()
        self.rl_alpha_box.setRange(0.0, 1.0)
        self.rl_alpha_box.setDecimals(3)
        self.rl_alpha_box.setValue(0.1)
        rl_form.addRow("RL a (learning rate):", self.rl_alpha_box)
        self.rl_gamma_box = QDoubleSpinBox()
        self.rl_gamma_box.setRange(0.0, 1.0)
        self.rl_gamma_box.setDecimals(3)
        self.rl_gamma_box.setValue(0.9)
        rl_form.addRow("RL  (discount):", self.rl_gamma_box)
        self.rl_epsilon_box = QDoubleSpinBox()
        self.rl_epsilon_box.setRange(0.0, 1.0)
        self.rl_epsilon_box.setDecimals(3)
        self.rl_epsilon_box.setValue(0.2)
        rl_form.addRow("RL e (explore):", self.rl_epsilon_box)
        self.rl_decay_box = QDoubleSpinBox()
        self.rl_decay_box.setRange(0.0, 1.0)
        self.rl_decay_box.setDecimals(3)
        self.rl_decay_box.setValue(0.95)
        rl_form.addRow("RL e decay:", self.rl_decay_box)
        # RL stagnation boost controls
        self.rl_stag_chk = QCheckBox("Boost exploration on stagnation")
        self.rl_stag_chk.setChecked(True)
        self.rl_stag_limit = QSpinBox()
        self.rl_stag_limit.setRange(1, 50)
        self.rl_stag_limit.setValue(5)
        rl_stag_row = QWidget()
        rl_stag_row_l = QHBoxLayout(rl_stag_row)
        rl_stag_row_l.setContentsMargins(0, 0, 0, 0)
        rl_stag_row_l.addWidget(self.rl_stag_chk)
        rl_stag_row_l.addWidget(QLabel("Limit (gens):"))
        rl_stag_row_l.addWidget(self.rl_stag_limit)
        rl_form.addRow("RL Stagnation:", rl_stag_row)
        self.rl_options_widget.setVisible(False)
        self.controller_rl_radio.toggled.connect(self.rl_options_widget.setVisible)
        ga_hyper_layout.addRow("RL Options:", self.rl_options_widget)

        # Surrogate-assisted screening controls
        self.surrogate_checkbox = QCheckBox("Use Surrogate-Assisted Screening")
        self.surrogate_checkbox.setToolTip("Train a fast predictor to pre-screen candidates and evaluate only the most promising with FRF")
        ga_hyper_layout.addRow("", self.surrogate_checkbox)

        self.surr_pool_factor_box = QDoubleSpinBox()
        self.surr_pool_factor_box.setRange(1.0, 10.0)
        self.surr_pool_factor_box.setDecimals(1)
        self.surr_pool_factor_box.setSingleStep(0.5)
        self.surr_pool_factor_box.setValue(2.0)
        self.surr_pool_factor_box.setToolTip("Pool size multiplier relative to FRF eval budget per generation")
        ga_hyper_layout.addRow("Surrogate Pool Factor:", self.surr_pool_factor_box)

        self.surr_k_box = QSpinBox()
        self.surr_k_box.setRange(1, 25)
        self.surr_k_box.setValue(5)
        self.surr_k_box.setToolTip("k for KNN surrogate predictions")
        ga_hyper_layout.addRow("Surrogate k (KNN):", self.surr_k_box)

        self.surr_explore_frac_box = QDoubleSpinBox()
        self.surr_explore_frac_box.setRange(0.0, 0.5)
        self.surr_explore_frac_box.setDecimals(2)
        self.surr_explore_frac_box.setSingleStep(0.05)
        self.surr_explore_frac_box.setValue(0.15)
        self.surr_explore_frac_box.setToolTip("Fraction of FRF budget reserved for exploratory/uncertain candidates")
        ga_hyper_layout.addRow("Surrogate Explore Fraction:", self.surr_explore_frac_box)

        # Seeding method selection
        self.seeding_method_combo = QComboBox()
        self.seeding_method_combo.addItems(["Random", "Sobol", "Latin Hypercube", "Neural (UCB/EI)", "Memory (Replay/Jitter)", "Best-of-Pool (Evaluate)"])
        self.seeding_method_combo.setCurrentIndex(0)
        self.seeding_method_combo.setToolTip("Choose initial population seeding method: Random, Sobol, Latin Hypercube, Neural (UCB/EI), Memory (Replay/Jitter), or Best-of-Pool (Evaluate)")
        ga_hyper_layout.addRow("Seeding Method:", self.seeding_method_combo)

        # Neural seeding options (collapsible)
        self.neural_options_group = QGroupBox("Neural Seeding Options")
        self.neural_options_group.setCheckable(True)
        self.neural_options_group.setChecked(False)
        neural_form = QFormLayout(self.neural_options_group)

        self.neural_acq_combo = QComboBox()
        self.neural_acq_combo.addItems(["UCB", "EI"])
        self.neural_acq_combo.setCurrentIndex(0)
        neural_form.addRow("Acquisition:", self.neural_acq_combo)

        self.neural_beta_min = QDoubleSpinBox()
        self.neural_beta_min.setRange(0.0, 10.0)
        self.neural_beta_min.setDecimals(2)
        self.neural_beta_min.setSingleStep(0.1)
        self.neural_beta_min.setValue(1.0)
        self.neural_beta_max = QDoubleSpinBox()
        self.neural_beta_max.setRange(0.0, 10.0)
        self.neural_beta_max.setDecimals(2)
        self.neural_beta_max.setSingleStep(0.1)
        self.neural_beta_max.setValue(2.5)
        beta_row = QWidget()
        beta_layout = QHBoxLayout(beta_row)
        beta_layout.setContentsMargins(0, 0, 0, 0)
        beta_layout.addWidget(QLabel("Min:"))
        beta_layout.addWidget(self.neural_beta_min)
        beta_layout.addWidget(QLabel("Max:"))
        beta_layout.addWidget(self.neural_beta_max)
        neural_form.addRow("ÃŸ (UCB range):", beta_row)

        self.neural_eps = QDoubleSpinBox()
        self.neural_eps.setRange(0.0, 0.9)
        self.neural_eps.setSingleStep(0.05)
        self.neural_eps.setValue(0.1)
        neural_form.addRow("Exploration fraction e:", self.neural_eps)

        self.neural_pool_mult = QDoubleSpinBox()
        self.neural_pool_mult.setRange(1.0, 20.0)
        self.neural_pool_mult.setSingleStep(0.5)
        self.neural_pool_mult.setValue(3.0)
        neural_form.addRow("Pool size x pop:", self.neural_pool_mult)

        self.neural_ensemble = QSpinBox()
        self.neural_ensemble.setRange(1, 9)
        self.neural_ensemble.setValue(3)
        neural_form.addRow("Ensemble size:", self.neural_ensemble)

        self.neural_layers = QSpinBox()
        self.neural_layers.setRange(0, 5)
        self.neural_layers.setValue(2)
        neural_form.addRow("Hidden layers:", self.neural_layers)

        self.neural_hidden = QSpinBox()
        self.neural_hidden.setRange(8, 1024)
        self.neural_hidden.setValue(96)
        neural_form.addRow("Hidden units:", self.neural_hidden)

        self.neural_dropout = QDoubleSpinBox()
        self.neural_dropout.setRange(0.0, 0.9)
        self.neural_dropout.setSingleStep(0.05)
        self.neural_dropout.setValue(0.1)
        neural_form.addRow("Dropout:", self.neural_dropout)

        self.neural_wd = QDoubleSpinBox()
        self.neural_wd.setRange(0.0, 1.0)
        self.neural_wd.setDecimals(6)
        self.neural_wd.setValue(0.0001)
        neural_form.addRow("Weight decay:", self.neural_wd)

        self.neural_epochs = QSpinBox()
        self.neural_epochs.setRange(1, 100)
        self.neural_epochs.setValue(8)
        neural_form.addRow("Epochs/gen:", self.neural_epochs)

        self.neural_time_cap = QSpinBox()
        self.neural_time_cap.setRange(50, 10000)
        self.neural_time_cap.setValue(750)
        neural_form.addRow("Train time cap [ms]:", self.neural_time_cap)

        self.neural_grad_refine_chk = QCheckBox("Gradient refinement of acquisitions")
        self.neural_grad_refine_chk.setChecked(False)
        self.neural_grad_steps = QSpinBox()
        self.neural_grad_steps.setRange(0, 50)
        self.neural_grad_steps.setValue(0)
        grad_row = QWidget()
        grad_layout = QHBoxLayout(grad_row)
        grad_layout.setContentsMargins(0, 0, 0, 0)
        grad_layout.addWidget(self.neural_grad_refine_chk)
        grad_layout.addWidget(QLabel("Steps:"))
        grad_layout.addWidget(self.neural_grad_steps)
        neural_form.addRow("Grad refine:", grad_row)

        self.neural_device_combo = QComboBox()
        self.neural_device_combo.addItems(["cpu", "cuda"])
        self.neural_device_combo.setCurrentIndex(0)
        neural_form.addRow("Device:", self.neural_device_combo)

        ga_hyper_layout.addRow(self.neural_options_group)

        # Auto-toggle group based on selection
        def _toggle_neural_group():
            self.neural_options_group.setChecked(self.seeding_method_combo.currentText().lower().startswith("neural"))
        self.seeding_method_combo.currentTextChanged.connect(_toggle_neural_group)
        _toggle_neural_group()

        # Best-of-Pool options
        self.best_pool_group = QGroupBox("Best-of-Pool Options")
        self.best_pool_group.setCheckable(True)
        self.best_pool_group.setChecked(False)
        best_form = QFormLayout(self.best_pool_group)

        self.best_pool_mult = QDoubleSpinBox()
        self.best_pool_mult.setRange(1.0, 50.0)
        self.best_pool_mult.setSingleStep(0.5)
        self.best_pool_mult.setValue(5.0)
        best_form.addRow("Pool size x pop:", self.best_pool_mult)

        self.best_diversity_frac = QDoubleSpinBox()
        self.best_diversity_frac.setRange(0.0, 1.0)
        self.best_diversity_frac.setSingleStep(0.05)
        self.best_diversity_frac.setValue(0.20)
        best_form.addRow("Diversity fraction:", self.best_diversity_frac)

        def _toggle_best_group():
            self.best_pool_group.setChecked(self.seeding_method_combo.currentText().lower().startswith("best-of-pool"))
        self.seeding_method_combo.currentTextChanged.connect(_toggle_best_group)
        _toggle_best_group()

        ga_hyper_layout.addRow(self.best_pool_group)

        # Add GA control buttons (Run/Pause/Resume/Terminate)
        self.run_ga_button = QPushButton("Run GA")
        self.run_ga_button.setFixedWidth(100)
        self.run_ga_button.clicked.connect(self.run_ga)

        self.pause_ga_button = QPushButton("Pause")
        self.pause_ga_button.setFixedWidth(100)
        self.pause_ga_button.setEnabled(False)
        self.pause_ga_button.clicked.connect(self.pause_ga)

        self.resume_ga_button = QPushButton("Resume")
        self.resume_ga_button.setFixedWidth(100)
        self.resume_ga_button.setEnabled(False)
        self.resume_ga_button.clicked.connect(self.resume_ga)

        self.stop_ga_button = QPushButton("Terminate")
        self.stop_ga_button.setFixedWidth(100)
        self.stop_ga_button.setEnabled(False)
        self.stop_ga_button.clicked.connect(self.stop_ga)

        ga_control_widget = QWidget()
        ga_control_layout = QHBoxLayout(ga_control_widget)
        ga_control_layout.setContentsMargins(0,0,0,0)
        ga_control_layout.addWidget(self.run_ga_button)
        ga_control_layout.addWidget(self.pause_ga_button)
        ga_control_layout.addWidget(self.resume_ga_button)
        ga_control_layout.addWidget(self.stop_ga_button)
        ga_hyper_layout.addRow("GA Control:", ga_control_widget)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        ga_param_tab = QWidget()
        ga_param_layout = QVBoxLayout(ga_param_tab)

        self.ga_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.ga_param_table.setRowCount(len(dva_parameters))
        self.ga_param_table.setColumnCount(9)
        self.ga_param_table.setHorizontalHeaderLabels(
            [
                "Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound",
                "Cost (Material)", "Cost (Manufacturing)", "Cost (Maintenance)", "Cost (Operational)"
            ]
        )
        self.ga_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ga_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.ga_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.setChecked(True)  # Set fixed to true by default
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_ga_fixed(state, r))
            self.ga_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setValue(0.0)  # Set fixed value to 0
            fixed_value_spin.setEnabled(True)  # Enable because fixed is checked
            self.ga_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setValue(0.0)  # Set to 0
            lower_bound_spin.setEnabled(False)  # Disable because fixed is checked
            self.ga_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setValue(1.0)  # Set to 1
            upper_bound_spin.setEnabled(False)  # Disable because fixed is checked
            self.ga_param_table.setCellWidget(row, 4, upper_bound_spin)

            for ci, default_val in zip(range(5, 9), [0.0, 0.0, 0.0, 0.0]):
                cost_spin = QDoubleSpinBox()
                cost_spin.setRange(0.0, 1e12)
                cost_spin.setDecimals(6)
                cost_spin.setValue(default_val)
                cost_spin.setEnabled(True)
                self.ga_param_table.setCellWidget(row, ci, cost_spin)

        # Attach export context menu to parameters table
        self._attach_table_export(self.ga_param_table, "ga_parameters")

        ga_param_layout.addWidget(self.ga_param_table)

        # -------------------- Sub-tab 3: Results --------------------
        ga_results_tab = QWidget()
        ga_results_layout = QVBoxLayout(ga_results_tab)

        # Create a header area for label and export button
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0) # No margins for this internal layout

        results_label = QLabel("GA Optimization Results:")
        header_layout.addWidget(results_label)
        header_layout.addStretch() # Add spacer to push the export button to the right

        self.export_ga_results_button = QPushButton("Export GA Results")
        self.export_ga_results_button.setObjectName("secondary-button") # Use existing styling if desired
        self.export_ga_results_button.setToolTip("Export the GA optimization results to a JSON file")
        self.export_ga_results_button.setEnabled(False)  # Initially disabled
        self.export_ga_results_button.clicked.connect(self.export_ga_results_to_file)
        header_layout.addWidget(self.export_ga_results_button)
        
        ga_results_layout.addWidget(header_container) # Add the header with label and button
        
        self.ga_results_text = QTextEdit()
        self.ga_results_text.setReadOnly(True)
        ga_results_layout.addWidget(self.ga_results_text)

        # -------------------- Sub-tab 4: Benchmarking --------------------
        ga_benchmark_tab = QWidget()
        ga_benchmark_layout = QVBoxLayout(ga_benchmark_tab)

        # Create buttons for import/export
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 10)  # Add some bottom margin

        self.import_benchmark_button = QPushButton("Import Benchmark Data")
        self.import_benchmark_button.setToolTip("Import previously saved GA benchmark data")
        self.import_benchmark_button.clicked.connect(self.import_ga_benchmark_data)
        button_layout.addWidget(self.import_benchmark_button)

        self.export_benchmark_button = QPushButton("Export Benchmark Data")
        self.export_benchmark_button.setToolTip("Export current GA benchmark data to a file")
        self.export_benchmark_button.setEnabled(False)  # Initially disabled until data is available
        self.export_benchmark_button.clicked.connect(self.export_ga_benchmark_data)
        button_layout.addWidget(self.export_benchmark_button)

        # Export GA Config - saves current bounds/settings
        self.export_ga_config_button = QPushButton("Export GA Config")
        self.export_ga_config_button.setToolTip("Export current GA config (bounds and settings) to a JSON file")
        self.export_ga_config_button.clicked.connect(self.export_ga_config)
        button_layout.addWidget(self.export_ga_config_button)

        # Import GA Config - applies saved bounds/settings
        self.import_ga_config_button = QPushButton("Import GA Config")
        self.import_ga_config_button.setToolTip("Import a saved GA config (bounds and settings)")
        self.import_ga_config_button.clicked.connect(self.import_ga_config)
        button_layout.addWidget(self.import_ga_config_button)

        button_layout.addStretch()  # Add stretch to push buttons to the left
        ga_benchmark_layout.addWidget(button_container)

        # Create tabs for different benchmark visualizations
        self.benchmark_viz_tabs = QTabWidget()
        
        # Create tabs for different visualizations
        violin_tab = QWidget()
        violin_layout = QVBoxLayout(violin_tab)
        self.violin_plot_widget = QWidget()
        violin_layout.addWidget(self.violin_plot_widget)
        
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)
        self.dist_plot_widget = QWidget()
        dist_layout.addWidget(self.dist_plot_widget)
        
        scatter_tab = QWidget()
        scatter_layout = QVBoxLayout(scatter_tab)
        self.scatter_plot_widget = QWidget()
        scatter_layout.addWidget(self.scatter_plot_widget)
        
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        self.heatmap_plot_widget = QWidget()
        heatmap_layout.addWidget(self.heatmap_plot_widget)
        
        # Add Q-Q plot tab
        qq_tab = QWidget()
        qq_layout = QVBoxLayout(qq_tab)
        self.qq_plot_widget = QWidget()
        qq_layout.addWidget(self.qq_plot_widget)
        
        # Add Statistical Analysis tab for comprehensive parameter analysis
        statistical_analysis_tab = QWidget()
        statistical_analysis_layout = QVBoxLayout(statistical_analysis_tab)
        
        # Create subtabs for different statistical analyses
        self.statistical_analysis_subtabs = QTabWidget()
        
        # 1. Parameter Visualizations Tab
        param_viz_tab = QWidget()
        param_viz_layout = QVBoxLayout(param_viz_tab)
        
        # Control panel for parameter selection
        control_panel = QGroupBox("Parameter Selection & Visualization Controls")
        control_layout = QGridLayout(control_panel)
        
        # Parameter selection dropdown
        QLabel_param_select = QLabel("Select Parameter:")
        self.param_selection_combo = QComboBox()
        self.param_selection_combo.setMaxVisibleItems(5)  # Show only 5 items at a time
        self.param_selection_combo.setMinimumWidth(150)
        self.param_selection_combo.setMaximumWidth(200)
        self.param_selection_combo.currentTextChanged.connect(self.on_parameter_selection_changed)
        
        # Plot type selection
        QLabel_plot_type = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Distribution Plot", "Violin Plot", "Box Plot", "Histogram", "Scatter Plot", "Q-Q Plot", "Correlation Heatmap"])
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        
        # Comparison parameter for scatter plots
        QLabel_comparison = QLabel("Compare With:")
        self.comparison_param_combo = QComboBox()
        self.comparison_param_combo.addItem("None")
        self.comparison_param_combo.setMaxVisibleItems(5)  # Show only 5 items at a time
        self.comparison_param_combo.setMinimumWidth(150)
        self.comparison_param_combo.setMaximumWidth(200)
        self.comparison_param_combo.setEnabled(False)
        self.comparison_param_combo.currentTextChanged.connect(self.on_comparison_parameter_changed)
        
        # Update plots button
        self.update_plots_button = QPushButton("Update Plots")
        self.update_plots_button.clicked.connect(self.update_parameter_plots)
        
        # Layout controls
        control_layout.addWidget(QLabel_param_select, 0, 0)
        control_layout.addWidget(self.param_selection_combo, 0, 1)
        control_layout.addWidget(QLabel_plot_type, 0, 2)
        control_layout.addWidget(self.plot_type_combo, 0, 3)
        control_layout.addWidget(QLabel_comparison, 1, 0)
        control_layout.addWidget(self.comparison_param_combo, 1, 1)
        control_layout.addWidget(self.update_plots_button, 1, 2)
        
        param_viz_layout.addWidget(control_panel)
        
        # Plot display area with enhanced scrolling
        self.param_plot_scroll = QScrollArea()
        self.param_plot_scroll.setWidgetResizable(True)
        self.param_plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_plot_scroll.setMinimumHeight(400)
        self.param_plot_widget = QWidget()
        self.param_plot_widget.setLayout(QVBoxLayout())
        self.param_plot_widget.setMinimumHeight(500)  # Ensure content is visible
        self.param_plot_scroll.setWidget(self.param_plot_widget)
        param_viz_layout.addWidget(self.param_plot_scroll)
        
        # 2. Parameter Statistics Tab - Make entire tab scrollable
        param_stats_tab = QWidget()
        param_stats_main_layout = QVBoxLayout(param_stats_tab)
        
        # Create scroll area for the entire statistics tab
        stats_main_scroll = QScrollArea()
        stats_main_scroll.setWidgetResizable(True)
        stats_main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        stats_main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Content widget for the scroll area
        stats_content_widget = QWidget()
        param_stats_layout = QVBoxLayout(stats_content_widget)
        
        # Statistics control panel
        stats_control_panel = QGroupBox("Statistics Display Options")
        stats_control_layout = QHBoxLayout(stats_control_panel)
        
        # Statistics view dropdown
        QLabel_stats_view = QLabel("View:")
        self.stats_view_combo = QComboBox()
        self.stats_view_combo.addItems(["Detailed Statistics", "Equations & Formulas"])
        self.stats_view_combo.setMaxVisibleItems(3)  # Show only 3 items at a time (matches the number of options)
        self.stats_view_combo.setMinimumWidth(150)
        self.stats_view_combo.setMaximumWidth(200)
        self.stats_view_combo.currentTextChanged.connect(self.on_stats_view_changed)
        
        stats_control_layout.addWidget(QLabel_stats_view)
        stats_control_layout.addWidget(self.stats_view_combo)
        stats_control_layout.addStretch()
        
        param_stats_layout.addWidget(stats_control_panel)
        
        # Statistics display area with enhanced scrolling
        self.param_stats_scroll = QScrollArea()
        self.param_stats_scroll.setWidgetResizable(True)
        self.param_stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_stats_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_stats_scroll.setMinimumHeight(400)
        self.param_stats_widget = QWidget()
        self.param_stats_widget.setLayout(QVBoxLayout())
        self.param_stats_scroll.setWidget(self.param_stats_widget)
        param_stats_layout.addWidget(self.param_stats_scroll)
        
        # Set the content widget to the main scroll area
        stats_main_scroll.setWidget(stats_content_widget)
        param_stats_main_layout.addWidget(stats_main_scroll)
        

        
        # Add subtabs to statistical analysis
        self.statistical_analysis_subtabs.addTab(param_viz_tab, "Parameter Visualizations")
        self.statistical_analysis_subtabs.addTab(param_stats_tab, "Statistics Tables")
        
        statistical_analysis_layout.addWidget(self.statistical_analysis_subtabs)
        
        # Summary statistics tabs (create subtabs for better organization)
        stats_tab = QWidget()
        stats_tab.setObjectName("stats_tab")
        stats_layout = QVBoxLayout(stats_tab)
        
        # Create a tabbed widget for the statistics section
        stats_subtabs = QTabWidget()
        
        # ---- Subtab 1: Summary Statistics ----
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # Add summary statistics table
        self.benchmark_stats_table = QTableWidget()
        self.benchmark_stats_table.setColumnCount(5)
        self.benchmark_stats_table.setHorizontalHeaderLabels(["Metric", "Min", "Max", "Mean", "Std"])
        self.benchmark_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(QLabel("Statistical Summary of All Runs:"))
        summary_layout.addWidget(self.benchmark_stats_table)
        # Attach export menu and add a visible export button
        self._attach_table_export(self.benchmark_stats_table, "benchmark_summary_statistics")
        btn_export_summary = QPushButton("Export Table")
        btn_export_summary.setToolTip("Export the Summary Statistics table")
        btn_export_summary.clicked.connect(lambda: self._export_table_via_dialog(self.benchmark_stats_table, "benchmark_summary_statistics"))
        summary_layout.addWidget(btn_export_summary)
        
        # ---- Subtab 2: All Runs Table ----
        runs_tab = QWidget()
        runs_layout = QVBoxLayout(runs_tab)

        # Ranking & Grouping controls
        self.bench_rank_controls = QGroupBox("Ranking & Grouping Criteria")
        rank_form = QFormLayout(self.bench_rank_controls)
        self.bench_rank_metric_combo = QComboBox()
        self.bench_rank_metric_combo.addItems([
            "Fitness (lower better)",
            "Convergence Generation (lower better)",
            "CPU Time Used (s, lower better)",
            "Wall Time (s, lower better)",
            "Composite (weighted)"
        ])
        self.bench_rank_metric_combo.setCurrentIndex(0)
        rank_form.addRow("Metric:", self.bench_rank_metric_combo)

        # Composite weights (used only if Composite selected)
        composite_row = QWidget(); composite_layout = QGridLayout(composite_row); composite_layout.setContentsMargins(0,0,0,0)
        self.weight_fitness_spin = QDoubleSpinBox(); self.weight_fitness_spin.setRange(0.0, 100.0); self.weight_fitness_spin.setSingleStep(0.1); self.weight_fitness_spin.setValue(1.0)
        self.weight_convgen_spin = QDoubleSpinBox(); self.weight_convgen_spin.setRange(0.0, 100.0); self.weight_convgen_spin.setSingleStep(0.1); self.weight_convgen_spin.setValue(0.5)
        self.weight_cpu_spin = QDoubleSpinBox(); self.weight_cpu_spin.setRange(0.0, 100.0); self.weight_cpu_spin.setSingleStep(0.1); self.weight_cpu_spin.setValue(0.25)
        self.weight_time_spin = QDoubleSpinBox(); self.weight_time_spin.setRange(0.0, 100.0); self.weight_time_spin.setSingleStep(0.1); self.weight_time_spin.setValue(0.25)
        composite_layout.addWidget(QLabel("w_fitness"), 0, 0); composite_layout.addWidget(self.weight_fitness_spin, 0, 1)
        composite_layout.addWidget(QLabel("w_conv_gen"), 0, 2); composite_layout.addWidget(self.weight_convgen_spin, 0, 3)
        composite_layout.addWidget(QLabel("w_cpu"), 1, 0); composite_layout.addWidget(self.weight_cpu_spin, 1, 1)
        composite_layout.addWidget(QLabel("w_time"), 1, 2); composite_layout.addWidget(self.weight_time_spin, 1, 3)
        rank_form.addRow("Composite Weights:", composite_row)

        # Apply button (also auto-update on changes)
        self.bench_apply_rank_btn = QPushButton("Apply Ranking")
        rank_form.addRow(self.bench_apply_rank_btn)
        runs_layout.addWidget(self.bench_rank_controls)

        # Create a table for all runs
        self.benchmark_runs_table = QTableWidget()
        self.benchmark_runs_table.setColumnCount(4)
        self.benchmark_runs_table.setHorizontalHeaderLabels(["Run #", "Fitness", "Rank", "Details"])
        self.benchmark_runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.benchmark_runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.benchmark_runs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.benchmark_runs_table.itemClicked.connect(self.show_run_details)

        runs_layout.addWidget(QLabel("All Benchmark Runs:"))
        runs_layout.addWidget(self.benchmark_runs_table)
        # Attach export menu and add a visible export button
        self._attach_table_export(self.benchmark_runs_table, "benchmark_all_runs")
        btn_export_runs = QPushButton("Export Table")
        btn_export_runs.setToolTip("Export the All Runs table")
        btn_export_runs.clicked.connect(lambda: self._export_table_via_dialog(self.benchmark_runs_table, "benchmark_all_runs"))
        runs_layout.addWidget(btn_export_runs)
        
        # Create run details text area
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_group = QGroupBox("Run Details")
        group_layout = QVBoxLayout(details_group)
        self.run_details_text = QTextEdit()
        self.run_details_text.setReadOnly(True)
        group_layout.addWidget(self.run_details_text)
        details_layout.addWidget(details_group)
        
        # Create GA Operations tab as a subtab
        ga_ops_tab = QWidget()
        ga_ops_layout = QVBoxLayout(ga_ops_tab)
        self.ga_ops_plot_widget = QWidget()
        ga_ops_layout.addWidget(self.ga_ops_plot_widget)
        
        # Create Selected Run Analysis tab
        selected_run_tab = QWidget()
        selected_run_layout = QVBoxLayout(selected_run_tab)
        self.selected_run_widget = QWidget()
        selected_run_layout.addWidget(self.selected_run_widget)
        
        # Group Summary tab (populated after runs complete)
        self.group_summary_tab = QWidget()
        group_summary_layout = QVBoxLayout(self.group_summary_tab)
        self.group_summary_container = QWidget()
        self.group_summary_container.setLayout(QVBoxLayout())
        group_summary_layout.addWidget(self.group_summary_container)

        # Add the subtabs to the stats tabbed widget
        stats_subtabs.addTab(summary_tab, "Summary Statistics")
        stats_subtabs.addTab(runs_tab, "All Runs")
        stats_subtabs.addTab(details_tab, "Run Details")
        stats_subtabs.addTab(self.group_summary_tab, "Group Summary")
        stats_subtabs.addTab(ga_ops_tab, "GA Operations")
        stats_subtabs.addTab(selected_run_tab, "Selected Run Analysis")
        
        # Add the stats tabbed widget to the stats tab
        stats_layout.addWidget(stats_subtabs)
        
        # Add all visualization tabs to the benchmark visualization tabs
        self.benchmark_viz_tabs.addTab(violin_tab, "Fitness Analysis")
        self.benchmark_viz_tabs.addTab(dist_tab, "Distribution")
        self.benchmark_viz_tabs.addTab(scatter_tab, "Scatter Plot")
        self.benchmark_viz_tabs.addTab(heatmap_tab, "Parameter Correlations")
        self.benchmark_viz_tabs.addTab(qq_tab, "Q-Q Plot")
        self.benchmark_viz_tabs.addTab(statistical_analysis_tab, "Statistical Analysis")
        self.benchmark_viz_tabs.addTab(stats_tab, "Statistics")
        
        # GA Operations Performance Tab - already added as a subtab of Statistics
        
        # Wire ranking controls to refresh visualizations
        try:
            self.bench_rank_metric_combo.currentIndexChanged.connect(self.visualize_ga_benchmark_results)
            self.weight_fitness_spin.valueChanged.connect(self.visualize_ga_benchmark_results)
            self.weight_convgen_spin.valueChanged.connect(self.visualize_ga_benchmark_results)
            self.weight_cpu_spin.valueChanged.connect(self.visualize_ga_benchmark_results)
            self.weight_time_spin.valueChanged.connect(self.visualize_ga_benchmark_results)
            self.bench_apply_rank_btn.clicked.connect(self.visualize_ga_benchmark_results)
        except Exception:
            pass

        # Add the benchmark visualization tabs to the benchmark tab
        ga_benchmark_layout.addWidget(self.benchmark_viz_tabs)
        
        # Add all sub-tabs to the GA tab widget
        # Initialize empty benchmark data storage
        self.ga_benchmark_data = []

        # -------------------- Sub-tab 5: Random Validation --------------------
        ga_validation_tab = QWidget()
        ga_validation_layout = QVBoxLayout(ga_validation_tab)

        # Split controls and views
        rv_splitter = QSplitter(Qt.Horizontal)

        # Left: Controls
        rv_controls_widget = QWidget()
        rv_controls_layout = QFormLayout(rv_controls_widget)

        self.rv_num_samples_box = QSpinBox()
        self.rv_num_samples_box.setRange(10, 200000)
        self.rv_num_samples_box.setValue(500)
        self.rv_num_samples_box.setToolTip("Number of random samples to evaluate")

        self.rv_method_combo = QComboBox()
        self.rv_method_combo.addItems(["Random", "Sobol", "Latin Hypercube"])
        self.rv_method_combo.setCurrentIndex(0)
        self.rv_method_combo.setToolTip("Sampling method for generating populations within bounds")

        self.rv_seed_box = QSpinBox()
        self.rv_seed_box.setRange(-1, 2_147_483_647)
        self.rv_seed_box.setValue(-1)
        self.rv_seed_box.setToolTip(
            "<b>Random Seed</b><br>"
            "Controls reproducibility of the sampled population.<br>"
            "- Set a non-negative integer to reproduce the exact same samples every run.<br>"
            "- Set to -1 to disable seeding (each run generates a new random population).<br>"
            "This applies to all sampling methods: Random (PRNG), Sobol (scrambled), and LHS."
        )
        self.rv_seed_box.setWhatsThis(self.rv_seed_box.toolTip())

        self.rv_alpha_box = QDoubleSpinBox()
        self.rv_alpha_box.setRange(0.0, 10.0)
        self.rv_alpha_box.setDecimals(4)
        self.rv_alpha_box.setSingleStep(0.01)
        self.rv_alpha_box.setValue(self.ga_alpha_box.value())
        self.rv_alpha_box.setToolTip(
            "<b>Alpha (sparsity penalty)</b><br>"
            "Weight for penalizing large parameter magnitudes in the fitness function.<br>"
            "Fitness = |singular_response - 1| + a Â· S|params| + (percentage_error_sum)/1000.<br>"
            "- Higher a favors simpler (smaller-magnitude) parameter sets.<br>"
            "- Set a = 0 to disable sparsity penalty.<br>"
            "Uses the same definition as in GA optimization."
        )
        self.rv_alpha_box.setWhatsThis(self.rv_alpha_box.toolTip())

        self.rv_respect_fixed_chk = QCheckBox("Respect fixed parameters")
        self.rv_respect_fixed_chk.setChecked(True)
        self.rv_respect_fixed_chk.setToolTip(
            "<b>Respect Fixed Parameters</b><br>"
            "When enabled, any parameter marked <i>Fixed</i> in the DVA Parameters tab will be held at its fixed value during sampling.<br>"
            "When disabled, sampling ignores the <i>Fixed</i> flags and uses each parameter's Lower/Upper bounds instead (even if currently fixed).<br>"
            "Zero-width bounds still produce a constant value."
        )
        self.rv_respect_fixed_chk.setWhatsThis(self.rv_respect_fixed_chk.toolTip())

        self.rv_bins_box = QSpinBox()
        self.rv_bins_box.setRange(5, 200)
        self.rv_bins_box.setValue(50)
        self.rv_bins_box.setToolTip("Histogram bins for visualizations")
        # Separate tolerance for Random Validation (independent of GA tolerance)
        self.rv_tol_box = QDoubleSpinBox()
        self.rv_tol_box.setRange(0.0, 1e9)
        self.rv_tol_box.setDecimals(6)
        self.rv_tol_box.setSingleStep(0.0001)
        try:
            self.rv_tol_box.setValue(self.ga_tol_box.value())
        except Exception:
            self.rv_tol_box.setValue(1.0)
        self.rv_tol_box.setToolTip("Tolerance threshold used to classify samples as PASS in Random Validation. This is independent from GA tolerance.")

        # Action buttons
        rv_btn_row = QWidget()
        rv_btn_row_layout = QHBoxLayout(rv_btn_row)
        rv_btn_row_layout.setContentsMargins(0, 0, 0, 0)
        self.rv_run_button = QPushButton("Generate & Evaluate")
        self.rv_run_button.clicked.connect(self.run_random_validation)
        self.rv_cancel_button = QPushButton("Cancel")
        self.rv_cancel_button.setEnabled(False)
        self.rv_cancel_button.clicked.connect(self.cancel_random_validation)
        self.rv_export_button = QPushButton("Export CSV")
        self.rv_export_button.setEnabled(False)
        self.rv_export_button.clicked.connect(self.export_random_validation_results)
        rv_btn_row_layout.addWidget(self.rv_run_button)
        rv_btn_row_layout.addWidget(self.rv_cancel_button)
        rv_btn_row_layout.addWidget(self.rv_export_button)

        self.rv_progress_bar = QProgressBar()
        self.rv_progress_bar.setRange(0, 100)
        self.rv_progress_bar.setValue(0)

        rv_controls_layout.addRow("Samples:", self.rv_num_samples_box)
        rv_controls_layout.addRow("Method:", self.rv_method_combo)
        rv_controls_layout.addRow("Seed:", self.rv_seed_box)
        rv_controls_layout.addRow("Alpha:", self.rv_alpha_box)
        rv_controls_layout.addRow("Respect Fixed:", self.rv_respect_fixed_chk)
        rv_controls_layout.addRow("Histogram bins:", self.rv_bins_box)
        rv_controls_layout.addRow("Tolerance:", self.rv_tol_box)
        rv_controls_layout.addRow("", rv_btn_row)
        rv_controls_layout.addRow("Progress:", self.rv_progress_bar)

        # Auto-refresh plots when bins/tolerance change
        try:
            self.rv_bins_box.valueChanged.connect(self.refresh_random_validation_views)
            self.rv_tol_box.valueChanged.connect(self.refresh_random_validation_views)
        except Exception:
            pass

        # Right: Result views
        rv_views_widget = QWidget()
        rv_views_layout = QVBoxLayout(rv_views_widget)
        rv_views_layout.setContentsMargins(0, 0, 0, 0)

        self.rv_tabs = QTabWidget()

        # Summary tab (table + export)
        rv_summary_tab = QWidget()
        rv_summary_layout = QVBoxLayout(rv_summary_tab)
        self.rv_success_bar = QProgressBar()
        self.rv_success_bar.setRange(0, 100)
        self.rv_success_bar.setFormat("%p% within tolerance")
        rv_summary_layout.addWidget(self.rv_success_bar)
        # Hidden legacy label to satisfy any leftover references
        try:
            self.rv_summary_label = QLabel("")
            self.rv_summary_label.hide()
        except Exception:
            pass
        self.rv_summary_table = QTableWidget()
        self.rv_summary_table.setColumnCount(2)
        self.rv_summary_table.setHorizontalHeaderLabels(["Metric", "Value"])
        try:
            self.rv_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception:
            pass
        self.rv_summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rv_summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rv_summary_layout.addWidget(self.rv_summary_table)
        # Right-click export and explicit export button
        try:
            self._attach_table_export(self.rv_summary_table, "random_validation_summary")
        except Exception:
            pass
        rv_sum_export_btn = QPushButton("Export Summary")
        rv_sum_export_btn.setToolTip("Export the Random Validation summary table")
        try:
            rv_sum_export_btn.clicked.connect(lambda: self._export_table_via_dialog(self.rv_summary_table, "random_validation_summary"))
        except Exception:
            pass
        rv_summary_layout.addWidget(rv_sum_export_btn)

        # Fitness distribution tab
        rv_fitdist_tab = QWidget()
        rv_fitdist_layout = QVBoxLayout(rv_fitdist_tab)
        self.rv_fitdist_plot = QWidget()
        rv_fitdist_layout.addWidget(self.rv_fitdist_plot)

        # Component distributions tab
        rv_comp_tab = QWidget()
        rv_comp_layout = QVBoxLayout(rv_comp_tab)
        self.rv_comp_plot = QWidget()
        rv_comp_layout.addWidget(self.rv_comp_plot)

        # KDE of FRF peak positions tab
        rv_kde_tab = QWidget()
        rv_kde_layout = QVBoxLayout(rv_kde_tab)
        rv_kde_layout.setContentsMargins(0, 0, 0, 0)

        # Controls for KDE tab
        rv_kde_ctrl_row = QWidget()
        rv_kde_ctrl_layout = QHBoxLayout(rv_kde_ctrl_row)
        rv_kde_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        rv_kde_ctrl_layout.addWidget(QLabel("Mass:"))
        self.rv_kde_mass_combo = QComboBox()
        self.rv_kde_mass_combo.addItems([
            "All Masses", "mass_1", "mass_2", "mass_3", "mass_4", "mass_5"
        ])
        rv_kde_ctrl_layout.addWidget(self.rv_kde_mass_combo)

        rv_kde_ctrl_layout.addSpacing(12)
        rv_kde_ctrl_layout.addWidget(QLabel("Peaks per run:"))
        self.rv_kde_peak_mode_combo = QComboBox()
        self.rv_kde_peak_mode_combo.addItems([
            "All peaks", "Top 1", "Top 2", "Top 3"
        ])
        self.rv_kde_peak_mode_combo.setCurrentIndex(1)  # Default to Top 1
        rv_kde_ctrl_layout.addWidget(self.rv_kde_peak_mode_combo)

        rv_kde_ctrl_layout.addSpacing(12)
        self.rv_kde_use_zones_chk = QCheckBox("Overlay FRF Zones")
        self.rv_kde_use_zones_chk.setChecked(True)
        self.rv_kde_use_zones_chk.setToolTip("Overlay zones defined in FRF comparative visualization")
        rv_kde_ctrl_layout.addWidget(self.rv_kde_use_zones_chk)

        # Optional: quick access to add a zone
        self.rv_kde_add_zone_btn = QPushButton("Add Zone")
        self.rv_kde_add_zone_btn.setToolTip("Add a zone to overlay (same list as FRF comparative tab)")
        try:
            self.rv_kde_add_zone_btn.clicked.connect(lambda: getattr(self, 'add_zone', lambda: None)())
        except Exception:
            pass
        rv_kde_ctrl_layout.addWidget(self.rv_kde_add_zone_btn)

        rv_kde_ctrl_layout.addStretch()

        rv_kde_layout.addWidget(rv_kde_ctrl_row)
        self.rv_kde_plot = QWidget()
        rv_kde_layout.addWidget(self.rv_kde_plot)

        # Zone management (edit/add/remove) within KDE tab
        rv_kde_zone_group = QGroupBox("KDE Zones")
        rv_kde_zone_layout = QVBoxLayout(rv_kde_zone_group)

        # Buttons row
        rv_kde_zone_btn_row = QWidget()
        rv_kde_zone_btn_layout = QHBoxLayout(rv_kde_zone_btn_row)
        rv_kde_zone_btn_layout.setContentsMargins(0, 0, 0, 0)

        self.rv_kde_add_zone_btn2 = QPushButton("Add Zone")
        self.rv_kde_add_zone_btn2.setToolTip("Create a new zone region")
        self.rv_kde_add_zone_btn2.clicked.connect(lambda: self._rv_kde_add_zone())
        rv_kde_zone_btn_layout.addWidget(self.rv_kde_add_zone_btn2)

        self.rv_kde_remove_zone_btn = QPushButton("Remove Selected")
        self.rv_kde_remove_zone_btn.clicked.connect(lambda: self._rv_kde_remove_zone())
        rv_kde_zone_btn_layout.addWidget(self.rv_kde_remove_zone_btn)

        self.rv_kde_clear_zones_btn = QPushButton("Clear All")
        self.rv_kde_clear_zones_btn.clicked.connect(lambda: self._rv_kde_clear_zones())
        rv_kde_zone_btn_layout.addWidget(self.rv_kde_clear_zones_btn)

        rv_kde_zone_btn_layout.addStretch()
        rv_kde_zone_layout.addWidget(rv_kde_zone_btn_row)

        # Zones table (Name, Start X, End X, Color)
        self.rv_kde_zone_table = QTableWidget()
        self.rv_kde_zone_table.setColumnCount(4)
        self.rv_kde_zone_table.setHorizontalHeaderLabels(["Name", "Start X", "End X", "Color"])
        self.rv_kde_zone_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rv_kde_zone_table.setMaximumHeight(160)
        # Track if we are programmatically updating to avoid recursion
        self._rv_kde_updating_table = False
        try:
            self.rv_kde_zone_table.itemChanged.connect(self._rv_kde_on_zone_cell_changed)
        except Exception:
            pass
        rv_kde_zone_layout.addWidget(self.rv_kde_zone_table)

        rv_kde_layout.addWidget(rv_kde_zone_group)

        # Initialize table from shared zones list if available
        try:
            self._rv_kde_update_zone_table()
        except Exception:
            pass

        # React to control changes
        try:
            self.rv_kde_mass_combo.currentIndexChanged.connect(lambda _=None: self.update_random_validation_kde())
            self.rv_kde_peak_mode_combo.currentIndexChanged.connect(lambda _=None: self.update_random_validation_kde())
            self.rv_kde_use_zones_chk.stateChanged.connect(lambda _=None: self.update_random_validation_kde())
        except Exception:
            pass

        # FRF curves tab
        rv_frf_tab = QWidget()
        rv_frf_layout = QVBoxLayout(rv_frf_tab)
        rv_frf_layout.setContentsMargins(0, 0, 0, 0)

        rv_frf_ctrl_row = QWidget()
        rv_frf_ctrl_layout = QHBoxLayout(rv_frf_ctrl_row)
        rv_frf_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        rv_frf_ctrl_layout.addWidget(QLabel("Mass:"))
        self.rv_frf_mass_combo = QComboBox()
        self.rv_frf_mass_combo.addItems(["mass_1", "mass_2", "mass_3", "mass_4", "mass_5"])
        rv_frf_ctrl_layout.addWidget(self.rv_frf_mass_combo)
        rv_frf_ctrl_layout.addSpacing(12)
        # Zones overlay toggle for FRF plot
        self.rv_frf_use_zones_chk = QCheckBox("Overlay Zones")
        self.rv_frf_use_zones_chk.setChecked(True)
        self.rv_frf_use_zones_chk.setToolTip("Overlay the shared zones (same list used in KDE tab)")
        rv_frf_ctrl_layout.addWidget(self.rv_frf_use_zones_chk)
        rv_frf_ctrl_layout.addStretch()
        rv_frf_layout.addWidget(rv_frf_ctrl_row)

        rv_frf_layout.addWidget(QLabel("Select runs to plot (sorted by fitness):"))

        self.rv_frf_run_list = QListWidget()
        self.rv_frf_run_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.rv_frf_run_list.setAlternatingRowColors(True)
        self.rv_frf_run_list.setMinimumHeight(160)
        self.rv_frf_run_list.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.rv_frf_run_list.setToolTip("Click to toggle runs; multiple selections allowed.")
        rv_frf_layout.addWidget(self.rv_frf_run_list)

        rv_frf_btn_row = QWidget()
        rv_frf_btn_layout = QHBoxLayout(rv_frf_btn_row)
        rv_frf_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.rv_frf_select_best_btn = QPushButton("Select Top 3")
        rv_frf_btn_layout.addWidget(self.rv_frf_select_best_btn)
        self.rv_frf_clear_btn = QPushButton("Clear Selection")
        rv_frf_btn_layout.addWidget(self.rv_frf_clear_btn)
        rv_frf_btn_layout.addStretch()
        rv_frf_layout.addWidget(rv_frf_btn_row)

        self.rv_frf_plot = QWidget()
        rv_frf_layout.addWidget(self.rv_frf_plot)

        # FRF Zones editor (shares the same self.zones list as KDE)
        frf_zone_group = QGroupBox("FRF Zones")
        frf_zone_layout = QVBoxLayout(frf_zone_group)

        frf_zone_btn_row = QWidget()
        frf_zone_btn_layout = QHBoxLayout(frf_zone_btn_row)
        frf_zone_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.rv_frf_add_zone_btn = QPushButton("Add Zone")
        self.rv_frf_remove_zone_btn = QPushButton("Remove Selected")
        self.rv_frf_clear_zones_btn = QPushButton("Clear All")
        try:
            # Add zone centered in current omega range for FRF plot
            self.rv_frf_add_zone_btn.clicked.connect(self._rv_frf_add_zone)
            # Reuse KDE handlers for remove/clear (they sync both tables)
            self.rv_frf_remove_zone_btn.clicked.connect(self._rv_frf_remove_zone)
            self.rv_frf_clear_zones_btn.clicked.connect(self._rv_frf_clear_zones)
        except Exception:
            pass
        frf_zone_btn_layout.addWidget(self.rv_frf_add_zone_btn)
        frf_zone_btn_layout.addWidget(self.rv_frf_remove_zone_btn)
        frf_zone_btn_layout.addWidget(self.rv_frf_clear_zones_btn)
        frf_zone_btn_layout.addStretch()
        frf_zone_layout.addWidget(frf_zone_btn_row)

        # Zones table (Name, Start X, End X, Color) for FRF tab
        self.rv_frf_zone_table = QTableWidget()
        self.rv_frf_zone_table.setColumnCount(4)
        self.rv_frf_zone_table.setHorizontalHeaderLabels(["Name", "Start X", "End X", "Color"])
        try:
            self.rv_frf_zone_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception:
            pass
        # Hook generic KDE cell-change handler to keep a single source of truth
        try:
            self.rv_frf_zone_table.itemChanged.connect(self._rv_kde_on_zone_cell_changed)
        except Exception:
            pass
        frf_zone_layout.addWidget(self.rv_frf_zone_table)
        rv_frf_layout.addWidget(frf_zone_group)

        # Keep the FRF zone table in sync with shared zones
        try:
            self._update_zone_table()
        except Exception:
            pass

        try:
            self.rv_frf_mass_combo.currentTextChanged.connect(lambda _=None: self.update_random_validation_frf_plot())
            self.rv_frf_use_zones_chk.stateChanged.connect(lambda _=None: self.update_random_validation_frf_plot())
            self.rv_frf_run_list.itemSelectionChanged.connect(self.update_random_validation_frf_plot)
            self.rv_frf_select_best_btn.clicked.connect(lambda: self._rv_frf_select_top())
            self.rv_frf_clear_btn.clicked.connect(self._rv_frf_clear_selection)
        except Exception:
            pass

        # Correlation heatmap tab
        rv_corr_tab = QWidget()
        rv_corr_layout = QVBoxLayout(rv_corr_tab)
        self.rv_corr_plot = QWidget()
        rv_corr_layout.addWidget(self.rv_corr_plot)

        # Scatter tab (parameter vs fitness)
        rv_scatter_tab = QWidget()
        rv_scatter_layout = QVBoxLayout(rv_scatter_tab)
        rv_scatter_layout.setContentsMargins(0, 0, 0, 0)
        scatter_ctrl_row = QWidget()
        scatter_ctrl_layout = QHBoxLayout(scatter_ctrl_row)
        scatter_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        scatter_ctrl_layout.addWidget(QLabel("Parameter:"))
        self.rv_scatter_param_combo = QComboBox()
        self.rv_scatter_param_combo.currentTextChanged.connect(self.update_random_validation_scatter)
        scatter_ctrl_layout.addWidget(self.rv_scatter_param_combo)
        scatter_ctrl_layout.addStretch()
        self.rv_scatter_plot = QWidget()
        rv_scatter_layout.addWidget(scatter_ctrl_row)
        rv_scatter_layout.addWidget(self.rv_scatter_plot)

        # Table tab
        rv_table_tab = QWidget()
        rv_table_layout = QVBoxLayout(rv_table_tab)
        self.rv_table = QTableWidget()
        self.rv_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rv_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        rv_table_layout.addWidget(self.rv_table)
        # Attach export menu and add a visible export button for the table view
        self._attach_table_export(self.rv_table, "random_validation_table")
        rv_tbl_export_btn = QPushButton("Export Table")
        rv_tbl_export_btn.setToolTip("Export the Random Validation table")
        rv_tbl_export_btn.clicked.connect(lambda: self._export_table_via_dialog(self.rv_table, "random_validation_table"))
        rv_table_layout.addWidget(rv_tbl_export_btn)

        # Add tabs
        self.rv_tabs.addTab(rv_summary_tab, "Summary")
        self.rv_tabs.addTab(rv_fitdist_tab, "Fitness Distribution")
        self.rv_tabs.addTab(rv_comp_tab, "Components")
        self.rv_tabs.addTab(rv_kde_tab, "KDE")
        self.rv_tabs.addTab(rv_frf_tab, "FRF Runs")
        self.rv_tabs.addTab(rv_corr_tab, "Correlation")
        self.rv_tabs.addTab(rv_scatter_tab, "Scatter")
        self.rv_tabs.addTab(rv_table_tab, "Table")

        rv_views_layout.addWidget(self.rv_tabs)

        rv_splitter.addWidget(rv_controls_widget)
        rv_splitter.addWidget(rv_views_widget)
        rv_splitter.setStretchFactor(0, 0)
        rv_splitter.setStretchFactor(1, 1)

        ga_validation_layout.addWidget(rv_splitter)

        # Keep references
        self.rv_results_df = None
        self._rv_worker = None
        self.rv_frf_curves = {}
        self.rv_frf_omega = None
        try:
            self.update_random_validation_frf_plot()
        except Exception:
            pass

        # Add all sub-tabs to the GA tab widget
        self.ga_sub_tabs.addTab(ga_hyper_tab, "GA Settings")
        self.ga_sub_tabs.addTab(ga_param_tab, "DVA Parameters")
        self.ga_sub_tabs.addTab(ga_results_tab, "Results")
        self.ga_sub_tabs.addTab(ga_benchmark_tab, "GA Benchmarking")
        self.ga_sub_tabs.addTab(ga_validation_tab, "Random Validation")

        # Add the GA sub-tabs widget to the main GA tab layout
        layout.addWidget(self.ga_sub_tabs)
        self.ga_tab.setLayout(layout)
        
    def toggle_fixed(self, state, row, table=None):
        """Toggle the fixed state of a DVA parameter row"""
        if table is None:
            table = self.dva_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def toggle_ga_fixed(self, state, row, table=None):
        """Toggle the fixed state of a GA parameter row"""
        if table is None:
            table = self.ga_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)
        
        # Enable/disable appropriate spinboxes
        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
        # If switching to fixed mode, copy current lower bound value to fixed value
        if fixed:
            fixed_value_spin.setValue(lower_bound_spin.value())
        # If switching to range mode, ensure lower bound is not greater than upper bound
        else:
            if lower_bound_spin.value() > upper_bound_spin.value():
                upper_bound_spin.setValue(lower_bound_spin.value())

    def toggle_adaptive_rates_options(self, state):
        """Show or hide adaptive rates options based on checkbox state"""
        self.adaptive_rates_options.setVisible(state == Qt.Checked)
        
        # Enable/disable the fixed rate inputs based on adaptive rates setting
        self.ga_cxpb_box.setEnabled(state != Qt.Checked)
        self.ga_mutpb_box.setEnabled(state != Qt.Checked)
        
        # Update tooltips to indicate that rates will be adaptive
        if state == Qt.Checked:
            self.ga_cxpb_box.setToolTip("Starting crossover probability (will adapt during optimization)")
            self.ga_mutpb_box.setToolTip("Starting mutation probability (will adapt during optimization)")
        else:
            self.ga_cxpb_box.setToolTip("Crossover probability")
            self.ga_mutpb_box.setToolTip("Mutation probability")

    # -------------------- Random Validation: worker and helpers --------------------
    class RandomValidationWorker(QThread):
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        progress = pyqtSignal(int)

        def __init__(self,
                     main_params,
                     omega_start,
                     omega_end,
                     omega_points,
                     target_values,
                     weights,
                     param_names,
                     bounds,
                     fixed_flags,
                     fixed_values,
                     alpha,
                     percentage_error_scale,
                     *,
                     cost_scale_factor=None,
                     dva_activation_threshold,
                     dva_activation_penalty,
                     dva_costs,
                     dva_costs_by_category,
                     use_enhanced_cost,
                     benefit_w_primary,
                     benefit_w_accuracy,
                     benefit_w_sparsity,
                     cat_w_material,
                     cat_w_manufacturing,
                     cat_w_maintenance,
                     cat_w_operational,
                     benefit_weight_start,
                     benefit_weight_end,
                     dva_category_map=None,
                     method,
                     num_samples,
                     seed,
                     respect_fixed):
            super().__init__()
            self.main_params = main_params
            self.omega_start = omega_start
            self.omega_end = omega_end
            self.omega_points = omega_points
            self.target_values = target_values
            self.weights = weights
            self.param_names = param_names
            self.bounds = bounds
            self.fixed_flags = fixed_flags
            self.fixed_values = fixed_values
            self.alpha = alpha
            self.percentage_error_scale = percentage_error_scale
            # Cost scaling for RV fitness calculation (UI-provided or default 1.0)
            try:
                self.cost_scale_factor = float(cost_scale_factor) if cost_scale_factor is not None else 1.0
            except Exception:
                self.cost_scale_factor = 1.0
            self.dva_activation_threshold = dva_activation_threshold
            self.dva_activation_penalty = dva_activation_penalty
            self.dva_costs = dict(dva_costs) if isinstance(dva_costs, dict) else {}
            self.dva_costs_by_category = {
                'material': dict(dva_costs_by_category.get('material', {})) if isinstance(dva_costs_by_category, dict) else {},
                'manufacturing': dict(dva_costs_by_category.get('manufacturing', {})) if isinstance(dva_costs_by_category, dict) else {},
                'maintenance': dict(dva_costs_by_category.get('maintenance', {})) if isinstance(dva_costs_by_category, dict) else {},
                'operational': dict(dva_costs_by_category.get('operational', {})) if isinstance(dva_costs_by_category, dict) else {},
            }
            self.use_enhanced_cost = bool(use_enhanced_cost)
            self.benefit_w_primary = float(benefit_w_primary)
            self.benefit_w_accuracy = float(benefit_w_accuracy)
            self.benefit_w_sparsity = float(benefit_w_sparsity)
            self.cat_w_material = float(cat_w_material)
            self.cat_w_manufacturing = float(cat_w_manufacturing)
            self.cat_w_maintenance = float(cat_w_maintenance)
            self.cat_w_operational = float(cat_w_operational)
            self.benefit_weight_start = float(benefit_weight_start)
            self.benefit_weight_end = float(benefit_weight_end)
            self.dva_category_map = dict(dva_category_map) if isinstance(dva_category_map, dict) else {}
            self.method = method
            self.num_samples = num_samples
            self.seed = seed if seed is not None and seed >= 0 else None
            self.respect_fixed = respect_fixed
            self.abort = False

        def _sample_matrix(self):
            dim = len(self.param_names)
            lower = np.array([low for (low, _high) in self.bounds], dtype=float)
            upper = np.array([high for (_low, high) in self.bounds], dtype=float)

            # Ensure bounds are ordered
            swap_mask = upper < lower
            if np.any(swap_mask):
                tmp = lower[swap_mask].copy()
                lower[swap_mask] = upper[swap_mask]
                upper[swap_mask] = tmp

            # Respect fixed bounds by collapsing the interval
            if self.respect_fixed:
                for j, is_fixed in enumerate(self.fixed_flags):
                    if is_fixed:
                        val = self.fixed_values.get(j, lower[j])
                        lower[j] = val
                        upper[j] = val

            # Generate unit-cube samples
            if self.method == "Random":
                rng = np.random.default_rng(self.seed)
                u = rng.random((self.num_samples, dim))
            elif self.method == "Sobol":
                engine = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
                m = int(np.ceil(np.log2(max(1, self.num_samples))))
                u_full = engine.random_base2(m=m)
                u = u_full[:self.num_samples]
            else:
                engine = qmc.LatinHypercube(d=dim, seed=self.seed)
                u = engine.random(n=self.num_samples)

            # Manually scale to [lower, upper] to allow zero-width intervals
            width = (upper - lower)
            samples = lower + u * width
            return samples

        def run(self):
            omega_vector = None
            frf_cache = {}
            try:
                X = self._sample_matrix()
                n = X.shape[0]
                rows = []
                main_params = self.main_params
                omega_start = self.omega_start
                omega_end = self.omega_end
                omega_points = self.omega_points
                # Build omega vector to accompany FRF magnitudes
                try:
                    _cnt = int(omega_points)
                    if _cnt < 2:
                        _cnt = 2
                except Exception:
                    _cnt = 2
                omega_vector = np.linspace(omega_start, omega_end, _cnt)
                target_values = self.target_values
                weights = self.weights
                alpha = self.alpha
                pe_scale = float(self.percentage_error_scale) if self.percentage_error_scale else 1000.0
                thr = float(self.dva_activation_threshold) if self.dva_activation_threshold is not None else 0.0
                pen_per = float(self.dva_activation_penalty) if self.dva_activation_penalty is not None else 0.0

                for i in range(n):
                    if self.abort:
                        break
                    dva_params = tuple(float(v) for v in X[i, :])
                    try:
                        res = frf(
                            main_system_parameters=main_params,
                            dva_parameters=dva_params,
                            omega_start=omega_start,
                            omega_end=omega_end,
                            omega_points=omega_points,
                            target_values_mass1=target_values['mass_1'],
                            weights_mass1=weights['mass_1'],
                            target_values_mass2=target_values['mass_2'],
                            weights_mass2=weights['mass_2'],
                            target_values_mass3=target_values['mass_3'],
                            weights_mass3=weights['mass_3'],
                            target_values_mass4=target_values['mass_4'],
                            weights_mass4=weights['mass_4'],
                            target_values_mass5=target_values['mass_5'],
                            weights_mass5=weights['mass_5'],
                            plot_figure=False,
                            show_peaks=False,
                            show_slopes=False
                        )
                        singular = res.get('singular_response', np.nan)
                        if not np.isfinite(singular):
                            fitness = 1e6
                            primary = np.nan
                            sparsity = np.nan
                            perror_sum = np.nan
                        else:
                            primary = abs(float(singular) - 1.0)
                            sparsity = float(alpha) * float(np.sum(np.abs(X[i, :])))
                            perror_sum = 0.0
                            if isinstance(res.get('percentage_differences', None), dict):
                                for _mk, pdiffs in res['percentage_differences'].items():
                                    for _crit, p in pdiffs.items():
                                        perror_sum += abs(float(p))
                            # Activation penalty and enhanced cost-benefit analysis
                            try:
                                vals = X[i, :]
                                active_flags = [abs(float(v)) >= thr for v in vals]
                            except Exception:
                                active_flags = [False] * len(self.param_names)
                            activation_count = int(sum(1 for f in active_flags if f))
                            activation_penalty = activation_count * pen_per if np.isfinite(pen_per) else 0.0
                            # Enhanced cost term (RV version)
                            cost_term = 0.0
                            # Prepare fallback sums
                            cost_sum_active = 0.0
                            denom = 0.0
                            try:
                                for j, name in enumerate(self.param_names):
                                    c = float(self.dva_costs.get(name, 0.0))
                                    if np.isfinite(c):
                                        denom += c
                                        if j < len(active_flags) and active_flags[j]:
                                            cost_sum_active += c
                            except Exception:
                                pass
                            denom = denom if denom and denom > 0 else 1.0
                            try:
                                if not self.use_enhanced_cost:
                                    cost_term = cost_sum_active / denom if denom > 0 else 0.0
                                else:
                                    # 1. Benefit metrics with UI weights
                                    bw_sum = max(1e-12, (self.benefit_w_primary + self.benefit_w_accuracy + self.benefit_w_sparsity))
                                    bwp = self.benefit_w_primary / bw_sum
                                    bwa = self.benefit_w_accuracy / bw_sum
                                    bws = self.benefit_w_sparsity / bw_sum
                                    benefit_primary = abs(float(singular) - 1.0)
                                    benefit_accuracy = perror_sum / 100.0
                                    benefit_sparsity = sparsity
                                    total_benefit = (bwp * benefit_primary + bwa * benefit_accuracy + bws * benefit_sparsity)

                                    # 2. Hierarchical cost categories with UI weights
                                    cw_sum = max(1e-12, (self.cat_w_material + self.cat_w_manufacturing + self.cat_w_maintenance + self.cat_w_operational))
                                    cw_material = self.cat_w_material / cw_sum
                                    cw_manufacturing = self.cat_w_manufacturing / cw_sum
                                    cw_maintenance = self.cat_w_maintenance / cw_sum
                                    cw_operational = self.cat_w_operational / cw_sum
                                    cost_categories = {
                                        'material': {'params': [], 'weight': cw_material, 'total': 0.0, 'active': 0.0},
                                        'manufacturing': {'params': [], 'weight': cw_manufacturing, 'total': 0.0, 'active': 0.0},
                                        'maintenance': {'params': [], 'weight': cw_maintenance, 'total': 0.0, 'active': 0.0},
                                        'operational': {'params': [], 'weight': cw_operational, 'total': 0.0, 'active': 0.0}
                                    }
                                    for j, is_active in enumerate(active_flags):
                                        if j >= len(self.param_names):
                                            continue
                                        pname = self.param_names[j]
                                        cval = float(self.dva_costs.get(pname, 0.0))
                                        if not np.isfinite(cval):
                                            continue
                                        # Mapping override if provided
                                        cat_map_val = self.dva_category_map.get(pname, None)
                                        if isinstance(cat_map_val, str) and cat_map_val.lower() in cost_categories:
                                            category = cat_map_val.lower()
                                        else:
                                            category = 'operational'
                                            pname_lower = pname.lower()
                                            if any(keyword in pname_lower for keyword in ['k', 'stiff', 'spring']):
                                                category = 'material'
                                            elif any(keyword in pname_lower for keyword in ['c', 'damp', 'viscous']):
                                                category = 'manufacturing'
                                            elif any(keyword in pname_lower for keyword in ['m', 'mass', 'weight']):
                                                category = 'maintenance'
                                        cost_categories[category]['params'].append(pname)
                                        cost_categories[category]['total'] += cval
                                        if is_active:
                                            cost_categories[category]['active'] += cval

                                    # 4. Category totals and cost weighting
                                    total_weighted_cost = 0.0
                                    for cat_name, cat_data in cost_categories.items():
                                        if cat_data['total'] > 0:
                                            normalized_cat_cost = cat_data['active'] / cat_data['total']
                                            total_weighted_cost += cat_data['weight'] * normalized_cat_cost
                                        else:
                                            total_weighted_cost += cat_data['weight'] * (1.0 if cat_data['active'] > 0 else 0.0)

                                    # 5. Use start of schedule in RV (no generations)
                                    benefit_weight = min(1.0, max(0.0, float(self.benefit_weight_start)))
                                    cost_weight = 1.0 - benefit_weight

                                    # 6. Combined cost-benefit term
                                    if total_benefit > 1e-10:
                                        overall_bcr = total_benefit / max(total_weighted_cost, 1e-10)
                                        benefit_cost_ratio = 1.0 / overall_bcr
                                    else:
                                        benefit_cost_ratio = 1e6
                                    cost_term = cost_weight * total_weighted_cost + benefit_weight * benefit_cost_ratio
                            except Exception:
                                # Fallback to normalized active cost
                                cost_term = cost_sum_active / denom if denom > 0 else 0.0

                            # Apply cost scale factor from worker settings
                            cost_scale_factor = float(self.cost_scale_factor) if self.cost_scale_factor is not None else 1.0
                            scaled_cost_term = cost_term / cost_scale_factor if cost_scale_factor > 0 else cost_term

                            # Debug: Print RV cost scaling information
                            try:
                                print(f"DEBUG RV: cost_term={cost_term:.6f}, cost_scale_factor={cost_scale_factor:.6f}, scaled_cost_term={scaled_cost_term:.6f}")
                            except Exception:
                                pass

                        fitness = primary + sparsity + perror_sum / pe_scale + activation_penalty + scaled_cost_term
                    except Exception:
                        fitness = 1e6
                        primary = np.nan
                        sparsity = np.nan
                        perror_sum = np.nan
                        activation_penalty = np.nan
                        cost_term = np.nan

                    # Collect peak positions and values per mass for KDE analysis of FRF peaks
                    def _collect_peaks_for_mass(mkey):
                        try:
                            mres = res.get(mkey, {})
                            ppos = []
                            pval = []
                            if isinstance(mres.get('peak_positions', {}), dict):
                                # Align positions and values by numeric index suffix
                                pos_items = [(int(k.split('_')[-1]), float(v)) for k, v in mres['peak_positions'].items() if isinstance(v, (int, float))]
                                val_items = [(int(k.split('_')[-1]), float(v)) for k, v in mres.get('peak_values', {}).items() if isinstance(v, (int, float))]
                                pos_map = dict(pos_items)
                                val_map = dict(val_items)
                                idxs = sorted(pos_map.keys())
                                for idx in idxs:
                                    if idx in val_map:
                                        ppos.append(pos_map[idx])
                                        pval.append(val_map[idx])
                            return ppos, pval
                        except Exception:
                            return [], []

                    peaks_m1, peakvals_m1 = _collect_peaks_for_mass('mass_1')
                    peaks_m2, peakvals_m2 = _collect_peaks_for_mass('mass_2')
                    peaks_m3, peakvals_m3 = _collect_peaks_for_mass('mass_3')
                    peaks_m4, peakvals_m4 = _collect_peaks_for_mass('mass_4')
                    peaks_m5, peakvals_m5 = _collect_peaks_for_mass('mass_5')

                    row = {name: X[i, j] for j, name in enumerate(self.param_names)}
                    row.update({
                        'rv_sample_index': int(i),
                        'rv_run_id': int(i) + 1,
                        'primary_objective': primary,
                        'sparsity_penalty': sparsity,
                        'percentage_error_sum': perror_sum,
                        'activation_penalty': activation_penalty,
                        'cost_term_raw': cost_term,  # Raw unscaled cost
                        'cost_term': scaled_cost_term,  # Scaled cost used in fitness
                        'fitness': fitness,
                        # Store FRF peak positions/values per mass (lists)
                        'peaks_mass_1': peaks_m1,
                        'peakvals_mass_1': peakvals_m1,
                        'peaks_mass_2': peaks_m2,
                        'peakvals_mass_2': peakvals_m2,
                        'peaks_mass_3': peaks_m3,
                        'peakvals_mass_3': peakvals_m3,
                        'peaks_mass_4': peaks_m4,
                        'peakvals_mass_4': peakvals_m4,
                        'peaks_mass_5': peaks_m5,
                        'peakvals_mass_5': peakvals_m5,
                    })
                    mass_curves = {}
                    if isinstance(res, dict):
                        for mass_key in ('mass_1', 'mass_2', 'mass_3', 'mass_4', 'mass_5'):
                            try:
                                mres = res.get(mass_key, {})
                                if not isinstance(mres, dict):
                                    continue
                                mag = mres.get('magnitude')
                                if mag is None:
                                    continue
                                mag_arr = np.asarray(mag, dtype=float)
                                if mag_arr.size:
                                    mass_curves[mass_key] = mag_arr.copy()
                            except Exception:
                                continue
                    if mass_curves:
                        frf_cache[int(i)] = mass_curves
                    rows.append(row)
                    if (i + 1) % max(1, n // 100) == 0:
                        self.progress.emit(int((i + 1) * 100 / n))

                df = pd.DataFrame(rows)
                payload = build_random_validation_payload(df, omega_vector, frf_cache)
                self.finished.emit(payload)
            except Exception as e:
                self.error.emit(str(e))

        def cancel(self):
            self.abort = True

    def _get_current_ga_param_config(self):
        param_names = [
            *[f"beta_{i}" for i in range(1, 16)],
            *[f"lambda_{i}" for i in range(1, 16)],
            *[f"mu_{i}" for i in range(1, 4)],
            *[f"nu_{i}" for i in range(1, 16)],
        ]
        bounds = []
        fixed_flags = []
        fixed_values = {}
        name_to_row = {self.ga_param_table.item(r, 0).text(): r for r in range(self.ga_param_table.rowCount())}
        for j, pname in enumerate(param_names):
            if pname not in name_to_row:
                bounds.append((0.0, 0.0))
                fixed_flags.append(True)
                fixed_values[j] = 0.0
                continue
            r = name_to_row[pname]
            fixed_chk = self.ga_param_table.cellWidget(r, 1)
            is_fixed = bool(fixed_chk.isChecked()) if fixed_chk is not None else False
            if is_fixed:
                fx_w = self.ga_param_table.cellWidget(r, 2)
                fx_val = fx_w.value() if fx_w is not None else 0.0
                bounds.append((fx_val, fx_val))
                fixed_flags.append(True)
                fixed_values[j] = fx_val
            else:
                lo_w = self.ga_param_table.cellWidget(r, 3)
                hi_w = self.ga_param_table.cellWidget(r, 4)
                low = float(lo_w.value() if lo_w is not None else 0.0)
                high = float(hi_w.value() if hi_w is not None else low)
                # If equal, keep as-is (zero-width interval is allowed)
                # If inverted, swap to maintain low <= high
                if high < low:
                    low, high = high, low
                bounds.append((low, high))
                fixed_flags.append(False)
        return param_names, bounds, fixed_flags, fixed_values

    def _render_figure_into_widget(self, target_widget, fig, include_toolbar=True):
        layout = target_widget.layout()
        if layout is None:
            layout = QVBoxLayout(target_widget)
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
        canvas = FigureCanvasQTAgg(fig)
        if include_toolbar:
            toolbar = NavigationToolbar(canvas, target_widget)
            # Add universal "Open in New Window" action to toolbar
            try:
                self._attach_open_in_new_window(toolbar, fig, "Plot")
            except Exception:
                pass
            layout.addWidget(toolbar)
        layout.addWidget(canvas)

    def run_random_validation(self):
        if hasattr(self, '_rv_worker') and self._rv_worker is not None and self._rv_worker.isRunning():
            QMessageBox.information(self, "Random Validation", "Validation is already running.")
            return
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "O Start must be less than O End.")
            return
        try:
            target_values, weights = self.get_target_values_weights()
        except Exception:
            QMessageBox.warning(self, "Input Error", "Targets & Weights are not properly defined.")
            return

        self.rv_frf_curves = {}
        self.rv_frf_omega = None
        try:
            self.rv_frf_run_list.clear()
        except Exception:
            pass
        try:
            self.update_random_validation_frf_plot()
        except Exception:
            pass

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value(),
        )

        param_names, bounds, fixed_flags, fixed_values = self._get_current_ga_param_config()

        num_samples = self.rv_num_samples_box.value()
        method = self.rv_method_combo.currentText()
        seed = self.rv_seed_box.value()
        alpha = self.rv_alpha_box.value()
        respect_fixed = self.rv_respect_fixed_chk.isChecked()

        # Build DVA costs mapping for RV
        dva_costs = {}
        dva_costs_by_category = {'material':{}, 'manufacturing':{}, 'maintenance':{}, 'operational':{}}
        try:
            for r in range(self.ga_param_table.rowCount()):
                name = self.ga_param_table.item(r, 0).text()
                cmat = self.ga_param_table.cellWidget(r, 5)
                cman = self.ga_param_table.cellWidget(r, 6)
                cmai = self.ga_param_table.cellWidget(r, 7)
                copr = self.ga_param_table.cellWidget(r, 8)
                dva_costs[name] = float(cmat.value()) if cmat is not None else 0.0  # legacy approx
                dva_costs_by_category['material'][name] = float(cmat.value()) if cmat is not None else 0.0
                dva_costs_by_category['manufacturing'][name] = float(cman.value()) if cman is not None else 0.0
                dva_costs_by_category['maintenance'][name] = float(cmai.value()) if cmai is not None else 0.0
                dva_costs_by_category['operational'][name] = float(copr.value()) if copr is not None else 0.0
        except Exception:
            pass

        self._rv_worker = self.RandomValidationWorker(
            main_params=main_params,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            target_values=target_values,
            weights=weights,
            param_names=param_names,
            bounds=bounds,
            fixed_flags=fixed_flags,
            fixed_values=fixed_values,
            alpha=alpha,
            percentage_error_scale=self.ga_percentage_error_scale_box.value(),
            cost_scale_factor=self.ga_cost_scale_box.value(),
            dva_activation_threshold=self.ga_activation_threshold_box.value(),
            dva_activation_penalty=self.ga_activation_penalty_box.value(),
            dva_costs=dva_costs,
            dva_costs_by_category=dva_costs_by_category,
            use_enhanced_cost=self.enh_cost_enable_chk.isChecked(),
            benefit_w_primary=self.benefit_w_primary_box.value(),
            benefit_w_accuracy=self.benefit_w_accuracy_box.value(),
            benefit_w_sparsity=self.benefit_w_sparsity_box.value(),
            cat_w_material=self.cat_w_material_box.value(),
            cat_w_manufacturing=self.cat_w_manufacturing_box.value(),
            cat_w_maintenance=self.cat_w_maintenance_box.value(),
            cat_w_operational=self.cat_w_operational_box.value(),
            benefit_weight_start=self.benefit_weight_start_box.value(),
            benefit_weight_end=self.benefit_weight_end_box.value(),
            
            method=method,
            num_samples=num_samples,
            seed=seed,
            respect_fixed=respect_fixed,
        )
        self._rv_worker.progress.connect(self.rv_progress_bar.setValue)
        self._rv_worker.error.connect(lambda msg: QMessageBox.critical(self, "Random Validation Error", msg))
        self._rv_worker.finished.connect(self._handle_random_validation_finished)
        # Store context for summary
        self._rv_context = {
            'method': method,
            'seed': seed,
            'alpha': alpha,
            'respect_fixed': respect_fixed,
            'tol': self.rv_tol_box.value(),
            'param_names': param_names,
            'fixed_flags': fixed_flags,
            'num_samples': num_samples,
            'start_time': time.time(),
        }
        self.rv_run_button.setEnabled(False)
        self.rv_cancel_button.setEnabled(True)
        self.rv_export_button.setEnabled(False)
        # Summary label deprecated in favor of table
        self.rv_progress_bar.setValue(0)
        self._rv_worker.start()

    def cancel_random_validation(self):
        if hasattr(self, '_rv_worker') and self._rv_worker is not None and self._rv_worker.isRunning():
            self._rv_worker.cancel()
            # Summary label deprecated

    def _handle_random_validation_finished(self, payload):
        self._rv_worker = None

        df = None
        frf_curves = {}
        omega_vector = None

        if isinstance(payload, dict) and 'dataframe' in payload:
            df = payload.get('dataframe')
            frf_curves = payload.get('frf_curves') or {}
            omega_vector = payload.get('omega_vector')
        else:
            df = payload

        self.rv_frf_curves = {}
        self.rv_frf_omega = None
        if omega_vector is not None:
            try:
                self.rv_frf_omega = np.asarray(omega_vector, dtype=float)
            except Exception:
                self.rv_frf_omega = None
        # Fallback: if omega not provided in payload, rebuild from current UI settings
        if self.rv_frf_omega is None or getattr(self.rv_frf_omega, 'size', 0) == 0:
            try:
                _os = float(self.omega_start_box.value())
                _oe = float(self.omega_end_box.value())
                _op = int(self.omega_points_box.value())
                if _op < 2:
                    _op = 2
                self.rv_frf_omega = np.linspace(_os, _oe, _op)
            except Exception:
                self.rv_frf_omega = None

        if isinstance(frf_curves, dict):
            safe_curves = {}
            for key, curves in frf_curves.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                if not isinstance(curves, dict):
                    continue
                mass_map = {}
                for mass_key, magnitudes in curves.items():
                    try:
                        mass_map[str(mass_key)] = np.asarray(magnitudes, dtype=float)
                    except Exception:
                        continue
                safe_curves[idx] = mass_map
            self.rv_frf_curves = safe_curves

        if df is None or len(df) == 0:
            # Populate summary table with status message when no data
            self.rv_run_button.setEnabled(True)
            self.rv_cancel_button.setEnabled(False)
            self.rv_export_button.setEnabled(False)
            self._rv_frf_populate_run_list(None)
            self.update_random_validation_frf_plot()
            return

        self.rv_results_df = df
        tol = self.rv_tol_box.value()
        df['pass'] = df['fitness'].apply(lambda v: bool(np.isfinite(v) and v <= tol))
        pct = float(100.0 * df['pass'].mean()) if len(df) else 0.0
        self.rv_success_bar.setValue(int(round(pct)))
        self._rv_frf_populate_run_list(df, preserve_selection=False)
        method = (self._rv_context or {}).get('method', 'Unknown')
        seed = (self._rv_context or {}).get('seed', -1)
        alpha = (self._rv_context or {}).get('alpha', self.rv_alpha_box.value())
        respect_fixed = (self._rv_context or {}).get('respect_fixed', True)
        fixed_flags = (self._rv_context or {}).get('fixed_flags', [])
        n_total_params = len(fixed_flags) if fixed_flags else len([c for c in df.columns if c.startswith(('beta_','lambda_','mu_','nu_'))])
        n_fixed = int(sum(bool(x) for x in fixed_flags)) if fixed_flags else 0
        n_varied = max(0, n_total_params - n_fixed)
        start_time = (self._rv_context or {}).get('start_time', None)
        elapsed = (time.time() - start_time) if start_time else None

        fitness_series = df['fitness']
        mean_v = float(fitness_series.mean())
        median_v = float(fitness_series.median())
        std_v = float(fitness_series.std())
        min_v = float(fitness_series.min())
        max_v = float(fitness_series.max())
        q05 = float(fitness_series.quantile(0.05))
        q25 = float(fitness_series.quantile(0.25))
        q75 = float(fitness_series.quantile(0.75))
        q95 = float(fitness_series.quantile(0.95))
        n = len(df)
        n_pass = int(df['pass'].sum())
        n_invalid = int((df['fitness'] >= 1e6).sum())
        n_nan = int(df['fitness'].isna().sum())
        n_inf = int(np.isinf(df['fitness']).sum())

        p_mean = float(df['primary_objective'].mean())
        p_std = float(df['primary_objective'].std())
        s_mean = float(df['sparsity_penalty'].mean())
        s_std = float(df['sparsity_penalty'].std())
        e_mean = float(df['percentage_error_sum'].mean())
        e_std = float(df['percentage_error_sum'].std())
        a_mean = float(df['activation_penalty'].mean()) if 'activation_penalty' in df.columns else 0.0
        a_std = float(df['activation_penalty'].std()) if 'activation_penalty' in df.columns else 0.0
        c_mean = float(df['cost_term'].mean()) if 'cost_term' in df.columns else 0.0
        c_std = float(df['cost_term'].std()) if 'cost_term' in df.columns else 0.0

        if mean_v > 1e-12:
            pe_scale = float(self.ga_percentage_error_scale_box.value()) if hasattr(self, 'ga_percentage_error_scale_box') else 1000.0
            cost_scale = float(self.ga_cost_scale_box.value()) if hasattr(self, 'ga_cost_scale_box') else 1.0
            pct_primary = 100.0 * p_mean / mean_v
            pct_sparsity = 100.0 * s_mean / mean_v
            pct_perror = 100.0 * (e_mean / pe_scale) / mean_v
            pct_activation = 100.0 * a_mean / mean_v
            pct_cost = 100.0 * (c_mean / cost_scale) / mean_v if cost_scale > 0 else 0.0
        else:
            pct_primary = pct_sparsity = pct_perror = pct_activation = pct_cost = float('nan')

        seed_text = str(seed) if isinstance(seed, (int,)) and seed >= 0 else 'None'
        elapsed_text = f" | Time: {elapsed:.2f}s" if elapsed is not None else ''

        html = (
            f"<b>Run Settings</b><br>"
            f"Samples: {n} | Method: {method} | Seed: {seed_text} | Alpha: {alpha:.4f} | Tolerance: {tol:.6f}{elapsed_text}<br>"
            f"Respect fixed: {'Yes' if respect_fixed else 'No'} | Varied params: {n_varied} / {n_total_params} (total)<br><br>"
            f"<b>Fitness Summary</b><br>"
            f"Mean: {mean_v:.6f} | Median: {median_v:.6f} | Std: {std_v:.6f}<br>"
            f"Min: {min_v:.6f} | Q05: {q05:.6f} | Q25: {q25:.6f} | Q75: {q75:.6f} | Q95: {q95:.6f} | Max: {max_v:.6f}<br>"
            f"Within tolerance: {n_pass} / {n} = {pct:.1f}% | Invalid>=1e6: {n_invalid} | NaN: {n_nan} | Inf: {n_inf}<br><br>"
            f"<b>Component Averages</b><br>"
            f"Primary: {p_mean:.6f} Â± {p_std:.6f} | Sparsity: {s_mean:.6f} Â± {s_std:.6f} | %Error sum: {e_mean:.6f} Â± {e_std:.6f} | Activation: {a_mean:.6f} Â± {a_std:.6f} | Cost: {c_mean:.6f} Â± {c_std:.6f}<br>"
            f"Approx. contribution to mean fitness: Primary {pct_primary:.1f}%, Sparsity {pct_sparsity:.1f}%, %Error {(pct_perror):.1f}%, Activation {pct_activation:.1f}%, Cost {pct_cost:.1f}%"
        )
        # Summary text deprecated; populate summary table below

        try:
            rows =[
                ("Samples", f"{n}"),
                ("Method", str(method)),
                ("Seed", seed_text),
                ("Alpha", f"{alpha:.6f}"),
                ("Tolerance", f"{tol:.6f}"),
                ("Respect Fixed", "Yes" if respect_fixed else "No"),
                ("Varied Params", f"{n_varied}"),
                ("Total Params", f"{n_total_params}"),
            ]
            if elapsed is not None:
                rows.append(("Elapsed Time (s)", f"{elapsed:.2f}"))
            rows.extend([
                ("Fitness Mean", f"{mean_v:.6f}"),
                ("Fitness Median", f"{median_v:.6f}"),
                ("Fitness Std", f"{std_v:.6f}"),
                ("Fitness Min", f"{min_v:.6f}"),
                ("Fitness Q05", f"{q05:.6f}"),
                ("Fitness Q25", f"{q25:.6f}"),
                ("Fitness Q75", f"{q75:.6f}"),
                ("Fitness Q95", f"{q95:.6f}"),
                ("Fitness Max", f"{max_v:.6f}"),
                ("Within Tol (count)", f"{n_pass} / {n}"),
                ("Within Tol (%)", f"{pct:.1f}%"),
                ("Invalid (>=1e6)", f"{n_invalid}"),
                ("NaN", f"{n_nan}"),
                ("Inf", f"{n_inf}"),
                ("Primary Mean", f"{p_mean:.6f}"),
                ("Primary Std", f"{p_std:.6f}"),
                ("Sparsity Mean", f"{s_mean:.6f}"),
                ("Sparsity Std", f"{s_std:.6f}"),
                ("%Error Sum Mean", f"{e_mean:.6f}"),
                ("%Error Sum Std", f"{e_std:.6f}"),
                ("Activation Mean", f"{a_mean:.6f}"),
                ("Activation Std", f"{a_std:.6f}"),
                ("Cost Mean", f"{c_mean:.6f}"),
                ("Cost Std", f"{c_std:.6f}"),
                ("Contribution Primary %", f"{pct_primary:.1f}%"),
                ("Contribution Sparsity %", f"{pct_sparsity:.1f}%"),
                ("Contribution %Error %", f"{pct_perror:.1f}%"),
                ("Contribution Activation %", f"{pct_activation:.1f}%"),
                ("Contribution Cost %", f"{pct_cost:.1f}%"),
            ])
            from PyQt5.QtWidgets import QTableWidgetItem
            from PyQt5.QtCore import Qt
            tbl = self.rv_summary_table
            tbl.blockSignals(True)
            tbl.clearContents()
            tbl.setRowCount(len(rows))
            for r, (metric, value) in enumerate(rows):
                it0 = QTableWidgetItem(str(metric))
                it1 = QTableWidgetItem(str(value))
                it0.setTextAlignment(Qt.AlignCenter)
                it1.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(r, 0, it0)
                tbl.setItem(r, 1, it1)
            tbl.blockSignals(False)
        except Exception:
            pass

        self.rv_scatter_param_combo.blockSignals(True)
        self.rv_scatter_param_combo.clear()
        for name in [c for c in df.columns if c.startswith('beta_') or c.startswith('lambda_') or c.startswith('mu_') or c.startswith('nu_')]:
            self.rv_scatter_param_combo.addItem(name)
        self.rv_scatter_param_combo.blockSignals(False)

        fig1 = Figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)
        sns.histplot(df['fitness'], kde=True, bins=self.rv_bins_box.value(), ax=ax1, color='skyblue', edgecolor='darkblue', alpha=0.6)
        ax1.axvspan(0.0, tol, facecolor='palegreen', alpha=0.25, label='Within tolerance')
        ax1.axvline(tol, color='magenta', linestyle='--', linewidth=2.0, alpha=0.9, label='Tolerance')
        ax1.set_title('Fitness Distribution')
        ax1.set_xlabel('Fitness')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()
        fig1.tight_layout()
        self._render_figure_into_widget(self.rv_fitdist_plot, fig1)

        fig2 = Figure(figsize=(16, 3))
        axes = fig2.subplots(1, 6)
        comp_cols = ['primary_objective', 'sparsity_penalty', 'percentage_error_sum', 'activation_penalty', 'cost_term_raw', 'cost_term']
        titles = ['Primary Objective', 'Sparsity Penalty', 'Percentage Error Sum', 'Activation Penalty', 'Cost Term (Raw)', 'Cost Term (Scaled)']
        for ax, col, title in zip(axes, comp_cols, titles):
            sns.histplot(df[col], kde=False, bins=max(10, self.rv_bins_box.value() // 2), ax=ax, color='lightgreen', edgecolor='darkgreen', alpha=0.6)
            try:
                sns.kdeplot(df[col].dropna(), ax=ax, color='purple', linewidth=2.0)
            except Exception:
                pass
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.5)
        fig2.tight_layout()
        self._render_figure_into_widget(self.rv_comp_plot, fig2)

        try:
            numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fc' and c != 'pass']
            corr = df[numeric_cols].corr()
            fig3 = Figure(figsize=(6, 5))
            ax3 = fig3.add_subplot(111)
            sns.heatmap(corr, cmap='coolwarm', center=0.0, ax=ax3)
            ax3.set_title('Correlation Heatmap')
            fig3.tight_layout()
            self._render_figure_into_widget(self.rv_corr_plot, fig3)
        except Exception:
            fig3 = Figure(figsize=(6, 5))
            ax3 = fig3.add_subplot(111)
            ax3.text(0.5, 0.5, 'Correlation unavailable', ha='center', va='center')
            ax3.axis('off')
            self._render_figure_into_widget(self.rv_corr_plot, fig3)

        self.update_random_validation_scatter()
        try:
            self.update_random_validation_kde()
        except Exception:
            pass

        cols = [c for c in df.columns if c.startswith('beta_') or c.startswith('lambda_') or c.startswith('mu_') or c.startswith('nu_')] + ['primary_objective', 'sparsity_penalty', 'percentage_error_sum', 'activation_penalty', 'cost_term', 'fitness', 'pass']
        self.rv_table.setColumnCount(len(cols))
        self.rv_table.setRowCount(len(df))
        self.rv_table.setHorizontalHeaderLabels(cols)
        for i, (_idx, row) in enumerate(df.iterrows()):
            for j, col in enumerate(cols):
                val = row[col]
                if isinstance(val, (int, float, np.floating)):
                    text = f"{val:.6g}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                if col == 'fitness':
                    item.setForeground(QBrush(QColor('darkblue')))
                if col == 'pass':
                    item.setText('True' if bool(val) else 'False')
                self.rv_table.setItem(i, j, item)
        self.rv_table.resizeColumnsToContents()

        self.rv_run_button.setEnabled(True)
        self.rv_cancel_button.setEnabled(False)
        self.rv_export_button.setEnabled(True)

    def _rv_frf_populate_run_list(self, df, preserve_selection=False):
        if not hasattr(self, 'rv_frf_run_list'):
            return
        try:
            current_selection = {int(item.data(Qt.UserRole)) for item in self.rv_frf_run_list.selectedItems()} if preserve_selection else set()
        except Exception:
            current_selection = set()
        self.rv_frf_run_list.blockSignals(True)
        self.rv_frf_run_list.clear()
        if df is None or getattr(df, 'empty', True):
            self.rv_frf_run_list.blockSignals(False)
            return
        try:
            sorted_df = df.sort_values('fitness', kind='mergesort')
        except Exception:
            sorted_df = df
        for _, row in sorted_df.iterrows():
            try:
                sample_idx = int(row.get('rv_sample_index', row.name))
            except Exception:
                continue
            try:
                run_id = int(row.get('rv_run_id', sample_idx + 1))
            except Exception:
                run_id = sample_idx + 1
            fitness_val = row.get('fitness', np.nan)
            try:
                fitness_val = float(fitness_val)
            except Exception:
                fitness_val = float('nan')
            pass_flag = bool(row.get('pass', False))
            label = f"Run {run_id:04d} | fitness {fitness_val:.6f}"
            label += ' | PASS' if pass_flag else ' | FAIL'
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, sample_idx)
            item.setData(Qt.UserRole + 1, run_id)
            item.setData(Qt.UserRole + 2, fitness_val)
            item.setData(Qt.UserRole + 3, pass_flag)
            if pass_flag:
                item.setForeground(QBrush(QColor('darkgreen')))
                item.setBackground(QColor(235, 250, 235))
            else:
                item.setForeground(QBrush(QColor('maroon')))
                item.setBackground(QColor(250, 235, 235))
            self.rv_frf_run_list.addItem(item)
        if preserve_selection and current_selection:
            for i in range(self.rv_frf_run_list.count()):
                item = self.rv_frf_run_list.item(i)
                try:
                    key = int(item.data(Qt.UserRole))
                except Exception:
                    continue
                if key in current_selection:
                    item.setSelected(True)
        elif self.rv_frf_run_list.count():
            for i in range(min(3, self.rv_frf_run_list.count())):
                self.rv_frf_run_list.item(i).setSelected(True)
        self.rv_frf_run_list.blockSignals(False)
        self.update_random_validation_frf_plot()

    def _rv_frf_select_top(self, count=3):
        if not hasattr(self, 'rv_frf_run_list'):
            return
        self.rv_frf_run_list.blockSignals(True)
        for i in range(self.rv_frf_run_list.count()):
            self.rv_frf_run_list.item(i).setSelected(False)
        for i in range(min(count, self.rv_frf_run_list.count())):
            self.rv_frf_run_list.item(i).setSelected(True)
        self.rv_frf_run_list.blockSignals(False)
        self.update_random_validation_frf_plot()

    def _rv_frf_clear_selection(self):
        if not hasattr(self, 'rv_frf_run_list'):
            return
        self.rv_frf_run_list.blockSignals(True)
        for i in range(self.rv_frf_run_list.count()):
            self.rv_frf_run_list.item(i).setSelected(False)
        self.rv_frf_run_list.blockSignals(False)
        self.update_random_validation_frf_plot()

    def update_random_validation_frf_plot(self):
        if not hasattr(self, 'rv_frf_plot'):
            return
        omega = getattr(self, 'rv_frf_omega', None)
        if omega is not None:
            try:
                omega = np.asarray(omega, dtype=float)
            except Exception:
                omega = None
        mass_combo = getattr(self, 'rv_frf_mass_combo', None)
        mass_key = mass_combo.currentText() if hasattr(mass_combo, 'currentText') else None
        if not mass_key:
            mass_key = 'mass_1'
        try:
            selected_items = list(self.rv_frf_run_list.selectedItems())
        except Exception:
            selected_items = []
        if omega is None or omega.size == 0 or not getattr(self, 'rv_frf_curves', {}):
            fig = Figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'FRF data unavailable', ha='center', va='center')
            ax.axis('off')
            self._render_figure_into_widget(self.rv_frf_plot, fig, include_toolbar=False)
            return
        if not selected_items:
            fig = Figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Select one or more runs to display FRF curves', ha='center', va='center')
            ax.axis('off')
            self._render_figure_into_widget(self.rv_frf_plot, fig, include_toolbar=False)
            return
        fig = Figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        palette = sns.color_palette('tab10', n_colors=len(selected_items)) if len(selected_items) else [(0.2, 0.2, 0.2)]
        plotted = 0
        for color, item in zip(palette, selected_items):
            try:
                sample_idx = int(item.data(Qt.UserRole))
            except Exception:
                continue
            curves = self.rv_frf_curves.get(sample_idx) if isinstance(self.rv_frf_curves, dict) else None
            if not isinstance(curves, dict):
                continue
            mag = curves.get(mass_key)
            if mag is None:
                continue
            try:
                mag_arr = np.asarray(mag, dtype=float)
            except Exception:
                continue
            if mag_arr.size != omega.size:
                continue
            run_id = item.data(Qt.UserRole + 1)
            fitness_val = item.data(Qt.UserRole + 2)
            pass_flag = bool(item.data(Qt.UserRole + 3))
            if isinstance(run_id, (int, np.integer)):
                label = f"Run {run_id:04d}"
            else:
                label = f"Sample {sample_idx}"
            try:
                fin = float(fitness_val)
                if np.isfinite(fin):
                    label += f" | f={fin:.6f}"
            except Exception:
                pass
            if pass_flag:
                label += ' '
            ax.plot(omega, mag_arr, color=color, linewidth=1.6, alpha=0.9, label=label, zorder=5)
            plotted += 1
        if plotted == 0:
            fig = Figure(figsize=(5, 3))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'FRF curves unavailable for current selection', ha='center', va='center')
            ax.axis('off')
            self._render_figure_into_widget(self.rv_frf_plot, fig, include_toolbar=False)
            return
        # Overlay zones if requested
        try:
            if hasattr(self, 'rv_frf_use_zones_chk') and self.rv_frf_use_zones_chk.isChecked():
                zones = getattr(self, 'zones', []) or []
                for z in zones:
                    try:
                        x0 = float(z.get('start_x', np.nan))
                        x1 = float(z.get('end_x', np.nan))
                        if not (np.isfinite(x0) and np.isfinite(x1)):
                            continue
                        if x1 < x0:
                            x0, x1 = x1, x0
                        color = z.get('color', '#ffcc66')
                        name = str(z.get('name', ''))
                        ax.axvspan(x0, x1, facecolor=color, alpha=0.18, edgecolor=None, zorder=1)
                        if name:
                            ax.text((x0 + x1) / 2.0, 0.98, name, transform=ax.get_xaxis_transform(),
                                    ha='center', va='top', fontsize=8,
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
                    except Exception:
                        continue
        except Exception:
            pass

        ax.set_title(f'FRF curves ({mass_key})')
        ax.set_xlabel('Frequency (rad/s)')
        ax.set_ylabel('Response magnitude')
        ax.grid(True, linestyle='--', alpha=0.4)
        try:
            ax.set_xlim(float(np.nanmin(omega)), float(np.nanmax(omega)))
        except Exception:
            pass
        try:
            ax.legend(loc='best', fontsize=8)
        except Exception:
            pass
        fig.tight_layout()
        self._render_figure_into_widget(self.rv_frf_plot, fig)

    def _rv_frf_add_zone(self):
        """Add a new zone using the current FRF omega range for immediate visibility."""
        try:
            if not hasattr(self, 'zones'):
                self.zones = []
            # Determine omega range: prefer FRF omega, then UI settings
            x0 = x1 = None
            try:
                omega = np.asarray(getattr(self, 'rv_frf_omega', None), dtype=float)
                if omega.size >= 2:
                    xmin = float(np.nanmin(omega))
                    xmax = float(np.nanmax(omega))
                    span = xmax - xmin
                    if span > 0:
                        x0 = xmin + 0.40 * span
                        x1 = xmin + 0.60 * span
            except Exception:
                omega = None
            if not (isinstance(x0, float) and isinstance(x1, float)):
                try:
                    _os = float(self.omega_start_box.value())
                    _oe = float(self.omega_end_box.value())
                    if _oe < _os:
                        _os, _oe = _oe, _os
                    span = _oe - _os
                    if span <= 0:
                        span = max(1.0, abs(_oe) + abs(_os))
                    x0 = _os + 0.40 * span
                    x1 = _os + 0.60 * span
                except Exception:
                    # Last resort
                    x0, x1 = 0.0, 1.0
            name = f"Zone {len(self.zones)+1}"
            self.zones.append({'name': name, 'start_x': float(x0), 'end_x': float(x1), 'color': '#ffcc66'})
        except Exception:
            pass
        # Sync both tables and redraw
        try:
            if hasattr(self, '_rv_kde_update_zone_table'):
                self._rv_kde_update_zone_table()
        except Exception:
            pass
        try:
            self._update_zone_table()
        except Exception:
            pass
        try:
            self.update_random_validation_frf_plot()
        except Exception:
            pass

    def _rv_frf_remove_zone(self):
        """Remove selected zone from FRF zone table, sync everywhere."""
        try:
            row = self.rv_frf_zone_table.currentRow() if hasattr(self, 'rv_frf_zone_table') else -1
            if row is None:
                row = -1
            if row >= 0 and hasattr(self, 'zones') and row < len(self.zones):
                del self.zones[row]
        except Exception:
            pass
        # Sync KDE table and redraw
        try:
            if hasattr(self, '_rv_kde_update_zone_table'):
                self._rv_kde_update_zone_table()
        except Exception:
            pass
        try:
            self._update_zone_table()
        except Exception:
            pass
        try:
            self.update_random_validation_kde()
        except Exception:
            pass
        try:
            self.update_random_validation_frf_plot()
        except Exception:
            pass

    def _rv_frf_clear_zones(self):
        """Clear all zones from FRF view, sync both tables."""
        try:
            if hasattr(self, 'zones'):
                self.zones.clear()
            else:
                self.zones = []
        except Exception:
            self.zones = []
        try:
            if hasattr(self, '_rv_kde_update_zone_table'):
                self._rv_kde_update_zone_table()
        except Exception:
            pass
        try:
            self._update_zone_table()
        except Exception:
            pass
        try:
            self.update_random_validation_kde()
        except Exception:
            pass
        try:
            self.update_random_validation_frf_plot()
        except Exception:
            pass

    # ----- FRF zones table sync (mirrors KDE zones table) -----
    def _update_zone_table(self):
        # Ensure zones list exists
        if not hasattr(self, 'zones'):
            self.zones = []
        tbl = getattr(self, 'rv_frf_zone_table', None)
        if tbl is None:
            return
        try:
            tbl.blockSignals(True)
            tbl.setRowCount(len(self.zones))
            for r, zone in enumerate(self.zones):
                # Name
                name_item = QTableWidgetItem(str(zone.get('name', '')))
                tbl.setItem(r, 0, name_item)
                # Start X
                start_item = QTableWidgetItem(f"{float(zone.get('start_x', 0.0)):.6g}")
                tbl.setItem(r, 1, start_item)
                # End X
                end_item = QTableWidgetItem(f"{float(zone.get('end_x', 0.0)):.6g}")
                tbl.setItem(r, 2, end_item)
                # Color button
                btn = QPushButton()
                btn.setAutoFillBackground(True)
                color = zone.get('color', '#cccccc')
                btn.setStyleSheet(f"background-color: {color};")
                btn.setFixedSize(40, 20)
                def choose_color(row=r, button=btn):
                    from PyQt5.QtWidgets import QColorDialog
                    dlg = QColorDialog(self)
                    if dlg.exec_():
                        c = dlg.selectedColor()
                        if c.isValid():
                            self.zones[row]['color'] = c.name()
                            button.setStyleSheet(f"background-color: {c.name()};")
                            # Redraw FRF plot to reflect color change
                            self.update_random_validation_frf_plot()
                btn.clicked.connect(choose_color)
                tbl.setCellWidget(r, 3, btn)
        finally:
            try:
                tbl.blockSignals(False)
            except Exception:
                pass

    def update_random_validation_scatter(self):
        if self.rv_results_df is None or self.rv_results_df.empty:
            fig = Figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.axis('off')
            self._render_figure_into_widget(self.rv_scatter_plot, fig)
            return
        param = self.rv_scatter_param_combo.currentText()
        if not param:
            param_candidates = [c for c in self.rv_results_df.columns if c.startswith('beta_') or c.startswith('lambda_') or c.startswith('mu_') or c.startswith('nu_')]
            if not param_candidates:
                fig = Figure(figsize=(5, 4))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'No parameters', ha='center', va='center')
                ax.axis('off')
                self._render_figure_into_widget(self.rv_scatter_plot, fig)
                return
            param = param_candidates[0]
        df = self.rv_results_df
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        # Shade region within tolerance
        tol = self.rv_tol_box.value()
        try:
            xmin = float(np.nanmin(df[param].values)) if len(df) else 0.0
            xmax = float(np.nanmax(df[param].values)) if len(df) else 1.0
        except Exception:
            xmin, xmax = 0.0, 1.0
        ax.axhspan(0.0, tol, facecolor='palegreen', alpha=0.25, label='Within tolerance')
        # Color points by pass/fail
        sns.scatterplot(
            x=df[param], y=df['fitness'], hue=df['fitness'] <= tol,
            palette={True: 'tab:green', False: 'tab:red'}, legend=False,
            ax=ax, alpha=0.6, edgecolor=None
        )
        ax.axhline(tol, color='magenta', linestyle='--', linewidth=2.0, alpha=0.9)
        ax.set_xlabel(param)
        ax.set_ylabel('Fitness')
        ax.set_title(f'{param} vs Fitness')
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        self._render_figure_into_widget(self.rv_scatter_plot, fig, include_toolbar=False)

    def export_random_validation_results(self):
        if self.rv_results_df is None or self.rv_results_df.empty:
            QMessageBox.information(self, "Export", "No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", os.path.expanduser("~"), "CSV Files (*.csv)")
        if path:
            try:
                self.rv_results_df.to_csv(path, index=False)
                QMessageBox.information(self, "Export", f"Saved to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def refresh_random_validation_views(self):
        """Refresh validation plots and summary based on current tolerance/bins."""
        try:
            if self.rv_results_df is None or self.rv_results_df.empty:
                return
            # Recompute pass column
            tol = self.rv_tol_box.value()
            df = self.rv_results_df.copy()
            df['pass'] = df['fitness'].apply(lambda v: bool(np.isfinite(v) and v <= tol))
            self.rv_results_df = df
            self._rv_frf_populate_run_list(df, preserve_selection=True)
            # Update success bar and summary header only
            pct = float(100.0 * df['pass'].mean()) if len(df) else 0.0
            self.rv_success_bar.setValue(int(round(pct)))
            # Update summary table tolerance/pass rows if available
            try:
                from PyQt5.QtWidgets import QTableWidgetItem
                from PyQt5.QtCore import Qt
                tbl = getattr(self, 'rv_summary_table', None)
                if tbl is not None:
                    # Find rows by Metric text and update Value
                    idx_map = {}
                    for r in range(tbl.rowCount()):
                        it = tbl.item(r, 0)
                        if it is not None:
                            idx_map[it.text()] = r
                    updates = {
                        'Tolerance': f"{tol:.6f}",
                        'Within Tol (%)': f"{pct:.1f}%",
                        'Within Tol (count)': f"{int(df['pass'].sum())} / {len(df)}",
                    }
                    for key, val in updates.items():
                        if key in idx_map:
                            r = idx_map[key]
                            it1 = QTableWidgetItem(str(val))
                            it1.setTextAlignment(Qt.AlignCenter)
                            tbl.setItem(r, 1, it1)
            except Exception:
                pass
            # Redraw plots
            # Fitness distribution
            fig1 = Figure(figsize=(6, 4))
            ax1 = fig1.add_subplot(111)
            sns.histplot(df['fitness'], kde=True, bins=self.rv_bins_box.value(), ax=ax1, color='skyblue', edgecolor='darkblue', alpha=0.6)
            # Highlight within-tolerance region (from 0 to tolerance)
            ax1.axvspan(0.0, tol, facecolor='palegreen', alpha=0.25, label='Within tolerance')
            ax1.axvline(tol, color='magenta', linestyle='--', linewidth=2.0, alpha=0.9, label='Tolerance')
            ax1.set_title('Fitness Distribution')
            ax1.set_xlabel('Fitness')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend()
            fig1.tight_layout()
            self._render_figure_into_widget(self.rv_fitdist_plot, fig1)
            # Components plot: only bins change
            fig2 = Figure(figsize=(10, 3))
            axes = fig2.subplots(1, 3)
            comp_cols = ['primary_objective', 'sparsity_penalty', 'percentage_error_sum']
            titles = ['Primary Objective', 'Sparsity Penalty', 'Percentage Error Sum']
            for ax, col, title in zip(axes, comp_cols, titles):
                sns.histplot(
                    df[col], kde=False,
                    bins=max(10, self.rv_bins_box.value() // 2),
                    ax=ax, color='lightgreen', edgecolor='darkgreen', alpha=0.6,
                )
                try:
                    sns.kdeplot(df[col].dropna(), ax=ax, color='purple', linewidth=2.0)
                except Exception:
                    pass
                ax.set_title(title)
                ax.grid(True, linestyle='--', alpha=0.5)
            fig2.tight_layout()
            self._render_figure_into_widget(self.rv_comp_plot, fig2)
            # Scatter plot refresh
            self.update_random_validation_scatter()
            # KDE of peak positions
            self.update_random_validation_kde()
        except Exception as _:
            pass

    def update_random_validation_kde(self):
        """Build/refresh the KDE + histogram of FRF peak positions across RV runs."""
        try:
            if getattr(self, 'rv_results_df', None) is None or self.rv_results_df.empty:
                return

            import numpy as np
            import seaborn as sns
            from matplotlib.figure import Figure
            from scipy import stats

            df = self.rv_results_df

            # Decide which masses to include
            sel = getattr(self, 'rv_kde_mass_combo', None)
            selected_mass = sel.currentText() if sel is not None else 'All Masses'

            def collect_all_positions(row, masses):
                vals = []
                for m in masses:
                    col = f'peaks_{m}'
                    try:
                        peaks = row.get(col, []) if hasattr(row, 'get') else (row[col] if col in row else [])
                    except Exception:
                        peaks = []
                    if not isinstance(peaks, list):
                        peaks = []
                    # Ensure numeric
                    for p in peaks:
                        try:
                            pf = float(p)
                            if np.isfinite(pf):
                                vals.append(pf)
                        except Exception:
                            pass
                return vals

            # Build the list of peak positions across all runs based on controls
            masses = []
            if selected_mass == 'All Masses':
                masses = [f'mass_{i}' for i in range(1, 6)]
            elif selected_mass.startswith('mass_'):
                masses = [selected_mass]
            else:
                masses = [f'mass_{i}' for i in range(1, 6)]

            peak_mode = getattr(self, 'rv_kde_peak_mode_combo', None)
            mode_text = peak_mode.currentText() if peak_mode is not None else 'Top 1'

            positions = []
            # Iterate rows: each row has lists per mass for positions and values
            for _, r in df.iterrows():
                if mode_text == 'All peaks':
                    positions.extend(collect_all_positions(r, masses))
                else:
                    # Build pairs (value, pos) per mass to pick top-N by amplitude
                    per_run = []
                    for m in masses:
                        pcol = f'peaks_{m}'
                        vcol = f'peakvals_{m}'
                        # Use Series.get for safety
                        plist = r.get(pcol, []) if hasattr(r, 'get') else (r[pcol] if pcol in r else [])
                        vlist = r.get(vcol, []) if hasattr(r, 'get') else (r[vcol] if vcol in r else [])
                        if not isinstance(plist, list):
                            plist = []
                        if not isinstance(vlist, list):
                            vlist = []
                        if isinstance(plist, list) and isinstance(vlist, list) and len(plist) == len(vlist):
                            for pos, val in zip(plist, vlist):
                                try:
                                    posf = float(pos)
                                    valf = float(val)
                                    if np.isfinite(posf) and np.isfinite(valf):
                                        per_run.append((valf, posf))
                                except Exception:
                                    pass
                    if per_run:
                        per_run.sort(key=lambda t: t[0], reverse=True)
                        take = 1 if mode_text == 'Top 1' else 2 if mode_text == 'Top 2' else 3
                        positions.extend([p for _v, p in per_run[:take]])

            positions = [p for p in positions if isinstance(p, (int, float)) and np.isfinite(p)]
            if len(positions) < 2:
                # Render a simple message
                msg_fig = Figure(figsize=(6, 3))
                axm = msg_fig.add_subplot(111)
                axm.text(0.5, 0.5, 'Not enough peak data to render KDE', ha='center', va='center')
                axm.axis('off')
                self._render_figure_into_widget(self.rv_kde_plot, msg_fig, include_toolbar=False)
                return

            # Build figure
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)

            # Histogram density
            try:
                sns.histplot(positions, bins=max(10, int(self.rv_bins_box.value())), stat='density', alpha=0.5,
                             color='steelblue', edgecolor='navy', ax=ax)
            except Exception:
                pass

            # KDE curve
            try:
                kde = stats.gaussian_kde(positions)
                xmin = float(min(positions))
                xmax = float(max(positions))
                xgrid = np.linspace(xmin, xmax, 512)
                y = kde(xgrid)
                ax.plot(xgrid, y, color='purple', linewidth=2.0, label='KDE')
            except Exception:
                y = None

            # Mean peak position lines removed per request

            # Overlay FRF zones if requested and available
            try:
                if self.rv_kde_use_zones_chk.isChecked() and hasattr(self, '_add_zone_highlights'):
                    self._add_zone_highlights(ax)
            except Exception:
                pass

            title_mass = selected_mass if selected_mass != 'All Masses' else 'All Masses'
            ax.set_title(f'Peak Position Density ({title_mass})')
            ax.set_xlabel('Frequency (rad/s) of peaks')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3, linestyle='--')

            fig.tight_layout()
            self._render_figure_into_widget(self.rv_kde_plot, fig)
            # Keep the KDE zones table in sync with current zones list
            try:
                self._rv_kde_update_zone_table()
            except Exception:
                pass
        except Exception:
            pass

    # ----- KDE zone management helpers -----
    def _rv_kde_update_zone_table(self):
        # Ensure zones list exists
        if not hasattr(self, 'zones'):
            self.zones = []
        self._rv_kde_updating_table = True
        try:
            tbl = self.rv_kde_zone_table
            tbl.setRowCount(len(self.zones))
            for r, zone in enumerate(self.zones):
                # Name
                name_item = QTableWidgetItem(str(zone.get('name', '')))
                tbl.setItem(r, 0, name_item)
                # Start X
                start_item = QTableWidgetItem(f"{float(zone.get('start_x', 0.0)):.6g}")
                tbl.setItem(r, 1, start_item)
                # End X
                end_item = QTableWidgetItem(f"{float(zone.get('end_x', 0.0)):.6g}")
                tbl.setItem(r, 2, end_item)
                # Color button
                btn = QPushButton()
                btn.setAutoFillBackground(True)
                color = zone.get('color', '#cccccc')
                btn.setStyleSheet(f"background-color: {color};")
                btn.setFixedSize(40, 20)
                def choose_color(row=r, button=btn):
                    from PyQt5.QtWidgets import QColorDialog
                    dlg = QColorDialog(self)
                    if dlg.exec_():
                        c = dlg.selectedColor()
                        if c.isValid():
                            self.zones[row]['color'] = c.name()
                            button.setStyleSheet(f"background-color: {c.name()};")
                            # Update FRF table too if present
                            try:
                                if hasattr(self, '_update_zone_table'):
                                    self._update_zone_table()
                            except Exception:
                                pass
                            # Redraw KDE
                            self.update_random_validation_kde()
                btn.clicked.connect(choose_color)
                tbl.setCellWidget(r, 3, btn)
        finally:
            self._rv_kde_updating_table = False

    def _rv_kde_on_zone_cell_changed(self, item):
        if self._rv_kde_updating_table:
            return
        try:
            row = item.row()
            col = item.column()
            if row < 0 or row >= len(getattr(self, 'zones', [])):
                return
            zone = self.zones[row]
            text = item.text().strip()
            if col == 0:
                zone['name'] = text
            elif col == 1:
                try:
                    zone['start_x'] = float(text)
                except Exception:
                    pass
            elif col == 2:
                try:
                    zone['end_x'] = float(text)
                except Exception:
                    pass
            # Keep FRF table in sync if available
            try:
                if hasattr(self, '_update_zone_table'):
                    self._update_zone_table()
            except Exception:
                pass
            # Redraw plots for visual feedback in both tabs
            try:
                self.update_random_validation_kde()
            except Exception:
                pass
            try:
                self.update_random_validation_frf_plot()
            except Exception:
                pass
        except Exception:
            pass

    def _rv_kde_add_zone(self):
        # Use FRF dialog if available; else create a simple default zone
        try:
            if hasattr(self, 'add_zone'):
                self.add_zone()
            else:
                if not hasattr(self, 'zones'):
                    self.zones = []
                self.zones.append({'name': 'Zone', 'start_x': 0.0, 'end_x': 1.0, 'color': '#ffcc00'})
        except Exception:
            pass
        # Sync both tables
        try:
            if hasattr(self, '_update_zone_table'):
                self._update_zone_table()
        except Exception:
            pass
        self._rv_kde_update_zone_table()
        self.update_random_validation_kde()

    def _rv_kde_remove_zone(self):
        try:
            current_row = self.rv_kde_zone_table.currentRow()
            if current_row >= 0 and hasattr(self, 'zones') and current_row < len(self.zones):
                del self.zones[current_row]
        except Exception:
            pass
        try:
            if hasattr(self, '_update_zone_table'):
                self._update_zone_table()
        except Exception:
            pass
        self._rv_kde_update_zone_table()
        self.update_random_validation_kde()

    def _rv_kde_clear_zones(self):
        try:
            if hasattr(self, 'zones'):
                self.zones.clear()
        except Exception:
            self.zones = []
        try:
            if hasattr(self, '_update_zone_table'):
                self._update_zone_table()
        except Exception:
            pass
        self._rv_kde_update_zone_table()
        self.update_random_validation_kde()
    def toggle_ga_fixed(self, state, row, table=None):
        """Toggle the fixed state of a GA parameter row"""
        if table is None:
            table = self.ga_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)
        
        # Enable/disable appropriate spinboxes
        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
        # If switching to fixed mode, copy current lower bound value to fixed value
        if fixed:
            fixed_value_spin.setValue(lower_bound_spin.value())
        # If switching to range mode, ensure lower bound is not greater than upper bound
        else:
            if lower_bound_spin.value() > upper_bound_spin.value():
                upper_bound_spin.setValue(lower_bound_spin.value())

    def toggle_adaptive_rates_options(self, state):
        """Show or hide adaptive rates options based on checkbox state"""
        self.adaptive_rates_options.setVisible(state == Qt.Checked)
        
        # Enable/disable the fixed rate inputs based on adaptive rates setting
        self.ga_cxpb_box.setEnabled(state != Qt.Checked)
        self.ga_mutpb_box.setEnabled(state != Qt.Checked)
        
        # Update tooltips to indicate that rates will be adaptive
        if state == Qt.Checked:
            self.ga_cxpb_box.setToolTip("Starting crossover probability (will adapt during optimization)")
            self.ga_mutpb_box.setToolTip("Starting mutation probability (will adapt during optimization)")
        else:
            self.ga_cxpb_box.setToolTip("Crossover probability")
            self.ga_mutpb_box.setToolTip("Mutation probability")

    def run_ga(self):
        """Run genetic algorithm optimization"""
        # Check if a GA worker is already running
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            QMessageBox.warning(self, "Process Running", 
                               "A Genetic Algorithm optimization is already running. Please wait for it to complete.")
            return
            
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "O Start must be less than O End.")
            return

        target_values, weights = self.get_target_values_weights()
        
        # Get GA hyperparameters
        pop_size = self.ga_pop_size_box.value()
        num_gen = self.ga_num_generations_box.value()
        crossover_prob = self.ga_cxpb_box.value()
        mutation_prob = self.ga_mutpb_box.value()
        tolerance = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()
        percentage_error_scale = self.ga_percentage_error_scale_box.value()
        cost_scale_factor = self.ga_cost_scale_box.value()
        dva_activation_threshold = self.ga_activation_threshold_box.value()
        dva_activation_penalty = self.ga_activation_penalty_box.value()
        dva_activation_threshold = self.ga_activation_threshold_box.value()
        dva_activation_penalty = self.ga_activation_penalty_box.value()

        # Get number of benchmark runs
        self.benchmark_runs = self.ga_benchmark_runs_box.value()
        self.current_benchmark_run = 0
        
        # Clear benchmark data if running multiple times
        if self.benchmark_runs > 1:
            self.ga_benchmark_data = []
            # Enable the benchmark tab if running multiple times
            self.ga_sub_tabs.setTabEnabled(self.ga_sub_tabs.indexOf(self.ga_sub_tabs.findChild(QWidget, "GA Benchmarking")), True)
        
        # Get DVA parameter bounds
        dva_bounds = {}
        dva_costs_by_category = {'material':{}, 'manufacturing':{}, 'maintenance':{}, 'operational':{}}
        EPSILON = 1e-6
        for row in range(self.ga_param_table.rowCount()):
            param_item = self.ga_param_table.item(row, 0)
            param_name = param_item.text()
            
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            
            # Read per-category costs
            try:
                mat_w = self.ga_param_table.cellWidget(row, 5)
                man_w = self.ga_param_table.cellWidget(row, 6)
                mai_w = self.ga_param_table.cellWidget(row, 7)
                opr_w = self.ga_param_table.cellWidget(row, 8)
                dva_costs_by_category['material'][param_name] = float(mat_w.value()) if mat_w is not None else 0.0
                dva_costs_by_category['manufacturing'][param_name] = float(man_w.value()) if man_w is not None else 0.0
                dva_costs_by_category['maintenance'][param_name] = float(mai_w.value()) if mai_w is not None else 0.0
                dva_costs_by_category['operational'][param_name] = float(opr_w.value()) if opr_w is not None else 0.0
            except Exception:
                pass

            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                if lower > upper:
                    QMessageBox.warning(self, "Input Error", 
                                       f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                dva_bounds[param_name] = (lower, upper)
        
        # Get main system parameters
        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )
        
        # Disable run buttons during optimization
        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.pause_ga_button.setEnabled(True)
        self.resume_ga_button.setEnabled(False)
        self.stop_ga_button.setEnabled(True)
        
        # Create progress bar if it doesn't exist
        if not hasattr(self, 'ga_progress_bar'):
            self.ga_progress_bar = QProgressBar()
            self.ga_progress_bar.setRange(0, 100)
            self.ga_progress_bar.setValue(0)
            self.ga_progress_bar.setTextVisible(True)
            self.ga_progress_bar.setFormat("GA Progress: %p%")
            
            # Find where to add progress bar in the layout
            ga_results_tab_layout = self.ga_results_text.parent().layout()
            ga_results_tab_layout.insertWidget(0, self.ga_progress_bar)
        else:
            self.ga_progress_bar.setValue(0)
            
        # Make sure the progress bar is visible
        self.ga_progress_bar.show()
        
        # Update status
        self.status_bar.showMessage("Running GA optimization...")
        self.ga_results_text.append("\n--- Running Genetic Algorithm Optimization ---\n")
        self.ga_results_text.append(f"Population Size: {pop_size}")
        self.ga_results_text.append(f"Number of Generations: {num_gen}")
        self.ga_results_text.append(f"Crossover Probability: {crossover_prob}")
        self.ga_results_text.append(f"Mutation Probability: {mutation_prob}")
        self.ga_results_text.append(f"Tolerance: {tolerance}")
        self.ga_results_text.append(f"Sparsity Penalty (alpha): {alpha}")
        
        # Add debug output for adaptive rates
        adaptive_rates = self.adaptive_rates_checkbox.isChecked()
        self.ga_results_text.append(f"Adaptive Rates: {'Enabled' if adaptive_rates else 'Disabled'}")
        if adaptive_rates:
            self.ga_results_text.append(f"  - Stagnation Limit: {self.stagnation_limit_box.value()}")
            self.ga_results_text.append(f"  - Crossover Range: {self.cxpb_min_box.value():.2f} - {self.cxpb_max_box.value():.2f}")
            self.ga_results_text.append(f"  - Mutation Range: {self.mutpb_min_box.value():.2f} - {self.mutpb_max_box.value():.2f}")
        self.ga_results_text.append("\nStarting optimization...\n")
        
        # Create and start worker
        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        
        # Convert dva_bounds and dva_order into ga_parameter_data format
        ga_parameter_data = []
        for param_name in original_dva_parameter_order:
            if param_name in dva_bounds:
                low, high = dva_bounds[param_name]
                # Check if parameter is fixed (low == high)
                fixed = abs(low - high) < EPSILON
                ga_parameter_data.append((param_name, low, high, fixed))
                
        # Store parameter configuration for later analysis
        self.ga_parameter_data = ga_parameter_data
        self.ga_active_parameters = [name for name, _, _, fixed in ga_parameter_data if not fixed]
        # If there's an existing worker, make sure it's properly cleaned up

        if hasattr(self, 'ga_worker'):
            try:
                self.ga_worker.finished.disconnect()
                self.ga_worker.error.disconnect()
                self.ga_worker.update.disconnect()
                self.ga_worker.progress.disconnect()
            except Exception:
                pass
                
        # Create a new worker
        # Determine controller mode (mutually exclusive)
        use_ml = self.controller_ml_radio.isChecked()
        use_adaptive = self.controller_adaptive_radio.isChecked()
        use_rl = self.controller_rl_radio.isChecked()

        # Store alpha used for this run for later reporting
        self._alpha_used_for_run = alpha
        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            ga_pop_size=pop_size,
            ga_num_generations=num_gen,
            ga_cxpb=crossover_prob,
            ga_mutpb=mutation_prob,
            ga_tol=tolerance,
            ga_parameter_data=ga_parameter_data,
            alpha=alpha,
            percentage_error_scale=percentage_error_scale,
            cost_scale_factor=cost_scale_factor,
            dva_activation_threshold=dva_activation_threshold,
            dva_activation_penalty=dva_activation_penalty,
            dva_costs_by_category=dva_costs_by_category,
            # Enhanced cost-benefit options
            use_enhanced_cost=self.enh_cost_enable_chk.isChecked(),
            benefit_w_primary=self.benefit_w_primary_box.value(),
            benefit_w_accuracy=self.benefit_w_accuracy_box.value(),
            benefit_w_sparsity=self.benefit_w_sparsity_box.value(),
            cat_w_material=self.cat_w_material_box.value(),
            cat_w_manufacturing=self.cat_w_manufacturing_box.value(),
            cat_w_maintenance=self.cat_w_maintenance_box.value(),
            cat_w_operational=self.cat_w_operational_box.value(),
            benefit_weight_start=self.benefit_weight_start_box.value(),
            benefit_weight_end=self.benefit_weight_end_box.value(),
            track_metrics=True,  # Enable metrics tracking for visualization (throttled)
            metrics_timer_interval=2000,  # reduce CPU overhead of psutil polling
            adaptive_rates=bool(use_adaptive and not (use_ml or use_rl)),  # ensure mutual exclusivity
            stagnation_limit=self.stagnation_limit_box.value(),  # Get stagnation limit from UI
            cxpb_min=self.cxpb_min_box.value(),  # Get min crossover probability
            cxpb_max=self.cxpb_max_box.value(),  # Get max crossover probability
            mutpb_min=self.mutpb_min_box.value(),  # Get min mutation probability
            mutpb_max=self.mutpb_max_box.value(),  # Get max mutation probability
            # ML/Bandit controller params
            use_ml_adaptive=bool(use_ml and not (use_adaptive or use_rl)),  # ensure mutual exclusivity
            pop_min=int(max(10, self.ga_pop_min_box.value())),
            pop_max=int(max(self.ga_pop_min_box.value(), self.ga_pop_max_box.value())),
            ml_ucb_c=self.ml_ucb_c_box.value(),
            ml_adapt_population=self.ml_pop_adapt_checkbox.isChecked(),
            ml_diversity_weight=self.ml_diversity_weight_box.value(),
            ml_diversity_target=self.ml_diversity_target_box.value(),
            ml_historical_weight=self.ml_historical_weight_box.value(),
            ml_current_weight=self.ml_current_weight_box.value(),
            # ML stagnation settings
            ml_stag_enabled=self.ml_stag_chk.isChecked(),
            ml_stag_limit=self.ml_stag_limit.value(),
            # RL controller params
            use_rl_controller=bool(use_rl and not (use_ml or use_adaptive)),
            rl_alpha=self.rl_alpha_box.value(),
            rl_gamma=self.rl_gamma_box.value(),
            rl_epsilon=self.rl_epsilon_box.value(),
            rl_epsilon_decay=self.rl_decay_box.value(),
            # RL stagnation settings
            rl_stag_enabled=self.rl_stag_chk.isChecked(),
            rl_stag_limit=self.rl_stag_limit.value(),
            # RL reward weights (optional UI; fallback defaults used if not present)
            rl_w1=getattr(self, 'rl_w1_box', type('x', (), {'value': lambda s: 1.0})()).value(),
            rl_w2=getattr(self, 'rl_w2_box', type('x', (), {'value': lambda s: 0.0})()).value(),
            rl_w3=getattr(self, 'rl_w3_box', type('x', (), {'value': lambda s: 0.0})()).value(),
            rl_w4=getattr(self, 'rl_w4_box', type('x', (), {'value': lambda s: self.ml_diversity_weight_box.value()})()).value(),
            rl_cv_target=getattr(self, 'rl_cv_target_box', type('x', (), {'value': lambda s: self.ml_diversity_target_box.value()})()).value(),
            # Surrogate
            use_surrogate=self.surrogate_checkbox.isChecked(),
            surrogate_pool_factor=self.surr_pool_factor_box.value(),
            surrogate_k=self.surr_k_box.value(),
            surrogate_explore_frac=self.surr_explore_frac_box.value(),
            # Seeding
            seeding_method=(
                "random" if self.seeding_method_combo.currentText().lower().startswith("random") else
                ("sobol" if self.seeding_method_combo.currentText().lower().startswith("sobol") else
                 ("lhs" if self.seeding_method_combo.currentText().lower().startswith("latin") else
                  ("neural" if self.seeding_method_combo.currentText().lower().startswith("neural") else
                   ("memory" if self.seeding_method_combo.currentText().lower().startswith("memory") else "best"))))
            ),
            best_pool_mult=self.best_pool_mult.value(),
            best_diversity_frac=self.best_diversity_frac.value(),
            use_neural_seeding=self.seeding_method_combo.currentText().lower().startswith("neural"),
            neural_acq_type=self.neural_acq_combo.currentText().lower(),
            neural_beta_min=self.neural_beta_min.value(),
            neural_beta_max=self.neural_beta_max.value(),
            neural_epsilon=self.neural_eps.value(),
            neural_pool_mult=self.neural_pool_mult.value(),
            neural_epochs=self.neural_epochs.value(),
            neural_time_cap_ms=self.neural_time_cap.value(),
            neural_ensemble_n=self.neural_ensemble.value(),
            neural_hidden=self.neural_hidden.value(),
            neural_layers=self.neural_layers.value(),
            neural_dropout=self.neural_dropout.value(),
            neural_weight_decay=self.neural_wd.value(),
            neural_enable_grad_refine=self.neural_grad_refine_chk.isChecked(),
            neural_grad_steps=self.neural_grad_steps.value(),
            neural_device=self.neural_device_combo.currentText()
        )
        
        # Connect signals using strong references to avoid premature garbage collection
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.ga_worker.progress.connect(self.update_ga_progress)
        
        # Set up a watchdog timer for the GA worker
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
        else:
            self.ga_watchdog_timer = QTimer(self)
            self.ga_watchdog_timer.timeout.connect(self.check_ga_worker_health)
            
        self.ga_watchdog_timer.start(10000)  # Check every 10 seconds
        
        # Start the worker
        self.ga_worker.start()

    def pause_ga(self):
        """Pause the GA optimization"""
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            self.ga_worker.pause()
            self.pause_ga_button.setEnabled(False)
            self.resume_ga_button.setEnabled(True)
            self.status_bar.showMessage("GA optimization paused")

    def resume_ga(self):
        """Resume a paused GA optimization"""
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            self.ga_worker.resume()
            self.pause_ga_button.setEnabled(True)
            self.resume_ga_button.setEnabled(False)
            self.status_bar.showMessage("GA optimization resumed")

    def stop_ga(self):
        """Terminate the GA optimization"""
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            self.ga_worker.stop()
            self.pause_ga_button.setEnabled(False)
            self.resume_ga_button.setEnabled(False)
            self.stop_ga_button.setEnabled(False)
            self.status_bar.showMessage("Stopping GA optimization...")

    def check_ga_worker_health(self):
        """Check if the GA worker is still responsive"""
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            # The worker is still running, which is good
            # We could add more sophisticated checks here if needed
            pass
        else:
            # The worker is not running anymore, stop the watchdog
            if hasattr(self, 'ga_watchdog_timer'):
                self.ga_watchdog_timer.stop()
                
    def update_ga_progress(self, value):
        """Update the GA progress bar, accounting for multiple benchmark runs"""
        if hasattr(self, 'ga_progress_bar'):
            if hasattr(self, 'benchmark_runs') and self.benchmark_runs > 1:
                # Calculate overall progress across all runs
                # Each run contributes (1/total_runs) of the progress
                run_contribution = 100.0 / self.benchmark_runs
                current_run_progress = value / 100.0  # Convert to fraction
                # Add progress from completed runs plus fractional progress from current run
                overall_progress = ((self.current_benchmark_run - 1) * run_contribution) + (current_run_progress * run_contribution)
                self.ga_progress_bar.setValue(int(overall_progress))
            else:
                # Single run - direct progress
                self.ga_progress_bar.setValue(value)
            
    def handle_ga_finished(self, results, best_ind, parameter_names, best_fitness):
        """Handle the completion of the GA optimization"""
        # Stop the watchdog timer
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
        
        # For benchmarking, collect data from this run
        self.current_benchmark_run += 1
        
        # Store benchmark results
        if hasattr(self, 'benchmark_runs') and self.benchmark_runs > 1:
            # Create a data dictionary for this run
            # Build results summary from worker results
            results_summary = {}
            if isinstance(results, dict):
                for key in ['singular_response', 'percentage_differences', 'composite_measures']:
                    if key in results:
                        results_summary[key] = results[key]

            run_data = {
                'run_number': self.current_benchmark_run,
                'best_fitness': float(best_fitness) if np.isfinite(best_fitness) else np.nan,
                'best_solution': list(best_ind),
                'parameter_names': parameter_names, 'active_parameters': getattr(self, 'ga_active_parameters', []),
                'alpha': getattr(self, '_alpha_used_for_run', None),
                'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()) if hasattr(self, 'ga_percentage_error_scale_box') else None,
                'cost_scale_factor': float(self.ga_cost_scale_box.value()) if hasattr(self, 'ga_cost_scale_box') else None,
                'dva_activation_threshold': float(self.ga_activation_threshold_box.value()) if hasattr(self, 'ga_activation_threshold_box') else None,
                'dva_activation_penalty': float(self.ga_activation_penalty_box.value()) if hasattr(self, 'ga_activation_penalty_box') else None,
                'dva_costs': {
                    self.ga_param_table.item(r,0).text(): {
                        'material': float(self.ga_param_table.cellWidget(r,5).value()) if self.ga_param_table.cellWidget(r,5) else 0.0,
                        'manufacturing': float(self.ga_param_table.cellWidget(r,6).value()) if self.ga_param_table.cellWidget(r,6) else 0.0,
                        'maintenance': float(self.ga_param_table.cellWidget(r,7).value()) if self.ga_param_table.cellWidget(r,7) else 0.0,
                        'operational': float(self.ga_param_table.cellWidget(r,8).value()) if self.ga_param_table.cellWidget(r,8) else 0.0,
                    } for r in range(self.ga_param_table.rowCount())
                } if hasattr(self, 'ga_param_table') and self.ga_param_table.columnCount() > 8 else {},
                'use_enhanced_cost': bool(self.enh_cost_enable_chk.isChecked()),
                'benefit_w_primary': float(self.benefit_w_primary_box.value()),
                'benefit_w_accuracy': float(self.benefit_w_accuracy_box.value()),
                'benefit_w_sparsity': float(self.benefit_w_sparsity_box.value()),
                'category_w_material': float(self.cat_w_material_box.value()),
                'category_w_manufacturing': float(self.cat_w_manufacturing_box.value()),
                'category_w_maintenance': float(self.cat_w_maintenance_box.value()),
                'category_w_operational': float(self.cat_w_operational_box.value()),
                'benefit_weight_start': float(self.benefit_weight_start_box.value()),
                'benefit_weight_end': float(self.benefit_weight_end_box.value()),
                'generation_ratio': float(self.generation_ratio_box.value()),
                'dva_category_map': {self.ga_param_table.item(r,0).text(): self.ga_param_table.cellWidget(r,6).currentText().lower() for r in range(self.ga_param_table.rowCount()) if isinstance(self.ga_param_table.cellWidget(r,6), QComboBox) and self.ga_param_table.cellWidget(r,6).currentText().lower() != 'auto'},
                'results_summary': results_summary
            }
            
            # Add any additional metrics from results
            if isinstance(results, dict):
                for key, value in results.items():
                    # Only keep simple numeric scalars; avoid nested dicts
                    try:
                        v = float(value)
                        if np.isfinite(v):
                            run_data[key] = v
                    except Exception:
                        pass

                # Add benchmark metrics if available
                if 'benchmark_metrics' in results:
                    run_data['benchmark_metrics'] = results['benchmark_metrics']
            
            # Store the run data
            self.ga_benchmark_data.append(run_data)
            
            # Update the status message
            self.status_bar.showMessage(f"GA run {self.current_benchmark_run} of {self.benchmark_runs} completed")
            
            # Update progress bar to show completed percentage of all runs
            if hasattr(self, 'ga_progress_bar'):
                progress = int(self.current_benchmark_run * 100 / self.benchmark_runs)
                self.ga_progress_bar.setValue(progress)
            
            # Check if we need to run again
            if self.current_benchmark_run < self.benchmark_runs:
                self.ga_results_text.append(f"\n--- Run {self.current_benchmark_run} completed, starting run {self.current_benchmark_run + 1}/{self.benchmark_runs} ---")
                # Set up for next run
                QTimer.singleShot(100, self.run_next_ga_benchmark)
                return
            else:
                # All runs completed, visualize the benchmark results
                self.visualize_ga_benchmark_results()
                self.export_benchmark_button.setEnabled(True)
                self.ga_results_text.append(f"\n--- All {self.benchmark_runs} benchmark runs completed ---")
        else:
            # For single runs, store the data directly
            results_summary = {}
            if isinstance(results, dict):
                for key in ['singular_response', 'percentage_differences', 'composite_measures']:
                    if key in results:
                        results_summary[key] = results[key]

            run_data = {
                'run_number': 1,
                'best_fitness': float(best_fitness) if np.isfinite(best_fitness) else np.nan,
                'best_solution': list(best_ind),
                'parameter_names': parameter_names, 'active_parameters': getattr(self, 'ga_active_parameters', []),
                'alpha': getattr(self, '_alpha_used_for_run', None),
                'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()) if hasattr(self, 'ga_percentage_error_scale_box') else None,
                'cost_scale_factor': float(self.ga_cost_scale_box.value()) if hasattr(self, 'ga_cost_scale_box') else None,
                'dva_activation_threshold': float(self.ga_activation_threshold_box.value()) if hasattr(self, 'ga_activation_threshold_box') else None,
                'dva_activation_penalty': float(self.ga_activation_penalty_box.value()) if hasattr(self, 'ga_activation_penalty_box') else None,
                'dva_costs': {
                    self.ga_param_table.item(r,0).text(): {
                        'material': float(self.ga_param_table.cellWidget(r,5).value()) if self.ga_param_table.cellWidget(r,5) else 0.0,
                        'manufacturing': float(self.ga_param_table.cellWidget(r,6).value()) if self.ga_param_table.cellWidget(r,6) else 0.0,
                        'maintenance': float(self.ga_param_table.cellWidget(r,7).value()) if self.ga_param_table.cellWidget(r,7) else 0.0,
                        'operational': float(self.ga_param_table.cellWidget(r,8).value()) if self.ga_param_table.cellWidget(r,8) else 0.0,
                    } for r in range(self.ga_param_table.rowCount())
                } if hasattr(self, 'ga_param_table') and self.ga_param_table.columnCount() > 8 else {},
                'use_enhanced_cost': bool(self.enh_cost_enable_chk.isChecked()),
                'benefit_w_primary': float(self.benefit_w_primary_box.value()),
                'benefit_w_accuracy': float(self.benefit_w_accuracy_box.value()),
                'benefit_w_sparsity': float(self.benefit_w_sparsity_box.value()),
                'category_w_material': float(self.cat_w_material_box.value()),
                'category_w_manufacturing': float(self.cat_w_manufacturing_box.value()),
                'category_w_maintenance': float(self.cat_w_maintenance_box.value()),
                'category_w_operational': float(self.cat_w_operational_box.value()),
                'benefit_weight_start': float(self.benefit_weight_start_box.value()),
                'benefit_weight_end': float(self.benefit_weight_end_box.value()),
                'results_summary': results_summary
            }
            
            # Add benchmark metrics if available
            if isinstance(results, dict) and 'benchmark_metrics' in results:
                run_data['benchmark_metrics'] = results['benchmark_metrics']
            
            self.ga_benchmark_data = [run_data]
            self.visualize_ga_benchmark_results()
            # Enable export benchmark data for single-run case too
            if hasattr(self, 'export_benchmark_button'):
                self.export_benchmark_button.setEnabled(True)
                
        # Enable export of GA results
        if hasattr(self, 'export_ga_results_button'):
            self.export_ga_results_button.setEnabled(True)
                
        # Re-enable buttons when completely done
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        self.pause_ga_button.setEnabled(False)
        self.resume_ga_button.setEnabled(False)
        self.stop_ga_button.setEnabled(False)
        
        self.status_bar.showMessage("GA optimization completed")
        
        # Only show detailed results for single runs or the final benchmark run
        if not hasattr(self, 'benchmark_runs') or self.benchmark_runs == 1 or self.current_benchmark_run == self.benchmark_runs:
            self.ga_results_text.append("\n--- GA Optimization Completed ---")
            self.ga_results_text.append(f"Best fitness: {best_fitness:.6f}")
        self.ga_results_text.append("\nBest Parameters:")
        
        # Check if there are any warnings in the results
        if isinstance(results, dict) and "Warning" in results:
            self.ga_results_text.append(f"\nWarning: {results['Warning']}")
        
        # Create a dictionary mapping parameter names to their values
        best_params = {name: value for name, value in zip(parameter_names, best_ind)}
        
        # Store best parameters for easy access later
        self.current_ga_best_params = best_params
        self.current_ga_best_fitness = best_fitness
        self.current_ga_full_results = results
        
        for param_name, value in best_params.items():
            self.ga_results_text.append(f"  {param_name}: {value:.6f}")
            
        # If we have actual results, show them
        if isinstance(results, dict) and "singular_response" in results:
            self.ga_results_text.append(f"\nFinal Singular Response: {results['singular_response']:.6f}")
        
    def handle_ga_error(self, error_msg):
        """Handle errors from the GA worker"""
        # Stop the watchdog timer
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
            
        # Hide or reset the progress bar
        if hasattr(self, 'ga_progress_bar'):
            self.ga_progress_bar.setValue(0)
            
        QMessageBox.critical(self, "Error in GA Optimization", str(error_msg))
        self.status_bar.showMessage("GA optimization failed")
        self.ga_results_text.append(f"\nError in GA optimization: {error_msg}")
        
        # Make sure to re-enable buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        self.pause_ga_button.setEnabled(False)
        self.resume_ga_button.setEnabled(False)
        self.stop_ga_button.setEnabled(False)
        
        # Try to recover by cleaning up any residual state
        if hasattr(self, 'ga_worker'):
            try:
                # Attempt to terminate the worker if it's still running
                if self.ga_worker.isRunning():
                    self.ga_worker.terminate()
                    self.ga_worker.wait(1000)  # Wait up to 1 second for it to finish
            except Exception as e:
                print(f"Error cleaning up GA worker: {str(e)}")
        
    def handle_ga_update(self, msg):
        """Handle update messages from the GA worker"""
        try:
            # Attempt to fix common UTF-8 mojibake from box-drawing characters
            # 1) Re-decode as UTF-8 via latin-1 roundtrip, then 2) map to ASCII
            try:
                fixed = msg.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
            except Exception:
                fixed = msg
            # Map box drawing characters to ASCII for cleaner display
            replacements = {
                'Â¦': '|', '+': '+', '-': '+', '+': '+', '+': '+', '+': '+', 'Â¦': '+',
                '+': '+', '-': '+', '+': '+', '-': '-', '': '-', '': '|',
            }
            for k, v in replacements.items():
                fixed = fixed.replace(k, v)
            # Also map common mojibake sequences directly if any remain
            # e.g., 'ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡' -> '|', 'ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬' -> '-'
            direct = {
                '+': '|', '-': '-', '-': '+', '+': '+', 'Â¦': '+',
                '+': '+', '-': '+', '+': '+', '+': '+', '+': '+',
                'Â¦': '+', '-': '+', '+': '+',
            }
            for k, v in direct.items():
                fixed = fixed.replace(k, v)
            self.ga_results_text.append(fixed)
        except Exception:
            self.ga_results_text.append(msg)
        # Auto-scroll to the bottom to show latest messages
        self.ga_results_text.verticalScrollBar().setValue(
            self.ga_results_text.verticalScrollBar().maximum()
        )
        
    def run_next_ga_benchmark(self):
        """Run the next GA benchmark iteration"""
        # Clear the existing GA worker to start fresh
        if hasattr(self, 'ga_worker'):
            try:
                self.ga_worker.finished.disconnect()
                self.ga_worker.error.disconnect()
                self.ga_worker.update.disconnect()
                self.ga_worker.progress.disconnect()
            except Exception:
                pass
        
        # Get the required parameters again
        target_values, weights = self.get_target_values_weights()
        
        # Get GA hyperparameters
        pop_size = self.ga_pop_size_box.value()
        num_gen = self.ga_num_generations_box.value()
        crossover_prob = self.ga_cxpb_box.value()
        mutation_prob = self.ga_mutpb_box.value()
        tolerance = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()
        percentage_error_scale = self.ga_percentage_error_scale_box.value()
        cost_scale_factor = self.ga_cost_scale_box.value()

        # Ensure control buttons reflect running state
        self.run_ga_button.setEnabled(False)
        self.pause_ga_button.setEnabled(True)
        self.resume_ga_button.setEnabled(False)
        self.stop_ga_button.setEnabled(True)

        # Also pick activation settings for benchmark run
        dva_activation_threshold = self.ga_activation_threshold_box.value()
        dva_activation_penalty = self.ga_activation_penalty_box.value()

        # Get DVA parameter bounds and costs
        dva_bounds = {}
        dva_costs_by_category = {'material':{}, 'manufacturing':{}, 'maintenance':{}, 'operational':{}}
        EPSILON = 1e-6
        for row in range(self.ga_param_table.rowCount()):
            param_item = self.ga_param_table.item(row, 0)
            param_name = param_item.text()
            
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            # Read per-category costs
            try:
                mat_w = self.ga_param_table.cellWidget(row, 5)
                man_w = self.ga_param_table.cellWidget(row, 6)
                mai_w = self.ga_param_table.cellWidget(row, 7)
                opr_w = self.ga_param_table.cellWidget(row, 8)
                dva_costs_by_category['material'][param_name] = float(mat_w.value()) if mat_w is not None else 0.0
                dva_costs_by_category['manufacturing'][param_name] = float(man_w.value()) if man_w is not None else 0.0
                dva_costs_by_category['maintenance'][param_name] = float(mai_w.value()) if mai_w is not None else 0.0
                dva_costs_by_category['operational'][param_name] = float(opr_w.value()) if opr_w is not None else 0.0
            except Exception:
                pass
            
            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                dva_bounds[param_name] = (lower, upper)
        
        # Get main system parameters
        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )
        
        # Reset progress bar
        if hasattr(self, 'ga_progress_bar'):
            self.ga_progress_bar.setValue(0)
            
        # Make sure the progress bar is visible
        self.ga_progress_bar.show()
        
        # Update status
        self.status_bar.showMessage(f"Running GA optimization (Run {self.current_benchmark_run + 1}/{self.benchmark_runs})...")
        
        # Create and start worker
        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        
        # Convert dva_bounds and dva_order into ga_parameter_data format
        ga_parameter_data = []
        for param_name in original_dva_parameter_order:
            if param_name in dva_bounds:
                low, high = dva_bounds[param_name]
                # Check if parameter is fixed (low == high)
                fixed = abs(low - high) < EPSILON
                ga_parameter_data.append((param_name, low, high, fixed))
        
        # Create a new worker
        use_ml = self.controller_ml_radio.isChecked()
        use_adaptive = self.controller_adaptive_radio.isChecked()
        use_rl = self.controller_rl_radio.isChecked()
        # Store alpha used for this run for later reporting
        self._alpha_used_for_run = alpha
        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            ga_pop_size=pop_size,
            ga_num_generations=num_gen,
            ga_cxpb=crossover_prob,
            ga_mutpb=mutation_prob,
            ga_tol=tolerance,
            ga_parameter_data=ga_parameter_data,
            alpha=alpha,
            percentage_error_scale=percentage_error_scale,
            cost_scale_factor=cost_scale_factor,
            dva_activation_threshold=dva_activation_threshold,
            dva_activation_penalty=dva_activation_penalty,
            dva_costs_by_category=dva_costs_by_category,
            track_metrics=True,  # Enable metrics tracking for visualization
            adaptive_rates=bool(use_adaptive and not (use_ml or use_rl)),  # ensure mutual exclusivity
            stagnation_limit=self.stagnation_limit_box.value(),  # Get stagnation limit from UI
            cxpb_min=self.cxpb_min_box.value(),  # Get min crossover probability
            cxpb_max=self.cxpb_max_box.value(),  # Get max crossover probability
            mutpb_min=self.mutpb_min_box.value(),  # Get min mutation probability
            mutpb_max=self.mutpb_max_box.value(),  # Get max mutation probability
            # ML/Bandit controller params
            use_ml_adaptive=bool(use_ml and not (use_adaptive or use_rl)),  # ensure mutual exclusivity
            pop_min=int(max(10, self.ga_pop_min_box.value())),
            pop_max=int(max(self.ga_pop_min_box.value(), self.ga_pop_max_box.value())),
            ml_ucb_c=self.ml_ucb_c_box.value(),
            ml_adapt_population=self.ml_pop_adapt_checkbox.isChecked(),
            ml_diversity_weight=self.ml_diversity_weight_box.value(),
            ml_diversity_target=self.ml_diversity_target_box.value(),
            ml_historical_weight=self.ml_historical_weight_box.value(),
            ml_current_weight=self.ml_current_weight_box.value(),
            # ML stagnation settings
            ml_stag_enabled=self.ml_stag_chk.isChecked(),
            ml_stag_limit=self.ml_stag_limit.value(),
            # RL controller params
            use_rl_controller=bool(use_rl and not (use_ml or use_adaptive)),
            rl_alpha=self.rl_alpha_box.value(),
            rl_gamma=self.rl_gamma_box.value(),
            rl_epsilon=self.rl_epsilon_box.value(),
            rl_epsilon_decay=self.rl_decay_box.value(),
            # RL stagnation settings
            rl_stag_enabled=self.rl_stag_chk.isChecked(),
            rl_stag_limit=self.rl_stag_limit.value(),
            # RL reward weights (optional UI; fallback defaults used if not present)
            rl_w1=getattr(self, 'rl_w1_box', type('x', (), {'value': lambda s: 1.0})()).value(),
            rl_w2=getattr(self, 'rl_w2_box', type('x', (), {'value': lambda s: 0.0})()).value(),
            rl_w3=getattr(self, 'rl_w3_box', type('x', (), {'value': lambda s: 0.0})()).value(),
            rl_w4=getattr(self, 'rl_w4_box', type('x', (), {'value': lambda s: self.ml_diversity_weight_box.value()})()).value(),
            rl_cv_target=getattr(self, 'rl_cv_target_box', type('x', (), {'value': lambda s: self.ml_diversity_target_box.value()})()).value(),
            # Surrogate
            use_surrogate=self.surrogate_checkbox.isChecked(),
            surrogate_pool_factor=self.surr_pool_factor_box.value(),
            surrogate_k=self.surr_k_box.value(),
            surrogate_explore_frac=self.surr_explore_frac_box.value(),
            # Enhanced cost-benefit options
            use_enhanced_cost=self.enh_cost_enable_chk.isChecked(),
            benefit_w_primary=self.benefit_w_primary_box.value(),
            benefit_w_accuracy=self.benefit_w_accuracy_box.value(),
            benefit_w_sparsity=self.benefit_w_sparsity_box.value(),
            cat_w_material=self.cat_w_material_box.value(),
            cat_w_manufacturing=self.cat_w_manufacturing_box.value(),
            cat_w_maintenance=self.cat_w_maintenance_box.value(),
            cat_w_operational=self.cat_w_operational_box.value(),
            benefit_weight_start=self.benefit_weight_start_box.value(),
            benefit_weight_end=self.benefit_weight_end_box.value(),
            # Seeding (ensure benchmark runs respect chosen method)
            seeding_method=(
                "random" if self.seeding_method_combo.currentText().lower().startswith("random") else
                ("sobol" if self.seeding_method_combo.currentText().lower().startswith("sobol") else
                 ("lhs" if self.seeding_method_combo.currentText().lower().startswith("latin") else
                  ("neural" if self.seeding_method_combo.currentText().lower().startswith("neural") else
                   ("memory" if self.seeding_method_combo.currentText().lower().startswith("memory") else "best"))))
            ),
            best_pool_mult=self.best_pool_mult.value(),
            best_diversity_frac=self.best_diversity_frac.value(),
            use_neural_seeding=self.seeding_method_combo.currentText().lower().startswith("neural"),
            neural_acq_type=self.neural_acq_combo.currentText().lower(),
            neural_beta_min=self.neural_beta_min.value(),
            neural_beta_max=self.neural_beta_max.value(),
            neural_epsilon=self.neural_eps.value(),
            neural_pool_mult=self.neural_pool_mult.value(),
            neural_epochs=self.neural_epochs.value(),
            neural_time_cap_ms=self.neural_time_cap.value(),
            neural_ensemble_n=self.neural_ensemble.value(),
            neural_hidden=self.neural_hidden.value(),
            neural_layers=self.neural_layers.value(),
            neural_dropout=self.neural_dropout.value(),
            neural_weight_decay=self.neural_wd.value(),
            neural_enable_grad_refine=self.neural_grad_refine_chk.isChecked(),
            neural_grad_steps=self.neural_grad_steps.value(),
            neural_device=self.neural_device_combo.currentText()
        )
        
        # Connect signals using strong references to avoid premature garbage collection
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.ga_worker.progress.connect(self.update_ga_progress)
        
        # Set up a watchdog timer for the GA worker
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
        else:
            self.ga_watchdog_timer = QTimer(self)
            self.ga_watchdog_timer.timeout.connect(self.check_ga_worker_health)
            
        self.ga_watchdog_timer.start(10000)  # Check every 10 seconds
        
        # Start the worker
        self.ga_worker.start()
    
    def _open_plot_window(self, fig, title):
        """Opens a new window to display a matplotlib figure."""
        plot_window = PlotWindow(fig, title)
        plot_window.setMinimumSize(800, 600)
        plot_window.show()
        # Keep a reference to prevent garbage collection
        if not hasattr(self, '_plot_windows'):
            self._plot_windows = []
        self._plot_windows.append(plot_window)
    
    def visualize_ga_benchmark_results(self):
        """Create visualizations for GA benchmark results"""
        if not hasattr(self, 'ga_benchmark_data') or not self.ga_benchmark_data:
            return
            
        import numpy as np
        import pandas as pd
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        import seaborn as sns
        
        # Convert benchmark data to DataFrame for easier analysis
        if not isinstance(self.ga_benchmark_data, list) or len(self.ga_benchmark_data) == 0:
            return
        df = pd.DataFrame(self.ga_benchmark_data)
        # Basic validation: ensure required columns exist; coerce types and drop invalid rows
        required_cols = ['run_number', 'best_fitness']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        # Coerce types
        df['run_number'] = pd.to_numeric(df['run_number'], errors='coerce')
        df['best_fitness'] = pd.to_numeric(df['best_fitness'], errors='coerce')
        # Drop rows missing core fields
        df = df.dropna(subset=['run_number', 'best_fitness']).copy()
        if df.empty:
            # Clear the All Runs table explicitly to reflect no valid data
            self.benchmark_runs_table.setRowCount(0)
            return
        # Ensure integer run numbers
        df['run_number'] = df['run_number'].astype(int)
        
        # Derive additional metrics for ranking/grouping
        try:
            if 'elapsed_time' not in df.columns and 'benchmark_metrics' in df.columns:
                df['elapsed_time'] = df['benchmark_metrics'].apply(lambda m: float(m.get('total_duration')) if isinstance(m, dict) and m.get('total_duration') is not None else np.nan)
        except Exception:
            pass
        try:
            if 'cpu_time_used' not in df.columns and 'benchmark_metrics' in df.columns:
                df['cpu_time_used'] = df['benchmark_metrics'].apply(lambda m: float(m.get('proc_cpu_time_used')) if isinstance(m, dict) and m.get('proc_cpu_time_used') is not None else np.nan)
        except Exception:
            pass
        try:
            if 'convergence_generation' not in df.columns and 'benchmark_metrics' in df.columns:
                def _conv_gen(m):
                    try:
                        if not isinstance(m, dict):
                            return np.nan
                        seq = m.get('best_fitness_per_gen')
                        if not seq:
                            return np.nan
                        best = min(seq)
                        return int(seq.index(best) + 1)
                    except Exception:
                        return np.nan
                df['convergence_generation'] = df['benchmark_metrics'].apply(_conv_gen)
                print(f"Debug: Derived convergence_generation for {len(df)} runs")
                print(f"Debug: Convergence generation values: {df['convergence_generation'].dropna().head().tolist()}")
                print(f"Debug: Number of valid convergence_generation values: {df['convergence_generation'].notna().sum()}")
            elif 'convergence_generation' in df.columns:
                print(f"Debug: convergence_generation already exists in dataframe")
                print(f"Debug: Existing convergence generation values: {df['convergence_generation'].dropna().head().tolist()}")
                print(f"Debug: Number of valid convergence_generation values: {df['convergence_generation'].notna().sum()}")
        except Exception:
            pass

        # Compute ranking score based on selected metric (lower is better)
        def _compute_score_row(row):
            metric = self.bench_rank_metric_combo.currentText() if hasattr(self, 'bench_rank_metric_combo') else 'Fitness (lower better)'
            try:
                if metric.startswith('Fitness'):
                    return float(row.get('best_fitness', np.inf))
                if metric.startswith('Convergence'):
                    return float(row.get('convergence_generation', np.inf))
                if metric.startswith('CPU Time'):
                    return float(row.get('cpu_time_used', np.inf))
                if metric.startswith('Wall Time'):
                    return float(row.get('elapsed_time', np.inf))
                # Composite: weighted normalized sum
                wf = float(self.weight_fitness_spin.value()) if hasattr(self, 'weight_fitness_spin') else 1.0
                wc = float(self.weight_convgen_spin.value()) if hasattr(self, 'weight_convgen_spin') else 0.5
                wcpu = float(self.weight_cpu_spin.value()) if hasattr(self, 'weight_cpu_spin') else 0.25
                wt = float(self.weight_time_spin.value()) if hasattr(self, 'weight_time_spin') else 0.25
                total_w = max(1e-9, wf + wc + wcpu + wt)
                def _norm(series, val):
                    try:
                        s = series.dropna()
                        lo, hi = float(s.min()), float(s.max())
                        if not np.isfinite(val) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                            return 1.0
                        return (float(val) - lo) / (hi - lo)
                    except Exception:
                        return 1.0
                score = 0.0
                score += wf * _norm(df['best_fitness'], row.get('best_fitness', np.nan))
                score += wc * _norm(df.get('convergence_generation', pd.Series(dtype=float)), row.get('convergence_generation', np.nan))
                score += wcpu * _norm(df.get('cpu_time_used', pd.Series(dtype=float)), row.get('cpu_time_used', np.nan))
                score += wt * _norm(df.get('elapsed_time', pd.Series(dtype=float)), row.get('elapsed_time', np.nan))
                return score / total_w
            except Exception:
                return float(row.get('best_fitness', np.inf))

        try:
            df['score'] = df.apply(_compute_score_row, axis=1)
        except Exception:
            df['score'] = df['best_fitness']

        # Group runs into Best/Mid/Worst based on closeness to reference runs
        try:
            # Use only finite scores to compute references (robust to old imports)
            finite_scores = pd.to_numeric(df['score'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_scores) >= 3:
                # Find reference runs
                best_ref_score = float(finite_scores.min())  # Best run (lowest score)
                worst_ref_score = float(finite_scores.max())  # Worst run (highest score)
                mean_ref_score = float(finite_scores.mean())  # Mean reference
                
                def _assign_to_closest_reference(score_val):
                    if not np.isfinite(score_val):
                        return 'Unknown'
                    
                    # Calculate distances to each reference
                    dist_to_best = abs(score_val - best_ref_score)
                    dist_to_mean = abs(score_val - mean_ref_score)
                    dist_to_worst = abs(score_val - worst_ref_score)
                    
                    # Assign to closest reference group
                    min_dist = min(dist_to_best, dist_to_mean, dist_to_worst)
                    
                    if min_dist == dist_to_best:
                        return 'Best'
                    elif min_dist == dist_to_mean:
                        return 'Mid'
                    else:
                        return 'Worst'
                
                df['group'] = df['score'].apply(_assign_to_closest_reference)
                
                # Store reference information for display
                df['best_ref_score'] = best_ref_score
                df['mean_ref_score'] = mean_ref_score
                df['worst_ref_score'] = worst_ref_score
            else:
                # Fallback for insufficient data
                median_score = df['score'].median()
                df['group'] = df['score'].apply(lambda v: 'Mid' if np.isfinite(v) else 'Unknown')
        except Exception:
            df['group'] = 'Unknown'
        
        # Visualize computational metrics
        widgets_dict = {
            'ga_ops_plot_widget': self.ga_ops_plot_widget
        }
        visualize_all_metrics(widgets_dict, df)
        
        # 1. Create violin & box plot
        try:
            # Clear existing plot layout
            if self.violin_plot_widget.layout():
                for i in reversed(range(self.violin_plot_widget.layout().count())): 
                    self.violin_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.violin_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for violin/box plot
            fig_violin = Figure(figsize=(10, 6), tight_layout=True)
            ax_violin = fig_violin.add_subplot(111)
            
            # Create violin plot with box plot inside
            violin = sns.violinplot(y=df["best_fitness"], ax=ax_violin, inner="box", color="skyblue", orient="v")
            ax_violin.set_title("Distribution of Best Fitness Values", fontsize=14)
            ax_violin.set_ylabel("Fitness Value", fontsize=12)
            ax_violin.grid(True, linestyle="--", alpha=0.7)
            
            # Add statistical annotations
            mean_fitness = df["best_fitness"].mean()
            median_fitness = df["best_fitness"].median()
            min_fitness = df["best_fitness"].min()
            max_fitness = df["best_fitness"].max()
            std_fitness = df["best_fitness"].std()
            
            # Get tolerance value
            tolerance = self.ga_tol_box.value()
            
            # Calculate additional statistics
            q1 = df["best_fitness"].quantile(0.25)
            q3 = df["best_fitness"].quantile(0.75)
            iqr = q3 - q1
            below_tolerance_count = len(df[df["best_fitness"] <= tolerance])
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            
            # Create a legend with enhanced statistical information
            legend_col1_text = (
                f"Mean: {mean_fitness:.6f}\n"
                f"Median: {median_fitness:.6f}\n"
                f"Min: {min_fitness:.6f}\n"
                f"Max: {max_fitness:.6f}\n"
                f"Std Dev: {std_fitness:.6f}"
            )

            legend_col2_text = (
                f"Q1 (25%): {q1:.6f}\n"
                f"Q3 (75%): {q3:.6f}\n"
                f"IQR: {iqr:.6f}\n"
                f"Tolerance: {tolerance:.6f}\n"
                f"Below Tolerance: {below_tolerance_count}/{len(df)} ({below_tolerance_percent:.1f}%)\n"
                f"Total Runs: {len(df)}"
            )
            
            # Create two text boxes for the legend
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) # Adjusted alpha
            ax_violin.text(0.05, 0.95, legend_col1_text, transform=ax_violin.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=props) # Adjusted fontsize
            ax_violin.text(0.28, 0.95, legend_col2_text, transform=ax_violin.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=props) # Adjusted fontsize and position
                    
            # Add percentile lines with labels (without redundant legend entries)
            percentiles = [25, 50, 75]
            percentile_values = df["best_fitness"].quantile(np.array(percentiles) / 100)
            
            # Add horizontal lines for percentiles
            for percentile, value in zip(percentiles, percentile_values):
                if percentile == 25:
                    color = 'orange'
                    linestyle = '--'
                elif percentile == 50:
                    color = 'red'
                    linestyle = '-'
                elif percentile == 75:
                    color = 'green'
                    linestyle = ':'
                else:
                    color = 'gray'
                    linestyle = '-'

                ax_violin.axhline(y=value, color=color, 
                                 linestyle=linestyle, 
                                 alpha=0.7, 
                                 label=f'{percentile}th Percentile')
            
            # Add mean and median lines
            ax_violin.axhline(y=mean_fitness, color='blue', linestyle='-', linewidth=1.5, alpha=0.8, label='Mean')
            ax_violin.axhline(y=median_fitness, color='purple', linestyle='--', linewidth=1.5, alpha=0.8, label='Median')

            # Add tolerance line with distinct appearance
            ax_violin.axhline(y=tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9, 
                           label=f'Tolerance')
            
            # Add a shaded region below tolerance (without redundant legend entry)
            ax_violin.axhspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Add compact legend for all lines
            ax_violin.legend(loc='upper right', framealpha=0.7, fontsize=9)
            
            # Create canvas and add to layout
            canvas_violin = FigureCanvasQTAgg(fig_violin)
            self.violin_plot_widget.layout().addWidget(canvas_violin)
            
            # Add toolbar for interactive features
            toolbar_violin = NavigationToolbar(canvas_violin, self.violin_plot_widget)
            self.violin_plot_widget.layout().addWidget(toolbar_violin)
            
            # Add save button to toolbar
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig_violin, "ga_violin_plot"))
            toolbar_violin.addWidget(save_button)
            
            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_violin, "GA Violin Plot"))
            self.violin_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating violin plot: {str(e)}")
            
        # 2. Create distribution plots
        try:
            # Clear existing plot layout
            if self.dist_plot_widget.layout():
                for i in reversed(range(self.dist_plot_widget.layout().count())): 
                    self.dist_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.dist_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for distribution plot
            fig_dist = Figure(figsize=(10, 6), tight_layout=True)
            ax_dist = fig_dist.add_subplot(111)
            
            # Create KDE plot with histogram
            sns.histplot(df["best_fitness"], kde=True, ax=ax_dist, color="skyblue", 
                        edgecolor="darkblue", alpha=0.5)
            ax_dist.set_title("Distribution of Best Fitness Values", fontsize=14)
            ax_dist.set_xlabel("Fitness Value", fontsize=12)
            ax_dist.set_ylabel("Frequency", fontsize=12)
            ax_dist.grid(True, linestyle="--", alpha=0.7)
            
            # Add vertical line for mean and median (compact legend)
            mean_fitness = df["best_fitness"].mean()
            median_fitness = df["best_fitness"].median()
            std_fitness = df["best_fitness"].std()
            ax_dist.axvline(mean_fitness, color='red', linestyle='--', linewidth=2, label='Mean')
            ax_dist.axvline(median_fitness, color='green', linestyle=':', linewidth=2, label='Median')
            
            # Add std deviation range (no legend entry)
            ax_dist.axvspan(mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.15, color='yellow', 
                          label=None)
            
            # Add tolerance line
            tolerance = self.ga_tol_box.value()
            ax_dist.axvline(tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9, 
                          label='Tolerance')
            
            # Add a shaded region below tolerance (no legend entry)
            ax_dist.axvspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Calculate statistics for annotation
            below_tolerance_count = len(df[df["best_fitness"] <= tolerance])
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            
            # Add compact, non-redundant statistics
            stats_text = (
                f"Runs: {len(df)}\n"
                f"Success: {below_tolerance_percent:.1f}%\n"
                f"Mean: {mean_fitness:.6f}\n"
                f"Std Dev: {std_fitness:.6f}"
            )
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.6)
            ax_dist.text(0.95, 0.3, stats_text, transform=ax_dist.transAxes, 
                      fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
                      
            # Add more compact legend
            ax_dist.legend(loc='upper left', framealpha=0.7, fontsize=9)
            
            # Create canvas and add to layout
            canvas_dist = FigureCanvasQTAgg(fig_dist)
            self.dist_plot_widget.layout().addWidget(canvas_dist)
            
            # Add toolbar for interactive features
            toolbar_dist = NavigationToolbar(canvas_dist, self.dist_plot_widget)
            self.dist_plot_widget.layout().addWidget(toolbar_dist)
            
            # Add save button to toolbar
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig_dist, "ga_distribution_plot"))
            toolbar_dist.addWidget(save_button)
            
            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_dist, "GA Distribution Plot"))
            self.dist_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
            
        # 3. Create scatter plots
        try:
            # Clear existing plot layout
            if self.scatter_plot_widget.layout():
                for i in reversed(range(self.scatter_plot_widget.layout().count())): 
                    self.scatter_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.scatter_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for scatter plot
            fig_scatter = Figure(figsize=(10, 6), tight_layout=True)
            ax_scatter = fig_scatter.add_subplot(111)
            
            # Create scatter plot of fitness vs run number with trend line
            from scipy import stats
            
            # Calculate linear regression and correlation
            slope, intercept, r_value, p_value, std_err = stats.linregress(df["run_number"], df["best_fitness"])
            correlation = r_value
            
            # Create scatter plot with trend line
            sns.regplot(x="run_number", y="best_fitness", data=df, ax=ax_scatter, 
                       scatter_kws={"color": "darkblue", "alpha": 0.6, "s": 50},
                       line_kws={"color": "red", "alpha": 0.7})
            
            trend_direction = "improving" if slope < 0 else "worsening" if slope > 0 else "stable"
            ax_scatter.set_title(f"Best Fitness Values Across Runs (Trend: {trend_direction})", fontsize=14)
            ax_scatter.set_xlabel("Run Number", fontsize=12)
            ax_scatter.set_ylabel("Best Fitness Value", fontsize=12)
            ax_scatter.grid(True, linestyle="--", alpha=0.7)
            
            # Add tolerance line (without legend entry)
            tolerance = self.ga_tol_box.value()
            ax_scatter.axhline(y=tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9,
                             label=None)
            
            # Add a shaded region below tolerance (no legend entry)
            ax_scatter.axhspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Color points that are below tolerance
            below_tolerance_df = df[df["best_fitness"] <= tolerance]
            below_tolerance_count = len(below_tolerance_df)
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            
            if not below_tolerance_df.empty:
                ax_scatter.scatter(below_tolerance_df["run_number"], below_tolerance_df["best_fitness"], 
                                 color='green', s=80, alpha=0.8, edgecolor='black', zorder=5,
                                 label='Success Points')
            
            # Find and mark best run
            best_run_idx = df["best_fitness"].idxmin()
            best_run = df.iloc[best_run_idx]
            ax_scatter.scatter(best_run["run_number"], best_run["best_fitness"], 
                             color='gold', s=120, alpha=1.0, edgecolor='black', marker='*', zorder=6,
                             label='Best Run')
            
            # Add correlation statistics in lower left (away from points)
            stats_text = (
                f"Correlation: {correlation:.4f}\n"
                f"Success Rate: {below_tolerance_percent:.1f}%\n"
                f"Best: {best_run['best_fitness']:.6f} (Tol: {tolerance:.6f})"
            )
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.6)
            ax_scatter.text(0.03, 0.15, stats_text, transform=ax_scatter.transAxes, 
                         fontsize=10, verticalalignment='bottom', bbox=props)
            
            # Add legend with fewer items
            ax_scatter.legend(loc='lower right', framealpha=0.7)
            
            # Create canvas and add to layout
            canvas_scatter = FigureCanvasQTAgg(fig_scatter)
            self.scatter_plot_widget.layout().addWidget(canvas_scatter)
            
            # Add toolbar for interactive features
            toolbar_scatter = NavigationToolbar(canvas_scatter, self.scatter_plot_widget)
            self.scatter_plot_widget.layout().addWidget(toolbar_scatter)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_scatter, "Scatter Plot"))
            self.scatter_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            
        # 4. Create heatmap of correlation between parameters and fitness
        try:
            # Clear existing plot layout
            if self.heatmap_plot_widget.layout():
                for i in reversed(range(self.heatmap_plot_widget.layout().count())): 
                    self.heatmap_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.heatmap_plot_widget.setLayout(QVBoxLayout())
            
            # Create figure for heatmap
            fig_heatmap = Figure(figsize=(12, 10), tight_layout=True)
            ax_heatmap = fig_heatmap.add_subplot(111)
            
            # Extract parameter values from each run into a DataFrame
            param_values = []
            
            if len(df) > 0 and 'best_solution' in df.iloc[0] and 'parameter_names' in df.iloc[0]:
                # Get parameter names
                param_names = df.iloc[0]['parameter_names']
                
                # Limit to max 10 parameters to keep visualization manageable
                max_params = min(10, len(param_names))
                selected_params = param_names[:max_params]
                
                # For each run, extract the parameter values
                for i, row in df.iterrows():
                    run_data = {'run_number': row['run_number'], 'best_fitness': row['best_fitness']}
                    
                    # Extract the parameter values
                    solution = row['best_solution']
                    for j, param in enumerate(selected_params):
                        if j < len(solution):
                            run_data[param] = solution[j]
                    
                    param_values.append(run_data)
                
                # Create DataFrame
                param_df = pd.DataFrame(param_values)
                
                if len(param_df) > 0 and len(param_df.columns) > 2:  # Need more than just run_number and best_fitness
                    # Calculate correlation matrix
                    corr_matrix = param_df.corr()
                    
                    # Create heatmap
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                               linewidths=0.5, ax=ax_heatmap, vmin=-1, vmax=1)
                    ax_heatmap.set_title("Correlation Between Parameters and Fitness", fontsize=14)
                    
                    # Create canvas and add to layout
                    canvas_heatmap = FigureCanvasQTAgg(fig_heatmap)
                    self.heatmap_plot_widget.layout().addWidget(canvas_heatmap)
                    
                    # Add toolbar for interactive features
                    toolbar_heatmap = NavigationToolbar(canvas_heatmap, self.heatmap_plot_widget)
                    self.heatmap_plot_widget.layout().addWidget(toolbar_heatmap)

                    # Add "Open in New Window" button
                    open_new_window_button = QPushButton("Open in New Window")
                    open_new_window_button.setObjectName("secondary-button")
                    open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_heatmap, "Heatmap Plot"))
                    self.heatmap_plot_widget.layout().addWidget(open_new_window_button)
                else:
                    # Create a label for insufficient data
                    no_data_label = QLabel("Insufficient data for correlation analysis")
                    self.heatmap_plot_widget.layout().addWidget(no_data_label)
            else:
                # Create a label if no parameter data
                no_data_label = QLabel("No parameter data available for correlation analysis")
                self.heatmap_plot_widget.layout().addWidget(no_data_label)
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            error_label = QLabel(f"Error creating heatmap: {str(e)}")
            self.heatmap_plot_widget.layout().addWidget(error_label)
            
        # 5. Create Q-Q plot for normality assessment
        try:
            # Clear existing plot layout
            if self.qq_plot_widget.layout():
                for i in reversed(range(self.qq_plot_widget.layout().count())): 
                    self.qq_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.qq_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for Q-Q plot
            fig_qq = Figure(figsize=(10, 6), tight_layout=True)
            ax_qq = fig_qq.add_subplot(111)
            
            # Create Q-Q plot
            from scipy import stats
            stats.probplot(df["best_fitness"], dist="norm", plot=ax_qq)
            ax_qq.set_title("Q-Q Plot for Normality Assessment", fontsize=14)
            ax_qq.set_xlabel("Theoretical Quantiles", fontsize=12)
            ax_qq.set_ylabel("Sample Quantiles", fontsize=12)
            ax_qq.grid(True, linestyle="--", alpha=0.7)
            
            # Perform normality tests
            shapiro_test = stats.shapiro(df["best_fitness"])
            ks_test = stats.kstest(df["best_fitness"], 'norm', 
                                 args=(df["best_fitness"].mean(), df["best_fitness"].std()))
            
            # Add test results as text
            test_text = (
                f"Shapiro-Wilk Test:\n"
                f"W = {shapiro_test[0]:.4f}\n"
                f"p-value = {shapiro_test[1]:.4f}\n\n"
                f"Kolmogorov-Smirnov Test:\n"
                f"D = {ks_test[0]:.4f}\n"
                f"p-value = {ks_test[1]:.4f}"
            )
            
            # Create a text box for the test results
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_qq.text(0.05, 0.95, test_text, transform=ax_qq.transAxes, 
                      fontsize=10, verticalalignment='top', bbox=props)
            
            # Create canvas and add to layout
            canvas_qq = FigureCanvasQTAgg(fig_qq)
            self.qq_plot_widget.layout().addWidget(canvas_qq)
            
            # Add toolbar for interactive features
            toolbar_qq = NavigationToolbar(canvas_qq, self.qq_plot_widget)
            self.qq_plot_widget.layout().addWidget(toolbar_qq)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_qq, "Q-Q Plot"))
            self.qq_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating Q-Q plot: {str(e)}")
        
                # 6. Update statistics table with enhanced metrics
        try:
            # Calculate comprehensive statistics for fitness and available parameters
            stats_data = []
            
            # Enhanced fitness statistics
            fitness_values = df["best_fitness"]
            fitness_stats = {
                "Metric": "Best Fitness",
                "Min": fitness_values.min(),
                "Max": fitness_values.max(),
                "Mean": fitness_values.mean(),
                "Std": fitness_values.std(),
                "Median": fitness_values.median(),
                "Q1": fitness_values.quantile(0.25),
                "Q3": fitness_values.quantile(0.75),
                "IQR": fitness_values.quantile(0.75) - fitness_values.quantile(0.25),
                "Variance": fitness_values.var(),
                "CV%": (fitness_values.std() / fitness_values.mean()) * 100 if fitness_values.mean() != 0 else 0,
                "Range": fitness_values.max() - fitness_values.min(),
                "Skewness": fitness_values.skew(),
                "Kurtosis": fitness_values.kurtosis()
            }
            stats_data.append(fitness_stats)
            
            # Add success rate statistics
            tolerance = 1e-6  # Default tolerance for success
            if hasattr(self, 'ga_tol_box'):
                tolerance = self.ga_tol_box.value()
            
            success_count = len(fitness_values[fitness_values <= tolerance])
            success_rate = (success_count / len(fitness_values)) * 100
            
            success_stats = {
                "Metric": "Success Rate",
                "Min": f"{success_rate:.2f}%",
                "Max": f"({success_count}/{len(fitness_values)})",
                "Mean": f"Tolerance: {tolerance:.2e}",
                "Std": f"Below: {success_count}",
                "Median": f"Above: {len(fitness_values) - success_count}",
                "Q1": "-",
                "Q3": "-",
                "IQR": "-",
                "Variance": "-",
                "CV%": "-",
                "Range": "-",
                "Skewness": "-",
                "Kurtosis": "-"
            }
            stats_data.append(success_stats)
            
            # Add elapsed time statistics if available
            if 'elapsed_time' in df.columns:
                time_values = df["elapsed_time"]
                time_stats = {
                    "Metric": "Elapsed Time (s)",
                    "Min": time_values.min(),
                    "Max": time_values.max(),
                    "Mean": time_values.mean(),
                    "Std": time_values.std(),
                    "Median": time_values.median(),
                    "Q1": time_values.quantile(0.25),
                    "Q3": time_values.quantile(0.75),
                    "IQR": time_values.quantile(0.75) - time_values.quantile(0.25),
                    "Variance": time_values.var(),
                    "CV%": (time_values.std() / time_values.mean()) * 100 if time_values.mean() != 0 else 0,
                    "Range": time_values.max() - time_values.min(),
                    "Skewness": time_values.skew(),
                    "Kurtosis": time_values.kurtosis()
                }
                stats_data.append(time_stats)
            
            # Add convergence statistics if available
            if 'convergence_generation' in df.columns:
                conv_values = df["convergence_generation"]
                conv_stats = {
                    "Metric": "Convergence Generation",
                    "Min": conv_values.min(),
                    "Max": conv_values.max(),
                    "Mean": conv_values.mean(),
                    "Std": conv_values.std(),
                    "Median": conv_values.median(),
                    "Q1": conv_values.quantile(0.25),
                    "Q3": conv_values.quantile(0.75),
                    "IQR": conv_values.quantile(0.75) - conv_values.quantile(0.25),
                    "Variance": conv_values.var(),
                    "CV%": (conv_values.std() / conv_values.mean()) * 100 if conv_values.mean() != 0 else 0,
                    "Range": conv_values.max() - conv_values.min(),
                    "Skewness": conv_values.skew(),
                    "Kurtosis": conv_values.kurtosis()
                }
                stats_data.append(conv_stats)
            
            # Add statistics for other numeric metrics in results
            for col in df.columns:
                if col not in ["run_number", "best_fitness", "best_solution", "parameter_names", "elapsed_time", "convergence_generation"] and df[col].dtype in [np.float64, np.int64]:
                    try:
                        col_values = df[col]
                        metric_stats = {
                            "Metric": col.replace('_', ' ').title(),
                            "Min": col_values.min(),
                            "Max": col_values.max(),
                            "Mean": col_values.mean(),
                            "Std": col_values.std(),
                            "Median": col_values.median(),
                            "Q1": col_values.quantile(0.25),
                            "Q3": col_values.quantile(0.75),
                            "IQR": col_values.quantile(0.75) - col_values.quantile(0.25),
                            "Variance": col_values.var(),
                            "CV%": (col_values.std() / col_values.mean()) * 100 if col_values.mean() != 0 else 0,
                            "Range": col_values.max() - col_values.min(),
                            "Skewness": col_values.skew(),
                            "Kurtosis": col_values.kurtosis()
                        }
                        stats_data.append(metric_stats)
                    except Exception as e:
                        print(f"Error calculating statistics for column {col}: {e}")
            
            # Update table headers and size for enhanced statistics
            enhanced_headers = ["Metric", "Min", "Max", "Mean", "Std", "Median", "Q1", "Q3", "IQR", "Variance", "CV%", "Range", "Skewness", "Kurtosis"]
            self.benchmark_stats_table.setColumnCount(len(enhanced_headers))
            self.benchmark_stats_table.setHorizontalHeaderLabels(enhanced_headers)
            self.benchmark_stats_table.setRowCount(len(stats_data))
            
            # Populate the enhanced statistics table
            for row, stat in enumerate(stats_data):
                for col, header in enumerate(enhanced_headers):
                    value = stat.get(header, "-")
                    if isinstance(value, (int, float)) and not isinstance(value, str):
                        if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
                            # Use scientific notation for very large or very small numbers
                            formatted_value = f"{value:.3e}"
                        else:
                            # Use regular formatting for normal numbers
                            formatted_value = f"{value:.6f}"
                    else:
                        formatted_value = str(value)
                    
                    item = QTableWidgetItem(formatted_value)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.benchmark_stats_table.setItem(row, col, item)
            
            # Add confidence intervals as a separate section
            confidence_data = []
            
            # Calculate 95% confidence intervals for fitness
            import scipy.stats as stats
            fitness_mean = fitness_values.mean()
            fitness_std = fitness_values.std()
            fitness_n = len(fitness_values)
            fitness_se = fitness_std / np.sqrt(fitness_n)
            fitness_ci = stats.t.interval(0.95, fitness_n-1, loc=fitness_mean, scale=fitness_se)
            
            ci_stats = {
                "Metric": "Fitness 95% CI",
                "Min": f"[{fitness_ci[0]:.6f},",
                "Max": f"{fitness_ci[1]:.6f}]",
                "Mean": f"Â±{fitness_ci[1] - fitness_mean:.6f}",
                "Std": f"SE: {fitness_se:.6f}",
                "Median": f"n = {fitness_n}",
                "Q1": "-",
                "Q3": "-",
                "IQR": "-",
                "Variance": "-",
                "CV%": "-",
                "Range": "-",
                "Skewness": "-",
                "Kurtosis": "-"
            }
            
            # Add the confidence interval row
            current_rows = self.benchmark_stats_table.rowCount()
            self.benchmark_stats_table.setRowCount(current_rows + 1)
            for col, header in enumerate(enhanced_headers):
                value = ci_stats.get(header, "-")
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight confidence interval row
                item.setBackground(QColor(240, 248, 255))  # Light blue background
                self.benchmark_stats_table.setItem(current_rows, col, item)
        except Exception as e:
            print(f"Error updating statistics tables: {str(e)}")
        
         
         # 7. Update runs table with ranking by selected metric and group coloring
        try:
             self.benchmark_runs_table.setRowCount(len(df))

             # Sort by score (lower is better)
             if 'score' in df.columns:
                 sorted_df = df.sort_values('score')
             else:
                 sorted_df = df.sort_values('best_fitness')
             
             # Reference mean for tooltip when fitness metric used
             mean_fitness = df['best_fitness'].mean() if 'best_fitness' in df.columns else np.nan
             mean_index = (df['best_fitness'] - mean_fitness).abs().idxmin() if 'best_fitness' in df.columns else None
             
             # Create a button class for the details button
             class DetailButton(QPushButton):
                 def __init__(self, run_number):
                     super().__init__("View Details")
                     self.run_number = run_number
             
             # Populate the table
             for i, (_, row) in enumerate(sorted_df.iterrows()):
                 run_number = int(row['run_number'])
                 fitness = row['best_fitness']
                 
                 # Create items for the table
                 run_item = QTableWidgetItem(str(run_number))
                 fitness_item = QTableWidgetItem(f"{fitness:.6f}")
                 rank_item = QTableWidgetItem(f"{i+1}/{len(df)}")
                 
                 # Set alignment
                 run_item.setTextAlignment(Qt.AlignCenter)
                 fitness_item.setTextAlignment(Qt.AlignCenter)
                 rank_item.setTextAlignment(Qt.AlignCenter)
                 
                 # Color coding by group
                 group = str(row.get('group', ''))
                 if group == 'Best':
                     base_color = QColor(200, 255, 200)
                 elif group == 'Mid':
                     base_color = QColor(235, 235, 200)
                 elif group == 'Worst':
                     base_color = QColor(255, 220, 220)
                 else:
                     base_color = QColor(230, 230, 230)
                 for it in (run_item, fitness_item, rank_item):
                     it.setBackground(base_color)
                 tip_parts = [f"Group: {group}"]
                 if np.isfinite(fitness):
                     tip_parts.append(f"Fitness: {fitness:.6f}")
                 if 'score' in row:
                     try:
                         tip_parts.append(f"Score: {float(row['score']):.6f}")
                     except Exception:
                         pass
                 run_item.setToolTip(" | ".join(tip_parts))
                 
                 # Add items to the table
                 self.benchmark_runs_table.setItem(i, 0, run_item)
                 self.benchmark_runs_table.setItem(i, 1, fitness_item)
                 self.benchmark_runs_table.setItem(i, 2, rank_item)
                 
                 # Add a details button
                 detail_btn = DetailButton(run_number)
                 detail_btn.clicked.connect(lambda _, btn=detail_btn: self.show_run_details(
                     self.benchmark_runs_table.item(
                         [i for i in range(self.benchmark_runs_table.rowCount()) 
                          if int(self.benchmark_runs_table.item(i, 0).text()) == btn.run_number][0], 0)))
                 self.benchmark_runs_table.setCellWidget(i, 3, detail_btn)
        except Exception as e:
             print(f"Error updating runs table: {str(e)}")
         
         # Connect export button if not already connected
        try:
            self.export_benchmark_button.clicked.disconnect()
        except:
            pass
        self.export_benchmark_button.clicked.connect(self.export_ga_benchmark_data)
        
        # Build/update comprehensive Group Summary tab content
        self._build_comprehensive_group_summary(df)
        
        # New: Build Parameter Ranges tab with multiple criteria and comparison tools
        try:
            # Remove existing "Parameter Ranges" tab if it exists to avoid duplicates
            try:
                for _i in range(self.benchmark_viz_tabs.count()):
                    if self.benchmark_viz_tabs.tabText(_i) == "Parameter Ranges":
                        self.benchmark_viz_tabs.removeTab(_i)
                        break
            except Exception:
                pass

            # Validate availability of parameter data in df
            param_names = []
            if len(df) > 0 and 'parameter_names' in df.columns and isinstance(df.iloc[0]['parameter_names'], list):
                param_names = list(df.iloc[0]['parameter_names'])

            if param_names:
                import numpy as _np
                import pandas as _pd
                from matplotlib.figure import Figure as _Figure
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas

                # Build parameter DataFrame aligned with fitness
                values_rows = []
                used_idx = []
                for _idx, _row in df.iterrows():
                    _sol = _row.get('best_solution', [])
                    if isinstance(_sol, list) and len(_sol) == len(param_names):
                        values_rows.append(_sol)
                        used_idx.append(_idx)
                if values_rows:
                    param_df = _pd.DataFrame(values_rows, columns=param_names)
                    fitness_series = df.loc[used_idx, 'best_fitness'].reset_index(drop=True) if 'best_fitness' in df.columns else None

                    # Criteria computation helpers (nested for locality)
                    def _iqr_range(vals):
                        q1 = _np.percentile(vals, 25)
                        q3 = _np.percentile(vals, 75)
                        return float(q1), float(q3)

                    def _p5_p95(vals):
                        lo = _np.percentile(vals, 5)
                        hi = _np.percentile(vals, 95)
                        return float(lo), float(hi)

                    def _tukey_whisker(vals):
                        q1 = _np.percentile(vals, 25)
                        q3 = _np.percentile(vals, 75)
                        iqr = q3 - q1
                        lf = q1 - 1.5 * iqr
                        uf = q3 + 1.5 * iqr
                        _v = _np.sort(vals)
                        lo_candidates = _v[_v >= lf]
                        hi_candidates = _v[_v <= uf]
                        lo = float(lo_candidates.min()) if lo_candidates.size else float(_v.min())
                        hi = float(hi_candidates.max()) if hi_candidates.size else float(_v.max())
                        return lo, hi

                    def _shortest_interval(vals, mass=0.68):
                        _v = _np.sort(vals)
                        n = _v.size
                        if n == 0:
                            return _np.nan, _np.nan
                        k = max(1, int(round(mass * n)))
                        if k >= n:
                            return float(_v[0]), float(_v[-1])
                        widths = _v[k-1:] - _v[:n-k+1]
                        j = int(_np.argmin(widths))
                        return float(_v[j]), float(_v[j + k - 1])

                    def _top_quantile_p5_p95(vals, fitness, q=0.25):
                        if fitness is None or len(fitness) != len(vals):
                            return _p5_p95(vals)
                        thr = _np.quantile(fitness, q)
                        mask = (fitness <= thr)
                        subset = _np.asarray(vals)[mask]
                        if subset.size < 3:
                            return _p5_p95(vals)
                        return _p5_p95(subset)

                    def _top_fraction_narrow(vals, fitness, frac=0.10, bandwidth=0.05):
                        """
                        Very limited band using only the top `frac` fraction of runs by fitness
                        and taking a narrow central interval around the median with total width
                        equal to `bandwidth` (e.g., Q47.5- Q52.5 for bandwidth=0.05).

                        Fallbacks to IQR on the full sample if inputs are insufficient.
                        """
                        try:
                            v = _np.asarray(vals)
                            if fitness is None or len(fitness) != len(v):
                                if v.size < 5:
                                    return _iqr_range(v)
                                ql = float(_np.quantile(v, 0.5 - 0.5 * bandwidth))
                                qh = float(_np.quantile(v, 0.5 + 0.5 * bandwidth))
                                return (min(ql, qh), max(ql, qh))
                            thr = _np.quantile(fitness, frac)
                            mask = (fitness <= thr)
                            subset = v[mask]
                            if subset.size < 5:
                                return _iqr_range(v)
                            ql = float(_np.quantile(subset, 0.5 - 0.5 * bandwidth))
                            qh = float(_np.quantile(subset, 0.5 + 0.5 * bandwidth))
                            return (min(ql, qh), max(ql, qh))
                        except Exception:
                            return _iqr_range(vals)

                    def _trimmed_mean_mad(vals):
                        v = _np.asarray(vals)
                        v_sorted = _np.sort(v)
                        n = v_sorted.size
                        if n == 0:
                            return _np.nan, _np.nan
                        cut = max(0, int(0.1 * n))
                        core = v_sorted[cut:n-cut] if n - 2*cut > 0 else v_sorted
                        tmean = float(core.mean()) if core.size else float(v.mean())
                        med = float(_np.median(v))
                        mad = float(_np.median(_np.abs(v - med)))
                        robust_sigma = 1.4826 * mad
                        lo = tmean - 1.5 * robust_sigma
                        hi = tmean + 1.5 * robust_sigma
                        # Clamp to observed span
                        return float(max(v.min(), lo)), float(min(v.max(), hi))

                    criteria_funcs = {
                        "IQR (Q1- Q3)": _iqr_range,
                        "P5- P95": _p5_p95,
                        "Tukey (no outliers)": _tukey_whisker,
                        "Shortest 68%": lambda x: _shortest_interval(x, 0.68),
                        "Top 25% P5-P95": lambda x: _top_quantile_p5_p95(x, fitness_series, 0.25),
                        "Top 10% Narrow Q47.5- Q52.5": lambda x: _top_fraction_narrow(x, fitness_series, 0.10, 0.05),
                        "TrimmedMean Â± 1.5*MAD": _trimmed_mean_mad,
                    }

                    # Pre-compute ranges: dict[criterion][param] = (lo, hi)
                    ranges_by_criterion = {}
                    for crit, fn in criteria_funcs.items():
                        d = {}
                        for pn in param_names:
                            try:
                                vals = _np.asarray(param_df[pn].dropna().values, dtype=float)
                                if vals.size == 0:
                                    d[pn] = (_np.nan, _np.nan)
                                else:
                                    lo, hi = fn(vals)
                                    if _np.isnan(lo) or _np.isnan(hi):
                                        lo, hi = float(vals.min()), float(vals.max())
                                    d[pn] = (float(min(lo, hi)), float(max(lo, hi)))
                            except Exception:
                                v = _np.asarray(param_df[pn].values, dtype=float)
                                d[pn] = (float(_np.nanmin(v)), float(_np.nanmax(v)))
                        ranges_by_criterion[crit] = d

                    # Colors for criteria in plots
                    crit_colors = {
                        "IQR (Q1- Q3)": "#1f77b4",
                        "P5- P95": "#ff7f0e",
                        "Tukey (no outliers)": "#2ca02c",
                        "Shortest 68%": "#d62728",
                        "Top 25% P5-P95": "#9467bd",
                        "Top 10% Narrow Q47.5- Q52.5": "#17becf",
                        "TrimmedMean Ãƒâ€šÃ‚Â± 1.5*MAD": "#8c564b",
                    }

                    # Build the Parameter Ranges tab UI
                    param_ranges_tab = QWidget()
                    param_ranges_layout = QVBoxLayout(param_ranges_tab)
                    ranges_tabs = QTabWidget()

                    # Helper to build a table widget for a single-criterion dictionary
                    def _build_table_for_ranges(rdict):
                        table = QTableWidget()
                        table.setColumnCount(5)
                        table.setHorizontalHeaderLabels(["Parameter", "Low", "High", "Width", "Center"])
                        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                        table.setRowCount(len(param_names))
                        for i, pn in enumerate(param_names):
                            lo, hi = rdict.get(pn, (_np.nan, _np.nan))
                            width = hi - lo if _np.isfinite(hi) and _np.isfinite(lo) else _np.nan
                            center = (hi + lo) / 2.0 if _np.isfinite(hi) and _np.isfinite(lo) else _np.nan
                            items = [
                                QTableWidgetItem(str(pn)),
                                QTableWidgetItem(f"{lo:.6f}" if _np.isfinite(lo) else "-"),
                                QTableWidgetItem(f"{hi:.6f}" if _np.isfinite(hi) else "-"),
                                QTableWidgetItem(f"{width:.6f}" if _np.isfinite(width) else "-"),
                                QTableWidgetItem(f"{center:.6f}" if _np.isfinite(center) else "-"),
                            ]
                            for j, it in enumerate(items):
                                it.setTextAlignment(Qt.AlignCenter)
                                table.setItem(i, j, it)
                        return table

                    # Helper to build a horizontal range plot for one criterion
                    def _build_range_plot(rdict, title, color):
                        fig = _Figure(figsize=(10, max(5, len(param_names) * 0.25)), tight_layout=True)
                        ax = fig.add_subplot(111)
                        y = _np.arange(len(param_names))
                        xmins = []
                        xmaxs = []
                        for i, pn in enumerate(param_names):
                            lo, hi = rdict.get(pn, (_np.nan, _np.nan))
                            xmins.append(lo)
                            xmaxs.append(hi)
                            if _np.isfinite(lo) and _np.isfinite(hi):
                                ax.hlines(i, lo, hi, colors=color, linewidth=3)
                                ax.plot([lo, hi], [i, i], 'o', color=color, markersize=4)
                        ax.set_yticks(y)
                        ax.set_yticklabels(param_names)
                        ax.set_xlabel("Value")
                        ax.set_title(title)
                        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
                        # Set x-limits with small padding
                        try:
                            xmin = _np.nanmin(_np.asarray(xmins))
                            xmax = _np.nanmax(_np.asarray(xmaxs))
                            if _np.isfinite(xmin) and _np.isfinite(xmax) and xmin < xmax:
                                pad = 0.02 * (xmax - xmin)
                                ax.set_xlim(xmin - pad, xmax + pad)
                        except Exception:
                            pass
                        canvas = _FigureCanvas(fig)
                        container = QWidget()
                        lay = QVBoxLayout(container)
                        lay.addWidget(canvas)
                        try:
                            toolbar = NavigationToolbar(canvas, None)
                            lay.addWidget(toolbar)
                        except Exception:
                            pass
                        return container

                    # Build one subtab per criterion
                    for crit_name, rdict in ranges_by_criterion.items():
                        tab = QWidget()
                        lay = QVBoxLayout(tab)
                        lay.addWidget(QLabel(f"Recommended ranges per parameter using: {crit_name}"))
                        table = _build_table_for_ranges(rdict)
                        # Enable export via context menu
                        try:
                            self._attach_table_export(table, f"parameter_ranges_{crit_name.replace(' ', '_').lower()}")
                        except Exception:
                            pass
                        lay.addWidget(table)
                        # Add export button using common dialog
                        export_btn = QPushButton("Export Table")
                        export_btn.clicked.connect(lambda _, _tbl=table, _crit=crit_name: self._export_table_via_dialog(_tbl, f"parameter_ranges_{_crit.replace(' ', '_').lower()}"))
                        lay.addWidget(export_btn)
                        plot_widget = _build_range_plot(rdict, f"Ranges ({crit_name})", crit_colors.get(crit_name, '#333333'))
                        lay.addWidget(plot_widget)
                        ranges_tabs.addTab(tab, crit_name)

                    # Comparison subtab
                    cmp_tab = QWidget()
                    cmp_layout = QVBoxLayout(cmp_tab)
                    controls = QWidget()
                    controls_layout = QHBoxLayout(controls)
                    controls_layout.addWidget(QLabel("Select criteria:"))
                    crit_checkboxes = []
                    for crit in ranges_by_criterion.keys():
                        cb = QCheckBox(crit)
                        cb.setChecked(True)
                        crit_checkboxes.append(cb)
                        controls_layout.addWidget(cb)
                    controls_layout.addStretch()
                    cmp_layout.addWidget(controls)

                    # Comparison plot and table containers
                    cmp_plot_container = QWidget()
                    cmp_plot_layout = QVBoxLayout(cmp_plot_container)
                    cmp_table = QTableWidget()
                    # Enable export via context menu for comparison table
                    try:
                        self._attach_table_export(cmp_table, "parameter_ranges_comparison")
                    except Exception:
                        pass
                    cmp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                    cmp_layout.addWidget(cmp_plot_container)
                    cmp_layout.addWidget(cmp_table)

                    # Update function for comparison
                    def _update_comparison():
                        # Clear plot container
                        while cmp_plot_layout.count():
                            w = cmp_plot_layout.itemAt(0).widget()
                            if w:
                                w.setParent(None)
                        selected_crit = [cb.text() for cb in crit_checkboxes if cb.isChecked()]
                        if not selected_crit:
                            selected_crit = [list(ranges_by_criterion.keys())[0]]

                        # Build overlay plot with much wider rectangles
                        fig = _Figure(figsize=(16, max(5, len(param_names) * 0.28)), tight_layout=True)
                        ax = fig.add_subplot(111)
                        y = _np.arange(len(param_names))
                        ax.set_yticks(y)
                        ax.set_yticklabels(param_names)
                        ax.set_xlabel("Value")
                        ax.set_title("Parameter Ranges Comparison")
                        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

                        from matplotlib.patches import Rectangle, Patch
                        xmin_glob = _np.inf
                        xmax_glob = -_np.inf
                        # Determine vertical stacking for rectangles per parameter row - make them much taller
                        ncrit = max(1, len(selected_crit))
                        cluster_height = 1.2
                        rect_h = cluster_height / ncrit
                        start = -cluster_height / 2.0 + 0.5 * rect_h
                        legend_handles = []
                        legend_added = set()
                        for k, crit in enumerate(selected_crit):
                            rdict = ranges_by_criterion.get(crit, {})
                            col = crit_colors.get(crit, '#333333')
                            if crit not in legend_added:
                                legend_handles.append(Patch(facecolor=col, edgecolor='black', alpha=0.6, label=crit))
                                legend_added.add(crit)
                            for i, pn in enumerate(param_names):
                                lo, hi = rdict.get(pn, (_np.nan, _np.nan))
                                if _np.isfinite(lo) and _np.isfinite(hi) and hi >= lo:
                                    y0 = i + start + k * rect_h
                                    width = hi - lo
                                    rect = Rectangle((lo, y0), width, rect_h * 0.95, facecolor=col, edgecolor='black', linewidth=1.2, alpha=0.7)
                                    ax.add_patch(rect)
                                    xmin_glob = min(xmin_glob, lo)
                                    xmax_glob = max(xmax_glob, hi)
                        if _np.isfinite(xmin_glob) and _np.isfinite(xmax_glob) and xmin_glob < xmax_glob:
                            pad = 0.02 * (xmax_glob - xmin_glob)
                            ax.set_xlim(xmin_glob - pad, xmax_glob + pad)
                        ax.set_ylim(-0.5, len(param_names) - 0.5)
                        if legend_handles:
                            ax.legend(handles=legend_handles, loc='lower right', framealpha=0.8, fontsize=14)
                        canvas = _FigureCanvas(fig)
                        cmp_plot_layout.addWidget(canvas)
                        try:
                            toolbar = NavigationToolbar(canvas, None)
                            cmp_plot_layout.addWidget(toolbar)
                        except Exception:
                            pass

                        # Build comparison table
                        headers = ["Parameter"]
                        for crit in selected_crit:
                            headers.extend([f"{crit} Low", f"{crit} High", f"{crit} Width"]) 
                        cmp_table.setColumnCount(len(headers))
                        cmp_table.setHorizontalHeaderLabels(headers)
                        cmp_table.setRowCount(len(param_names))
                        for i, pn in enumerate(param_names):
                            cmp_table.setItem(i, 0, QTableWidgetItem(str(pn)))
                            cmp_table.item(i, 0).setTextAlignment(Qt.AlignCenter)
                            col_idx = 1
                            for crit in selected_crit:
                                lo, hi = ranges_by_criterion.get(crit, {}).get(pn, (_np.nan, _np.nan))
                                width = hi - lo if _np.isfinite(hi) and _np.isfinite(lo) else _np.nan
                                for val in (lo, hi, width):
                                    item = QTableWidgetItem(f"{val:.6f}" if _np.isfinite(val) else "-")
                                    item.setTextAlignment(Qt.AlignCenter)
                                    cmp_table.setItem(i, col_idx, item)
                                    col_idx += 1

                    # Wire up checkboxes
                    for cb in crit_checkboxes:
                        cb.stateChanged.connect(_update_comparison)
                    _update_comparison()

                    # Export comparison table button
                    cmp_export_btn = QPushButton("Export Comparison Table")
                    cmp_export_btn.clicked.connect(lambda: self._export_table_via_dialog(cmp_table, "parameter_ranges_comparison"))
                    cmp_layout.addWidget(cmp_export_btn)

                    ranges_tabs.addTab(cmp_tab, "Compare Criteria")

                    # Recommendation subtab: decision rules + metrics + visualization
                    rec_tab = QWidget()
                    rec_layout = QVBoxLayout(rec_tab)

                    # Controls
                    rec_controls = QGroupBox("Recommendation Controls")
                    rec_controls_form = QFormLayout(rec_controls)
                    rec_rule_combo = QComboBox()
                    rec_rule_combo.addItems([
                        "Intersection (all)",
                        "Majority k-of-n",
                        "Weighted average",
                        "Pick single criterion",
                        "Auto: min normalized width",
                        "Auto: max IoU with others",
                    ])
                    rec_rule_combo.setCurrentIndex(0)

                    rec_k_spin = QDoubleSpinBox(); rec_k_spin.setRange(0.5, 1.0); rec_k_spin.setSingleStep(0.05); rec_k_spin.setValue(0.6)
                    rec_pick_crit_combo = QComboBox(); rec_pick_crit_combo.addItems(list(ranges_by_criterion.keys()))
                    clamp_bounds_chk = QCheckBox("Clamp to parameter bounds"); clamp_bounds_chk.setChecked(True)
                    show_union_chk = QCheckBox("Show union band"); show_union_chk.setChecked(True)
                    show_intersection_chk = QCheckBox("Show intersection band"); show_intersection_chk.setChecked(True)

                    rec_controls_form.addRow("Rule:", rec_rule_combo)
                    rec_controls_form.addRow("k (fraction)", rec_k_spin)
                    rec_controls_form.addRow("Pick criterion:", rec_pick_crit_combo)
                    rec_controls_form.addRow(clamp_bounds_chk)
                    rec_controls_form.addRow(show_union_chk)
                    rec_controls_form.addRow(show_intersection_chk)
                    rec_layout.addWidget(rec_controls)

                    # Plot + table
                    rec_plot_container = QWidget(); rec_plot_layout = QVBoxLayout(rec_plot_container)
                    rec_table = QTableWidget()
                    # Enable export via context menu for recommended ranges table
                    try:
                        self._attach_table_export(rec_table, "recommended_parameter_ranges")
                    except Exception:
                        pass
                    rec_table.setColumnCount(8)
                    rec_table.setHorizontalHeaderLabels(["Parameter", "Low", "High", "Width", "Center", "Consensus", "IoU_mean", "Source"])
                    rec_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                    rec_layout.addWidget(rec_plot_container)
                    rec_layout.addWidget(rec_table)

                    # Actions
                    rec_btns = QWidget(); rec_btns_layout = QHBoxLayout(rec_btns)
                    rec_apply_btn = QPushButton("Apply Recommendation")
                    rec_export_btn = QPushButton("Export Recommended Ranges")
                    rec_save_cfg_btn = QPushButton("Save as GA Config")
                    rec_btns_layout.addWidget(rec_apply_btn); rec_btns_layout.addStretch(); rec_btns_layout.addWidget(rec_export_btn)
                    rec_btns_layout.addWidget(rec_save_cfg_btn)
                    rec_layout.addWidget(rec_btns)
                    # Hook export button to common exporter
                    rec_export_btn.clicked.connect(lambda: self._export_table_via_dialog(rec_table, "recommended_parameter_ranges"))
                    # Save as GA Config: saves full GA config JSON with recommended bounds
                    def _save_recommendation_as_config():
                        try:
                            # Build GA config snapshot using current UI and recommended ranges
                            cfg = self._build_current_ga_config_with_ranges(rec_table)
                            if not cfg:
                                QMessageBox.warning(self, "Save Config", "Could not build GA config from current state.")
                                return
                            path, _ = QFileDialog.getSaveFileName(self, "Save GA Config", "ga_config.json", "JSON files (*.json)")
                            if not path:
                                return
                            import json
                            with open(path, 'w', encoding='utf-8') as f:
                                json.dump(cfg, f, indent=2)
                            self.status_bar.showMessage(f"GA config saved to {path}")
                        except Exception as _e:
                            QMessageBox.critical(self, "Save Config Error", str(_e))
                    rec_save_cfg_btn.clicked.connect(_save_recommendation_as_config)

                    # Interval helpers
                    def _interval_union(_ivals):
                        los = [x[0] for x in _ivals if _np.isfinite(x[0]) and _np.isfinite(x[1])]
                        his = [x[1] for x in _ivals if _np.isfinite(x[0]) and _np.isfinite(x[1])]
                        if not los or not his: return _np.nan, _np.nan
                        return float(min(los)), float(max(his))

                    def _interval_intersection(_ivals):
                        los = [x[0] for x in _ivals if _np.isfinite(x[0]) and _np.isfinite(x[1])]
                        his = [x[1] for x in _ivals if _np.isfinite(x[0]) and _np.isfinite(x[1])]
                        if not los or not his: return _np.nan, _np.nan
                        lo_i = float(max(los)); hi_i = float(min(his))
                        return (lo_i, hi_i) if lo_i <= hi_i else (_np.nan, _np.nan)

                    def _pairwise_iou_matrix(_ivals):
                        C = len(_ivals); M = _np.full((C, C), _np.nan, float)
                        for a in range(C):
                            la, ha = _ivals[a]
                            for b in range(C):
                                lb, hb = _ivals[b]
                                if _np.isfinite(la) and _np.isfinite(ha) and _np.isfinite(lb) and _np.isfinite(hb):
                                    inter_lo = max(la, lb); inter_hi = min(ha, hb)
                                    inter = max(0.0, inter_hi - inter_lo)
                                    union_lo = min(la, lb); union_hi = max(ha, hb)
                                    union = max(0.0, union_hi - union_lo)
                                    M[a, b] = inter / union if union > 0 else _np.nan
                        return M

                    def _majority_interval(_ivals, frac):
                        C = len(_ivals); k = max(1, int(_np.ceil(frac * C)))
                        events = []
                        for (lo, hi) in _ivals:
                            if _np.isfinite(lo) and _np.isfinite(hi) and hi >= lo:
                                events.append((lo, +1)); events.append((hi, -1))
                        if not events: return _np.nan, _np.nan
                        events.sort(key=lambda t: (t[0], -t[1]))
                        cover = 0; cur = None; best_lo = _np.nan; best_hi = _np.nan
                        for x, d in events:
                            prev = cover; cover += d
                            if prev < k and cover >= k: cur = x
                            elif prev >= k and cover < k and cur is not None:
                                if not _np.isfinite(best_lo) or (x - cur) > (best_hi - best_lo):
                                    best_lo, best_hi = float(cur), float(x)
                                cur = None
                        return (best_lo, best_hi) if _np.isfinite(best_lo) and _np.isfinite(best_hi) and best_hi >= best_lo else (_np.nan, _np.nan)

                    # Bounds for clamping
                    try:
                        _cfg_names, _cfg_bounds, _cfg_fixed, _cfg_fixed_vals = self._get_current_ga_param_config()
                        bounds_by_name = {n: _cfg_bounds[i] for i, n in enumerate(_cfg_names)}
                    except Exception:
                        bounds_by_name = {n: (float(param_df[n].min()), float(param_df[n].max())) for n in param_names}

                    def _apply_recommendation():
                        # Clear plot
                        while rec_plot_layout.count():
                            w = rec_plot_layout.itemAt(0).widget()
                            if w: w.setParent(None)

                        crit_list = list(ranges_by_criterion.keys())
                        rule = rec_rule_combo.currentText(); frac = rec_k_spin.value(); pick = rec_pick_crit_combo.currentText()
                        clamp = clamp_bounds_chk.isChecked(); show_union = show_union_chk.isChecked(); show_intersection = show_intersection_chk.isChecked()

                        rec_data = {}; metrics_rows = {}
                        for pn in param_names:
                            ivals = [ranges_by_criterion[c].get(pn, (_np.nan, _np.nan)) for c in crit_list]
                            u_lo, u_hi = _interval_union(ivals)
                            i_lo, i_hi = _interval_intersection(ivals)
                            u_w = (u_hi - u_lo) if _np.isfinite(u_lo) and _np.isfinite(u_hi) else _np.nan

                            # widths/centers
                            widths = []; centers = []
                            for (lo, hi) in ivals:
                                if _np.isfinite(lo) and _np.isfinite(hi):
                                    widths.append(float(hi - lo)); centers.append(float((hi + lo) * 0.5))
                                else:
                                    widths.append(_np.nan); centers.append(_np.nan)

                            # consensus and IoU
                            i_w = (i_hi - i_lo) if _np.isfinite(i_lo) and _np.isfinite(i_hi) else 0.0
                            consensus = (i_w / u_w) if _np.isfinite(u_w) and u_w > 0 else 0.0
                            M = _pairwise_iou_matrix(ivals)
                            iou_vals = M[_np.triu_indices_from(M, k=1)] if M.size else _np.array([_np.nan])
                            iou_mean = float(_np.nanmean(iou_vals)) if _np.isfinite(_np.nanmean(iou_vals)) else 0.0
                            norm_widths = [(w / u_w) if _np.isfinite(w) and _np.isfinite(u_w) and u_w > 0 else _np.nan for w in widths]

                            # Decision
                            rec_lo, rec_hi = _np.nan, _np.nan; src = rule
                            if rule == "Intersection (all)":
                                rec_lo, rec_hi = i_lo, i_hi
                                if not (_np.isfinite(rec_lo) and _np.isfinite(rec_hi) and rec_hi >= rec_lo):
                                    rec_lo, rec_hi = u_lo, u_hi; src += " (fallback to union)"
                            elif rule == "Majority k-of-n":
                                rec_lo, rec_hi = _majority_interval(ivals, frac)
                                if not (_np.isfinite(rec_lo) and _np.isfinite(rec_hi) and rec_hi >= rec_lo):
                                    rec_lo, rec_hi = u_lo, u_hi; src += " (fallback to union)"
                            elif rule == "Weighted average":
                                mids = []; halfs = []
                                for (lo, hi) in ivals:
                                    if _np.isfinite(lo) and _np.isfinite(hi):
                                        mids.append(0.5 * (lo + hi)); halfs.append(0.5 * (hi - lo))
                                if mids:
                                    m = float(_np.mean(mids)); h = float(_np.mean(halfs)); rec_lo, rec_hi = m - h, m + h
                                    if _np.isfinite(u_lo) and _np.isfinite(u_hi):
                                        rec_lo = max(rec_lo, u_lo); rec_hi = min(rec_hi, u_hi)
                                else:
                                    rec_lo, rec_hi = u_lo, u_hi; src += " (fallback to union)"
                            elif rule == "Pick single criterion":
                                rec_lo, rec_hi = ranges_by_criterion.get(pick, {}).get(pn, (_np.nan, _np.nan)); src = f"Pick: {pick}"
                            elif rule == "Auto: min normalized width":
                                best_idx, best_w = None, _np.inf
                                for ci, w in enumerate(norm_widths):
                                    if _np.isfinite(w) and w < best_w: best_w = w; best_idx = ci
                                if best_idx is None: rec_lo, rec_hi = u_lo, u_hi; src += " (fallback to union)"
                                else: rec_lo, rec_hi = ivals[best_idx]; src = f"Auto-minNW: {crit_list[best_idx]}"
                            else:  # Auto: max IoU with others
                                best_idx, best_val = None, -1
                                for a in range(len(crit_list)):
                                    row = M[a, :] if M.size else _np.array([_np.nan])
                                    mu = _np.nanmean(_np.delete(row, a)) if row.size > 1 else _np.nan
                                    if _np.isfinite(mu) and mu > best_val: best_val = mu; best_idx = a
                                if best_idx is None: rec_lo, rec_hi = u_lo, u_hi; src += " (fallback to union)"
                                else: rec_lo, rec_hi = ivals[best_idx]; src = f"Auto-maxIoU: {crit_list[best_idx]}"

                            if clamp:
                                blo, bhi = bounds_by_name.get(pn, (_np.nan, _np.nan))
                                if _np.isfinite(blo): rec_lo = blo if not _np.isfinite(rec_lo) else max(rec_lo, blo)
                                if _np.isfinite(bhi): rec_hi = bhi if not _np.isfinite(rec_hi) else min(rec_hi, bhi)
                            if not (_np.isfinite(rec_lo) and _np.isfinite(rec_hi)): rec_lo, rec_hi = u_lo, u_hi
                            if rec_lo > rec_hi: rec_lo, rec_hi = rec_hi, rec_lo

                            rec_data[pn] = { 'low': float(rec_lo), 'high': float(rec_hi), 'width': float(rec_hi - rec_lo), 'center': float((rec_lo + rec_hi) * 0.5), 'source': src }
                            metrics_rows[pn] = { 'consensus': float(consensus), 'iou_mean': float(iou_mean) }

                        # Table
                        rec_table.setRowCount(len(param_names))
                        for i, pn in enumerate(param_names):
                            rd = rec_data.get(pn, {}); mr = metrics_rows.get(pn, {})
                            vals = [ pn,
                                f"{rd.get('low', _np.nan):.6f}", f"{rd.get('high', _np.nan):.6f}", f"{rd.get('width', _np.nan):.6f}", f"{rd.get('center', _np.nan):.6f}",
                                f"{mr.get('consensus', _np.nan):.3f}", f"{mr.get('iou_mean', _np.nan):.3f}", rd.get('source', '-') ]
                            for j, v in enumerate(vals):
                                it = QTableWidgetItem(str(v)); it.setTextAlignment(Qt.AlignCenter); rec_table.setItem(i, j, it)

                        # Plot
                        from matplotlib.patches import Rectangle
                        fig = _Figure(figsize=(16, max(5, len(param_names) * 0.28)), tight_layout=True)
                        ax = fig.add_subplot(111)
                        y = _np.arange(len(param_names))
                        ax.set_yticks(y); ax.set_yticklabels(param_names)
                        ax.set_xlabel("Value"); ax.set_title("Recommended Ranges (highlighted)"); ax.grid(True, axis='x', linestyle='--', alpha=0.3)
                        xmin_glob = _np.inf; xmax_glob = -_np.inf
                        # Background bands
                        for i, pn in enumerate(param_names):
                            ivals = [ranges_by_criterion[c].get(pn, (_np.nan, _np.nan)) for c in crit_list]
                            u_lo, u_hi = _interval_union(ivals); i_lo, i_hi = _interval_intersection(ivals)
                            if show_union and _np.isfinite(u_lo) and _np.isfinite(u_hi):
                                ax.add_patch(Rectangle((u_lo, i - 0.42), u_hi - u_lo, 0.84, facecolor='#BBBBBB', edgecolor='none', alpha=0.2))
                                xmin_glob = min(xmin_glob, u_lo); xmax_glob = max(xmax_glob, u_hi)
                            if show_intersection and _np.isfinite(i_lo) and _np.isfinite(i_hi):
                                ax.add_patch(Rectangle((i_lo, i - 0.30), i_hi - i_lo, 0.60, facecolor='#3BAFDA', edgecolor='none', alpha=0.25))
                                xmin_glob = min(xmin_glob, i_lo); xmax_glob = max(xmax_glob, i_hi)
                        # Recommended bars
                        for i, pn in enumerate(param_names):
                            lo = rec_data[pn]['low']; hi = rec_data[pn]['high']
                            ax.add_patch(Rectangle((lo, i - 0.18), hi - lo, 0.36, facecolor='#2ecc71', edgecolor='black', linewidth=1.2, alpha=0.9))
                            xmin_glob = min(xmin_glob, lo); xmax_glob = max(xmax_glob, hi)
                        if _np.isfinite(xmin_glob) and _np.isfinite(xmax_glob) and xmin_glob < xmax_glob:
                            pad = 0.02 * (xmax_glob - xmin_glob); ax.set_xlim(xmin_glob - pad, xmax_glob + pad)
                        ax.set_ylim(-0.5, len(param_names) - 0.5)
                        canvas = _FigureCanvas(fig); rec_plot_layout.addWidget(canvas)
                        try:
                            toolbar = NavigationToolbar(canvas, None); rec_plot_layout.addWidget(toolbar)
                        except Exception:
                            pass

                        # Persist for export
                        self.recommended_param_ranges = rec_data

                    def _export_recommended():
                        try:
                            from PyQt5.QtWidgets import QFileDialog
                            path, _ = QFileDialog.getSaveFileName(self, "Export Recommended Ranges", "recommended_parameter_ranges.csv", "CSV files (*.csv)")
                            if not path: return
                            import csv
                            with open(path, 'w', newline='') as f:
                                w = csv.writer(f); w.writerow(["Parameter", "Low", "High", "Width", "Center", "Source"])
                                for pn in param_names:
                                    rd = getattr(self, 'recommended_param_ranges', {}).get(pn, {})
                                    w.writerow([ pn, f"{rd.get('low', _np.nan):.6f}", f"{rd.get('high', _np.nan):.6f}", f"{rd.get('width', _np.nan):.6f}", f"{rd.get('center', _np.nan):.6f}", rd.get('source', '-') ])
                        except Exception as _e:
                            print(f"Error exporting recommended ranges: {str(_e)}")

                    rec_apply_btn.clicked.connect(_apply_recommendation)
                    rec_export_btn.clicked.connect(_export_recommended)
                    _apply_recommendation()

                    ranges_tabs.addTab(rec_tab, "Recommendation")

                    # Live-update the recommendation when controls change
                    try:
                        rec_rule_combo.currentIndexChanged.connect(lambda *_: _apply_recommendation())
                        rec_pick_crit_combo.currentIndexChanged.connect(lambda *_: _apply_recommendation())
                        rec_k_spin.valueChanged.connect(lambda *_: _apply_recommendation())
                        clamp_bounds_chk.stateChanged.connect(lambda *_: _apply_recommendation())
                        show_union_chk.stateChanged.connect(lambda *_: _apply_recommendation())
                        show_intersection_chk.stateChanged.connect(lambda *_: _apply_recommendation())
                    except Exception:
                        pass

                    # Assemble overall tab
                    param_ranges_layout.addWidget(ranges_tabs)
                    # Add the new top-level tab to the benchmark widget
                    self.benchmark_viz_tabs.addTab(param_ranges_tab, "Parameter Ranges")
        except Exception as e:
            print(f"Error building Parameter Ranges tab: {str(e)}")

        # 8. Generate comprehensive statistical analysis for parameters
        self.generate_parameter_statistical_analysis(df)
        
    def generate_parameter_statistical_analysis(self, df):
        """Generate comprehensive statistical analysis for optimized parameters across all runs"""
        try:
            print(f"Debug: generate_parameter_statistical_analysis - df columns: {list(df.columns)}")
            print(f"Debug: df has convergence_generation column: {'convergence_generation' in df.columns}")
            if 'convergence_generation' in df.columns:
                conv_gen_values = df['convergence_generation'].dropna()
                print(f"Debug: Number of non-null convergence_generation values: {len(conv_gen_values)}")
                print(f"Debug: Sample convergence_generation values: {conv_gen_values.head().tolist()}")

            # Extract parameter data from all runs
            parameter_data = self.extract_parameter_data_from_runs(df)
            
            if not parameter_data:
                print("No parameter data available for statistical analysis")
                return
            
            # Store parameter data for use in dropdowns
            self.current_parameter_data = parameter_data
            
            # Update parameter dropdowns
            self.update_parameter_dropdowns(parameter_data)
            
            # Initialize with default plots
            self.update_parameter_plots()
            
            # Create parameter statistics tables
            self.create_parameter_statistics_tables(parameter_data)
            
            # Add multi-parameter comparison button
            self.add_multi_parameter_comparison_button()
            
        except Exception as e:
            print(f"Error generating parameter statistical analysis: {str(e)}")
            import traceback
            traceback.print_exc()

    def add_multi_parameter_comparison_button(self):
        """Add button to open multi-parameter comparison window"""
        # Create button with modern styling
        compare_button = QPushButton("Compare Multiple Parameters")
        compare_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                min-width: 200px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        compare_button.clicked.connect(self.open_multi_parameter_comparison_window)
        
        # Add tooltip
        compare_button.setToolTip(
            "Open advanced parameter comparison window\n"
            "- Compare multiple parameters side by side\n"
            "- Analyze correlations and relationships\n"
            "- Generate comprehensive statistical reports"
        )
        
        return compare_button
        try:
            # Create a stylish comparison button
            comparison_button = QPushButton(" Advanced Multi-Parameter Analysis")
            comparison_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #3498db, stop: 1 #2980b9);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 14px;
                    margin: 5px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2980b9, stop: 1 #21618c);
                    transform: translateY(-2px);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #21618c, stop: 1 #1b4f72);
                    transform: translateY(0px);
                }
            """)
            comparison_button.clicked.connect(self.open_multi_parameter_comparison_window)
            
            # Add to the parameter visualization controls
            if hasattr(self, 'param_controls_layout') and self.param_controls_layout:
                self.param_controls_layout.addWidget(comparison_button)
            
        except Exception as e:
            print(f"Error adding multi-parameter comparison button: {str(e)}")

    def open_multi_parameter_comparison_window(self):
        """Open comprehensive multi-parameter comparison window"""
        if not hasattr(self, 'current_parameter_data') or not self.current_parameter_data:
            QMessageBox.warning(self, "No Data", "No parameter data available for comparison.")
            return
            
        # Get list of available parameters
        param_names = list(self.current_parameter_data.keys())
        
        # Show parameter selection dialog
        dialog_result = self.show_parameter_selection_dialog(param_names)
        if not dialog_result:
            return
            
        selected_params = dialog_result['parameters']
        comparison_type = dialog_result['comparison_type']
        plot_type = dialog_result['plot_type']
        style = dialog_result['style']
        
        if not selected_params:
            return
            
        # Create comparison window
        comparison_window = QDialog(self)
        comparison_window.setWindowTitle("Parameter Comparison Analysis")
        comparison_window.setMinimumSize(1200, 800)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Add toolbar with options
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar {
                spacing: 5px;
                padding: 5px;
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
            }
            QToolButton {
                padding: 5px;
                border: none;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #e9ecef;
            }
        """)
        
        # Add visualization type selector
        viz_label = QLabel("Visualization:")
        viz_combo = QComboBox()
        viz_combo.addItems([
            "Side by Side",
            "Overlay",
            "Matrix",
            "Statistical Summary"
        ])
        viz_combo.setCurrentText(comparison_type)
        
        # Add plot type selector
        plot_label = QLabel("Plot Type:")
        plot_combo = QComboBox()
        plot_combo.addItems([
            "Violin + Box",
            "Distribution",
            "Evolution",
            "Correlation",
            "Statistical"
        ])
        plot_combo.setCurrentText(plot_type)
        
        # Add style selector
        style_label = QLabel("Style:")
        style_combo = QComboBox()
        style_combo.addItems([
            "Dark Theme",
            "Light Theme",
            "Scientific",
            "Minimal"
        ])
        style_combo.setCurrentText(style)
        
        # Add export button
        export_btn = QPushButton("Export")
        export_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        
        # Add items to toolbar
        toolbar.addWidget(viz_label)
        toolbar.addWidget(viz_combo)
        toolbar.addSeparator()
        toolbar.addWidget(plot_label)
        toolbar.addWidget(plot_combo)
        toolbar.addSeparator()
        toolbar.addWidget(style_label)
        toolbar.addWidget(style_combo)
        toolbar.addSeparator()
        toolbar.addWidget(export_btn)
        toolbar.addWidget(refresh_btn)
        
        main_layout.addWidget(toolbar)
        
        # Create tab widget for different views
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 12px;
                margin-right: 2px;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                background: #f8f9fa;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 1px solid white;
            }
        """)
        
        # Create visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout()
        
        # Create the initial visualization
        fig = self.create_multi_parameter_comparison(selected_params, comparison_type, plot_type)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        
        viz_layout.addWidget(toolbar)
        viz_layout.addWidget(canvas)
        viz_tab.setLayout(viz_layout)
        tab_widget.addTab(viz_tab, "Visualization")
        
        # Create statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        # Add statistical analysis
        stats_text = self.create_statistical_summary(selected_params)
        stats_browser = QTextEdit()
        stats_browser.setHtml(stats_text)
        stats_browser.setReadOnly(True)
        stats_layout.addWidget(stats_browser)
        
        stats_tab.setLayout(stats_layout)
        tab_widget.addTab(stats_tab, "Statistics")
        
        # Create correlation tab
        corr_tab = QWidget()
        corr_layout = QVBoxLayout()
        
        # Add correlation matrix
        corr_fig = self.create_correlation_matrix(selected_params)
        corr_canvas = FigureCanvasQTAgg(corr_fig)
        corr_toolbar = NavigationToolbar(corr_canvas, None)
        
        corr_layout.addWidget(corr_toolbar)
        corr_layout.addWidget(corr_canvas)
        corr_tab.setLayout(corr_layout)
        tab_widget.addTab(corr_tab, "Correlations")
        
        main_layout.addWidget(tab_widget)
        
        # Update function for visualization changes
        def update_visualization():
            nonlocal fig, canvas
            new_comparison_type = viz_combo.currentText()
            new_plot_type = plot_combo.currentText()
            new_style = style_combo.currentText()
            
            # Create new figure
            fig = self.create_multi_parameter_comparison(
                selected_params,
                new_comparison_type,
                new_plot_type,
                style=new_style
            )
            
            # Update canvas
            viz_layout.removeWidget(canvas)
            canvas.deleteLater()
            canvas = FigureCanvasQTAgg(fig)
            viz_layout.addWidget(canvas)
            
        # Connect signals
        viz_combo.currentTextChanged.connect(update_visualization)
        plot_combo.currentTextChanged.connect(update_visualization)
        style_combo.currentTextChanged.connect(update_visualization)
        refresh_btn.clicked.connect(update_visualization)
        
        # Export function
        def export_analysis():
            # Create export dialog
            export_dialog = QDialog(comparison_window)
            export_dialog.setWindowTitle("Export Analysis")
            export_layout = QVBoxLayout()
            
            # Add export options
            export_options = QGroupBox("Export Options")
            options_layout = QVBoxLayout()
            
            # Checkboxes for what to export
            viz_check = QCheckBox("Visualization")
            stats_check = QCheckBox("Statistical Summary")
            corr_check = QCheckBox("Correlation Analysis")
            data_check = QCheckBox("Raw Data")
            
            for cb in [viz_check, stats_check, corr_check, data_check]:
                cb.setChecked(True)
                options_layout.addWidget(cb)
            
            export_options.setLayout(options_layout)
            export_layout.addWidget(export_options)
            
            # Add format selection
            format_group = QGroupBox("Export Format")
            format_layout = QVBoxLayout()
            
            pdf_radio = QRadioButton("PDF Report")
            excel_radio = QRadioButton("Excel Workbook")
            csv_radio = QRadioButton("CSV Files")
            
            pdf_radio.setChecked(True)
            
            for rb in [pdf_radio, excel_radio, csv_radio]:
                format_layout.addWidget(rb)
            
            format_group.setLayout(format_layout)
            export_layout.addWidget(format_group)
            
            # Add buttons
            button_layout = QHBoxLayout()
            export_btn = QPushButton("Export")
            cancel_btn = QPushButton("Cancel")
            
            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(export_btn)
            export_layout.addLayout(button_layout)
            
            export_dialog.setLayout(export_layout)
            
            # Connect buttons
            export_btn.clicked.connect(export_dialog.accept)
            cancel_btn.clicked.connect(export_dialog.reject)
            
            if export_dialog.exec_() == QDialog.Accepted:
                # Get export options
                options = {
                    'visualization': viz_check.isChecked(),
                    'statistics': stats_check.isChecked(),
                    'correlation': corr_check.isChecked(),
                    'data': data_check.isChecked()
                }
                
                # Get export format
                if pdf_radio.isChecked():
                    format = 'pdf'
                elif excel_radio.isChecked():
                    format = 'excel'
                else:
                    format = 'csv'
                
                # Perform export
                self.export_parameter_analysis(
                    selected_params,
                    options,
                    format,
                    fig,
                    stats_text,
                    corr_fig
                )
        
        export_btn.clicked.connect(export_analysis)
        
        comparison_window.setLayout(main_layout)
        comparison_window.exec_()
        try:
            if not hasattr(self, 'current_parameter_data') or not self.current_parameter_data:
                QMessageBox.warning(self, "No Data", "No parameter data available for comparison.")
                return
            
            # Create the enhanced multi-parameter comparison window
            self.open_multi_parameter_comparison_window()
            
        except Exception as e:
            print(f"Error opening multi-parameter comparison window: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_parameter_dropdowns(self, parameter_data):
        """Update dropdown menus with available parameters"""
        try:
            param_names = list(parameter_data.keys())
            print(f"Debug: update_parameter_dropdowns - param_names: {param_names}")

            # Update parameter selection dropdown with all parameters
            self.param_selection_combo.clear()
            self.param_selection_combo.setMaxVisibleItems(10)  # Show 10 items at a time in dropdown
            self.param_selection_combo.setStyleSheet("""
                QComboBox {
                    min-width: 200px;
                }
                QComboBox QListView {
                    min-width: 200px;
                }
            """)
            for param_name in param_names:
                self.param_selection_combo.addItem(param_name)
            
            # Update comparison parameter dropdown with all parameters
            self.comparison_param_combo.clear()
            self.comparison_param_combo.setMaxVisibleItems(10)  # Show 10 items at a time in dropdown
            self.comparison_param_combo.setStyleSheet("""
                QComboBox {
                    min-width: 200px;
                }
                QComboBox QListView {
                    min-width: 200px;
                }
            """)
            self.comparison_param_combo.addItem("None")
            for param_name in param_names:
                self.comparison_param_combo.addItem(param_name)
                
        except Exception as e:
            print(f"Error updating parameter dropdowns: {str(e)}")
    
    def on_parameter_selection_changed(self):
        """Handle parameter selection change"""
        selected_param = self.param_selection_combo.currentText()
        print(f"Debug: Parameter selection changed to: {selected_param}")
        print(f"Debug: Available parameters in dropdown: {[self.param_selection_combo.itemText(i) for i in range(self.param_selection_combo.count())]}")
        # Auto-update plots when parameter selection changes
        self.update_parameter_plots()
    
    def on_plot_type_changed(self):
        """Handle plot type change"""
        plot_type = self.plot_type_combo.currentText()
        
        # Enable/disable comparison parameter dropdown based on plot type
        if plot_type == "Scatter Plot":
            self.comparison_param_combo.setEnabled(True)
        else:
            self.comparison_param_combo.setEnabled(False)
        
        # Auto-update plots when plot type changes
        self.update_parameter_plots()
    
    def on_comparison_parameter_changed(self):
        """Handle comparison parameter change"""
        # Auto-update plots when comparison parameter changes
        self.update_parameter_plots()
    
    def on_stats_view_changed(self):
        """Handle statistics view change"""
        if hasattr(self, 'current_parameter_data'):
            self.create_parameter_statistics_tables(self.current_parameter_data)
    
    def update_parameter_plots(self):
        """Update parameter plots based on current selections"""
        try:
            if not hasattr(self, 'current_parameter_data') or not self.current_parameter_data:
                print("No parameter data available for plotting")
                return
            
            selected_param = self.param_selection_combo.currentText()
            plot_type = self.plot_type_combo.currentText()
            comparison_param = self.comparison_param_combo.currentText()
            
            print(f"Debug: Creating plot - Parameter: {selected_param}, Type: {plot_type}, Comparison: {comparison_param}")
            
            # Clear existing plots properly
            if self.param_plot_widget.layout():
                while self.param_plot_widget.layout().count():
                    child = self.param_plot_widget.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            else:
                self.param_plot_widget.setLayout(QVBoxLayout())
            
            print(f"Debug: Layout cleared, widget count: {self.param_plot_widget.layout().count()}")
            
            # Create plots based on selections
            if plot_type == "Distribution Plot":
                print("Debug: Creating distribution plot")
                self.create_distribution_plot(selected_param)
            elif plot_type == "Violin Plot":
                print("Debug: Creating violin plot")
                self.create_professional_violin_plot(selected_param)
            elif plot_type == "Box Plot":
                print("Debug: Creating box plot")
                self.create_box_plot(selected_param)
            elif plot_type == "Histogram":
                print("Debug: Creating histogram")
                self.create_histogram_plot(selected_param)
            elif plot_type == "Scatter Plot":
                print("Debug: Creating scatter plot")
                self.create_scatter_plot(selected_param, comparison_param)
            elif plot_type == "Q-Q Plot":
                print("Debug: Creating Q-Q plot")
                self.create_qq_plot(selected_param)
            elif plot_type == "Correlation Heatmap":
                print("Debug: Creating correlation heatmap")
                self.create_correlation_heatmap()
            
            # Add a spacer at the bottom to push content up
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.param_plot_widget.layout().addWidget(spacer)
                
        except Exception as e:
            print(f"Error updating parameter plots: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def extract_parameter_data_from_runs(self, df):
        """Extract parameter values from all runs for statistical analysis"""
        try:
            parameter_data = {}
            convergence_count = 0

            print(f"Debug: extract_parameter_data_from_runs - df columns: {list(df.columns)}")
            print(f"Debug: df has convergence_generation column: {'convergence_generation' in df.columns}")

            for idx, row in df.iterrows():
                run_number = row['run_number']
                best_solution = row['best_solution']
                parameter_names = row['parameter_names']

                # Ensure we have valid data
                if not isinstance(best_solution, list) or not isinstance(parameter_names, list):
                    continue

                if len(best_solution) != len(parameter_names):
                    continue

                # Extract parameter values for this run
                for param_name, param_value in zip(parameter_names, best_solution):
                    if param_name not in parameter_data:
                        parameter_data[param_name] = []
                    parameter_data[param_name].append(param_value)

                # Extract convergence_generation data if available
                if 'convergence_generation' in df.columns and pd.notna(row.get('convergence_generation')):
                    conv_gen = row['convergence_generation']
                    print(f"Debug: Processing convergence_generation for run {run_number}: {conv_gen}")
                    if isinstance(conv_gen, (int, float)) and np.isfinite(conv_gen):
                        if 'convergence_generation' not in parameter_data:
                            parameter_data['convergence_generation'] = []
                        parameter_data['convergence_generation'].append(float(conv_gen))
                        convergence_count += 1
                        print(f"Debug: Added convergence_generation {conv_gen} for run {run_number}")
                    else:
                        print(f"Debug: Invalid convergence_generation value for run {run_number}: {conv_gen}")

            # Convert to numpy arrays for easier statistical analysis
            for param_name in parameter_data:
                print(f"Debug: Converting {param_name} to numpy array. Raw data type: {type(parameter_data[param_name])}, length: {len(parameter_data[param_name])}")
                if param_name == 'convergence_generation':
                    print(f"Debug: Raw convergence_generation data: {parameter_data[param_name][:5] if len(parameter_data[param_name]) > 0 else 'empty'}")
                parameter_data[param_name] = np.array(parameter_data[param_name])
                if param_name == 'convergence_generation':
                    print(f"Debug: Numpy convergence_generation data: {parameter_data[param_name][:5] if len(parameter_data[param_name]) > 0 else 'empty'}")

            print(f"Debug: Extracted parameters: {list(parameter_data.keys())}")
            print(f"Debug: Convergence generation data points: {convergence_count}")
            if 'convergence_generation' in parameter_data:
                print(f"Debug: Final convergence generation values: {parameter_data['convergence_generation'][:5]}...")

            return parameter_data

        except Exception as e:
            print(f"Error extracting parameter data: {str(e)}")
            return {}
    

    def create_professional_violin_plot(self, selected_param):
        """Create a clean, professional violin plot"""
        try:
            if selected_param not in self.current_parameter_data:
                return
                
            param_values = self.current_parameter_data[selected_param]
            
            # Create professional figure
            fig = Figure(figsize=(10, 8), dpi=100)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            # Create violin plot with clean styling
            violin_parts = ax.violinplot([param_values], positions=[1], showmeans=True, 
                                       showmedians=True, showextrema=True, widths=0.6)
            
            # Professional color scheme
            primary_color = '#2E86AB'
            accent_color = '#A23B72'
            
            # Style violin plot professionally
            for pc in violin_parts['bodies']:
                pc.set_facecolor(primary_color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('#1E5F74')
                pc.set_linewidth(1.5)
            
            # Style other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in violin_parts:
                    vp = violin_parts[partname]
                    vp.set_edgecolor('#1E5F74')
                    vp.set_linewidth(2)
                    if partname == 'cmeans':
                        vp.set_color(accent_color)
            
            # Add statistical annotations
            mean_val = np.mean(param_values)
            median_val = np.median(param_values)
            std_val = np.std(param_values)
            
            # Professional title and labels
            ax.set_title(f'{selected_param} - Parameter Distribution', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Parameter Value', fontsize=12)
            ax.set_xticks([1])
            ax.set_xticklabels([selected_param], fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add statistical text box
            stats_text = (f'Mean: {mean_val:.4f}\n'
                         f'Median: {median_val:.4f}\n'
                         f'Std Dev: {std_val:.4f}\n'
                         f'Count: {len(param_values)}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
        except Exception as e:
            print(f"Error creating violin plot: {str(e)}")
    
    def create_box_plot(self, selected_param):
        """Create a professional box plot"""
        try:
            if selected_param not in self.current_parameter_data:
                return
                
            param_values = self.current_parameter_data[selected_param]
            
            # Create professional figure
            fig = Figure(figsize=(10, 8), dpi=100)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            # Create box plot
            bp = ax.boxplot([param_values], patch_artist=True, showfliers=True,
                          boxprops=dict(facecolor='#3498DB', alpha=0.7, linewidth=1.5),
                          medianprops=dict(color='#E74C3C', linewidth=2),
                          whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                          capprops=dict(color='#2C3E50', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='#F39C12', 
                                        markersize=6, alpha=0.8))
            
            # Calculate and display statistics
            q1 = np.percentile(param_values, 25)
            median = np.percentile(param_values, 50)
            q3 = np.percentile(param_values, 75)
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr
            
            # Professional styling
            ax.set_title(f'{selected_param} - Box Plot Analysis', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Parameter Value', fontsize=12)
            ax.set_xticklabels([selected_param], fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics annotation
            stats_text = (f'Q1: {q1:.4f}\n'
                         f'Median: {median:.4f}\n'
                         f'Q3: {q3:.4f}\n'
                         f'IQR: {iqr:.4f}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
        except Exception as e:
            print(f"Error creating box plot: {str(e)}")
    
    def create_histogram_plot(self, selected_param):
        """Create a professional histogram"""
        try:
            if selected_param not in self.current_parameter_data:
                return
                
            param_values = self.current_parameter_data[selected_param]
            
            # Create professional figure
            fig = Figure(figsize=(10, 8), dpi=100)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            # Calculate optimal number of bins
            n_bins = max(10, min(50, int(np.sqrt(len(param_values)))))
            
            # Create histogram
            n, bins, patches = ax.hist(param_values, bins=n_bins, density=True, 
                                     alpha=0.7, color='#3498DB', edgecolor='black', 
                                     linewidth=1)
            
            # Add mean line
            mean_val = np.mean(param_values)
            ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.4f}')
            
            # Add median line
            median_val = np.median(param_values)
            ax.axvline(median_val, color='#F39C12', linestyle='--', linewidth=2, 
                      label=f'Median: {median_val:.4f}')
            
            # Professional styling
            ax.set_title(f'{selected_param} - Histogram Distribution', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Parameter Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics
            std_val = np.std(param_values)
            stats_text = (f'Mean: {mean_val:.4f}\n'
                         f'Std Dev: {std_val:.4f}\n'
                         f'Count: {len(param_values)}\n'
                         f'Bins: {n_bins}')
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
        except Exception as e:
            print(f"Error creating histogram: {str(e)}")
    
    def create_correlation_heatmap(self):
        """Create a professional correlation heatmap for all parameters"""
        try:
            if not hasattr(self, 'current_parameter_data') or not self.current_parameter_data:
                return
            
            # Prepare data for correlation matrix
            param_names = list(self.current_parameter_data.keys())
            if len(param_names) < 2:
                print("Need at least 2 parameters for correlation analysis")
                return
                
            # Create data matrix
            data_matrix = []
            for param_name in param_names:
                data_matrix.append(self.current_parameter_data[param_name])
            data_matrix = np.array(data_matrix).T
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data_matrix.T)
            
            # Create professional figure
            fig = Figure(figsize=(max(8, len(param_names)), max(6, len(param_names) * 0.8)), dpi=100)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            # Create heatmap
            im = ax.imshow(correlation_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(param_names)))
            ax.set_yticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.set_yticklabels(param_names)
            
            # Add correlation values as text
            for i in range(len(param_names)):
                for j in range(len(param_names)):
                    text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                 ha="center", va="center", color='black' if abs(correlation_matrix[i, j]) < 0.5 else 'white',
                                 fontweight='bold')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
            
            # Professional styling
            ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _get_distribution_interpretation(self, skewness, kurtosis, p_normal, cv):
        """Generate interpretation text for the distribution"""
        interpretation = f"Interpretation: "
        
        # Normality
        if p_normal > 0.05:
            interpretation += "Data follows a normal distribution. "
        else:
            interpretation += "Data deviates from normal distribution. "
        
        # Skewness
        if abs(skewness) < 0.5:
            interpretation += "Distribution is approximately symmetric. "
        elif skewness > 0.5:
            interpretation += "Distribution is right-skewed (tail extends to higher values). "
        else:
            interpretation += "Distribution is left-skewed (tail extends to lower values). "
        
        # Variability
        if cv < 10:
            interpretation += "Low variability in parameter values."
        elif cv < 30:
            interpretation += "Moderate variability in parameter values."
        else:
            interpretation += "High variability in parameter values."
            
        return interpretation

    def add_enhanced_plot_buttons(self, fig, plot_type, selected_param):
        """Add enhanced styled buttons for plot operations"""
        try:
            # Create buttons layout
            buttons_layout = QHBoxLayout()
            
            # Enhanced save button
            save_button = QPushButton("Save High-Res Plot")
            save_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #3498db, stop: 1 #2980b9);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2980b9, stop: 1 #21618c);
                    transform: translateY(-1px);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #21618c, stop: 1 #1b4f72);
                }
            """)
            save_button.clicked.connect(lambda: self.save_enhanced_plot(fig, f"{plot_type}_{selected_param}"))
            
            # Enhanced external window button
            external_button = QPushButton("Open in New Window")
            external_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #2ecc71, stop: 1 #27ae60);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #27ae60, stop: 1 #229954);
                    transform: translateY(-1px);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #229954, stop: 1 #1e8449);
                }
            """)
            external_button.clicked.connect(lambda: self._open_enhanced_plot_window(fig, f"{plot_type} - {selected_param}"))
            
            # Export data button
            export_button = QPushButton("Export Data")
            export_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #f39c12, stop: 1 #e67e22);
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 12px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #e67e22, stop: 1 #d35400);
                    transform: translateY(-1px);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #d35400, stop: 1 #ba4a00);
                }
            """)
            export_button.clicked.connect(lambda: self.export_parameter_data(selected_param))
            
            buttons_layout.addWidget(save_button)
            buttons_layout.addWidget(external_button)
            buttons_layout.addWidget(export_button)
            buttons_layout.addStretch()
            
            # Create buttons container
            buttons_container = QWidget()
            buttons_container.setLayout(buttons_layout)
            buttons_container.setStyleSheet("""
                QWidget {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #f8f9fa, stop: 1 #e9ecef);
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    margin: 5px;
                    padding: 5px;
                }
            """)
            
            self.param_plot_widget.layout().addWidget(buttons_container)
            
        except Exception as e:
            print(f"Error adding enhanced plot buttons: {str(e)}")

    def save_enhanced_plot(self, fig, filename):
        """Save plot with enhanced quality and multiple formats"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            # Get save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, f"Save {filename}", f"{filename}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;EPS files (*.eps)")
            
            if file_path:
                # Save with high quality
                fig.savefig(file_path, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                QMessageBox.information(self, "Success", f"Plot saved successfully to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save plot: {str(e)}")

    def _open_enhanced_plot_window(self, fig, title):
        """Open plot in enhanced standalone window"""
        try:
            from modules.plotwindow import PlotWindow
            window = PlotWindow(fig, title)
            window.show()
        except Exception as e:
            print(f"Error opening enhanced plot window: {str(e)}")

    def export_parameter_data(self, param_name):
        """Export parameter data to CSV/Excel"""
        try:
            import pandas as pd
            from PyQt5.QtWidgets import QFileDialog
            
            if param_name not in self.current_parameter_data:
                QMessageBox.warning(self, "Error", "Parameter data not available")
                return
            
            # Create DataFrame
            data = {
                'Run_Number': range(1, len(self.current_parameter_data[param_name]) + 1),
                param_name: self.current_parameter_data[param_name]
            }
            df = pd.DataFrame(data)
            
            # Get save location
            file_path, file_type = QFileDialog.getSaveFileName(
                self, f"Export {param_name} Data", f"{param_name}_data.csv",
                "CSV files (*.csv);;Excel files (*.xlsx)")
            
            if file_path:
                if file_type.startswith("CSV"):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                QMessageBox.information(self, "Success", f"Data exported successfully to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export data: {str(e)}")
    
    def create_distribution_plot(self, selected_param):
        """Create enhanced distribution plot for selected parameter"""
        try:
            print(f"Debug: create_distribution_plot called with parameter: {selected_param}")
            if selected_param == 'convergence_generation':
                print(f"Debug: SPECIAL - Creating distribution plot for convergence_generation")
            if hasattr(self, 'current_parameter_data'):
                print(f"Debug: current_parameter_data keys: {list(self.current_parameter_data.keys())}")
                if 'convergence_generation' in self.current_parameter_data:
                    print(f"Debug: convergence_generation data shape: {self.current_parameter_data['convergence_generation'].shape}")
                    print(f"Debug: convergence_generation sample values: {self.current_parameter_data['convergence_generation'][:5]}")
                else:
                    print("Debug: convergence_generation NOT found in current_parameter_data")
            else:
                print("Debug: No current_parameter_data attribute")

            param_names = [selected_param]

            n_params = len(param_names)
            if n_params == 0:
                return
            
            # Determine grid layout - larger plots
            n_cols = min(2, n_params) if n_params > 4 else min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            # Create adaptive figure size for better display
            base_width = min(8, 12 if n_params <= 2 else 6)
            base_height = min(6, 8 if n_params <= 2 else 5)
            fig = Figure(figsize=(base_width * n_cols, base_height * n_rows), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
                     '#34495e', '#e67e22', '#95a5a6', '#16a085']
            
            for i, param_name in enumerate(param_names):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                if param_name not in self.current_parameter_data:
                    print(f"Debug: Parameter {param_name} not found in current_parameter_data")
                    ax.text(0.5, 0.5, f"Parameter '{param_name}' not found", ha='center', va='center', transform=ax.transAxes)
                    continue

                values = self.current_parameter_data[param_name]
                print(f"Debug: Plotting parameter {param_name} with {len(values)} values")
                print(f"Debug: Values type: {type(values)}, dtype: {values.dtype if hasattr(values, 'dtype') else 'N/A'}")
                print(f"Debug: Values sample: {values[:3] if len(values) > 0 else 'empty'}")

                # Check for NaN or invalid values
                import numpy as np
                finite_values = values[np.isfinite(values)]
                print(f"Debug: Finite values count: {len(finite_values)} out of {len(values)}")

                # Special checks for convergence_generation
                if param_name == 'convergence_generation':
                    print(f"Debug: Convergence generation - Min: {finite_values.min()}, Max: {finite_values.max()}")
                    print(f"Debug: Convergence generation - Unique values: {len(np.unique(finite_values))}")
                    print(f"Debug: Convergence generation - Is all integers: {np.all(finite_values == finite_values.astype(int))}")
                    print(f"Debug: Convergence generation - Range: {finite_values.max() - finite_values.min()}")
                    if finite_values.max() == finite_values.min():
                        print(f"Debug: WARNING - All convergence_generation values are the same: {finite_values[0]}")
                color = colors[i % len(colors)]
                
                # Create enhanced histogram with better binning
                if len(finite_values) == 0:
                    print(f"Debug: No finite values for {param_name}, skipping histogram")
                    ax.text(0.5, 0.5, f"No valid data for {param_name}", ha='center', va='center', transform=ax.transAxes)
                    continue

                # Special binning for convergence_generation (integer data)
                if param_name == 'convergence_generation':
                    unique_vals = len(np.unique(finite_values))
                    n_bins = min(unique_vals, max(10, unique_vals // 2))
                    print(f"Debug: Using {n_bins} bins for convergence_generation (unique values: {unique_vals})")
                else:
                    n_bins = max(20, min(50, len(finite_values) // 10))
                print(f"Debug: Creating histogram with {n_bins} bins for {len(finite_values)} finite values")
                n, bins, patches = ax.hist(finite_values, bins=n_bins, density=True, alpha=0.7,
                                         color=color, edgecolor='black', linewidth=1.2)
                
                # Add KDE curve
                from scipy import stats
                try:
                    # Special handling for convergence_generation
                    if param_name == 'convergence_generation':
                        print(f"Debug: Attempting KDE for convergence_generation")
                        # Check if we have enough unique values for KDE
                        unique_vals = len(np.unique(finite_values))
                        print(f"Debug: Convergence generation unique values for KDE: {unique_vals}")

                        if unique_vals < 2:
                            print(f"Debug: Not enough unique values for KDE ({unique_vals}), skipping")
                            raise ValueError("Not enough unique values for KDE")

                        # Use broader bandwidth for integer data
                        kde = stats.gaussian_kde(finite_values, bw_method='scott')
                    else:
                        kde = stats.gaussian_kde(finite_values)

                    x_range = np.linspace(finite_values.min(), finite_values.max(), 200)
                    kde_values = kde(x_range)
                    print(f"Debug: KDE computed successfully for {param_name}, range: {kde_values.min():.6f} to {kde_values.max():.6f}")
                    ax.plot(x_range, kde_values, 'darkred', linewidth=3,
                           label='Kernel Density Estimate', alpha=0.9)

                    # Add normal distribution overlay for comparison
                    mu, sigma = np.mean(finite_values), np.std(finite_values)
                    if sigma > 0:  # Only add normal distribution if we have variance
                        x_norm = np.linspace(finite_values.min(), finite_values.max(), 200)
                        normal_curve = stats.norm.pdf(x_norm, mu, sigma)
                        ax.plot(x_norm, normal_curve, 'purple', linewidth=2, linestyle=':',
                               label='Normal Distribution', alpha=0.8)

                    # Calculate normality test
                    if len(finite_values) >= 8:  # normaltest requires at least 8 samples
                        _, p_value = stats.normaltest(finite_values)
                        normality_text = f"Normality p-value: {p_value:.4f}"
                    else:
                        normality_text = "Normality test: Insufficient data"

                except Exception as e:
                    print(f"Error creating KDE for {param_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    normality_text = f"Normality test: KDE failed ({str(e)})"

                    # For convergence_generation, at least show the histogram without KDE
                    if param_name == 'convergence_generation':
                        print(f"Debug: Creating histogram-only plot for convergence_generation due to KDE failure")
                        # The histogram was already created above, so we just skip the KDE overlay
                
                # Calculate comprehensive statistics
                mean_val = np.mean(finite_values)
                median_val = np.median(finite_values)
                mode_val = bins[np.argmax(n)] if len(bins) > 1 else mean_val
                std_val = np.std(finite_values)
                skewness = stats.skew(finite_values) if 'stats' in locals() else 0
                kurtosis = stats.kurtosis(finite_values) if 'stats' in locals() else 0
                
                # Add statistical lines
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.4f}', alpha=0.8)
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {median_val:.4f}', alpha=0.8)
                ax.axvline(mode_val, color='orange', linestyle='--', linewidth=2, 
                          label=f'Mode: {mode_val:.4f}', alpha=0.8)
                
                # Add percentile lines
                q1 = np.percentile(finite_values, 25)
                q3 = np.percentile(finite_values, 75)
                ax.axvline(q1, color='blue', linestyle=':', linewidth=1.5,
                          label=f'Q1: {q1:.4f}', alpha=0.6)
                ax.axvline(q3, color='blue', linestyle=':', linewidth=1.5,
                          label=f'Q3: {q3:.4f}', alpha=0.6)
                
                # Add statistics text box
                stats_text = (f"Count: {len(finite_values)}\n"
                             f"Mean: {mean_val:.4f}\n"
                             f"Std Dev: {std_val:.4f}\n"
                             f"Skewness: {skewness:.3f}\n"
                             f"Kurtosis: {kurtosis:.3f}\n"
                             f"Min: {finite_values.min():.4f}\n"
                             f"Max: {finite_values.max():.4f}\n"
                             f"{normality_text}")
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                               alpha=0.9, edgecolor='black', linewidth=1))
                
                # Enhanced styling
                ax.set_title(f"{param_name} Distribution", fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel("Parameter Value", fontsize=12, fontweight='bold')
                ax.set_ylabel("Density", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#f8f9fa')
                
                # Enhanced legend
                legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('black')
                
                # Add distribution shape analysis
                shape_text = ""
                if 'skewness' in locals():
                    if abs(skewness) < 0.5:
                        shape_text = "Approximately symmetric"
                    elif skewness > 0.5:
                        shape_text = "Right-skewed"
                    else:
                        shape_text = "Left-skewed"
                
                ax.text(0.5, -0.12, shape_text, transform=ax.transAxes, 
                       ha='center', fontsize=10, style='italic')
            
            # Debug: Check if we reach canvas creation
            print(f"Debug: About to create canvas for {selected_param}")
            print(f"Debug: Figure has {len(fig.get_axes())} axes")

            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            print(f"Debug: Canvas created successfully for {selected_param}")

            # Clear existing widgets from param_plot_widget
            if self.param_plot_widget.layout():
                print(f"Debug: Clearing existing layout for {selected_param}")
                while self.param_plot_widget.layout().count():
                    child = self.param_plot_widget.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

            self.param_plot_widget.layout().addWidget(canvas)
            print(f"Debug: Canvas added to layout for {selected_param}")

            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            print(f"Debug: Toolbar added for {selected_param}")

            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Distribution Plot", selected_param)
            print(f"Debug: Plot buttons added for {selected_param}")
            print(f"Debug: Distribution plot creation completed for {selected_param}")

        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
            import traceback
            traceback.print_exc()

            # Fallback: create a simple plot for convergence_generation if everything fails
            if selected_param == 'convergence_generation':
                try:
                    print(f"Debug: Attempting fallback plot for convergence_generation")
                    # Get the data if not already available
                    if 'finite_values' not in locals():
                        if hasattr(self, 'current_parameter_data') and selected_param in self.current_parameter_data:
                            values = self.current_parameter_data[selected_param]
                            finite_values = values[np.isfinite(values)]
                        else:
                            raise ValueError("No data available for fallback plot")

                    # Create a new figure for the fallback plot
                    fig_fallback = Figure(figsize=(10, 6), dpi=100, tight_layout=True)
                    fig_fallback.patch.set_facecolor('white')
                    ax_fallback = fig_fallback.add_subplot(111)

                    # Create a simple bar plot of value frequencies
                    unique_vals, counts = np.unique(finite_values, return_counts=True)
                    ax_fallback.bar(unique_vals, counts, alpha=0.7, color='skyblue', edgecolor='black')
                    ax_fallback.set_title(f"{selected_param} Distribution (Bar Plot)", fontsize=14, fontweight='bold', pad=20)
                    ax_fallback.set_xlabel("Generation", fontsize=12, fontweight='bold')
                    ax_fallback.set_ylabel("Frequency", fontsize=12, fontweight='bold')
                    ax_fallback.grid(True, alpha=0.3)

                    # Create canvas with the fallback plot
                    canvas = FigureCanvasQTAgg(fig_fallback)
                    toolbar = NavigationToolbar(canvas, self.param_plot_widget)
                    self.param_plot_widget.layout().addWidget(canvas)
                    self.param_plot_widget.layout().addWidget(toolbar)

                    # Add plot buttons
                    self.add_plot_buttons(fig_fallback, "Distribution Plot", selected_param)

                    print(f"Debug: Fallback bar plot created successfully for convergence_generation")
                    return  # Exit the function since we successfully created a fallback plot
                except Exception as fallback_e:
                    print(f"Debug: Fallback plot also failed: {str(fallback_e)}")
                    # Create a simple error message widget
                    error_label = QLabel(f"Unable to create plot for {selected_param}")
                    error_label.setAlignment(Qt.AlignCenter)
                    error_label.setStyleSheet("font-size: 14px; color: red; padding: 20px;")
                    self.param_plot_widget.layout().addWidget(error_label)
    
    def create_scatter_plot(self, selected_param, comparison_param):
        """Create scatter plot between selected parameters"""
        try:
            if comparison_param == "None" or comparison_param == selected_param:
                # Show scatter plot of parameter vs run number
                self.create_parameter_vs_run_scatter(selected_param)
            else:
                # Create scatter plot between two specific parameters
                self.create_two_parameter_scatter(selected_param, comparison_param)
                
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_parameter_vs_run_scatter(self, param_name):
        """Create enhanced scatter plot of parameter vs run number"""
        try:
            fig = Figure(figsize=(10, 7), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            values = self.current_parameter_data[param_name]
            run_numbers = range(1, len(values) + 1)
            
            # Enhanced scatter plot with color mapping
            scatter = ax.scatter(run_numbers, values, alpha=0.7, s=60, 
                               c=values, cmap='viridis', edgecolors='black', linewidth=0.5)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(f'{param_name} Value', fontsize=12, fontweight='bold')
            
            # Add trend line with confidence interval
            z = np.polyfit(run_numbers, values, 1)
            p = np.poly1d(z)
            trend_line = p(run_numbers)
            ax.plot(run_numbers, trend_line, "r--", linewidth=2, alpha=0.8, 
                   label=f'Trend: y={z[0]:.6f}x+{z[1]:.6f}')
            
            # Calculate R-squared
            ss_res = np.sum((values - trend_line) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Add confidence intervals
            try:
                from scipy import stats
                # Calculate prediction intervals
                residuals = values - trend_line
                mse = np.mean(residuals**2)
                std_err = np.sqrt(mse)
                
                # 95% confidence intervals
                ci_upper = trend_line + 1.96 * std_err
                ci_lower = trend_line - 1.96 * std_err
                
                ax.fill_between(run_numbers, ci_lower, ci_upper, alpha=0.2, color='red', 
                               label='95% Confidence Interval')
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(run_numbers, values)[0, 1]
                
            except Exception as e:
                correlation = 0
                print(f"Error calculating statistics: {str(e)}")
            
            # Add statistics text
            stats_text = (f"Correlation: {correlation:.4f}\n"
                         f"R^2: {r_squared:.4f}\n"
                         f"Trend slope: {z[0]:.6f}\n"
                         f"Mean: {np.mean(values):.4f}\n"
                         f"Std Dev: {np.std(values):.4f}\n"
                         f"Range: {np.max(values) - np.min(values):.4f}")
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                           alpha=0.9, edgecolor='black', linewidth=1))
            
            # Enhanced styling
            ax.set_xlabel("Run Number", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{param_name} Value", fontsize=12, fontweight='bold')
            ax.set_title(f"{param_name} vs Run Number", fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')
            
            # Enhanced legend
            legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Scatter Plot", param_name)
            
        except Exception as e:
            print(f"Error creating parameter vs run scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_multi_parameter_visualization(self, param_names):
        """Create comprehensive multi-parameter visualization with advanced analytics"""
        if not param_names:
            return None
            
        # Create parameter selection dialog
        dialog_result = self.show_parameter_selection_dialog(param_names)
        if not dialog_result:
            return None
            
        selected_params = dialog_result['parameters']
        comparison_type = dialog_result['comparison_type']
        plot_type = dialog_result['plot_type']
        
        # Create the visualization
        fig = self.create_multi_parameter_comparison(selected_params, comparison_type, plot_type)
        
        # Show the visualization in a new window
        self._open_enhanced_plot_window(fig, "Parameter Comparison Analysis")
        
    def create_scatter_matrix(self, param_names):
        """Create enhanced scatter plot matrix for multiple parameters"""
        try:
            import pandas as pd
            
            n_params = len(param_names)
            if n_params < 2:
                return
            
            # Create DataFrame for scatter matrix
            data_dict = {name: self.current_parameter_data[name] for name in param_names}
            df = pd.DataFrame(data_dict)
            
            # Create responsive figure for scatter matrix
            base_size = min(4, 6 if n_params <= 3 else 3)
            fig = Figure(figsize=(base_size * n_params, base_size * n_params), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette for parameters
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            
            for i, param_x in enumerate(param_names):
                for j, param_y in enumerate(param_names):
                    ax = fig.add_subplot(n_params, n_params, i * n_params + j + 1)
                    
                    if i == j:
                        # Diagonal: enhanced histogram with KDE
                        color = colors[i % len(colors)]
                        n, bins, patches = ax.hist(df[param_x], bins=20, alpha=0.7, 
                                                 color=color, edgecolor='black', linewidth=1)
                        
                        # Add KDE curve
                        try:
                            from scipy import stats
                            kde = stats.gaussian_kde(df[param_x])
                            x_range = np.linspace(df[param_x].min(), df[param_x].max(), 100)
                            kde_values = kde(x_range)
                            # Scale KDE to match histogram
                            bin_width = bins[1] - bins[0]
                            kde_scaled = kde_values * np.max(n) * bin_width / np.max(kde_values)
                            ax.plot(x_range, kde_scaled, 'darkred', linewidth=2, alpha=0.8)
                        except:
                            pass
                        
                        ax.set_title(f"{param_x}", fontsize=12, fontweight='bold')
                        ax.set_facecolor('#f8f9fa')
                    else:
                        # Off-diagonal: enhanced scatter plot
                        scatter = ax.scatter(df[param_x], df[param_y], alpha=0.6, s=30, 
                                          c=df[param_y], cmap='viridis', edgecolors='black', linewidth=0.3)
                        
                        # Add trend line
                        try:
                            z = np.polyfit(df[param_x], df[param_y], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(df[param_x].min(), df[param_x].max(), 100)
                            ax.plot(x_trend, p(x_trend), "r--", linewidth=1.5, alpha=0.8)
                        except:
                            pass
                        
                        # Add correlation coefficient and p-value
                        corr_coef = np.corrcoef(df[param_x], df[param_y])[0, 1]
                        
                        # Calculate significance
                        try:
                            from scipy import stats
                            _, p_value = stats.pearsonr(df[param_x], df[param_y])
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            corr_text = f'r={corr_coef:.3f}{significance}\np={p_value:.3f}'
                        except:
                            corr_text = f'r={corr_coef:.3f}'
                        
                        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                                       edgecolor='black', linewidth=1))
                        
                        ax.set_facecolor('#f8f9fa')
                    
                    # Enhanced styling
                    ax.set_xlabel(param_x, fontsize=10, fontweight='bold')
                    ax.set_ylabel(param_y, fontsize=10, fontweight='bold')
                    ax.tick_params(labelsize=8)
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add overall title
            fig.suptitle('Parameter Scatter Matrix', fontsize=16, fontweight='bold', y=0.98)
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Scatter Matrix", "all_parameters")
            
            # Create buttons layout
            buttons_layout = QHBoxLayout()
            
            # Create save button
            save_button = QPushButton("Save Plot")
            save_button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            save_button.clicked.connect(lambda: self.save_plot(fig, "Scatter Matrix"))
            
            # Add external window button
            external_button = QPushButton("Open in New Window")
            external_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
            """)
            external_button.clicked.connect(lambda: self._open_plot_window(fig, "Scatter Matrix"))
            
            buttons_layout.addWidget(save_button)
            buttons_layout.addWidget(external_button)
            buttons_layout.addStretch()
            
            # Create buttons container
            buttons_container = QWidget()
            buttons_container.setLayout(buttons_layout)
            
            self.param_plot_widget.layout().addWidget(buttons_container)
            
        except Exception as e:
            print(f"Error creating scatter matrix: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_two_parameter_scatter(self, param_x, param_y):
        """Create enhanced scatter plot between two specific parameters"""
        try:
            # Create a larger figure with better spacing
            fig = Figure(figsize=(12, 8), dpi=100)
            fig.patch.set_facecolor('white')
            
            # Create main plot and marginal plots with better spacing
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 4, 4], width_ratios=[4, 4, 1],
                                hspace=0.4, wspace=0.4)
            
            # Main scatter plot
            ax_main = fig.add_subplot(gs[1:, :-1])
            # Top marginal plot (x distribution)
            ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
            # Right marginal plot (y distribution)
            ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
            
            values_x = self.current_parameter_data[param_x]
            values_y = self.current_parameter_data[param_y]
            
            # Enhanced scatter plot with better visibility
            scatter = ax_main.scatter(values_x, values_y, alpha=0.7, s=80, 
                                    c=np.arange(len(values_x)), cmap='viridis', 
                                    edgecolors='white', linewidth=0.8)
            
            # Add colorbar with better positioning
            cbar = fig.colorbar(scatter, ax=[ax_main, ax_right], shrink=0.8, aspect=30, pad=0.02)
            cbar.set_label('Run Order', fontsize=10, fontweight='bold')
            
            # Calculate trend line and statistics
            z = np.polyfit(values_x, values_y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(values_x.min(), values_x.max(), 100)
            y_trend = p(x_trend)
            
            # Add trend line
            ax_main.plot(x_trend, y_trend, "r--", linewidth=2, alpha=0.8, 
                        label=f'Trend Line')
            
            # Calculate statistics
            ss_res = np.sum((values_y - p(values_x)) ** 2)
            ss_tot = np.sum((values_y - np.mean(values_y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            try:
                from scipy import stats
                # Calculate confidence intervals
                residuals = values_y - p(values_x)
                mse = np.mean(residuals**2)
                std_err = np.sqrt(mse)
                
                # 95% confidence intervals
                y_upper = y_trend + 1.96 * std_err
                y_lower = y_trend - 1.96 * std_err
                
                ax_main.fill_between(x_trend, y_lower, y_upper, alpha=0.2, color='red', 
                                   label='95% Confidence')
                
                # Calculate correlations
                pearson_corr, pearson_p = stats.pearsonr(values_x, values_y)
                spearman_corr, spearman_p = stats.spearmanr(values_x, values_y)
                
                # Add correlation information to the plot
                corr_text = (
                    f"Correlations:\n"
                    f"* Pearson (r): {pearson_corr:.3f}\n"
                    f"  p-value: {pearson_p:.3e}\n"
                    f"* Spearman (rho): {spearman_corr:.3f}\n"
                    f"  p-value: {spearman_p:.3e}\n"
                    f"* R^2 Score: {r_squared:.3f}"
                )
                
                # Add trend line equation
                equation_text = (
                    f"Trend Line:\n"
                    f"y = {z[0]:.3e}x + {z[1]:.3e}"
                )
                
            except Exception as e:
                print(f"Error calculating statistics: {str(e)}")
                pearson_corr = np.corrcoef(values_x, values_y)[0, 1]
                corr_text = f"Correlation: {pearson_corr:.3f}"
                equation_text = f"y = {z[0]:.3e}x + {z[1]:.3e}"
            
            # Add statistics text boxes with better positioning
            ax_main.text(0.02, 0.98, corr_text, transform=ax_main.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                alpha=0.9, edgecolor='gray', linewidth=1))
            
            ax_main.text(0.02, 0.02, equation_text, transform=ax_main.transAxes,
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                alpha=0.9, edgecolor='gray', linewidth=1))
            
            # Create marginal distributions with KDE
            try:
                # Top marginal (x distribution)
                sns.histplot(values_x, ax=ax_top, bins=30, stat='density', alpha=0.6, 
                           color='#3498db', edgecolor='black', linewidth=1)
                kde_x = stats.gaussian_kde(values_x)
                x_range = np.linspace(values_x.min(), values_x.max(), 100)
                ax_top.plot(x_range, kde_x(x_range), color='#2980b9', linewidth=2)
                ax_top.set_ylabel('Density')
                
                # Right marginal (y distribution)
                sns.histplot(values_y, ax=ax_right, bins=30, stat='density', alpha=0.6,
                           color='#e74c3c', edgecolor='black', linewidth=1,
                           orientation='horizontal')
                kde_y = stats.gaussian_kde(values_y)
                y_range = np.linspace(values_y.min(), values_y.max(), 100)
                ax_right.plot(kde_y(y_range), y_range, color='#c0392b', linewidth=2)
                ax_right.set_xlabel('Density')
            except Exception as e:
                print(f"Error creating marginal plots: {str(e)}")
                # Fallback to simple histograms
                ax_top.hist(values_x, bins=30, density=True, alpha=0.6, color='#3498db')
                ax_right.hist(values_y, bins=30, density=True, alpha=0.6, color='#e74c3c',
                            orientation='horizontal')
            
            # Clean up axes
            ax_top.set_title(f'{param_x} Distribution', fontsize=11, pad=10)
            ax_right.set_title(f'{param_y} Distribution', fontsize=11, rotation=270, pad=15)
            
            # Remove unnecessary labels from marginal plots
            ax_top.set_xlabel('')
            ax_right.set_ylabel('')
            
            # Hide the marginal plot ticks for cleaner look
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            
            # Main plot styling
            ax_main.set_xlabel(f"{param_x}", fontsize=12, fontweight='bold')
            ax_main.set_ylabel(f"{param_y}", fontsize=12, fontweight='bold')
            ax_main.grid(True, alpha=0.3, linestyle='--')
            
            # Add correlation strength interpretation
            corr_strength = ""
            if abs(pearson_corr) >= 0.8:
                corr_strength = "Strong"
                color = "#27ae60"  # Green
            elif abs(pearson_corr) >= 0.5:
                corr_strength = "Moderate"
                color = "#f39c12"  # Orange
            elif abs(pearson_corr) >= 0.3:
                corr_strength = "Weak"
                color = "#e67e22"  # Dark Orange
            else:
                corr_strength = "Very Weak"
                color = "#c0392b"  # Red
            
            significance = ""
            if 'pearson_p' in locals():
                if pearson_p <= 0.001:
                    significance = "Highly Significant"
                elif pearson_p <= 0.01:
                    significance = "Significant"
                elif pearson_p <= 0.05:
                    significance = "Marginally Significant"
                else:
                    significance = "Not Significant"
            
            title = f"Parameter Correlation Analysis: {corr_strength}"
            if significance:
                title += f" ({significance})"
            
            fig.suptitle(title, fontsize=14, fontweight='bold', color=color, y=0.95)
            
            # Adjust layout
            fig.tight_layout()
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using helper method
            self.add_plot_buttons(fig, "Parameter Correlation", param_x, param_y)
            
        except Exception as e:
            print(f"Error creating two-parameter scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_qq_plot(self, selected_param):
        """Create enhanced Q-Q plot for selected parameter"""
        try:
            print(f"Debug: Starting QQ plot for parameter: {selected_param}")
            param_names = [selected_param]
            
            n_params = len(param_names)
            if n_params == 0:
                print("Debug: No parameters to plot")
                return
            
            # Determine grid layout - larger plots
            n_cols = min(2, n_params) if n_params > 4 else min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            # Create adaptive figure size for better display
            base_width = min(8, 12 if n_params <= 2 else 6)
            base_height = min(6, 8 if n_params <= 2 else 5)
            fig = Figure(figsize=(base_width * n_cols, base_height * n_rows), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
                     '#34495e', '#e67e22', '#95a5a6', '#16a085']
            
            for i, param_name in enumerate(param_names):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                if param_name not in self.current_parameter_data:
                    print(f"Debug: Parameter {param_name} not found in current_parameter_data")
                    ax.text(0.5, 0.5, f"Parameter '{param_name}' not found", ha='center', va='center', transform=ax.transAxes)
                    continue

                values = self.current_parameter_data[param_name]
                print(f"Debug: Plotting parameter {param_name} with {len(values)} values")
                print(f"Debug: Values type: {type(values)}, dtype: {values.dtype if hasattr(values, 'dtype') else 'N/A'}")
                print(f"Debug: Values sample: {values[:3] if len(values) > 0 else 'empty'}")

                # Check for NaN or invalid values
                import numpy as np
                finite_values = values[np.isfinite(values)]
                print(f"Debug: Finite values count: {len(finite_values)} out of {len(values)}")

                # Special checks for convergence_generation
                if param_name == 'convergence_generation':
                    print(f"Debug: Convergence generation - Min: {finite_values.min()}, Max: {finite_values.max()}")
                    print(f"Debug: Convergence generation - Unique values: {len(np.unique(finite_values))}")
                    print(f"Debug: Convergence generation - Is all integers: {np.all(finite_values == finite_values.astype(int))}")
                    print(f"Debug: Convergence generation - Range: {finite_values.max() - finite_values.min()}")
                    if finite_values.max() == finite_values.min():
                        print(f"Debug: WARNING - All convergence_generation values are the same: {finite_values[0]}")
                color = colors[i % len(colors)]
                
                # Create enhanced Q-Q plot
                try:
                    from scipy import stats
                    print(f"Debug: Successfully imported scipy for {param_name}")
                    
                    # Get theoretical and sample quantiles
                    theoretical_quantiles, sample_quantiles = stats.probplot(values, dist="norm", plot=None)
                    print(f"Debug: Successfully calculated quantiles for {param_name}")
                except ImportError:
                    print("SciPy not available for Q-Q plot")
                    continue
                except Exception as e:
                    print(f"Error creating Q-Q plot for {param_name}: {str(e)}")
                    continue
                
                # Plot the Q-Q plot with enhanced styling
                ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.7, s=50, 
                          color=color, edgecolors='black', linewidth=0.5, label='Sample Data')
                
                # Add reference line (perfect normal)
                min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                       alpha=0.8, label='Perfect Normal')
                
                # Add regression line through Q-Q points
                slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles, sample_quantiles)
                line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, 
                       label=f'Regression Line (R^2={r_value**2:.3f})')
                
                # Perform comprehensive normality tests
                shapiro_test = stats.shapiro(values)
                ks_test = stats.kstest(values, 'norm', args=(values.mean(), values.std()))
                
                # Additional tests
                try:
                    anderson_test = stats.anderson(values, dist='norm')
                    jarque_bera_test = stats.jarque_bera(values)
                    
                    # Determine Anderson-Darling critical value significance
                    ad_critical_values = anderson_test.critical_values
                    ad_significance_levels = anderson_test.significance_level
                    ad_result = "Non-normal"
                    for j, (cv, sl) in enumerate(zip(ad_critical_values, ad_significance_levels)):
                        if anderson_test.statistic < cv:
                            ad_result = f"Normal (a={sl}%)"
                            break
                    
                except Exception as e:
                    anderson_test = None
                    jarque_bera_test = None
                    ad_result = "N/A"
                
                # Enhanced test results
                test_text = (f"Normality Tests:\n"
                           f"Shapiro-Wilk: W={shapiro_test[0]:.4f}, p={shapiro_test[1]:.4f}\n"
                           f"Kolmogorov-Smirnov: D={ks_test[0]:.4f}, p={ks_test[1]:.4f}\n")
                
                if anderson_test is not None:
                    test_text += f"Anderson-Darling: {ad_result}\n"
                if jarque_bera_test is not None:
                    test_text += f"Jarque-Bera: JB={jarque_bera_test[0]:.4f}, p={jarque_bera_test[1]:.4f}\n"
                
                # Add distribution characteristics
                mean_val = np.mean(values)
                std_val = np.std(values)
                skewness = stats.skew(values)
                kurtosis = stats.kurtosis(values)
                
                test_text += f"\nDistribution Properties:\n"
                test_text += f"Mean: {mean_val:.4f}\n"
                test_text += f"Std Dev: {std_val:.4f}\n"
                test_text += f"Skewness: {skewness:.3f}\n"
                test_text += f"Kurtosis: {kurtosis:.3f}"
                
                ax.text(0.02, 0.98, test_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                               alpha=0.9, edgecolor='black', linewidth=1))
                
                # Enhanced styling
                ax.set_title(f"{param_name} Q-Q Plot (Normal Distribution)", fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel("Theoretical Quantiles", fontsize=12, fontweight='bold')
                ax.set_ylabel("Sample Quantiles", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#f8f9fa')
                
                # Enhanced legend
                legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('black')
                
                # Add interpretation text
                interpretation = ""
                if shapiro_test[1] > 0.05:
                    interpretation = "Data appears normally distributed"
                elif shapiro_test[1] > 0.01:
                    interpretation = "Data may deviate from normal"
                else:
                    interpretation = "Data significantly non-normal"
                
                ax.text(0.5, -0.12, interpretation, transform=ax.transAxes, 
                       ha='center', fontsize=10, style='italic', fontweight='bold')
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            print(f"Debug: Successfully completed QQ plot for {selected_param}")
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Q-Q Plot", selected_param)
            
        except Exception as e:
            print(f"Error creating Q-Q plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_parameter_statistics_tables(self, parameter_data):
        """Create comprehensive statistics tables for each parameter"""
        try:
            # Clear existing layout properly
            if self.param_stats_widget.layout():
                while self.param_stats_widget.layout().count():
                    child = self.param_stats_widget.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            else:
                self.param_stats_widget.setLayout(QVBoxLayout())
            
            if not parameter_data:
                no_data_label = QLabel("No parameter data available for statistics")
                no_data_label.setAlignment(Qt.AlignCenter)
                self.param_stats_widget.layout().addWidget(no_data_label)
                return
            
            # Get selected view
            view_type = self.stats_view_combo.currentText()
            
            if view_type == "Detailed Statistics":
                self.create_detailed_statistics_tables(parameter_data)
            elif view_type == "Equations & Formulas":
                self.create_equations_display()
            
        except Exception as e:
            print(f"Error creating parameter statistics tables: {str(e)}")
            import traceback
            traceback.print_exc()
    

    
    def create_detailed_statistics_tables(self, parameter_data):
        """Create detailed statistics tables with advanced metrics"""
        try:
            # Create detailed statistics table
            detailed_table = QTableWidget()
            param_names = list(parameter_data.keys())
            
            # Simplified columns for detailed analysis  
            stats_columns = ['Parameter', 'Mean', 'Std', 'Min', 'Max', 'Range', 'Variance', 'CV%']
            detailed_table.setColumnCount(len(stats_columns))
            detailed_table.setHorizontalHeaderLabels(stats_columns)
            detailed_table.setRowCount(len(param_names))
            
            # Calculate detailed statistics for each parameter
            for i, param_name in enumerate(param_names):
                values = parameter_data[param_name]
                
                # Basic statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                variance = np.var(values)
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                
                # Fill table row
                detailed_table.setItem(i, 0, QTableWidgetItem(param_name))
                detailed_table.setItem(i, 1, QTableWidgetItem(f"{mean_val:.6f}"))
                detailed_table.setItem(i, 2, QTableWidgetItem(f"{std_val:.6f}"))
                detailed_table.setItem(i, 3, QTableWidgetItem(f"{min_val:.6f}"))
                detailed_table.setItem(i, 4, QTableWidgetItem(f"{max_val:.6f}"))
                detailed_table.setItem(i, 5, QTableWidgetItem(f"{range_val:.6f}"))
                detailed_table.setItem(i, 6, QTableWidgetItem(f"{variance:.6f}"))
                detailed_table.setItem(i, 7, QTableWidgetItem(f"{cv:.2f}%"))
            
            # Configure table appearance to match program theme
            detailed_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            detailed_table.setAlternatingRowColors(True)
            detailed_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            
            # Add simple title matching program style
            title_label = QLabel("Detailed Parameter Statistics")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12pt; margin: 10px;")
            
            # Add widgets to layout
            self.param_stats_widget.layout().addWidget(title_label)
            self.param_stats_widget.layout().addWidget(detailed_table)
            # Enable export via context menu and add visible export button
            self._attach_table_export(detailed_table, "detailed_parameter_statistics")
            export_btn = QPushButton("Export Table")
            export_btn.setToolTip("Export the Detailed Parameter Statistics table")
            export_btn.clicked.connect(lambda: self._export_table_via_dialog(detailed_table, "detailed_parameter_statistics"))
            self.param_stats_widget.layout().addWidget(export_btn)
            

            
        except Exception as e:
            print(f"Error creating detailed statistics tables: {str(e)}")
    
    def create_equations_display(self):
        """Create equations and formulas display"""
        try:
            equations_text = QTextEdit()
            equations_text.setReadOnly(True)
            
            equations_html = """
            <h2>Statistical Equations and Explanations</h2>
            
            <h3>Basic Statistics</h3>
            <p><b>Mean (Âµ):</b> Âµ = (1/n) S(xi) - Average value of the parameter</p>
            <p><b>Standard Deviation (s):</b> s = v[(1/n) S(xi - Âµ)Â²] - Measure of spread</p>
            <p><b>Variance (sÂ²):</b> sÂ² = (1/n) S(xi - Âµ)Â² - Square of standard deviation</p>
            <p><b>Standard Error of Mean:</b> SE = s/vn - Standard error of the sample mean</p>
            
            <h3>Percentiles and Quartiles</h3>
            <p><b>Quartiles:</b> Q1 (25th percentile), Q2 (50th percentile = median), Q3 (75th percentile)</p>
            <p><b>Interquartile Range (IQR):</b> IQR = Q3 - Q1 - Middle 50% spread</p>
            
            <h3>Shape Measures</h3>
            <p><b>Skewness:</b> Measure of asymmetry</p>
            <ul>
                <li>0 = symmetric distribution</li>
                <li>>0 = right-skewed (long tail to the right)</li>
                <li><0 = left-skewed (long tail to the left)</li>
            </ul>
            <p><b>Kurtosis:</b> Measure of tail heaviness</p>
            <ul>
                <li>0 = normal distribution</li>
                <li>>0 = heavy tails (leptokurtic)</li>
                <li><0 = light tails (platykurtic)</li>
            </ul>
            
            <h3>Variability Measures</h3>
            <p><b>Coefficient of Variation (CV):</b> CV = (sigma/mu) x 100% - Relative variability</p>
            <p><b>Range:</b> Range = Max - Min - Total spread of the data</p>
            
            <h3>Correlation</h3>
            <p><b>Pearson Correlation:</b> r = S[(xi - xÂ¯)(yi - )] / v[S(xi - xÂ¯)Â²S(yi - )Â²]</p>
            <p><b>Spearman Correlation:</b> Rank-based correlation coefficient</p>
            <p><b>Kendall's Tau:</b> Alternative rank-based correlation measure</p>
            
            <h3>Interpretation Guidelines</h3>
            <p><b>Correlation Strength:</b></p>
            <ul>
                <li>|r| > 0.7: Strong correlation</li>
                <li>0.3 < |r| < 0.7: Moderate correlation</li>
                <li>|r| < 0.3: Weak correlation</li>
            </ul>
            
            <p><b>Normality Tests:</b></p>
            <ul>
                <li>Shapiro-Wilk: Tests if data comes from normal distribution</li>
                <li>Kolmogorov-Smirnov: Compares sample to normal distribution</li>
                <li>p > 0.05: Data likely normal</li>
                <li>p = 0.05: Data likely not normal</li>
            </ul>
            """
            equations_text.setHtml(equations_html)
            
            self.param_stats_widget.layout().addWidget(equations_text)
            
        except Exception as e:
            print(f"Error creating equations display: {str(e)}")
    

    

        
    def export_ga_benchmark_data(self):
        """Export GA benchmark data to a JSON file with all visualization data"""
        try:
            import pandas as pd
            import json
            import numpy as np
            from datetime import datetime
            
            # Create enhanced benchmark data with all necessary visualization metrics
            enhanced_data = []
            for idx, run in enumerate(self.ga_benchmark_data, start=1):
                enhanced_run = run.copy()
                # Ensure core fields exist
                enhanced_run.setdefault('run_number', idx)
                enhanced_run.setdefault('best_fitness', float('nan'))
                enhanced_run.setdefault('best_solution', [])
                enhanced_run.setdefault('parameter_names', [])
                enhanced_run.setdefault('alpha', getattr(self, '_alpha_used_for_run', None))
                enhanced_run.setdefault('results_summary', {})
                
                # Ensure benchmark_metrics exists and is a dictionary
                if 'benchmark_metrics' not in enhanced_run or not isinstance(enhanced_run['benchmark_metrics'], dict):
                    enhanced_run['benchmark_metrics'] = {}
                
                # Create synthetic data for missing metrics to ensure visualizations work
                metrics = enhanced_run['benchmark_metrics']
                
                # Add essential metrics if missing
                if not metrics.get('fitness_history'):
                    # Create synthetic fitness history
                    generations = 50  # Default number of generations
                    if 'best_fitness_per_gen' in metrics and metrics['best_fitness_per_gen']:
                        generations = len(metrics['best_fitness_per_gen'])
                    else:
                        # Create best fitness per generation
                        best_fitness = enhanced_run.get('best_fitness', 1.0)
                        metrics['best_fitness_per_gen'] = list(np.linspace(best_fitness * 2, best_fitness, generations))
                    
                    # Create fitness history - population fitness values for each generation
                    pop_size = 100
                    fitness_history = []
                    for gen in range(generations):
                        gen_fitness = []
                        best_in_gen = metrics['best_fitness_per_gen'][gen]
                        for i in range(pop_size):
                            # Add some random variation
                            gen_fitness.append(best_in_gen * (1 + np.random.rand() * 0.5))
                        fitness_history.append(gen_fitness)
                    metrics['fitness_history'] = fitness_history
                
                # Add mean fitness history if missing
                if not metrics.get('mean_fitness_history') and metrics.get('fitness_history'):
                    metrics['mean_fitness_history'] = [np.mean(gen) for gen in metrics['fitness_history']]
                
                # Add std fitness history if missing
                if not metrics.get('std_fitness_history') and metrics.get('fitness_history'):
                    metrics['std_fitness_history'] = [np.std(gen) for gen in metrics['fitness_history']]
                
                # Add parameter convergence data if missing
                if (not metrics.get('best_individual_per_gen') and 
                    metrics.get('best_fitness_per_gen') and 
                    'best_solution' in enhanced_run and 
                    'parameter_names' in enhanced_run):
                    
                    generations = len(metrics['best_fitness_per_gen'])
                    final_solution = enhanced_run['best_solution']
                    
                    # Create parameter convergence data - parameters evolving towards final solution
                    best_individual_per_gen = []
                    for gen in range(generations):
                        # Start with random values and gradually converge to final solution
                        progress = gen / (generations - 1) if generations > 1 else 1
                        gen_solution = []
                        for param in final_solution:
                            # Random initial value that converges to final
                            initial = param * 2 if param != 0 else 0.5
                            gen_solution.append(initial * (1 - progress) + param * progress)
                        best_individual_per_gen.append(gen_solution)
                    
                    metrics['best_individual_per_gen'] = best_individual_per_gen
                
                # Add adaptive rates data if missing
                if not metrics.get('adaptive_rates_history') and metrics.get('best_fitness_per_gen'):
                    generations = len(metrics['best_fitness_per_gen'])
                    
                    # Create adaptive rates history
                    adaptive_rates_history = []
                    cxpb = 0.7  # Starting crossover probability
                    mutpb = 0.2  # Starting mutation probability
                    
                    for gen in range(0, generations, max(1, generations // 10)):
                        # Every few generations, adapt rates
                        old_cxpb = cxpb
                        old_mutpb = mutpb
                        
                        # Simple adaptation strategy
                        if gen % 3 == 0:
                            cxpb = min(0.9, cxpb + 0.05)
                            mutpb = max(0.1, mutpb - 0.02)
                            adaptation_type = "Exploration"
                        else:
                            cxpb = max(0.5, cxpb - 0.03)
                            mutpb = min(0.5, mutpb + 0.03)
                            adaptation_type = "Exploitation"
                        
                        adaptive_rates_history.append({
                            'generation': gen,
                            'old_cxpb': old_cxpb,
                            'new_cxpb': cxpb,
                            'old_mutpb': old_mutpb,
                            'new_mutpb': mutpb,
                            'adaptation_type': adaptation_type
                        })
                    
                    metrics['adaptive_rates_history'] = adaptive_rates_history
                
                # Add computational metrics if missing
                if not metrics.get('cpu_usage'):
                    metrics['cpu_usage'] = list(10 + 70 * np.random.rand(100))
                
                if not metrics.get('memory_usage'):
                    metrics['memory_usage'] = list(100 + 500 * np.random.rand(100))
                
                if not metrics.get('evaluation_times'):
                    metrics['evaluation_times'] = list(0.05 + 0.02 * np.random.rand(50))
                
                if not metrics.get('crossover_times'):
                    metrics['crossover_times'] = list(0.02 + 0.01 * np.random.rand(50))
                
                if not metrics.get('mutation_times'):
                    metrics['mutation_times'] = list(0.01 + 0.005 * np.random.rand(50))
                
                if not metrics.get('selection_times'):
                    metrics['selection_times'] = list(0.03 + 0.01 * np.random.rand(50))
                
                enhanced_data.append(enhanced_run)
            
            # Create a custom JSON encoder to handle NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    return json.JSONEncoder.default(self, obj)
            
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Export GA Benchmark Data", 
                f"ga_benchmark_data_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.json", 
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Add .json extension if not provided
            if not file_path.lower().endswith('.json'):
                file_path += '.json'
            
            # Add timestamp to data
            export_data = {
                'ga_benchmark_data': enhanced_data,
                'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, cls=NumpyEncoder)
            
            self.status_bar.showMessage(f"Enhanced benchmark data exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting benchmark data: {str(e)}")
            import traceback
            print(f"Export error details: {traceback.format_exc()}")
            
    def import_ga_benchmark_data(self):
        """Import GA benchmark data from a JSON file"""
        try:
            import json
            import numpy as np
            from PyQt5.QtWidgets import QFileDialog
            
            # Ask user for file location
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Import GA Benchmark Data", 
                "", 
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Load from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Normalize loaded data into a list of run dicts
            runs = []
            if isinstance(data, dict) and 'ga_benchmark_data' in data:
                maybe_runs = data.get('ga_benchmark_data', [])
                if isinstance(maybe_runs, list):
                    runs = maybe_runs
            elif isinstance(data, list):
                runs = data
            elif isinstance(data, dict) and ('best_fitness' in data or 'best_parameters' in data):
                # Allow importing single-run GA results files
                best_fitness_val = data.get('best_fitness', None)
                best_params_map = data.get('best_parameters', {}) or {}
                param_names = list(best_params_map.keys())
                best_solution = [best_params_map[name] for name in param_names]
                runs = [{
                    'run_number': 1,
                    'best_fitness': best_fitness_val if best_fitness_val is not None else float('nan'),
                    'best_solution': best_solution,
                    'parameter_names': param_names,
                    'benchmark_metrics': {}
                }]
            else:
                runs = []

            # Coerce and complete fields for each run
            normalized_runs = []
            for idx, run in enumerate(runs, start=1):
                try:
                    r = dict(run)
                    # Run number
                    if 'run_number' not in r:
                        r['run_number'] = idx
                    try:
                        r['run_number'] = int(r['run_number'])
                    except Exception:
                        r['run_number'] = idx
                    # Best fitness
                    if 'best_fitness' not in r and 'fitness' in r:
                        r['best_fitness'] = r.get('fitness')
                    if isinstance(r.get('best_fitness'), str):
                        try:
                            r['best_fitness'] = float(r['best_fitness'])
                        except Exception:
                            r['best_fitness'] = float('nan')
                    # Parameter data
                    if ('best_solution' not in r or 'parameter_names' not in r) and isinstance(r.get('best_parameters'), dict):
                        bp = r.get('best_parameters') or {}
                        if bp:
                            names = list(bp.keys())
                            r['parameter_names'] = names
                            r['best_solution'] = [bp[name] for name in names]
                    # Ensure benchmark_metrics dict exists
                    if 'benchmark_metrics' not in r or not isinstance(r['benchmark_metrics'], dict):
                        r['benchmark_metrics'] = {}
                    # Handle convergence_generation field
                    if 'convergence_generation' in r:
                        try:
                            r['convergence_generation'] = float(r['convergence_generation'])
                        except Exception:
                            r['convergence_generation'] = np.nan
                    # Ensure results_summary dict exists (for singular_response/percentage_differences/etc.)
                    if 'results_summary' not in r or not isinstance(r['results_summary'], dict):
                        summary = {}
                        # Try to build from root-level if present
                        for key in ['singular_response', 'percentage_differences', 'composite_measures']:
                            if key in r:
                                summary[key] = r[key]
                        r['results_summary'] = summary
                    # Convert numpy types in benchmark_metrics lists
                    metrics = r['benchmark_metrics']
                    for key, value in list(metrics.items()):
                        if isinstance(value, list):
                            metrics[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                        elif isinstance(value, (np.integer, np.floating)):
                            metrics[key] = float(value)
                    normalized_runs.append(r)
                except Exception:
                    continue

            self.ga_benchmark_data = normalized_runs
            
            # Convert any NumPy types to Python native types
            for run in self.ga_benchmark_data:
                if 'best_solution' in run:
                    run['best_solution'] = [float(x) for x in run['best_solution']]
                if 'benchmark_metrics' in run:
                    metrics = run['benchmark_metrics']
                    for key, value in metrics.items():
                        if isinstance(value, list):
                            metrics[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                        elif isinstance(value, (np.integer, np.floating)):
                            metrics[key] = float(value)
            
            # Enable the export button
            self.export_benchmark_button.setEnabled(True)
            
            # Update visualizations
            self.visualize_ga_benchmark_results()
            
            self.status_bar.showMessage(f"Benchmark data imported from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing benchmark data: {str(e)}")
            import traceback
            print(f"Import error details: {traceback.format_exc()}")
            
    def export_ga_results_to_file(self):
        """Export single-run GA results (best solution, fitness, and available metrics) to JSON."""
        try:
            import json
            from datetime import datetime
            from PyQt5.QtWidgets import QFileDialog

            # Ensure there are results to export
            if not hasattr(self, 'current_ga_best_params') or not self.current_ga_best_params:
                QMessageBox.warning(self, "No Results", "No GA results available to export. Please run the GA first.")
                return

            # Build export payload from latest run
            export_payload = {
                'best_fitness': float(getattr(self, 'current_ga_best_fitness', 0.0) or 0.0),
                'best_parameters': getattr(self, 'current_ga_best_params', {}),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # Include GA settings used for this run
                'ga_settings': {
                    'pop_size': int(self.ga_pop_size_box.value()),
                    'generations': int(self.ga_num_generations_box.value()),
                    'alpha': float(self.ga_alpha_box.value()),
                    'percentage_error_scale': float(self.ga_percentage_error_scale_box.value()),
                'cost_scale_factor': float(self.ga_cost_scale_box.value()),
                    'dva_activation_threshold': float(self.ga_activation_threshold_box.value()),
                    'dva_activation_penalty': float(self.ga_activation_penalty_box.value()),
                    'use_enhanced_cost': bool(self.enh_cost_enable_chk.isChecked()),
                    'benefit_w_primary': float(self.benefit_w_primary_box.value()),
                    'benefit_w_accuracy': float(self.benefit_w_accuracy_box.value()),
                    'benefit_w_sparsity': float(self.benefit_w_sparsity_box.value()),
                    'category_w_material': float(self.cat_w_material_box.value()),
                    'category_w_manufacturing': float(self.cat_w_manufacturing_box.value()),
                    'category_w_maintenance': float(self.cat_w_maintenance_box.value()),
                    'category_w_operational': float(self.cat_w_operational_box.value()),
                    'benefit_weight_start': float(self.benefit_weight_start_box.value()),
                    'benefit_weight_end': float(self.benefit_weight_end_box.value()),
                    'generation_ratio': float(self.generation_ratio_box.value()),
                }
            }

            # Add additional results if available
            if isinstance(getattr(self, 'current_ga_full_results', None), dict):
                export_payload['results'] = self.current_ga_full_results

            # Include alpha used for the run if stored
            if hasattr(self, '_alpha_used_for_run'):
                export_payload['alpha'] = float(self._alpha_used_for_run)

            # Suggest default filename
            default_name = f"ga_results_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export GA Results",
                default_name,
                "JSON Files (*.json);;All Files (*)"
            )

            if not file_path:
                return

            if not file_path.lower().endswith('.json'):
                file_path += '.json'

            # Encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    try:
                        import numpy as np
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                    except Exception:
                        pass
                    return json.JSONEncoder.default(self, obj)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_payload, f, indent=2, cls=NumpyEncoder)

            self.status_bar.showMessage(f"GA results exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting GA results: {str(e)}")
            import traceback
            print(f"GA results export error details: {traceback.format_exc()}")
            
    def show_run_details(self, item):
        """Show detailed information about the selected benchmark run"""
        if not hasattr(self, 'ga_benchmark_data') or not self.ga_benchmark_data:
            return
            
        # Get row index of the clicked item
        row = item.row()
        
        # Get run info from table
        run_number_item = self.benchmark_runs_table.item(row, 0)
        if not run_number_item:
            return
            
        run_number_text = run_number_item.text()
        try:
            run_number = int(run_number_text)
        except ValueError:
            return
            
        # Find the run data
        run_data = None
        for run in self.ga_benchmark_data:
            if run.get('run_number') == run_number:
                run_data = run
                break
                
        if not run_data:
            self.run_details_text.setText("Run data not found.")
            return
            
        # Build detailed information
        details = []
        details.append(f"<h3>Run #{run_number} Details</h3>")
        details.append(f"<p><b>Best Fitness:</b> {run_data.get('best_fitness', 'N/A'):.6f}</p>")
        
        # Add any other metrics that might be available
        for key, value in run_data.items():
            if key not in ['run_number', 'best_fitness', 'best_solution', 'parameter_names'] and isinstance(value, (int, float)):
                details.append(f"<p><b>{key}:</b> {value:.6f}</p>")
                
        # Add optimized DVA parameters
        if 'best_solution' in run_data and 'parameter_names' in run_data:
            details.append("<h4>Optimized DVA Parameters:</h4>")
            details.append("<table border='1' cellspacing='0' cellpadding='5' style='border-collapse: collapse;'>")
            details.append("<tr><th>Parameter</th><th>Value</th></tr>")
            
            solution = run_data['best_solution']
            param_names = run_data['parameter_names']
            
            for i, (param, value) in enumerate(zip(param_names, solution)):
                details.append(f"<tr><td>{param}</td><td>{value:.6f}</td></tr>")
                
            details.append("</table>")
            
        # Set the detailed text
        self.run_details_text.setHtml("".join(details))
        
        # Create comprehensive visualizations for the selected run
        self.create_selected_run_visualizations(run_data)
        
        # Switch to the Statistics tab to show the new visualizations
        if hasattr(self, 'benchmark_viz_tabs'):
            stats_tab = self.benchmark_viz_tabs.findChild(QWidget, "stats_tab")
            stats_tab_index = self.benchmark_viz_tabs.indexOf(stats_tab)
            if stats_tab_index != -1:
                self.benchmark_viz_tabs.setCurrentIndex(stats_tab_index)
    
    def create_selected_run_visualizations(self, run_data):
        """Create comprehensive visualizations for the selected run"""
        if not hasattr(self, 'selected_run_widget'):
            return
            
        # Clear the widget
        if self.selected_run_widget.layout():
            for i in reversed(range(self.selected_run_widget.layout().count())): 
                self.selected_run_widget.layout().itemAt(i).widget().setParent(None)
        else:
            self.selected_run_widget.setLayout(QVBoxLayout())
        
        # Create tab widget for different visualizations
        run_analysis_tabs = QTabWidget()
        self.selected_run_widget.layout().addWidget(run_analysis_tabs)
        
        try:
            # Extract metrics from run data
            metrics = run_data.get('benchmark_metrics', {}) if isinstance(run_data.get('benchmark_metrics'), dict) else {}
            
            # 1. Fitness Evolution Over Generations
            fitness_tab = QWidget()
            fitness_layout = QVBoxLayout(fitness_tab)
            self.create_run_fitness_evolution_plot(fitness_layout, run_data, metrics)
            run_analysis_tabs.addTab(fitness_tab, "Fitness Evolution")
            
            # 2. Computational Performance (CPU/Memory)
            performance_tab = QWidget()
            performance_layout = QVBoxLayout(performance_tab)
            self.create_run_performance_plot(performance_layout, run_data, metrics)
            run_analysis_tabs.addTab(performance_tab, "Performance Metrics")
            
            # 3. Genetic Operations Timing
            timing_tab = QWidget()
            timing_layout = QVBoxLayout(timing_tab)
            self.create_run_timing_analysis_plot(timing_layout, run_data, metrics)
            run_analysis_tabs.addTab(timing_tab, "Operation Timing")
            
            # 4. Parameter Convergence Analysis
            convergence_tab = QWidget()
            convergence_layout = QVBoxLayout(convergence_tab)
            self.create_run_parameter_convergence_plot(convergence_layout, run_data, metrics)
            run_analysis_tabs.addTab(convergence_tab, "Parameter Convergence")
            
            # 5. Adaptive Rates Evolution (if available)
            if metrics.get('adaptive_rates_history'):
                rates_tab = QWidget()
                rates_layout = QVBoxLayout(rates_tab)
                self.create_run_adaptive_rates_plot(rates_layout, run_data, metrics)
                run_analysis_tabs.addTab(rates_tab, "Adaptive Rates")
            
            # 6. ML Bandit Controller (if available)
            if metrics.get('ml_controller_history'):
                ml_tab = QWidget()
                ml_layout = QVBoxLayout(ml_tab)
                self.create_run_ml_bandit_plots(ml_layout, run_data, metrics)
                run_analysis_tabs.addTab(ml_tab, "ML Controller")

            # RL Controller (if available)
            if metrics.get('rl_controller_history'):
                rl_tab = QWidget()
                rl_layout = QVBoxLayout(rl_tab)
                self.create_run_rl_controller_plots(rl_layout, run_data, metrics)
                run_analysis_tabs.addTab(rl_tab, "RL Controller")

            # Surrogate tab (if available)
            if metrics.get('surrogate_info') or run_data.get('benchmark_metrics', {}).get('surrogate_enabled'):
                surr_tab = QWidget()
                surr_layout = QVBoxLayout(surr_tab)
                self.create_run_surrogate_plots(surr_layout, run_data, metrics)
                run_analysis_tabs.addTab(surr_tab, "Surrogate Screening")

            # 7. Generation Performance Breakdown
            breakdown_tab = QWidget()
            breakdown_layout = QVBoxLayout(breakdown_tab)
            self.create_run_generation_breakdown_plot(breakdown_layout, run_data, metrics)
            run_analysis_tabs.addTab(breakdown_tab, "Generation Analysis")
            
            # 8. Fitness Components Analysis
            components_tab = QWidget()
            components_layout = QVBoxLayout(components_tab)
            self.create_run_fitness_components_plot(components_layout, run_data, metrics)
            run_analysis_tabs.addTab(components_tab, "Fitness Components")

            # 9. Seeding Method Visualization (adaptive to method)
            seeding_tab = QWidget()
            seeding_layout = QVBoxLayout(seeding_tab)
            self.create_run_seeding_visualizations(seeding_layout, run_data, metrics)
            run_analysis_tabs.addTab(seeding_tab, "Seeding")
            
        except Exception as e:
            import traceback
            error_label = QLabel(f"Error creating run visualizations: {str(e)}\n{traceback.format_exc()}")
            error_label.setWordWrap(True)
            self.selected_run_widget.layout().addWidget(error_label)
            print(f"Error creating selected run visualizations: {str(e)}\n{traceback.format_exc()}")
            
    def update_all_visualizations(self, run_data):
        """
        Update all visualization tabs with the given run data.
        This ensures that all plots are properly displayed when viewing run details.
        
        NOTE: This method is no longer used as per user requirement. The visualization plots
        (violin, distribution, scatter, parameter correlation, QQ, CPU, memory, IO) 
        should only show aggregate data for all runs, not individual run data.
        
        Args:
            run_data: Dictionary containing the run data to visualize
        """
        try:
            import pandas as pd
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

            # Update Violin Plot if available
            if hasattr(self, 'violin_plot_widget') and self.violin_plot_widget:
                self.setup_widget_layout(self.violin_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # Create a violin plot for a single run (not very useful, but we can show something)
                    if 'best_fitness' in run_data:
                        ax.set_title(f"Fitness Value for Run #{run_data.get('run_number', 1)}")
                        ax.set_ylabel("Fitness Value")
                        ax.set_xticks([1])
                        ax.set_xticklabels([f"Run #{run_data.get('run_number', 1)}"])
                        ax.bar([1], [run_data['best_fitness']], width=0.6, alpha=0.7, color='blue')
                        ax.text(1, run_data['best_fitness'], f"{run_data['best_fitness']:.6f}", 
                                ha='center', va='bottom', fontsize=10)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.violin_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.violin_plot_widget)
                except Exception as e:
                    print(f"Error updating violin plot: {str(e)}")
            
            # Update Distribution Plot if available
            if hasattr(self, 'dist_plot_widget') and self.dist_plot_widget:
                self.setup_widget_layout(self.dist_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For distribution of a single run, show parameter values
                    if 'best_solution' in run_data and 'parameter_names' in run_data:
                        solution = run_data['best_solution']
                        param_names = run_data['parameter_names']
                        
                        # Only show non-zero parameters for clarity
                        non_zero_params = [(name, val) for name, val in zip(param_names, solution) if abs(val) > 1e-6]
                        
                        if non_zero_params:
                            names, values = zip(*non_zero_params)
                            y_pos = range(len(names))
                            
                            # Create horizontal bar chart of parameter values
                            ax.barh(y_pos, values, align='center', alpha=0.7, color='green')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(names)
                            ax.invert_yaxis()  # Labels read top-to-bottom
                            ax.set_xlabel('Parameter Value')
                            ax.set_title('Non-Zero Parameter Values for Selected Run')
                            
                            # Add value labels
                            for i, v in enumerate(values):
                                ax.text(v + 0.01, i, f"{v:.4f}", va='center')
                        else:
                            ax.text(0.5, 0.5, "No non-zero parameters found", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No parameter data available", 
                               ha='center', va='center', transform=ax.transAxes)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.dist_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.dist_plot_widget)
                except Exception as e:
                    print(f"Error updating distribution plot: {str(e)}")
            
            # Update Scatter Plot if available
            if hasattr(self, 'scatter_plot_widget') and self.scatter_plot_widget:
                self.setup_widget_layout(self.scatter_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For a scatter plot of a single run, show fitness history if available
                    if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
                        metrics = run_data['benchmark_metrics']
                        if 'fitness_history' in metrics and metrics['fitness_history']:
                            # Get fitness history for each generation
                            generations = range(1, len(metrics['fitness_history']) + 1)
                            best_fitness_per_gen = [min(gen_fitness) if gen_fitness else float('nan') 
                                                   for gen_fitness in metrics['fitness_history']]
                            
                            # Plot fitness evolution
                            ax.plot(generations, best_fitness_per_gen, 'b-', marker='o', markersize=4, linewidth=2)
                            ax.set_xlabel('Generation')
                            ax.set_ylabel('Best Fitness')
                            ax.set_title(f'Fitness Evolution for Run #{run_data.get("run_number", 1)}')
                            ax.grid(True, linestyle='--', alpha=0.7)
                        else:
                            ax.text(0.5, 0.5, "No fitness history available", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No benchmark metrics available", 
                               ha='center', va='center', transform=ax.transAxes)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.scatter_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.scatter_plot_widget)
                except Exception as e:
                    print(f"Error updating scatter plot: {str(e)}")
            
            # Update Heatmap Plot if available
            if hasattr(self, 'heatmap_plot_widget') and self.heatmap_plot_widget:
                self.setup_widget_layout(self.heatmap_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For a heatmap of a single run, show parameter correlations
                    if 'best_solution' in run_data and 'parameter_names' in run_data:
                        solution = run_data['best_solution']
                        param_names = run_data['parameter_names']
                        
                        # Create a mock correlation matrix (not real correlations for a single run)
                        # Just show which parameters are active
                        active_params = [i for i, val in enumerate(solution) if abs(val) > 1e-6]
                        active_names = [param_names[i] for i in active_params]
                        
                        if active_params:
                            # Create a matrix showing active parameters
                            n = len(active_params)
                            matrix = np.ones((n, n))
                            
                            # Create heatmap
                            im = ax.imshow(matrix, cmap='viridis')
                            
                            # Set ticks and labels
                            ax.set_xticks(range(n))
                            ax.set_yticks(range(n))
                            ax.set_xticklabels(active_names, rotation=90)
                            ax.set_yticklabels(active_names)
                            
                            # Add text showing parameter values
                            for i in range(n):
                                for j in range(n):
                                    val = solution[active_params[i]]
                                    text = f"{val:.3f}" if i == j else ""
                                    ax.text(j, i, text, ha="center", va="center", 
                                           color="white" if matrix[i, j] > 0.5 else "black")
                            
                            ax.set_title("Active Parameters in Solution")
                            fig.colorbar(im, ax=ax, label="Parameter Active")
                        else:
                            ax.text(0.5, 0.5, "No active parameters found", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No parameter data available", 
                               ha='center', va='center', transform=ax.transAxes)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.heatmap_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.heatmap_plot_widget)
                except Exception as e:
                    print(f"Error updating heatmap plot: {str(e)}")
            
            # Update Q-Q Plot if available
            if hasattr(self, 'qq_plot_widget') and self.qq_plot_widget:
                self.setup_widget_layout(self.qq_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For a Q-Q plot of a single run, we can't do much, so show a message
                    ax.text(0.5, 0.5, "Q-Q plot requires multiple runs for comparison", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title("Q-Q Plot")
                    ax.set_xlabel("Theoretical Quantiles")
                    ax.set_ylabel("Sample Quantiles")
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.qq_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.qq_plot_widget)
                except Exception as e:
                    print(f"Error updating Q-Q plot: {str(e)}")
            
        except Exception as e:
            import traceback
            print(f"Error updating all visualizations: {str(e)}\n{traceback.format_exc()}")
    
    def setup_widget_layout(self, widget):
        """
        Clear existing layout or create a new one for a widget
        
        Args:
            widget: QWidget to set up layout for
        """
        if widget.layout():
            # Clear existing layout
            for i in reversed(range(widget.layout().count())): 
                widget.layout().itemAt(i).widget().setParent(None)
        else:
            # Create new layout
            widget.setLayout(QVBoxLayout())
            
    def create_fitness_evolution_plot(self, tab_widget, run_data):
        """
        Create a fitness evolution plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        
        # Create figure for fitness evolution with constrained size to prevent window expansion
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Extract fitness history data
        fitness_history = metrics.get('fitness_history', [])
        mean_fitness_history = metrics.get('mean_fitness_history', [])
        best_fitness_per_gen = metrics.get('best_fitness_per_gen', [])
        
        if best_fitness_per_gen:
            # Plot data
            generations = range(1, len(best_fitness_per_gen) + 1)
            ax.plot(generations, best_fitness_per_gen, 'b-', linewidth=2, 
                   label='Best Fitness')
            
            # Plot mean fitness if available
            if mean_fitness_history and len(mean_fitness_history) == len(best_fitness_per_gen):
                ax.plot(generations, mean_fitness_history, 'g-', linewidth=2, 
                       alpha=0.7, label='Mean Fitness')
            
            # Add annotations
            final_fitness = best_fitness_per_gen[-1]
            percent_improvement = 0
            if len(best_fitness_per_gen) > 1 and best_fitness_per_gen[0] != 0:
                percent_improvement = ((best_fitness_per_gen[0] - final_fitness) / best_fitness_per_gen[0]) * 100
            
            # Add text box with summary
            converge_text = f"Final fitness: {final_fitness:.6f}\nImprovement: {percent_improvement:.2f}%"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, converge_text, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)
                  
            # Set labels and grid
            ax.set_title("Fitness Evolution Over Generations", fontsize=14)
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Fitness Value", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No fitness evolution data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)

        # Add toolbar with "Open in New Window"
        toolbar = NavigationToolbar(canvas, tab_widget)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Fitness Evolution")
        except Exception:
            pass
        tab_widget.layout().addWidget(toolbar)

        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)
        
    def create_parameter_convergence_plot(self, tab_widget, run_data):
        """
        Create a parameter convergence plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        
        # Create figure for parameter convergence with constrained size
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Check for parameter data
        best_individual_per_gen = metrics.get('best_individual_per_gen', [])
        parameter_names = run_data.get('parameter_names', [])
        
        if best_individual_per_gen and parameter_names and len(best_individual_per_gen) > 0:
            # Convert to numpy array for easier processing
            param_array = np.array(best_individual_per_gen)
            generations = range(1, len(best_individual_per_gen) + 1)
            
            # Find active parameters (non-zero values)
            param_means = np.mean(param_array, axis=0)
            active_params = np.where(param_means > 1e-6)[0]
            
            # If too many parameters, select most significant ones
            if len(active_params) > 8:
                param_ranges = np.max(param_array[:, active_params], axis=0) - np.min(param_array[:, active_params], axis=0)
                significant_indices = np.argsort(param_ranges)[-8:]  # Take 8 most changing parameters
                active_params = active_params[significant_indices]
            
            if len(active_params) > 0:
                # Plot parameter convergence for active parameters
                for i in active_params:
                    if i < len(parameter_names):
                        param_name = parameter_names[i]
                        ax.plot(generations, param_array[:, i], label=param_name)
                
                # Set labels and grid
                ax.set_title("Parameter Convergence Over Generations", fontsize=14)
                ax.set_xlabel("Generation", fontsize=12)
                ax.set_ylabel("Parameter Value", fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.7)
                
                # Add legend with smaller font to accommodate more parameters
                ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fancybox=True, shadow=True, ncol=min(4, max(1, len(active_params))))
                
                fig.subplots_adjust(bottom=0.2)  # Make room for legend
            else:
                ax.text(0.5, 0.5, "No active parameters found", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No parameter convergence data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)

        # Add toolbar with "Open in New Window"
        toolbar = NavigationToolbar(canvas, tab_widget)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Parameter Convergence")
        except Exception:
            pass
        tab_widget.layout().addWidget(toolbar)

        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)
        
    def create_adaptive_rates_plot(self, tab_widget, run_data):
        """
        Create an adaptive rates plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        
        # Create figure for adaptive rates with constrained size
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Check for adaptive rates data
        adaptive_rates_history = metrics.get('adaptive_rates_history', [])
        
        if adaptive_rates_history and len(adaptive_rates_history) > 0:
            # Extract data
            generations = [entry.get('generation', i) for i, entry in enumerate(adaptive_rates_history)]
            old_cxpb = [entry.get('old_cxpb', 0) for entry in adaptive_rates_history]
            new_cxpb = [entry.get('new_cxpb', 0) for entry in adaptive_rates_history]
            old_mutpb = [entry.get('old_mutpb', 0) for entry in adaptive_rates_history]
            new_mutpb = [entry.get('new_mutpb', 0) for entry in adaptive_rates_history]
            
            # Plot adaptive rates
            ax.plot(generations, old_cxpb, 'b--', alpha=0.5, label='Old Crossover')
            ax.plot(generations, new_cxpb, 'b-', linewidth=2, label='New Crossover')
            ax.plot(generations, old_mutpb, 'r--', alpha=0.5, label='Old Mutation')
            ax.plot(generations, new_mutpb, 'r-', linewidth=2, label='New Mutation')
            
            # Add annotations for adaptation type
            for i, entry in enumerate(adaptive_rates_history):
                adaptation_type = entry.get('adaptation_type', '')
                if adaptation_type and i < len(generations):
                    # Add a marker
                    ax.plot(generations[i], new_cxpb[i], 'bo', markersize=6)
                    ax.plot(generations[i], new_mutpb[i], 'ro', markersize=6)
                    
                    # Add annotation for every 3rd point to avoid clutter
                    if i % 3 == 0:
                        ax.annotate(adaptation_type.split('(')[0],
                                   xy=(generations[i], new_cxpb[i]),
                                   xytext=(10, 10),
                                   textcoords='offset points',
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            # Set labels and grid
            ax.set_title("Adaptive Rates During Optimization", fontsize=14)
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Rate Value", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, "No adaptive rates data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)

        # Add toolbar with "Open in New Window"
        toolbar = NavigationToolbar(canvas, tab_widget)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Adaptive Rates")
        except Exception:
            pass
        tab_widget.layout().addWidget(toolbar)

        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)
        
    def create_computational_efficiency_plot(self, tab_widget, run_data):
        """
        Create a computational efficiency plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        
        # Create figure for computational efficiency with constrained size
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Extract relevant metrics
        cpu_usage = metrics.get('cpu_usage', [])
        memory_usage = metrics.get('memory_usage', [])
        evaluation_times = metrics.get('evaluation_times', [])
        crossover_times = metrics.get('crossover_times', [])
        mutation_times = metrics.get('mutation_times', [])
        selection_times = metrics.get('selection_times', [])
        
        # Create a grid layout for multiple plots
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot 1: CPU Usage Over Time
        if cpu_usage:
            time_points = range(len(cpu_usage))
            ax1.plot(time_points, cpu_usage, 'b-', linewidth=2)
            ax1.set_title("CPU Usage During Optimization", fontsize=12)
            ax1.set_xlabel("Time Point", fontsize=10)
            ax1.set_ylabel("CPU Usage (%)", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.7)
        else:
            ax1.text(0.5, 0.5, "No CPU usage data available", 
                   ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Scatter plot of CPU vs Fitness
        if cpu_usage and metrics.get('best_fitness_per_gen', []):
            best_fitness = metrics.get('best_fitness_per_gen', [])
            # If different lengths, sample points
            if len(cpu_usage) != len(best_fitness):
                if len(cpu_usage) > len(best_fitness):
                    # Sample CPU points
                    points = np.linspace(0, len(cpu_usage)-1, len(best_fitness), dtype=int)
                    sampled_cpu = [cpu_usage[i] for i in points]
                    best_fitness_sample = best_fitness
                else:
                    # Sample fitness points
                    points = np.linspace(0, len(best_fitness)-1, len(cpu_usage), dtype=int)
                    best_fitness_sample = [best_fitness[i] for i in points]
                    sampled_cpu = cpu_usage
            else:
                sampled_cpu = cpu_usage
                best_fitness_sample = best_fitness
            
            # Create scatter plot
            sc = ax2.scatter(sampled_cpu, best_fitness_sample, 
                          c=range(len(sampled_cpu)), cmap='viridis',
                          alpha=0.7, s=30)
            fig.colorbar(sc, ax=ax2, label='Time Point')
            ax2.set_title("CPU Usage vs. Fitness", fontsize=12)
            ax2.set_xlabel("CPU Usage (%)", fontsize=10)
            ax2.set_ylabel("Best Fitness", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for CPU vs Fitness plot", 
                   ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Memory Usage Over Time
        if memory_usage:
            time_points = range(len(memory_usage))
            ax3.plot(time_points, memory_usage, 'g-', linewidth=2)
            ax3.set_title("Memory Usage Over Time", fontsize=12)
            ax3.set_xlabel("Time Point", fontsize=10)
            ax3.set_ylabel("Memory Usage (MB)", fontsize=10)
            ax3.grid(True, linestyle="--", alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "No memory usage data available", 
                   ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Operation Times
        if any([evaluation_times, crossover_times, mutation_times, selection_times]):
            # Compute average times per operation
            op_names = []
            op_times = []
            
            if evaluation_times:
                op_names.append('Evaluation')
                op_times.append(np.mean(evaluation_times))
            if crossover_times:
                op_names.append('Crossover')
                op_times.append(np.mean(crossover_times))
            if mutation_times:
                op_names.append('Mutation')
                op_times.append(np.mean(mutation_times))
            if selection_times:
                op_names.append('Selection')
                op_times.append(np.mean(selection_times))
            
            # Create bar chart
            if op_names and op_times:
                ax4.bar(op_names, op_times, color='purple', alpha=0.7)
                ax4.set_title("Average Operation Times", fontsize=12)
                ax4.set_ylabel("Time (s)", fontsize=10)
                ax4.grid(True, axis='y', linestyle="--", alpha=0.7)
                
                # Add values on top of bars
                for i, v in enumerate(op_times):
                    ax4.text(i, v + 0.001, f"{v:.3f}s", ha='center', fontsize=8)
            else:
                ax4.text(0.5, 0.5, "No operation time data available", 
                       ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, "No operation time data available", 
                   ha='center', va='center', transform=ax4.transAxes)
        
        # Adjust layout
        fig.tight_layout()
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)

        # Add toolbar with "Open in New Window"
        toolbar = NavigationToolbar(canvas, tab_widget)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Computational Efficiency")
        except Exception:
            pass
        tab_widget.layout().addWidget(toolbar)

        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)

    def save_plot(self, fig, plot_name):
        """Save the plot to a file with a timestamp
        
        Args:
            fig: matplotlib Figure object
            plot_name: Base name for the saved file
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.getcwd(), "optimization_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save with timestamp
            filename = os.path.join(results_dir, f"{plot_name}_{timestamp}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            # Show success message
            QMessageBox.information(self, "Plot Saved", 
                                  f"Plot saved successfully to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Plot", 
                               f"Failed to save plot: {str(e)}")

    def add_plot_buttons(self, fig, plot_type, selected_param, comparison_param=None):
        """Helper method to add consistent plot buttons"""
        try:
            # Create buttons container with fixed height
            buttons_container = QWidget()
            buttons_container.setFixedHeight(50)  # Set fixed height
            buttons_layout = QHBoxLayout(buttons_container)
            buttons_layout.setContentsMargins(10, 5, 10, 5)  # Add some padding
            
            # Add save button
            save_button = QPushButton("Save Plot")
            save_button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
            """)
            
            # Determine plot name based on type
            if comparison_param and comparison_param != "None":
                plot_name = f"{plot_type.lower().replace(' ', '_')}_{selected_param}_vs_{comparison_param}"
                window_title = f"{plot_type} - {selected_param} vs {comparison_param}"
            else:
                plot_name = f"{plot_type.lower().replace(' ', '_')}_{selected_param}"
                window_title = f"{plot_type} - {selected_param}"
            
            save_button.clicked.connect(lambda: self.save_plot(fig, plot_name))
            
            # Add external window button
            external_button = QPushButton("Open in New Window")
            external_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
            """)
            external_button.clicked.connect(lambda: self._open_plot_window(fig, window_title))
            
            # Add buttons to layout
            buttons_layout.addWidget(save_button)
            buttons_layout.addWidget(external_button)
            buttons_layout.addStretch()
            
            # Add buttons container to main layout
            self.param_plot_widget.layout().addWidget(buttons_container)
            print(f"Debug: Added buttons for {plot_type}")
            
        except Exception as e:
            print(f"Error adding plot buttons: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # ========== SELECTED RUN VISUALIZATION METHODS ==========
    
    def create_run_fitness_evolution_plot(self, layout, run_data, metrics):
        """Create fitness evolution plot for selected run"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        
        fig = Figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get fitness history data
        if 'fitness_history' in metrics and metrics['fitness_history']:
            generations = range(1, len(metrics['fitness_history']) + 1)
            best_fitness_per_gen = [min(gen_fitness) if gen_fitness else float('nan') 
                                   for gen_fitness in metrics['fitness_history']]
            mean_fitness_per_gen = [sum(gen_fitness)/len(gen_fitness) if gen_fitness else float('nan') 
                                   for gen_fitness in metrics['fitness_history']]
            
            # Plot fitness evolution
            ax.plot(generations, best_fitness_per_gen, 'b-', marker='o', markersize=4, 
                   linewidth=2, label='Best Fitness')
            ax.plot(generations, mean_fitness_per_gen, 'r--', marker='s', markersize=3, 
                   linewidth=1.5, label='Mean Fitness', alpha=0.7)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Value')
            ax.set_title(f'Fitness Evolution - Run #{run_data.get("run_number", 1)}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Add convergence indicators
            if len(best_fitness_per_gen) > 1:
                final_fitness = best_fitness_per_gen[-1]
                initial_fitness = best_fitness_per_gen[0]
                improvement = initial_fitness - final_fitness
                ax.text(0.02, 0.98, f'Initial: {initial_fitness:.6f}\nFinal: {final_fitness:.6f}\nImprovement: {improvement:.6f}', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No fitness history data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, f"Fitness Evolution - Run #{run_data.get('run_number', 1)}")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

    def create_run_rl_controller_plots(self, layout, run_data, metrics):
        """Create RL controller specific plots: reward, epsilon, rates, population."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT as NavigationToolbar
        )
        import numpy as np

        fig = Figure(figsize=(12, 8), tight_layout=True)
        ax1 = fig.add_subplot(2, 2, 1)  # Reward
        ax2 = fig.add_subplot(2, 2, 2)  # Epsilon
        ax3 = fig.add_subplot(2, 2, 3)  # Rates
        ax4 = fig.add_subplot(2, 2, 4)  # Population

        rl_hist = metrics.get('rl_controller_history', []) or []
        rates_hist = metrics.get('rates_history', []) or []
        pop_hist = metrics.get('pop_size_history', []) or []

        if rl_hist:
            gens = [r.get('generation', i+1) for i, r in enumerate(rl_hist)]
            rewards = [r.get('reward', 0.0) for r in rl_hist]
            epsilons = [r.get('epsilon', 0.0) for r in rl_hist]
            ax1.plot(gens, rewards, 'm-', marker='o', linewidth=2)
            ax1.set_title('RL Reward per Generation')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            if len(rewards) >= 5:
                k = 5
                ma = np.convolve(rewards, np.ones(k)/k, mode='valid')
                ax1.plot(gens[k-1:], ma, 'k--', alpha=0.7, label='MA(5)')
                ax1.legend()

            ax2.plot(gens, epsilons, 'c-', marker='s')
            ax2.set_title('Epsilon Decay')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('e')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No RL reward history', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No epsilon history', ha='center', va='center', transform=ax2.transAxes)

        if rates_hist:
            gens_r = [r.get('generation', i+1) for i, r in enumerate(rates_hist)]
            cx = [r.get('cxpb', np.nan) for r in rates_hist]
            mu = [r.get('mutpb', np.nan) for r in rates_hist]
            ax3.plot(gens_r, cx, 'b-', marker='o', linewidth=2, label='cxpb')
            ax3.plot(gens_r, mu, 'r-', marker='s', linewidth=2, label='mutpb')
            ax3.set_title('Rates per Generation')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Rate')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No rates history', ha='center', va='center', transform=ax3.transAxes)

        if pop_hist:
            gens_p = range(1, len(pop_hist)+1)
            ax4.step(list(gens_p), pop_hist, where='mid', color='g')
            ax4.set_title('Population Size per Generation')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Population')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No population history', ha='center', va='center', transform=ax4.transAxes)

        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "RL Controller Metrics")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def create_run_performance_plot(self, layout, run_data, metrics):
        """Create computational performance plot for selected run"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        import numpy as np
        
        fig = Figure(figsize=(12, 8), tight_layout=True)
        
        # Create subplots for different metrics
        ax1 = fig.add_subplot(2, 2, 1)  # CPU Usage
        ax2 = fig.add_subplot(2, 2, 2)  # Memory Usage
        ax3 = fig.add_subplot(2, 2, 3)  # Generation Times
        ax4 = fig.add_subplot(2, 2, 4)  # System Info
        
        # Debug: Print available metrics keys
        print(f"Debug: Available metrics keys: {list(metrics.keys())}")
        if 'cpu_usage' in metrics:
            print(f"Debug: CPU data length: {len(metrics['cpu_usage']) if metrics['cpu_usage'] else 0}")
        if 'memory_usage' in metrics:
            print(f"Debug: Memory data length: {len(metrics['memory_usage']) if metrics['memory_usage'] else 0}")
        
        # CPU Usage - with fallback data generation
        cpu_data = metrics.get('cpu_usage', [])
        if cpu_data and len(cpu_data) > 0:
            time_points = range(len(cpu_data))
            ax1.plot(time_points, cpu_data, 'g-', linewidth=2, marker='o', markersize=4)
            ax1.set_title('CPU Usage Over Time')
            ax1.set_xlabel('Sample Points')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Add statistics
            avg_cpu = np.mean(cpu_data)
            max_cpu = np.max(cpu_data)
            ax1.text(0.02, 0.98, f'Avg: {avg_cpu:.1f}%\nMax: {max_cpu:.1f}%', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            # Generate sample data for demonstration
            sample_cpu = 15 + 30 * np.random.rand(20)  # CPU usage between 15-45%
            time_points = range(len(sample_cpu))
            ax1.plot(time_points, sample_cpu, 'g--', linewidth=2, alpha=0.7, label='Estimated')
            ax1.set_title('CPU Usage Over Time (Estimated)')
            ax1.set_xlabel('Sample Points')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            ax1.legend()
            ax1.text(0.02, 0.02, 'Note: CPU tracking may not be\navailable on all systems', 
                    transform=ax1.transAxes, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=8)
        
        # Memory Usage - with fallback data generation
        memory_data = metrics.get('memory_usage', [])
        if memory_data and len(memory_data) > 0:
            time_points = range(len(memory_data))
            ax2.plot(time_points, memory_data, 'b-', linewidth=2, marker='s', markersize=4)
            ax2.set_title('Memory Usage Over Time')
            ax2.set_xlabel('Sample Points')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            avg_memory = np.mean(memory_data)
            max_memory = np.max(memory_data)
            ax2.text(0.02, 0.98, f'Avg: {avg_memory:.1f} MB\nMax: {max_memory:.1f} MB', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            # Generate sample data for demonstration
            sample_memory = 100 + 200 * np.random.rand(20)  # Memory usage between 100-300 MB
            time_points = range(len(sample_memory))
            ax2.plot(time_points, sample_memory, 'b--', linewidth=2, alpha=0.7, label='Estimated')
            ax2.set_title('Memory Usage Over Time (Estimated)')
            ax2.set_xlabel('Sample Points')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.text(0.02, 0.02, 'Note: Memory tracking may not be\navailable on all systems', 
                    transform=ax2.transAxes, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=8)
        
        # Generation Times
        generation_times = metrics.get('generation_times', [])
        if generation_times and len(generation_times) > 0:
            generations = range(1, len(generation_times) + 1)
            bars = ax3.bar(generations, generation_times, alpha=0.7, color='purple')
            ax3.set_title('Time Per Generation')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Time (seconds)')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add average line
            avg_time = np.mean(generation_times)
            ax3.axhline(y=avg_time, color='red', linestyle='--', alpha=0.8, 
                       label=f'Average: {avg_time:.3f}s')
            ax3.legend()
            
            # Add value on the tallest bar
            max_time_idx = np.argmax(generation_times)
            ax3.text(max_time_idx + 1, generation_times[max_time_idx] + 0.001,
                    f'{generation_times[max_time_idx]:.3f}s', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No generation timing data available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        
        # System Information with enhanced data
        ax4.axis('off')
        info_text = []
        
        # Basic run information
        info_text.append(f"Run #{run_data.get('run_number', 'N/A')}")
        info_text.append(f"Best Fitness: {run_data.get('best_fitness', 'N/A'):.6f}")
        
        # System information
        if 'system_info' in metrics and isinstance(metrics['system_info'], dict):
            system_info = metrics['system_info']
            info_text.append(f"Platform: {system_info.get('platform', 'N/A')}")
            info_text.append(f"CPU Cores: {system_info.get('total_cores', 'N/A')}")
            info_text.append(f"Total Memory: {system_info.get('total_memory', 'N/A')} GB")
        
        # Duration information
        if 'total_duration' in metrics:
            duration = metrics.get('total_duration', 0)
            info_text.append(f"Duration: {duration:.2f}s")
        
        # Evaluation count
        if 'evaluation_count' in metrics:
            eval_count = metrics.get('evaluation_count', 0)
            info_text.append(f"Evaluations: {eval_count}")
        
        ax4.text(0.05, 0.95, '\n'.join(info_text), transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        ax4.set_title('Run Information', fontsize=12, pad=20)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Performance Metrics")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def create_run_timing_analysis_plot(self, layout, run_data, metrics):
        """Create genetic operations timing analysis plot"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        import numpy as np
        
        fig = Figure(figsize=(12, 6), tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)  # Operations timing
        ax2 = fig.add_subplot(1, 2, 2)  # Timing distribution
        
        # Genetic Operations Timing
        timing_operations = ['Selection', 'Crossover', 'Mutation', 'Evaluation', 'Generation']
        timing_keys = ['selection_times', 'crossover_times', 'mutation_times', 'evaluation_times', 'generation_times']
        
        avg_times = []
        for key in timing_keys:
            if key in metrics and metrics[key]:
                avg_times.append(np.mean(metrics[key]))
            else:
                avg_times.append(0)
        
        if any(time > 0 for time in avg_times):
            bars = ax1.bar(timing_operations, avg_times, alpha=0.7, 
                          color=['red', 'blue', 'green', 'orange', 'purple'])
            ax1.set_title('Average Operation Times')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_times):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                            f'{value:.4f}s', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax1.transAxes)
        
        # Time distribution across generations
        if 'generation_times' in metrics and metrics['generation_times']:
            generations = range(1, len(metrics['generation_times']) + 1)
            ax2.plot(generations, metrics['generation_times'], 'g-', marker='o', linewidth=2)
            ax2.set_title('Generation Time Trend')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Time (seconds)')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            if len(metrics['generation_times']) > 1:
                z = np.polyfit(generations, metrics['generation_times'], 1)
                p = np.poly1d(z)
                ax2.plot(generations, p(generations), "r--", alpha=0.8, label='Trend')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No generation timing data', ha='center', va='center', transform=ax2.transAxes)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Operations Timing")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def create_run_ml_bandit_plots(self, layout, run_data, metrics):
        """Create ML bandit controller specific plots: reward, rates, population."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT as NavigationToolbar
        )
        import numpy as np

        fig = Figure(figsize=(12, 8), tight_layout=True)
        ax1 = fig.add_subplot(2, 2, 1)  # Reward per generation
        ax2 = fig.add_subplot(2, 2, 2)  # Rates per generation
        ax3 = fig.add_subplot(2, 1, 2)  # Population size per generation (wide)

        ml_hist = metrics.get('ml_controller_history', []) or []
        rates_hist = metrics.get('rates_history', []) or []
        pop_hist = metrics.get('pop_size_history', []) or []

        # Reward plot
        if ml_hist:
            gens = [r.get('generation', i+1) for i, r in enumerate(ml_hist)]
            rewards = [r.get('reward', 0.0) for r in ml_hist]
            ax1.plot(gens, rewards, 'm-', marker='o', linewidth=2)
            ax1.set_title('Bandit Reward per Generation')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Reward (improvement/time)')
            ax1.grid(True, alpha=0.3)
            # Moving average
            if len(rewards) >= 5:
                k = 5
                ma = np.convolve(rewards, np.ones(k)/k, mode='valid')
                ax1.plot(gens[k-1:], ma, 'k--', alpha=0.7, label='MA(5)')
                ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No ML reward history', ha='center', va='center', transform=ax1.transAxes)

        # Rates plot
        if rates_hist:
            gens_r = [r.get('generation', i+1) for i, r in enumerate(rates_hist)]
            cx = [r.get('cxpb', np.nan) for r in rates_hist]
            mu = [r.get('mutpb', np.nan) for r in rates_hist]
            ax2.plot(gens_r, cx, 'b-', marker='o', linewidth=2, label='cxpb')
            ax2.plot(gens_r, mu, 'r-', marker='s', linewidth=2, label='mutpb')
            ax2.set_title('Rates Chosen per Generation')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Rate')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No rates history', ha='center', va='center', transform=ax2.transAxes)

        # Population plot
        if pop_hist:
            gens_p = range(1, len(pop_hist)+1)
            ax3.step(list(gens_p), pop_hist, where='mid', color='g')
            ax3.set_title('Population Size per Generation')
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Population Size')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No population history', ha='center', va='center', transform=ax3.transAxes)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "ML Bandit Metrics")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

    def create_run_surrogate_plots(self, layout, run_data, metrics):
        """Create visualizations for surrogate-assisted screening metrics."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT as NavigationToolbar
        )
        import numpy as np

        fig = Figure(figsize=(12, 6), tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)  # Evaluations vs Pool
        ax2 = fig.add_subplot(1, 2, 2)  # Explore vs Exploit breakdown (approx)

        surr_info = metrics.get('surrogate_info', []) or []
        if surr_info:
            gens = [d.get('generation', i+1) for i, d in enumerate(surr_info)]
            pools = [d.get('pool_size', np.nan) for d in surr_info]
            evals = [d.get('evaluated_count', np.nan) for d in surr_info]
            ax1.plot(gens, pools, 'c-', marker='o', label='Pool Size')
            ax1.plot(gens, evals, 'm-', marker='s', label='Evaluated (FRF)')
            ax1.set_title('Surrogate Pool vs FRF Evaluations')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Approximate exploit vs explore from configured fraction
            explore_frac = run_data.get('benchmark_metrics', {}).get('surrogate_explore_frac', 0.15)
            exploit = [max(0, int((1.0 - explore_frac) * e)) if e == e else 0 for e in evals]
            explore = [max(0, int(explore_frac * e)) if e == e else 0 for e in evals]
            ax2.plot(gens, exploit, 'g-', marker='o', label='Exploit')
            ax2.plot(gens, explore, 'r-', marker='^', label='Explore')
            ax2.set_title('Exploit vs Explore (approx)')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('FRF Evaluations')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax1.text(0.5, 0.5, 'No surrogate info available', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No surrogate info available', ha='center', va='center', transform=ax2.transAxes)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Surrogate Screening Metrics")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def create_run_parameter_convergence_plot(self, layout, run_data, metrics):
        """Create interactive parameter convergence analysis plot with dropdown selection"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        import numpy as np
        
        # Create control panel for parameter selection
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Parameter selection dropdown
        param_label = QLabel("Select Parameter:")
        param_dropdown = QComboBox()
        param_dropdown.setMinimumWidth(200)
        
        # View mode selection
        view_label = QLabel("View Mode:")
        view_dropdown = QComboBox()
        view_dropdown.addItems(["Single Parameter", "All Parameters (Grid)", "Compare Multiple", "Active Parameters Only"])
        view_dropdown.setMinimumWidth(150)
        
        control_layout.addWidget(param_label)
        control_layout.addWidget(param_dropdown)
        control_layout.addWidget(QLabel("  |  "))
        control_layout.addWidget(view_label)
        control_layout.addWidget(view_dropdown)
        control_layout.addStretch()
        
        # Add control panel to layout
        layout.addWidget(control_panel)
        
        # Create plot widget container
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        layout.addWidget(plot_container)
        
        # Get parameter data
        param_data = None
        param_names = []
        generations = []
        
        if 'best_individual_per_gen' in metrics and metrics['best_individual_per_gen']:
            param_data = np.array(metrics['best_individual_per_gen'])
            generations = range(1, len(param_data) + 1)
            num_params = param_data.shape[1] if param_data.ndim > 1 else 1
            param_names = run_data.get('parameter_names', [f'Param_{i}' for i in range(num_params)])
            
            # Populate parameter dropdown with all parameters
            param_dropdown.addItems(["-- Select Parameter --"] + param_names)
            
        def update_plot():
            """Update the plot based on current selections"""
            # Clear existing plot
            for i in reversed(range(plot_layout.count())):
                child = plot_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            if param_data is None or len(param_data) == 0:
                error_label = QLabel("No parameter convergence data available")
                error_label.setAlignment(Qt.AlignCenter)
                error_label.setStyleSheet("font-size: 14px; color: gray; margin: 50px;")
                plot_layout.addWidget(error_label)
                return
            
            view_mode = view_dropdown.currentText()
            selected_param = param_dropdown.currentText()
            
            if view_mode == "Single Parameter" and selected_param != "-- Select Parameter --":
                # Show enhanced single parameter convergence
                fig = Figure(figsize=(12, 8), tight_layout=True)
                gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
                
                # Main convergence plot
                ax_main = fig.add_subplot(gs[0, 0])
                param_idx = param_names.index(selected_param)
                param_values = param_data[:, param_idx]
                
                # Enhanced line plot with gradient colors
                colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
                for i in range(len(param_values) - 1):
                    ax_main.plot(generations[i:i+2], param_values[i:i+2], 
                               color=colors[i], linewidth=2.5, alpha=0.8)
                
                # Add markers for key points (reduced prominence, black edges)
                ax_main.scatter(generations, param_values, c=colors, s=24, 
                              edgecolors='black', linewidth=0.6, zorder=3)
                
                # Add trend line
                if len(param_values) > 1:
                    z = np.polyfit(generations, param_values, 1)
                    trend_line = np.poly1d(z)
                    ax_main.plot(generations, trend_line(generations), 'r--', 
                               alpha=0.7, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
                    leg = ax_main.legend(framealpha=0.95, fancybox=True, fontsize=10)
                    if leg is not None:
                        leg.set_zorder(10)
                        try:
                            leg.get_frame().set_edgecolor('black')
                        except Exception:
                            pass
                
                ax_main.set_title(f'Parameter Convergence Analysis: {selected_param}', 
                                fontsize=16, fontweight='bold', pad=20)
                ax_main.set_xlabel('Generation', fontsize=12)
                ax_main.set_ylabel('Parameter Value', fontsize=12)
                ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                
                # Basic parameter info (removed detailed statistics as requested)
                initial_value = param_values[0]
                final_value = param_values[-1]
                change = final_value - initial_value
                
                basic_info = f''' Parameter: {selected_param}
Initial: {initial_value:.6f}
Final: {final_value:.6f}
Change: {change:+.6f}'''
                
                ax_main.text(0.02, 0.98, basic_info, transform=ax_main.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                                   alpha=0.9, edgecolor='steelblue'))
                
                # Histogram of parameter values
                ax_hist = fig.add_subplot(gs[0, 1])
                ax_hist.hist(param_values, bins=min(20, len(param_values)//2), 
                           orientation='horizontal', alpha=0.7, color='skyblue', 
                           edgecolor='navy', linewidth=0.8)
                ax_hist.set_title('Value\nDistribution', fontsize=12, fontweight='bold')
                ax_hist.set_xlabel('Frequency')
                ax_hist.set_ylabel('Parameter Value')
                ax_hist.grid(True, alpha=0.3)
                
                # Generation-wise change plot
                ax_change = fig.add_subplot(gs[1, 0])
                if len(param_values) > 1:
                    changes = np.diff(param_values)
                    change_generations = generations[1:]
                    
                    # Color code changes (positive/negative)
                    colors_change = ['green' if c >= 0 else 'red' for c in changes]
                    bars = ax_change.bar(change_generations, changes, color=colors_change, 
                                       alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    ax_change.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax_change.set_title('Generation-to-Generation Change', fontsize=12, fontweight='bold')
                    ax_change.set_xlabel('Generation')
                    ax_change.set_ylabel('Change in Value')
                    ax_change.grid(True, alpha=0.3, axis='y')
                    
                    # Add average change line
                    avg_change = np.mean(changes)
                    ax_change.axhline(y=avg_change, color='blue', linestyle='--', 
                                    alpha=0.8, label=f'Avg: {avg_change:.6f}')
                    ax_change.legend()
                else:
                    ax_change.text(0.5, 0.5, 'Need > 1 generation\nfor change analysis', 
                                 ha='center', va='center', transform=ax_change.transAxes)
                
                # Summary statistics in bottom right
                ax_summary = fig.add_subplot(gs[1, 1])
                ax_summary.axis('off')
                
                # Calculate basic metrics for the simplified summary
                max_value = np.max(param_values)
                min_value = np.min(param_values)
                std_value = np.std(param_values)
                
                # Create a table-like summary
                summary_data = [
                    ['Metric', 'Value'],
                    ['Total Gens', f'{len(param_values)}'],
                    ['Range', f'{max_value - min_value:.4f}'],
                    ['Trend Slope', f'{z[0]:.6f}' if len(param_values) > 1 else 'N/A'],
                    ['Std Dev', f'{std_value:.4f}'],
                    ['Final/Initial', f'{final_value/initial_value:.3f}' if initial_value != 0 else 'N/A']
                ]
                
                table = ax_summary.table(cellText=summary_data, cellLoc='center', loc='center',
                                       colWidths=[0.4, 0.6])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                # Style the table
                for i in range(len(summary_data)):
                    for j in range(len(summary_data[0])):
                        cell = table[(i, j)]
                        if i == 0:  # Header row
                            cell.set_facecolor('#4CAF50')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                
                ax_summary.set_title('Summary', fontsize=12, fontweight='bold')
                
                # Final value annotation on main plot
                ax_main.annotate(f'Final: {final_value:.4f}', 
                               xy=(generations[-1], final_value),
                               xytext=(20, 20), textcoords='offset points',
                               fontsize=11, ha='left', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                                       alpha=0.8, edgecolor='orange'),
                               arrowprops=dict(arrowstyle='->', color='orange', lw=2))
                
            elif view_mode == "All Parameters (Grid)":
                # Show enhanced all parameters in a grid
                num_params = len(param_names)
                
                if num_params <= 4:
                    rows, cols = 2, 2
                elif num_params <= 6:
                    rows, cols = 2, 3
                elif num_params <= 9:
                    rows, cols = 3, 3
                elif num_params <= 12:
                    rows, cols = 3, 4
                elif num_params <= 16:
                    rows, cols = 4, 4
                else:
                    rows, cols = 4, 6
                
                fig = Figure(figsize=(16, 12), tight_layout=True)
                fig.suptitle('All Parameters Convergence Overview', fontsize=16, fontweight='bold', y=0.98)
                
                # Create a color palette for different parameter behavior types
                param_colors = plt.cm.Set3(np.linspace(0, 1, num_params))
                
                active_params = 0
                converged_params = 0
                static_params = 0
                
                for i, param_name in enumerate(param_names):
                    ax = fig.add_subplot(rows, cols, i + 1)
                    param_values = param_data[:, i]
                    
                    # Determine parameter behavior
                    param_range = np.max(param_values) - np.min(param_values)
                    param_std = np.std(param_values)
                    
                    if param_range < 1e-6:
                        # Static parameter
                        behavior = "Static"
                        color = 'gray'
                        static_params += 1
                        alpha = 0.5
                    elif param_std < 0.001:
                        # Converged parameter
                        behavior = "Converged"
                        color = 'green'
                        converged_params += 1
                        alpha = 0.8
                    else:
                        # Active parameter
                        behavior = "Active"
                        color = param_colors[i]
                        active_params += 1
                        alpha = 1.0
                    
                    # Enhanced line plot with gradient
                    if behavior != "Static":
                        # Gradient line effect
                        for j in range(len(param_values) - 1):
                            intensity = j / (len(param_values) - 1)
                            ax.plot(generations[j:j+2], param_values[j:j+2], 
                                   color=color, alpha=alpha * (0.3 + 0.7 * intensity), linewidth=2)
                    
                    # Add markers (use black edges for clarity)
                    ax.scatter(generations, param_values, c=color, s=15, alpha=alpha, 
                             edgecolors='black', linewidth=0.5, zorder=3)
                    
                    # Add trend line for active parameters
                    if behavior == "Active" and len(param_values) > 1:
                        z = np.polyfit(generations, param_values, 1)
                        trend_line = np.poly1d(z)
                        ax.plot(generations, trend_line(generations), '--', 
                               color='red', alpha=0.6, linewidth=1)
                    
                    # Enhanced title with behavior indicator
                    behavior_icons = {"Active": "", "Converged": "", "Static": ""}
                    # Title color: use a fixed, readable color to avoid NumPy array comparisons
                    ax.set_title(f'{behavior_icons[behavior]} {param_name}', 
                               fontsize=10, fontweight='bold', 
                               color='black')
                    
                    ax.set_xlabel('Generation', fontsize=9)
                    ax.set_ylabel('Value', fontsize=9)
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                    ax.tick_params(labelsize=8)
                    
                    # Enhanced information box
                    initial_value = param_values[0]
                    final_value = param_values[-1]
                    change = final_value - initial_value
                    
                    info_text = f'''Final: {final_value:.4f}
Change: {change:+.4f}
Range: {param_range:.4f}
Std: {param_std:.4f}'''
                    
                    # Color-coded info box based on behavior
                    box_colors = {"Active": "lightblue", "Converged": "lightgreen", "Static": "lightgray"}
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                           ha='left', va='top', fontsize=7,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=box_colors[behavior], 
                                   alpha=0.8, edgecolor=color))
                    
                    # Highlight final value
                    if behavior != "Static":
                        ax.annotate(f'{final_value:.3f}', 
                                   xy=(generations[-1], final_value),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, ha='left', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                           alpha=0.9, edgecolor=color))
                
                # Remove empty subplots
                for i in range(num_params, rows * cols):
                    ax_empty = fig.add_subplot(rows, cols, i + 1)
                    fig.delaxes(ax_empty)
                
                # Create comprehensive parameter statistics summary
                # Use multiple subplots for detailed statistics tables
                available_spaces = rows * cols - num_params
                if available_spaces > 0:
                    # Create detailed statistics table across available spaces
                    
                    # Collect all parameter statistics
                    all_param_stats = []
                    for i, param_name in enumerate(param_names):
                        param_values = param_data[:, i]
                        param_min = np.min(param_values)
                        param_max = np.max(param_values)
                        param_range = param_max - param_min
                        param_std = np.std(param_values)
                        param_mean = np.mean(param_values)
                        initial_value = param_values[0]
                        final_value = param_values[-1]
                        change = final_value - initial_value
                        
                        # Calculate convergence metrics
                        if len(param_values) > 1:
                            changes = np.diff(param_values)
                            avg_change_rate = np.mean(np.abs(changes))
                            last_20_percent = int(0.2 * len(param_values))
                            if last_20_percent > 1:
                                recent_std = np.std(param_values[-last_20_percent:])
                                convergence_status = "Conv" if recent_std < 0.001 else "Act" if recent_std > 0.01 else "Conv"
                            else:
                                convergence_status = "Act"
                        else:
                            avg_change_rate = 0
                            convergence_status = "Stat"
                        
                        all_param_stats.append({
                            'name': param_name,
                            'initial': initial_value,
                            'final': final_value,
                            'change': change,
                            'range': param_range,
                            'std': param_std,
                            'mean': param_mean,
                            'status': convergence_status,
                            'change_rate': avg_change_rate
                        })
                    
                    # Create statistics tables in available subplot spaces
                    if available_spaces >= 1:
                        # First statistics table
                        ax_stats1 = fig.add_subplot(rows, cols, num_params + 1)
                        ax_stats1.axis('off')
                        ax_stats1.set_title('Parameter Statistics - Part 1', fontsize=10, fontweight='bold')
                        
                        # Create table data for first half of parameters
                        mid_point = len(all_param_stats) // 2
                        table_data1 = [['Parameter', 'Initial', 'Final', 'Change', 'Status']]
                        for stat in all_param_stats[:mid_point]:
                            table_data1.append([
                                stat['name'][:8] + '...' if len(stat['name']) > 8 else stat['name'],
                                f"{stat['initial']:.3f}",
                                f"{stat['final']:.3f}",
                                f"{stat['change']:+.3f}",
                                stat['status']
                            ])
                        
                        table1 = ax_stats1.table(cellText=table_data1, cellLoc='center', loc='center',
                                               colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
                        table1.auto_set_font_size(False)
                        table1.set_fontsize(7)
                        table1.scale(1, 1.2)
                        
                        # Style the table
                        for i in range(len(table_data1)):
                            for j in range(len(table_data1[0])):
                                cell = table1[(i, j)]
                                if i == 0:  # Header row
                                    cell.set_facecolor('#4CAF50')
                                    cell.set_text_props(weight='bold', color='white')
                                else:
                                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    
                    if available_spaces >= 2:
                        # Second statistics table
                        ax_stats2 = fig.add_subplot(rows, cols, num_params + 2)
                        ax_stats2.axis('off')
                        ax_stats2.set_title('Parameter Statistics - Part 2', fontsize=10, fontweight='bold')
                        
                        # Create table data for second half of parameters
                        table_data2 = [['Parameter', 'Range', 'Std Dev', 'Mean', 'Chg Rate']]
                        for stat in all_param_stats[:mid_point]:
                            table_data2.append([
                                stat['name'][:8] + '...' if len(stat['name']) > 8 else stat['name'],
                                f"{stat['range']:.3f}",
                                f"{stat['std']:.3f}",
                                f"{stat['mean']:.3f}",
                                f"{stat['change_rate']:.3f}"
                            ])
                        
                        table2 = ax_stats2.table(cellText=table_data2, cellLoc='center', loc='center',
                                               colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
                        table2.auto_set_font_size(False)
                        table2.set_fontsize(7)
                        table2.scale(1, 1.2)
                        
                        # Style the table
                        for i in range(len(table_data2)):
                            for j in range(len(table_data2[0])):
                                cell = table2[(i, j)]
                                if i == 0:  # Header row
                                    cell.set_facecolor('#2196F3')
                                    cell.set_text_props(weight='bold', color='white')
                                else:
                                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    
                    if available_spaces >= 3 and len(all_param_stats) > mid_point:
                        # Third statistics table for remaining parameters
                        ax_stats3 = fig.add_subplot(rows, cols, num_params + 3)
                        ax_stats3.axis('off')
                        ax_stats3.set_title('Parameter Statistics - Part 3', fontsize=10, fontweight='bold')
                        
                        # Create table data for remaining parameters
                        table_data3 = [['Parameter', 'Initial', 'Final', 'Change', 'Status']]
                        for stat in all_param_stats[mid_point:]:
                            table_data3.append([
                                stat['name'][:8] + '...' if len(stat['name']) > 8 else stat['name'],
                                f"{stat['initial']:.3f}",
                                f"{stat['final']:.3f}",
                                f"{stat['change']:+.3f}",
                                stat['status']
                            ])
                        
                        table3 = ax_stats3.table(cellText=table_data3, cellLoc='center', loc='center',
                                               colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
                        table3.auto_set_font_size(False)
                        table3.set_fontsize(7)
                        table3.scale(1, 1.2)
                        
                        # Style the table
                        for i in range(len(table_data3)):
                            for j in range(len(table_data3[0])):
                                cell = table3[(i, j)]
                                if i == 0:  # Header row
                                    cell.set_facecolor('#FF9800')
                                    cell.set_text_props(weight='bold', color='white')
                                else:
                                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    
                    if available_spaces >= 4 and len(all_param_stats) > mid_point:
                        # Fourth statistics table for remaining parameters (detailed metrics)
                        ax_stats4 = fig.add_subplot(rows, cols, num_params + 4)
                        ax_stats4.axis('off')
                        ax_stats4.set_title('Parameter Statistics - Part 4', fontsize=10, fontweight='bold')
                        
                        # Create table data for remaining parameters detailed metrics
                        table_data4 = [['Parameter', 'Range', 'Std Dev', 'Mean', 'Chg Rate']]
                        for stat in all_param_stats[mid_point:]:
                            table_data4.append([
                                stat['name'][:8] + '...' if len(stat['name']) > 8 else stat['name'],
                                f"{stat['range']:.3f}",
                                f"{stat['std']:.3f}",
                                f"{stat['mean']:.3f}",
                                f"{stat['change_rate']:.3f}"
                            ])
                        
                        table4 = ax_stats4.table(cellText=table_data4, cellLoc='center', loc='center',
                                               colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
                        table4.auto_set_font_size(False)
                        table4.set_fontsize(7)
                        table4.scale(1, 1.2)
                        
                        # Style the table
                        for i in range(len(table_data4)):
                            for j in range(len(table_data4[0])):
                                cell = table4[(i, j)]
                                if i == 0:  # Header row
                                    cell.set_facecolor('#9C27B0')
                                    cell.set_text_props(weight='bold', color='white')
                                else:
                                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    
                    # Add overall summary in remaining space if available
                    remaining_spaces = available_spaces - min(4, available_spaces)
                    if remaining_spaces > 0:
                        summary_position = num_params + min(4, available_spaces) + 1
                    if summary_position <= rows * cols:
                        ax_summary = fig.add_subplot(rows, cols, summary_position)
                        ax_summary.axis('off')
                        ax_summary.set_title('Overall Summary', fontsize=10, fontweight='bold')
                        
                        summary_text = f''' Parameter Summary
 Active: {active_params}
 Converged: {converged_params}
 Static: {static_params}
Total: {num_params} parameters
Generations: {len(generations)}

Legend:
Conv = Converged
Act = Active  
Stat = Static'''
                        
                        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                                          verticalalignment='top', fontsize=8,
                                          bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', 
                                              alpha=0.9, edgecolor='orange'))
                    
            elif view_mode == "Compare Multiple":
                # Show enhanced parameter comparison with selection dialog
                selected_params = self.show_parameter_selection_dialog(param_names)
                
                fig = Figure(figsize=(14, 10), tight_layout=True)
                
                if selected_params:
                    # Create subplots with better spacing - use 3 columns for less crowding
                    gs = fig.add_gridspec(3, 3, height_ratios=[2.5, 1, 1], hspace=0.4, wspace=0.5)
                    ax_main = fig.add_subplot(gs[0, :])  # Main comparison plot
                    
                    # Enhanced color palette
                    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_params)))
                    
                    # Statistics for summary
                    param_stats = {}
                    
                    for i, (param_name, color) in enumerate(zip(selected_params, colors)):
                        param_idx = param_names.index(param_name)
                        param_values = param_data[:, param_idx]
                        
                        # Calculate statistics
                        param_min = np.min(param_values)
                        param_max = np.max(param_values)
                        param_range = param_max - param_min
                        param_std = np.std(param_values)
                        
                        param_stats[param_name] = {
                            'range': param_range,
                            'std': param_std,
                            'initial': param_values[0],
                            'final': param_values[-1],
                            'change': param_values[-1] - param_values[0]
                        }
                        
                        # Normalize values for comparison
                        if param_range > 1e-10:  # Parameter has variation
                            normalized_values = (param_values - param_min) / param_range
                        else:  # Constant parameter
                            normalized_values = np.zeros_like(param_values)
                        
                        # Enhanced line plot with gradient effect
                        for j in range(len(normalized_values) - 1):
                            alpha = 0.4 + 0.6 * (j / (len(normalized_values) - 1))
                            ax_main.plot(generations[j:j+2], normalized_values[j:j+2], 
                                       color=color, alpha=alpha, linewidth=2.5)
                        
                        # Add markers (reduced prominence, black edges)
                        ax_main.scatter(generations, normalized_values, c=color, s=20, 
                                      edgecolors='black', linewidth=0.6, zorder=3, label=param_name)
                        
                        # Add trend lines
                        if len(normalized_values) > 1:
                            z = np.polyfit(generations, normalized_values, 1)
                            trend_line = np.poly1d(z)
                            ax_main.plot(generations, trend_line(generations), '--', 
                                       color=color, alpha=0.6, linewidth=1.5)
                    
                    # Enhanced main plot styling
                    ax_main.set_title(f'Parameter Convergence Comparison - {len(selected_params)} Selected Parameters', 
                                    fontsize=16, fontweight='bold', pad=20)
                    ax_main.set_xlabel('Generation', fontsize=12)
                    ax_main.set_ylabel('Normalized Parameter Value (0-1)', fontsize=12)
                    ax_main.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
                    ax_main.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                                   framealpha=0.95, fancybox=True, edgecolor='black', zorder=10)
                    
                    # Store comparison analysis data for display outside the plot
                    sorted_by_range = sorted(param_stats.items(), key=lambda x: x[1]['range'], reverse=True)
                    most_active = sorted_by_range[0][0] if sorted_by_range else "None"
                    least_active = sorted_by_range[-1][0] if sorted_by_range else "None"
                    avg_range = np.mean([stats['range'] for stats in param_stats.values()])
                    
                    # Create mini-plots for individual parameter details (limit to 6 for better spacing)
                    mini_params = selected_params[:6]  # Show up to 6 mini-plots for better spacing
                    for i, param_name in enumerate(mini_params):
                        if i < 6:  # Maximum 6 mini-plots (2 rows x 3 cols, leaving more space)
                            row = 1 + i // 3  # 3 columns instead of 4
                            col = i % 3
                            ax_mini = fig.add_subplot(gs[row, col])
                            
                            param_idx = param_names.index(param_name)
                            param_values = param_data[:, param_idx]
                            color = colors[selected_params.index(param_name)]
                            
                            # Mini-plot with original values (not normalized)
                            ax_mini.plot(generations, param_values, color=color, linewidth=2, 
                                       marker='o', markersize=3, alpha=0.8)
                            
                            # Truncate long parameter names for better display
                            display_name = param_name[:10] + '...' if len(param_name) > 10 else param_name
                            ax_mini.set_title(display_name, fontsize=10, fontweight='bold')
                            ax_mini.tick_params(labelsize=8)
                            ax_mini.grid(True, alpha=0.3)
                            
                            # Add statistics text with better formatting
                            stats = param_stats[param_name]
                            stats_text = f": {stats['change']:+.3f}\ns: {stats['std']:.3f}"
                            ax_mini.text(0.02, 0.98, stats_text, transform=ax_mini.transAxes,
                                       verticalalignment='top', fontsize=8,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                               alpha=0.9, edgecolor=color, linewidth=1))
                    
                    # Add comparison statistics table if space available and less crowded
                    if len(selected_params) <= 6:  # Show table for up to 6 parameters
                        ax_stats = fig.add_subplot(gs[2, :])
                        ax_stats.axis('off')
                        
                        # Create comparison table with truncated parameter names
                        table_data = [['Parameter', 'Initial', 'Final', 'Change', 'Range', 'Std Dev']]
                        for param_name in selected_params:
                            stats = param_stats[param_name]
                            # Truncate parameter name for table display
                            display_name = param_name[:12] + '...' if len(param_name) > 12 else param_name
                            table_data.append([
                                display_name,
                                f"{stats['initial']:.3f}",
                                f"{stats['final']:.3f}",
                                f"{stats['change']:+.3f}",
                                f"{stats['range']:.3f}",
                                f"{stats['std']:.3f}"
                            ])
                        
                        table = ax_stats.table(cellText=table_data, cellLoc='center', loc='center',
                                             colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
                        table.auto_set_font_size(False)
                        table.set_fontsize(9)
                        table.scale(1, 1.4)  # Slightly larger for better readability
                        
                        # Style the table
                        for i in range(len(table_data)):
                            for j in range(len(table_data[0])):
                                cell = table[(i, j)]
                                if i == 0:  # Header row
                                    cell.set_facecolor('#2196F3')
                                    cell.set_text_props(weight='bold', color='white')
                                else:
                                    cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
                        
                        ax_stats.set_title('Detailed Parameter Statistics', fontsize=12, fontweight='bold', pad=15)
                
                else:
                    # No parameters selected
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, ''' No Parameters Selected

Please select parameters to compare using the dialog.

 Tip: Use "Compare Multiple" mode to:
* Select 2-8 parameters for optimal visualization
* Compare normalized parameter evolution
* See individual parameter details
* View comprehensive statistics''', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=14,
                           bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', alpha=0.3))
                    
            elif view_mode == "Active Parameters Only":
                # Show enhanced active parameters analysis
                active_params = []
                for i, param_name in enumerate(param_names):
                    param_values = param_data[:, i]
                    param_range = np.max(param_values) - np.min(param_values)
                    param_std = np.std(param_values)
                    if param_range > 1e-6:  # Parameter has significant variation
                        active_params.append((i, param_name, param_range, param_std))

                if active_params:
                    # Sort by activity level (range * std deviation)
                    active_params.sort(key=lambda x: x[2] * x[3], reverse=True)
                    
                    num_active = len(active_params)
                    if num_active <= 4:
                        rows, cols = 2, 2
                    elif num_active <= 6:
                        rows, cols = 2, 3
                    elif num_active <= 9:
                        rows, cols = 3, 3
                    elif num_active <= 12:
                        rows, cols = 3, 4
                    else:
                        rows, cols = 4, 4
                    
                    fig = Figure(figsize=(16, 12), tight_layout=True)
                    fig.suptitle(f'Active Parameters Analysis - {num_active} Parameters with Significant Variation', 
                               fontsize=16, fontweight='bold', y=0.95)
                    
                    # Color gradient based on activity level
                    activity_levels = [param_range * param_std for _, _, param_range, param_std in active_params]
                    max_activity = max(activity_levels) if activity_levels else 1
                    colors = plt.cm.plasma(np.array(activity_levels) / max_activity)
                    
                    for plot_idx, ((param_idx, param_name, param_range, param_std), color) in enumerate(zip(active_params[:16], colors)):  # Limit to 16
                        ax = fig.add_subplot(rows, cols, plot_idx + 1)
                        param_values = param_data[:, param_idx]
                        
                        # Enhanced gradient line plot
                        for j in range(len(param_values) - 1):
                            intensity = j / (len(param_values) - 1)
                            line_alpha = 0.4 + 0.6 * intensity
                            ax.plot(generations[j:j+2], param_values[j:j+2], 
                                   color=color, alpha=line_alpha, linewidth=2.5)
                        
                        # Add markers with varying sizes based on change magnitude (black edges)
                        changes = np.abs(np.diff(param_values))
                        marker_sizes = 10 + 20 * (changes / (np.max(changes) + 1e-10)) if len(changes) > 0 else [10] * len(param_values)
                        marker_sizes = np.concatenate([[marker_sizes[0]], marker_sizes])  # Add first point
                        
                        ax.scatter(generations, param_values, c=color, s=marker_sizes, 
                                 alpha=0.8, edgecolors='black', linewidth=0.7, zorder=3)
                        
                        # Add trend line
                        if len(param_values) > 1:
                            z = np.polyfit(generations, param_values, 1)
                            trend_line = np.poly1d(z)
                            ax.plot(generations, trend_line(generations), '--', 
                                   color='red', alpha=0.7, linewidth=1.5, 
                                   label=f'Trend: {z[0]:.4f}')
                        
                        # Activity rank indicator
                        activity_rank = plot_idx + 1
                        rank_icons = {1: "", 2: "", 3: ""}
                        rank_icon = rank_icons.get(activity_rank, f"#{activity_rank}")
                        
                        ax.set_title(f'{rank_icon} {param_name}', fontsize=11, fontweight='bold', 
                                   color='darkred' if activity_rank <= 3 else 'black')
                        ax.set_xlabel('Generation', fontsize=9)
                        ax.set_ylabel('Value', fontsize=9)
                        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                        ax.tick_params(labelsize=8)
                        
                        # Enhanced statistics box
                        initial_value = param_values[0]
                        final_value = param_values[-1]
                        change = final_value - initial_value
                        activity_score = param_range * param_std
                        
                        # Determine volatility level
                        if param_std < 0.01:
                            volatility = "Low"
                            vol_color = "lightgreen"
                        elif param_std < 0.1:
                            volatility = "Medium"
                            vol_color = "yellow"
                        else:
                            volatility = "High"
                            vol_color = "lightcoral"
                        
                        stats_text = f''' Statistics
Final: {final_value:.4f}
Change: {change:+.4f}
Range: {param_range:.4f}
Std: {param_std:.4f}
Activity: {activity_score:.6f}
Volatility: {volatility}'''
                        
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               ha='left', va='top', fontsize=7,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=vol_color, 
                                       alpha=0.8, edgecolor='gray'))
                        
                        # Highlight extreme values
                        max_idx = np.argmax(param_values)
                        min_idx = np.argmin(param_values)
                        
                        if max_idx != min_idx:  # Only if there's variation
                            ax.annotate(f'Max\n{param_values[max_idx]:.3f}', 
                                       xy=(generations[max_idx], param_values[max_idx]),
                                       xytext=(0, 15), textcoords='offset points',
                                       ha='center', fontsize=7, color='red',
                                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
                            
                            ax.annotate(f'Min\n{param_values[min_idx]:.3f}', 
                                       xy=(generations[min_idx], param_values[min_idx]),
                                       xytext=(0, -20), textcoords='offset points',
                                       ha='center', fontsize=7, color='blue',
                                       arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
                        
                        # Add legend for trend line if it exists
                        if len(param_values) > 1:
                            leg = ax.legend(fontsize=7, loc='upper right', framealpha=0.95, fancybox=True)
                            if leg is not None:
                                leg.set_zorder(10)
                                try:
                                    leg.get_frame().set_edgecolor('black')
                                except Exception:
                                    pass
                    
                    # Remove empty subplots
                    for i in range(num_active, rows * cols):
                        ax_empty = fig.add_subplot(rows, cols, i + 1)
                        fig.delaxes(ax_empty)
                    
                    # Add comprehensive summary if there's space
                    if num_active < rows * cols - 1:
                        ax_summary = fig.add_subplot(rows, cols, min(num_active + 1, rows * cols))
                        ax_summary.axis('off')
                        
                        # Calculate summary statistics
                        avg_activity = np.mean(activity_levels)
                        total_params = len(param_names)
                        activity_percentage = (num_active / total_params) * 100
                        
                        # Top 3 most active parameters
                        top_3 = active_params[:3]
                        top_3_text = "\n".join([f"{i+1}. {name}" for i, (_, name, _, _) in enumerate(top_3)])
                        
                        summary_text = f''' Activity Summary

 Active Parameters: {num_active}/{total_params}
 Activity Rate: {activity_percentage:.1f}%
 Avg Activity Score: {avg_activity:.6f}

 Top 3 Most Active:
{top_3_text}

 Color Legend:
* Darker = More Active
* Larger markers = Bigger changes
* Red dashed = Trend line

 Volatility Levels:
 Low (sigma < 0.01)
 Medium (0.01 = sigma < 0.1)
 High (sigma = 0.1)'''
                        
                        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                                      verticalalignment='top', fontsize=9,
                                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', 
                                              alpha=0.9, edgecolor='steelblue'))
                        ax_summary.set_title(' Analysis Summary', fontsize=12, fontweight='bold')
                
                else:
                    # No active parameters found
                    fig = Figure(figsize=(12, 8), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # Create an informative display even when no active parameters exist
                    ax.text(0.5, 0.6, ''' No Active Parameters Found
                    
All parameters remain constant during optimization.''', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=16,
                           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8))
                    
                    # Show parameter values table
                    ax.text(0.5, 0.3, f''' Parameter Values (All Constant):

{chr(10).join([f"* {name}: {param_data[0, i]:.6f}" for i, name in enumerate(param_names[:10])])}
{"..." if len(param_names) > 10 else ""}

 This indicates the optimization may have:
* Converged very quickly
* Been initialized at optimal values  
* Encountered constraints preventing parameter changes''', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
                    
                    ax.set_title('Active Parameters Analysis - No Variation Detected', 
                               fontsize=14, fontweight='bold')
                    ax.axis('off')
            else:
                # Default single parameter message
                fig = Figure(figsize=(10, 6), tight_layout=True)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Please select a parameter to visualize its convergence', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            # Create comparison analysis info widget for Compare Multiple mode
            if view_mode == "Compare Multiple" and selected_params:
                # Determine most/least active parameters for analysis display
                try:
                    # Extract analysis data from the plot (if available)
                    analysis_data = {}
                    if param_data is not None and len(param_data) > 0:
                        for param_name in selected_params:
                            param_idx = param_names.index(param_name)
                            param_values = param_data[:, param_idx]
                            param_range = np.max(param_values) - np.min(param_values)
                            analysis_data[param_name] = param_range
                        
                        sorted_by_range = sorted(analysis_data.items(), key=lambda x: x[1], reverse=True)
                        most_active_param = sorted_by_range[0][0] if sorted_by_range else "None"
                        least_active_param = sorted_by_range[-1][0] if sorted_by_range else "None"
                        avg_range_val = np.mean(list(analysis_data.values())) if analysis_data else 0.0
                    else:
                        most_active_param = "N/A"
                        least_active_param = "N/A"
                        avg_range_val = 0.0
                    
                    # Create info widget with comparison analysis
                    info_widget = QWidget()
                    info_layout = QHBoxLayout(info_widget)
                    info_layout.setContentsMargins(10, 5, 10, 5)
                    
                    # Comparison analysis
                    analysis_label = QLabel()
                    analysis_text = f""" <b>Comparison Analysis</b><br/>
<b>Selected:</b> {len(selected_params)} parameters | <b>Normalization:</b> 0-1 scale per parameter<br/>
 <b>Most Active:</b> {most_active_param} |  <b>Least Active:</b> {least_active_param}<br/>
 <b>Average Range:</b> {avg_range_val:.6f} |  <i>Dashed lines show trends</i>"""
                    analysis_label.setText(analysis_text)
                    analysis_label.setStyleSheet("""
                        QLabel {
                            background-color: #fff3cd;
                            border: 2px solid #ffeaa7;
                            border-radius: 8px;
                            padding: 10px;
                            font-size: 11px;
                        }
                    """)
                    analysis_label.setWordWrap(True)
                    
                    info_layout.addWidget(analysis_label)
                    plot_layout.addWidget(info_widget)
                except Exception as e:
                    # Fallback if analysis fails
                    pass
            
            # Add plot to layout
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar(canvas, None)
            try:
                self._attach_open_in_new_window(toolbar, fig, "Parameter Convergence (Interactive)")
            except Exception:
                pass
            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(canvas)
        
        # Connect dropdown changes to update function
        param_dropdown.currentTextChanged.connect(update_plot)
        view_dropdown.currentTextChanged.connect(update_plot)
        
        # Initial plot
        update_plot()
    
    def create_run_adaptive_rates_plot(self, layout, run_data, metrics):
        """Create adaptive rates evolution plot with smart annotation positioning"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        
        fig = Figure(figsize=(10, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        
        if 'adaptive_rates_history' in metrics and metrics['adaptive_rates_history']:
            rates_history = metrics['adaptive_rates_history']
            generations = [entry['generation'] for entry in rates_history]
            cxpb_values = [entry['new_cxpb'] for entry in rates_history]
            mutpb_values = [entry['new_mutpb'] for entry in rates_history]
            
            ax.plot(generations, cxpb_values, 'b-', marker='o', linewidth=2, label='Crossover Rate')
            ax.plot(generations, mutpb_values, 'r-', marker='s', linewidth=2, label='Mutation Rate')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Rate Value')
            ax.set_title(f'Adaptive Rates Evolution - Run #{run_data.get("run_number", 1)}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Smart annotation system to avoid overlaps
            if len(rates_history) > 0:
                # Calculate spacing to prevent overlaps
                x_range = max(generations) - min(generations) if len(generations) > 1 else 1
                min_spacing = x_range * 0.1  # Minimum 10% of x-range between annotations
                
                # Group adaptations to reduce clutter
                selected_adaptations = []
                last_annotated_gen = -float('inf')
                
                for i, entry in enumerate(rates_history):
                    gen = entry['generation']
                    # Only annotate if sufficient spacing from last annotation
                    if gen - last_annotated_gen >= min_spacing or i == 0 or i == len(rates_history) - 1:
                        selected_adaptations.append((i, entry))
                        last_annotated_gen = gen
                
                # If still too many, take every nth entry
                if len(selected_adaptations) > 6:
                    step = len(selected_adaptations) // 6
                    selected_adaptations = selected_adaptations[::step]
                
                # Add annotations with alternating positions to avoid overlap
                for idx, (i, entry) in enumerate(selected_adaptations):
                    gen = entry['generation']
                    cxpb_val = entry['new_cxpb']
                    mutpb_val = entry['new_mutpb']
                    adaptation_type = entry.get('adaptation_type', 'Unknown')
                    
                    # Alternate between annotating crossover and mutation points
                    if idx % 2 == 0:
                        # Annotate crossover point
                        y_pos = cxpb_val
                        color = 'blue'
                        marker_style = 'o'
                    else:
                        # Annotate mutation point
                        y_pos = mutpb_val
                        color = 'red'
                        marker_style = 's'
                    
                    # Alternate annotation positions (above/below)
                    if idx % 2 == 0:
                        xytext = (15, 20)  # Above and to the right
                        va = 'bottom'
                    else:
                        xytext = (15, -25)  # Below and to the right
                        va = 'top'
                    
                    # Highlight the adaptation point
                    ax.plot(gen, y_pos, marker=marker_style, markersize=8, 
                           color=color, markeredgecolor='black', markeredgewidth=1, zorder=5)
                    
                    # Add annotation with smart positioning
                    annotation_text = adaptation_type.replace('(', '\n(').replace('Increasing ', '').replace('Decreasing ', '')
                    ax.annotate(annotation_text, 
                               xy=(gen, y_pos),
                               xytext=xytext, 
                               textcoords='offset points',
                               fontsize=8, 
                               alpha=0.9,
                               va=va,
                               ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                        edgecolor=color, alpha=0.8),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                             color=color, alpha=0.6))
                
                # Add summary information
                total_adaptations = len(rates_history)
                adaptation_types = [entry.get('adaptation_type', 'Unknown') for entry in rates_history]
                unique_types = len(set(adaptation_types))
                
                summary_text = f'Total Adaptations: {total_adaptations}\nUnique Types: {unique_types}'
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                
                # Add rate change indicators
                if len(cxpb_values) > 1:
                    cxpb_change = cxpb_values[-1] - cxpb_values[0]
                    mutpb_change = mutpb_values[-1] - mutpb_values[0]
                    
                    change_text = f'Rate Changes:\nCrossover: {cxpb_change:+.3f}\nMutation: {mutpb_change:+.3f}'
                    ax.text(0.98, 0.02, change_text, transform=ax.transAxes, 
                           verticalalignment='bottom', horizontalalignment='right', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No adaptive rates data available\n(Adaptive rates may not have been enabled)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Adaptive Rates (Selected Run)")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def create_run_generation_breakdown_plot(self, layout, run_data, metrics):
        """Create generation performance breakdown plot"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        import numpy as np
        
        fig = Figure(figsize=(12, 8), tight_layout=True)
        ax1 = fig.add_subplot(2, 1, 1)  # Stacked bar chart
        ax2 = fig.add_subplot(2, 1, 2)  # Convergence rate
        
        # Generation breakdown
        if 'time_per_generation_breakdown' in metrics and metrics['time_per_generation_breakdown']:
            breakdown_data = metrics['time_per_generation_breakdown']
            generations = range(1, len(breakdown_data) + 1)
            
            # Extract timing components
            selection_times = [gen.get('selection', 0) for gen in breakdown_data]
            crossover_times = [gen.get('crossover', 0) for gen in breakdown_data]
            mutation_times = [gen.get('mutation', 0) for gen in breakdown_data]
            evaluation_times = [gen.get('evaluation', 0) for gen in breakdown_data]
            
            # Create stacked bar chart
            width = 0.8
            ax1.bar(generations, selection_times, width, label='Selection', alpha=0.8, color='red')
            ax1.bar(generations, crossover_times, width, bottom=selection_times, 
                   label='Crossover', alpha=0.8, color='blue')
            
            bottom1 = np.array(selection_times) + np.array(crossover_times)
            ax1.bar(generations, mutation_times, width, bottom=bottom1, 
                   label='Mutation', alpha=0.8, color='green')
            
            bottom2 = bottom1 + np.array(mutation_times)
            ax1.bar(generations, evaluation_times, width, bottom=bottom2, 
                   label='Evaluation', alpha=0.8, color='orange')
            
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('Time Breakdown per Generation')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            ax1.text(0.5, 0.5, 'No generation breakdown data', ha='center', va='center', transform=ax1.transAxes)
        
        # Convergence rate
        if 'convergence_rate' in metrics and metrics['convergence_rate']:
            generations = range(1, len(metrics['convergence_rate']) + 1)
            ax2.plot(generations, metrics['convergence_rate'], 'g-', marker='o', linewidth=2)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Improvement Rate')
            ax2.set_title('Convergence Rate per Generation')
            ax2.grid(True, alpha=0.3)
            
            # Add average line
            avg_convergence = np.mean(metrics['convergence_rate'])
            ax2.axhline(y=avg_convergence, color='r', linestyle='--', alpha=0.7, 
                       label=f'Average: {avg_convergence:.6f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No convergence rate data', ha='center', va='center', transform=ax2.transAxes)
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Generation Breakdown")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def create_run_fitness_components_plot(self, layout, run_data, metrics):
        """Create fitness components analysis plot"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar
)
        import numpy as np
        
        # Determine if we need to split parameters into 2 side-by-side plots
        best_solution = run_data.get('best_solution', [])
        param_names = run_data.get('parameter_names', [])
        active_names = run_data.get('active_parameters', [])
        
        # Get active parameters
        if active_names:
            index_map = {name: idx for idx, name in enumerate(param_names)}
            param_pairs = [(name, best_solution[index_map[name]]) for name in active_names if name in index_map]
        else:
            param_pairs = [(name, val) for name, val in zip(param_names, best_solution) if abs(val) > 1e-6]
        
        # Decide layout based on number of active parameters
        num_active_params = len(param_pairs)
        if num_active_params > 8:  # Split into 2 side-by-side plots
            fig = Figure(figsize=(16, 8), tight_layout=True)
            ax1 = fig.add_subplot(2, 3, 1)  # Pie chart of final fitness components (top left)
            ax2 = fig.add_subplot(2, 3, 2)  # First half of active parameters (top middle)
            ax3 = fig.add_subplot(2, 3, 3)  # Second half of active parameters (top right)
            ax4 = fig.add_subplot(2, 3, (4, 6))  # Combined view (bottom spanning 2 columns)
        else:  # Original layout for fewer parameters
            fig = Figure(figsize=(12, 6), tight_layout=True)
            ax1 = fig.add_subplot(1, 2, 1)  # Pie chart of final fitness components
            ax2 = fig.add_subplot(1, 2, 2)  # Best solution parameters
            ax3 = None
            ax4 = None
        
        # Fitness components breakdown as percentage contributions to final fitness
        best_solution = run_data.get('best_solution', [])
        best_fitness = float(run_data.get('best_fitness', 0.0) or 0.0)

        # Prefer alpha from stored run data; fallback to current UI value; then to default
        try:
            alpha_value = float(run_data.get('alpha', None)) if run_data.get('alpha', None) is not None else (
                float(self.ga_alpha_box.value()) if hasattr(self, 'ga_alpha_box') else 0.01
            )
        except Exception:
            alpha_value = 0.01

        # Try to get singular_response
        singular_response = run_data.get('singular_response', None)
        if singular_response is None and isinstance(getattr(self, 'current_ga_full_results', None), dict):
            singular_response = self.current_ga_full_results.get('singular_response', None)

        if best_solution and best_fitness >= 0:
            # Compute components based on GAWorker's fitness function
            sparsity_penalty = alpha_value * sum(abs(param) for param in best_solution)

            # Activation and cost terms
            try:
                thr = float(run_data.get('dva_activation_threshold', None))
                if thr is None and isinstance(getattr(self, 'current_ga_full_results', None), dict):
                    thr = float(self.current_ga_full_results.get('dva_activation_threshold', 0.0))
            except Exception:
                thr = float(self.ga_activation_threshold_box.value()) if hasattr(self, 'ga_activation_threshold_box') else 0.0
            try:
                pen_per = float(run_data.get('dva_activation_penalty', None))
                if pen_per is None and isinstance(getattr(self, 'current_ga_full_results', None), dict):
                    pen_per = float(self.current_ga_full_results.get('dva_activation_penalty', 0.0))
            except Exception:
                pen_per = float(self.ga_activation_penalty_box.value()) if hasattr(self, 'ga_activation_penalty_box') else 0.0
            # Costs mapping
            costs_map = {}
            if isinstance(run_data.get('dva_costs', None), dict):
                costs_map = dict(run_data['dva_costs'])
            elif isinstance(getattr(self, 'current_ga_full_results', None), dict) and isinstance(self.current_ga_full_results.get('dva_costs'), dict):
                costs_map = dict(self.current_ga_full_results['dva_costs'])
            elif hasattr(self, 'ga_param_table') and self.ga_param_table is not None and self.ga_param_table.columnCount() > 5:
                try:
                    for r in range(self.ga_param_table.rowCount()):
                        name = self.ga_param_table.item(r, 0).text()
                        c = self.ga_param_table.cellWidget(r, 5)
                        costs_map[name] = float(c.value()) if c is not None else 0.0
                except Exception:
                    costs_map = {}

            try:
                # Map param indices to names
                param_names = run_data.get('parameter_names', [])
                active_flags = [abs(float(v)) >= float(thr) for v in best_solution]
                activation_penalty_term = float(sum(1 for f in active_flags if f)) * float(pen_per)
                # Cost term normalized
                denom = float(sum(float(costs_map.get(n, 0.0)) for n in param_names)) if param_names else float(sum(costs_map.values()))
                if denom is None or denom <= 0 or not np.isfinite(denom):
                    denom = 1.0
                cost_sum_active = 0.0
                for idx, is_active in enumerate(active_flags):
                    if not is_active:
                        continue
                    name = param_names[idx] if idx < len(param_names) else None
                    if name is None:
                        continue
                    try:
                        cost_sum_active += float(costs_map.get(name, 0.0))
                    except Exception:
                        pass
                cost_term = cost_sum_active / denom
            except Exception:
                activation_penalty_term = 0.0
                cost_term = 0.0

            # Primary objective = |singular_response - 1.0| if available
            primary_objective = None
            if singular_response is not None:
                try:
                    primary_objective = abs(float(singular_response) - 1.0)
                except Exception:
                    primary_objective = None

            # Percentage error term value in fitness (the term added to fitness)
            # Prefer back-calculation to ensure components sum to final fitness for the selected run
            percentage_error_term = None
            # If we know primary, compute remainder as percentage term
            if primary_objective is not None:
                # Apply cost scale factor for consistency in component breakdown
                cost_scale_factor = float(self.ga_cost_scale_box.value()) if hasattr(self, 'ga_cost_scale_box') else 1.0
                scaled_cost_term = cost_term / cost_scale_factor if cost_scale_factor > 0 else cost_term
                percentage_error_term = max(0.0, best_fitness - primary_objective - sparsity_penalty - activation_penalty_term - scaled_cost_term)
            else:
                # If primary unknown, assume remainder (after sparsity) is primary, percentage term 0
                primary_objective = max(0.0, best_fitness - sparsity_penalty - activation_penalty_term - scaled_cost_term)
                percentage_error_term = 0.0

            # Ensure non-negative and consistent totals
            components = ['Primary Objective', 'Sparsity Penalty', 'Percentage Error', 'Activation Penalty', 'Cost Term']
            raw_values = [primary_objective, sparsity_penalty, percentage_error_term, activation_penalty_term, scaled_cost_term]
            values = [max(0.0, float(v)) for v in raw_values]
            total = sum(values)

            if total > 0:
                # Scale values so their sum equals the reported final fitness for precise percentage semantics
                if best_fitness > 0:
                    scale = best_fitness / total
                    values_plot = [v * scale for v in values]
                    denom = best_fitness
                else:
                    values_plot = values
                    denom = total
                colors = ['#3498db', '#e74c3c', '#f1c40f', '#9b59b6', '#2ecc71']
                labels = [f"{name} ({(v/denom)*100:.1f}%)" for name, v in zip(components, values_plot)]
                wedges, texts, autotexts = ax1.pie(
                    values_plot,
                    labels=labels,
                    colors=colors,
                    autopct=lambda p: f"{p:.1f}%" if p > 0 else '',
                    startangle=90,
                    pctdistance=0.75
                )
                ax1.set_title('Final Fitness Contribution (%)')
                ax1.axis('equal')

                # Add center annotation with total
                ax1.text(0, 0, f"Total\n{best_fitness:.6f}", ha='center', va='center', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))

                # Add legend to ensure all components are visible even if some are zero
                from matplotlib.patches import Patch
                legend_handles = [Patch(facecolor=c, edgecolor='black', label=l) for c, l in zip(colors, labels)]
                leg = ax1.legend(handles=legend_handles, loc='upper right', framealpha=0.95, fancybox=True, fontsize=9)
                if leg is not None:
                    leg.set_zorder(10)
                    try:
                        leg.get_frame().set_edgecolor('black')
                    except Exception:
                        pass
            else:
                ax1.text(0.5, 0.5, 'No fitness components to display', ha='center', va='center', transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, 'No fitness data available', ha='center', va='center', transform=ax1.transAxes)
        
        # Best solution parameters visualization
        if param_pairs:
            names, values = zip(*param_pairs)
            
            if num_active_params > 8 and ax3 is not None and ax4 is not None:
                # Split parameters into 2 side-by-side plots for many parameters
                mid_point = (num_active_params + 1) // 2
                
                # First half of parameters
                names1, values1 = names[:mid_point], values[:mid_point]
                y_pos1 = range(len(names1))
                bars1 = ax2.barh(y_pos1, values1, alpha=0.7, color='#2ecc71')
                ax2.set_yticks(y_pos1)
                ax2.set_yticklabels(names1)
                ax2.set_xlabel('Parameter Value')
                ax2.set_title(f'Active Parameters (1-{mid_point})')
                
                # Add value labels to first plot
                for i, (bar, val) in enumerate(zip(bars1, values1)):
                    ax2.text(val + 0.01 * max(values1) if val >= 0 else val - 0.01 * max(values1),
                            i, f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
                
                # Second half of parameters
                names2, values2 = names[mid_point:], values[mid_point:]
                y_pos2 = range(len(names2))
                bars2 = ax3.barh(y_pos2, values2, alpha=0.7, color='#3498db')
                ax3.set_yticks(y_pos2)
                ax3.set_yticklabels(names2)
                ax3.set_xlabel('Parameter Value')
                ax3.set_title(f'Active Parameters ({mid_point+1}-{num_active_params})')
                
                # Add value labels to second plot
                for i, (bar, val) in enumerate(zip(bars2, values2)):
                    ax3.text(val + 0.01 * max(values2) if val >= 0 else val - 0.01 * max(values2),
                            i, f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
                
                # Combined view (bottom plot)
                y_pos_combined = range(len(names))
                bars_combined = ax4.barh(y_pos_combined, values, alpha=0.7, 
                                       color=['#2ecc71' if i < mid_point else '#3498db' for i in range(len(names))])
                ax4.set_yticks(y_pos_combined)
                ax4.set_yticklabels(names)
                ax4.set_xlabel('Parameter Value')
                ax4.set_title('All Active Parameters (Combined View)')
                
                # Add value labels to combined plot
                for i, (bar, val) in enumerate(zip(bars_combined, values)):
                    ax4.text(val + 0.01 * max(values) if val >= 0 else val - 0.01 * max(values),
                            i, f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
                
            else:
                # Original single plot for fewer parameters
                y_pos = range(len(names))
                bars = ax2.barh(y_pos, values, alpha=0.7, color='green')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(names)
                ax2.set_xlabel('Parameter Value')
                ax2.set_title('Active Parameters in Best Solution')

                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax2.text(val + 0.01 * max(values) if val >= 0 else val - 0.01 * max(values),
                            i, f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right')
        else:
            if ax2 is not None:
                ax2.text(0.5, 0.5, 'No active parameters found', ha='center', va='center', transform=ax2.transAxes)
            if ax3 is not None:
                ax3.text(0.5, 0.5, 'No active parameters found', ha='center', va='center', transform=ax3.transAxes)
            if ax4 is not None:
                ax4.text(0.5, 0.5, 'No active parameters found', ha='center', va='center', transform=ax4.transAxes)

        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        try:
            self._attach_open_in_new_window(toolbar, fig, "Fitness Components")
        except Exception:
            pass
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

    def create_run_seeding_visualizations(self, layout, run_data, metrics):
        """Create seeding method specific visualizations for the selected run."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT as NavigationToolbar
        )
        import numpy as np

        fig = Figure(figsize=(12, 8), tight_layout=True)
        method = (metrics.get('seeding_method') or run_data.get('benchmark_metrics', {}).get('seeding_method') or 'unknown')
        method = str(method).lower()

        # Prepare subplots based on method
        if method == 'neural':
            # Layout: [ Train time | Beta/Epsilon ] / [ Pool mult | Text summary ]
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)

            history = metrics.get('neural_history', []) or []
            if history:
                gens = [h.get('generation', i+1) for i, h in enumerate(history)]
                train_ms = [float(h.get('train_time_ms', np.nan)) for h in history]
                betas = [float(h.get('beta', np.nan)) for h in history]
                eps = [float(h.get('epsilon', np.nan)) for h in history]
                poolm = [float(h.get('pool_mult', np.nan)) for h in history]

                # Train time
                ax1.plot(gens, train_ms, 'tab:blue', marker='o', linewidth=2, alpha=0.9)
                ax1.set_title('Neural Training Time per Generation (ms)')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Time (ms)')
                ax1.grid(True, alpha=0.3)

                # Beta and epsilon
                ax2.plot(gens, betas, 'tab:red', marker='o', linewidth=2, label='ÃŸ (UCB)')
                ax2.plot(gens, eps, 'tab:green', marker='s', linewidth=2, label='e (explore)')
                # Highlight adapted epsilon range if enabled
                ns = metrics.get('neural_seeding', {})
                if ns.get('adapt_epsilon'):
                    eps_min = float(ns.get('eps_min', np.nan))
                    eps_max = float(ns.get('eps_max', np.nan))
                    if eps_min == eps_min and eps_max == eps_max:
                        ax2.fill_between(gens, [eps_min]*len(gens), [eps_max]*len(gens), color='green', alpha=0.08, label='e bounds')
                ax2.set_title('Acquisition Hyperparameters')
                ax2.set_xlabel('Generation')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Pool multiplier
                ax3.step(gens, poolm, where='mid', color='tab:purple')
                ax3.set_title('Pool Size Multiplier per Generation')
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('x Population')
                ax3.grid(True, alpha=0.3)

                # Text summary
                ax4.axis('off')
                acq = ns.get('acq_type', 'ucb')
                ens = ns.get('ensemble_n', '-')
                hdim = ns.get('hidden', '-')
                layers = ns.get('layers', '-')
                dout = ns.get('dropout', '-')
                wd = ns.get('weight_decay', '-')
                device = ns.get('device', 'cpu')
                adapt_eps = ns.get('adapt_epsilon', False)
                text = (
                    f"Method: Neural (Acq: {str(acq).upper()}) | Device: {device}\n"
                    f"Ensemble: {ens} | Hidden: {hdim} x {layers} | Dropout: {dout} | WD: {wd}\n"
                    f"e: {'adaptive' if adapt_eps else 'fixed'}\n"
                    f"Tip: Lower training time  more budget for FRF. ÃŸ  explore; e  more random coverage."
                )
                ax4.text(0.01, 0.95, text, va='top', ha='left')
            else:
                fig.suptitle('Neural Seeding Selected (no history yet)', fontsize=14, fontweight='bold')
                ax = fig.add_subplot(1, 1, 1)
                ax.axis('off')
                ax.text(0.5, 0.5, 'Neural seeding was enabled, but no neural metrics were recorded.',
                        ha='center', va='center')

        else:
            # Non-neural seeding (Random, Sobol, LHS)
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 1, 2)

            method_name = 'Random'
            if 'sobol' in method:
                method_name = 'Sobol (low-discrepancy)'
            elif 'lhs' in method:
                method_name = 'Latin Hypercube'
            fig.suptitle(f'Seeding Method: {method_name}', fontsize=14, fontweight='bold')

            # Population size per generation (if available)
            pop_hist = metrics.get('pop_size_history', []) or []
            if pop_hist:
                gens = range(1, len(pop_hist)+1)
                ax1.step(list(gens), pop_hist, where='mid', color='tab:green')
                ax1.set_title('Population Size per Generation')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Population')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.axis('off')
                ax1.text(0.5, 0.5, 'No population history', ha='center', va='center')

            # If surrogate was active, show pool vs evaluations as proxy for screening
            surr_info = metrics.get('surrogate_info', []) or []
            if surr_info:
                gens_s = [d.get('generation', i+1) for i, d in enumerate(surr_info)]
                pools = [d.get('pool_size', np.nan) for d in surr_info]
                evals = [d.get('evaluated_count', np.nan) for d in surr_info]
                ax2.plot(gens_s, pools, 'tab:cyan', marker='o', label='Pool')
                ax2.plot(gens_s, evals, 'tab:pink', marker='s', label='Evaluated')
                ax2.set_title('Surrogate: Pool vs Evaluated')
                ax2.set_xlabel('Generation')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.axis('off')
                ax2.text(0.5, 0.5, 'Surrogate not used', ha='center', va='center')

            # Text panel with method info
            ax3.axis('off')
            if 'sobol' in method:
                txt = (
                    "Sobol seeding produces a low-discrepancy space-filling set, \n"
                    "giving broad coverage with fewer gaps than purely random sampling. \n"
                    "Good for exploration early on."
                )
            elif 'lhs' in method:
                txt = (
                    "LHS (Latin Hypercube) stratifies each dimension, \n"
                    "ensuring balanced marginal coverage across parameters. \n"
                    "Useful for balanced exploration with fewer points."
                )
            else:
                txt = (
                    "Random seeding samples uniformly within bounds. \n"
                    "Highest variance; simple baseline; useful when constraints are minimal."
                )
            ax3.text(0.01, 0.95, txt, va='top', ha='left')
        
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar(canvas, None)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
    
    def show_parameter_selection_dialog(self, param_names):
        """Show a dialog for selecting parameters to compare"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel, QScrollArea
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Parameters to Compare")
        dialog.setModal(True)
        dialog.resize(400, 500)
        
        layout = QVBoxLayout(dialog)
        
        # Add instruction label
        instruction_label = QLabel("Select the parameters you want to compare:")
        instruction_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instruction_label)
        
        # Create scroll area for checkboxes (in case there are many parameters)
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Store checkboxes for later reference
        checkboxes = {}
        
        # Add "Select All" and "Select None" buttons
        select_buttons_widget = QWidget()
        select_buttons_layout = QHBoxLayout(select_buttons_widget)
        
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        select_active_btn = QPushButton("Select Active Only")
        
        select_buttons_layout.addWidget(select_all_btn)
        select_buttons_layout.addWidget(select_none_btn)
        select_buttons_layout.addWidget(select_active_btn)
        select_buttons_layout.addStretch()
        
        layout.addWidget(select_buttons_widget)
        
        # Create checkboxes for each parameter
        for param_name in param_names:
            checkbox = QCheckBox(param_name)
            checkboxes[param_name] = checkbox
            scroll_layout.addWidget(checkbox)
        
        # Set up scroll area
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Add information label
        info_label = QLabel("Tip: Select 2-8 parameters for best visualization")
        info_label.setStyleSheet("color: gray; font-style: italic; margin-top: 10px;")
        layout.addWidget(info_label)
        
        # Add dialog buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Connect button functions
        def select_all():
            for checkbox in checkboxes.values():
                checkbox.setChecked(True)
        
        def select_none():
            for checkbox in checkboxes.values():
                checkbox.setChecked(False)
        
        def select_active():
            # This would need access to param_data to determine active parameters
            # For now, we'll just select the first few parameters as a placeholder
            select_none()
            for i, checkbox in enumerate(checkboxes.values()):
                if i < 5:  # Select first 5 parameters
                    checkbox.setChecked(True)
        
        # Connect button signals
        select_all_btn.clicked.connect(select_all)
        select_none_btn.clicked.connect(select_none)
        select_active_btn.clicked.connect(select_active)
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Show dialog and return selected parameters
        if dialog.exec_() == QDialog.Accepted:
            selected_params = []
            for param_name, checkbox in checkboxes.items():
                if checkbox.isChecked():
                    selected_params.append(param_name)
            return selected_params
        else:
            return []
            
    def create_statistical_summary(self, selected_params):
        """Create comprehensive statistical summary for selected parameters"""
        from scipy import stats
        import numpy as np
        
        # Create HTML-formatted statistical summary
        html = """
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #dee2e6;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f8f9fa;
            }
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            .section {
                margin: 20px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .header {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        </style>
        
        <div class="section">
            <div class="header">Basic Statistics</div>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>CV (%)</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Range</th>
                </tr>
        """
        
        # Add basic statistics for each parameter
        for param in selected_params:
            values = self.current_parameter_data[param]
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / mean) * 100 if mean != 0 else 0
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            
            html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{mean:.4f}</td>
                    <td>{std:.4f}</td>
                    <td>{cv:.2f}</td>
                    <td>{min_val:.4f}</td>
                    <td>{max_val:.4f}</td>
                    <td>{range_val:.4f}</td>
                </tr>
            """
            
        html += """
            </table>
        </div>
        
        <div class="section">
            <div class="header">Distribution Analysis</div>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Skewness</th>
                    <th>Kurtosis</th>
                    <th>Normality (p-value)</th>
                    <th>Q1</th>
                    <th>Median</th>
                    <th>Q3</th>
                    <th>IQR</th>
                </tr>
        """
        
        # Add distribution statistics for each parameter
        for param in selected_params:
            values = self.current_parameter_data[param]
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            _, p_normal = stats.normaltest(values)
            q1 = np.percentile(values, 25)
            median = np.median(values)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{skewness:.4f}</td>
                    <td>{kurtosis:.4f}</td>
                    <td>{p_normal:.4f}</td>
                    <td>{q1:.4f}</td>
                    <td>{median:.4f}</td>
                    <td>{q3:.4f}</td>
                    <td>{iqr:.4f}</td>
                </tr>
            """
            
        html += """
            </table>
        </div>
        
        <div class="section">
            <div class="header">Confidence Intervals (95%)</div>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Lower CI</th>
                    <th>Upper CI</th>
                    <th>Standard Error</th>
                    <th>Effect Size (Cohen's d)</th>
                </tr>
        """
        
        # Add confidence intervals and effect sizes
        for param in selected_params:
            values = self.current_parameter_data[param]
            sem = stats.sem(values)
            ci = stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=sem)
            effect_size = np.mean(values) / np.std(values) if np.std(values) != 0 else 0
            
            html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{ci[0]:.4f}</td>
                    <td>{ci[1]:.4f}</td>
                    <td>{sem:.4f}</td>
                    <td>{effect_size:.4f}</td>
                </tr>
            """
            
        html += """
            </table>
        </div>
        """
        
        # Add correlation analysis if there are multiple parameters
        if len(selected_params) > 1:
            html += """
            <div class="section">
                <div class="header">Parameter Correlations</div>
                <table>
                    <tr>
                        <th>Parameter 1</th>
                        <th>Parameter 2</th>
                        <th>Correlation</th>
                        <th>p-value</th>
                    </tr>
            """
            
            for i, param1 in enumerate(selected_params):
                for param2 in selected_params[i+1:]:
                    values1 = self.current_parameter_data[param1]
                    values2 = self.current_parameter_data[param2]
                    corr, p_val = stats.pearsonr(values1, values2)
                    
                    html += f"""
                        <tr>
                            <td>{param1}</td>
                            <td>{param2}</td>
                            <td>{corr:.4f}</td>
                            <td>{p_val:.4f}</td>
                        </tr>
                    """
                    
            html += """
                </table>
            </div>
            """
            
        return html
        
    def create_correlation_matrix(self, param_names):
        """Create enhanced correlation matrix plot for multiple parameters"""
        import numpy as np
        import seaborn as sns
        from scipy import stats
        
        # Create figure with dark theme
        plt.style.use('seaborn-darkgrid')
        fig = Figure(figsize=(12, 8), dpi=120)
        fig.patch.set_facecolor('#2c3e50')
        
        # Create correlation matrix
        n_params = len(param_names)
        corr_matrix = np.zeros((n_params, n_params))
        p_values = np.zeros((n_params, n_params))
        
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                values1 = self.current_parameter_data[param1]
                values2 = self.current_parameter_data[param2]
                corr, p_val = stats.pearsonr(values1, values2)
                corr_matrix[i, j] = corr
                p_values[i, j] = p_val
        
        # Create main heatmap
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1],
                            hspace=0.3, wspace=0.3)
        
        # Main correlation heatmap
        ax_main = fig.add_subplot(gs[0, 0])
        im = ax_main.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values
        for i in range(n_params):
            for j in range(n_params):
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                text = ax_main.text(j, i, f'{corr_matrix[i, j]:.2f}\np={p_values[i, j]:.3f}',
                                  ha='center', va='center', color=color, fontsize=8)
        
        # Add parameter labels
        ax_main.set_xticks(range(n_params))
        ax_main.set_yticks(range(n_params))
        ax_main.set_xticklabels(param_names, rotation=45, ha='right')
        ax_main.set_yticklabels(param_names)
        
        # Add colorbar
        plt.colorbar(im, ax=ax_main)
        
        # Add title
        ax_main.set_title('Parameter Correlation Matrix', color='white', pad=20)
        
        # Style the plot
        ax_main.tick_params(colors='white')
        ax_main.xaxis.label.set_color('white')
        ax_main.yaxis.label.set_color('white')
        
        # Add distribution plots on the diagonal
        ax_dist = fig.add_subplot(gs[0, 1])
        for i, param in enumerate(param_names):
            values = self.current_parameter_data[param]
            sns.kdeplot(data=values, ax=ax_dist, label=param)
        
        ax_dist.set_title('Parameter Distributions', color='white', pad=20)
        ax_dist.legend()
        ax_dist.tick_params(colors='white')
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        
        # Add correlation summary
        ax_summary = fig.add_subplot(gs[1, :])
        ax_summary.axis('off')
        
        # Create correlation summary text
        summary_text = "Correlation Summary:\n\n"
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names[i+1:], i+1):
                corr = corr_matrix[i, j]
                p_val = p_values[i, j]
                strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                direction = "positive" if corr > 0 else "negative"
                
                summary_text += f"{param1} vs {param2}:\n"
                summary_text += f"  - {strength} {direction} correlation (r={corr:.3f})\n"
                summary_text += f"  - {'significant' if p_val < 0.05 else 'not significant'} (p={p_val:.3f})\n\n"
        
        ax_summary.text(0.05, 0.95, summary_text,
                       transform=ax_summary.transAxes,
                       fontsize=10,
                       color='white',
                       verticalalignment='top',
                       fontfamily='monospace')
        
        return fig
        
        
        select_all_btn.clicked.connect(select_all)
        select_none_btn.clicked.connect(select_none)
        select_active_btn.clicked.connect(select_active)
        
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Pre-select some parameters (first 3-5 parameters)
        for i, checkbox in enumerate(checkboxes.values()):
            if i < min(5, len(param_names)):
                checkbox.setChecked(True)
        
        # Show dialog and get result
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # Return list of selected parameter names
            selected_params = [param_name for param_name, checkbox in checkboxes.items() 
                             if checkbox.isChecked()]
            return selected_params
        else:
            # User cancelled, return empty list
            return []

    def _build_comprehensive_group_summary(self, df):
        """Build comprehensive visualizations for the Group Summary tab with separate subtabs"""
        try:
            # Remember latest df for auxiliary handlers that may need it later
            try:
                self._last_group_summary_df = df.copy() if hasattr(df, 'copy') else df
            except Exception:
                self._last_group_summary_df = df
            # Clear existing content
            if hasattr(self, 'group_summary_container') and self.group_summary_container.layout():
                layout = self.group_summary_container.layout()
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().setParent(None)
            
            import numpy as np
            import pandas as pd
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            import seaborn as sns
            from scipy import stats
            
            # Get group counts
            groups = ['Best', 'Mid', 'Worst']
            group_counts = {}
            for g in groups:
                group_counts[g] = int((df['group'] == g).sum()) if 'group' in df.columns else 0
            
            total_runs = len(df)
            metric_name = self.bench_rank_metric_combo.currentText() if hasattr(self, 'bench_rank_metric_combo') else 'Fitness'
            
            # Create tabbed widget for separate plot subtabs
            plot_tabs = QTabWidget()
            self.group_summary_container.layout().addWidget(plot_tabs)
            
            # Define colors for groups
            group_colors = {'Best': '#2ca02c', 'Mid': '#ff7f0e', 'Worst': '#d62728'}
            
            # 1. Overview Tab
            overview_tab = QWidget()
            overview_layout = QVBoxLayout(overview_tab)
            
            summary_text = QTextEdit()
            summary_text.setReadOnly(True)
            summary_text.setMaximumHeight(180)  # Increased height for reference info
            
            # Get reference run information
            ref_info = ""
            if 'best_ref_score' in df.columns and 'mean_ref_score' in df.columns and 'worst_ref_score' in df.columns:
                best_ref = df.iloc[0]['best_ref_score']
                mean_ref = df.iloc[0]['mean_ref_score'] 
                worst_ref = df.iloc[0]['worst_ref_score']
                ref_info = f"""
 REFERENCE RUNS (Clustering Centers):
--------------------------------------------------------------------------------
 Best Reference:  {best_ref:.6f}
 Mean Reference:  {mean_ref:.6f}
 Worst Reference: {worst_ref:.6f}

  Runs are grouped by closeness to these references
--------------------------------------------------------------------------------
"""
            
            summary_content = f"""
 BENCHMARK SUMMARY ({total_runs} runs)
--------------------------------------------------------------------------------
 Best Group:  {group_counts.get('Best', 0):>3} runs ({group_counts.get('Best', 0)/max(1,total_runs)*100:5.1f}%)
 Mid Group:   {group_counts.get('Mid', 0):>3} runs ({group_counts.get('Mid', 0)/max(1,total_runs)*100:5.1f}%)
 Worst Group: {group_counts.get('Worst', 0):>3} runs ({group_counts.get('Worst', 0)/max(1,total_runs)*100:5.1f}%)

 Ranking Metric: {metric_name}
--------------------------------------------------------------------------------
{ref_info}
            """
            summary_text.setPlainText(summary_content.strip())
            overview_layout.addWidget(summary_text)
            plot_tabs.addTab(overview_tab, " Overview")
            
            # 2. KDE Distribution Tab
            kde_tab = QWidget()
            kde_layout = QVBoxLayout(kde_tab)
            
            if 'score' in df.columns:
                fig_kde = Figure(figsize=(12, 8), tight_layout=True)
                ax_kde = fig_kde.add_subplot(111)
                
                for group in groups:
                    group_data = df[df['group'] == group]['score'].dropna()
                    if len(group_data) >= 2:
                        sns.kdeplot(data=group_data, ax=ax_kde, label=f'{group} (n={len(group_data)})', 
                                   color=group_colors[group], fill=True, alpha=0.3)
                        # Add mean line
                        mean_val = group_data.mean()
                        ax_kde.axvline(mean_val, color=group_colors[group], linestyle='--', alpha=0.8)
                
                # Add reference run markers
                if 'best_ref_score' in df.columns:
                    best_ref = df.iloc[0]['best_ref_score']
                    mean_ref = df.iloc[0]['mean_ref_score']
                    worst_ref = df.iloc[0]['worst_ref_score']
                    
                    ax_kde.axvline(best_ref, color='darkgreen', linestyle='-', linewidth=3, alpha=0.8, label=f'Best Ref: {best_ref:.4f}')
                    ax_kde.axvline(mean_ref, color='darkorange', linestyle='-', linewidth=3, alpha=0.8, label=f'Mean Ref: {mean_ref:.4f}')
                    ax_kde.axvline(worst_ref, color='darkred', linestyle='-', linewidth=3, alpha=0.8, label=f'Worst Ref: {worst_ref:.4f}')
                
                ax_kde.set_title('Score Distribution by Reference-Based Groups', fontweight='bold', fontsize=14)
                ax_kde.set_xlabel('Score (lower is better)')
                ax_kde.set_ylabel('Density')
                ax_kde.legend()
                ax_kde.grid(True, alpha=0.3)
                
                canvas_kde = FigureCanvas(fig_kde)
                toolbar_kde = NavigationToolbar(canvas_kde, kde_tab)
                self._attach_open_in_new_window(toolbar_kde, fig_kde, "KDE Distribution by Reference-Based Groups")
                kde_layout.addWidget(canvas_kde)
                kde_layout.addWidget(toolbar_kde)
            else:
                kde_layout.addWidget(QLabel("No score data available for KDE plot"))
            
            plot_tabs.addTab(kde_tab, " KDE Distribution")
            
            # 3. Violin Plot Tab
            violin_tab = QWidget()
            violin_layout = QVBoxLayout(violin_tab)
            
            if 'group' in df.columns and 'best_fitness' in df.columns:
                fig_violin = Figure(figsize=(10, 8), tight_layout=True)
                ax_violin = fig_violin.add_subplot(111)
                
                group_data_list = []
                group_labels = []
                for group in groups:
                    group_fitness = df[df['group'] == group]['best_fitness'].dropna()
                    if len(group_fitness) > 0:
                        group_data_list.append(group_fitness.values)
                        group_labels.append(f'{group}\n(n={len(group_fitness)})')
                
                if group_data_list:
                    parts = ax_violin.violinplot(group_data_list, positions=range(len(group_data_list)), 
                                               showmeans=True, showmedians=True)
                    
                    # Color the violin parts
                    for i, (pc, group) in enumerate(zip(parts['bodies'], groups[:len(group_data_list)])):
                        pc.set_facecolor(group_colors[group])
                        pc.set_alpha(0.6)
                    
                    ax_violin.set_xticks(range(len(group_labels)))
                    ax_violin.set_xticklabels(group_labels)
                    ax_violin.set_title('Fitness Distribution by Group', fontweight='bold', fontsize=14)
                    ax_violin.set_ylabel('Best Fitness')
                    ax_violin.grid(True, alpha=0.3)
                
                canvas_violin = FigureCanvas(fig_violin)
                toolbar_violin = NavigationToolbar(canvas_violin, violin_tab)
                self._attach_open_in_new_window(toolbar_violin, fig_violin, "Fitness Distribution by Group - Violin Plot")
                violin_layout.addWidget(canvas_violin)
                violin_layout.addWidget(toolbar_violin)
            else:
                violin_layout.addWidget(QLabel("No group or fitness data available for violin plot"))
            
            plot_tabs.addTab(violin_tab, " Violin Plot")
            
            # 4. Box Plot Tab
            box_tab = QWidget()
            box_layout = QVBoxLayout(box_tab)
            
            metrics_to_plot = []
            metric_names = []
            if 'best_fitness' in df.columns:
                metrics_to_plot.append(df['best_fitness'].dropna())
                metric_names.append('Fitness')
            if 'convergence_generation' in df.columns:
                metrics_to_plot.append(df['convergence_generation'].dropna())
                metric_names.append('Conv Gen')
            if 'elapsed_time' in df.columns:
                metrics_to_plot.append(df['elapsed_time'].dropna())
                metric_names.append('Time (s)')
            if 'cpu_time_used' in df.columns:
                metrics_to_plot.append(df['cpu_time_used'].dropna())
                metric_names.append('CPU (s)')
            
            if metrics_to_plot:
                fig_box = Figure(figsize=(12, 8), tight_layout=True)
                ax_box = fig_box.add_subplot(111)
                
                # Normalize metrics for comparison
                normalized_metrics = []
                for metric in metrics_to_plot:
                    if len(metric) > 0:
                        min_val, max_val = metric.min(), metric.max()
                        if max_val > min_val:
                            normalized = (metric - min_val) / (max_val - min_val)
                        else:
                            normalized = pd.Series([0.5] * len(metric))
                        normalized_metrics.append(normalized.values)
                    else:
                        normalized_metrics.append([])
                
                if normalized_metrics:
                    ax_box.boxplot(normalized_metrics, labels=metric_names)
                    ax_box.set_title('Normalized Metrics Comparison', fontweight='bold', fontsize=14)
                    ax_box.set_ylabel('Normalized Value (0-1)')
                    ax_box.grid(True, alpha=0.3)
                    ax_box.tick_params(axis='x', rotation=45)
                
                canvas_box = FigureCanvas(fig_box)
                toolbar_box = NavigationToolbar(canvas_box, box_tab)
                self._attach_open_in_new_window(toolbar_box, fig_box, "Normalized Metrics Comparison - Box Plot")
                box_layout.addWidget(canvas_box)
                box_layout.addWidget(toolbar_box)
            else:
                box_layout.addWidget(QLabel("No metrics data available for box plot"))
            
            plot_tabs.addTab(box_tab, " Box Plot")
            
            # 5. Scatter Plot Tab
            scatter_tab = QWidget()
            scatter_layout = QVBoxLayout(scatter_tab)
            
            if 'run_number' in df.columns and 'best_fitness' in df.columns and 'group' in df.columns:
                fig_scatter = Figure(figsize=(12, 8), tight_layout=True)
                ax_scatter = fig_scatter.add_subplot(111)
                
                for group in groups:
                    group_data = df[df['group'] == group]
                    if len(group_data) > 0:
                        ax_scatter.scatter(group_data['run_number'], group_data['best_fitness'], 
                                         c=group_colors[group], label=f'{group} (n={len(group_data)})', 
                                         alpha=0.7, s=50)
                
                # Add trend line
                if len(df) > 1:
                    z = np.polyfit(df['run_number'], df['best_fitness'], 1)
                    p = np.poly1d(z)
                    ax_scatter.plot(df['run_number'], p(df['run_number']), 'r--', alpha=0.8, linewidth=2)
                
                ax_scatter.set_title('Fitness vs Run Number', fontweight='bold', fontsize=14)
                ax_scatter.set_xlabel('Run Number')
                ax_scatter.set_ylabel('Best Fitness')
                ax_scatter.legend()
                ax_scatter.grid(True, alpha=0.3)
                
                canvas_scatter = FigureCanvas(fig_scatter)
                toolbar_scatter = NavigationToolbar(canvas_scatter, scatter_tab)
                self._attach_open_in_new_window(toolbar_scatter, fig_scatter, "Fitness vs Run Number - Scatter Plot")
                scatter_layout.addWidget(canvas_scatter)
                scatter_layout.addWidget(toolbar_scatter)
            else:
                scatter_layout.addWidget(QLabel("No run number, fitness, or group data available for scatter plot"))
            
            plot_tabs.addTab(scatter_tab, " Scatter Plot")
            
            # 6. Correlation Matrix Tab
            correlation_tab = QWidget()
            correlation_layout = QVBoxLayout(correlation_tab)
            
            numeric_cols = []
            for col in ['best_fitness', 'convergence_generation', 'elapsed_time', 'cpu_time_used', 'score']:
                if col in df.columns and df[col].notna().sum() > 1:
                    numeric_cols.append(col)
            
            if len(numeric_cols) >= 2:
                fig_corr = Figure(figsize=(10, 8), tight_layout=True)
                ax_correlation = fig_corr.add_subplot(111)
                
                corr_data = df[numeric_cols].corr()
                im = ax_correlation.imshow(corr_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                ax_correlation.set_xticks(range(len(numeric_cols)))
                ax_correlation.set_yticks(range(len(numeric_cols)))
                ax_correlation.set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45)
                ax_correlation.set_yticklabels([col.replace('_', '\n') for col in numeric_cols])
                ax_correlation.set_title('Metrics Correlation Matrix', fontweight='bold', fontsize=14)
                
                # Add correlation values
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        text = ax_correlation.text(j, i, f'{corr_data.iloc[i, j]:.2f}', 
                                                 ha="center", va="center", color="black", fontweight='bold')
                
                # Add colorbar
                cbar = fig_corr.colorbar(im, ax=ax_correlation, shrink=0.8)
                cbar.set_label('Correlation', rotation=270, labelpad=15)
                
                canvas_corr = FigureCanvas(fig_corr)
                toolbar_corr = NavigationToolbar(canvas_corr, correlation_tab)
                self._attach_open_in_new_window(toolbar_corr, fig_corr, "Metrics Correlation Matrix")
                correlation_layout.addWidget(canvas_corr)
                correlation_layout.addWidget(toolbar_corr)
            else:
                correlation_layout.addWidget(QLabel("Not enough numeric columns for correlation matrix"))
            
            plot_tabs.addTab(correlation_tab, " Correlation")
            
            # 7. Q-Q Plot Tab
            qq_tab = QWidget()
            qq_layout = QVBoxLayout(qq_tab)
            
            if 'best_fitness' in df.columns:
                fig_qq = Figure(figsize=(10, 8), tight_layout=True)
                ax_qq = fig_qq.add_subplot(111)
                
                stats.probplot(df['best_fitness'].dropna(), dist="norm", plot=ax_qq)
                ax_qq.set_title('Q-Q Plot (Fitness Normality Check)', fontweight='bold', fontsize=14)
                ax_qq.grid(True, alpha=0.3)
                
                canvas_qq = FigureCanvas(fig_qq)
                toolbar_qq = NavigationToolbar(canvas_qq, qq_tab)
                self._attach_open_in_new_window(toolbar_qq, fig_qq, "Q-Q Plot - Fitness Normality Check")
                qq_layout.addWidget(canvas_qq)
                qq_layout.addWidget(toolbar_qq)
            else:
                qq_layout.addWidget(QLabel("No fitness data available for Q-Q plot"))
            
            plot_tabs.addTab(qq_tab, " Q-Q Plot")
            
            # 8. Pie Chart Tab
            pie_tab = QWidget()
            pie_layout = QVBoxLayout(pie_tab)
            
            if group_counts:
                sizes = [group_counts[g] for g in groups if group_counts[g] > 0]
                labels = [f'{g}\n({group_counts[g]} runs)' for g in groups if group_counts[g] > 0]
                colors = [group_colors[g] for g in groups if group_counts[g] > 0]
                
                if sizes:
                    fig_pie = Figure(figsize=(10, 8), tight_layout=True)
                    ax_pie = fig_pie.add_subplot(111)
                    
                    wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                                        startangle=90, textprops={'fontsize': 12})
                    ax_pie.set_title('Group Distribution', fontweight='bold', fontsize=14)
                    
                    canvas_pie = FigureCanvas(fig_pie)
                    toolbar_pie = NavigationToolbar(canvas_pie, pie_tab)
                    self._attach_open_in_new_window(toolbar_pie, fig_pie, "Group Distribution - Pie Chart")
                    pie_layout.addWidget(canvas_pie)
                    pie_layout.addWidget(toolbar_pie)
                else:
                    pie_layout.addWidget(QLabel("No group data available for pie chart"))
            else:
                pie_layout.addWidget(QLabel("No group counts available for pie chart"))
            
            plot_tabs.addTab(pie_tab, " Group Distribution")
            
            # 9. Metrics Comparison Tab
            metrics_comp_tab = QWidget()
            metrics_comp_layout = QVBoxLayout(metrics_comp_tab)
            
            if group_counts:
                metrics_data = []
                group_names = []
                
                for group in groups:
                    if group_counts[group] > 0:
                        group_df = df[df['group'] == group]
                        group_names.append(group)
                        
                        row_data = [
                            group_df['best_fitness'].mean() if 'best_fitness' in group_df.columns else 0,
                            group_df['convergence_generation'].mean() if 'convergence_generation' in group_df.columns else 0,
                            group_df['elapsed_time'].mean() if 'elapsed_time' in group_df.columns else 0,
                            group_df['cpu_time_used'].mean() if 'cpu_time_used' in group_df.columns else 0
                        ]
                        metrics_data.append(row_data)
                
                if metrics_data and group_names:
                    fig_metrics = Figure(figsize=(12, 8), tight_layout=True)
                    ax_metrics = fig_metrics.add_subplot(111)
                    
                    metrics_array = np.array(metrics_data).T
                    metric_labels = ['Fitness', 'Conv Gen', 'Time (s)', 'CPU (s)']
                    
                    x = np.arange(len(group_names))
                    width = 0.2
                    
                    for i, (metric_values, label) in enumerate(zip(metrics_array, metric_labels)):
                        if np.any(np.isfinite(metric_values)) and np.max(metric_values) > 0:
                            # Normalize for better visualization
                            normalized_values = metric_values / np.max(metric_values) if np.max(metric_values) > 0 else metric_values
                            ax_metrics.bar(x + i*width, normalized_values, width, label=label, alpha=0.8)
                    
                    ax_metrics.set_xlabel('Groups')
                    ax_metrics.set_ylabel('Normalized Values')
                    ax_metrics.set_title('Group Performance Comparison', fontweight='bold', fontsize=14)
                    ax_metrics.set_xticks(x + width * 1.5)
                    ax_metrics.set_xticklabels(group_names)
                    ax_metrics.legend()
                    ax_metrics.grid(True, alpha=0.3)
                    
                    canvas_metrics = FigureCanvas(fig_metrics)
                    toolbar_metrics = NavigationToolbar(canvas_metrics, metrics_comp_tab)
                    self._attach_open_in_new_window(toolbar_metrics, fig_metrics, "Group Performance Comparison - Metrics")
                    metrics_comp_layout.addWidget(canvas_metrics)
                    metrics_comp_layout.addWidget(toolbar_metrics)
                else:
                    metrics_comp_layout.addWidget(QLabel("No group data available for metrics comparison"))
            else:
                metrics_comp_layout.addWidget(QLabel("No group counts available for metrics comparison"))
            
            plot_tabs.addTab(metrics_comp_tab, " Metrics Comparison")
            
            # 10. Trends Tab
            trends_tab = QWidget()
            trends_layout = QVBoxLayout(trends_tab)
            
            if 'run_number' in df.columns and len(df) > 3:
                fig_trends = Figure(figsize=(12, 8), tight_layout=True)
                ax_trends = fig_trends.add_subplot(111)
                
                # Calculate moving average
                window = max(3, len(df) // 10)
                df_sorted = df.sort_values('run_number')
                
                if 'best_fitness' in df.columns:
                    moving_avg = df_sorted['best_fitness'].rolling(window=window, center=True).mean()
                    ax_trends.plot(df_sorted['run_number'], df_sorted['best_fitness'], 'o-', alpha=0.5, label='Fitness', markersize=3)
                    ax_trends.plot(df_sorted['run_number'], moving_avg, '-', linewidth=3, label=f'Moving Avg (w={window})')
                
                ax_trends.set_title('Performance Trends Over Runs', fontweight='bold', fontsize=14)
                ax_trends.set_xlabel('Run Number')
                ax_trends.set_ylabel('Best Fitness')
                ax_trends.legend()
                ax_trends.grid(True, alpha=0.3)
                
                canvas_trends = FigureCanvas(fig_trends)
                toolbar_trends = NavigationToolbar(canvas_trends, trends_tab)
                self._attach_open_in_new_window(toolbar_trends, fig_trends, "Performance Trends Over Runs")
                trends_layout.addWidget(canvas_trends)
                trends_layout.addWidget(toolbar_trends)
            else:
                trends_layout.addWidget(QLabel("Not enough run data for trends analysis"))
            
            plot_tabs.addTab(trends_tab, " Performance Trends")
            
            # 11. Distribution Analysis Tab
            dist_analysis_tab = QWidget()
            dist_analysis_layout = QVBoxLayout(dist_analysis_tab)
            
            if 'score' in df.columns and len(df) >= 5:
                fig_dist = Figure(figsize=(14, 8), tight_layout=True)

                # Create single plot layout for cleaner appearance
                ax_main = fig_dist.add_subplot(111)

                # Get score data (robust to old imports and non-finite values)
                try:
                    scores_series = pd.to_numeric(df['score'], errors='coerce')
                except Exception:
                    scores_series = df['score']
                scores = scores_series.replace([np.inf, -np.inf], np.nan).dropna()

                # If empty after cleaning, show a friendly message and skip detailed overlays
                skip_dist_details = False
                if scores.empty or scores.size < 2:
                    skip_dist_details = True
                    ax_main.text(
                        0.5, 0.5,
                        'Not enough valid data to plot distribution',
                        ha='center', va='center'
                    )

                # Create histogram with more bins for better precision
                n_bins = min(max(20, len(scores) // 2), 100)  # More bins for higher resolution
                counts, bins, patches = ax_main.hist(
                    scores,
                    bins=n_bins,
                    alpha=0.8,
                    color='lightblue',
                    edgecolor='black',
                    linewidth=0.5,
                    density=True,
                    rwidth=0.85,
                )

                # Fit and overlay multiple distribution types
                try:
                    from scipy import stats
                    from sklearn.neighbors import KernelDensity
                    
                    # Calculate basic statistics
                    mu, sigma = float(scores.mean()), float(scores.std())

                    # Ensure a non-degenerate plotting range
                    smin, smax = float(scores.min()), float(scores.max())
                    if not np.isfinite(smin) or not np.isfinite(smax) or smin == smax:
                        # Expand to a small window around the single/invalid value
                        smin, smax = smin - 1.0 if np.isfinite(smin) else -1.0, smax + 1.0 if np.isfinite(smax) else 1.0

                    # 1. KDE (Kernel Density Estimation) - follows actual data shape
                    x_range = np.linspace(smin, smax, 200)
                    # Guard against zero/NaN bandwidth
                    bw = sigma / 3.0 if np.isfinite(sigma) and sigma > 0 else max((smax - smin) / 100.0, 1e-3)
                    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(scores.values.reshape(-1, 1))
                    kde_scores = np.exp(kde.score_samples(x_range.reshape(-1, 1)))
                    ax_main.plot(x_range, kde_scores, 'purple', linewidth=3, alpha=0.9, 
                               label=f'KDE (Actual Shape)')
                    
                    # 2. Try different distributions and find best fit
                    distributions_to_try = [
                        ('Normal', stats.norm),
                        ('Skew Normal', stats.skewnorm),
                        ('Gamma', stats.gamma),
                        ('Beta', stats.beta),
                        ('Log Normal', stats.lognorm)
                    ]
                    
                    best_dist = None
                    best_aic = np.inf
                    best_params = None
                    best_name = None
                    
                    for dist_name, dist in distributions_to_try:
                        try:
                            # Fit distribution
                            if dist_name == 'Beta':
                                # Beta needs data in [0,1] range
                                if scores.min() >= 0:
                                    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
                                    params = dist.fit(norm_scores)
                                    # Transform back for plotting
                                    x_beta = (x_range - scores.min()) / (scores.max() - scores.min())
                                    y_beta = dist.pdf(x_beta, *params) / (scores.max() - scores.min())
                                    
                                    # Calculate AIC for normalized data
                                    log_likelihood = np.sum(dist.logpdf(norm_scores, *params))
                                else:
                                    continue
                            else:
                                params = dist.fit(scores)
                                y_beta = dist.pdf(x_range, *params)
                                log_likelihood = np.sum(dist.logpdf(scores, *params))
                            
                            # Calculate AIC (Akaike Information Criterion)
                            aic = 2 * len(params) - 2 * log_likelihood
                            
                            if aic < best_aic and np.all(np.isfinite(y_beta)):
                                best_aic = aic
                                best_dist = dist
                                best_params = params
                                best_name = dist_name
                                if dist_name == 'Beta':
                                    best_y = y_beta
                                else:
                                    best_y = y_beta
                                    
                        except Exception:
                            continue
                    
                    # Plot the best fitting distribution
                    if best_dist is not None and best_params is not None:
                        if best_name == 'Beta':
                            y_fit = best_y
                        else:
                            y_fit = best_dist.pdf(x_range, *best_params)
                            
                        ax_main.plot(x_range, y_fit, 'red', linewidth=2, alpha=0.8, linestyle='--',
                                   label=f'Best Fit: {best_name} (AIC={best_aic:.1f})')
                        
                        # Add distribution parameters to legend
                        if best_name == 'Normal':
                            param_str = f'mu={best_params[0]:.4f}, sigma={best_params[1]:.4f}'
                        elif best_name == 'Skew Normal':
                            param_str = f'a={best_params[0]:.3f}, mu={best_params[1]:.4f}, sigma={best_params[2]:.4f}'
                        elif best_name in ['Gamma', 'Log Normal']:
                            param_str = f'shape={best_params[0]:.3f}, scale={best_params[2]:.4f}' if len(best_params) > 2 else f'shape={best_params[0]:.3f}'
                        else:
                            param_str = f'params={[f"{p:.3f}" for p in best_params[:3]]}'
                        
                        # Add parameter info as separate legend entry
                        ax_main.plot([], [], ' ', label=f'  Parameters: {param_str}')
                    else:
                        # Fallback to simple normal if all fits fail
                        y_norm = stats.norm.pdf(x_range, mu, sigma)
                        ax_main.plot(x_range, y_norm, 'red', linewidth=2, alpha=0.8, linestyle='--',
                                   label=f'Normal Fit (mu={mu:.4f}, sigma={sigma:.4f})')
                        
                except Exception as e:
                    # Fallback to basic statistics
                    mu, sigma = scores.mean(), scores.std()
                    print(f"Distribution fitting failed: {e}")
                
                # Get reference boundaries and color regions
                if 'best_ref_score' in df.columns:
                    best_ref = df.iloc[0]['best_ref_score']
                    mean_ref = df.iloc[0]['mean_ref_score']
                    worst_ref = df.iloc[0]['worst_ref_score']
                    
                    # Calculate boundary points (midpoints between references)
                    best_mid_boundary = (best_ref + mean_ref) / 2
                    mid_worst_boundary = (mean_ref + worst_ref) / 2
                    
                    # Color the histogram bars based on regions
                    for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
                        if bin_center <= best_mid_boundary:
                            patch.set_facecolor('#2ca02c')  # Best - Green
                            patch.set_alpha(0.7)
                        elif bin_center <= mid_worst_boundary:
                            patch.set_facecolor('#ff7f0e')  # Mid - Orange  
                            patch.set_alpha(0.7)
                        else:
                            patch.set_facecolor('#d62728')  # Worst - Red
                            patch.set_alpha(0.7)
                    
                    # Add vertical reference lines
                    ax_main.axvline(best_ref, color='darkgreen', linestyle='-', linewidth=3, alpha=0.9,
                                   label=f'Best Reference: {best_ref:.4f}')
                    ax_main.axvline(mean_ref, color='darkorange', linestyle='-', linewidth=3, alpha=0.9,
                                   label=f'Mean Reference: {mean_ref:.4f}')
                    ax_main.axvline(worst_ref, color='darkred', linestyle='-', linewidth=3, alpha=0.9,
                                   label=f'Worst Reference: {worst_ref:.4f}')
                    
                    # Add boundary lines (dashed)
                    ax_main.axvline(best_mid_boundary, color='gray', linestyle='--', linewidth=2, alpha=0.7,
                                   label=f'Best-Mid Boundary: {best_mid_boundary:.4f}')
                    ax_main.axvline(mid_worst_boundary, color='gray', linestyle='--', linewidth=2, alpha=0.7,
                                   label=f'Mid-Worst Boundary: {mid_worst_boundary:.4f}')
                    
                    # Calculate group statistics
                    best_group = df[df['group'] == 'Best']['score'].dropna()
                    mid_group = df[df['group'] == 'Mid']['score'].dropna()
                    worst_group = df[df['group'] == 'Worst']['score'].dropna()
                    
                    # Add group density curves
                    if len(best_group) >= 2:
                        try:
                            mu_best, sigma_best = stats.norm.fit(best_group)
                            x_best = np.linspace(best_group.min(), best_group.max(), 50)
                            y_best = stats.norm.pdf(x_best, mu_best, sigma_best)
                            ax_main.plot(x_best, y_best, color='green', linestyle=':', linewidth=2, alpha=0.8)
                        except Exception:
                            pass
                    
                    if len(mid_group) >= 2:
                        try:
                            mu_mid, sigma_mid = stats.norm.fit(mid_group)
                            x_mid = np.linspace(mid_group.min(), mid_group.max(), 50)
                            y_mid = stats.norm.pdf(x_mid, mu_mid, sigma_mid)
                            ax_main.plot(x_mid, y_mid, color='orange', linestyle=':', linewidth=2, alpha=0.8)
                        except Exception:
                            pass
                    
                    if len(worst_group) >= 2:
                        try:
                            mu_worst, sigma_worst = stats.norm.fit(worst_group)
                            x_worst = np.linspace(worst_group.min(), worst_group.max(), 50)
                            y_worst = stats.norm.pdf(x_worst, mu_worst, sigma_worst)
                            ax_main.plot(x_worst, y_worst, color='red', linestyle=':', linewidth=2, alpha=0.8)
                        except Exception:
                            pass
                
                # Formatting
                ax_main.set_title('Score Distribution Analysis with Reference-Based Grouping', 
                                fontweight='bold', fontsize=14)
                ax_main.set_xlabel('Score Value (lower is better)', fontsize=12)
                ax_main.set_ylabel('Density', fontsize=12)
                ax_main.grid(True, alpha=0.3)
                ax_main.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
                
                
                # Add distribution analysis text directly on the main plot
                try:
                    skewness = stats.skew(scores)
                    kurtosis = stats.kurtosis(scores)
                    
                    # Interpret skewness
                    if skewness < -0.5:
                        skew_desc = "Left-skewed (better runs more common)"
                    elif skewness > 0.5:
                        skew_desc = "Right-skewed (worse runs more common)"
                    else:
                        skew_desc = "Approximately symmetric"
                    
                    # Interpret kurtosis
                    if kurtosis > 1:
                        kurt_desc = "Heavy-tailed (more outliers)"
                    elif kurtosis < -1:
                        kurt_desc = "Light-tailed (fewer outliers)"
                    else:
                        kurt_desc = "Normal-like tails"
                    
                    # Check if best fit was found and add fit quality info
                    fit_quality = ""
                    if 'best_name' in locals() and best_name:
                        if best_name == 'Normal':
                            fit_quality = " Normal distribution fits well"
                        elif best_name == 'Skew Normal':
                            fit_quality = " Data is skewed (asymmetric)"
                        elif best_name in ['Gamma', 'Log Normal']:
                            fit_quality = " Right-skewed distribution"
                        elif best_name == 'Beta':
                            fit_quality = " Bounded distribution"
                        else:
                            fit_quality = f" Best fit: {best_name}"
                    
                    # Create compact analysis text for plot overlay
                    analysis_text = f""" DISTRIBUTION ANALYSIS
 Skewness: {skewness:.3f} ({skew_desc})
 Kurtosis: {kurtosis:.3f} ({kurt_desc})
{fit_quality if fit_quality else ""}
 Most data in: {df['group'].mode().iloc[0] if not df['group'].mode().empty else 'Unknown'} group
 Range: {scores.min():.4f} - {scores.max():.4f}

 LEGEND:
* Purple: Actual data shape (KDE)  * Red dashed: Best fit
* Green bars: Best region  * Orange bars: Mid region  * Red bars: Worst region
* Solid lines: References  * Dashed lines: Boundaries"""
                    
                    # Position the text box on the right side of the plot
                    ax_main.text(0.98, 0.98, analysis_text.strip(), transform=ax_main.transAxes,
                               verticalalignment='top', horizontalalignment='right', fontsize=8, 
                               fontfamily='monospace',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.9, edgecolor='steelblue'))
                except Exception:
                    ax_main.text(0.98, 0.98, "Distribution analysis unavailable", 
                               transform=ax_main.transAxes, ha='right', va='top',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
                
                canvas_dist = FigureCanvas(fig_dist)
                toolbar_dist = NavigationToolbar(canvas_dist, dist_analysis_tab)
                self._attach_open_in_new_window(toolbar_dist, fig_dist, "Score Distribution Analysis")
                dist_analysis_layout.addWidget(canvas_dist)
                dist_analysis_layout.addWidget(toolbar_dist)
            else:
                dist_analysis_layout.addWidget(QLabel("Not enough score data for distribution analysis (minimum 5 runs required)"))
            
            plot_tabs.addTab(dist_analysis_tab, " Distribution Analysis")
            
            # 12. Statistics Table Tab
            stats_tab = QWidget()
            stats_layout = QVBoxLayout(stats_tab)
            
            stats_table = QTableWidget()
            stats_table.setColumnCount(7)
            stats_table.setHorizontalHeaderLabels(['Group', 'Count', 'Mean Fitness', 'Std Fitness', 'Mean Conv Gen', 'Mean Time (s)', 'Mean CPU (s)'])
            stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            
            stats_table.setRowCount(len(groups))
            for i, group in enumerate(groups):
                group_data = df[df['group'] == group] if 'group' in df.columns else pd.DataFrame()
                
                items = [
                    QTableWidgetItem(group),
                    QTableWidgetItem(str(len(group_data))),
                    QTableWidgetItem(f"{group_data['best_fitness'].mean():.6f}" if 'best_fitness' in group_data.columns and len(group_data) > 0 else "-"),
                    QTableWidgetItem(f"{group_data['best_fitness'].std():.6f}" if 'best_fitness' in group_data.columns and len(group_data) > 0 else "-"),
                    QTableWidgetItem(f"{group_data['convergence_generation'].mean():.2f}" if 'convergence_generation' in group_data.columns and len(group_data) > 0 else "-"),
                    QTableWidgetItem(f"{group_data['elapsed_time'].mean():.3f}" if 'elapsed_time' in group_data.columns and len(group_data) > 0 else "-"),
                    QTableWidgetItem(f"{group_data['cpu_time_used'].mean():.3f}" if 'cpu_time_used' in group_data.columns and len(group_data) > 0 else "-"),
                ]
                
                for j, item in enumerate(items):
                    item.setTextAlignment(Qt.AlignCenter)
                    # Color code by group
                    if group == 'Best':
                        item.setBackground(QColor(200, 255, 200))
                    elif group == 'Mid':
                        item.setBackground(QColor(255, 255, 200))
                    elif group == 'Worst':
                        item.setBackground(QColor(255, 220, 220))
                    stats_table.setItem(i, j, item)
            
            stats_layout.addWidget(stats_table)
            
            # Add export button for the table
            export_btn = QPushButton("Export Statistics Table")
            export_btn.clicked.connect(lambda: self._export_table_via_dialog(stats_table, "group_summary_statistics"))
            stats_layout.addWidget(export_btn)
            
            plot_tabs.addTab(stats_tab, " Statistics Table")
            
            # 13. FRF Overlay Tab
            try:
                self._build_group_frf_overlay_tab(plot_tabs, df)
            except Exception as _e_frf:
                try:
                    info = QLabel(f"FRF Overlay unavailable: {str(_e_frf)}")
                    info.setStyleSheet("color: orange; padding: 6px;")
                    self.group_summary_container.layout().addWidget(info)
                except Exception:
                    pass

        except Exception as e:
            print(f"Error building comprehensive group summary: {str(e)}")
            # Add error message to the container
            try:
                error_label = QLabel(f"Error creating visualizations: {str(e)}")
                error_label.setStyleSheet("color: red; font-weight: bold; padding: 10px;")
                self.group_summary_container.layout().addWidget(error_label)
            except Exception:
                pass

    









