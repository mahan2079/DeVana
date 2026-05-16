"""
Lightweight fallback for computational metrics visualizations.

This module provides no-op or minimal UI stubs for functions imported by
the GA/PSO mixins. It avoids hard runtime dependencies when the full
visualization package is not present.
"""

from typing import Any, Dict


def visualize_all_metrics(widgets_dict: Dict[str, Any], df: Any) -> None:
    """Stub: safely do nothing when asked to visualize metrics.

    Args:
        widgets_dict: Mapping of widget names to Qt widgets.
        df: Any tabular-like object (e.g., pandas.DataFrame).
    """
    try:
        # Intentionally minimal; real implementation can populate widgets_dict
        # with plots based on df. This stub keeps the app functional.
        return
    except Exception:
        return


def create_ga_visualizations(parent_widget: Any, run_data: Dict[str, Any]) -> None:
    """Stub: add a minimal placeholder label into the parent widget layout."""
    try:
        # Import Qt lazily to avoid issues in headless contexts
        from PyQt5.QtWidgets import QVBoxLayout, QLabel

        if parent_widget is None:
            return
        if not parent_widget.layout():
            parent_widget.setLayout(QVBoxLayout())
        parent_widget.layout().addWidget(
            QLabel("Computational metrics visualization is not available in this build.")
        )
    except Exception:
        return


def ensure_all_visualizations_visible(widget: Any) -> None:
    """Stub: ensure child widgets are visible if a layout exists."""
    try:
        if widget is None or not widget.layout():
            return
        layout = widget.layout()
        for i in range(layout.count()):
            child = layout.itemAt(i).widget()
            if child is not None:
                child.setVisible(True)
    except Exception:
        return