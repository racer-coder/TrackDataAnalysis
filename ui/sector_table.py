# Copyright 2026. MIT License (see LICENSE).

"""Sector comparison table: shows time per sector per lap with delta highlighting."""

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import state


class SectorTable(QWidget):
    """Component showing a table of sector times per lap with delta coloring."""

    def __init__(self, data_view, state=None):
        super().__init__()
        self.data_view = data_view

        self.label = QLabel()
        self.label.setStyleSheet("color: #888; padding: 4px;")

        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0c0c0c;
                color: #ddd;
                gridline-color: #333;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #ccc;
                padding: 4px;
                border: 1px solid #333;
                font-weight: bold;
            }
        """)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.cellClicked.connect(self._on_cell_clicked)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        data_view.values_change.connect(self.recompute)
        data_view.data_change.connect(self.recompute)
        self.recompute()

    def save_state(self):
        return {'type': 'sector_table', 'base': self.parentWidget().save_state()}

    def channels(self):
        return set()

    def addChannel(self, ch):
        pass  # Not channel-based

    def updateCursor(self, old_cursor):
        pass

    def recompute(self):
        """Rebuild the sector table from current track/lap data."""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        if not self.data_view.log_files:
            self.label.setText("No data loaded.")
            return

        track = self.data_view.track
        if not track or not track.sector_sets:
            self.label.setText("No sectors defined. Use Tools > Track editor to define sectors.")
            return

        # Get the first sector set
        sector_set_name = list(track.sector_sets.keys())[0]
        sectors = track.sector_sets[sector_set_name]
        markers = sorted(sectors.markers, key=lambda m: m._dist if m._dist is not None else 0)

        if len(markers) < 2:
            self.label.setText("Not enough sector markers.")
            return

        # Build sector names from consecutive marker pairs
        sector_names = [m.name for m in markers]

        # Collect all laps from the first log file
        logref = self.data_view.log_files[0]
        laps = logref.laps

        if not laps:
            self.label.setText("No laps available.")
            return

        # Compute sector times for each lap
        # Each marker._dist is distance from start/finish in meters
        sector_dists = [m._dist for m in markers if m._dist is not None]
        if not sector_dists:
            self.label.setText("Sector distances not computed.")
            return

        # Total track distance (last coord's distance value)
        track_dist = track.coords[-1][3] if track.coords else sector_dists[-1]

        # Build table: rows = laps, columns = [Lap, Total] + sectors
        num_sectors = len(sector_names)
        col_headers = ["Total"] + sector_names
        row_headers = []
        self.table.setColumnCount(len(col_headers))
        self.table.setHorizontalHeaderLabels(col_headers)
        self.table.setRowCount(len(laps))

        # sector_times[lap_idx][sector_idx] = time in seconds
        sector_times = []
        lap_totals = []

        for lap_idx, lap in enumerate(laps):
            lap_dist = lap.end.dist - lap.start.dist
            lap_duration = lap.duration() / 1000.0  # ms to seconds

            # If lap distance is too short (incomplete), skip
            if lap_dist < track_dist * 0.5:
                sector_times.append([None] * num_sectors)
                lap_totals.append(lap_duration)
                continue

            times = []
            prev_time = 0.0
            for i, dist in enumerate(sector_dists):
                # Convert sector distance to time within this lap
                try:
                    t = lap.lapDist2Time(dist) / 1000.0  # ms to seconds
                except:
                    t = None
                if t is not None and t > prev_time:
                    times.append(t - prev_time)
                    prev_time = t
                else:
                    times.append(None)
            sector_times.append(times)
            lap_totals.append(lap_duration)

        # Find best sector times (for delta coloring)
        best_sectors = [None] * num_sectors
        for si in range(num_sectors):
            valid = [st[si] for st in sector_times if st[si] is not None and st[si] > 0]
            if valid:
                best_sectors[si] = min(valid)

        best_total = min([t for t in lap_totals if t > 0], default=None)

        # Populate table
        ref_lap = self.data_view.ref_lap

        for lap_idx, lap in enumerate(laps):
            duration = lap_totals[lap_idx]

            # Lap number
            row_headers.append('Lap %d' % lap_idx)

            # Total time
            total_item = QTableWidgetItem(_format_time(duration))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if best_total and duration > 0:
                delta = duration - best_total
                _color_item(total_item, delta)
            self.table.setItem(lap_idx, 0, total_item)

            # Sector times
            for si in range(num_sectors):
                st_val = sector_times[lap_idx][si]
                if st_val is not None and st_val > 0:
                    item = QTableWidgetItem(_format_time(st_val))
                    if best_sectors[si]:
                        delta = st_val - best_sectors[si]
                        _color_item(item, delta)
                else:
                    item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(lap_idx, 1 + si, item)

            # Highlight reference lap row
            if ref_lap and lap.num == ref_lap.num and lap.log == ref_lap.log:
                for col in range(self.table.columnCount()):
                    cell = self.table.item(lap_idx, col)
                    if cell:
                        font = cell.font()
                        font.setBold(True)
                        cell.setFont(font)

        self.table.setVerticalHeaderLabels(row_headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.label.setText(f"Sectors: {sector_set_name} ({num_sectors} sectors, {len(laps)} laps)")

    def _on_cell_clicked(self, row, col):
        """Click a row to set it as reference lap."""
        logref = self.data_view.log_files[0] if self.data_view.log_files else None
        if logref and row < len(logref.laps):
            self.data_view.ref_lap = logref.laps[row]
            self.data_view.values_change.emit()


def _format_time(seconds):
    """Format seconds as M:SS.mmm"""
    if seconds is None or seconds <= 0:
        return "—"
    mins = int(seconds) // 60
    secs = seconds - mins * 60
    if mins > 0:
        return f"{mins}:{secs:06.3f}"
    return f"{secs:.3f}"


def _color_item(item, delta_seconds):
    """Color a table cell green (faster) or red (slower) based on delta."""
    if abs(delta_seconds) < 0.001:
        # Best time — bright green
        item.setForeground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
    elif delta_seconds < 0.1:
        # Very close — light green
        item.setForeground(QtGui.QBrush(QtGui.QColor(100, 220, 100)))
    elif delta_seconds < 0.3:
        # Moderate — yellow
        item.setForeground(QtGui.QBrush(QtGui.QColor(220, 220, 80)))
    elif delta_seconds < 0.5:
        # Getting slower — orange
        item.setForeground(QtGui.QBrush(QtGui.QColor(220, 150, 50)))
    else:
        # Significantly slower — red
        item.setForeground(QtGui.QBrush(QtGui.QColor(220, 80, 80)))
