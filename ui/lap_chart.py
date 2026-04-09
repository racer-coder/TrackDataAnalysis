# Copyright 2026. MIT License (see LICENSE).

"""Lap time bar chart: visual comparison of lap times with color coding."""

from PySide6 import QtGui
from PySide6.QtCore import QRect, QRectF, Qt

from . import state, widgets


class LapChart(widgets.MouseHelperWidget):
    """Component showing horizontal bar chart of lap times."""

    def __init__(self, data_view, st=None):
        super().__init__()
        self.data_view = data_view
        self.setMinimumSize(200, 100)
        data_view.values_change.connect(self.update)
        data_view.data_change.connect(self.update)

    def save_state(self):
        return {'type': 'lap_chart', 'base': {}}

    def channels(self):
        return set()

    def addChannel(self, ch):
        pass

    def updateCursor(self, old_cursor):
        pass

    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)
        ph.painter.fillRect(ph.rect, QtGui.QColor(12, 12, 12))

        if not self.data_view.log_files:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, "No data loaded")
            return

        logref = self.data_view.log_files[0]
        laps = logref.laps
        if not laps:
            return

        ref_lap = self.data_view.ref_lap
        alt_lap = self.data_view.alt_lap

        # Get durations (skip laps with zero/negative time)
        lap_data = []
        for lap in laps:
            dur = lap.duration() / 1000.0
            if dur > 0:
                lap_data.append((lap, dur))

        if not lap_data:
            return

        durations = [d[1] for d in lap_data]
        best_time = min(durations)
        worst_time = max(durations)
        time_range = worst_time - best_time if worst_time > best_time else 1.0

        # Layout
        scale = ph.scale
        margin_left = int(55 * scale)
        margin_right = int(80 * scale)
        margin_top = int(30 * scale)
        margin_bottom = int(10 * scale)

        chart_w = ph.size.width() - margin_left - margin_right
        chart_h = ph.size.height() - margin_top - margin_bottom

        if chart_w < 50 or chart_h < 20:
            return

        n_laps = len(lap_data)
        bar_spacing = max(1, int(2 * scale))
        bar_h = max(4, (chart_h - bar_spacing * (n_laps - 1)) // n_laps)

        # Title
        title_font = ph.painter.font()
        title_font.setPixelSize(int(14 * scale))
        title_font.setBold(True)
        ph.painter.setFont(title_font)
        ph.painter.setPen(QtGui.QColor(200, 200, 200))
        ph.painter.drawText(margin_left, 0, chart_w, margin_top,
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                            "Lap Times")

        # Bar font
        bar_font = ph.painter.font()
        bar_font.setPixelSize(max(8, int(min(11, bar_h * 0.7) * scale)))
        bar_font.setBold(False)
        ph.painter.setFont(bar_font)

        # Reset mouse helpers
        self.mouse_helpers = []

        # Draw bars
        for i, (lap, dur) in enumerate(lap_data):
            y = margin_top + i * (bar_h + bar_spacing)

            # Bar width proportional to time (inverted: shorter = longer bar to show "better")
            # Actually, direct proportional is more intuitive
            bar_min_time = best_time * 0.98  # some padding
            bar_max_time = worst_time * 1.02
            frac = (dur - bar_min_time) / (bar_max_time - bar_min_time) if bar_max_time > bar_min_time else 0.5
            bar_w = max(4, int(frac * chart_w))

            # Color: green (best) → yellow → red (worst)
            t = (dur - best_time) / time_range if time_range > 0 else 0
            t = max(0, min(1, t))
            color = _time_gradient(t)

            # Draw bar
            bar_rect = QRect(margin_left, y, bar_w, bar_h)
            ph.painter.fillRect(bar_rect, color)

            # Highlight ref/alt lap
            if ref_lap and lap.num == ref_lap.num and lap.log == ref_lap.log:
                pen = QtGui.QPen(QtGui.QColor(255, 255, 255), 2 * scale)
                ph.painter.setPen(pen)
                ph.painter.drawRect(bar_rect)
            elif alt_lap and lap.num == alt_lap.num and lap.log == alt_lap.log:
                pen = QtGui.QPen(QtGui.QColor(180, 180, 180), 1 * scale)
                ph.painter.setPen(pen)
                ph.painter.drawRect(bar_rect)

            # Lap number label (left of bar)
            ph.painter.setPen(QtGui.QColor(180, 180, 180))
            ph.painter.drawText(0, y, margin_left - int(5 * scale), bar_h,
                                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                f"Lap {lap.num}")

            # Time label (right of bar)
            delta_str = ""
            if dur > best_time + 0.001:
                delta_str = f"  +{dur - best_time:.3f}"
            ph.painter.setPen(QtGui.QColor(200, 200, 200))
            ph.painter.drawText(margin_left + bar_w + int(5 * scale), y,
                                margin_right, bar_h,
                                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                f"{_format_time(dur)}{delta_str}")

            # Click region to select lap
            click_rect = QRectF(0, y, ph.size.width(), bar_h)
            self.mouse_helpers.append(
                widgets.MouseHelperItem(
                    click_rect,
                    Qt.CursorShape.PointingHandCursor,
                    [widgets.MouseHelperClick(
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier,
                        False,
                        lambda pos, lap=lap: self._select_ref(lap),
                        None)],
                    {}))

    def _select_ref(self, lap):
        self.data_view.ref_lap = lap
        self.data_view.values_change.emit()


def _time_gradient(t):
    """Return color from green (t=0) through yellow to red (t=1)."""
    if t < 0.5:
        r = int(2 * t * 220)
        g = 200
    else:
        r = 220
        g = int(200 * (1 - (t - 0.5) * 2))
    return QtGui.QColor(r, g, 40)


def _format_time(seconds):
    mins = int(seconds) // 60
    secs = seconds - mins * 60
    if mins > 0:
        return f"{mins}:{secs:06.3f}"
    return f"{secs:.3f}"
