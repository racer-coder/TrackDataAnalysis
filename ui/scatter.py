
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""XY Scatter plot: plot two channels against each other."""

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, Qt

from . import state, widgets


class Scatter(widgets.MouseHelperWidget):
    """Component showing XY scatter plot of two channels."""

    def __init__(self, data_view, st=None):
        super().__init__()
        self.data_view = data_view
        self.channel_list = list(st['channels']) if st and 'channels' in st else []
        self.setMinimumSize(200, 100)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        data_view.values_change.connect(self.update)

    def save_state(self):
        return {
            'type': 'scatter',
            'base': self.parentWidget().save_state(),
            'channels': self.channel_list,
        }

    def channels(self):
        return set(self.channel_list)

    def addChannel(self, ch):
        if ch in self.channel_list:
            self.channel_list.remove(ch)
        else:
            self.channel_list.append(ch)
        self.data_view.data_change.emit()
        self.update()

    def updateCursor(self, old_cursor):
        pass

    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)
        ph.painter.fillRect(ph.rect, QtGui.QColor(12, 12, 12))
        ph.painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        dv = self.data_view
        if not dv.ref_lap:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, "No data loaded")
            return

        if len(self.channel_list) < 2:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            msg = "Add 2 channels: first = X axis, second = Y axis"
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, msg)
            return

        ch_x = self.channel_list[0]
        ch_y = self.channel_list[1]

        scale = ph.scale
        margin_left = int(60 * scale)
        margin_right = int(15 * scale)
        margin_top = int(20 * scale)
        margin_bottom = int(35 * scale)

        chart_w = ph.size.width() - margin_left - margin_right
        chart_h = ph.size.height() - margin_top - margin_bottom
        if chart_w < 40 or chart_h < 20:
            return

        # Draw axes
        axis_pen = QtGui.QPen(QtGui.QColor(80, 80, 80))
        axis_pen.setWidth(max(1, int(scale)))
        ph.painter.setPen(axis_pen)
        ph.painter.drawLine(
            int(margin_left), int(margin_top + chart_h),
            int(margin_left + chart_w), int(margin_top + chart_h))
        ph.painter.drawLine(
            int(margin_left), int(margin_top),
            int(margin_left), int(margin_top + chart_h))

        label_font = ph.painter.font()
        label_font.setPixelSize(max(8, int(10 * scale)))
        ph.painter.setFont(label_font)

        def plot_lap(lap, color, dot_size):
            try:
                cd_x = lap.get_channel_data(ch_x, None, dv.maths)
                cd_y = lap.get_channel_data(ch_y, None, dv.maths)
            except Exception:
                return None, None
            if (cd_x is None or cd_y is None or
                    not hasattr(cd_x, 'values') or not hasattr(cd_y, 'values') or
                    len(cd_x.values) == 0 or len(cd_y.values) == 0):
                return None, None

            # Align on timecodes: interpolate Y onto X timecodes
            vals_x = cd_x.values
            vals_y = np.interp(cd_x.timecodes, cd_y.timecodes, cd_y.values)

            x_min, x_max = np.min(vals_x), np.max(vals_x)
            y_min, y_max = np.min(vals_y), np.max(vals_y)
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0

            pen = QtGui.QPen(color)
            pen.setWidth(1)
            ph.painter.setPen(pen)
            ph.painter.setBrush(QtGui.QBrush(
                QtGui.QColor(color.red(), color.green(), color.blue(), 120)))

            for i in range(0, len(vals_x), max(1, len(vals_x) // 2000)):
                sx = margin_left + ((vals_x[i] - x_min) / x_range) * chart_w
                sy = margin_top + chart_h - ((vals_y[i] - y_min) / y_range) * chart_h
                ph.painter.drawEllipse(QPointF(sx, sy), dot_size, dot_size)

            return (cd_x, x_min, x_max), (cd_y, y_min, y_max)

        # Plot alt lap first (behind)
        dot_sz = max(1.5, 2 * scale)
        if dv.alt_lap:
            plot_lap(dv.alt_lap, state.lap_colors[1], dot_sz)

        # Plot ref lap
        x_info, y_info = plot_lap(dv.ref_lap, state.lap_colors[0], dot_sz)

        if x_info is None:
            return

        cd_x, x_min, x_max = x_info
        cd_y, y_min, y_max = y_info

        # Axis labels
        ph.painter.setPen(QtGui.QColor(160, 160, 160))
        x_units = cd_x.units if cd_x.units else ''
        y_units = cd_y.units if cd_y.units else ''

        # X axis label
        x_label = f"{ch_x} ({x_units})" if x_units else ch_x
        ph.painter.drawText(
            int(margin_left), int(margin_top + chart_h + 5 * scale),
            int(chart_w), int(margin_bottom),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            x_label)

        # X axis range
        ph.painter.drawText(
            int(margin_left), int(margin_top + chart_h + 2 * scale),
            int(chart_w / 3), int(15 * scale),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{x_min:.1f}")
        ph.painter.drawText(
            int(margin_left + chart_w * 2 / 3), int(margin_top + chart_h + 2 * scale),
            int(chart_w / 3), int(15 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"{x_max:.1f}")

        # Y axis label (draw vertically would be ideal, but keep simple)
        y_label = f"{ch_y} ({y_units})" if y_units else ch_y
        ph.painter.drawText(
            0, int(margin_top), int(margin_left - 5 * scale), int(20 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            y_label)

        # Y axis range
        ph.painter.drawText(
            0, int(margin_top + chart_h - 15 * scale),
            int(margin_left - 5 * scale), int(15 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            f"{y_min:.1f}")
        ph.painter.drawText(
            0, int(margin_top), int(margin_left - 5 * scale), int(15 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"{y_max:.1f}")
