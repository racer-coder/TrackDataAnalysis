
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""XY Scatter plot: plot two channels against each other."""

import bisect

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, Qt

from . import channels
from . import state
from . import widgets


class Scatter(widgets.MouseHelperWidget):
    """Component showing XY scatter plot of two channels."""

    def __init__(self, data_view, state=None):
        super().__init__()
        self.data_view = data_view
        self.channel_list = list(state['channels']) if state and 'channels' in state else []
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

        chfont = QtGui.QFont('Tahoma')
        chfont.setPixelSize(widgets.deviceScale(self, 13))
        ph.painter.setFont(chfont)

        axfont = QtGui.QFont('Tahoma')
        axfont.setPixelSize(widgets.deviceScale(self, 11.25))

        dv = self.data_view
        if not dv.ref_lap:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, "No data loaded")
            return

        if len(self.channel_list) == 0:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            msg = "Add 2 channels: first = X axis, second = Y axis"
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, msg)
            return

        if len(self.channel_list) == 1:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            msg = "Add 1 more channel: X axis = '%s', next channel = Y axis" % self.channel_list[0]
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
        axis_pen = QtGui.QPen(QtGui.QColor(192, 192, 192))
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

        data = []
        # Traverse laps in reverse order so main lap is on top, also
        # cd_x/cd_y will represent the reference lap.
        for lap, color, idx in self.data_view.get_laps()[::-1]:
            try:
                cd_x = self.data_view.get_channel_data(lap, ch_x)
                cd_y = self.data_view.get_channel_data(lap, ch_y)
            except Exception:
                continue
            if cd_x is None or cd_y is None:
                continue

            # Prune data to just the lap in question
            start_idx = bisect.bisect_left(cd_x.timecodes, lap.start.time)
            end_idx = bisect.bisect_right(cd_x.timecodes, lap.end.time)

            if start_idx == end_idx:
                continue

            # Align on timecodes: interpolate Y onto X timecodes
            vals_x = cd_x.values[start_idx:end_idx]
            vals_y = np.interp(cd_x.timecodes[start_idx:end_idx], cd_y.timecodes, cd_y.values)

            data.append((color if idx != 1 else channels.colors[cd_y.color],
                         vals_x, vals_y))

        if not data:
            return

        x_min = min(np.min(vals_x) for _, vals_x, _ in data)
        x_max = max(np.max(vals_x) for _, vals_x, _ in data)
        y_min = min(np.min(vals_y) for _, _, vals_y in data)
        y_max = max(np.max(vals_y) for _, _, vals_y in data)
        x_range = (x_max - x_min) or 1.
        y_range = (y_max - y_min) or 1.

        dot_size = max(1.5, 2 * scale)
        for color, vals_x, vals_y in data:
            pen = QtGui.QPen(color)
            pen.setWidth(1)
            ph.painter.setPen(pen)
            ph.painter.setBrush(QtGui.QBrush(
                QtGui.QColor(color.red(), color.green(), color.blue(), 128)))

            vals_x = margin_left + (vals_x - x_min) * (chart_w / x_range)
            vals_y = margin_top + chart_h - (vals_y - y_min) * (chart_h / y_range)

            for i in range(0, len(vals_x), max(1, len(vals_x) // 10000)):
                ph.painter.drawEllipse(QPointF(vals_x[i], vals_y[i]), dot_size, dot_size)

        # Axis labels
        ph.painter.setPen(QtGui.QColor(192, 192, 192))
        x_units = cd_x.units if cd_x.units else ''
        y_units = cd_y.units if cd_y.units else ''

        ph.painter.setFont(axfont)

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
            '%.*f' % (cd_x.dec_pts, x_min))
        ph.painter.drawText(
            int(margin_left + chart_w * 2 / 3), int(margin_top + chart_h + 2 * scale),
            int(chart_w / 3), int(15 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            '%.*f' % (cd_x.dec_pts, x_max))

        # Y axis range
        ph.painter.drawText(
            0, int(margin_top + chart_h - 15 * scale),
            int(margin_left - 5 * scale), int(15 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            '%.*f' % (cd_y.dec_pts, y_min))
        ph.painter.drawText(
            0, int(margin_top), int(margin_left - 5 * scale), int(15 * scale),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            '%.*f' % (cd_y.dec_pts, y_max))

        # Y axis label (draw vertically would be ideal, but keep simple)
        y_label = f"{ch_y} ({y_units})" if y_units else ch_y
        ph.painter.rotate(-90)
        ph.painter.drawText(
            #0, int(margin_top + 15 * scale), int(margin_left - 5 * scale), int(20 * scale),
            -int(margin_top + chart_h), 0, int(chart_h), int(20 * scale),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            y_label)
