
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""XY Scatter plot: plot two channels against each other."""

import bisect

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, Qt

from . import channels
from . import graphhelper
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

        gh = graphhelper.GraphHelper(self, ph)
        gh.setArea(QRectF(QPointF(0, 0), ph.size).adjusted(0, 25 * ph.scale, 0, 0), 1, 2)

        ph.painter.setFont(gh.channel_font)

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
            step = max(1, (end_idx - start_idx) // 10000)
            vals_x = cd_x.values[start_idx:end_idx:step]
            vals_y = np.interp(cd_x.timecodes[start_idx:end_idx:step], cd_y.timecodes, cd_y.values)

            data.append((color if idx != 1 else channels.colors[cd_y.color],
                         vals_x, vals_y))

        if not data:
            return

        gh.setXAxis(min(np.min(vals_x) for _, vals_x, _ in data),
                    max(np.max(vals_x) for _, vals_x, _ in data))
        gh.setYAxis(min(np.min(vals_y) for _, _, vals_y in data),
                    max(np.max(vals_y) for _, _, vals_y in data))

        gh.paintXGrid()
        gh.paintYGrid()

        ph.painter.save()
        ph.painter.setClipRect(gh.graph_area)
        dot_size = max(1.5, 2 * ph.scale)
        for color, vals_x, vals_y in data:
            pen = QtGui.QPen(color)
            pen.setWidth(1)
            ph.painter.setPen(pen)
            ph.painter.setBrush(QtGui.QBrush(
                QtGui.QColor(color.red(), color.green(), color.blue(), 128)))

            vals_x = gh.x_axis.calc(vals_x).data
            vals_y = gh.y_axis.calc(vals_y).data

            for x, y in zip(vals_x, vals_y):
                ph.painter.drawEllipse(QPointF(x, y), dot_size, dot_size)
        ph.painter.restore()

        # Graph frame and axis
        gh.paintGraphFrame()
        gh.paintXAxis()
        gh.paintYAxis()

        # Axis labels
        ph.painter.setPen(gh.axis_pen)
        ph.painter.setFont(gh.axis_font)
        x_units = cd_x.units if cd_x.units else ''
        y_units = cd_y.units if cd_y.units else ''

        # X axis label
        x_label = f"{ch_x} ({x_units})" if x_units else ch_x
        axis_space = 16 * ph.scale
        ph.painter.drawText(
            int(gh.graph_area.left()), int(gh.graph_area.bottom() + axis_space),
            int(gh.graph_area.width()), int(axis_space),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            x_label)

        # Y axis label (draw vertically would be ideal, but keep simple)
        y_label = f"{ch_y} ({y_units})" if y_units else ch_y
        ph.painter.save()
        ph.painter.rotate(-90)
        ph.painter.drawText(
            -int(gh.graph_area.bottom()), 0, int(gh.graph_area.height()), int(20 * ph.scale),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            y_label)
        ph.painter.restore()
