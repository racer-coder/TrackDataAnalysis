
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""XY Scatter plot: plot two channels against each other."""

import bisect

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt

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
        self.update() # can we be more efficient?

    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)
        ph.painter.fillRect(ph.rect, QtGui.QColor(12, 12, 12))

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

        data = []
        # Traverse laps in reverse order so main lap is on top, also
        # cds will represent the reference lap.
        for lap, color, idx in self.data_view.get_laps()[::-1]:
            cds = [self.data_view.get_channel_data(lap, ch) for ch in self.channel_list]
            if cds[0] is None:
                continue # Need X axis

            # Prune data to just the lap in question
            start_idx = bisect.bisect_left(cds[0].timecodes, lap.start.time)
            end_idx = bisect.bisect_right(cds[0].timecodes, lap.end.time)

            if start_idx == end_idx:
                continue

            # Align on timecodes: interpolate Y onto X timecodes
            step = max(1, (end_idx - start_idx) // 10000)
            vals_x = cds[0].values[start_idx:end_idx:step]
            vals_y = np.interp(cds[0].timecodes[start_idx:end_idx:step],
                               cds[1].timecodes, cds[1].values)

            data.append(([color if idx != 1 else channels.colors[cd.color]
                          for cd in cds[1:]],
                         vals_x, [np.interp(cds[0].timecodes[start_idx:end_idx:step],
                                            cd.timecodes, cd.values)
                                  for cd in cds[1:]],
                         [cd.interp(self.data_view.cursor2outTime(lap))
                          for cd in cds]))

        if not data:
            return

        gh.setXAxis(min(np.min(vals_x) for _, vals_x, _, _ in data),
                    max(np.max(vals_x) for _, vals_x, _, _ in data))
        gh.setYAxis(min(np.min(vals_y) for _, _, vals, _ in data for vals_y in vals),
                    max(np.max(vals_y) for _, _, vals, _ in data for vals_y in vals))

        gh.paintXGrid()
        gh.paintYGrid()

        ph.painter.save()
        ph.painter.setClipRect(gh.graph_area)
        ph.painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        dot_size = max(1.5, 2 * ph.scale)
        for colors, vals_x, vals, _ in data:
            vals_x = gh.x_axis.calc(vals_x).data
            for color, vals_y in zip(colors[::-1], vals[::-1]):
                pen = QtGui.QPen(color)
                pen.setWidth(1)
                ph.painter.setPen(pen)
                ph.painter.setBrush(QtGui.QBrush(
                    QtGui.QColor(color.red(), color.green(), color.blue(), 128)))

                vals_y = gh.y_axis.calc(vals_y).data
                for x, y in zip(vals_x, vals_y):
                    ph.painter.drawEllipse(QPointF(x, y), dot_size, dot_size)
        ph.painter.restore()

        # paint cursors
        ph.painter.save()
        ph.painter.setClipRect(gh.graph_area)
        cursor_width = 6 * ph.scale
        for colors, _, _, vals in data:
            val_x = gh.x_axis.calc(vals[0])
            for color, val_y in zip(colors[::-1], vals[1:][::-1]):
                val_y = gh.y_axis.calc(val_y)

                pens = [QtGui.QPen(Qt.black), QtGui.QPen(color)]
                pens[0].setWidth(5)
                pens[1].setWidth(3)
                for pen in pens:
                    pen.setCapStyle(Qt.RoundCap)
                    ph.painter.setPen(pen)
                    ph.painter.drawLine(val_x - cursor_width, val_y, val_x + cursor_width, val_y)
                    ph.painter.drawLine(val_x, val_y - cursor_width, val_x, val_y + cursor_width)
        ph.painter.restore()

        # Axis labels
        ph.painter.setPen(gh.axis_pen)
        ph.painter.setFont(gh.axis_font)
        x_units = cds[0].units if cds[0].units else ''

        # X axis label
        x_label = f"{ch_x} [{x_units}]" if x_units else ch_x
        axis_space = 16 * ph.scale
        val_space = 40 * ph.scale
        nref = 2 if self.data_view.alt_lap else 1
        if nref == 2:
            val_space += 8 * ph.scale
        rect = ph.painter.drawText(
            int(gh.graph_area.left()), int(gh.graph_area.bottom() + axis_space),
            int(gh.graph_area.width() - val_space * 0.5  * nref),
            int(axis_space),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            x_label)
        ph.painter.drawText(rect.right() + 10 * ph.scale,
                            int(gh.graph_area.bottom() + axis_space),
                            int(gh.graph_area.width()), int(axis_space),
                            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                            '%s%.*f' % ('\u278a ' if nref == 2 else '',
                                        cds[0].dec_pts, data[-1][3][0]))
        if self.data_view.alt_lap:
            ph.painter.drawText(rect.right() + 10 * ph.scale + val_space,
                                int(gh.graph_area.bottom() + axis_space),
                                int(gh.graph_area.width()), int(axis_space),
                                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                                '\u278b %.*f' % (cds[0].dec_pts, data[-2][3][0]))


        # Y axis legend
        names_with_units = [f'{ch} [{cd.units}]' if cd.units else ch
                            for ch, cd in zip(self.channel_list[1:], cds[1:])]
        # font for data stats
        font = gh.channel_font
        ph.painter.setFont(font)
        fontMetrics = QtGui.QFontMetrics(font)
        M_space = fontMetrics.horizontalAdvance('\u25bc ')
        self.channel_name_width = max(fontMetrics.horizontalAdvance(name) for name in names_with_units) + 2 * M_space
        self.channel_ind_width = M_space if self.channel_name_width else 0
        self.channel_value_width = M_space + max(
            fontMetrics.horizontalAdvance('%.*f' % (cds[1].dec_pts, gh.y_axis.logical_min_val)),
            fontMetrics.horizontalAdvance('%.*f' % (cds[1].dec_pts, gh.y_axis.logical_max_val)))
        self.channel_opt_width = 2 * (self.channel_value_width
                                      if self.channel_name_width and self.data_view.alt_lap
                                      else 0)
        # color background for text
        ph.painter.fillRect(QRectF(gh.graph_area.topLeft(),
                                   QSizeF(self.channel_ind_width + self.channel_name_width +
                                          self.channel_value_width + self.channel_opt_width,
                                          12 + fontMetrics.height() * len(names_with_units))),
                            QtGui.QColor(32, 32, 32, 160))

        pen2 = QtGui.QPen(state.lap_colors[1])
        pen2.setStyle(Qt.SolidLine)
        next_y = gh.graph_area.top()
        start_x = gh.graph_area.left() + 6
        for idx, (name, cd) in enumerate(zip(names_with_units, cds[1:])):
            # set pen for data
            y = next_y
            next_y += fontMetrics.height()

            pen = QtGui.QPen(channels.colors[cd.color])
            ph.painter.setPen(pen)

            if idx == len(names_with_units) - 1:
                ph.painter.drawText(start_x, y,
                                    self.channel_ind_width, fontMetrics.height(),
                                    Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                    '\u25aa')
            ph.painter.drawText(start_x + self.channel_ind_width, y,
                                self.channel_name_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                name)
            ph.painter.drawText(start_x + self.channel_ind_width + self.channel_name_width, y,
                                self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '%.*f' % (cd.dec_pts, data[-1][3][idx+1]))

            if not self.data_view.alt_lap: continue
            ph.painter.setPen(pen2)
            ph.painter.drawText(start_x + self.channel_ind_width
                                + self.channel_name_width + self.channel_value_width,
                                y, self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '%.*f' % (cd.dec_pts, data[-2][3][idx+1]))

        # Graph frame and axis
        gh.paintGraphFrame()
        gh.paintXAxis()
        gh.paintYAxis()
