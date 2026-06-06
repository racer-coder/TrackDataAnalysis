
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""Histogram chart: value distribution for selected channels."""

import bisect
import math

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, Qt

from . import channels
from . import graphhelper
from . import state
from . import widgets


class Histogram(widgets.MouseHelperWidget):
    """Component showing histogram of channel data distribution."""

    NUM_BINS = 30

    def __init__(self, data_view, state=None):
        super().__init__()
        self.data_view = data_view
        self.channel_list = list(state['channels']) if state and 'channels' in state else []
        self.setMinimumSize(200, 100)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        self.cumulative = QtGui.QAction('Show cumulative percentage', self)
        self.cumulative.setCheckable(True)
        self.cumulative.toggled.connect(self.toggle_cumulative)
        self.addAction(self.cumulative)
        data_view.values_change.connect(self.update)

    def save_state(self):
        return {
            'type': 'histogram',
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

    def toggle_cumulative(self, flag):
        self.update()

    def _datasets(self):
        laps = self.data_view.get_laps()
        for ch in self.channel_list:
            for lap, color, idx in laps:
                data = self.data_view.get_channel_data(lap, ch)
                if idx == 1:
                    color = channels.colors[data.color]

                start_idx = bisect.bisect_left(data.timecodes, lap.start.time)
                end_idx = min(bisect.bisect_right(data.timecodes, lap.end.time),
                              len(data.timecodes) - 1)

                if end_idx < start_idx:
                    continue

                weights = (data.timecodes[start_idx+1:end_idx+1] -
                           data.timecodes[start_idx:end_idx])
                values = data.values[start_idx:end_idx]
                selector = np.isfinite(values)
                values = np.compress(selector, values)
                if len(values) == 0:
                    continue

                yield (ch, color, values, weights)


    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)
        ph.painter.fillRect(ph.rect, QtGui.QColor(12, 12, 12))

        gh = graphhelper.GraphHelper(self, ph)
        gh.setArea(QRectF(QPointF(0, 0), ph.size).adjusted(0, 25 * ph.scale, 0, 0), 1, 1)

        ph.painter.setFont(gh.channel_font)

        dv = self.data_view
        if not dv.ref_lap or not self.channel_list:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            msg = "No data loaded" if not dv.ref_lap else "Add channels to view histogram"
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, msg)
            return

        scale = ph.scale

        # Gather all the raw data (channel lap data)
        data = list(self._datasets())
        # Calculate histogram boundaries based on min/max of all the data
        bin_edges = np.histogram_bin_edges([(np.min(values), np.max(values))
                                            for _, _, values, _ in data],
                                           bins=self.NUM_BINS)

        # Y Axis scaling
        gh.setYAxis(0, 100)

        # Custom built X axis using bins
        gh.x_axis = graphhelper.AxisGrid(bin_edges[0], bin_edges[-1], bin_edges[1] - bin_edges[0],
                                         gh.graph_area.width() / (bin_edges[-1] - bin_edges[0]),
                                         gh.graph_area.left(),
                                         bin_edges[0])

        n_channels = len(data)
        bar_width_per_ch = max(1, gh.graph_area.width() / self.NUM_BINS / max(1, n_channels))

        # Draw dotted grid lines
        gh.paintXGrid()
        gh.paintYGrid()

        legend_y = int(5 * scale)

        for ci, ch in enumerate(self.channel_list):
            # Legend entry
            prop = self.data_view.get_channel_prop(ch)
            ph.painter.setFont(gh.channel_font)
            ph.painter.setPen(channels.colors[prop.color])
            label = f"{ch} [{prop.units}]" if prop.units else ch
            ph.painter.drawText(
                int(gh.graph_area.left() + ci * 150 * scale), legend_y,
                int(150 * scale), int(18 * scale),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label)

        for ci, (ch, color, values, weights) in enumerate(data):
            alpha_color = QtGui.QColor(color.red(), color.green(), color.blue(), 192)

            counts = np.histogram(values, bins=bin_edges, weights=weights)[0]
            counts = counts * (100. / (np.sum(counts) or 1)) # Normalize the sum to 100%

            total = 0
            for bi, count in enumerate(counts):
                if count == 0:
                    continue
                total += count
                bx = gh.graph_area.left() + (bi * n_channels + ci) * bar_width_per_ch
                ph.painter.fillRect(QRectF(QPointF(bx, gh.y_axis.calc(count)),
                                           QPointF(bx + bar_width_per_ch, gh.graph_area.bottom())),
                                    alpha_color)
                if self.cumulative.isChecked():
                    ph.painter.drawRect(
                        QRectF(QPointF(bx, gh.y_axis.calc(total)),
                               QPointF(bx + bar_width_per_ch, gh.graph_area.bottom())))


        gh.paintGraphFrame()

        # X-axis labels
        gh.x_axis.logical_tick_spacing *= math.ceil(
            60 * ph.scale / (gh.x_axis.logical_tick_spacing * gh.x_axis.pixel_val_spacing))
        gh.paintXAxis(subtick=1)

        # Y-axis label
        gh.paintYLabel('Percent')
        gh.paintYAxis()
