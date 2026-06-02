
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""Histogram chart: value distribution for selected channels."""

import bisect

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QRectF, Qt

from . import channels
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
                values = values[selector]
                if len(values) == 0:
                    continue

                yield (ch, color, values, weights)


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
        if not dv.ref_lap or not self.channel_list:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            msg = "No data loaded" if not dv.ref_lap else "Add channels to view histogram"
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, msg)
            return

        scale = ph.scale
        margin_left = int(50 * scale)
        margin_right = int(10 * scale)
        margin_top = int(25 * scale)
        margin_bottom = int(30 * scale)

        chart_w = ph.size.width() - margin_left - margin_right
        chart_h = ph.size.height() - margin_top - margin_bottom
        if chart_w < 40 or chart_h < 20:
            return

        # Gather all the raw data (channel lap data)
        data = list(self._datasets())
        # Calculate histogram boundaries based on min/max of all the data
        bin_edges = np.histogram_bin_edges([(np.min(values), np.max(values))
                                            for _, _, values, _ in data],
                                           bins=self.NUM_BINS)

        # Y Axis scaling
        max_count = 100

        n_channels = len(data)
        bar_width_per_ch = max(1, chart_w / self.NUM_BINS / max(1, n_channels))

        # Axis pen
        axis_color = QtGui.QColor(192, 192, 192)
        axis_pen = QtGui.QPen(axis_color)
        axis_pen.setWidth(max(1, int(scale)))
        ph.painter.setPen(axis_pen)
        ph.painter.drawLine(
            int(margin_left), int(margin_top + chart_h),
            int(margin_left + chart_w), int(margin_top + chart_h))
        ph.painter.drawLine(
            int(margin_left), int(margin_top),
            int(margin_left), int(margin_top + chart_h))

        # Draw dotted grid lines
        grid_pen = QtGui.QPen(QtGui.QColor(64, 64, 64))
        grid_pen.setStyle(Qt.DotLine)
        ph.painter.setPen(grid_pen)
        for bi in range(1, len(bin_edges) - 1):
            bx = margin_left + bi * n_channels * bar_width_per_ch
            ph.painter.drawLine(bx, margin_top, bx, margin_top + chart_h)
        for i in range(10, 100, 10):
            by = margin_top + i / 100. * chart_h
            ph.painter.drawLine(margin_left, by, margin_left + chart_w, by)

        legend_y = int(5 * scale)

        for ci, ch in enumerate(self.channel_list):
            # Legend entry
            prop = self.data_view.get_channel_prop(ch)
            ph.painter.setFont(chfont)
            ph.painter.setPen(channels.colors[prop.color])
            label = f"{ch} ({prop.units})" if prop.units else ch
            ph.painter.drawText(
                int(margin_left + ci * 150 * scale), legend_y,
                int(150 * scale), int(18 * scale),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label)

        for ci, (ch, color, values, weights) in enumerate(data):
            alpha_color = QtGui.QColor(color.red(), color.green(), color.blue(), 192)

            counts = np.histogram(values, bins=bin_edges, weights=weights)[0]
            counts = counts * (100. / (np.sum(counts) or 1)) # Normalize the sum to 100%

            for bi, count in enumerate(counts):
                if count == 0:
                    continue
                bar_h = (count / max_count) * chart_h
                bx = margin_left + (bi * n_channels + ci) * bar_width_per_ch
                by = margin_top + chart_h - bar_h
                ph.painter.fillRect(QRectF(bx, by, bar_width_per_ch, bar_h), alpha_color)

        # X-axis labels (min and max)
        ph.painter.setFont(axfont)
        ph.painter.setPen(axis_color)
        ph.painter.drawText(
            int(margin_left), int(margin_top + chart_h + 2 * scale),
            int(chart_w / 2), int(margin_bottom),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            f"{bin_edges[0]:.1f}")
        ph.painter.drawText(
            int(margin_left + chart_w / 2), int(margin_top + chart_h + 2 * scale),
            int(chart_w / 2), int(margin_bottom),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            f"{bin_edges[-1]:.1f}")

        # Y-axis label
        ph.painter.setFont(axfont)
        ph.painter.setPen(axis_color)
        ph.painter.drawText(
            0, int(margin_top), int(margin_left - 5 * scale), int(chart_h),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            "Percent")
