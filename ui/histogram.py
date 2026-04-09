
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""Histogram chart: value distribution for selected channels."""

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QRectF, Qt

from . import state, widgets


class Histogram(widgets.MouseHelperWidget):
    """Component showing histogram of channel data distribution."""

    NUM_BINS = 30

    def __init__(self, data_view, st=None):
        super().__init__()
        self.data_view = data_view
        self.channel_list = list(st['channels']) if st and 'channels' in st else []
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

    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)
        ph.painter.fillRect(ph.rect, QtGui.QColor(12, 12, 12))
        ph.painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

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

        n_channels = len(self.channel_list)
        bar_width_per_ch = max(1, chart_w / self.NUM_BINS / max(1, n_channels))

        # Axis pen
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

        legend_y = int(5 * scale)

        for ci, ch in enumerate(self.channel_list):
            try:
                cd = dv.ref_lap.get_channel_data(ch, None, dv.maths)
            except Exception:
                continue
            if cd is None or not hasattr(cd, 'values') or len(cd.values) == 0:
                continue

            values = cd.values[np.isfinite(cd.values)]
            if len(values) == 0:
                continue

            counts, bin_edges = np.histogram(values, bins=self.NUM_BINS)
            max_count = np.max(counts) if np.max(counts) > 0 else 1

            color_idx = ci % len(state.lap_colors)
            bar_color = state.lap_colors[color_idx]
            alpha_color = QtGui.QColor(bar_color.red(), bar_color.green(),
                                        bar_color.blue(), 180)

            for bi in range(len(counts)):
                if counts[bi] == 0:
                    continue
                bar_h = (counts[bi] / max_count) * chart_h
                bx = margin_left + (bi * n_channels + ci) * bar_width_per_ch
                by = margin_top + chart_h - bar_h
                ph.painter.fillRect(QRectF(bx, by, bar_width_per_ch, bar_h), alpha_color)

            # X-axis labels (min and max)
            ph.painter.setPen(QtGui.QColor(160, 160, 160))
            units = cd.units if cd.units else ''
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

            # Legend entry
            ph.painter.setPen(bar_color)
            label = f"{ch} ({units})" if units else ch
            ph.painter.drawText(
                int(margin_left + ci * 150 * scale), legend_y,
                int(150 * scale), int(18 * scale),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label)

        # Y-axis label
        ph.painter.setPen(QtGui.QColor(120, 120, 120))
        ph.painter.drawText(
            0, int(margin_top), int(margin_left - 5 * scale), int(chart_h),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
            "Count")
