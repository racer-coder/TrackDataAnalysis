
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from dataclasses import dataclass
import math

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPen,
)

from . import widgets

def roundUpHumanNumber(num):
    if num <= 0: return 1
    l10 = math.log10(num)
    w = math.ceil(l10)
    d = 1
    if 10. ** (l10 - w + 1) < 2:
        d = 5
    elif 10. ** (l10 - w + 1) < 5:
        d = 2
    return 10. ** w / d

@dataclass()
class AxisGrid:
    logical_min_val: float
    logical_max_val: float
    logical_tick_spacing: float
    pixel_val_spacing: float
    pixel_offset: float

    def calc(self, logical):
        return (logical - self.logical_min_val) * self.pixel_val_spacing + self.pixel_offset

    def invert(self, physical):
        return (physical - self.pixel_offset) / self.pixel_val_spacing + self.logical_min_val

    def invertRelative(self, physical):
        return physical / self.pixel_val_spacing

    def tickCoords(self):
        i = np.arange(math.ceil(self.logical_min_val / self.logical_tick_spacing),
                      math.ceil(self.logical_max_val / self.logical_tick_spacing))
        return memoryview(self.calc(i * self.logical_tick_spacing))

def makeAxisGrid(ph, min_val, max_val, pixel_base, pixel_range, tick_factor):
    pixel_per_val = pixel_range / (max_val - min_val)
    return AxisGrid(
        min_val, max_val,
        roundUpHumanNumber(ph.scale * tick_factor / abs(pixel_per_val)),
        pixel_per_val, pixel_base)

class GraphHelper:
    def __init__(self, widget, ph):
        self.widget = widget
        self.ph = ph

        self.axis_font = QFont('Tahoma')
        self.axis_font.setPixelSize(widgets.deviceScale(self.widget, 11.25))

        self.channel_font = QFont('Tahoma')
        self.channel_font.setPixelSize(widgets.deviceScale(self.widget, 13))

        self.axis_pen = QPen(QColor(192, 192, 192))
        self.grid_pen = QPen(QColor(64, 64, 64))
        self.grid_pen.setStyle(Qt.DotLine)

    def setArea(self, rect, n_y_axis, n_x_axis):
        self.total_area = rect.adjusted(0, 0, -1, 0)
        self.full_graph_area = self.total_area.adjusted(n_y_axis * 50 * self.ph.scale,
                                                        0, 0,
                                                        -n_x_axis * 16 * self.ph.scale)
        self.graph_area = self.full_graph_area


    def setXAxis(self, min_val, max_val):
        self.x_axis = makeAxisGrid(self.ph, min_val, max_val,
                                   self.graph_area.left(), self.graph_area.width(), 60)

    def setYAxis(self, min_val, max_val):
        self.y_axis = makeAxisGrid(self.ph, min_val, max_val,
                                   self.graph_area.top() + self.graph_area.height(),
                                   -self.graph_area.height(), 14)

    def paintGraphFrame(self):
        self.ph.painter.setPen(self.axis_pen)
        self.ph.painter.drawRect(self.full_graph_area)

    def paintXGrid(self):
        self.ph.painter.setPen(self.grid_pen)
        for gx in self.x_axis.tickCoords():
            self.ph.painter.drawLine(gx, self.graph_area.top(), gx, self.graph_area.bottom())

    def paintYGrid(self):
        self.ph.painter.setPen(self.grid_pen)
        for gy in self.y_axis.tickCoords():
            self.ph.painter.drawLine(self.graph_area.left(), gy, self.graph_area.right(), gy)

    def paintYAxis(self):
        if self.graph_area.left() < self.ph.rect.left():
            return # nothing to do for this update

        self.ph.painter.setFont(self.axis_font)
        text_height = QFontMetrics(self.axis_font).height()
        self.ph.painter.setPen(self.axis_pen)

        self.ph.painter.drawLine(self.graph_area.topLeft(), self.graph_area.bottomLeft())

        exp = int(math.floor(math.log10(self.y_axis.logical_tick_spacing) + .01))
        exp = max(0, -exp)
        i = np.arange(int(self.y_axis.logical_min_val / self.y_axis.logical_tick_spacing),
                      int(self.y_axis.logical_max_val / self.y_axis.logical_tick_spacing) + 1)
        atc = i * self.y_axis.logical_tick_spacing
        for tc, y in zip(atc, self.y_axis.calc(atc)):
            if y + text_height / 2 <= self.graph_area.bottom():
                self.ph.painter.drawText(0, y - 25, self.graph_area.left() - 4, 50,
                                         Qt.AlignVCenter | Qt.AlignRight | Qt.TextSingleLine,
                                         '%.*f' % (exp, tc))
        spacing = self.y_axis.logical_tick_spacing / 5
        ai = np.arange(int(self.y_axis.logical_min_val / spacing) + 1,
                       int(self.y_axis.logical_max_val / spacing) + 1)
        for i, y in zip(ai.data, self.y_axis.calc(ai * spacing).data):
            self.ph.painter.drawLine(self.graph_area.left(), y,
                                     self.graph_area.left() - (2 if i % 5 else 4), y)

    def paintXAxis(self, x_axis=None, index=0, time_format=False):
        if not x_axis: x_axis = self.x_axis
        y_offset = self.full_graph_area.bottom() + 16 * self.ph.scale * index

        self.ph.painter.setFont(self.axis_font)
        self.ph.painter.setPen(self.axis_pen)

        self.ph.painter.drawLine(self.full_graph_area.left(), y_offset,
                                 self.full_graph_area.right(), y_offset)

        exp = int(math.floor(math.log10(x_axis.logical_tick_spacing) + .01)) - 3
        if time_format:
            formatter = '%.0f:%02d' if exp >= 0 else ('%%.0f:%%0%d.%df' % (3 - exp, -exp))
        else:
            formatter = '%%.%df' % max(-exp, 0)
        ai = np.arange(int(math.ceil(x_axis.logical_min_val / x_axis.logical_tick_spacing)),
                       int(math.ceil(x_axis.logical_max_val / x_axis.logical_tick_spacing)) + 1)
        atc = ai * x_axis.logical_tick_spacing
        ax = x_axis.calc(atc)
        for tc, x in zip(atc, ax):
            self.ph.painter.drawText(x - 100, y_offset + 4, 200, 50,
                                     Qt.AlignHCenter | Qt.AlignTop | Qt.TextSingleLine,
                                     formatter %
                                     ((math.copysign(math.trunc(tc / 60000), tc),
                                       abs(tc) % 60000 / 1000) if time_format else tc))

        spacing = x_axis.logical_tick_spacing / 5
        ai = np.arange(int(math.ceil(x_axis.logical_min_val / spacing)),
                       int(math.ceil(x_axis.logical_max_val / spacing)))
        atc = ai * spacing
        ax = x_axis.calc(atc)
        for i, tc, x in zip(ai.data, atc.data, ax.data):
            self.ph.painter.drawLine(x, y_offset,
                                     x, y_offset + (2 if i % 5 else 4))
