
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
    logical_zero: float

    def calc(self, logical):
        return (logical - self.logical_min_val) * self.pixel_val_spacing + self.pixel_offset

    def invert(self, physical):
        return (physical - self.pixel_offset) / self.pixel_val_spacing + self.logical_min_val

    def invertRelative(self, physical):
        return physical / self.pixel_val_spacing

    def tickIndices(self, subtick=1):
        spacing = self.logical_tick_spacing / subtick
        return np.arange(math.ceil((self.logical_min_val - self.logical_zero) / spacing),
                         math.floor((self.logical_max_val - self.logical_zero) / spacing + 1)) / subtick

    def tickVals(self, subtick=1):
        return self.logical_zero + self.logical_tick_spacing * self.tickIndices(subtick=subtick)

    def tickCoords(self, subtick=1):
        return memoryview(self.calc(self.tickVals(subtick=subtick)))

def makeAxisGrid(ph, min_val, max_val, pixel_base, pixel_range, tick_factor, zero_val=0):
    pixel_per_val = pixel_range / (max_val - min_val)
    return AxisGrid(
        min_val, max_val,
        roundUpHumanNumber(ph.scale * tick_factor / abs(pixel_per_val)),
        pixel_per_val, pixel_base, zero_val)

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


    def setXAxis(self, min_val, max_val, zero_val=0):
        self.x_axis = makeAxisGrid(self.ph, min_val, max_val,
                                   self.graph_area.left(), self.graph_area.width(), 60, zero_val)

    def setYAxis(self, min_val, max_val, zero_val=0):
        self.y_axis = makeAxisGrid(self.ph, min_val, max_val,
                                   self.graph_area.top() + self.graph_area.height(),
                                   -self.graph_area.height(), 14, zero_val)

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
        atc = self.y_axis.tickVals()
        for tc, y in zip(atc, self.y_axis.calc(atc)):
            if y + text_height / 2 <= self.graph_area.bottom():
                self.ph.painter.drawText(0, y - 25, self.graph_area.left() - 4, 50,
                                         Qt.AlignVCenter | Qt.AlignRight | Qt.TextSingleLine,
                                         '%.*f' % (exp, tc))
        ai = self.y_axis.tickIndices(subtick=5)
        for i, y in zip(ai.data, self.y_axis.calc(ai * self.y_axis.logical_tick_spacing +
                                                  self.y_axis.logical_zero).data):
            self.ph.painter.drawLine(self.graph_area.left(), y,
                                     self.graph_area.left() - (2 if i % 1 else 4), y)

    def paintXAxis(self, x_axis=None, index=0, time_format=False, subtick=5):
        if not x_axis: x_axis = self.x_axis
        y_offset = self.full_graph_area.bottom() + 16 * self.ph.scale * index

        self.ph.painter.setFont(self.axis_font)
        self.ph.painter.setPen(self.axis_pen)

        self.ph.painter.drawLine(self.full_graph_area.left(), y_offset,
                                 self.full_graph_area.right(), y_offset)

        exp = int(math.floor(math.log10(x_axis.logical_tick_spacing) + .01))
        if time_format:
            exp -= 3
            formatter = '%.0f:%02d' if exp >= 0 else ('%%.0f:%%0%d.%df' % (3 - exp, -exp))
        else:
            formatter = '%%.%df' % max(-exp, 0)
        atc = x_axis.tickVals()
        for tc, x in zip(atc, x_axis.calc(atc)):
            self.ph.painter.drawText(x - 100, y_offset + 4, 200, 50,
                                     Qt.AlignHCenter | Qt.AlignTop | Qt.TextSingleLine,
                                     formatter %
                                     ((math.copysign(math.trunc(tc / 60000), tc),
                                       abs(tc) % 60000 / 1000) if time_format else tc))

        ai = x_axis.tickIndices(subtick=subtick)
        ax = x_axis.calc(ai * x_axis.logical_tick_spacing + x_axis.logical_zero)
        for i, x in zip(ai.data, ax.data):
            self.ph.painter.drawLine(x, y_offset,
                                     x, y_offset + (2 if i % 1 else 4))
