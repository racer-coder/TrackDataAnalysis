
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
from dataclasses import dataclass
import math

import numpy as np

from PySide2 import QtGui
from PySide2.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide2.QtWidgets import (
    QAction,
    QApplication,
    QMenu,
)

from data import unitconv
from data.distance import ChannelData
from . import channels
from .math import channel_editor
from . import state
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

class TimeDist(widgets.MouseHelperWidget):
    font_map = {
        'axis': ('Tahoma', 11.25),
        'channel': ('Tahoma', 13),
    }
    CURSOR_WIDTH = 10 # scale?

    def __init__(self, dataView, lapView=None, state=None):
        super().__init__()
        self.dataView = dataView
        if state:
            self.lapView = state['lapView']
            self.channelGroups = state['channels']
            self.current_channel = state['selected']
            # timeslip is handled later
        else:
            assert lapView is not None
            self.lapView = lapView
            self.channelGroups = []
            self.current_channel = (None, None)
        self.x_axis = None
        self.zoom_highlight = None

        self.cursorClick = widgets.MouseHelperItem(
            clicks=[widgets.MouseHelperClick(Qt.LeftButton,
                                             state_capture=self.cursorJump,
                                             move=self.cursorDrag),
                    widgets.MouseHelperClick(Qt.LeftButton, double=True,
                                             state_capture=self.zoom_sel_start,
                                             move=self.zoom_sel_drag,
                                             release=self.zoom_sel_release)],
            wheel=self.graph_wheel)
        self.addMouseHelperTop(self.cursorClick)

        self.xaxisClick = widgets.MouseHelperItem(
            cursor=Qt.OpenHandCursor,
            clicks=[widgets.MouseHelperClick(Qt.LeftButton,
                                             state_capture=self.xaxisCapture,
                                             move=self.xaxisDrag)])
        if self.lapView:
            self.addMouseHelperTop(self.xaxisClick)

        self.offsetClick = widgets.MouseHelperItem(
            cursor=Qt.OpenHandCursor,
            clicks=[widgets.MouseHelperClick(Qt.LeftButton,
                                             state_capture=self.offsetCapture,
                                             move=self.offsetDrag)])
        if self.lapView:
            self.addMouseHelperTop(self.offsetClick)

        self.leftClick = widgets.MouseHelperItem(
            cursor = Qt.SizeHorCursor,
            clicks=[widgets.MouseHelperClick(Qt.LeftButton, move=self.leftDrag)])
        self.rightClick = widgets.MouseHelperItem(
            cursor = Qt.SizeHorCursor,
            clicks=[widgets.MouseHelperClick(Qt.LeftButton, move=self.rightDrag)])
        if not self.lapView:
            self.addMouseHelperTop([self.leftClick, self.rightClick])

        self.graph_mouse_helpers = []
        self.addMouseHelperTop(self.graph_mouse_helpers)
        self.channel_mouse_helpers = []
        self.addMouseHelperTop(self.channel_mouse_helpers)

        self.time_slip = QAction('Show Time Slip', self)
        self.time_slip.setCheckable(True)
        self.time_slip.setChecked(bool(state and state['timeslip']))
        self.time_slip.toggled.connect(self.toggle_time_slip)
        if self.lapView:
            self.addAction(self.time_slip)

        self.setAcceptDrops(True) # for reordering channels

        self.axis_pen = QtGui.QPen(QtGui.QColor(192, 192, 192))
        self.drop_indicator = None

    def save_state(self):
        return {'type': 'timedist',
                'base': self.parentWidget().save_state(),
                'lapView': self.lapView,
                'channels': self.channelGroups,
                'selected': self.current_channel,
                'timeslip': self.time_slip.isChecked(),
                }

    def toggle_time_slip(self, flag):
        if flag:
            self.channelGroups.append(['Time Slip'])
        else:
            self.tryRemoveChannel('Time Slip')
        self.update()

    def leftDrag(self, rel_pos, abs_pos, saved_state):
        self.dataView.zoom_window = (
            self.dataView.makeTD(self.x_axis.invert(abs_pos.x()), False),
            self.dataView.zoom_window[1])
        self.dataView.values_change.emit()

    def rightDrag(self, rel_pos, abs_pos, saved_state):
        span = self.dataView.getLapValue(self.dataView.ref_lap)
        self.dataView.zoom_window = (
            self.dataView.zoom_window[0],
            self.dataView.makeTD(self.x_axis.invert(abs_pos.x()) - (span[1] - span[0]), True))
        self.dataView.values_change.emit()

    def offsetCapture(self, absPos):
        idx = int((absPos.y() - self.offsetClick.geometry.y()) / (16 * self.devicePixelRatioF()))
        return (idx, self.shift_axis[idx][0].offset)

    def offsetDrag(self, relPos, absPos, axis):
        rel = self.x_axis.invertRelative(relPos.x())
        self.shift_axis[axis[0]][0].offset = self.dataView.makeTD(
            self.dataView.getTDValue(axis[1]) - rel, False)
        self.dataView.values_change.emit()

    def channelName(self, ch, units=None):
        if not units and self.dataView.ref_lap and ch in self.dataView.channel_properties:
            units = self.dataView.channel_properties[ch].units
        return '%s [%s]' % (ch, unitconv.display_text(units)) if units else ch

    def graph_wheel(self, angle):
        if angle.x():
            rel = self.x_axis.invertRelative(angle.x())
            self.dataView.zoom_window = (
                self.dataView.makeTD(self.dataView.getTDValue(self.dataView.zoom_window[0]) - rel,
                                     False),
                self.dataView.makeTD(self.dataView.getTDValue(self.dataView.zoom_window[1]) - rel,
                                     True))
            self.dataView.values_change.emit()
        if angle.y():
            window_size = self.dataView.windowSize2Mode()
            cursor_mode = self.dataView.getTDValue(self.dataView.cursor_time)
            cursor_to_window = cursor_mode - self.dataView.getTDValue(self.dataView.zoom_window[0])
            new_window_size = window_size * 2 ** (angle.y() / 540)
            new_c2w = cursor_to_window * new_window_size / window_size
            z0 = cursor_mode - new_c2w
            lap_range = self.dataView.getLapValue(self.dataView.ref_lap)
            z1 = new_window_size - lap_range[1] + lap_range[0] + z0
            self.dataView.zoom_window = (
                self.dataView.makeTD(z0, False),
                self.dataView.makeTD(z1, True))
            self.dataView.values_change.emit()

    def cursorJump(self, absPos):
        old_cursor = self.dataView.cursor_time
        self.dataView.cursor_time = self.dataView.makeTD(self.x_axis.invert(absPos.x()), False)
        self.dataView.cursor_change.emit(old_cursor)

    def cursorDrag(self, relPos, absPos, savedState):
        self.cursorJump(absPos)

    def zoom_sel_start(self, absPos):
        return min(max(self.graph_x, absPos.x()), self.graph_max)

    def zoom_sel_drag(self, relPos, absPos, start_x):
        self.zoom_highlight = (start_x, self.zoom_sel_start(absPos))
        old_cursor = self.dataView.cursor_time
        self.dataView.cursor_time = self.dataView.makeTD(self.x_axis.invert(absPos.x()), False)
        self.dataView.cursor_change.emit(old_cursor)
        self.update()

    def zoom_sel_release(self, relPos, absPos, start_x):
        zoom_highlight = (self.x_axis.invert(start_x),
                          self.x_axis.invert(self.zoom_sel_start(absPos)))
        self.zoom_highlight = None

        lap_range = self.dataView.getLapValue(self.dataView.ref_lap)
        self.dataView.zoom_window = (
            self.dataView.makeTD(zoom_highlight[0], False),
            self.dataView.makeTD(zoom_highlight[1] - lap_range[1] + lap_range[0], True))
        old_cursor = self.dataView.cursor_time
        self.dataView.cursor_time = self.dataView.zoom_window[0]
        self.dataView.cursor_change.emit(old_cursor)
        self.dataView.values_change.emit()

    def xaxisCapture(self, absPos):
        return self.dataView.zoom_window

    def xaxisDrag(self, relPos, absPos, origZoom):
        rel = self.x_axis.invertRelative(relPos.x())
        self.dataView.zoom_window = (
            self.dataView.makeTD(self.dataView.getTDValue(origZoom[0]) - rel, False),
            self.dataView.makeTD(self.dataView.getTDValue(origZoom[1]) - rel, True))
        self.dataView.values_change.emit()

    def selectFont(self, why):
        stats = self.font_map[why]
        font = QtGui.QFont(stats[0])
        font.setPixelSize(widgets.deviceScale(self, stats[1]))
        return font

    def calc_time_slip(self, laps, var):
        if not laps: return
        target_window = self.dataView.lapTime2Mode(self.dataView.ref_lap,
                                                   self.dataView.ref_lap.duration())
        for lapref, _color, _ in laps:
            start_idx = max(0,
                            bisect.bisect_left(lapref.log.log.dist_map_time,
                                               self.dataView.offMode2outTime(lapref, 0)) - 1)
            end_idx = max(0,
                          bisect.bisect_left(
                              lapref.log.log.dist_map_time,
                              self.dataView.offMode2outTime(lapref, target_window)) - 1)

            time_base = lapref.start.time + lapref.offset.time
            dist_base = lapref.start.dist + lapref.offset.dist
            var.append(ChannelData.from_data(
                timecodes = lapref.log.log.dist_map_time[start_idx:end_idx],
                distances = lapref.log.log.dist_map_dist[start_idx:end_idx],
                values = [] if lapref == self.dataView.ref_lap else
                (np.subtract(lapref.log.log.dist_map_time[start_idx:end_idx], time_base) -
                 self.dataView.ref_lap.offDist2Time(lapref.log.log.dist_map_dist[start_idx:end_idx] - dist_base)) / 1000,
                units = 's',
                dec_pts = 3,
                interpolate = True))

    def paintGraph(self, ph, y_offset, height, channel_list, graph_idx):
        if not self.x_axis: return
        # get laps
        laps = self.dataView.get_laps()
        laps[0] = (self.dataView.ref_lap, None, 0) # special color to mean use channel color
        # get data
        data = [d
                for ch in channel_list
                for d in (self.time_slip_data if ch == 'Time Slip' else
                          [self.dataView.get_channel_data(self.dataView.ref_lap, ch)])]
        # calc min/max data
        dmin = min([d.min for d in data if d.min is not None], default=0)
        dmax = max([d.max for d in data if d.max is not None], default=0)
        if dmin == dmax:
            dmin -= 1
            dmax += 1
        # add buffer
        diff = dmax - dmin
        dmax += diff * .03
        dmin -= diff * .03
        # construct y axis base and spacing
        y_axis = AxisGrid(dmax, dmin,
                          roundUpHumanNumber((dmax-dmin) / (height / ph.scale / 14)),
                          height / (dmin - dmax), y_offset)
        # set pen for grid
        pen = QtGui.QPen(QtGui.QColor(64, 64, 64))
        pen.setStyle(Qt.DotLine)
        ph.painter.setPen(pen)
        # draw x grid
        i = np.arange(math.ceil(self.x_axis.logical_min_val / self.x_axis.logical_tick_spacing),
                      math.ceil(self.x_axis.logical_max_val / self.x_axis.logical_tick_spacing))
        agx = self.x_axis.calc(i * self.x_axis.logical_tick_spacing)
        for gx in memoryview(agx):
            ph.painter.drawLine(gx, y_offset, gx, y_offset + height)

        # draw y grid
        i = np.arange(math.ceil(dmin / y_axis.logical_tick_spacing),
                      math.ceil(dmax / y_axis.logical_tick_spacing))
        agy = y_axis.calc(i * y_axis.logical_tick_spacing)
        for gy in memoryview(agy):
            ph.painter.drawLine(self.graph_x, gy, ph.size.width(), gy)
        # draw data
        draw_channels = channel_list
        if graph_idx == self.current_channel[0]:
            draw_channels = ([d for d in draw_channels if d != self.current_channel[1]] +
                             [d for d in draw_channels if d == self.current_channel[1]])
        for lidx, (lap, color, _) in list(enumerate(laps if self.lapView else laps[:1]))[::-1]:
            for ch in draw_channels:
                if ch == 'Time Slip':
                    d = self.time_slip_data[lidx]
                else:
                    d = self.dataView.get_channel_data(lap, ch)
                if not len(d.values): continue
                # set pen for data
                pen = QtGui.QPen(color if color else channels.colors[d.color])
                ph.painter.setPen(pen)

                xa = d.timecodes if self.dataView.mode_time else d.distances
                lap_base = self.dataView.getTDValue(lap.start) + self.dataView.getTDValue(lap.offset)
                search = self.x_axis.invert(max(ph.rect.left(), self.graph_x) - 0.5) + lap_base
                start_idx = max(0, bisect.bisect_left(xa, search) - 1)
                search = self.x_axis.invert(max(ph.rect.right() + 0.5, self.graph_x)) + lap_base
                end_idx = min(len(xa), bisect.bisect_right(xa, search) + 1)
                xa = memoryview(np.round(self.x_axis.calc(np.subtract(xa[start_idx:end_idx], lap_base))).astype(int))
                dv = np.round(y_axis.calc(d.values[start_idx:end_idx])).astype(int)
                dvd1 = dv.data
                xa_uniqval, xa_uniqidx = np.unique(xa, return_index=True)
                umin = np.minimum.reduceat(dv, xa_uniqidx)
                umax = np.maximum.reduceat(dv, xa_uniqidx)
                if d.interpolate:
                    dvd2 = dv[1:].data
                else:
                    dvd2 = dvd1
                    dvval = dv[xa_uniqidx[1:]-1]
                    np.minimum(umin[1:], dvval, out=umin[1:])
                    np.maximum(umax[1:], dvval, out=umax[1:])
                ph.painter.save()
                ph.painter.setClipRect(self.graph_x, y_offset, ph.size.width(), height)
                # paint lines that live across pixel columns
                for idx in memoryview(xa_uniqidx[1:] - 1):
                    ph.painter.drawLine(xa[idx], dvd1[idx], xa[idx+1], dvd2[idx])
                # paint lines within pixel columns
                for x, y1, y2 in zip(xa_uniqval.data, umin.data, umax.data):
                    if y1 != y2:
                        ph.painter.drawLine(x, y1, x, y2)
                ph.painter.restore()
        # font for data stats
        font = self.selectFont('channel')
        ph.painter.setFont(font)
        fontMetrics = QtGui.QFontMetrics(font)
        # color background for text
        ph.painter.fillRect(QRect(QPoint(self.graph_x, y_offset),
                                  QSize(self.channel_ind_width + self.channel_name_width + self.channel_value_width + self.channel_opt_width,
                                        12 + fontMetrics.height() * len(channel_list))),
                            QtGui.QColor(32, 32, 32, 160))
        nminmax = 4 if self.dataView.alt_lap else 2
        ph.painter.fillRect(QRect(QPoint(ph.size.width() - self.channel_minmax_width * nminmax,
                                         y_offset),
                                  QSize(self.channel_minmax_width * nminmax,
                                        12 + fontMetrics.height() * len(channel_list))),
                            QtGui.QColor(32, 32, 32, 160))
        # text for data
        pen2 = QtGui.QPen(state.lap_colors[1])
        pen2.setStyle(Qt.SolidLine)
        next_y = y_offset
        for (color, ch, lap), d in zip(
                [trip
                 for ch in channel_list
                 for trip in ([(c, 'Time Slip', l) for l, c, _ in laps] if ch == 'Time Slip' else
                              [(channels.colors[self.dataView.get_channel_prop(ch).color], ch,
                                self.dataView.ref_lap)])],
                data):
            first_timeslip = ch == 'Time Slip' and (len(self.time_slip_data) <= 1 or
                                                    d == self.time_slip_data[1])
            if color is None:
                if not first_timeslip:
                    continue
                color = state.lap_colors[0] # We don't have any alternate laps, but we still want to show an entry for Time Slip
            # set pen for data
            y = next_y
            next_y += fontMetrics.height()

            pen = QtGui.QPen(color)
            ph.painter.setPen(pen)

            if graph_idx == self.current_channel[0] and ch == self.current_channel[1]:
                ph.painter.drawText(self.graph_x + 6, y,
                                    self.channel_ind_width, fontMetrics.height(),
                                    Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                    '\u25aa')
            ph.painter.drawText(self.graph_x + 6 + self.channel_ind_width, y,
                                self.channel_name_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                self.channelName(ch, d.units))
            if ch != 'Time Slip' or first_timeslip:
                chh = max(len(self.time_slip_data) - 1, 1) if ch == 'Time Slip' else 1
                self.channel_mouse_helpers.append(
                    widgets.MouseHelperItem(
                        geometry=QRectF(self.graph_x + 6 + self.channel_ind_width, y,
                                        self.channel_name_width, chh * fontMetrics.height()),
                        clicks=[widgets.MouseHelperClick(button_type=Qt.LeftButton,
                                                         state_capture=self.selectChannel,
                                                         move=self.moveChannel)],
                        channel=(graph_idx, ch),
                        drop_rect=QRect(self.graph_x, y - 2,
                                        ph.size.width() - self.graph_x, 4)))
            if not len(d.values): continue
            main_val = d.interp(self.dataView.cursor2outTime(lap))
            self.cursor_values.append(y_axis.calc(main_val))
            ph.painter.drawText(self.graph_x + 6 + self.channel_ind_width + self.channel_name_width, y,
                                self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '%.*f' % (d.dec_pts, main_val))

            # print min/max for zoom window
            if not self.lapView: continue
            zoomsize = self.dataView.windowSize2Mode()
            drange = d.values[
                bisect.bisect_left(
                    d.timecodes if self.dataView.mode_time else d.distances,
                    self.dataView.offMode2outMode(
                        lap, self.dataView.getTDValue(self.dataView.zoom_window[0])))
                : bisect.bisect_left(
                    d.timecodes if self.dataView.mode_time else d.distances,
                    self.dataView.offMode2outMode(
                        lap, zoomsize + self.dataView.getTDValue(self.dataView.zoom_window[0])))]
            ph.painter.drawText(ph.size.width() - self.channel_minmax_width * nminmax + 6,
                                y, self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '\u25bc' + (' %.*f' % (d.dec_pts, np.min(drange)) if len(drange) else ''))
            ph.painter.drawText(ph.size.width() - self.channel_minmax_width * nminmax / 2 + 6,
                                y, self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '\u25b2' + (' %.*f' % (d.dec_pts, np.max(drange)) if len(drange) else ''))
            if not self.dataView.alt_lap or ch == 'Time Slip': continue
            d2 = self.dataView.get_channel_data(self.dataView.alt_lap, ch)
            if not len(d2.values): continue
            ph.painter.setPen(pen2)
            ph.painter.drawText(self.graph_x + 6 + self.channel_ind_width
                                + self.channel_name_width + self.channel_value_width,
                                y, self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '%.*f' % (d.dec_pts,
                                          d2.interp(self.dataView.cursor2outTime(self.dataView.alt_lap))))
            ph.painter.drawText(self.graph_x + 6 + self.channel_ind_width
                                + self.channel_name_width + self.channel_value_width
                                + self.channel_opt_width / 2,
                                y, self.channel_value_width, fontMetrics.height(),
                                Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                '\u0394 %.*f' % (d.dec_pts, d2.values[start_idx] - main_val))
            drange = d.values[
                bisect.bisect_left(
                    d.timecodes if self.dataView.mode_time else d.distances,
                    self.dataView.offMode2outMode(
                        self.dataView.alt_lap,
                        self.dataView.getTDValue(self.dataView.zoom_window[0])))
                : bisect.bisect_left(
                    d.timecodes if self.dataView.mode_time else d.distances,
                    self.dataView.offMode2outMode(
                        self.dataView.alt_lap,
                        zoomsize + self.dataView.getTDValue(self.dataView.zoom_window[0])))]
            if len(drange):
                ph.painter.drawText(ph.size.width() - self.channel_minmax_width * 3 + 6,
                                    y, self.channel_value_width, fontMetrics.height(),
                                    Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                    '%.*f' % (d.dec_pts, np.min(drange)))
                ph.painter.drawText(ph.size.width() - self.channel_minmax_width + 6,
                                    y, self.channel_value_width, fontMetrics.height(),
                                    Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine,
                                    '%.*f' % (d.dec_pts, np.max(drange)))

        self.graph_mouse_helpers.append(
            widgets.MouseHelperItem(
                geometry=QRectF(self.graph_x, y_offset,
                                ph.size.width() - self.graph_x, (next_y - y_offset + height) / 2),
                graph_idx=(graph_idx, False),
                drop_rect=QRect(self.graph_x, next_y - 2,
                                ph.size.width() - self.graph_x, 4)))
        self.graph_mouse_helpers.append(
            widgets.MouseHelperItem(
                geometry=QRectF(self.graph_x, (next_y + y_offset + height)/2,
                                ph.size.width() - self.graph_x, (y_offset - next_y + height)/2),
                graph_idx=(graph_idx+1, True),
                drop_rect=QRect(self.graph_x, (next_y + y_offset + height)/2,
                                ph.size.width() - self.graph_x, (y_offset - next_y + height)/2)))

        # draw Y axis
        self.paintYAxis(ph, y_offset, height, y_axis)

    def paintYAxis(self, ph, y_offset, height, y_axis):
        if self.graph_x < ph.rect.left():
            return # nothing to do

        ph.painter.setFont(self.axis_font)
        ph.painter.setPen(self.axis_pen)

        ph.painter.drawLine(self.graph_x, y_offset, self.graph_x, y_offset + height)

        exp = int(math.floor(math.log10(y_axis.logical_tick_spacing) + .01))
        exp = max(0, -exp)
        i = np.arange(int(y_axis.logical_max_val / y_axis.logical_tick_spacing) + 1,
                      int(y_axis.logical_min_val / y_axis.logical_tick_spacing) + 1)
        atc = i * y_axis.logical_tick_spacing
        for tc, y in zip(atc, y_axis.calc(atc)):
            ph.painter.drawText(0, y - 25, self.graph_x - 4, 50,
                                Qt.AlignVCenter | Qt.AlignRight | Qt.TextSingleLine,
                                '%.*f' % (exp, tc))
        spacing = y_axis.logical_tick_spacing / 5
        ai = np.arange(int(y_axis.logical_max_val / spacing) + 1,
                       int(y_axis.logical_min_val / spacing) + 1)
        for i, y in zip(ai.data, y_axis.calc(ai * spacing).data):
            ph.painter.drawLine(self.graph_x, y,
                                self.graph_x - (2 if i % 5 else 4), y)

    def paintXAxis(self, ph, y_offset, x_axis):
        if not self.dataView.ref_lap: return

        ph.painter.setFont(self.axis_font)
        ph.painter.setPen(self.axis_pen)

        ph.painter.drawLine(self.graph_x, y_offset, ph.size.width(), y_offset)

        exp = int(math.floor(math.log10(x_axis.logical_tick_spacing) + .01)) - 3
        if self.dataView.mode_time:
            formatter = '%.0f:%02d' if exp >= 0 else ('%%.0f:%%0%d.%df' % (3 - exp, -exp))
        else:
            formatter = '%%.%df' % max(-exp, 0)
        ai = np.arange(int(math.ceil(x_axis.logical_min_val / x_axis.logical_tick_spacing)),
                       int(math.ceil(x_axis.logical_max_val / x_axis.logical_tick_spacing)) + 1)
        atc = ai * x_axis.logical_tick_spacing
        ax = x_axis.calc(atc)
        for tc, x in zip(atc, ax):
            ph.painter.drawText(x - 100, y_offset + 4, 200, 50,
                                Qt.AlignHCenter | Qt.AlignTop | Qt.TextSingleLine,
                                formatter %
                                ((math.copysign(math.trunc(tc / 60000), tc),
                                  abs(tc) % 60000 / 1000) if self.dataView.mode_time else tc))

        spacing = x_axis.logical_tick_spacing / 5
        ai = np.arange(int(math.ceil(x_axis.logical_min_val / spacing)),
                       int(math.ceil(x_axis.logical_max_val / spacing)))
        atc = ai * spacing
        ax = x_axis.calc(atc)
        for i, tc, x in zip(ai.data, atc.data, ax.data):
            ph.painter.drawLine(x, y_offset,
                                x, y_offset + (2 if i % 5 else 4))

    def minmax_width(self, channel_font_metrics, d):
        if isinstance(d, list):
            return max(self.minmax_width(channel_font_metrics, x) for x in d)
        return max(channel_font_metrics.horizontalAdvance('%.*f' % (d.dec_pts, d.min or 0)),
                   channel_font_metrics.horizontalAdvance('%.*f' % (d.dec_pts, d.max or 0)))

    def paintEvent(self, event):
        ph = widgets.makePaintHelper(self, event)

        self.axis_font = self.selectFont('axis')

        self.graph_x = 50 * ph.scale
        self.graph_max = ph.size.width()

        self.time_slip_data = []
        if self.time_slip.isChecked():
            self.calc_time_slip(self.dataView.get_laps(), self.time_slip_data)

        channel_font_metrics = QtGui.QFontMetrics(self.selectFont('channel'))
        M_space = channel_font_metrics.horizontalAdvance('\u25bc ')
        self.channel_name_width = max(
            [channel_font_metrics.horizontalAdvance(self.channelName(ch)) + 2 * M_space
             for grp in self.channelGroups
             for ch in grp], default=0)
        self.channel_ind_width = M_space if self.channel_name_width else 0
        self.channel_value_width = M_space + max([
            self.minmax_width(channel_font_metrics,
                              self.time_slip_data if ch == 'Time Slip' else
                              self.dataView.get_channel_data(self.dataView.ref_lap, ch))
            for grp in self.channelGroups
            for ch in grp
            if self.dataView.ref_lap], default=0)
        self.channel_opt_width = 2 * (self.channel_value_width
                                      if self.channel_name_width and self.dataView.alt_lap
                                      and self.lapView else 0)
        self.channel_minmax_width = (
            channel_font_metrics.horizontalAdvance('M ') + self.channel_value_width
            if self.channel_name_width and self.lapView else 0)
        self.graph_mouse_helpers.clear()
        self.channel_mouse_helpers.clear()

        # calculate X axis spacing
        self.x_axis = None
        if self.dataView.ref_lap:
            if self.lapView:
                data_range = self.dataView.windowSize2Mode()
                zero_offset = self.dataView.getTDValue(self.dataView.zoom_window[0])
            else:
                data_range = self.dataView.outTime2Mode(
                    self.dataView.ref_lap,
                    self.dataView.ref_lap.log.laps[-1].end.time)
                zero_offset = -self.dataView.getLapValue(self.dataView.ref_lap)[0] - self.dataView.getTDValue(self.dataView.ref_lap.offset)
            if data_range > 0: # maybe we have no distance data?
                est_spacing = roundUpHumanNumber(data_range / ((ph.size.width() - self.graph_x) / ph.scale / 60))
                self.x_axis = AxisGrid(zero_offset, zero_offset + data_range, est_spacing,
                                       (ph.size.width() - self.graph_x) / data_range, self.graph_x)
        if not self.dataView.mode_offset or not self.lapView or self.dataView.active_component != self:
            self.shift_axis = []
        else:
            self.shift_axis = [
                (lap, AxisGrid(zero_offset + self.dataView.getTDValue(lap.offset),
                               zero_offset + self.dataView.getTDValue(lap.offset) + data_range,
                               est_spacing,
                               (ph.size.width() - self.graph_x) / data_range, self.graph_x))
                for lap, color, idx in self.dataView.get_laps()]
        y_div = ph.size.height() - 16 * ph.scale * (1 + len(self.shift_axis))

        # grey out area outside of the current lap
        if self.x_axis:
            start_x = self.x_axis.calc(0)
            if start_x > self.graph_x:
                ph.painter.fillRect(QRect(self.graph_x, 0, start_x - self.graph_x, y_div),
                                    QtGui.QColor(48, 48, 48))
            end_x = self.x_axis.calc(
                self.dataView.lapTime2Mode(self.dataView.ref_lap, self.dataView.ref_lap.duration()))
            if end_x < ph.size.width():
                ph.painter.fillRect(QRect(end_x, 0, ph.size.width() - end_x, y_div),
                                    QtGui.QColor(48, 48, 48))

        # lap window
        font = self.axis_font
        ph.painter.setFont(font)
        fontMetrics = QtGui.QFontMetrics(font)
        graph_y = 4 + fontMetrics.height()
        ph.painter.setPen(self.axis_pen)
        # draw outline
        ph.painter.drawRect(self.graph_x, 0, ph.size.width() - self.graph_x - 1, graph_y - 2)
        # draw laps
        if self.x_axis:
            right_x = self.graph_x * 2
            for lap in self.dataView.ref_lap.log.laps:
                start_x = self.x_axis.calc(
                    self.dataView.getTDValue(lap.start) -
                    self.dataView.getTDValue(self.dataView.ref_lap.start))
                end_x = self.x_axis.calc(
                    self.dataView.getTDValue(lap.end) -
                    self.dataView.getTDValue(self.dataView.ref_lap.start))
                text = str(lap.num)
                width = fontMetrics.horizontalAdvance(text) * 1.5
                if start_x + end_x - width < right_x:
                    continue
                ph.painter.drawText((start_x + end_x - width) / 2, 1, width, graph_y,
                                    Qt.AlignTop | Qt.AlignHCenter | Qt.TextSingleLine,
                                    text)
                right_x = start_x + end_x + width

        # paint each channel group graph
        self.cursor_values = []
        groups = self.channelGroups or [[]]
        # each graph
        last_cutoff = graph_y
        separation = 6
        for i, grp in enumerate(groups):
            next_cutoff = graph_y + i * separation + (i + 1) * (y_div - graph_y - separation * (len(groups) - 1)) // len(groups)
            self.paintGraph(ph, last_cutoff, next_cutoff - last_cutoff, grp, i)
            last_cutoff = next_cutoff + separation

        # separation lines
        pen = QtGui.QPen(QtGui.QColor(64, 64, 64))
        pen.setStyle(Qt.SolidLine)
        ph.painter.setPen(pen)
        for i in range(1, len(groups)):
            cutoff = graph_y + i * separation + i * (y_div - graph_y - separation * (len(groups) - 1)) / len(groups)
            ph.painter.drawLine(self.graph_x, cutoff,
                                ph.size.width(), cutoff)
            ph.painter.drawLine(self.graph_x, cutoff - separation,
                                ph.size.width(), cutoff - separation)

        # draw lap boundaries
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setStyle(Qt.DashLine)
        ph.painter.setPen(pen)
        if self.x_axis:
            for lap in self.dataView.ref_lap.log.laps:
                x = self.x_axis.calc(
                    self.dataView.getTDValue(lap.start) -
                    self.dataView.getTDValue(self.dataView.ref_lap.start))
                if x > self.graph_x:
                    ph.painter.drawLine(x, 0, x, y_div)
                x = self.x_axis.calc(
                    self.dataView.getTDValue(lap.end) -
                    self.dataView.getTDValue(self.dataView.ref_lap.start))
                if x > self.graph_x:
                    ph.painter.drawLine(x, 0, x, y_div)

        # draw zoom window
        if not self.lapView and self.x_axis:
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen.setStyle(Qt.SolidLine)
            pen.setWidth(2)
            ph.painter.setPen(pen)
            x1 = round(self.x_axis.calc(self.dataView.getTDValue(self.dataView.zoom_window[0])))
            x2 = round(self.x_axis.calc(
                self.dataView.lapTime2Mode(self.dataView.ref_lap, self.dataView.ref_lap.duration())
                + self.dataView.getTDValue(self.dataView.zoom_window[1])))
            ph.painter.drawRect(x1, graph_y + 1, x2 - x1, y_div - graph_y - 2)

            self.leftClick.geometry.setRect(x1 - 3, graph_y + 1, 7, y_div - graph_y - 2)
            self.rightClick.geometry.setRect(x2 - 3, graph_y + 1, 7, y_div - graph_y - 2)


        # frame graph area
        ph.painter.setPen(self.axis_pen)
        ph.painter.drawRect(self.graph_x, graph_y,
                            ph.size.width() - self.graph_x - 1, y_div - graph_y)

        # paint X axis
        if self.x_axis:
            self.paintXAxis(ph, y_div, self.x_axis)
            for idx, axis in enumerate(self.shift_axis):
                self.paintXAxis(ph, y_div + 16 * ph.scale * (1 + idx), axis[1])
            ph.painter.save()
            f = QtGui.QColor(192, 192, 192)
            b = QtGui.QColor(0, 0, 0)
            if not self.dataView.mode_time: f, b = b, f
            ph.painter.setBackground(QtGui.QBrush(b))
            ph.painter.setBackgroundMode(Qt.OpaqueMode)
            ph.painter.setPen(QtGui.QPen(f))
            ph.painter.drawText(0, y_div + 4, self.graph_x, 50,
                                Qt.AlignTop | Qt.AlignRight | Qt.TextSingleLine,
                                'm:s' if self.dataView.mode_time else 'meter')
            ph.painter.restore()

        # draw drag and drop indicator, if in progress
        if self.drop_indicator:
            ph.painter.fillRect(self.drop_indicator, QtGui.QColor(64, 64, 255, 192))

        # draw zoom selection, if in progress
        if self.zoom_highlight:
            ph.painter.save()
            ph.painter.setCompositionMode(ph.painter.CompositionMode_Difference)

            l = min(self.zoom_highlight)
            h = max(self.zoom_highlight)
            ph.painter.fillRect(QRect(QPoint(l, 0), QPoint(h, y_div)),
                                QtGui.QColor(255, 255, 255))

            ph.painter.restore()

        # paint cursor
        if self.x_axis:
            x = self.x_axis.calc(self.dataView.getTDValue(self.dataView.cursor_time))
            if x >= self.graph_x and x < ph.size.width():
                pen = QtGui.QPen(QtGui.QColor(255, 255, 0))
                pen.setStyle(Qt.SolidLine)
                ph.painter.setPen(pen)
                ph.painter.drawLine(x, 0, x, y_div)
                pen.setWidth(2)
                ph.painter.setPen(pen)
                for val in self.cursor_values:
                    # adjust drawing position due to pen width=2
                    ph.painter.drawLine(max(x - self.CURSOR_WIDTH + 1, self.graph_x), val,
                                        x + self.CURSOR_WIDTH - 1, val)

        # update mouse helper boundaries
        if self.x_axis:
            self.cursorClick.geometry.setCoords(self.graph_x, 0, ph.size.width(), y_div)
            self.xaxisClick.geometry.setRect(0, y_div, ph.size.width(), 16 * ph.scale)
            self.offsetClick.geometry.setCoords(0, y_div + 16 * ph.scale, ph.size.width(), ph.size.height())
        else:
            self.cursorClick.geometry.setRect(0, 0, 0, 0)
            self.xaxisClick.geometry.setRect(0, 0, 0, 0)
            self.offsetClick.geometry.setRect(0, 0, 0, 0)
        self.lookupCursor()

    def tryRemoveChannel(self, ch):
        for grp in self.channelGroups:
            if ch in grp:
                grp.remove(ch)
                if not grp:
                    self.channelGroups.remove(grp)
                self.update_time_slip_check()
                return True
        return False

    def tryAddChannelExistingGroup(self, ch):
        if self.dataView.ref_lap:
            for grp in self.channelGroups:
                if grp and (self.dataView.channel_properties[ch].units ==
                            self.dataView.get_channel_prop(grp[0]).units):
                    grp.append(ch)
                    return True
        return False

    def addChannel(self, ch):
        # remove if channel already exists
        if not self.tryRemoveChannel(ch):
            if ((QtGui.QGuiApplication.keyboardModifiers() & Qt.ControlModifier)
                or not self.tryAddChannelExistingGroup(ch)):
                self.channelGroups.append([ch])
        self.dataView.data_change.emit()
        self.update()

    def channels(self):
        return {ch
                for grp in self.channelGroups
                for ch in grp
                if ch != 'Time Slip'}

    def updateCursor(self, old_cursor):
        if not old_cursor: return # something other than cursor updated (video_alignment?)
        if not self.x_axis: # maybe we were never painted because we're hidden?
            self.update()
            return
        old_cursor_x = self.x_axis.calc(self.dataView.getTDValue(old_cursor))
        new_cursor_x = self.x_axis.calc(self.dataView.getTDValue(self.dataView.cursor_time))
        ratio = self.devicePixelRatioF()
        minx = int((min(old_cursor_x, new_cursor_x) - self.CURSOR_WIDTH) / ratio)
        maxx = int((max(old_cursor_x, new_cursor_x) + self.CURSOR_WIDTH) / ratio) + 1
        self.update(minx, 0, maxx - minx + 1, self.height())
        self.update(
            int((self.graph_x + self.channel_ind_width + self.channel_name_width) / ratio), 0,
            int((self.channel_value_width + self.channel_opt_width) / ratio) + 1, self.height())

    def selectChannel(self, abs_pos):
        self.current_channel = self.getLastMouseHelperData('channel')
        self.update()

    def moveChannel(self, delta_pos, abs_pos, state):
        if delta_pos.manhattanLength() < QApplication.startDragDistance(): return
        channels.initiate_drag(self, self.dataView, self.current_channel[1])

    def acceptable_drop(self, e):
        return e.source() == self or (e.mimeData().hasText() and
                                      (e.mimeData().text() != 'Time Slip' or self.lapView))

    def dragEnterEvent(self, e):
        if self.acceptable_drop(e):
            e.accept()

    def dragMoveEvent(self, e):
        drop_ind = self.getEventMouseHelperData('drop_rect', e.posF())
        if drop_ind != self.drop_indicator:
            self.drop_indicator = drop_ind
            self.update()

    def dragLeaveEvent(self, e):
        self.drop_indicator = None
        self.update()

    def dropEvent(self, e):
        self.drop_indicator = None
        if not self.acceptable_drop(e): return
        dst_ch = self.getEventMouseHelperData('channel', e.posF())
        if e.source() == self:
            src_grp = self.channelGroups[self.current_channel[0]]
            channel = self.current_channel[1]
        else:
            src_grp = None
            channel = e.mimeData().text()
        if not dst_ch:
            grp = self.getEventMouseHelperData('graph_idx', e.posF())
            if grp is None:
                return # Shouldn't happen
            if src_grp is not None:
                src_grp.remove(channel)
                if not src_grp and (grp[1] or grp[0] != self.current_channel[0]):
                    self.channelGroups[self.current_channel[0]] = None
            if grp[1]:
                self.channelGroups.insert(grp[0], [])
            if not self.channelGroups:
                self.channelGroups.append([]) # dropping into an empty graph
            grp = self.channelGroups[grp[0]]
            if channel not in grp:
                grp.append(channel)
        else:
            grp = self.channelGroups[dst_ch[0]]
            dst_idx = grp.index(dst_ch[1])
            if src_grp is not None:
                src_idx = src_grp.index(channel)
                # src/dst may be equal if the groups are different, so don't short circuit equality
                if src_idx < dst_idx:
                    grp.insert(dst_idx, channel)
                    src_grp.remove(channel)
                else: # >=
                    src_grp.remove(channel)
                    grp.insert(dst_idx, channel)
                if not src_grp:
                    self.channelGroups[self.current_channel[0]] = None
            elif channel not in grp:
                grp.insert(dst_idx, channel)
        if None in self.channelGroups:
            self.channelGroups.remove(None)
        for idx, g in enumerate(self.channelGroups):
            if g is grp:
                self.current_channel = (idx, channel)
        e.accept()
        self.update_time_slip_check()
        self.dataView.data_change.emit()
        self.update()

    def update_time_slip_check(self):
        if self.time_slip.isChecked() != any('Time Slip' in chlist
                                             for chlist in self.channelGroups):
            self.time_slip.toggle()

    def channelMenuRemove(self, ch):
        self.tryRemoveChannel(ch[1])
        self.dataView.data_change.emit()
        self.update()

    def contextMenuEvent(self, event):
        ch = self.getLastMouseHelperData('channel')
        event.accept()
        if ch:
            # channel specific context menu
            menu = QMenu()
            menu.addAction(ch[1]) # dummy entry so the user knows exactly what we're operating on
            menu.addSeparator()
            menu.addAction('Edit expression...' if ch[1] in self.dataView.maths.channel_map
                           else 'Edit channel...').triggered.connect(
                                   lambda: channel_editor(self, self.dataView, ch[1]))
            menu.addSeparator()
            menu.addAction('Remove channel').triggered.connect(lambda: self.channelMenuRemove(ch))
            menu.exec_(event.globalPos())
        else:
            # general widget context menu
            QMenu.exec_(self.actions(), event.globalPos(), None, self)
