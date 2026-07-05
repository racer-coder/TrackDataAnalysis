
# Copyright 2026, Scott Smith.  MIT License (see LICENSE).

import bisect
import math

import numpy as np

from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFontMetrics,
    QGuiApplication,
    QPainter,
    QPen,
)
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QMenu,
)

from data import unitconv
from data.distance import ChannelData
from . import channels
from .math import channel_editor
from . import graphhelper
from . import state
from . import widgets

# XXX
# . in gps speed, add option for time slip
# . allow speed chart to drag cursor
# . allow speed chart to zoom

def _calc_p2m(*, t, d, v):
    # empirically seems correct (when validated against _calc_time)
    return (9 * d*d - 12*(t*v)**2 + math.sqrt(3*(3*d-2*v*t)**3*(d+2*v*t))) / (16 * t**3)

def _calc_time(*, p2m, d, v):
    return ((3*p2m*d + v**3) ** (2./3) - v*v) / (2*p2m) # validated by hand

def _calc_finalv_from_time(*, p2m, t, v): # p2m = p/m
    return np.sqrt(v*v + 2*p2m*t) # validated by hand

def _calc_finalv_from_dist(*, p2m, d, v): # p2m = p/m
    # return _calc_finalv_from_time(p2m=p2m, v=v, t=_calc_time(p2m=p2m, d=d, v=v))
    return np.cbrt(v**3 + 3*p2m*d) # validated by hand

class EfficiencyChart(widgets.MouseHelperWidget):
    def __init__(self, lc):
        super().__init__()
        self.lc = lc
        self.x_axis = None

        self.cursor_pos = 0
        self.cursorClick = widgets.MouseHelperItem(
            clicks=[widgets.MouseHelperClick(Qt.LeftButton,
                                             state_capture=self.cursorJump,
                                             move=self.cursorDrag)])
        self.addMouseHelperTop(self.cursorClick)

    def cursorJump(self, absPos):
        if self.x_axis:
            self.cursor_pos = self.x_axis.invert(absPos.x())
            self.parentWidget().update()

    def cursorDrag(self, relPos, absPos, savedState):
        self.cursorJump(absPos)

    def paintEvent(self, e):
        self.cursorClick.geometry.setRect(0, 0, 0, 0)

        ph = widgets.makePaintHelper(self, e)

        if not self.lc.lcdata:
            return

        gh = graphhelper.GraphHelper(self, ph)
        gh.setArea(QRectF(QPointF(0, 0), ph.size), 1, 1)

        gh.setXAxis(0, self.lc.lcdata[-1][7])

        gh.setYAxis(0, 30)

        self.x_axis = gh.x_axis
        self.cursorClick.geometry.setRect(gh.graph_area.left(), 0,
                                          ph.size.width(), gh.graph_area.bottom())

        lap = self.lc.dataView.ref_lap
        lt = (lap.end.time - lap.start.time) / 1000

        gh.paintXGrid()
        gh.paintYGrid()

        ph.painter.save()
        ph.painter.setClipRect(gh.graph_area)

        pen = QPen(QColor('cyan'))
        pen.setWidth(2)
        ph.painter.setPen(pen)

        x1 = gh.x_axis.calc(0)
        y1 = gh.y_axis.calc(0)
        for a in self.lc.lcdata:
            x2 = gh.x_axis.calc(a[7])
            y2 = gh.y_axis.calc(100 * ((lt + a[7]) / (lt * (1 - a[6])) - 1))
            # y2 = gh.y_axis.calc(a[6] * 100)
            ph.painter.drawLine(x1, y1, x2, y2)
            x1 = x2
            y1 = y2

        #pen = QPen(QColor('yellow'))
        #pen.setWidth(1)
        #ph.painter.setPen(pen)

        #x1 = None
        #y1 = None
        #for a in self.lc.lcdata:
        #    x2 = gh.x_axis.calc(a[7])
        #    y2 = gh.y_axis.calc(100 * (1 - (1 - a[5]) * lt / (lt + a[7])))
        #    if x1 is not None:
        #        ph.painter.drawLine(x1, y1, x2, y2)
        #    x1 = x2
        #    y1 = y2

        pen = QPen(QColor('green'))
        pen.setWidth(2)
        ph.painter.setPen(pen)
        x = gh.x_axis.calc(self.cursor_pos)
        ph.painter.drawLine(x, 0, x, gh.full_graph_area.bottom())

        ph.painter.restore()

        gh.paintGraphFrame()

        gh.paintXAxis()

        gh.paintYLabel('% more driving time per fuel used')
        gh.paintYAxis()

class SpeedChart(widgets.MouseHelperWidget):
    CURSOR_WIDTH = 10 # scale?

    def __init__(self, lc):
        super().__init__()
        self.lc = lc

    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)

        if not self.lc.lcdata:
            return

        gh = graphhelper.GraphHelper(self, ph)
        gh.setArea(QRectF(QPointF(0, 0), ph.size), 1, 1)

        data_range = self.lc.dataView.windowSize2Mode()
        zero_offset = self.lc.dataView.getTDValue(self.lc.dataView.zoom_window[0])
        gh.setXAxis(zero_offset, zero_offset + data_range)

        d = self.lc.dataView.get_channel_data(self.lc.dataView.ref_lap, 'GPS Speed')
        gh.setYAxis(d.min, d.max)

        lap = self.lc.dataView.ref_lap
        lift_zones = {}
        for adj in self.lc.lcdata:
            if adj[7] > self.lc.effChart.cursor_pos:
                break
            lift_zones[adj[0]] = adj

        ph.painter.save()
        ph.painter.setClipRect(gh.graph_area)
        for adj in lift_zones.values():
            ph.painter.fillRect(QRect(
                QPoint(gh.x_axis.calc(self.lc.dataView.getTDValue(adj[1])), 0),
                QPoint(gh.x_axis.calc(self.lc.dataView.getTDValue(adj[2])), gh.full_graph_area.bottom())),
                                QColor(32, 32, 32))
        ph.painter.restore()

        gh.paintXGrid()
        gh.paintYGrid()

        ph.painter.save()
        ph.painter.setClipRect(gh.graph_area)

        ph.painter.setPen(QPen(state.lap_colors[1]))
        for adj in lift_zones.values():
            r = adj[2].dist - adj[1].dist
            k = np.linspace(0, r, max(2, math.ceil(r))) # every meter
            v = _calc_finalv_from_dist(p2m=adj[9], d=k, v=adj[8])
            k += adj[1].dist
            if self.lc.dataView.mode_time:
                # base k relative to entire session
                k += lap.start.dist
                # convert k to time
                k = np.interp(k, d.distances, d.timecodes)
                # rebase back to 0 indexing
                k -= lap.start.time
            v = unitconv.convert(v, 'm/s', d.units)
            k = memoryview(gh.x_axis.calc(k))
            v = memoryview(gh.y_axis.calc(v))
            for x1, y1, x2, y2 in zip(k[:-1], v[:-1], k[1:], v[1:]):
                ph.painter.drawLine(x1, y1, x2, y2)

        pen = QPen(channels.colors[d.color])
        ph.painter.setPen(pen)

        xa = d.timecodes if self.lc.dataView.mode_time else d.distances
        lap_base = self.lc.dataView.getTDValue(lap.start) + self.lc.dataView.getTDValue(lap.offset)
        search = gh.x_axis.invert(max(ph.rect.left(), gh.graph_area.left()) - 0.5) + lap_base
        start_idx = max(0, bisect.bisect_left(xa, search) - 1)
        search = gh.x_axis.invert(max(ph.rect.right() + 0.5, gh.graph_area.left())) + lap_base
        end_idx = min(len(xa), bisect.bisect_right(xa, search) + 1)
        xa = memoryview(np.round(gh.x_axis.calc(np.subtract(xa[start_idx:end_idx], lap_base))).astype(int))
        dv = np.round(gh.y_axis.calc(np.asarray(d.values[start_idx:end_idx]))).astype(int)
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

        # paint lines that live across pixel columns
        for idx in memoryview(xa_uniqidx[1:] - 1):
            ph.painter.drawLine(xa[idx], dvd1[idx], xa[idx+1], dvd2[idx])
        # paint lines within pixel columns
        for x, y1, y2 in zip(xa_uniqval.data, umin.data, umax.data):
            if y1 != y2:
                ph.painter.drawLine(x, y1, x, y2)

        x1 = gh.x_axis.calc(0)
        y1 = gh.y_axis.calc(0)
        for a in self.lc.lcdata:
            x2 = gh.x_axis.calc(a[7])
            y2 = gh.y_axis.calc(a[6] * 100)
            ph.painter.drawLine(x1, y1, x2, y2)
            x1 = x2
            y1 = y2

        ph.painter.restore()

        x = gh.x_axis.calc(self.lc.dataView.getTDValue(self.lc.dataView.cursor_time))
        if x >= gh.graph_area.left() and x < ph.size.width():
            pen = QPen(QColor(255, 255, 0))
            pen.setStyle(Qt.SolidLine)
            ph.painter.setPen(pen)
            ph.painter.drawLine(x, 0, x, gh.full_graph_area.bottom())
            pen.setWidth(2)
            ph.painter.setPen(pen)
            val = gh.y_axis.calc(d.interp(self.lc.dataView.cursor2outTime(lap)))
            # adjust drawing position due to pen width=2
            ph.painter.drawLine(max(x - self.CURSOR_WIDTH + 1, gh.graph_area.left()), val,
                                x + self.CURSOR_WIDTH - 1, val)

        gh.paintGraphFrame()

        gh.paintXAxis(time_format=self.lc.dataView.mode_time)
        gh.paintYAxis()

class LiftCoast(widgets.MouseHelperWidget):
    def __init__(self, dataView, state=None):
        super().__init__()
        self.dataView = dataView

        self.effChart = EfficiencyChart(self)
        self.speedChart = SpeedChart(self)

        g_layout = QGridLayout()
        g_layout.addWidget(self.effChart, 0, 0)
        g_layout.addWidget(self.speedChart, 1, 0)
        self.setLayout(g_layout)

        dataView.values_change.connect(self.recompute)
        self.recompute()

    def save_state(self):
        return {'type': 'liftcoast',
                'base': self.parentWidget().save_state(),
                }

    def addChannel(self, ch):
        pass

    def channels(self):
        return {}

    def updateCursor(self, old_cursor):
        self.speedChart.update()

    def paintEvent(self, event):
        pass

    def recompute(self):
        self.lcdata = None

        if not self.dataView.ref_lap:
            print('No reflap')
            return

        speed = self.dataView.get_channel_data(self.dataView.ref_lap, 'GPS Speed')
        if not speed or len(speed.distances) == 0:
            print('No speed')
            return

        accel = self.dataView.get_channel_data(self.dataView.ref_lap, 'APS')
        if not accel or len(accel.distances) == 0:
            print('No accel')
            return
        accel_threshold = 90

        est_power = 220 * 1000 / 1.36 # W
        est_driveline_efficiency = 0.85 # % power from engine that makes it to wheels, used to calc driveline drag on decel
        est_mass = 2500 / 2.205 # kg

        brake = self.dataView.get_channel_data(self.dataView.ref_lap, 'BrakeSwitch')
        if not brake or len(brake.distances) == 0:
            print('No brake')
            return
        brake_threshold = 0.5

        fuel = self.dataView.get_channel_data(self.dataView.ref_lap, 'FuelUsed')
        if not fuel or len(fuel.distances) == 0:
            print('No fuel')
            return

        # fix units for speed
        speed = speed.change_units('m/s')

        # pick range
        start_idx = bisect.bisect_left(speed.timecodes, self.dataView.ref_lap.start.time)
        end_idx = min(bisect.bisect_right(speed.timecodes, self.dataView.ref_lap.end.time),
                      len(speed.timecodes) - 1)
        timecodes = speed.timecodes[start_idx:end_idx]
        distances = speed.distances[start_idx:end_idx]
        if len(timecodes) == 0 or len(distances) == 0:
            print('No lap data of interest')
            return

        # recast everything in terms of speed distances
        speed = speed.values[start_idx:end_idx]
        accel = accel.interp_many(distances, mode_time=False) > accel_threshold
        brake = brake.interp_many(distances, mode_time=False) > brake_threshold
        fuel = fuel.interp_many(distances, mode_time=False)
        fuel = np.subtract(fuel, fuel[0])
        fuel = fuel / fuel[-1]
        timecodes = np.subtract(timecodes, timecodes[0])
        distances = distances - distances[0]

        # find braking zones
        min_braking_distance = 20 # meters
        braking_zones = [] # indices into array
        has_accel = False
        for i in range(len(brake)):
            if brake[i]:
                if has_accel and not brake[i-1]:
                    braking_zones.append(i)
            elif has_accel and brake[i-1]:
                has_accel = False
                if i - braking_zones[-1] < min_braking_distance:
                    braking_zones.pop()
            elif accel[i]:
                has_accel = True

        # for each braking zone, analyze benefit of lifting
        alladv = []
        for zone in braking_zones:
            adv = []
            start_brake = zone
            for lift in range(zone - 1, 0, -1):
                if not accel[lift]:
                    if accel[lift + 1]:
                        break
                    continue
                t = (timecodes[zone] - timecodes[lift]) / 1000.
                d = distances[zone] - distances[lift]
                v = speed[lift]
                p2m = _calc_p2m(t=t, d=d, v=v) - est_power / (est_driveline_efficiency * est_mass)
                while True:
                    t = (timecodes[start_brake] - timecodes[lift]) / 1000.
                    d = distances[start_brake] - distances[lift]
                    newt = _calc_time(p2m=p2m, d=d, v=v)
                    finalv = _calc_finalv_from_time(p2m=p2m, t=newt, v=v)
                    if finalv >= speed[start_brake + 1]:
                        break
                    if not brake[start_brake + 1]:
                        # if we let off the brake, then these are impossible to use lift zones
                        newt = math.nan
                        break
                    start_brake += 1
                if math.isfinite(newt):
                    fsave = fuel[start_brake] - fuel[lift]
                    tcost = newt - t
                    new_row = [zone,
                               state.TimeDistRef(timecodes[lift], distances[lift]),
                               state.TimeDistRef(timecodes[start_brake], distances[start_brake]),
                               fsave,
                               tcost,
                               fsave/tcost,
                               0, # filled later as total fsave
                               0, # filled later as total tcost
                               v,
                               p2m]
                    # Discard any intermediate steps that have a worse
                    # incremental efficiency than this one.  It just
                    # makes the search space too difficult to deal
                    # with.
                    while len(adv) >= 2 and ((new_row[3]-adv[-2][3]) / (new_row[4]-adv[-2][4]) >
                                             (adv[-1][3]-adv[-2][3]) / (adv[-1][4]-adv[-2][4])):
                        adv.pop()
                    adv.append(new_row)
            # Rewrite steps as incremental steps so they can be sorted later with other results
            for i in range(len(adv) - 1, 0, -1):
                adv[i][3] -= adv[i-1][3]
                adv[i][4] -= adv[i-1][4]
                adv[i][5] = adv[i][3] / adv[i][4]
            alladv.extend(adv)

        alladv.sort(key=lambda a: a[5], reverse=True)

        cf = 0
        ct = 0
        for a in alladv:
            cf += a[3]
            ct += a[4]
            a[6] = cf
            a[7] = ct

        self.lcdata = alladv

        self.update()
