
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
import concurrent.futures
import urllib.request

import numpy as np

from PySide2 import QtGui
from PySide2.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide2.QtWidgets import (
    QAction,
    QWidget,
)

import data.gps as gps
from . import widgets

worker = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='maploader')

tile_cache = {}

def closure(func, *args):
    return lambda *args2: func(*args, *args2)

def maptiler_get_map(x, y, zoom, api_key):
    # lat/long are in degrees
    with urllib.request.urlopen(
            'https://api.maptiler.com/tiles/satellite-v2/%d/%d/%d.jpg?key=%s'
            % (zoom, x, y, api_key)) as f:
        return f.read()

class MapBaseWidget(QWidget):
    async_update = Signal(object, object) # key, Future

    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view
        self.async_update.connect(self.handle_update)

        self.satellite = QAction('Show satellite', self)
        self.satellite.setCheckable(True)
        self.satellite.setChecked(True)
        self.satellite.toggled.connect(self.update)
        self.addAction(self.satellite)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        self.setMinimumSize(100, 100) # arbitrary, prevents QT rendering bug with docks

    def handle_update(self, k, fut):
        pm = QtGui.QPixmap()
        if not pm.loadFromData(fut.result()):
            print('failed to load data for tile', k)
        tile_cache[k] = pm
        self.update()

    def sizeHint(self):
        return QSize(400, 400)

    def paint_satellite(self, ph, lla_min, lla_max):
        # first, a center
        center = (np.array(lla_min) + np.array(lla_max)) / 2

        # second, compute scale for lat/long
        delta = 1/1024. # approx 300 ft or so
        x, y, z = gps.lla2ecef(np.array([center[0], center[0] + delta, center[0]]),
                               np.array([center[1], center[1],         center[1] + delta]),
                               center[2])
        lat_scale = np.linalg.norm(np.array([x[0] - x[1], y[0] - y[1], z[0] - z[1]]))
        long_scale = np.linalg.norm(np.array([x[0] - x[2], y[0] - y[2], z[0] - z[2]]))

        # third, calc bounds for gps, and map to fit in ph.size
        lat_min = lla_min[0]
        lat_range = lla_max[0] - lat_min
        long_min = lla_min[1]
        long_range = lla_max[1] - long_min

        # What's the point of having GPS data if it doesn't move?
        if lat_range == 0 or long_range == 0: return False

        scale = 0.9 * min(ph.size.height() / (lat_range * lat_scale),
                          ph.size.width() / (long_range * long_scale))
        lat_scale *= -scale # invert display
        long_scale *= scale

        lat_base = lat_min + lat_range / 2 - ph.size.height() / lat_scale / 2
        long_base = long_min + long_range / 2 - ph.size.width() / long_scale / 2

        self.lat_scale = lat_scale
        self.long_scale = long_scale
        self.lat_base = lat_base
        self.long_base = long_base

        # map to corresponding tiles
        if self.satellite.isChecked() and self.data_view.maps_key:
            tile_corner1 = np.array(gps.llz2web(lat_base, long_base))
            tile_corner2 = np.array(gps.llz2web(ph.size.height()/lat_scale+lat_base,
                                                ph.size.width()/long_scale+long_base))
            zoom = int(np.ceil(np.max(
                np.log(np.array([ph.size.width(), ph.size.height()]) * (1/512)
                       / (tile_corner2 - tile_corner1))) / np.log(2)))
            zoom = min(zoom, 20) # maptiler goes to 22, but it isn't very helpful
            tile_corner1 = np.floor(tile_corner1 * 2 ** zoom).astype(np.int32)
            tile_corner2 = np.ceil(tile_corner2 * 2 ** zoom).astype(np.int32)
            for x in range(tile_corner1[0], tile_corner2[0]):
                for y in range(tile_corner1[1], tile_corner2[1]):
                    k = (x, y, zoom)
                    if k not in tile_cache:
                        f = worker.submit(maptiler_get_map, x, y, zoom, self.data_view.maps_key[1])
                        tile_cache[k] = f
                        f.add_done_callback(closure(self.async_update.emit, k))
                    # find lat/long of boundaries of tile
                    la, lo = gps.web2ll([x, x+1], [y, y+1], zoom)
                    # map lat/long to screen
                    sy = (la - lat_base) * lat_scale
                    sx = (lo - long_base) * long_scale
                    tgt_rect = QRectF(QPointF(sx[0], sy[0]), QPointF(sx[1], sy[1]))
                    pm = tile_cache[k]
                    if isinstance(pm, QtGui.QPixmap): # might still be a future
                        ph.painter.drawPixmap(tgt_rect, pm, pm.rect())

        # map attribution
        font = QtGui.QFont('Tahoma')
        font.setPixelSize(widgets.deviceScale(self, 13))
        ph.painter.setFont(font)
        pen = QtGui.QPen(QtGui.QColor(224, 224, 224))
        pen.setStyle(Qt.SolidLine)
        pen.setWidth(1)
        ph.painter.setPen(pen)
        ph.painter.drawText(0, 0, ph.size.width(), ph.size.height(),
                            Qt.AlignRight | Qt.AlignBottom,
                            '\u00a9 MapTiler  \u00a9 OpenStreetMap contributors')

        return True


class MapWidget(MapBaseWidget):
    def paintEvent(self, event):
        ph = widgets.makePaintHelper(self, event)

        ph.painter.fillRect(QRectF(QPoint(0, 0), ph.size),
                            QtGui.QColor(0, 0, 0))

        dv = self.data_view
        lap = dv.ref_lap
        if not lap: return

        key_channels = lap.log.log.get_key_channel_map()
        gps_lat = lap.log.log.get_channel_data(key_channels[1], unit='deg')
        gps_long = lap.log.log.get_channel_data(key_channels[2], unit='deg')
        gps_alt = lap.log.log.get_channel_data(key_channels[3], unit='m')

        if not len(gps_lat.values) or not len(gps_long.values):
            return

        zoom_start = bisect.bisect_left(
            gps_lat.timecodes, lap.start.time + lap.offset.time + dv.zoom_window[0].time)
        zoom_end = bisect.bisect_left(
            gps_lat.timecodes, lap.end.time + lap.offset.time + dv.zoom_window[1].time)
        zoom_lat = gps_lat.values[max(0, zoom_start-1) : zoom_end]
        zoom_long = gps_long.values[max(0, zoom_start-1) : zoom_end]
        zoom_alt = gps_alt.values[max(0, zoom_start-1) : zoom_end] if len(gps_alt.values) else np.array(0.)

        if not len(zoom_lat): # zoom window too small or ??
            return

        if not self.paint_satellite(ph,
                                    (np.min(zoom_lat), np.min(zoom_long), np.min(zoom_alt)),
                                    (np.max(zoom_lat), np.max(zoom_long), np.max(zoom_alt))):
            return

        # get gps data for the lap
        start_idx = bisect.bisect_left(gps_lat.timecodes, lap.start.time)
        end_idx = bisect.bisect_right(gps_lat.timecodes, lap.end.time)
        # we assume timecodes in long match lat
        lap_y = memoryview((gps_lat.values[start_idx:end_idx] - self.lat_base) * self.lat_scale)
        lap_x = memoryview((gps_long.values[start_idx:end_idx] - self.long_base) * self.long_scale)

        zoom_y = memoryview((zoom_lat - self.lat_base) * self.lat_scale)
        zoom_x = memoryview((zoom_long - self.long_base) * self.long_scale)

        # Background track outline
        pen = QtGui.QPen(QtGui.QColor(192, 192, 192))
        pen.setWidth(widgets.deviceScale(self, 1.25))
        ph.painter.setPen(pen)
        if start_idx < zoom_start:
            for i in range(min(len(lap_x), zoom_start - start_idx)):
                ph.painter.drawLine(lap_x[i-1], lap_y[i-1], lap_x[i], lap_y[i])
        if zoom_end < end_idx:
            for i in range(max(0, zoom_end - start_idx), len(lap_x)):
                ph.painter.drawLine(lap_x[i-1], lap_y[i-1], lap_x[i], lap_y[i])

        # Zoom window track outline
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setWidth(widgets.deviceScale(self, 2))
        ph.painter.setPen(pen)
        for i in range(1, len(zoom_x)):
            ph.painter.drawLine(zoom_x[i-1], zoom_y[i-1], zoom_x[i], zoom_y[i])

        # track position
        msize = widgets.deviceScale(self, 4)
        for lap, color, _idx in dv.get_laps()[::-1]:
            pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
            pen.setStyle(Qt.SolidLine)
            ph.painter.setPen(pen)
            ph.painter.setBrush(QtGui.QBrush(color))
            key_channels = lap.log.log.get_key_channel_map()
            gps_lat = lap.log.log.get_channel_data(key_channels[1], unit='deg')
            gps_long = lap.log.log.get_channel_data(key_channels[2], unit='deg')
            lap_out_time = dv.cursor2outTime(lap)
            ph.painter.drawEllipse(
                QPointF((gps_long.interp(lap_out_time) - self.long_base) * self.long_scale,
                        (gps_lat.interp(lap_out_time) - self.lat_base) * self.lat_scale),
                msize, msize)
