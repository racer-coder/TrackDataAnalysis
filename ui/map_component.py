
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""Map component: GPS track outline with cursor position dot."""

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, QPoint, Qt
from PySide6.QtWidgets import QWidget

from . import state, widgets


class MapComponent(QWidget):
    """Component showing GPS track outline on black background with cursor dot."""

    def __init__(self, data_view, st=None):
        super().__init__()
        self.data_view = data_view
        self.channel_list = list(st['channels']) if st and 'channels' in st else []
        self.setMinimumSize(100, 100)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        data_view.values_change.connect(self.update)
        data_view.cursor_change.connect(lambda old: self.update())

    def save_state(self):
        return {
            'type': 'map_component',
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
        self.update()

    def paintEvent(self, e):
        ph = widgets.makePaintHelper(self, e)
        ph.painter.fillRect(ph.rect, QtGui.QColor(12, 12, 12))

        dv = self.data_view
        track = dv.track
        if not track or not track.coords:
            ph.painter.setPen(QtGui.QColor(100, 100, 100))
            ph.painter.drawText(ph.rect, Qt.AlignmentFlag.AlignCenter, "No track data")
            return

        coords = np.array(track.coords)
        lats = coords[:, 0]
        lons = coords[:, 1]

        if len(lats) < 2:
            return

        lat_min, lat_max = np.min(lats), np.max(lats)
        lon_min, lon_max = np.min(lons), np.max(lons)
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        if lat_range == 0 or lon_range == 0:
            return

        # Compute scale preserving aspect ratio (approximate cos correction)
        mid_lat = np.radians((lat_min + lat_max) / 2)
        cos_lat = np.cos(mid_lat)

        margin = 0.08
        usable_w = ph.size.width() * (1 - 2 * margin)
        usable_h = ph.size.height() * (1 - 2 * margin)

        scale = min(usable_h / lat_range, usable_w / (lon_range * cos_lat))

        # Map coords to pixels (lat inverted for screen Y)
        cx = ph.size.width() / 2
        cy = ph.size.height() / 2
        mid_lon = (lon_min + lon_max) / 2
        mid_lat_val = (lat_min + lat_max) / 2

        def to_screen(lat, lon):
            sx = cx + (lon - mid_lon) * cos_lat * scale
            sy = cy - (lat - mid_lat_val) * scale
            return sx, sy

        # Draw track outline
        pen = QtGui.QPen(QtGui.QColor(100, 100, 100))
        pen.setWidth(max(1, int(1.5 * ph.scale)))
        ph.painter.setPen(pen)
        ph.painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # Speed color coding if a speed channel is available
        speed_values = None
        speed_ch = None
        for ch in self.channel_list:
            if 'speed' in ch.lower():
                speed_ch = ch
                break

        if speed_ch and dv.ref_lap:
            try:
                cd = dv.ref_lap.get_channel_data(speed_ch, None, dv.maths)
                if cd and hasattr(cd, 'values') and len(cd.values) > 0:
                    # Interpolate speed onto track distances
                    dists = coords[:, 3]
                    speed_values = np.interp(dists, cd.distances, cd.values)
            except Exception:
                speed_values = None

        if speed_values is not None and len(speed_values) == len(lats):
            smin, smax = np.min(speed_values), np.max(speed_values)
            srange = smax - smin if smax > smin else 1.0
            for i in range(1, len(lats)):
                t = (speed_values[i] - smin) / srange
                r = int(255 * t)
                b = int(255 * (1 - t))
                pen.setColor(QtGui.QColor(r, 40, b))
                ph.painter.setPen(pen)
                x0, y0 = to_screen(lats[i - 1], lons[i - 1])
                x1, y1 = to_screen(lats[i], lons[i])
                ph.painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        else:
            pen.setColor(QtGui.QColor(180, 180, 180))
            ph.painter.setPen(pen)
            for i in range(1, len(lats)):
                x0, y0 = to_screen(lats[i - 1], lons[i - 1])
                x1, y1 = to_screen(lats[i], lons[i])
                ph.painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # Draw cursor dot for each active lap
        dot_size = max(3, int(5 * ph.scale))
        for lap, color, _idx in dv.get_laps():
            try:
                key_channels = lap.log.log.get_key_channel_map()
                gps_lat = lap.log.log.get_channel_data(key_channels[1], unit='deg')
                gps_lon = lap.log.log.get_channel_data(key_channels[2], unit='deg')
                out_time = dv.cursor2outTime(lap)
                clat = gps_lat.interp(out_time)
                clon = gps_lon.interp(out_time)
                sx, sy = to_screen(clat, clon)
                ph.painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
                ph.painter.setBrush(QtGui.QBrush(color))
                ph.painter.drawEllipse(QPointF(sx, sy), dot_size, dot_size)
            except Exception:
                pass

        # Label
        font = ph.painter.font()
        font.setPixelSize(int(11 * ph.scale))
        ph.painter.setFont(font)
        ph.painter.setPen(QtGui.QColor(120, 120, 120))
        ph.painter.drawText(
            int(5 * ph.scale), int(5 * ph.scale),
            int(ph.size.width()), int(20 * ph.scale),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            track.name if track.name else "Track Map")
