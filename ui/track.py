
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import configparser
import copy
import dataclasses
import json
import os
import time

from PySide2.QtCore import QAbstractItemModel, QModelIndex, QPoint, QRectF, QStandardPaths, Qt, Signal
from PySide2.QtGui import QColor, QPen
from PySide2.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

import dacite
import numpy as np

from data import gps
from .map import MapBaseWidget
from . import state
from . import widgets

def track_dir():
    track_dir = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation) + '/tracks'
    os.makedirs(track_dir, exist_ok=True)
    return track_dir

strtable = ''.join([chr(48 + j) if j < 10 else chr(97 - 10 + j) for j in range(36)])
def strcode(i):
    i = min(int(i), 1295)
    return strtable[i // 36] + strtable[i % 36]

def strdeg(d):
    return strcode((d + 180) * 1296 / 360)

def coord_fname(lat, lon):
    return strdeg(lat) + strdeg(lon)

def find_many_crossing(xyz, latlon, adder=0):
    # divide and conquer to reduce search space for each subsequent search
    if len(xyz) * len(latlon) < 8192:
        dist = gps.find_crossing_idx(xyz, latlon)
        dist[:,0] += adder
        return dist
    else:
        idx = len(latlon) // 2
        crossing = gps.find_crossing_idx(xyz, latlon[idx])
        cidx = int(crossing[0])
        return np.row_stack([find_many_crossing(xyz[:cidx+2], latlon[:idx], adder),
                             [[crossing[0] + adder, crossing[1]]],
                             find_many_crossing(xyz[cidx:], latlon[idx+1:], adder + crossing[0])])

def load_track(lat, lon):
    # given a bunch of in-order lat,lon coordinates, try to find a stored track that matches
    possible = {coord_fname(la, lo)
                for la in (np.min(lat), np.max(lat))
                for lo in (np.min(lon), np.max(lon))}
    for entry in os.scandir(track_dir()):
        if entry.name[:4].lower() in possible and entry.name[-4:].lower() == '.trk':
            t1 = time.perf_counter()
            with open(entry.path, 'rt', encoding='utf-8') as f:
                track = dacite.from_dict(data_class=state.Track,
                                         data=json.load(f),
                                         config=dacite.Config(cast=[tuple]))
            t2 = time.perf_counter()
            na = np.array(track.coords)
            xyzd = np.column_stack(list(gps.lla2ecef(na[:,0], na[:,1], 0.)) + [np.arange(len(na))])
            t3 = time.perf_counter()
            dist = find_many_crossing(xyzd, np.array(list(zip(lat, lon))))
            t4 = time.perf_counter()
            print('load track', t2-t1, t3-t2, t4-t3)
            if np.max(dist[:,1]) < 15 and np.sum(dist[1:,0] > dist[:-1,0]) / len(dist) > 0.9:
                return track

def save_track(track):
    dirbase = track_dir()
    if not track.file_name:
        coords = np.array(track.coords)
        fbase = coord_fname(np.mean(coords[:,0]), np.mean(coords[:,1]))
        for idx in range(36*36):
            fname = fbase + strcode(idx) + '.trk'
            if not os.path.isfile(os.path.join(dirbase, fname)):
                track.file_name = fname
                break
    with state.atomic_write(os.path.join(dirbase, track.file_name)) as f:
        json.dump(dataclasses.asdict(track,
                                     dict_factory=lambda x: {a:b for a, b in x if a[0] != '_'}),
                  f, indent=4)

def select_track(logref):
    t1 = time.perf_counter()

    if len(logref.laps) < 3:
        return None
    lap = logref.best_lap

    # grab GPS coordinates, select roughly every 10 meters
    key_channels = logref.log.get_key_channel_map()
    gps_speed = logref.log.get_channel_data(key_channels[0], unit='m/s')
    gps_lat = logref.log.get_channel_data(key_channels[1], unit='deg')
    gps_long = logref.log.get_channel_data(key_channels[2], unit='deg')
    gps_alt = logref.log.get_channel_data(key_channels[3], unit='m')

    # NOTE: GPS coordintes from synchronized time loggers (like MoTeC)
    # which log every X Hz regardless of when data shows up may have
    # repeats or skipped GPS entries.  Be wary relying on GPS data on
    # those loggers without heavy filtering!

    if not len(gps_lat.values) or not len(gps_long.values):
        return None

    marker_sample = 1 # meters
    map_sample = 10 # units of marker_sample
    dist = np.linspace(lap.start.dist, lap.end.dist,
                       num=round((lap.end.dist - lap.start.dist) / map_sample) * map_sample + 1)
    speed = gps_speed.interp_many(dist, mode_time=False)
    lat = gps_lat.interp_many(dist, mode_time=False)
    lon = gps_long.interp_many(dist, mode_time=False)
    if len(gps_alt.values):
        alt = gps_alt.interp_many(dist, mode_time=False)
    else:
        alt = np.zeros((len(dist),))
    dist -= dist[0]

    # fudge the data so we start and finish at the exact same point (if they're close)
    if np.linalg.norm(np.array(gps.lla2ecef(lat[0], lon[0], alt[0])) -
                      np.array(gps.lla2ecef(lat[-1], lon[-1], alt[-1]))) < 20:
        lat += (lat[0] - lat[-1]) * np.linspace(0, 1, num=len(lat))
        lon += (lon[0] - lon[-1]) * np.linspace(0, 1, num=len(lon))
        alt += (alt[0] - alt[-1]) * np.linspace(0, 1, num=len(alt))

    track = load_track(lat[::map_sample], lon[::map_sample])
    if track:
        return track

    # build track
    try:
        name = logref.log.get_metadata()['Venue']
    except KeyError:
        name = 'Unknown'
    latrad = lat * (np.pi / 180)
    lonrad = lon * (np.pi /180)
    hsample = 4 # units of marker_sample.  Have > 1 to work around MoTeC GPS sampling issues
    orig_index = hsample // 2
    heading = np.arctan2(np.cos(latrad[hsample:]) * np.sin(lonrad[hsample:] - lonrad[:-hsample]),
                         np.cos(latrad[:-hsample]) * np.sin(latrad[hsample:])
                         - np.sin(latrad[:-hsample]) * np.cos(latrad[hsample:])
                         * np.cos(lonrad[hsample:] - lonrad[:-hsample])) * (180 / np.pi)

    tsample = 10 # units of marker_sample
    orig_index += tsample // 2
    turn = heading[tsample:] - heading[:-tsample]
    turn = (turn + 540) % 360 - 180 # normalize to -180 to 180
    turn *= speed[orig_index:-orig_index]

    # sliding window calculation
    window = 4
    orig_index -= 1
    turn = np.concatenate([[0.], np.cumsum(turn)])
    orig_index += window // 2
    turn = (turn[window:] - turn[:-window]) * (1./window)

    t2 = time.perf_counter()

    type_map = ['Straight', 'Right', 'Left']
    typ = (turn > 150) + 2 * (turn < -150)

    # find points where typ changes
    typidx = list(memoryview(np.concatenate([[0],
                                             np.nonzero(typ[1:] != typ[:-1])[0] + 1,
                                             [len(typ)]])))
    # check for short straights and join correctly (same turn->same turn - extend, different turn - find 0 crossing)
    min_straight = 25 # meters
    for i in range(len(typidx), 1, -1):
        # need the out of bounds check here because this loop can
        # delete more than 1 element, causing the out of bounds check
        # to fail in later iterations
        if i + 2 >= len(typidx):
            continue

        # look at the straight from [i] to [i+1], if short, merge into turn starting at [i-1] and [i+1]
        if typ[typidx[i]] == 0 and typidx[i+1] - typidx[i] < min_straight:
            if typ[typidx[i-1]] == typ[typidx[i+1]]:
                # same direction turn, just have one big turn
                typidx.pop(i)
                typidx.pop(i)
            else:
                # different direction turns, find the best zero crossing
                subset = turn[typidx[i]:typidx[i+1]]
                zero_cross = np.nonzero((subset[1:] > 0) != (subset[:-1] > 0))[0]
                if len(zero_cross):
                    newidx = int(zero_cross[0]) + typidx[i] + 1
                    typ[newidx - 1] = typ[typidx[i] - 1]
                    typ[newidx] = typ[typidx[i+1]]
                    typidx[i+1] = newidx
                    typidx.pop(i)

    # check for short turns and drop?
    min_turn = 10 # meters
    for i in range(len(typidx), 1, -1):
        if i + 2 < len(typidx) and typ[typidx[i]] != 0 and typidx[i+1] - typidx[i] < min_turn:
            typidx.pop(i)
            typidx.pop(i)

    sectors = [state.Marker('Hi', lat[i + orig_index], lon[i + orig_index], type_map[typ[i - 1]])
               for i in typidx[1:-1]]

    num = 0
    for s in sectors:
        if s.typ != 'Straight':
            num += 1
            s.name = 'Turn %d' % num
    last = num
    num = 1
    for s in sectors:
        if s.typ == 'Straight':
            s.name = 'Str %d-%d' % (last, num if sectors[-1] != s else 1)
        else:
            last = num
            num += 1

    t4 = time.perf_counter()

    track = state.Track(
        name = name,
        file_name = None,
        coords = [(float(la), float(lo), float(al), float(di))
                  for la, lo, al, di in zip(*map(lambda x: x[::map_sample],
                                                 (lat, lon, alt, dist)))],
        sector_sets = {'Default': state.Sectors('Default', sectors)})


    t5 = time.perf_counter()
    print('select track: %.3f %.3f %.3f' % (t2-t1, t4-t2, t5-t4))
    return track


@dataclasses.dataclass
class IndexDetails:
    name: str
    obj: object
    ordered_src: list[object]
    key: object # either str or int, key into src_obj
    src_obj: object # list[object] or dict[str, object]
    parent: object

class TrackTreeModel(QAbstractItemModel):
    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view

    def child(self, index):
        if index.isValid():
            obj = index.internalPointer()
            if isinstance(obj, state.Track):
                src = sorted(obj.sector_sets.keys())
                k = src[index.row()]
                return IndexDetails(k, obj.sector_sets[k], src, k, obj.sector_sets, obj)
            if isinstance(obj, state.Sectors):
                src = obj.markers
                k = index.row()
                return IndexDetails(src[k].name, src[k], src, k, src, obj)
        return IndexDetails(None, self.data_view.track, [], None, {}, None)

    def data(self, index, role):
        child = self.child(index)
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return child.name
            if index.column() == 1 and isinstance(child.obj, state.Marker):
                return child.obj.typ

    HEADINGS = ('Name', 'Type')
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADINGS[section]

    def index(self, row, col, parent):
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        child = self.child(parent)
        return self.createIndex(row, col, child.obj)

    def parent(self, index):
        if not index.isValid(): return index
        parent = index.internalPointer()
        if isinstance(parent, state.Track):
            return QModelIndex()
        if isinstance(parent, state.Sectors):
            k = [v for _, v in sorted(self.data_view.track.sector_sets.items())]
            return self.createIndex(k.index(parent), 0, self.data_view.track)
        return QModelIndex()

    def rowCount(self, parent):
        parent = self.child(parent).obj
        if isinstance(parent, state.Track):
            return len(parent.sector_sets)
        if isinstance(parent, state.Sectors):
            return len(parent.markers)
        return 0

    def columnCount(self, parent):
        return 2

class TrackSectorsMapWidget(MapBaseWidget):
    marker_select = Signal(int)

    def __init__(self, data_view):
        super().__init__(data_view)
        self.setMouseTracking(True)
        self.current_marker = None

    def mouseMoveEvent(self, e):
        if self.current_marker:
            coords = np.array(self.data_view.track.coords)
            eyx = np.array([[e.localPos().y(), e.localPos().x()]])
            yx = ((coords[:,:2] - np.array([[self.lat_base, self.long_base]]))
                  * np.array([[self.lat_scale, self.long_scale]]))
            v = yx[1:] - yx[:-1]
            yx = yx[:-1]

            # yx + t*v - m -> min dist
            # sum(sq(yx + t*v - m)) -> min
            # sum(t*t*v*v + 2*t*v*(yx-m) + (yx-m)**2) -> min
            # sum(2*t*v*v + 2*v*(yx-m)) = 0
            # t = sum(v*(m-yx)) / sum(v*v)
            t = np.clip(np.sum((eyx - yx) * v, axis=1) / np.sum(v*v, axis=1), 0, 1)
            t = t.reshape((len(t), 1))
            idx = np.argmin(np.sum(np.square(yx + t * v - eyx), axis=1))

            lat, lon = (coords[idx] + t[idx] * (coords[idx + 1] - coords[idx]))[:2]
            dist = gps.find_crossing_dist(
                np.column_stack(list(gps.lla2ecef(coords[:, 0], coords[:, 1], 0.))
                                + [coords[:, 3]]),
                (lat, lon))
            # make sure dist is within reasonable bounds
            if self.current_marker_idx and dist < self.sectors.markers[self.current_marker_idx - 1]._dist + 10:
                return
            try:
                if dist > self.sectors.markers[self.current_marker_idx + 1]._dist - 10:
                    return
            except IndexError:
                pass

            self.current_marker.lat = lat
            self.current_marker.lon = lon
            self.current_marker._dist = dist
            self.update()
            self.data_view.values_change.emit()

    def mousePressEvent(self, e):
        if self.sectors:
            markers = self.sectors.markers
            self.current_marker_idx = min(
                range(len(markers)),
                key=lambda idx: np.linalg.norm(
                    [e.localPos().x() - (markers[idx].lon - self.long_base) * self.long_scale,
                     e.localPos().y() - (markers[idx].lat - self.lat_base) * self.lat_scale]))
            self.current_marker = markers[self.current_marker_idx]
            self.mouseMoveEvent(e)
            self.marker_select.emit(self.current_marker_idx)

    def mouseReleaseEvent(self, e):
        if self.current_marker:
            self.mouseMoveEvent(e)
            self.current_marker = None
            self.update()

    def crossing_vector(self, coords, xyz, lat, lon, extent):
        idx = int(gps.find_crossing_idx(xyz, (lat, lon))[0])
        vect = coords[idx + 1] - coords[idx]
        vect = np.array([vect[0] * self.lat_scale,
                         -vect[1] * self.long_scale])
        vect /= np.linalg.norm(vect)
        pt = np.array([(lon - self.long_base) * self.long_scale,
                       (lat - self.lat_base) * self.lat_scale])
        p1 = pt + vect * extent
        p2 = pt - vect * extent
        return (p1[0], p1[1], p2[0], p2[1])

    def paintEvent(self, event):
        ph = widgets.makePaintHelper(self, event)

        ph.painter.fillRect(QRectF(QPoint(0, 0), ph.size),
                            QColor(0, 0, 0))

        coords = np.array(self.data_view.track.coords)[:,:3]

        self.paint_satellite(ph,
                             np.min(coords, axis=0),
                             np.max(coords, axis=0))

        if self.sector_idx is not None:
            pen = QPen(QColor(192, 192, 192))
            pen.setWidth(widgets.deviceScale(self, 1.25))
            ph.painter.setPen(pen)
            y = ((coords[:,0] - self.lat_base) * self.lat_scale).data
            x = ((coords[:,1] - self.long_base) * self.long_scale).data
            for i in range(1, len(y)):
                ph.painter.drawLine(x[i-1], y[i-1], x[i], y[i])

        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(widgets.deviceScale(self, 2))
        ph.painter.setPen(pen)
        draw_coords = coords
        xyz = np.column_stack(list(gps.lla2ecef(coords[:,0], coords[:,1], 0)))
        if self.sector_idx is not None:
            m1 = self.sectors.markers[self.sector_idx - 1]
            m2 = self.sectors.markers[self.sector_idx]
            c1 = int(1 + gps.find_crossing_idx(xyz, (m1.lat, m1.lon))[0])
            c2 = int(np.ceil(gps.find_crossing_idx(xyz, (m2.lat, m2.lon))[0]))
            draw_coords = [np.array([m1.lat, m1.lon], dtype=np.float64)]
            if c1 <= c2:
                draw_coords.append(coords[c1:c2,:2])
            else:
                draw_coords.append(coords[c1:,:2])
                draw_coords.append(coords[:c2,:2])
            draw_coords = np.row_stack(draw_coords +
                                       [np.array([m2.lat, m2.lon], dtype=np.float64)])
        y = ((draw_coords[:,0] - self.lat_base) * self.lat_scale).data
        x = ((draw_coords[:,1] - self.long_base) * self.long_scale).data
        for i in range(1, len(y)):
            ph.painter.drawLine(x[i-1], y[i-1], x[i], y[i])

        pen = QPen(QColor(0, 0, 160))
        pen.setWidth(widgets.deviceScale(self, 3))
        ph.painter.setPen(pen)
        ph.painter.drawLine(*self.crossing_vector(coords, xyz, coords[0,0], coords[0,1],
                                                  widgets.deviceScale(self, 15)))
        if self.sectors:
            for m in self.sectors.markers:
                pen = QPen(QColor(255, 255, 255))
                pen.setWidth(widgets.deviceScale(self, 1.25))
                if m == self.current_marker:
                    pen = QPen(QColor(0, 0, 160))
                    pen.setWidth(widgets.deviceScale(self, 2))
                elif self.sector_idx is not None and m == self.sectors.markers[self.sector_idx]:
                    pen.setWidth(widgets.deviceScale(self, 2))
                ph.painter.setPen(pen)
                ph.painter.drawLine(*self.crossing_vector(coords, xyz, m.lat, m.lon,
                                                          widgets.deviceScale(self, 15)))

class TrackDialog(QDialog):
    def __init__(self, data_view):
        super().__init__()

        self.left = QFormLayout()
        name_edit = QLineEdit(data_view.track.name)
        self.left.addRow('Name', name_edit)

        self.tree_view = QTreeView()
        self.tree_model = TrackTreeModel(data_view)
        self.tree_view.setModel(self.tree_model)
        self.tree_view.expandAll()
        self.tree_view.selectionModel().currentChanged.connect(self.sector_click)
        self.left.addRow(self.tree_view)

        buttons = QHBoxLayout()
        self.split_button = QPushButton('Split')
        self.split_button.clicked.connect(self.split_sector)
        self.split_button.setEnabled(False)
        buttons.addWidget(self.split_button)
        self.remove_button = QPushButton('Remove')
        self.remove_button.clicked.connect(self.remove_marker)
        self.remove_button.setEnabled(False)
        buttons.addWidget(self.remove_button)
        self.left.addRow(buttons)

        # XXX . button to add sector set?
        # XXX . button to regenerate default markers?

        self.section_name = QLineEdit('')
        self.section_name.setEnabled(False)
        self.section_name.textEdited.connect(self.name_edit)
        self.left.addRow('Section Name', self.section_name)
        self.left.labelForField(self.section_name).setEnabled(False)
        self.section_type = QComboBox()
        self.section_type.setEditable(True)
        self.section_type.setEnabled(False)
        self.section_type.lineEdit().textEdited.connect(self.type_edit)
        self.section_type.activated.connect(self.type_edit)
        self.left.addRow('Section Type', self.section_type)
        self.left.labelForField(self.section_type).setEnabled(False)

        dlgbutton = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dlgbutton.accepted.connect(self.accept)
        dlgbutton.rejected.connect(self.reject)
        self.left.addRow(dlgbutton) # slight violation of UI standards by forcing this on left half

        split = QSplitter()
        left_w = QWidget()
        left_w.setLayout(self.left)
        split.addWidget(left_w)

        self.map_widget = TrackSectorsMapWidget(data_view)
        self.map_widget.sectors = None
        self.map_widget.sector_idx = None
        self.map_widget.marker_select.connect(self.marker_select)
        split.addWidget(self.map_widget)
        lay = QVBoxLayout()
        lay.addWidget(split)

        self.setLayout(lay)
        try:
            self.restoreGeometry(
                bytes.fromhex(data_view.config.get('main', 'trackeditor_geometry')))
            self.tree_view.header().restoreState(
                bytes.fromhex(data_view.config.get('main', 'trackeditor_header')))
        except configparser.NoOptionError:
            pass

    def hideEvent(self, ev):
        self.tree_model.data_view.config['main']['trackeditor_geometry'] = bytes(
            self.saveGeometry()).hex()
        self.tree_model.data_view.config['main']['trackeditor_header'] = bytes(
            self.tree_view.header().saveState()).hex()
        super().hideEvent(ev)

    def split_sector(self):
        dist = (self.map_widget.sectors.markers[self.map_widget.sector_idx]._dist +
                (self.map_widget.sectors.markers[self.map_widget.sector_idx-1]._dist
                 if self.map_widget.sector_idx else 0)) / 2
        coords = np.array(self.tree_model.data_view.track.coords)
        m = state.Marker('New Sector',
                         np.interp(dist, coords[:, 3], coords[:, 0]),
                         np.interp(dist, coords[:, 3], coords[:, 1]),
                         '',
                         _dist = dist)
        self.map_widget.sectors.markers.insert(self.map_widget.sector_idx, m)
        self.tree_model.layoutChanged.emit()
        self.update_selection()
        self.tree_model.data_view.values_change.emit()
        self.map_widget.update()

    def remove_marker(self):
        self.map_widget.sectors.markers.pop(self.map_widget.sector_idx)
        self.tree_model.layoutChanged.emit()
        self.update_selection()
        self.tree_model.data_view.values_change.emit()
        self.map_widget.update()

    def marker_select(self, idx):
        self.tree_view.setCurrentIndex(
            self.tree_model.createIndex(idx, 0, self.map_widget.sectors))

    def name_edit(self):
        self.map_widget.sectors.markers[self.map_widget.sector_idx].name = self.section_name.text()
        self.tree_model.dataChanged.emit(
            self.tree_model.createIndex(self.map_widget.sector_idx, 0, self.map_widget.sectors),
            self.tree_model.createIndex(self.map_widget.sector_idx, 0, self.map_widget.sectors))
        self.tree_model.data_view.values_change.emit()

    def type_edit(self):
        self.map_widget.sectors.markers[self.map_widget.sector_idx].typ = \
            self.section_type.currentText()
        self.tree_model.dataChanged.emit(
            self.tree_model.createIndex(self.map_widget.sector_idx, 1, self.map_widget.sectors),
            self.tree_model.createIndex(self.map_widget.sector_idx, 1, self.map_widget.sectors))
        self.tree_model.data_view.values_change.emit()

    def update_selection(self):
        indexes = self.tree_view.selectedIndexes()
        self.sector_click(indexes[0] if indexes else None)

    def sector_click(self, index):
        child = self.tree_model.child(index) if index else None
        if child and isinstance(child.obj, state.Marker):
            self.map_widget.sectors = child.parent
            self.map_widget.sector_idx = child.key

            self.section_name.setText(child.obj.name)
            self.section_name.setEnabled(True)
            self.left.labelForField(self.section_name).setEnabled(True)

            self.section_type.clear()
            self.section_type.insertItems(0, sorted({obj.typ for obj in child.parent.markers}))
            self.section_type.setCurrentText(child.obj.typ)
            self.section_type.setEnabled(True)
            self.left.labelForField(self.section_type).setEnabled(True)

            self.split_button.setEnabled(True)
            self.remove_button.setEnabled(len(child.parent.markers) > 2)
        else:
            self.map_widget.sectors = child.obj if child else None
            self.map_widget.sector_idx = None

            self.section_name.setText('')
            self.section_name.setEnabled(False)
            self.left.labelForField(self.section_name).setEnabled(False)

            self.section_type.clear()
            self.section_type.clearEditText()
            self.section_type.setEnabled(False)
            self.left.labelForField(self.section_type).setEnabled(False)

            self.split_button.setEnabled(False)
            self.remove_button.setEnabled(False)
        self.map_widget.update()


def track_editor(parent, data_view):
    if data_view.track:
        orig_track = copy.deepcopy(data_view.track)
        dia = TrackDialog(data_view)
        if dia.exec_():
            save_track(data_view.track)
        else:
            data_view.track = orig_track
            data_view.values_change.emit()
