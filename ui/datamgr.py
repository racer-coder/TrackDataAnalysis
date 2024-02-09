
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
from dataclasses import dataclass
import json
import os
import sys
import threading

from PySide2.QtCore import QFileSystemWatcher, QRect, QSize, QStandardPaths, Qt, Signal
from PySide2 import QtGui
from PySide2.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QLineEdit,
    QMenu,
    QProgressDialog,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import data.aim_xrk
import data.autosport_labs
import data.motec
from .dockers import (FastTableModel,
                      FastItemDelegate,
                      TempDockWidget,
                      TextMatcher)
from . import channels
from . import state
from . import widgets

@dataclass
class DataModelLap:
    lap: state.LapRef
    best_lap: int

    def present(self, index, model):
        col = index.column()
        if col == 0:
            return (model.right_style, str(self.lap.num))
        if col == 1:
            return (model.right_style, state.format_time(self.lap.duration()))
        if col == 2:
            return (([state.lap_colors[0]] + model.center_style[1:], '\u278a')
                    if self.lap == model.data_view.ref_lap else (model.center_style, '\u2d54'))
        if col == 3:
            return (([state.lap_colors[1]] + model.center_style[1:], '\u278b')
                    if self.lap == model.data_view.alt_lap else (model.center_style, '\u2d54'))
        if col == 4:
            for idx, (lap, color) in enumerate(model.data_view.extra_laps):
                if lap == self.lap:
                    return ([color] + model.center_style[1:], chr(0x278c + idx))
            return (model.center_style, '\u2d54')

        if col == 5:
            return (model.right_style,
                    state.format_time(self.lap.duration() - self.best_lap))
        if col == 6:
            return (model.right_style,
                    state.format_time(self.lap.offset.time) if self.lap.offset.time else None)
        if col == 7:
            return (model.right_style,
                    '%.2f' % self.lap.offset.dist if self.lap.offset.dist else None)
        return None

@dataclass
class DataModelSection:
    text: str
    lap: state.LapRef = None

    def present(self, index, model):
        col = index.column()
        if col == 0:
            return (model.section_style, self.text)
        return (model.section_style, None)

class DataDockModel(FastTableModel):
    headings = ['Lap', 'Time', 'R', 'A', '3', 'Delta', 'Time off', 'Dist off']

    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view
        self.laps = []
        self.right_style = [QtGui.QPen(QtGui.QColor(192, 192, 192)), None, None, Qt.AlignRight]
        self.center_style = self.right_style[:3] + [Qt.AlignHCenter]
        self.section_style = [QtGui.QPen(QtGui.QColor(255, 255, 255)), QtGui.QColor(64, 64, 64),
                              None, Qt.AlignLeft]

    def set_data(self, laps, font, section_font):
        old_len = len(self.laps)
        self.laps = laps
        self.right_style[2] = font
        self.center_style[2] = font
        self.section_style[2] = section_font
        super().set_data(self.headings, len(laps))

    def present(self, index):
        return self.laps[index.row()].present(index, self)

class DataDockWidget(TempDockWidget):
    metadata_ver = 1
    wake_watcher = Signal(str)
    status_msg = Signal(str)
    add_watch = Signal(str)

    def __init__(self, mainwindow, toolbar):
        super().__init__('Data', mainwindow, toolbar, True)

        self.margin = 6

        self.model = DataDockModel(mainwindow.data_view)
        self.deleg = FastItemDelegate(self.model, self.margin)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(self.deleg)
        self.table.setSelectionMode(self.table.SingleSelection)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setMinimumSectionSize(10)
        self.table.setHorizontalScrollMode(self.table.ScrollPerPixel)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setMinimumSectionSize(5)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        self.table.pressed.connect(self.clickCell)

        pal = self.table.palette()
        pal.setColor(pal.Base, QtGui.QColor(0, 0, 0))
        self.table.setPalette(pal)

        self.setWidget(self.table)

        mainwindow.data_view.values_change.connect(self.recompute)
        mainwindow.data_view.data_change.connect(self.recompute)
        self.recompute()

        self.config = mainwindow.config
        self.data_view = mainwindow.data_view
        self.status_msg.connect(mainwindow.statusBar().showMessage)
        self.metadata_fname = (QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
                               + '/metadata.json')
        self.metadata_cache = {}
        self.load_metadata_cache()
        self.keep_watching = True
        self.watch_deleted = 0
        self.watcher = QFileSystemWatcher()
        self.watcher.directoryChanged.connect(self.add_watch_dir)
        self.add_watch.connect(self.add_watch_dir)
        self.watchqueue = [] # order isn't important
        self.watch_semaphore = threading.Semaphore(0)
        threading.Thread(target = self.process_loop).start()
        try:
            for d in ['sampledata']: #json.loads(mainwindow.config.get('main', 'watched_dirs')):
                self.add_watch_dir(d)
        except:
            pass

    def rewrite_metadata_cache(self):
        with open(self.metadata_fname, 'wt') as f:
            f.write('%d\n' % self.metadata_ver)
            for obj in self.metadata_cache.items():
                f.write(json.dumps(obj) + '\n')
        self.watch_deleted = 0

    def load_metadata_cache(self):
        try:
            with open(self.metadata_fname, 'rt') as f:
                assert json.loads(f.readline()) == self.metadata_ver
                for line in f:
                    obj = json.loads(line)
                    self.metadata_cache[obj['path']] = obj
        except: # catch all error handling, grotesque
            self.rewrite_metadata_cache()

    def add_watch_dir(self, d):
        self.watcher.addPath(d)
        self.watchqueue.append(d)
        self.watch_semaphore.release()

    def stop_metadata_scan(self):
        self.keep_watching = False
        self.watch_semaphore.release()

    def process_loop(self):
        while self.keep_watching:
            sys.setswitchinterval(0.0001) # need crazy value to keep responsiveness, thanks GIL
            self.process_watch()
            sys.setswitchinterval(0.005) # back to default, thanks GIL
            if not self.watchqueue:
                self.watch_semaphore.acquire()

    def process_watch(self):
        if not self.watchqueue: return

        f = None
        todo = self.watchqueue.pop()

        self.status_msg.emit('Scanning ' + todo)
        new_files = []
        new_dirs = []
        for obj in os.scandir(todo):
            if obj.is_dir():
                new_dirs.append(obj.path + os.pathsep)
                # need to send this back to the main thread to call addPath on self.watcher safely
                self.add_watch.emit(obj.path)
            else:
                stat = obj.stat()
                new_files.append((obj.path, stat.st_size, stat.st_mtime))

        keep = sorted([fname for fname, _, _ in new_files] + new_dirs)
        to_del = [old for old in self.metadata_cache.keys()
                  if old.startswith(todo) and
                  not old.startswith(keep[min(bisect.bisect_left(keep, old),
                                              len(keep) - 1)])]
        for elem in to_del:
            del self.metadata_cache[elem]
            self.watch_deleted += 1

        for path, size, mtime in new_files:
            if not self.keep_watching:
                break
            if path in self.metadata_cache:
                entry = self.metadata_cache[path]
                if entry['size'] == size and entry['mtime'] == mtime:
                    continue

            if path.endswith('.xrk'):
                builder = data.aim_xrk.AIMXRK
            elif path.endswith('.ld'):
                builder = data.motec.MOTEC
            elif path.endswith('.log'):
                builder = data.autosport_labs.AutosportLabs
            else:
                continue # not a file we care about

            self.status_msg.emit('Reading ' + path)
            try:
                obj = builder(path, lambda x, y: None).get_metadata()
                readable = True
            except:
                obj = None
                readable = False

            metadata =  {'path': path,
                         'size': size,
                         'mtime': mtime,
                         'readable': readable,
                         'metadata': obj}
            self.metadata_cache[path] = metadata
            if not f:
                # rewrite the cache if there is enough garbage
                if self.watch_deleted > len(self.metadata_cache):
                    self.rewrite_metadata_cache()
                f = open(self.metadata_fname, 'at')
            f.write(json.dumps(metadata) + '\n')

        self.status_msg.emit('')
        if f:
            f.close()

    def open_from_db(self):
        layout = QGridLayout()

        search = QLineEdit()
        search.setClearButtonEnabled(True)
        search.setPlaceholderText('Search')
        layout.addWidget(search, 0, 0, 1, 1)

        files = QTableWidget()
        files.setSelectionMode(files.SingleSelection)
        files.setSelectionBehavior(files.SelectRows)
        files.setShowGrid(False)
        files.horizontalHeader().setHighlightSections(False)
        files.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        files.setHorizontalScrollMode(files.ScrollPerPixel)
        files.verticalHeader().hide()
        files.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        files.setEditTriggers(files.NoEditTriggers)
        dblist = [d for d in self.metadata_cache.values()
                  if d['readable']]
        collist = ['Log Date', 'Log Time', 'Venue', 'Driver']
        files.setRowCount(len(dblist))
        files.setColumnCount(len(collist))
        for i, f in enumerate(dblist):
            for j, c in enumerate(collist):
                if c in f['metadata']:
                    files.setItem(i, j, QTableWidgetItem(f['metadata'][c]))
        layout.addWidget(files, 1, 0, 1, 1)

        metadata = QTableWidget()
        metadata.setSelectionMode(metadata.NoSelection)
        metadata.setShowGrid(False)
        metadata.horizontalHeader().hide()
        metadata.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        metadata.setHorizontalScrollMode(metadata.ScrollPerPixel)
        metadata.verticalHeader().hide()
        metadata.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        metadata.setEditTriggers(metadata.NoEditTriggers)
        layout.addWidget(metadata, 0, 1, 2, 1)

        def update_matches(txt):
            matcher = TextMatcher(txt)
            for i, f in enumerate(dblist):
                files.setRowHidden(i, not any(matcher.match(str(d))
                                              for d in f['metadata'].values()))
        search.textChanged.connect(update_matches)

        def cell_selected():
            row = files.selectedIndexes()[0].row()
            f = dblist[row]['metadata']
            metadata.setRowCount(len(f))
            metadata.setColumnCount(2)
            for i, (k, v) in enumerate(sorted(f.items())):
                metadata.setItem(i, 0, QTableWidgetItem(k))
                metadata.setItem(i, 1, QTableWidgetItem(str(v)))
        files.itemSelectionChanged.connect(cell_selected)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(bbox, 2, 0, 1, 2)

        dia = QDialog(self)
        dia.setWindowTitle('Open log file from database')
        dia.setLayout(layout)
        dia.resize(QSize(widgets.devicePointScale(self, 600),
                         widgets.devicePointScale(self, 400)))

        files.cellActivated.connect(dia.accept)

        bbox.accepted.connect(dia.accept)
        bbox.rejected.connect(dia.reject)

        if not dia.exec_(): return
        selection = files.selectedIndexes()
        if not selection: return
        self.open_file(dblist[selection[0].row()]['path'])

    def open_from_file(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open data file for analysis',
                                                self.config.get('main', 'last_open_dir',
                                                                fallback=os.getcwd()),
                                                'Data files (*.xrk *.ld *.log)')[0]
        if file_name:
            self.open_file(file_name)

    def open_file(self, file_name):
        if file_name.endswith('.xrk'):
            builder = data.aim_xrk.AIMXRK
        elif file_name.endswith('.ld'):
            builder = data.motec.MOTEC
        elif file_name.endswith('.log'):
            builder = data.autosport_labs.AutosportLabs
        else:
            QMessageBox.critical(self, 'Unknown extension',
                                 'Unable to determine format for file.',
                                 QMessageBox.Ok)
            return

        progress = QProgressDialog('Processing file', 'Cancel', 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(1000)
        def update_progress(pos, total):
            progress.setMaximum(total)
            progress.setValue(pos)
            if progress.wasCanceled():
                raise KeyboardInterrupt # really?

        try:
            obj = builder(file_name, update_progress)
        except KeyboardInterrupt:
            return # abort load
        finally:
            progress.deleteLater()

        logref = state.LogRef(data.distance.DistanceWrapper(obj))
        logref.update_laps()
        self.data_view.log_files.append(logref)
        if len(self.data_view.log_files) > 1:
            # do it all again, but for multiple logs
            data.distance.unify_lap_distance([logref.log for logref in self.data_view.log_files])
            for logref in self.data_view.log_files:
                logref.update_laps()
            # this sucks but we've lost our mapping to the original laps
            self.data_view.alt_lap = None
            self.data_view.extra_laps = []

        channels.update_channel_properties(self.data_view)

        laps = logref.laps
        if not laps:
            best_lap = None
        else:
            best_lap = min(laps[1:-1] if len(laps) >= 3 else laps,
                           key=lambda x: x.duration())
        self.data_view.ref_lap = best_lap
        self.data_view.values_change.emit()
        self.data_view.data_change.emit()
        self.config['main']['last_open_dir'] = os.path.dirname(file_name)

    def clickCell(self, index):
        row = index.row()
        col = index.column()
        lapref = self.model.laps[row].lap
        if not lapref: return # must be a section divider
        data_view = self.mainwindow.data_view
        if col <= 2:
            data_view.ref_lap = lapref
            data_view.zoom_window = (state.TimeDistRef(0, 0), state.TimeDistRef(0, 0))
        elif col == 3:
            if data_view.alt_lap == lapref:
                data_view.alt_lap = None
            else:
                data_view.alt_lap = lapref
        elif col == 4:
            removed = [(l, c) for l, c in data_view.extra_laps if l != lapref]
            if len(removed) != len(data_view.extra_laps):
                data_view.extra_laps = removed
            elif len(data_view.extra_laps) < 8: # some sane upper bound
                used_colors = [c for l, c in data_view.extra_laps]
                for color in state.lap_colors[2:]:
                    if color not in used_colors:
                        break
                data_view.extra_laps.append((lapref, color))
        data_view.values_change.emit()

    def best_lap(self, logref):
        laps = logref.laps
        return min(laps[1:-1] if len(laps) >= 3 else laps,
                   key=lambda x: x.duration()).duration()

    def recompute(self):
        font_size = 12
        font = QtGui.QFont('Tahoma')
        font.setPixelSize(widgets.devicePointScale(self, font_size))
        section_font = QtGui.QFont(font)
        section_font.setBold(True)
        metrics = QtGui.QFontMetrics(font)
        logs = [(logref, self.best_lap(logref)) for logref in self.mainwindow.data_view.log_files]
        self.table.clearSpans()
        laps = []
        last_date = None
        logs.sort(key=lambda r: (r[0].log.get_metadata()['Log Date'],
                                 r[0].log.get_metadata()['Log Time']))
        for logref, best_lap in logs:
            metadata = logref.log.get_metadata()
            date = metadata['Log Date']
            if date != last_date:
                self.table.setSpan(len(laps), 0, 1, 8)
                laps.append(DataModelSection('\u25bc ' + date))
            self.table.setSpan(len(laps), 0, 1, 8)
            laps.append(DataModelSection('  \u25bc %s' % ', '.join(
                filter(bool, [metadata['Log Time'],
                              metadata.get('Session', None),
                              metadata.get('Driver', None)]))))
            laps += [DataModelLap(lap, best_lap) for lap in logref.laps]
        self.model.set_data(laps, font, section_font)
        self.deleg.metrics = metrics
        self.table.setColumnWidth(0, self.margin * 2 + metrics.horizontalAdvance('Lap 888'))
        self.table.setColumnWidth(1, self.margin * 2 + metrics.horizontalAdvance('88:88.888'))
        self.table.setColumnWidth(2, self.margin * 2 + metrics.horizontalAdvance('M'))
        self.table.setColumnWidth(3, self.margin * 2 + metrics.horizontalAdvance('M'))
        self.table.setColumnWidth(4, self.margin * 2 + metrics.horizontalAdvance('M'))
        self.table.setColumnWidth(5, self.margin * 2 + metrics.horizontalAdvance('+88:88.888'))
        self.table.setColumnWidth(6, self.margin * 2 + metrics.horizontalAdvance('88:88.888'))
        self.table.setColumnWidth(7, self.margin * 2 + metrics.horizontalAdvance('+88888'))
        for row in range(self.model.rowCount(None)):
            self.table.setRowHeight(row, metrics.height())
