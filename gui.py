#!/usr/bin/env python3.10

# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import configparser
import dataclasses
import json
import os
import sys

from PySide2.QtCore import QSize, QStandardPaths, Qt, Signal
from PySide2.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import dacite

import ui.channels
import ui.components
import ui.datamgr
import ui.dockers
import ui.layout
import ui.math
import ui.state
import ui.timedist
import ui.track
import ui.widgets
from version import version

class TimeDistStatus(QLabel):
    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view
        data_view.cursor_change.connect(self.updateCursor)
        data_view.values_change.connect(self.updateCursor)
        self.updateCursor()

    def updateCursor(self, old_cursor=None):
        token = '\u25fe'
        self.setText('%s Time %.4f [s]   %s Distance %d [m]' %
                     (token if self.data_view.mode_time else '',
                      self.data_view.cursor_time.time / 1000,
                      token if not self.data_view.mode_time else '',
                      self.data_view.cursor_time.dist))

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    cursor_change = Signal(ui.state.TimeDistRef)
    values_change = Signal()
    data_change = Signal()

    def __init__(self):
        super().__init__()

        self.config = configparser.ConfigParser()
        self.config['main'] = {} # base structure initialization
        self.config_fname = (QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
                             + '/config.ini')
        self.config.read(self.config_fname)

        # upgrade from old versions
        if 'database_scan_dirs' not in self.config['main']:
            self.config['main']['database_scan_dirs'] = json.dumps(
                [d for d in ('C:\\MoTeC\\Logged Data',
                             'C:\\AIM_SPORT\\RaceStudio3\\user\\data')
                 if sys.platform == 'win32' and os.path.exists(d)])

        self.data_view = ui.state.DataView(ref_lap=None,
                                           alt_lap=None,
                                           extra_laps=[],
                                           cursor_time=ui.state.TimeDistRef(0., 0.),
                                           zoom_window=(ui.state.TimeDistRef(0., 0.),
                                                        ui.state.TimeDistRef(0., 0.)),
                                           mode_time=True,
                                           mode_offset=False,
                                           log_files=[],
                                           active_component=None,
                                           video_alignment={},
                                           maps_key=None,

                                           channel_overrides={},
                                           channel_properties={},
                                           channel_defaults={},

                                           maths=ui.state.Maths(),
                                           track=None,

                                           cursor_change=self.cursor_change,
                                           values_change=self.values_change,
                                           data_change=self.data_change,

                                           config=self.config)

        try:
            ui.math.set_user_func_dir(self.data_view, self.config.get('main', 'user_func_path'))
        except configparser.NoOptionError:
            pass

        try:
            self.data_view.maps_key = json.loads(self.config.get('main', 'maps_key'))
        except configparser.NoOptionError:
            pass

        self.workspace_dir = (QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
                              + '/workspaces')
        os.makedirs(self.workspace_dir, exist_ok=True)
        try:
            self.workspace_fname = self.config.get('main', 'workspace')
        except configparser.NoOptionError:
            self.workspace_fname = None
        self.update_title()

        menu = self.menuBar()
        file_menu = menu.addMenu('&File')
        layout_menu = menu.addMenu('Layout')
        add_menu = menu.addMenu('Add')
        data_menu = menu.addMenu('Data')
        self.comp_menu = menu.addMenu('Component')
        tools_menu = menu.addMenu('Tools')

        self.statusBar().addWidget(TimeDistStatus(self.data_view))

        toolbar = QToolBar()
        toolbar.setObjectName('DockerBar')
        toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        self.dockwidgets = []

        ui.dockers.ChannelsDockWidget(self, toolbar)
        self.datamgr = ui.datamgr.DataDockWidget(self, toolbar)
        ui.dockers.ValuesDockWidget(self, toolbar)
        ui.dockers.MapDockWidget(self, toolbar)

        lapwidget = ui.widgets.LapWidget(self.data_view)

        measures = ui.components.ComponentManager(self.data_view, add_menu)

        self.layout_mgr = ui.layout.LayoutManager(measures, self.data_view, layout_menu)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self.layout_mgr)
        vbox.addWidget(lapwidget)
        vbox.addWidget(measures, 1)

        dummy = QWidget()
        dummy.setLayout(vbox)
        self.setCentralWidget(dummy)

        file_menu.addAction('Open from db...').triggered.connect(self.datamgr.open_from_db)
        file_menu.addAction('Open from file...').triggered.connect(self.datamgr.open_from_file)
        file_menu.addAction('Close all log files').triggered.connect(self.datamgr.close_all_logs)
        file_menu.addSeparator()
        file_menu.addAction('New Workspace').triggered.connect(self.new_workspace)
        file_menu.addAction('Open Workspace...').triggered.connect(self.open_workspace)
        file_menu.addAction('Save Workspace').triggered.connect(self.save_workspace)
        file_menu.addAction('Save Workspace As...').triggered.connect(self.save_as_workspace)
        file_menu.addSeparator()
        file_menu.addAction('Preferences...').triggered.connect(self.preferences)
        file_menu.addSeparator()
        file_menu.addAction('Exit').triggered.connect(self.close)

        self.comp_menu.addAction('Hi') # dummy entry so Mac OSX will show the Component menu
        self.comp_menu.aboutToShow.connect(self.setup_component_menu)

        data_menu.addAction('Details...').triggered.connect(self.show_details)
        data_menu.addSeparator()
        data_menu.addAction('Time/Distance').triggered.connect(self.toggle_time_dist)
        self.data_offsets = data_menu.addAction('Show Data Offsets')
        self.data_offsets.triggered.connect(self.toggle_data_offsets)
        self.data_offsets.setCheckable(True)
        data_menu.addAction('Swap Ref/Alt Laps').triggered.connect(self.swap_ref_alt)
        data_menu.addSeparator()
        data_menu.addAction('Zoom to default').triggered.connect(self.zoom_default)

        tools_menu.addAction('Track editor...').triggered.connect(self.track_editor)
        tools_menu.addAction('Math channels...').triggered.connect(self.math_editor)

        try:
            self.restoreGeometry(bytes.fromhex(self.config.get('main', 'geometry')))
            self.restoreState(bytes.fromhex(self.config.get('main', 'widgets')))
        except configparser.NoOptionError:
            pass

        try:
            if self.workspace_fname:
                self.load_workspace(self.workspace_fname)
        except FileNotFoundError:
            pass

        for f in app.arguments()[1:]:
            self.datamgr.open_file(f)

    def sizeHint(self):
        return QSize(ui.widgets.deviceScale(self, 800), ui.widgets.deviceScale(self, 600))

    def update_title(self):
        if self.workspace_fname:
            self.config['main']['workspace'] = self.workspace_fname
            self.setWindowTitle("Track Data Analysis %s - %s" % (
                version, os.path.splitext(os.path.basename(self.workspace_fname))[0]))
        else:
            self.config.remove_option('main', 'workspace')
            self.setWindowTitle("Track Data Analysis")

    def setup_component_menu(self):
        # Can't use clear(), it may delete the actions
        for a in self.comp_menu.actions():
            self.comp_menu.removeAction(a)
        self.comp_menu.addActions(self.data_view.active_component.actions())

    def show_details(self):
        if not self.data_view.ref_lap:
            return

        layout = QVBoxLayout()

        metadata = sorted(self.data_view.ref_lap.log.log.get_metadata().items())
        metadata.append(('Filename',
                         os.path.basename(self.data_view.ref_lap.log.log.get_filename())))
        metadata.append(('Dirname', os.path.dirname(self.data_view.ref_lap.log.log.get_filename())))
        table = QTableWidget(len(metadata), 2)
        table.setSelectionMode(table.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().hide()
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.verticalHeader().hide()
        table.setEditTriggers(table.NoEditTriggers)
        row = 0
        for name, val in metadata:
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(str(val)))
            row += 1
        layout.addWidget(table)

        bbox = QDialogButtonBox(QDialogButtonBox.Close)
        layout.addWidget(bbox)

        dia = QDialog()
        dia.setWindowTitle('Details')
        dia.setLayout(layout)

        bbox.rejected.connect(dia.accept)
        dia.exec_()

    def toggle_time_dist(self):
        self.data_view.mode_time = not self.data_view.mode_time
        self.data_view.values_change.emit()

    def swap_ref_alt(self):
        if self.data_view.alt_lap:
            self.data_view.ref_lap, self.data_view.alt_lap = (self.data_view.alt_lap,
                                                              self.data_view.ref_lap)
            self.data_view.values_change.emit()
            self.data_view.data_change.emit()

    def zoom_default(self):
        self.data_view.zoom_window=(ui.state.TimeDistRef(0., 0.),
                                    ui.state.TimeDistRef(0., 0.))
        self.data_view.values_change.emit()

    def math_editor(self):
        ui.math.math_editor(self, self.data_view)

    def track_editor(self):
        ui.track.track_editor(self, self.data_view)

    def toggle_data_offsets(self, flag):
        if flag:
            self.data_view.mode_offset = flag
            self.data_view.active_component.update()
        else:
            laps = [lap for log in self.data_view.log_files for lap in log.laps]
            if any(lap.offset.time != 0 for lap in laps if lap):
                ret = QMessageBox.warning(self, 'Warning', 'Turning off data offsets will discard '
                                          'the offsets, continue?',
                                          QMessageBox.Discard | QMessageBox.Cancel,
                                          QMessageBox.Cancel)
                if ret == QMessageBox.Cancel:
                    # toggle back
                    self.data_offsets.setChecked(True)
                    return
            # zero all offsets
            for lap in laps:
                lap.offset = ui.state.TimeDistRef(0, 0)
            self.data_view.mode_offset = flag
            self.data_view.values_change.emit()

    def preferences(self):
        layout = QFormLayout()
        maptiler_key = QLineEdit(self.data_view.maps_key[1] if self.data_view.maps_key else '')
        layout.addRow('Maptiler Key', maptiler_key)

        ulayout = QHBoxLayout()
        userfuncpath = QLineEdit(self.config.get('main', 'user_func_path', fallback=''))
        ulayout.addWidget(userfuncpath)
        userfuncbutton = QToolButton()
        def userfunc_dialog():
            d = QFileDialog.getExistingDirectory(self, 'Directory for user math functions',
                                                 dir=userfuncpath.text())
            if d:
                userfuncpath.setText(d)
        userfuncbutton.clicked.connect(userfunc_dialog)
        userfuncbutton.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        ulayout.addWidget(userfuncbutton)
        layout.addRow('Path for user math functions', ulayout)

        sdlayout = QGridLayout()
        scandirs = QListWidget()
        scandirs.addItems(json.loads(self.config.get('main', 'database_scan_dirs')))

        def add_scan_dir():
            d = QFileDialog.getExistingDirectory(
                self, 'Directory to scan for logs')
            if d:
                d = os.path.normpath(d)
                if d not in [scandirs.item(i).text()
                             for i in range(scandirs.count())]:
                    scandirs.addItem(d)

        def rem_scan_dir():
            for i in scandirs.selectedIndexes():
                scandirs.takeItem(i.row())

        sdlayout.addWidget(scandirs, 0, 0, 1, 2)
        but = QPushButton('Add directory...')
        but.clicked.connect(add_scan_dir)
        sdlayout.addWidget(but, 1, 0, 1, 1)
        but = QPushButton('Remove directory')
        but.clicked.connect(rem_scan_dir)
        sdlayout.addWidget(but, 1, 1, 1, 1)

        scandirbox = QGroupBox('Log file directories')
        scandirbox.setLayout(sdlayout)
        layout.addRow(scandirbox)

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addRow(bbox)

        dia = QDialog()
        dia.setWindowTitle('Preferences')
        dia.setLayout(layout)

        bbox.accepted.connect(dia.accept)
        bbox.rejected.connect(dia.reject)

        if dia.exec_():
            self.data_view.maps_key = ['maptiler', maptiler_key.text()]
            self.config['main']['user_func_path'] = userfuncpath.text()
            ui.math.set_user_func_dir(self.data_view, userfuncpath.text())
            self.config['main']['database_scan_dirs'] = json.dumps(
                [scandirs.item(i).text() for i in range(scandirs.count())])
            self.datamgr.update_scan_dirs()

    def new_workspace(self):
        ret = QMessageBox.warning(self, 'Warning',
                                  'Creating a new workspace will discard the current one.',
                                  QMessageBox.Discard | QMessageBox.Save | QMessageBox.Cancel,
                                  QMessageBox.Cancel)
        if ret == QMessageBox.Cancel:
            return
        if ret == QMessageBox.Save:
            self.save_workspace()
        self.workspace_fname = None
        self.update_title()
        self.layout_mgr.new_layout()

    def open_workspace(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open workspace',
                                                self.workspace_dir, 'Workspaces (*.json)')[0]
        if file_name:
            self.load_workspace(file_name)
            self.workspace_fname = file_name
            self.update_title()

    def load_workspace(self, file_name):
        with open(file_name, 'rt', encoding='utf-8') as f:
            ws_data = json.load(f)
        self.layout_mgr.load_state(ws_data['layout'])
        self.data_view.video_alignment = ws_data['videos']
        self.data_view.channel_overrides = ws_data.get('channels', {})
        self.data_view.maths.groups = {k: dacite.from_dict(data_class=ui.state.MathGroup, data=v)
                                       for k, v in ws_data.get('math_groups', {}).items()}
        self.data_view.math_invalidate()
        ui.channels.update_channel_properties(self.data_view)
        self.data_view.values_change.emit()
        self.data_view.data_change.emit() # XXX make a new signal for log file / workspace changes

    def save_workspace(self):
        if not self.workspace_fname: return self.save_as_workspace()
        with ui.state.atomic_write(self.workspace_fname) as f:
            json.dump({'layout': self.layout_mgr.save_state(),
                       'videos': self.data_view.video_alignment,
                       'channels': self.data_view.channel_overrides,
                       'math_groups': {k: dataclasses.asdict(v)
                                       for k, v in self.data_view.maths.groups.items()},
                       }, f, indent=4)
        return True

    def save_as_workspace(self):
        file_name = QFileDialog.getSaveFileName(
            self, 'Save workspace',
            self.workspace_fname if self.workspace_fname else self.workspace_dir,
            'Workspaces (*.json)')[0]
        if not file_name: return False
        if not file_name.endswith('.json'):
            file_name += '.json'
        self.workspace_fname = file_name
        self.update_title()
        return self.save_workspace()

    def closeEvent(self, e):
        e.ignore()
        if not self.workspace_fname:
            ret = QMessageBox.warning(self, 'Warning',
                                      'Save workspace before exitting?',
                                      QMessageBox.Discard | QMessageBox.Save | QMessageBox.Cancel,
                                      QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                return
            if ret == QMessageBox.Discard:
                e.accept()
            elif not self.save_workspace():
                return
        elif not self.save_workspace():
            return
        self.datamgr.stop_metadata_scan()
        self.config['main']['geometry'] = bytes(self.saveGeometry()).hex()
        self.config['main']['widgets'] = bytes(self.saveState()).hex()
        self.config['main']['maps_key'] = json.dumps(self.data_view.maps_key)
        with ui.state.atomic_write(self.config_fname) as f:
            self.config.write(f)
        e.accept()




app = QApplication(sys.argv)
app.setApplicationName('TrackDataAnalysis')

window = MainWindow()
window.show()

# Start the event loop.
app.exec_()
