#!/usr/bin/env python3.10

# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import configparser
import json
import os
import pprint
import sys

from PySide2.QtCore import QSize, QStandardPaths, Qt, Signal
from PySide2.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import data.aim_xrk
import data.autosport_labs
import data.distance
import data.motec
import ui.channels
import ui.components
import ui.dockers
import ui.layout
import ui.state
import ui.timedist
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

                                           cursor_change=self.cursor_change,
                                           values_change=self.values_change,
                                           data_change=self.data_change)

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

        self.statusBar().addWidget(TimeDistStatus(self.data_view))

        toolbar = QToolBar()
        toolbar.setObjectName('DockerBar')
        toolbar.setMovable(False)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        self.dockwidgets = []

        ui.dockers.ChannelsDockWidget(self, toolbar)
        ui.dockers.DataDockWidget(self, toolbar)
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

        file_menu.addAction('Open...').triggered.connect(self.open)
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

        data_menu.addAction('Time/Distance').triggered.connect(self.toggle_time_dist)
        self.data_offsets = data_menu.addAction('Show Data Offsets')
        self.data_offsets.triggered.connect(self.toggle_data_offsets)
        self.data_offsets.setCheckable(True)
        data_menu.addAction('Swap Ref/Alt Laps').triggered.connect(self.swap_ref_alt)
        data_menu.addSeparator()
        data_menu.addAction('Zoom to default').triggered.connect(self.zoom_default)

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
            self.open_file(f)

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

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(bbox)

        dia = QDialog(self)
        dia.setWindowTitle('Preferences')
        dia.setLayout(layout)

        bbox.accepted.connect(dia.accept)
        bbox.rejected.connect(dia.reject)

        if dia.exec_():
            self.data_view.maps_key = ['maptiler', maptiler_key.text()]

    def open(self):
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

        logref = ui.state.LogRef(data.distance.DistanceWrapper(obj))
        pprint.pprint(logref.log.get_metadata())
        logref.laps = [
            ui.state.LapRef(logref,
                            lap.num,
                            ui.state.TimeDistRef(lap.start_time,
                                                 logref.log.outTime2Dist(lap.start_time)),
                            ui.state.TimeDistRef(lap.end_time,
                                                 logref.log.outTime2Dist(lap.end_time)),
                            ui.state.TimeDistRef(0., 0.))
            for lap in logref.log.get_laps()]
        self.data_view.log_files.append(logref)
        ui.channels.update_channel_properties(self.data_view)

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
        with open(file_name, 'rt') as f:
            ws_data = json.load(f)
        self.layout_mgr.load_state(ws_data['layout'])
        self.data_view.video_alignment = ws_data['videos']
        self.data_view.maps_key = ws_data['maps_key']
        self.data_view.channel_overrides = ws_data.get('channels', {})
        ui.channels.update_channel_properties(self.data_view)
        self.data_view.data_change.emit() # XXX make a new signal for log file / workspace changes

    def save_workspace(self):
        if not self.workspace_fname: return self.save_as_workspace()
        new_name = self.workspace_fname + '.new'
        with open(new_name, 'wt') as f:
            json.dump({'layout': self.layout_mgr.save_state(),
                       'videos': self.data_view.video_alignment,
                       'maps_key': self.data_view.maps_key,
                       'channels': self.data_view.channel_overrides,
                       }, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(new_name, self.workspace_fname) # atomic replace
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
        self.config['main']['geometry'] = bytes(self.saveGeometry()).hex()
        self.config['main']['widgets'] = bytes(self.saveState()).hex()
        new_name = self.config_fname + '.new'
        with open(new_name, 'wt') as f:
            self.config.write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(new_name, self.config_fname)
        e.accept()




app = QApplication(sys.argv)
app.setApplicationName('TrackDataAnalysis')

window = MainWindow()
window.show()

# Start the event loop.
app.exec_()
