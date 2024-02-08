
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from dataclasses import dataclass

from PySide2.QtCore import QRect, QSize, Qt
from PySide2 import QtGui
from PySide2.QtWidgets import (
    QHeaderView,
    QLineEdit,
    QMenu,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .dockers import FastTableModel, FastItemDelegate, TempDockWidget
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
