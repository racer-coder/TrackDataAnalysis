
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
from dataclasses import dataclass
import math
import sys

from PySide2.QtCore import QAbstractTableModel, QRect, QSize, Qt
from PySide2 import QtGui
from PySide2.QtWidgets import (
    QAbstractItemDelegate,
    QDockWidget,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QMenu,
    QStyle,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data import unitconv
from . import channels
from . import map
from . import state
from . import widgets

class FastTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.heading = []
        self.row_count = 0
        self.col_count = 0

    def set_data(self, heading, nrows):
        self.heading = heading
        self.row_count = nrows
        self.col_count = len(heading)
        self.layoutChanged.emit()

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.heading[section]
        return None

    # Called too often, make them fast
    def rowCount(self, index): return self.row_count
    def columnCount(self, index): return self.col_count

    def data(self, index, role): return None

    # end user must override
    def present(self, index):
        return None


class FastItemDelegate(QAbstractItemDelegate):
    def __init__(self, model, margin):
        super().__init__()
        self.model = model
        self.margin = margin
        self.metrics = None # only one fontmetrics - not worried about section names exceeding width

    def paint(self, painter, opt, index):
        (fg, bg, font, align), txt = self.model.present(index)
        if bg and not (opt.state & QStyle.State_Selected):
            painter.fillRect(opt.rect, bg)
        if txt:
            painter.setPen(fg)
            painter.setFont(font)
            rect = opt.rect.adjusted(self.margin, 0, -self.margin, 0)
            painter.drawText(rect, align,
                             self.metrics.elidedText(txt, Qt.ElideRight, rect.width()))

    def sizeHint(self):
        return QSize(1, 1) # who cares

    def set_metrics(self, metrics):
        self.metrics = metrics


class TempDockWidget(QDockWidget):
    def __init__(self, title, mainwindow, toolbar, prefer_float):
        super().__init__(title)
        self.setObjectName(title) # for saving/restoring state

        self.mainwindow = mainwindow
        self.toolbar = toolbar
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.button = widgets.RotatedPushButton(title, toolbar)
        self.button.setFocusPolicy(Qt.NoFocus) # Undo setting from QAbstractButton / style
        self.button.setCheckable(True)
        self.button.clicked.connect(self.clicked)
        self.button_action = toolbar.addWidget(self.button)
        self.visibilityChanged.connect(self.button.setChecked)
        mainwindow.dockwidgets.append(self)
        self.mainwindow.addDockWidget(
            Qt.LeftDockWidgetArea if prefer_float else Qt.RightDockWidgetArea, self)
        self.prefer_float = prefer_float
        self.hide()

    def clicked(self):
        if not self.button.isChecked():
            self.hide()
        elif not self.prefer_float:
            self.show()
        else:
            self.setFloating(True)
            mwg = self.mainwindow.geometry()
            tbg = self.toolbar.geometry()
            if sys.platform == 'win32':
                reserve_height = mwg.y() - self.mainwindow.frameGeometry().y()
            else:
                reserve_height = 0
            self.setGeometry(mwg.x() + tbg.x() + tbg.width(), mwg.y() + tbg.y() + reserve_height,
                             self.sizeHint().width(), tbg.height() - reserve_height)
            self.show()

            # unshow any currently floating dock
            for d in self.mainwindow.dockwidgets:
                if d != self and d.isFloating() and not d.isHidden():
                    d.hide()

class DataDockModel(QAbstractTableModel):
    headings = ['Lap', 'Time', 'R', 'A', '3', 'Delta', 'Time off', 'Dist off']
    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view
        self.laps = []

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headings[section]
        return None

    def set_data(self, laps):
        old_len = len(self.laps)
        self.laps = laps
        self.layoutChanged.emit()

    def rowCount(self, index): return len(self.laps)
    def columnCount(self, index): return 8

    def data(self, index, role):
        if role == Qt.DisplayRole:
            lapref, best_lap = self.laps[index.row()]
            col = index.column()
            if col == 0: return str(lapref.num)
            if col == 1: return state.format_time(lapref.duration())
            if col == 2: return '\u278a' if lapref == self.data_view.ref_lap else '\u2d54'
            if col == 3: return '\u278b' if lapref == self.data_view.alt_lap else '\u2d54'
            if col == 4: return ([chr(0x278c + idx)
                                  for idx, (lap, color) in enumerate(self.data_view.extra_laps)
                                  if lap == lapref] +
                                 ['\u2d54'])[0]
            if col == 5: return state.format_time(lapref.duration() - best_lap)
            if col == 6: return state.format_time(lapref.offset.time) if lapref.offset.time else ''
            if col == 7: return '%.2f' % lapref.offset.dist if lapref.offset.dist else ''
            return None
        if role == Qt.TextAlignmentRole:
            col = index.column()
            # XXX Can't get QT to respect any vertical alignment here...
            if col >= 2 and col <= 4: return Qt.AlignHCenter #| Qt.AlignVCenter
            return Qt.AlignRight # | Qt.AlignVCenter
        if role == Qt.FontRole:
            col = index.column()
            return self.font if col < 2 or col > 4 else self.big_font
        return None

class DataDockWidget(TempDockWidget):
    def __init__(self, mainwindow, toolbar):
        super().__init__('Data', mainwindow, toolbar, True)

        self.model = DataDockModel(mainwindow.data_view)
        self.table = QTableView()
        self.table.setModel(self.model)
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
        self.setWidget(self.table)

        self.margin = 6
        mainwindow.data_view.values_change.connect(self.recompute)
        mainwindow.data_view.data_change.connect(self.recompute)
        self.recompute()

    def clickCell(self, index):
        row = index.row()
        col = index.column()
        lapref = self.model.laps[row][0]
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
        self.model.font = QtGui.QFont('Tahoma')
        self.model.font.setPixelSize(widgets.devicePointScale(self, font_size))
        self.model.big_font = QtGui.QFont('Tahoma')
        self.model.big_font.setPixelSize(widgets.devicePointScale(self, font_size * 1))
        logs = [(logref, self.best_lap(logref)) for logref in self.mainwindow.data_view.log_files]
        laps = [(lap, best_lap) for logref, best_lap in logs for lap in logref.laps]
        self.model.set_data(laps)
        metrics = QtGui.QFontMetrics(self.model.font)
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

class TextMatcher:
    def __init__(self, txt):
        self.text_hints = txt.lower().split(' ')

    def match(self, other):
        return all(any(piece.startswith(h)
                       for piece in other.lower().replace('_', ' ').split())
                   for h in self.text_hints)

class ChannelsDockWidget(TempDockWidget):
    def __init__(self, mainwindow, toolbar):
        super().__init__('Channels', mainwindow, toolbar, True)

        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self.textChanged)

        self.chList = QListWidget()
        self.chList.setSelectionMode(QListWidget.NoSelection)
        self.chList.itemActivated.connect(self.activateItem)
        self.chList.customContextMenuRequested.connect(self.context_menu)
        self.chList.setContextMenuPolicy(Qt.CustomContextMenu)

        self.matcher = TextMatcher('')
        self.recompute()
        mainwindow.data_view.data_change.connect(self.recompute)

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.chList)

        widget = QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

    def activateItem(self, item):
        ac = self.mainwindow.data_view.active_component
        if ac:
            ac.addChannel(item.text())

    def textChanged(self, txt):
        self.matcher = TextMatcher(txt)
        self.update_hidden()

    def context_menu(self, pos):
        ch = self.chList.itemAt(pos).text()
        # channel specific context menu
        menu = QMenu()
        menu.addAction(ch) # dummy entry so the user knows exactly what we're operating on
        menu.addSeparator()
        menu.addAction('Edit channel...').triggered.connect(
            lambda: channels.channel_editor(self, self.mainwindow.data_view, ch))
        ac = self.mainwindow.data_view.active_component
        if ac:
            menu.addSeparator()
            act = menu.addAction('Remove channel' if ch in ac.channels() else 'Add channel')
            act.triggered.connect(lambda: ac.addChannel(ch))
        menu.exec_(self.mapToGlobal(pos))

    def update_hidden(self):
        for i in range(self.chList.count()):
            it = self.chList.item(i)
            hide = not self.matcher.match(it.text())
            if hide != it.isHidden():
                it.setHidden(hide)

    def recompute(self):
        current = self.chList.currentItem()
        if current: current = current.text()
        items = [ch
                 for logfile in self.mainwindow.data_view.log_files
                 for ch in logfile.log.get_channels()]
        self.chList.clear()
        self.chList.addItems(items)
        self.update_hidden()
        if current:
            for it in self.chList.findItems(current, Qt.MatchExactly):
                self.chList.setCurrentItem(it)
                break

        ac = self.mainwindow.data_view.active_component
        if ac:
            chSet = ac.channels()
        else:
            chSet = set()
        for i in range(self.chList.count()):
            it = self.chList.item(i)
            if it.text() in chSet:
                it.setBackgroundColor(QtGui.QColor(255, 255, 0))
            else:
                it.setBackground(QtGui.QBrush(Qt.NoBrush))


@dataclass
class ValuesTableSection:
    model: object
    icon: object
    name: object
    units: object

    def __init__(self, model, title):
        self.model = model
        self.units = (model.section_style, None)
        self.icon = (model.section_style, '\u25bc')
        self.name = (model.section_style, title)

    def value(self, lap, delta): return (self.model.section_style, None)

@dataclass
class ValuesTableChannel:
    model: 'ValuesTableModel'
    data_view: object
    channel: str
    dec_pts: int

    icon: object
    name: object
    units: object

    def __init__(self, icon, model, data_view, channel, units = None, dec_pts = None):
        # properties might not exist if the channel comes from a component
        props = data_view.get_channel_prop(channel) if data_view else None
        self.model = model
        self.data_view = data_view
        self.channel = channel
        self.icon = ([channels.colors[props.color]] + model.channel_style[1:], '\u25a0') if icon else (model.channel_style, None)
        self.name = (model.channel_style, channel)
        if units is None and data_view.ref_lap:
            units = props.units
            dec_pts = props.dec_pts
        self.units = (model.channel_style,
                      unitconv.display_text(units) or None) # in case units == '', None is faster
        self.dec_pts = dec_pts

    def _calc(self, lap):
        # XXX MOVE INTO state.py
        d = self.data_view.get_channel_data(lap[0], self.channel)
        start_idx = max(0, bisect.bisect_left(d.timecodes, lap[1]) - 1)
        # interpolate between start_idx and start_idx+1?
        if start_idx >= len(d.values):
            return None
        return d.values[start_idx]

    def _format(self, v): return '%.*f' % (self.dec_pts, v)
    def _format_delta(self, v): return '%+.*f' % (self.dec_pts, v)

    def value(self, lap, delta):
        v = self._calc(lap)
        if v is None:
            return (self.model.value_style, None)
        if delta is None or delta == lap:
            return (self.model.value_style, self._format(v))
        b = self._calc(delta)
        return (self.model.delta_style, self._format_delta(v-b))

class ValuesTableFunc(ValuesTableChannel):
    def __init__(self, model, channel, func, units):
        super().__init__(False, model, None, channel, units, 0)
        self.func = func

    def _calc(self, lap): return self.func(lap[0])

class ValuesTableFuncTime(ValuesTableFunc):
    def _format(self, ms):
        return '%d:%06.3f' % (math.copysign(math.trunc(ms / 60000), ms),
                              abs(ms) % 60000 / 1000)

    def _format_delta(self, ms):
        return '%+.f:%06.3f' % (math.copysign(math.trunc(ms / 60000), ms),
                               abs(ms) % 60000 / 1000)

class ValuesTableModel(FastTableModel):
    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view
        self.section_style = [QtGui.QPen(QtGui.QColor(255, 255, 255)), QtGui.QColor(64, 64, 64),
                              None, Qt.AlignTop | Qt.AlignLeft]
        self.channel_style = [QtGui.QPen(QtGui.QColor(192, 192, 192)), QtGui.QColor(32, 32, 32),
                              None, Qt.AlignTop | Qt.AlignLeft]
        self.value_style   = [QtGui.QPen(QtGui.QColor(192, 192, 192)), None,
                              None, Qt.AlignTop | Qt.AlignRight]
        self.delta_style   = [QtGui.QPen(QtGui.QColor(192, 192, 192)), QtGui.QColor(48, 48, 48),
                              None, Qt.AlignTop | Qt.AlignRight]
        self.rows = []
        self.laps = []
        self.delta_idx = None

    def set_data(self, heading, rows, laps, delta_idx):
        self.rows = rows
        self.laps = laps
        self.delta_idx = delta_idx
        self.update_cursor()
        super().set_data(heading, len(rows))

    def update_cursor(self):
        self.laps = [(l, float(self.data_view.cursor2outTime(l))) for l, old_time in self.laps]

    def present(self, index):
        handler = self.rows[index.row()]
        col = index.column() - 2
        if col == -2: return handler.icon
        if col == -1: return handler.name
        if col == len(self.laps): return handler.units
        return handler.value(
            self.laps[col],
            None if self.delta_idx is None else self.laps[self.delta_idx])


class ValuesDockWidget(TempDockWidget):
    def __init__(self, mainwindow, toolbar):
        super().__init__('Values', mainwindow, toolbar, False)

        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self.text_changed)

        self.margin = 2

        self.model = ValuesTableModel(mainwindow.data_view)
        self.deleg = FastItemDelegate(self.model, self.margin)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegate(self.deleg)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setHorizontalScrollMode(self.table.ScrollPerPixel)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setMinimumSectionSize(10)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().sectionPressed.connect(self.section_pressed)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setMinimumSectionSize(5)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setEditTriggers(self.table.NoEditTriggers)
        self.table.activated.connect(self.activate_cell)
        self.table.customContextMenuRequested.connect(self.context_menu)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

        mainwindow.data_view.cursor_change.connect(self.update_cursor)
        mainwindow.data_view.values_change.connect(self.recompute)
        mainwindow.data_view.data_change.connect(self.recompute)

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.table)

        widget = QWidget()
        widget.setLayout(layout)
        self.setWidget(widget)

        pal = self.table.palette()
        pal.setColor(pal.Base, QtGui.QColor(0, 0, 0))
        self.table.setPalette(pal)

        self.text_hints = TextMatcher('')
        self.delta_mode = None

        self.recompute()

    def sizeHint(self):
        return QSize(400, 400)

    def context_menu(self, pos):
        row = self.model.rows[self.table.indexAt(pos).row()]
        if type(row) is ValuesTableChannel:
            ch = row.channel
            # channel specific context menu
            menu = QMenu()
            menu.addAction(ch) # dummy entry so the user knows exactly what we're operating on
            menu.addSeparator()
            menu.addAction('Edit channel...').triggered.connect(
                lambda: channels.channel_editor(self, self.mainwindow.data_view, ch))
            ac = self.mainwindow.data_view.active_component
            if ac:
                menu.addSeparator()
                act = menu.addAction('Remove channel' if ch in ac.channels() else 'Add channel')
                act.triggered.connect(lambda: ac.addChannel(ch))
            menu.exec_(self.mapToGlobal(pos))

    def activate_cell(self, index):
        ac = self.mainwindow.data_view.active_component
        if ac:
            row = self.model.rows[index.row()]
            if type(row) == ValuesTableChannel:
                ac.addChannel(row.channel)
        self.table.clearSelection()

    def section_pressed(self, idx):
        idx -= 2
        if idx >= 0 and idx < self.model.columnCount(None) - 3:
            self.delta_mode = None if idx == self.delta_mode else idx
        self.recompute()

    def text_changed(self, txt):
        self.text_hints = TextMatcher(txt)
        self.recompute()

    def update_cursor(self, old_cursor):
        self.model.update_cursor()
        # QAbstractItemView doesn't handle non singular dataChanged
        # events correctly.  So grab the underlying viewport and
        # update it manually.
        # start at row=1 to skip the section heading due to span issues
        self.table.viewport().update(
            QRect(self.table.visualRect(self.model.createIndex(1, 2)).topLeft(),
                  self.table.visualRect(
                      self.model.createIndex(self.model.row_count - 1,
                                             self.model.col_count - 2)).bottomRight()))

    def recompute(self):
        font = QtGui.QFont('Tahoma')
        font.setPixelSize(widgets.devicePointScale(self, 11.25))
        self.font_metrics = QtGui.QFontMetrics(font)
        section_font = QtGui.QFont(font)
        section_font.setBold(True)
        self.model.section_style[2] = section_font
        self.model.channel_style[2] = font
        self.model.value_style[2] = font
        self.model.delta_style[2] = font
        self.deleg.set_metrics(self.font_metrics)

        dv = self.mainwindow.data_view

        laps = [('R', dv.ref_lap), ('A', dv.alt_lap)] + [
            (chr(ord('3') + idx), lap) for idx, (lap, color) in enumerate(dv.extra_laps)]
        laps = [l for l in laps if l[1]] # filter out any unset laps (ref_lap, alt_lap?)

        self.table.clearSpans()
        if self.delta_mode is not None and self.delta_mode >= len(laps):
            self.delta_mode = None
        heading = ['', ''] + [l[0] + (' \u0394' if i == self.delta_mode else '') for i, l in enumerate(laps)] + ['']

        ac = self.mainwindow.data_view.active_component
        acset = set(ac.channels()) if ac else set()

        allch = list({ch
                      for logfile in self.mainwindow.data_view.log_files
                      for ch in logfile.log.get_channels()})
        allch.sort()

        rows = []
        rowhide = []

        # Cursor
        self.table.setSpan(len(rows), 1, 1, len(laps) + 2)
        rowhide.append(False)
        rows.append(ValuesTableSection(self.model, 'Cursor'))

        rowhide.append(False)
        rows.append(ValuesTableFuncTime(self.model, 'Time', dv.cursor2offTime, ''))

        rowhide.append(False)
        rows.append(ValuesTableFuncTime(self.model, 'Session Time', dv.cursor2outTime, ''))

        rowhide.append(False)
        rows.append(ValuesTableFunc(self.model, 'Dist', dv.cursor2offDist, 'm'))

        rowhide.append(False)
        rows.append(ValuesTableFunc(self.model, 'Session Dist', dv.cursor2outDist, 'm'))

        # Component
        self.table.setSpan(len(rows), 1, 1, len(laps) + 2)
        rowhide.append(False)
        rows.append(ValuesTableSection(self.model, 'Component/Graph'))

        for ch in sorted(list(acset)):
            rowhide.append(False)
            rows.append(ValuesTableChannel(True, self.model, dv, ch))

        # Channels
        self.table.setSpan(len(rows), 1, 1, len(laps) + 2)
        rowhide.append(False)
        rows.append(ValuesTableSection(self.model, 'Channels'))

        for ch in allch:
            rowhide.append(not self.text_hints.match(ch))
            rows.append(ValuesTableChannel(ch in acset, self.model, dv, ch))

        old_col_count = self.model.columnCount(None)
        self.model.set_data(heading, rows,
                            [(l[1], 0) for l in laps],
                            self.delta_mode)
        if old_col_count != self.model.columnCount(None):
            self.table.setColumnWidth(
                0, self.margin * 2 + self.font_metrics.horizontalAdvance('\u25bc'))
            self.table.setColumnWidth(
                1, self.margin * 2 + 1 + max([self.font_metrics.horizontalAdvance(ch)
                                              for ch in allch + ['Session Time', 'Session Dist']]))
            for c in range(2, self.model.columnCount(None) - 1):
                self.table.setColumnWidth(
                    c, self.margin * 2 + self.font_metrics.horizontalAdvance('+MMMM.MMM'))
            self.table.setColumnWidth(
                self.model.columnCount(None) - 1,
                self.margin * 2 + max([self.font_metrics.horizontalAdvance(r.units[1])
                                       for r in rows]))

        for row in range(self.model.rowCount(None)):
            self.table.setRowHeight(row, self.font_metrics.height())

        for row, hide in enumerate(rowhide):
            self.table.setRowHidden(row, hide)


class MapDockWidget(TempDockWidget):
    def __init__(self, mainwindow, toolbar):
        super().__init__('Map', mainwindow, toolbar, False)
        self.setWidget(map.MapWidget(mainwindow.data_view))

        # Need to simulate updates that MapWidget would receive if it were a component
        mainwindow.data_view.cursor_change.connect(self.update_cursor)
        mainwindow.data_view.values_change.connect(self.update)

    def update_cursor(self, old_cursor):
        self.update()
