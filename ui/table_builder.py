
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import configparser

import numpy as np

from PySide2 import QtGui
from PySide2.QtCore import QSize, Qt, Signal
from PySide2.QtWidgets import (
    QAction,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .dockers import TextMatcher # move to general tools?
from . import widgets

class ChannelSelect(QDialog):
    def __init__(self, data_view):
        super().__init__()
        self.setWindowTitle('Channel selection')
        self.data_view = data_view

        # Largely copied from dockers.py / ChannelsDockWidget
        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.edit.setPlaceholderText('Channel search')
        self.edit.textChanged.connect(self.text_changed)

        self.chlist = QListWidget()
        self.chlist.itemActivated.connect(self.accept)
        self.chlist.addItems(sorted(data_view.channel_properties.keys()))

        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)

        vbox = QVBoxLayout()
        vbox.addWidget(self.edit)
        vbox.addWidget(self.chlist)
        vbox.addWidget(bbox)

        self.setLayout(vbox)
        try:
            self.restoreGeometry(
                bytes.fromhex(data_view.config.get('main', 'channel_select_geometry')))
        except configparser.NoOptionError:
            pass

    def hideEvent(self, ev):
        self.data_view.config['main']['channel_select_geometry'] = bytes(
            self.saveGeometry()).hex()
        super().hideEvent(ev)

    def text_changed(self, txt):
        matcher = TextMatcher(txt)
        for i in range(self.chlist.count()):
            it = self.chlist.item(i)
            hide = not matcher.match(it.text())
            if hide != it.isHidden():
                it.setHidden(hide)

class ChannelEdit(QLineEdit):
    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view
        self.setReadOnly(True)

        icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        action = self.addAction(icon, self.TrailingPosition)
        action.triggered.connect(self.selected)

    def selected(self):
        dia = ChannelSelect(self.data_view)
        if dia.exec_():
            items = dia.chlist.selectedItems()
            if len(items) == 1:
                self.setText(items[0].text())

class TBAxis(QGroupBox):
    update = Signal()
    def __init__(self, title, data_view, state):
        super().__init__(title)

        self.ch = ChannelEdit(data_view)
        self.ch.setText(state.get('channel', ''))
        self.spacing = QLineEdit()
        self.spacing.setText(state.get('spacing', ''))
        self.how_close = QLineEdit()
        self.how_close.setText(state.get('how_close', ''))

        form = QFormLayout()
        form.addRow('Channel', self.ch)
        form.addRow('Spacing', self.spacing)
        form.addRow('+/- %', self.how_close)

        self.setLayout(form)
        self.setCheckable(True)
        self.setChecked(state.get('enabled?', True))

        self.toggled.connect(self.update)
        self.ch.textChanged.connect(self.update)
        self.spacing.textChanged.connect(self.update)
        self.how_close.textChanged.connect(self.update)

    def save_state(self):
        return {'enabled?': self.isChecked(),
                'channel': self.ch.text(),
                'spacing': self.spacing.text(),
                'how_close': self.how_close.text()}

    def recompute(self, data_view, timecodes):
        if not self.isChecked():
            return (np.zeros(timecodes.shape),
                    np.full(timecodes.shape, True))
        values = data_view.get_channel_data(data_view.ref_lap, self.ch.text())
        if not values or len(values.timecodes) == 0:
            return ('No data for axis channel', None)
        values = values.interp_many(timecodes)
        try:
            spacing = float(self.spacing.text())
        except ValueError:
            return ('Cannot parse axis spacing', None)
        nearest = np.round(values / spacing) * spacing
        try:
            keep = np.abs((values - nearest) / spacing) < float(self.how_close.text()) / 100
        except ValueError:
            return ('Cannot parse axis closeness', None)
        return (nearest, keep)


class TableBuilder(QWidget):
    def __init__(self, data_view, state={}):
        super().__init__()
        self.data_view = data_view

        self.x_box = TBAxis('X Axis', data_view, state.get('x_axis', {}))
        self.x_box.update.connect(self.recompute)
        self.y_box = TBAxis('Y Axis', data_view, state.get('y_axis', {}))
        self.y_box.update.connect(self.recompute)

        v_box = QGroupBox('Value')
        form = QFormLayout()
        self.value_ch = ChannelEdit(data_view)
        self.value_ch.setText(state.get('value', ''))
        self.value_ch.textChanged.connect(self.recompute)
        form.addRow('Channel', self.value_ch)
        v_box.setLayout(form)

        self.f_box = QGroupBox('Filter')
        self.f_box.setCheckable(True)
        self.f_box.setChecked(state.get('filter?', False))
        self.f_box.toggled.connect(self.recompute)
        form = QFormLayout()
        self.filter_ch = ChannelEdit(data_view)
        self.filter_ch.setText(state.get('filter', ''))
        self.filter_ch.textChanged.connect(self.recompute)
        form.addRow('Channel', self.filter_ch)
        self.f_box.setLayout(form)

        self.table = QTableWidget()
        self.table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.table.setAlternatingRowColors(True)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.x_box)
        v_layout.addWidget(self.y_box)
        v_layout.addWidget(v_box)
        v_layout.addWidget(self.f_box)

        h_layout = QHBoxLayout()
        h_layout.addLayout(v_layout, stretch=1)
        h_layout.addWidget(self.table, stretch=3)
        self.setLayout(h_layout)

        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        data_view.values_change.connect(self.recompute)
        self.recompute()

    def save_state(self):
        return {'type': 'table_builder',
                'base': self.parentWidget().save_state(),
                'x_axis': self.x_box.save_state(),
                'y_axis': self.y_box.save_state(),
                'value': self.value_ch.text(),
                'filter?': self.f_box.isChecked(),
                'filter': self.filter_ch.text(),
                }

    def channels(self):
        return set()

    def updateCursor(self, old_cursor):
        pass

    def paintEvent(self, event):
        ph = widgets.makePaintHelper(self, event)
        ph.painter.fillRect(ph.rect,
                            QtGui.QColor(224, 224, 224))

    def clear_table(self, reason):
        print('CLEAR TABLE:', reason)
        pass

    def recompute(self):
        if not self.data_view.ref_lap:
            return self.clear_table('No ref lap')
        # load values (time and data)
        if not self.value_ch.text():
            return self.clear_table('No values channel')
        values = self.data_view.get_channel_data(self.data_view.ref_lap, self.value_ch.text())
        if not values or len(values.timecodes) == 0:
            return self.clear_table('No values data')

        timecodes = values.timecodes
        values = values.values

        # Check filter.  Skip data filtered out
        if self.f_box.isChecked():
            if not self.filter_ch.text():
                return self.clear_table('No filter channel specified')
            filt = self.data_view.get_channel_data(self.data_view.ref_lap,
                                                   self.filter_ch.text())
            if not filt or len(filt.timecodes) == 0:
                return self.clear_table('No filter data')
            filt = filt.interp_many(timecodes) != 0
        else:
            filt = np.full(timecodes.shape, True)

        # Add X axis data.  Skip data not near cell
        x_axis, keep = self.x_box.recompute(self.data_view, timecodes)
        if isinstance(x_axis, str):
            return self.clear_table(x_axis)
        filt = filt & keep

        # Add Y axis data.  Skip data not near cell
        y_axis, keep = self.y_box.recompute(self.data_view, timecodes)
        if isinstance(y_axis, str):
            return self.clear_table(y_axis)
        filt = filt & keep

        # Filter data using all restrictions
        values = np.column_stack([x_axis, y_axis, values])[filt]
        values = values[np.argsort(values[:,1], kind='stable')]
        values = values[np.argsort(values[:,0], kind='stable')]

        sep = values[1:, :2] != values[:-1, :2]
        sep = np.nonzero(np.logical_or(sep[:, 0], sep[:, 1]))[0]
        sep = np.concatenate([[0], sep + 1])
        key = values[sep, :2]
        tot = np.add.reduceat(values[:, 2], sep)
        cnt = np.add.reduceat(np.ones(len(values)), sep)

        res = np.column_stack([key, tot / cnt, cnt])

        # Update table with data
        self.table.clear()

        # Add x axis
        xaxis = np.unique(res[:, 0])
        self.table.setColumnCount(len(xaxis))
        # XXX need decpts from axis
        self.table.setHorizontalHeaderLabels(['%.3f' % v for v in xaxis])
        xaxis = {xaxis[i]: i for i in range(len(xaxis))}

        # Add y axis
        yaxis = np.unique(res[:, 1])
        self.table.setRowCount(len(yaxis))
        # XXX need decpts from axis
        self.table.setVerticalHeaderLabels(['%.3f' % v for v in yaxis])
        yaxis = {yaxis[i]: i for i in range(len(yaxis))}

        # Iterate over res, setting text
        for row in res:
            item = QTableWidgetItem('%.3f\n%d samples' % (row[2], row[3]))
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.table.setItem(yaxis[row[1]], xaxis[row[0]], item)

        self.table.horizontalHeader().resizeSections(QHeaderView.ResizeToContents)
        self.table.verticalHeader().resizeSections(QHeaderView.ResizeToContents)
