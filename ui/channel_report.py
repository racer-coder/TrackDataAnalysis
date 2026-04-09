
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

"""Channel report: min/max/avg statistics table for channels."""

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import state


class ChannelReport(QWidget):
    """Component showing a table of channel statistics (min, max, avg, range)."""

    COLUMNS = ['Channel', 'Units', 'Min', 'Max', 'Avg', 'Range']

    def __init__(self, data_view, st=None):
        super().__init__()
        self.data_view = data_view
        self.channel_list = list(st['channels']) if st and 'channels' in st else []
        self.setMinimumSize(200, 100)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        # Dark styling
        self.table.setStyleSheet(
            "QTableWidget { background-color: #0c0c0c; color: #ddd; "
            "gridline-color: #333; }"
            "QHeaderView::section { background-color: #1a1a1a; color: #bbb; "
            "border: 1px solid #333; padding: 3px; }"
            "QTableWidget::item { padding: 2px 6px; }"
            "QTableWidget::item:selected { background-color: #333; }")

        layout.addWidget(self.table)

        data_view.values_change.connect(self._rebuild)
        self._rebuild()

    def save_state(self):
        return {
            'type': 'channel_report',
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
        self._rebuild()

    def updateCursor(self, old_cursor):
        pass

    def _rebuild(self):
        dv = self.data_view
        self.table.setRowCount(0)

        if not dv.ref_lap or not self.channel_list:
            return

        rows = []
        for ch in self.channel_list:
            try:
                cd = dv.ref_lap.get_channel_data(ch, None, dv.maths)
            except Exception:
                continue
            if cd is None or not hasattr(cd, 'values') or len(cd.values) == 0:
                continue

            values = cd.values[np.isfinite(cd.values)]
            if len(values) == 0:
                continue

            units = cd.units if cd.units else ''
            vmin = float(np.min(values))
            vmax = float(np.max(values))
            vavg = float(np.mean(values))
            vrange = vmax - vmin

            dec = cd.dec_pts if hasattr(cd, 'dec_pts') else 2
            fmt = f'{{:.{dec}f}}'

            rows.append((ch, units,
                         fmt.format(vmin), fmt.format(vmax),
                         fmt.format(vavg), fmt.format(vrange)))

        self.table.setRowCount(len(rows))
        for r, (ch, units, smin, smax, savg, srange) in enumerate(rows):
            for c, text in enumerate([ch, units, smin, smax, savg, srange]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    if c < 2 else
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
