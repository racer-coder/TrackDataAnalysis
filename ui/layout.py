
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from dataclasses import dataclass
import typing

from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QTabBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

@dataclass(eq=False)
class Worksheet:
    name: str
    mode_time: bool
    components: typing.List

    def save_state(self):
        return self.__dict__

    @classmethod
    def load_state(cls, data):
        return Worksheet(**data)

@dataclass(eq=False)
class Workbook:
    name: str
    last_sheet: Worksheet
    sheets: typing.List[Worksheet]

    def save_state(self):
        return {'name': self.name,
                'last': self.sheets.index(self.last_sheet),
                'sheets': [ws.save_state() for ws in self.sheets]}

    @classmethod
    def load_state(cls, data):
        sheets = [Worksheet.load_state(ws) for ws in data['sheets']]
        return Workbook(data['name'],
                        sheets[data['last']],
                        sheets)

class LayoutTree(QTreeWidget):
    reordered = Signal()

    def setItemDropEnabled(self, i, e):
        i.setFlags((i.flags() & ~Qt.ItemIsDropEnabled) | (Qt.ItemIsDropEnabled if e else 0))

    def dragEnterEvent(self, e):
        if len(self.selectedItems()) == 1:
            is_wb = type(self.selectedItems()[0].data(0, Qt.UserRole)) is Workbook
            self.setItemDropEnabled(self.invisibleRootItem(), is_wb)
            for i in range(self.topLevelItemCount()):
                self.setItemDropEnabled(self.topLevelItem(i), not is_wb)
            super().dragEnterEvent(e)

    def dropEvent(self, e):
        super().dropEvent(e)
        for i in range(self.topLevelItemCount()):
            self.topLevelItem(i).setExpanded(True) # not sure why this keeps getting reset
        self.reordered.emit()

class LayoutEditor(QDialog):
    worktypemap = {Workbook: 'workbook', Worksheet: 'worksheet'}

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Layout Editor')
        layout = QGridLayout()

        self.tree = LayoutTree()
        self.tree.setColumnCount(1)
        self.tree.setItemsExpandable(False)
        self.tree.setDragDropMode(self.tree.InternalMove)
        self.tree.setDefaultDropAction(Qt.IgnoreAction)
        self.tree.setEditTriggers(self.tree.DoubleClicked | self.tree.SelectedClicked |
                                  self.tree.EditKeyPressed)
        self.tree.header().hide()
        current = None
        for wb in parent.workspace:
            maybe_current, _ = self.insertWorkbook(wb)
            current = current or maybe_current
        if current:
            self.tree.setCurrentItem(current)
        self.tree.itemSelectionChanged.connect(self.selectionChanged)
        self.tree.itemChanged.connect(self.itemChanged)
        self.tree.reordered.connect(self.reordered)
        layout.addWidget(self.tree, 0, 0, 4, 1)

        rename_b = QPushButton('Rename')
        rename_b.clicked.connect(self.rename)
        layout.addWidget(rename_b, 0, 1, 1, 1)

        delete_b = QPushButton('Delete')
        delete_b.clicked.connect(self.deleteWorkitem)
        layout.addWidget(delete_b, 1, 1, 1, 1)

        addbook_b = QPushButton('New Workbook')
        addbook_b.clicked.connect(self.addWorkbook)
        layout.addWidget(addbook_b, 2, 1, 1, 1)

        addsheet_b = QPushButton('New Worksheet')
        addsheet_b.clicked.connect(self.addWorksheet)
        layout.addWidget(addsheet_b, 3, 1, 1, 1)

        dlgbutton = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dlgbutton.accepted.connect(self.accept)
        dlgbutton.rejected.connect(self.reject)
        layout.addWidget(dlgbutton, 4, 0, 1, 2)

        for i in range(4):
            layout.setRowStretch(i, 1)
        layout.setRowStretch(4, 0)

        self.setLayout(layout)

    def insertWorksheet(self, wbitem, ws):
        shitem = QTreeWidgetItem(wbitem)
        shitem.setFlags((shitem.flags() | Qt.ItemIsEditable) & ~Qt.ItemIsDropEnabled)
        shitem.setText(0, ws.name)
        shitem.setData(0, Qt.UserRole, ws)
        return shitem

    def insertWorkbook(self, wb):
        current = None
        wbitem = QTreeWidgetItem(self.tree)
        wbitem.setFlags(wbitem.flags() | Qt.ItemIsEditable)
        wbitem.setExpanded(True)
        wbitem.setText(0, wb.name)
        wbitem.setData(0, Qt.UserRole, wb)
        for sh in wb.sheets:
            shitem = self.insertWorksheet(wbitem, sh)
            if sh == self.parent().current_sheet:
                current = shitem
        return current, shitem

    def selectionChanged(self):
        sel = self.tree.selectedItems()
        if len(sel) != 1: return
        sel = sel[0].data(0, Qt.UserRole)
        self.parent().selectSheet(sel if type(sel) == Worksheet else sel.last_sheet)

    def reordered(self):
        parent = self.parent()
        parent.workspace = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            wb = it.data(0, Qt.UserRole)
            wb.sheets = [it.child(j).data(0, Qt.UserRole) for j in range(it.childCount())]
            if not wb.sheets: continue # skip empty Workbooks
            if wb.last_sheet not in wb.sheets:
                wb.last_sheet = wb.sheets[0]
            parent.workspace.append(wb)
        parent.workspaceUpdated()

    def itemChanged(self, it, c):
        if it.data(0, Qt.UserRole):
            it.data(0, Qt.UserRole).name = it.text(0) # propogate name change to underlying workspace
            self.parent().workspaceUpdated() # update tabbar/combo box as needed

    def addWorkbook(self):
        new_name, ok = QInputDialog.getText(self.parent(), 'New Workbook',
                                            'Enter a name for the new workbook',
                                            text='Workbook %d' % self.tree.topLevelItemCount())
        if ok and new_name:
            ws=Worksheet('Worksheet', True, [])
            wb=Workbook(new_name, ws, [ws])
            _, it = self.insertWorkbook(wb)
            self.reordered()
            self.tree.setCurrentItem(it)

    def addWorksheet(self):
        sel = self.tree.selectedItems()
        if len(sel) != 1: return
        sel = sel[0]
        while sel.parent():
            sel = sel.parent()
        new_name, ok = QInputDialog.getText(
            self.parent(), 'New Worksheet',
            'Enter a name for the new worksheet under book "%s"' % sel.text(0),
            text='Worksheet %d' % sel.childCount())
        if ok and new_name:
            ws = Worksheet(new_name, True, [])
            it = self.insertWorksheet(sel, ws)
            self.reordered()
            self.tree.setCurrentItem(it)

    def deleteWorkitem(self):
        sel = self.tree.selectedItems()
        if len(sel) != 1: return
        sel = sel[0]
        ret = QMessageBox.warning(self.parent(), 'Warning',
                                  'Are you sure you want to delete %s "%s?"' %
                                  (self.worktypemap[type(sel.data(0, Qt.UserRole))],
                                   sel.text(0)),
                                  QMessageBox.Yes | QMessageBox.No,
                                  QMessageBox.No)
        if ret == QMessageBox.Yes:
            sel.parent().takeChild(sel.parent().indexOfChild(sel))
            self.reordered() # big hammer

    def rename(self):
        sel = self.tree.selectedItems()
        if len(sel) != 1: return
        sel = sel[0]
        t = self.worktypemap[type(sel.data(0, Qt.UserRole))]
        new_name, ok = QInputDialog.getText(self.parent(), 'Rename',
                                            'Enter a new name for %s "%s"' % (t, sel.text(0)))
        if ok and new_name:
            sel.setText(0, new_name)
            sel.data(0, Qt.UserRole).name = new_name
            self.parent().workspaceUpdated()


class LayoutManager(QWidget):
    def __init__(self, measures, data_view, layout_menu):
        super().__init__()

        self.measures = measures
        self.data_view = data_view

        self.workbook_selector = QComboBox()
        self.workbook_selector.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.workbook_selector.activated.connect(self.comboActivated)
        self.workbook_selector.currentIndexChanged.connect(self.comboChange)
        self.current_book = None
        self.current_sheet = None
        self.tabbar = QTabBar()
        self.tabbar.setMovable(True)
        self.tabbar.tabMoved.connect(self.tabMoved)
        self.tabbar.tabBarClicked.connect(self.tabClicked)
        self.tabbar.currentChanged.connect(self.tabSelected)

        self.new_layout()

        tabbar_more = QToolButton()
        tabbar_more.setText('+')
        tabbar_more.clicked.connect(self.newWorksheet)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        hbox.addWidget(self.workbook_selector)
        hbox.addWidget(self.tabbar)
        hbox.addWidget(tabbar_more)
        hbox.addWidget(QWidget(), 1)

        layout_menu.addAction('Layout Editor...').triggered.connect(self.layoutEditor)
        layout_menu.addSeparator()
        layout_menu.addAction('New Workbook').triggered.connect(self.newWorkbook)
        layout_menu.addAction('New Worksheet').triggered.connect(self.newWorksheet)

        self.setLayout(hbox)

    def new_layout(self):
        ws = Worksheet('Worksheet', True, [])
        wb = Workbook('Workbook', ws, [ws])
        self.workspace = [wb]
        self.last_book = wb
        self.populateComboBox()

    def layoutEditor(self):
        self.saveCurrentTab()
        dlg = LayoutEditor(self)
        dlg.exec_()
        # XXX cancel button for dialog to revert to previous design

    def tabMoved(self, t, f):
        self.updateWorkbook()

    def tabClicked(self, idx):
        new_sheet = self.tabbar.tabData(idx)
        if new_sheet:
            self.current_book.last_sheet = new_sheet

    def tabSelected(self, idx):
        # don't update self.current_book.last_sheet here, this routine
        # gets called on workbook change.
        new_sheet = self.tabbar.tabData(idx)
        if new_sheet != self.current_sheet: # can be same if following a move
            self.saveCurrentTab()
            self.current_sheet = new_sheet
            self.loadCurrentTab()

    def selectSheet(self, sheet):
        if sheet in self.current_book.sheets:
            self.current_book.last_sheet = sheet
            self.tabbar.setCurrentIndex(self.current_book.sheets.index(sheet))
        else:
            for wb in self.workspace:
                if sheet in wb.sheets:
                    wb.last_sheet = sheet
                    self.last_book = wb
                    self.workbook_selector.setCurrentIndex(self.workspace.index(wb))
                    break

    def populateComboBox(self):
        while self.workbook_selector.count():
            self.workbook_selector.removeItem(0)
        self.workbook_selector.addItems([w.name for w in self.workspace])
        if self.last_book in self.workspace:
            self.workbook_selector.setCurrentIndex(self.workspace.index(self.last_book))

    # notification from LayoutEditor
    def workspaceUpdated(self):
        # update combobox
        self.populateComboBox()

    def populateTabBar(self):
        # remove old contents
        for i in range(self.tabbar.count()):
            self.tabbar.setTabData(i, None) # to prevent craziness when removing tabs
        while self.tabbar.count():
            self.tabbar.removeTab(0)
        # add new contents
        for sheet in self.current_book.sheets:
            idx = self.tabbar.addTab(sheet.name)
            self.tabbar.setTabData(idx, sheet)
        # properly select last sheet
        self.tabbar.setCurrentIndex(self.current_book.sheets.index(self.current_book.last_sheet))
        self.tabSelected(self.tabbar.currentIndex()) # in case it was 0

    def comboActivated(self, idx):
        self.last_book = self.workspace[idx]

    def comboChange(self, idx):
        self.current_book = self.workspace[idx]
        self.populateTabBar()

    def updateWorkbook(self):
        self.current_book.sheets = [self.tabbar.tabData(i) for i in range(self.tabbar.count())]

    def newWorksheet(self, *args):
        name = 'Worksheet %d' % self.tabbar.count()
        worksheet = Worksheet(name=name, mode_time=self.data_view.mode_time, components=[])
        idx = self.tabbar.addTab(name)
        self.tabbar.setTabData(idx, worksheet)
        self.last_book.last_sheet = worksheet
        self.updateWorkbook()
        self.tabbar.setCurrentIndex(idx)

    def newWorkbook(self, *args):
        ws = Worksheet('Worksheet', True, [])
        wb = Workbook('Workbook %d' % len(self.workspace), ws, [ws])
        self.workspace.append(wb)
        self.last_book = wb
        self.workbook_selector.addItem(wb.name)
        self.workbook_selector.setCurrentIndex(len(self.workspace) - 1)

    def saveCurrentTab(self):
        data = self.current_sheet
        if data:
            data.mode_time = self.data_view.mode_time
            data.components = self.measures.save_state()

    def loadCurrentTab(self):
        data = self.current_sheet
        if data:
            self.data_view.mode_time = data.mode_time
            self.measures.load_state(data.components)
            self.data_view.values_change.emit() # time/dist mode change
        else:
            self.data_view.mode_time = True
            self.measures.load_state([])
            self.data_view.values_change.emit()

    def save_state(self):
        self.saveCurrentTab()
        return {'last': self.workspace.index(self.last_book),
                'books': [wb.save_state() for wb in self.workspace]}

    def load_state(self, data):
        # Scan first so if we crash, last_book is still kinda correct
        ws = [Workbook.load_state(wb) for wb in data['books']]
        self.workspace, self.last_book = ws, ws[data['last']]
        self.populateComboBox()
