
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import configparser
from dataclasses import dataclass
import importlib.abc
import importlib.machinery
import os
import sys
import types

from PySide2.QtCore import QAbstractItemModel, QFileSystemWatcher, QMimeData, QModelIndex, Qt
from PySide2.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat
from PySide2.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTreeView,
)

import numpy as np

from data import math_eval, unitconv
from . import channels
from . import state

class Highlighter(QSyntaxHighlighter):
    def __init__(self, doc, data_view, error_label):
        super().__init__(doc)
        self.data_view = data_view
        self.error_label = error_label
        self.lex = math_eval.ExprLex()

    def maybe_format(self, fmt, start, end):
        block = self.currentBlock()
        start = max(start - block.position(), 0)
        end = min(end - block.position(), block.length())
        if end > start:
            self.setFormat(start, end - start, fmt)

    def highlightBlock(self, text):
        err_format = QTextCharFormat()
        err_format.setBackground(QColor(255, 160, 160))
        comment_format = QTextCharFormat()
        comment_format.setFontItalic(True)
        comment_format.setForeground(QColor(96, 96, 96))
        var_format = QTextCharFormat()
        var_format.setForeground(QColor(0, 0, 224))
        var_format.setFontWeight(75)
        unit_format = QTextCharFormat()
        unit_format.setForeground(QColor(64, 64, 64,))
        unit_format.setFontWeight(75)
        lit_format = QTextCharFormat()
        lit_format.setForeground(QColor(192, 0, 0))
        error_value = 0
        err_status = 'Ok'
        try:
            all_text = self.document().toPlainText()
            # base formatting
            for tok in self.lex.tokenize(all_text):
                if tok.type == 'COMMENT':
                    self.maybe_format(comment_format, tok.index, tok.end)
                if tok.type == 'VAR':
                    self.maybe_format(var_format, tok.index, tok.end)
                if tok.type == 'UNIT':
                    self.maybe_format(unit_format, tok.index, tok.end)
                if tok.type == 'INT' or tok.type == 'FLOAT':
                    self.maybe_format(lit_format, tok.index, tok.end)
            # does it parse?  if not, we'll error out and highlight
            p = math_eval.compile(all_text)
            # great it parses.  but do we know all the variables?  Not fatal, but alert the user...
            for tok in self.lex.tokenize(all_text):
                if tok.type == 'VAR':
                    if tok.value[1:-1] not in self.data_view.channel_properties:
                        self.maybe_format(err_format, tok.index, tok.end)
                        err_status = 'Unknown variable(s) (maybe not available in these log files?)'
            if err_status == 'Ok':
                # look for improper units
                for tok in self.lex.tokenize(all_text):
                    if tok.type == 'UNIT':
                        if tok.value[1:-1] not in unitconv.unit_map:
                            self.maybe_format(err_format, tok.index, tok.end)
                            err_status = 'Unknown units'
            if err_status == 'Ok' and self.data_view.ref_lap:
                try:
                    err_status = 'Ok.  Value at cursor: %f' % (
                        p.values(self.data_view.ref_lap.log,
                                 np.array([self.data_view.cursor2outTime(self.data_view.ref_lap)]))[0])
                except BaseException as err:
                    err_status = 'Parses correctly, but unable to evaluate (%s).' % err
        except math_eval.LexError as err:
            self.maybe_format(err_format, err.error_index, len(all_text))
            error_value = err.error_index + (2 << 24)
            err_status = err.args[0]
        except math_eval.ParseError as err:
            if err.token:
                self.maybe_format(err_format, err.token.index, err.token.end)
                error_value = err.token.index + (3 << 24)
            else:
                error_value = len(all_text) + (3 << 24)
            err_status = err.args[0]
        self.error_label.setText(err_status)
        self.setCurrentBlockState(error_value)

class ExpressionEditor(QDialog):
    def __init__(self, parent, data_view, base=None):
        super().__init__(parent)
        self.setWindowTitle('Expression Editor')
        self.config = data_view.config
        self.old_expr = base

        grid = QGridLayout()

        layout = QFormLayout()

        self.name_edit = QLineEdit(base.name if base else '')
        layout.addRow('Name', self.name_edit)

        self.enabled = QCheckBox('')
        self.enabled.setChecked(base.enabled if base else True)
        layout.addRow('Enabled', self.enabled)

        self.unit_edit = QLineEdit(base.unit if base else '') # XXX should be dropdown
        layout.addRow('Units', self.unit_edit)

        self.decpts_edit = QLineEdit(str(base.dec_pts) if base else '0')
        layout.addRow('Decimal places', self.decpts_edit)

        self.interpolate = QCheckBox('')
        self.interpolate.setChecked(base.interpolate if base else True)
        layout.addRow('Interpolate', self.interpolate)

        self.sample_rate = QComboBox()
        self.sample_rate.addItem('auto', 0)
        for rate in (1, 2, 5, 10, 20, 25, 50, 100):
            self.sample_rate.addItem('%d Hz' % rate, rate)
        self.sample_rate.setCurrentIndex(
            self.sample_rate.findData(base.sample_rate if base else 0))
        layout.addRow('Sample rate', self.sample_rate)

        self.color_edit = QComboBox()
        channels.add_channel_colors(self.color_edit)
        self.color_edit.setCurrentIndex(base.color if base else 0)
        layout.addRow('Color', self.color_edit)

        self.expr_unit_edit = QLineEdit(base.expr_unit if base else '') # XXX another dropdown
        layout.addRow('Expression units', self.expr_unit_edit)

        grid.addLayout(layout, 0, 0, 2, 1)

        error_label = QLabel()
        error_label.setWordWrap(True)
        grid.addWidget(error_label, 1, 1, 1, 1)

        self.expression_edit = QPlainTextEdit(base.expression if base else '')
        self.highlighter = Highlighter(self.expression_edit.document(), data_view, error_label)
        grid.addWidget(self.expression_edit, 0, 1, 1, 1)

        dlgbutton = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dlgbutton.accepted.connect(self.validate)
        dlgbutton.rejected.connect(self.reject)
        grid.addWidget(dlgbutton, 2, 0, 2, 2)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        self.setLayout(grid)
        try:
            self.restoreGeometry(
                bytes.fromhex(self.config.get('main', 'expressioneditor_geometry')))
        except configparser.NoOptionError:
            pass

    def hideEvent(self, ev):
        self.config['main']['expressioneditor_geometry'] = bytes(self.saveGeometry()).hex()
        super().hideEvent(ev)

    def validate(self):
        enabled = self.enabled.isChecked()
        name = self.name_edit.text()
        if not name:
            QMessageBox.warning(self, 'Error', 'Please enter a name for the expression',
                                QMessageBox.Ok)
            return
        units = self.unit_edit.text()
        try:
            dec_pts = int(self.decpts_edit.text())
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Decimal points must be an integer',
                                QMessageBox.Ok)
            return
        interpolate = self.interpolate.isChecked()
        sample_rate = self.sample_rate.currentData()
        color = self.color_edit.currentData()
        expr_unit = self.expr_unit_edit.text()
        expression = self.expression_edit.toPlainText()
        try:
            math_eval.compile(expression)
        except (math_eval.LexError, math_eval.ParseError):
            if QMessageBox.No == QMessageBox.warning(
                    self, 'Error', 'Unable to parse expression; save as incomplete?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
                return
        if not self.old_expr:
            self.old_expr = state.MathExpr('', False, '', 0, False, 0, 0, '', '', '')
        self.old_expr.name = name
        self.old_expr.enabled = enabled
        self.old_expr.unit = units
        self.old_expr.dec_pts = dec_pts
        self.old_expr.interpolate = interpolate
        self.old_expr.color = color
        self.old_expr.sample_rate = sample_rate
        self.old_expr.expr_unit = expr_unit
        self.old_expr.expression = expression
        self.new_expr = self.old_expr
        self.accept()

@dataclass
class IndexDetails:
    name: str
    obj: object
    ordered_src: list[object]
    key: object # either str or int, key into src_obj
    src_obj: object # list[object] or dict[str, object]
    parent: object

class MathTreeModel(QAbstractItemModel):
    def __init__(self, data_view):
        super().__init__()
        self.data_view = data_view

    def child(self, index):
        if index.isValid():
            obj = index.internalPointer()
            if isinstance(obj, state.Maths):
                src = sorted(obj.groups.keys())
                k = src[index.row()]
                return IndexDetails(k, obj.groups[k], src, k, obj.groups, obj) # MathGroup
            if isinstance(obj, state.MathGroup):
                src = obj.expressions
                k = index.row()
                return IndexDetails(src[k].name, src[k], src, k, src, obj) # MathExpr
        return IndexDetails(None, self.data_view.maths, [], None, {}, None)

    def data(self, index, role):
        child = self.child(index)
        if role == Qt.DisplayRole or role == Qt.EditRole:
            if index.column() == 0:
                return child.name
            if index.column() == 1:
                if isinstance(child.obj, state.MathExpr):
                    return child.obj.expression.replace('\n', ' ')
            if index.column() == 2:
                return child.obj.comment.replace('\n', ' ')
        if role == Qt.CheckStateRole and index.column() == 0:
            return Qt.Checked if child.obj.enabled else Qt.Unchecked

    def setData(self, index, value, role):
        child = self.child(index)
        if index.column() == 0:
            if role == Qt.EditRole:
                if isinstance(child.src_obj, dict):
                    if value in child.src_obj: # whether no name change or whether renaming to a dup
                        return False
                    self.layoutAboutToBeChanged.emit()
                    child.src_obj[value] = child.obj
                    del child.src_obj[child.name]
                    self.layoutChanged.emit()
                else:
                    child.obj.name = value
                    self.dataChanged.emit(index, index)
                redo_math(self.data_view) # reordering groups can cause many effects
                return True
            if role == Qt.CheckStateRole:
                child.obj.enabled = bool(value)
                self.dataChanged.emit(index, index, [role])
                redo_math(self.data_view)
                return True
        return super().setData(index, value, role)

    def flags(self, index):
        if not index.isValid():
            return super().flags(index)
        f = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() == 0:
            f |= Qt.ItemIsUserCheckable
            child = self.child(index)
            if isinstance(child.obj, state.MathExpr):
                f |= Qt.ItemIsDragEnabled
            else:
                f |= Qt.ItemIsDropEnabled | Qt.ItemIsEditable
        return f

    HEADINGS = ('Name', 'Summary', 'Comment')
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADINGS[section]
        return None

    def index(self, row, col, parent):
        if not self.hasIndex(row, col, parent):
            return QModelIndex()
        child = self.child(parent)
        return self.createIndex(row, col, child.obj)

    def parent(self, index):
        if not index.isValid(): return index
        parent = index.internalPointer()
        if isinstance(parent, state.Maths):
            return QModelIndex()
        if isinstance(parent, state.MathGroup):
            k = [v for _, v in sorted(self.data_view.maths.groups.items())]
            return self.createIndex(k.index(parent), 0, self.data_view.maths)
        return QModelIndex() # should never get here

    def rowCount(self, parent):
        parent = self.child(parent).obj
        if isinstance(parent, state.Maths):
            return len(parent.groups)
        if isinstance(parent, state.MathGroup):
            return len(parent.expressions)
        return 0

    def columnCount(self, parent):
        return 3

    def supportedDropActions(self):
        return Qt.MoveAction

    def mimeTypes(self):
        return ['text/plain']

    def mimeData(self, indices):
        mime = QMimeData()
        index = indices[0]
        mime.setText('%d/%d' % (self.parent(index).row(), index.row()))
        return mime

    def decode_mime(self, data, parent):
        if not parent.isValid():
            return None
        if not isinstance(parent.internalPointer(), state.Maths):
            return None
        if not data.hasText():
            return None
        p = data.text().split('/')
        if len(p) != 2:
            return None
        try:
            pi0 = int(p[0])
            pi1 = int(p[1])
        except ValueError:
            return None
        return (sorted(self.data_view.maths.groups.keys())[pi0], pi1)

    def canDropMimeData(self, data, action, row, column, parent):
        return bool(self.decode_mime(data, parent))

    def dropMimeData(self, data, action, row, column, parent):
        drop_pos = self.decode_mime(data, parent)
        if not drop_pos:
            return False

        src_group = self.data_view.maths.groups[drop_pos[0]]
        src_row = drop_pos[1]

        # 1. Drop between expressions row=expression row, index.row() = group index, index.ip=maths
        # 2. Drop on a group: row==-1, index.row() = group index, index.ip = maths
        dst_group = parent.internalPointer().groups
        dst_group = dst_group[sorted(dst_group.keys())[min(parent.row(), len(dst_group) - 1)]]
        dst_row = max(row, 0)

        self.layoutAboutToBeChanged.emit()
        child = src_group.expressions[src_row]
        src_group.expressions[src_row] = None
        dst_group.expressions.insert(dst_row, child)
        src_group.expressions.remove(None)
        self.layoutChanged.emit()
        redo_math(self.data_view)
        return True

class MathEditor(QDialog):
    def __init__(self, parent, data_view):
        super().__init__(parent)
        self.data_view = data_view

        self.setWindowTitle('Math Editor')
        layout = QGridLayout()

        self.tree_view = QTreeView()
        self.tree_model = MathTreeModel(data_view)
        self.tree_view.setModel(self.tree_model)
        self.tree_view.expandAll()
        self.tree_view.setDragDropMode(self.tree_view.InternalMove)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.activated.connect(self.activated_item)
        layout.addWidget(self.tree_view, 0, 0, 6, 1)

        but = QPushButton('New Group')
        but.clicked.connect(self.create_group)
        layout.addWidget(but, 0, 1, 1, 1)

        but = QPushButton('New Expression')
        but.clicked.connect(self.create_expr)
        layout.addWidget(but, 1, 1, 1, 1)

        but = QPushButton('Edit')
        but.clicked.connect(self.edit_something)
        layout.addWidget(but, 2, 1, 1, 1)

        but = QPushButton('Comment')
        but.clicked.connect(self.comment_something)
        layout.addWidget(but, 3, 1, 1, 1)

        but = QPushButton('Delete')
        but.clicked.connect(self.delete_something)
        layout.addWidget(but, 4, 1, 1, 1)

        dlgbutton = QDialogButtonBox(QDialogButtonBox.Ok)
        dlgbutton.accepted.connect(self.accept)
        layout.addWidget(dlgbutton, 5, 1, 1, 1)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)

        self.setLayout(layout)
        try:
            self.restoreGeometry(
                bytes.fromhex(self.data_view.config.get('main', 'matheditor_geometry')))
            self.tree_view.header().restoreState(
                bytes.fromhex(self.data_view.config.get('main', 'matheditor_header')))
        except configparser.NoOptionError:
            pass

    def hideEvent(self, ev):
        self.data_view.config['main']['matheditor_geometry'] = bytes(self.saveGeometry()).hex()
        self.data_view.config['main']['matheditor_header'] = bytes(
            self.tree_view.header().saveState()).hex()
        super().hideEvent(ev)

    def create_group(self):
        new_name, ok = QInputDialog.getText(self, 'New Math Group',
                                            'Enter a name for the new math group')
        if ok and new_name and new_name not in self.data_view.maths.groups:
            self.tree_model.layoutAboutToBeChanged.emit()
            self.data_view.maths.groups[new_name] = state.MathGroup(True, [], '')
            self.tree_model.layoutChanged.emit()
            self.tree_view.setExpanded(
                self.tree_model.createIndex(
                    sorted(self.data_view.maths.groups.keys()).index(new_name), 0,
                    self.data_view.maths),
                True)

    def get_single_index(self):
        ind = [i for i in self.tree_view.selectedIndexes() if i.column() == 0]
        return ind[0] if len(ind) == 1 else None

    def get_single_child(self):
        index = self.get_single_index()
        return self.tree_model.child(index) if index else None

    def comment_something(self):
        child = self.get_single_child()
        if not child:
            return
        new_comment, ok = QInputDialog.getMultiLineText(self, 'Comment',
                                                        'Enter a comment for "%s"' % child.name,
                                                        child.obj.comment)
        if ok:
            child.obj.comment = new_comment
            indexes = self.tree_view.selectedIndexes()
            self.tree_model.dataChanged.emit(indexes[0], indexes[-1])

    def edit_something(self):
        self.activated_item(self.get_single_index())

    def activated_item(self, index):
        if not index:
            return
        child = self.tree_model.child(index)
        if isinstance(child.obj, state.MathGroup):
            new_name, ok = QInputDialog.getText(
                self, 'Rename Math Group', 'Enter a name for the new math group "%s"' % child.name,
                text=child.name)
            if ok and new_name and new_name != child.name:
                if new_name in child.src_obj:
                    QMessageBox.warning(self, 'Conflict', 'That name is already in use',
                                        QMessageBox.Ok)
                else:
                    self.tree_model.layoutAboutToBeChanged.emit()
                    del child.src_obj[child.name]
                    child.src_obj[new_name] = child.obj
                    self.tree_model.layoutChanged.emit()
        else:
            dlg = ExpressionEditor(self, self.data_view, child.obj)
            try:
                if dlg.exec_():
                    self.tree_model.layoutAboutToBeChanged.emit()
                    self.tree_model.layoutChanged.emit()
            finally:
                dlg.deleteLater()
        redo_math(self.data_view)

    def delete_something(self):
        child = self.get_single_child()
        if child and QMessageBox.Yes == QMessageBox.warning(
                self, 'Warning',
                'Are you sure you want to delete %s "%s?"'
                % ('group' if isinstance(child.obj, state.MathGroup) else 'expression',
                   child.name),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
            self.tree_model.layoutAboutToBeChanged.emit()
            if isinstance(child.src_obj, dict):
                del child.src_obj[child.name]
            else:
                child.src_obj.remove(child.obj)
            self.tree_model.layoutChanged.emit()
            redo_math(self.data_view)

    def create_expr(self):
        child = self.get_single_child()
        if not child:
            if not self.data_view.maths.groups.keys():
                self.create_group()
                if not self.data_view.maths.groups.keys():
                    return
            group = QInputDialog.getItem(self, 'Math Group', 'Select a group to receive the new expression',
                                         sorted(self.data_view.maths.groups.keys()),
                                         editable=False)
            if not group[1] or not group[0]:
                return
            group = self.data_view.maths.groups[group[0]]
        elif isinstance(child.obj, state.MathGroup):
            group = child.obj
        elif isinstance(child.parent, state.MathGroup):
            group = child.parent
        else:
            return # ??
        dlg = ExpressionEditor(self, self.data_view)
        try:
            if dlg.exec_():
                self.tree_model.layoutAboutToBeChanged.emit()
                group.expressions.append(dlg.new_expr)
                self.tree_model.layoutChanged.emit()
                redo_math(self.data_view)
        finally:
            dlg.deleteLater()

def math_editor(parent, data_view):
    dlg = MathEditor(parent, data_view)
    try:
        dlg.exec_()
    finally:
        dlg.deleteLater()

def redo_math(data_view):
    data_view.math_invalidate()
    channels.update_channel_properties(data_view)
    data_view.values_change.emit()
    data_view.data_change.emit()

def channel_editor(parent, data_view, channel):
    if channel in data_view.maths.channel_map:
        dlg = ExpressionEditor(parent, data_view, data_view.maths.channel_map[channel][0])
        try:
            if dlg.exec_():
                redo_math(data_view)
        finally:
            dlg.deleteLater()
    else:
        channels.channel_editor(parent, data_view, channel)

class MathModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        super().__init__()
        self.path = None
        dummy_module = types.ModuleType('usermathfunc')
        dummy_module.__path__ = []
        sys.modules['usermathfunc'] = dummy_module

    def set_path(self, path):
        self.path = path

    def find_spec(self, fullname, path, target=None):
        if self.path and fullname.startswith('usermathfunc'):
            return importlib.machinery.ModuleSpec(
                fullname, importlib.machinery.SourceFileLoader(
                    fullname, os.path.join(self.path, *fullname.split('.')[1:]) + '.py'))
        return None

def set_user_func_dir(data_view, dir_name):
    if not data_view.maths.watcher:
        data_view.maths.watcher = QFileSystemWatcher()
        data_view.maths.watcher.directoryChanged.connect(lambda: clear_user_module(data_view))
    data_view.maths.watcher.removePaths(data_view.maths.watcher.directories())
    if dir_name:
        data_view.maths.watcher.addPath(dir_name)
    if not data_view.maths.finder:
        data_view.maths.finder = MathModuleFinder()
        sys.meta_path.append(data_view.maths.finder)
    data_view.maths.finder.set_path(dir_name)

def clear_user_module(data_view):
    to_del = [k for k in sys.modules.keys() if k.startswith('usermathfunc.')]
    for k in to_del:
        del sys.modules[k]
    data_view.math_invalidate()
    data_view.values_change.emit()
