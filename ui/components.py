
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import enum

from PySide2 import QtGui
from PySide2.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide2.QtWidgets import (
    QAction,
    QVBoxLayout,
    QWidget,
)

from . import widgets
from . import timedist
from . import video

class ComponentManager(QWidget):
    factory = {
        'timedist': timedist.TimeDist,
        'video': video.Video,
    }

    def __init__(self, dataView, addMenu):
        super().__init__()
        self.dataView = dataView
        dataView.cursor_change.connect(self.updateCursor)
        dataView.values_change.connect(self.updateValues)

        addMenu.addAction('Time/Distance Graph').triggered.connect(self.newTDGraph)
        addMenu.addAction('Session Graph').triggered.connect(self.newSessionGraph)
        addMenu.addAction('Video').triggered.connect(self.newVideo)

        act = QAction('Paste', self)
        act.triggered.connect(self.paste_component)
        self.addAction(act)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        self.component_clipboard = None

    def newTDGraph(self):
        ComponentBase(self, None, self.dataView,
                      timedist.TimeDist(self.dataView, True))

    def newSessionGraph(self):
        ComponentBase(self, None, self.dataView,
                      timedist.TimeDist(self.dataView, False))

    def newVideo(self):
        ComponentBase(self, None, self.dataView, video.Video(self.dataView))

    def paintEvent(self, e: QtGui.QPaintEvent):
        ph = widgets.makePaintHelper(self, e)
        x1, y1, x2, y2 = ph.rect.getCoords()
        sz = int(100 * widgets.deviceScaleFactor(self))
        for cx in range(int(x1 / sz), int(x2 / sz) + 1):
            for cy in range(int(y1 / sz), int(y2 / sz) + 1):
                bright = 0 if (cx ^ cy) & 1 else 24
                ph.painter.fillRect(QRect(cx * sz, cy * sz, sz, sz),
                                    QtGui.QColor(bright, bright, bright))

    def resizeLambda(self):
        return lambda p: QPointF(p.x() * self.size().width(),
                                 p.y() * self.size().height())

    def invertLambda(self):
        return lambda p: QPointF(p.x() / self.size().width(),
                                 p.y() / self.size().height())

    def resizeEvent(self, event):
        m = self.resizeLambda()
        for cb in self.findChildren(ComponentBase):
            cb.parentResize(m)

    def updateCursor(self, old_cursor):
        for cb in self.findChildren(ComponentBase):
            cb.childWidget.updateCursor(old_cursor)

    def updateValues(self):
        # manually update cursor of videos.  I can't get signals to
        # work properly without crashing, so....
        for cb in self.findChildren(ComponentBase):
            if type(cb.childWidget) is video.Video:
                cb.childWidget.updateCursor(None)
            else:
                cb.update()

    def save_state(self):
        return [cb.childWidget.save_state() for cb in self.findChildren(ComponentBase)]

    def load_state(self, state):
        # delete all widgets
        self.dataView.active_component = None
        for cb in self.findChildren(ComponentBase):
            cb.setParent(None) # immediately remove from list of children, so save_state won't find it
            cb.deleteLater()
        # install new widgest
        for widg in state:
            ComponentBase(self, widg['base'], self.dataView,
                          self.factory[widg['type']](self.dataView, state=widg))
        if not state:
            # need to emit this due to change in active_component/focus
            self.dataView.data_change.emit()

    def cut_component(self, cb):
        self.copy_component(cb)
        self.dataView.active_component = None
        cb.setParent(None)
        cb.deleteLater()

    def copy_component(self, cb):
        self.component_clipboard = cb.childWidget.save_state()

    def paste_component(self):
        widg = self.component_clipboard
        if widg:
            ComponentBase(self, widg['base'], self.dataView,
                          self.factory[widg['type']](self.dataView, state=widg))

class ResizerMode(enum.Flag):
    MOVE = 0
    TOP = 1
    BOTTOM = 2
    LEFT = 4
    RIGHT = 8


class ComponentBase(QWidget):
    """
    created (translated to pyqt) by Aleksandr Korabelnikov (nesoriti@yandex.ru)
    origin was written in c++ by Aleksey Osipov (aliks-os@yandex.ru)
    wiki: https://wiki.qt.io/Widget-moveable-and-resizeable

    distributed without any warranty. Code bellow can contains mistakes taken from c++ version
    and/or created by my own

    allow to move and resize by user"""
    menu = None
    mode = ResizerMode.MOVE
    position = None
    inFocus = Signal(bool)
    outFocus = Signal(bool)

    def __init__(self, parent, g, data_view, cWidget):
        super().__init__(parent=parent)

        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setVisible(True)
        self.setAutoFillBackground(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        if g:
            # g[0] -> maximize?
            self.fracGeometry = QRectF(g[1], g[2], g[3], g[4])
        else:
            self.fracGeometry = QRectF(0., 0., 0.75, 0.75)

        self.data_view = data_view
        self.vLayout = QVBoxLayout(self)
        self.vLayout.setContentsMargins(4, 4, 4, 4)

        self.m_infocus = False
        self.m_isEditing = True
        self.pressedGeometry = None

        self.parentResize(parent.resizeLambda())

        self.setChildWidget(cWidget)
        self.setFocus()


    def save_state(self):
        return (False, self.fracGeometry.x(), self.fracGeometry.y(),
                self.fracGeometry.width(), self.fracGeometry.height())

    def setChildWidget(self, cWidget):
        if cWidget:
            act = QAction('', cWidget)
            act.setSeparator(True)
            cWidget.addAction(act)

            act = QAction('Cut', cWidget)
            act.triggered.connect(lambda: self.parent().cut_component(self))
            cWidget.addAction(act)

            act = QAction('Copy', cWidget)
            act.triggered.connect(lambda: self.parent().copy_component(self))
            cWidget.addAction(act)

            act = QAction('Paste', cWidget)
            act.triggered.connect(self.parent().paste_component)
            cWidget.addAction(act)

            self.childWidget = cWidget
            self.childWidget.setMouseTracking(True)
            self.vLayout.addWidget(cWidget)

    def parentResize(self, m):
        self.setGeometry(QRectF(m(self.fracGeometry.topLeft()),
                                m(self.fracGeometry.bottomRight())).toRect())

    def saveGeometry(self):
        m = self.parentWidget().invertLambda()
        geof = QRectF(self.geometry())
        self.fracGeometry = QRectF(m(geof.topLeft()),
                                   m(geof.bottomRight()))

    def focusInEvent(self, event):
        self.data_view.active_component = self.childWidget
        self.m_infocus = True
        self.raise_()
        self.update()
        self.inFocus.emit(True)
        self.data_view.data_change.emit()

    def focusOutEvent(self, event):
        if not self.m_isEditing:
            return
        self.mode = ResizerMode.MOVE
        self.outFocus.emit(False)
        self.m_infocus = False
        self.update()

    def paintEvent(self, e: QtGui.QPaintEvent):
        ph = widgets.makePaintHelper(self, e)

        box = QRectF(QPoint(0, 0), ph.size)

        ph.painter.fillRect(box, QtGui.QColor(0, 0, 0))

        pen_width = int(ph.scale + 0.5)
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0)
                         if self.m_infocus else QtGui.QColor(64, 64, 64))
        pen.setWidth(pen_width)
        rect = box.adjusted(pen_width / 2, pen_width / 2, -pen_width/2, -pen_width/2)
        ph.painter.setPen(pen)
        ph.painter.drawRect(rect)

    #def keyPressEvent(self, e: QtGui.QKeyEvent):
    #    if not self.m_isEditing: return
    #    if e.key() == Qt.Key_Delete:
    #        self.deleteLater()
    #    # Moving container with arrows
    #    elif QApplication.keyboardModifiers() == Qt.ControlModifier:
    #        if   e.key() == Qt.Key_Up:    self.move(self.x(),     self.y() - 1)
    #        elif e.key() == Qt.Key_Down:  self.move(self.x(),     self.y() + 1)
    #        elif e.key() == Qt.Key_Left:  self.move(self.x() - 1, self.y())
    #        elif e.key() == Qt.Key_Right: self.move(self.x() + 1, self.y())
    #        else: return
    #    elif QApplication.keyboardModifiers() == Qt.ShiftModifier:
    #        if   e.key() == Qt.Key_Up:    self.resize(self.width(),     self.height() - 1)
    #        elif e.key() == Qt.Key_Down:  self.resize(self.width(),     self.height() + 1)
    #        elif e.key() == Qt.Key_Left:  self.resize(self.width() - 1, self.height())
    #        elif e.key() == Qt.Key_Right: self.resize(self.width() + 1, self.height())
    #        else: return
    #    else:
    #        return
    #    self.saveGeometry()

    def setCursorShape(self, e_pos: QPoint):
        diff = 4

        flags = ResizerMode.MOVE

        if e_pos.y() < diff:                  flags |= ResizerMode.TOP
        if e_pos.y() >= self.height() - diff: flags |= ResizerMode.BOTTOM
        if e_pos.x() < diff:                  flags |= ResizerMode.LEFT
        if e_pos.x() >= self.width() - diff:  flags |= ResizerMode.RIGHT

        if not flags:
            self.setCursor(QtGui.QCursor(Qt.ArrowCursor))
        elif flags == ResizerMode.LEFT or flags == ResizerMode.RIGHT:
            self.setCursor(QtGui.QCursor(Qt.SizeHorCursor))
        elif flags == ResizerMode.TOP or flags == ResizerMode.BOTTOM:
            self.setCursor(QtGui.QCursor(Qt.SizeVerCursor))
        elif bool(flags & ResizerMode.TOP) == bool(flags & ResizerMode.LEFT):
            self.setCursor(QtGui.QCursor(Qt.SizeFDiagCursor))
        else:
            self.setCursor(QtGui.QCursor(Qt.SizeBDiagCursor))
        self.mode = flags

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.pressedGeometry = self.geometry().translated(-e.globalPos())
        if self.m_isEditing and e.button() == Qt.LeftButton:
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self.pressedGeometry = None
        super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if not self.m_isEditing:
            super().mouseMoveEvent(e)
            return
        if not (e.buttons() & Qt.LeftButton) or not self.pressedGeometry:
            self.setCursorShape(e.pos())
            e.accept()
            return

        if self.mode == ResizerMode.MOVE:
            toMove = e.globalPos() + self.pressedGeometry.topLeft()
            self.move(
                QPoint(min(max(toMove.x(), 0), self.parentWidget().width() - self.width()),
                       min(max(toMove.y(), 0), self.parentWidget().height() - self.height())))
        else:
            newGeo = self.geometry()
            minSize = self.minimumSizeHint()
            if self.mode & ResizerMode.TOP:
                newGeo.setTop(max(min(e.globalPos().y() + self.pressedGeometry.top(),
                                      newGeo.bottom() - minSize.height()),
                                  0))
            if self.mode & ResizerMode.BOTTOM:
                newGeo.setBottom(min(max(e.globalPos().y() + self.pressedGeometry.bottom(),
                                         newGeo.top() + minSize.height()),
                                     self.parentWidget().height()))
            if self.mode & ResizerMode.LEFT:
                newGeo.setLeft(max(min(e.globalPos().x() + self.pressedGeometry.left(),
                                       newGeo.right() - minSize.width()),
                                   0))
            if self.mode & ResizerMode.RIGHT:
                newGeo.setRight(min(max(e.globalPos().x() + self.pressedGeometry.right(),
                                        newGeo.left() + minSize.width()),
                                    self.parentWidget().width()))
            m = self.parentWidget().invertLambda()
            self.fracGeometry = QRectF(m(QPointF(newGeo.topLeft())),
                                       m(QPointF(newGeo.bottomRight())))
            self.setGeometry(newGeo)
        self.saveGeometry()
        e.accept()
