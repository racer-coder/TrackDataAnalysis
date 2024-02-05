
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from collections import namedtuple
from dataclasses import dataclass, field
import os.path

from PySide2 import QtGui
from PySide2.QtCore import QPointF, QRect, QRectF, QSize, QSizeF, Qt
from PySide2.QtWidgets import (
    QPushButton,
    QStyle,
    QStylePainter,
    QStyleOptionButton,
    QWidget,
)

from . import state

def deviceScaleFactor(widget):
    # QT_SCALE_FACTOR -> widget.devicePixelRatioF()
    # QT_FONT_DPI -> widget.logicalDpiX() (and probably Y)
    # Windows: devicePixelRatioF == 1, logicalDpiX varies with Settings.Display.Scale * 96
    # Linux: ?? devicePixelRatioF == 1, logicalDpiX == 96
    # Mac: ?? devicePixelRatioF == 2?, logicalDpiX == ??
    return widget.devicePixelRatioF() * widget.logicalDpiX() / 96

def deviceScale(widget, sz):
    '''Scales to pixels'''
    return int(deviceScaleFactor(widget) * sz + 0.5)

def devicePointScale(widget, sz):
    '''Scales to points'''
    return int(sz * widget.logicalDpiX() / 96 + 0.5)

PaintHelper = namedtuple('PaintHelper', ['painter', 'rect', 'size', 'ratio', 'scale'])
def makePaintHelper(widget, paintEvent):
    ratio = widget.devicePixelRatioF() # note this is ONLY device ratio, not full scale
    painter = QtGui.QPainter(widget)
    painter.scale(1 / ratio, 1 / ratio)

    rect = QRectF(paintEvent.rect())
    rect = QRectF(rect.topLeft() * ratio, rect.bottomRight() * ratio)
    return PaintHelper(painter=painter,
                       rect=rect,
                       size=QSizeF(widget.geometry().size()) * ratio,
                       ratio=ratio,
                       scale=deviceScaleFactor(widget))

class RotatedPushButton(QPushButton):
    def paintEvent(self, event):
        styleopt = QStyleOptionButton()
        self.initStyleOption(styleopt)
        styleopt.rect = styleopt.rect.transposed()

        painter = QStylePainter(self)
        painter.rotate(-90)
        painter.translate(-self.height(), 0)
        painter.drawControl(QStyle.CE_PushButton, styleopt)

    # don't override minimumSizeHint, it just calls sizeHint()

    def sizeHint(self):
        size = super().sizeHint()
        size.transpose()
        return size

@dataclass
class MouseHelperClick:
    button_type: int = Qt.NoButton # Qt.MouseButton (Qt.LeftButton, Qt.RightButton)
    modifier_type: int = 0 # Qt.Modifier (Qt.SHIFT, Qt.CTRL, Qt.ALT)
    double: bool = False # If for double clicks
    state_capture: object = None # (abs_pos (QPointF))
    move: object = None # (delta_pos (QPointF), abs_pos (QPointF), state)
    release: object = None # (delta_pos (QPointF), abs_pos (QPointF), state)

@dataclass
class MouseHelperItem:
    geometry: QRectF = field(default_factory=QRectF) # QRectF
    cursor: int = Qt.ArrowCursor # CursorShape
    clicks: list = field(default_factory=list) # list of MouseHelperClick
    wheel: object = None # (angle_delta)
    data: dict = field(default_factory=dict) # user settable data for this region

    def __init__(self, geometry=None, cursor=Qt.ArrowCursor, clicks=None,
                 wheel=None, **kwargs):
        self.geometry = geometry or QRectF()
        self.cursor = cursor
        self.clicks = clicks or []
        self.wheel = wheel
        self.data = kwargs

class MouseHelperWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__mouseHelpers = [] # list of MouseHelperItem
        self.__lastLocalPos = QPointF(0, 0)
        self.__basePos = None # QPointF
        self.__curClick = None # MouseHelperClick
        self.__savedState = None # object
        self.setMouseTracking(True)

    def lookupCursor(self):
        for lst in self.__mouseHelpers:
            for obj in lst:
                if obj.geometry.contains(self.__lastLocalPos):
                    self.setCursor(QtGui.QCursor(obj.cursor))
                    return True
        self.setCursor(QtGui.QCursor(Qt.ArrowCursor))
        return False

    def addMouseHelperTop(self, mhi):
        if type(mhi) == MouseHelperItem: mhi = [mhi]
        self.__mouseHelpers.insert(0, mhi)
        self.lookupCursor()

    def addMouseHelperBottom(self, mhi):
        if type(mhi) == MouseHelperItem: mhi = [mhi]
        self.__mouseHelpers.append(mhi)
        self.lookupCursor()

    def getLastMouseHelperData(self, key, pixpos = None):
        for lst in self.__mouseHelpers:
            for obj in lst:
                if obj.geometry.contains(pixpos or self.__lastLocalPos) and key in obj.data:
                    return obj.data[key]
        return None

    def getEventMouseHelperData(self, key, pos): # position is from event, so not scaled
        return self.getLastMouseHelperData(key, pos * self.devicePixelRatioF())

    def __handleClick(self, e: QtGui.QMouseEvent, dbl: bool):
        self.__lastLocalPos = e.localPos() * self.devicePixelRatioF()
        if self.__curClick:
            e.accept()
            return True # still processing first click
        for lst in self.__mouseHelpers:
            for obj in lst:
                if obj.geometry.contains(self.__lastLocalPos) and obj.clicks:
                    for clk in obj.clicks:
                        if (clk.button_type == e.button() and
                            clk.modifier_type == e.modifiers() and
                            clk.double == dbl):
                            self.__basePos = e.globalPos() * self.devicePixelRatioF()
                            self.__curClick = clk
                            self.__savedState = clk.state_capture(self.__lastLocalPos) if clk.state_capture else None
                            QtGui.QGuiApplication.setOverrideCursor(self.cursor())
                            e.accept()
                            return True
        return False

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self.__handleClick(e, False): super().mousePressEvent(e)

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent):
        if not self.__handleClick(e, True): super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        self.__lastLocalPos = e.localPos() * self.devicePixelRatioF()
        if self.__curClick:
            if (self.__curClick.button_type == e.buttons() and
                self.__curClick.modifier_type == e.modifiers()):
                p = e.globalPos() * self.devicePixelRatioF()
                if self.__curClick.move:
                    self.__curClick.move(p - self.__basePos, self.__lastLocalPos,
                                         self.__savedState)
                e.accept()
                return
            if self.__curClick.release:
                self.__curClick.release(self.__lastLocalPos - self.__basePos,
                                        self.__lastLocalPos, self.__savedState)
            self.__curClick = None
            self.__savedState = None
            QtGui.QGuiApplication.restoreOverrideCursor()
        self.lookupCursor()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self.__lastLocalPos = e.localPos() * self.devicePixelRatioF()
        if self.__curClick:
            if self.__curClick.release:
                self.__curClick.release(self.__lastLocalPos - self.__basePos,
                                        self.__lastLocalPos, self.__savedState)
            self.__curClick = None
            self.__savedState = None
            QtGui.QGuiApplication.restoreOverrideCursor()
            self.lookupCursor()
            e.accept()
        else:
            super().mouseReleaseEvent(e)

    def wheelEvent(self, event):
        self.__lastLocalPos = event.position() * self.devicePixelRatioF()
        for lst in self.__mouseHelpers:
            for obj in lst:
                if obj.geometry.contains(self.__lastLocalPos) and obj.wheel:
                    obj.wheel(event.angleDelta())
                    event.accept()
                    return
        super().wheelEvent(event)

class LapWidget(MouseHelperWidget):
    def __init__(self, dataView):
        super().__init__()
        self.dataView = dataView
        self.scale = 1
        dataView.cursor_change.connect(self.updateCursor)
        dataView.values_change.connect(self.update)

        self.leftClick = MouseHelperItem(
            cursor = Qt.SizeHorCursor,
            clicks=[MouseHelperClick(Qt.LeftButton, move=self.leftDrag)])
        self.addMouseHelperBottom(self.leftClick)

        self.rightClick = MouseHelperItem(
            cursor = Qt.SizeHorCursor,
            clicks=[MouseHelperClick(Qt.LeftButton, move=self.rightDrag)])
        self.addMouseHelperBottom(self.rightClick)

        self.dragWindow = MouseHelperItem(
            cursor = Qt.OpenHandCursor,
            clicks=[MouseHelperClick(Qt.LeftButton, state_capture=self.windowDragCapture,
                                     move=self.windowDrag)])
        self.addMouseHelperBottom(self.dragWindow)

        self.dblClick = MouseHelperItem(
            clicks=[MouseHelperClick(Qt.LeftButton, double=True, state_capture=self.selectLap)])
        self.addMouseHelperBottom(self.dblClick)

    def leftDrag(self, rel_pos, abs_pos, saved_state):
        self.dataView.zoom_window = (
            self.dataView.makeTD(abs_pos.x() / self.scale - self.dataView.outTime2Mode(self.dataView.ref_lap, self.dataView.ref_lap.lap.start_time),
                                 False),
            self.dataView.zoom_window[1])
        self.dataView.values_change.emit()

    def rightDrag(self, rel_pos, abs_pos, saved_state):
        self.dataView.zoom_window = (
            self.dataView.zoom_window[0],
            self.dataView.makeTD(abs_pos.x() / self.scale - self.dataView.outTime2Mode(self.dataView.ref_lap, self.dataView.ref_lap.lap.end_time),
                                 True))
        self.dataView.values_change.emit()

    def windowDragCapture(self, abs_pos):
        return self.dataView.zoom_window

    def windowDrag(self, rel_pos, abs_pos, saved_state):
        self.dataView.zoom_window = (
            self.dataView.makeTD(rel_pos.x() / self.scale + self.dataView.getTDValue(saved_state[0]),
                                 False),
            self.dataView.makeTD(rel_pos.x() / self.scale + self.dataView.getTDValue(saved_state[1]),
                                 True))
        self.dataView.values_change.emit()

    def selectLap(self, abs_pos):
        tc = self.dataView.outMode2Time(self.dataView.ref_lap, abs_pos.x() / self.scale)
        for lap in self.dataView.ref_lap.log.laps:
            if tc >= lap.lap.start_time and tc <= lap.lap.end_time:
                self.dataView.ref_lap = lap
                self.dataView.zoom_window = (state.TimeDistRef(0, 0), state.TimeDistRef(0, 0))
                self.dataView.values_change.emit()
                break

    def getFont(self, big):
        font = QtGui.QFont('Tahoma')
        font.setPixelSize(deviceScale(self, 11.25))
        return font

    def updateCursor(self, old_cursor):
        self.update()

    def timeCalc(self, time):
        return round(self.dataView.outTime2Mode(self.dataView.ref_lap, time) * self.scale)

    def sizeHint(self):
        fontMetrics = QtGui.QFontMetrics(self.getFont(False))
        return QSize(200, 3 * fontMetrics.height() / self.devicePixelRatioF() + 2)

    def paintEvent(self, e):
        ph = makePaintHelper(self, e)
        font = self.getFont(False)
        bigfont = self.getFont(True)
        metrics = QtGui.QFontMetrics(font)
        fh = metrics.height()
        icon_width = QtGui.QFontMetrics(bigfont).horizontalAdvance(chr(0x278a))
        pen = QtGui.QPen(QtGui.QColor(192, 192, 192))
        ph.painter.setPen(pen)

        ph.painter.fillRect(0, 0, ph.size.width(), ph.size.height(),
                            QtGui.QColor(0, 0, 0))
        self.leftClick.geometry.setRect(0, 0, 0, 0)
        self.rightClick.geometry.setRect(0, 0, 0, 0)
        self.dragWindow.geometry.setRect(0, 0, 0, 0)
        self.dblClick.geometry.setRect(0, 0, 0, 0)
        lapy = 2 * fh
        ph.painter.drawRect(0, lapy, ph.size.width() - 1, ph.size.height() - 1 - lapy)
        if self.dataView.ref_lap:
            size = 0
            lapx = 4
            for pos, (l, c, idx) in enumerate(self.dataView.get_laps()):
                y = (pos & 1) * fh
                ph.painter.setFont(bigfont)
                ph.painter.setPen(c)
                ph.painter.drawText(lapx, y, ph.size.width(), fh,
                                    Qt.AlignTop | Qt.AlignLeft, chr(0x2789 + idx))

                ph.painter.setFont(font)
                ph.painter.setPen(pen)
                txt = '[%s] Lap %d [%s]' % (state.format_time(l.lap.duration()), l.lap.num,
                                            os.path.basename(l.log.log.get_filename()))
                size = max(size, metrics.horizontalAdvance(txt))
                ph.painter.drawText(lapx + icon_width, y, ph.size.width(), fh,
                                    Qt.AlignTop | Qt.AlignLeft, txt)
                if y:
                    lapx += icon_width + size + metrics.horizontalAdvance('MMMMM')

            duration = self.dataView.outTime2Mode(self.dataView.ref_lap,
                                                  self.dataView.ref_lap.log.laps[-1].lap.end_time)
            if not duration: # bug out early if no real data
                self.lookupCursor()
                return
            self.scale = ph.size.width () / duration
            # draw grey zone
            ph.painter.fillRect(
                QRect(1, 1 + lapy,
                      self.timeCalc(self.dataView.ref_lap.lap.start_time) - 1,
                      ph.size.height() - 2 - lapy),
                QtGui.QColor(48, 48, 48))
            ph.painter.fillRect(
                QRect(self.timeCalc(self.dataView.ref_lap.lap.end_time), 1 + lapy,
                      ph.size.width() - 1 - self.timeCalc(self.dataView.ref_lap.lap.end_time),
                      ph.size.height() - 2 - lapy),
                QtGui.QColor(48, 48, 48))
            # draw laps and lap markers
            for lap in self.dataView.ref_lap.log.laps:
                start_x = self.timeCalc(lap.lap.start_time)
                end_x = self.timeCalc(lap.lap.end_time)
                ph.painter.drawText(start_x, 1 + lapy, end_x - start_x, ph.size.height() - lapy,
                                    Qt.AlignTop | Qt.AlignHCenter | Qt.TextSingleLine,
                                    str(lap.lap.num))
            # draw lap boundaries
            pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
            pen.setStyle(Qt.DashLine)
            ph.painter.setPen(pen)
            for lap in self.dataView.ref_lap.log.laps:
                x = self.timeCalc(lap.lap.start_time)
                ph.painter.drawLine(x, lapy, x, ph.size.height())
                x = self.timeCalc(lap.lap.end_time)
                ph.painter.drawLine(x, lapy, x, ph.size.height())

            # draw cursor
            pen = QtGui.QPen(QtGui.QColor(255, 255, 0))
            pen.setStyle(Qt.SolidLine)
            ph.painter.setPen(pen)
            x = self.timeCalc(self.dataView.ref_lap.lap.start_time + self.dataView.cursor_time.time)
            ph.painter.drawLine(x, 1 + lapy, x, ph.size.height() - 2)

            # draw zoom window
            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen.setStyle(Qt.SolidLine)
            ph.painter.setPen(pen)
            x1 = self.timeCalc(self.dataView.ref_lap.lap.start_time + self.dataView.zoom_window[0].time)
            self.leftClick.geometry.setRect(x1 - 3, lapy, 7, ph.size.height() - lapy)
            x2 = self.timeCalc(self.dataView.ref_lap.lap.end_time + self.dataView.zoom_window[1].time)
            self.rightClick.geometry.setRect(x2 - 3, lapy, 7, ph.size.height() - lapy)
            ph.painter.drawRect(x1, 1 + lapy, x2 - x1, ph.size.height() - 2 - lapy)

            self.dragWindow.geometry.setCoords(x1, lapy, x2, ph.size.height())
            self.dblClick.geometry.setCoords(0, lapy, ph.size.width(), ph.size.height())
        self.lookupCursor()
