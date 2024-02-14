
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import ctypes
import locale
import math
import os
import platform

import glfw
from PySide2 import QtGui
from PySide2.QtCore import QSize, Qt, Signal
from PySide2.QtWidgets import (
    QAction,
    QFileDialog,
    QGridLayout,
    QOpenGLWidget,
    QWidget,
)

# Might need to add current dir to path to find mpv dll/so
os.environ['PATH'] = os.path.dirname(__file__) + os.pathsep + os.environ['PATH']
from . import mpv
from . import widgets
from .timedist import roundUpHumanNumber, AxisGrid


class GetProcAddressGetter:
    """This wrapper class is necessary because the required function
    pointers were only exposed from Qt 6.5 onwards
    https://bugreports.qt.io/browse/PYSIDE-971
    """

    def __init__(self):
        self.surface = QtGui.QOffscreenSurface()
        self.surface.create()

        if not glfw.init():
            raise AssertionError('Cannot initialize OpenGL')

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(1, 1, "tda-OpenGL", None, None)

        glfw.make_context_current(window)
        QtGui.QOpenGLContext.currentContext().makeCurrent(self.surface)

    def wrap(self, _, name: bytes):
        return ctypes.cast(glfw.get_proc_address(name.decode('utf8')), ctypes.c_void_p).value


#class Video(QVideoWidget):
class OneVideo(QOpenGLWidget):
    onUpdate = Signal()
    mpv_result = Signal(int, object, object) # mechanism to queue responses back to GUI thread

    def __init__(self, data_view, secondary):
        super().__init__()
        self.setMouseTracking(True) # Normally set by ComponentBase
        self.data_view = data_view
        self.secondary = secondary
        self.lname = None
        self.vname = None

        #self.media_player = QMediaPlayer()
        #self.media_player.setVideoOutput(self)
        #self.media_player.setMedia(QUrl.fromLocalFile(self.fname))

        locale.setlocale(locale.LC_NUMERIC, 'C')
        self._get_proc_address_resolver = None
        extra_args = {}
        if platform.system() == 'Windows':
            # for some reason, wasapi stutters on unpause on Windows
            # 11; openal / SDL seem better
            extra_args['ao'] = 'openal'
        self.player = mpv.MPV(vo='libmpv',
                              pause=True,
                              aid='no' if secondary else 'auto',
                              keep_open='always',
                              **extra_args)
        self.onUpdate.connect(self.process_update)
        self.ctx = None

        self.mpv_result.connect(self.process_result, Qt.QueuedConnection)
        self.async_outstanding = 0
        self.last_seek_exact = True
        self.last_seek_time = 0
        self.last_pos_recv = 0
        self.player.observe_property('time-pos', self.emit_time)
        self.player.observe_property('seeking', self.emit_seeking)
        self.control_cursor = 0
        self.frame_step_wait_seek = False
        self.playback_rate = 1 # for secondary

    def emit_time(self, name, newval): self.mpv_result.emit(0, self.update_time, newval)
    def emit_seeking(self, name, newval): self.mpv_result.emit(0, self.update_seeking, newval)

    def process_update(self):
        if self.ctx.update():
            self.update() #??

    def done_seeking(self):
        self.update_time(self.player.time_pos) # catch any last update
        # XXX if secondary, we should probably realign ourselves with cursor time....
        self.frame_step_wait_seek = False

    def update_seeking(self, val):
        if not val and self.frame_step_wait_seek:
            self.done_seeking()

    def seek_cmd_done(self, fut):
        self.frame_step_wait_seek = True
        self.control_cursor -= 1 # disable interpretation of any more updates
        if not self.player.seeking:
            self.done_seeking()

    def play_cb(self):
        self.control_cursor += 1
        if self.secondary: self.playback_rate = 1 # reset adaptation
        # Make this async for consistency, probably also helps when running two videos at once
        self.mpv_command_async(None, 'set', 'pause', 'no')

    def pause_cb(self):
        self.mpv_command_async(self.seek_cmd_done, 'set', 'pause', 'yes')

    def next_frame(self):
        self.control_cursor += 1
        self.mpv_command_async(self.seek_cmd_done, 'frame-step')

    def prev_frame(self):
        self.control_cursor += 1
        self.mpv_command_async(self.seek_cmd_done, 'frame-back-step')

    def process_result(self, cnt, func, val):
        self.async_outstanding -= cnt
        if func: func(val)
        if self.async_outstanding == 0:
            self.async_idle()

    def mpv_command_async(self, callback, *args):
        fut = self.player.command_async(*args)
        self.async_outstanding += 1
        # following may call back synchronously
        fut.add_done_callback(lambda f: self.mpv_result.emit(1, callback, f))

    def async_idle(self):
        # Called when there are no more seeks outstanding.  Do one
        # final 'exact' seek if the last seek was not exact.
        if not self.last_seek_exact:
            self.updateCursor(None)

    def addChannel(self, ch):
        pass # not supported

    def updateCursor(self, old_cursor):
        if self.control_cursor or self.frame_step_wait_seek or not self.ctx: return
        lap = self.data_view.alt_lap if self.secondary else self.data_view.ref_lap
        if not lap: return
        if lap.log.log.get_filename() != self.lname:
            # Try to infer video filename.  Use self.lname to gate trying more than once.
            self.lname = lap.log.log.get_filename()
            if self.lname not in self.data_view.video_alignment:
                base = os.path.splitext(lap.log.log.get_filename())[0]
                vfile = None
                for ext in ('.mp4', '.mov'):
                    if os.path.exists(base + ext):
                        vfile = base + ext
                        break
                if not vfile: return
                self.data_view.video_alignment[self.lname] = (vfile, 0)
        if not self.vname or lap.log.video_file != self.vname:
            if not lap.log.video_file:
                if self.lname not in self.data_view.video_alignment: return
                lap.log.video_file, lap.log.video_alignment = self.data_view.video_alignment[self.lname]
            self.player.play(lap.log.video_file)
            self.player.wait_for_property('seekable', timeout=0.5)
            self.vname = lap.log.video_file
        seek_time = max(0, lap.log.video_alignment + self.data_view.cursor2outTime(lap)) / 1000
        if (abs(seek_time - self.last_seek_time) < 0.002 and
            (self.async_outstanding or self.last_seek_exact)):
            return
        self.last_seek_exact = (self.async_outstanding == 0
                                or (seek_time >= self.last_pos_recv and
                                    seek_time < self.last_pos_recv + 1))
        self.last_seek_time = seek_time
        # self.media_player.setPosition(int(self.last_seek_time + 0.5))
        self.mpv_command_async(None, 'seek', self.last_seek_time, 'absolute',
                               'exact' if self.last_seek_exact else 'keyframes')

    def update_time(self, newt):
        self.last_pos_recv = newt
        if self.control_cursor or self.frame_step_wait_seek:
            if not self.secondary:
                old_cursor = self.data_view.cursor_time
                self.data_view.cursor_time = self.data_view.outTime2cursor(
                    self.data_view.ref_lap, newt * 1000 - self.data_view.ref_lap.log.video_alignment)
                self.data_view.cursor_change.emit(old_cursor)
            else:
                # feedback loop to keep video playback in sync
                catchup_time = 0.1
                cursor_time = self.data_view.cursor_time.time
                # how far will ref lap advance in the next catchup_time?
                catchup_mode = (self.data_view.offTime2outMode(self.data_view.ref_lap,
                                                               cursor_time + catchup_time * 1000) -
                                self.data_view.offTime2outMode(self.data_view.ref_lap,
                                                               cursor_time))
                # what should our local video time be in catchup_time given what ref_lap is doing
                vtime = (self.data_view.alt_lap.log.video_alignment +
                         self.data_view.offMode2outTime(
                             self.data_view.alt_lap,
                             self.data_view.getTDValue(self.data_view.cursor_time) + catchup_mode))
                delta = vtime / 1000 - newt # in seconds
                newscale = delta / catchup_time
                newscale = 0.75 * self.playback_rate + 0.25 * newscale # dampen oscillations
                self.playback_rate = newscale
                newscale = min(newscale, 2)
                self.mpv_command_async(None, 'set', 'speed', newscale)
            self.last_seek_time = newt
            self.last_seek_exact = True

    def initializeGL(self):
        if self._get_proc_address_resolver is None:
            self._get_proc_address_resolver = mpv.MpvGlGetProcAddressFn(GetProcAddressGetter().wrap)
        self.ctx = mpv.MpvRenderContext(
            self.player, 'opengl',
            opengl_init_params={'get_proc_address': self._get_proc_address_resolver})
        self.ctx.update_cb = self.onUpdate.emit
        self.updateCursor(None)

    def paintGL(self):
        pixel_size = self.devicePixelRatioF() * self.size()
        self.ctx.render(flip_y=True,
                        opengl_fbo={'w': pixel_size.width(), 'h': pixel_size.height(),
                                    'fbo': self.defaultFramebufferObject()})


class AlignmentSlider(widgets.MouseHelperWidget):
    def __init__(self, data_view):
        super().__init__()
        self.setMouseTracking(True) # Normally set by ComponentBase

        self.data_view = data_view
        self.zoom_window = 60000
        self.xaxis_click = widgets.MouseHelperItem(
            cursor=Qt.OpenHandCursor,
            clicks=[widgets.MouseHelperClick(Qt.LeftButton,
                                             state_capture=self.xaxis_capture,
                                             move=self.xaxis_drag)])
        self.addMouseHelperTop(self.xaxis_click)

    def sizeHint(self):
        metrics = QtGui.QFontMetrics(self.select_font())
        return QSize(200, 2 * (4 + metrics.height()))

    def xaxis_capture(self, absPos):
        return self.data_view.ref_lap.log.video_alignment

    def xaxis_drag(self, relPos, absPos, orig_align):
        rel = self.x_axis.invertRelative(relPos.x())
        log = self.data_view.ref_lap.log
        log.video_alignment = orig_align - rel
        self.data_view.video_alignment[log.log.get_filename()] = (log.video_file, log.video_alignment)
        self.data_view.cursor_change.emit(None)

    def select_font(self):
        font = QtGui.QFont('Tahoma')
        font.setPixelSize(widgets.deviceScale(self, 11.25))
        return font

    def wheelEvent(self, event):
        self.zoom_window *= 2 ** (event.angleDelta().y() / 540)
        event.accept()
        self.update()

    def paintEvent(self, event):
        # shamelessly copied from TimeDist paintXAxis
        ph = widgets.makePaintHelper(self, event)
        self.xaxis_click.geometry.setRect(0, 0, ph.size.width(), ph.size.height())
        session_time = self.data_view.cursor2outTime(self.data_view.ref_lap)
        zero_offset = session_time + self.data_view.ref_lap.log.video_alignment - self.zoom_window / 2
        est_spacing = roundUpHumanNumber(self.zoom_window / (ph.size.width() / ph.scale / 60))
        self.x_axis = AxisGrid(zero_offset, zero_offset + self.zoom_window, est_spacing,
                               ph.size.width() / self.zoom_window, 0)
        if not self.data_view.ref_lap: return

        font = self.select_font()
        ph.painter.setFont(font)
        fontMetrics = QtGui.QFontMetrics(font)

        pen = QtGui.QPen(QtGui.QColor(192, 192, 192))
        pen.setStyle(Qt.SolidLine)
        ph.painter.setPen(pen)

        y_offset = fontMetrics.height() + 4

        tc = session_time
        ph.painter.drawText(0, 0, ph.size.width(), fontMetrics.height(),
                            Qt.AlignHCenter | Qt.AlignBottom | Qt.TextSingleLine,
                            '%.0f:%06.3f' % (math.copysign(math.trunc(tc / 60000), tc),
                                             abs(tc) % 60000 / 1000))
        ph.painter.drawLine(ph.size.width() / 2, y_offset, ph.size.width() / 2, y_offset - 4)
        ph.painter.drawLine(0, y_offset, ph.size.width(), y_offset)

        exp = int(math.floor(math.log10(self.x_axis.logical_tick_spacing) + .01)) - 3
        formatter = '%.0f:%02d' if exp >= 0 else ('%%.0f:%%0%d.%df' % (3 - exp, -exp))
        for i in range(int(math.ceil(self.x_axis.logical_min_val / self.x_axis.logical_tick_spacing)),
                       int(math.ceil(self.x_axis.logical_max_val / self.x_axis.logical_tick_spacing)) + 1):
            tc = i * self.x_axis.logical_tick_spacing
            x = self.x_axis.calc(tc)
            ph.painter.drawText(x - 100, y_offset + 4, 200, 50,
                                Qt.AlignHCenter | Qt.AlignTop | Qt.TextSingleLine,
                                formatter %
                                (math.copysign(math.trunc(tc / 60000), tc),
                                 abs(tc) % 60000 / 1000))
        spacing = self.x_axis.logical_tick_spacing / 5
        for i in range(int(math.ceil(self.x_axis.logical_min_val / spacing)),
                       int(math.ceil(self.x_axis.logical_max_val / spacing))):
            tc = i * spacing
            x = self.x_axis.calc(tc)
            ph.painter.drawLine(x, y_offset,
                                x, y_offset + (2 if i % 5 else 4))


class Video(QWidget):
    def __init__(self, data_view, state=None):
        super().__init__()
        self.data_view = data_view

        self.play_button = QAction('Play', self)
        #button.triggered.connect(self.media_player.play)
        self.play_button.triggered.connect(self.play_cb)
        self.addAction(self.play_button)

        act = QAction('Next frame', self)
        act.triggered.connect(self.next_frame)
        self.addAction(act)

        act = QAction('Prev frame', self)
        act.triggered.connect(self.prev_frame)
        self.addAction(act)

        act = QAction(self)
        act.setSeparator(True)
        self.addAction(act)

        act = QAction('Load video for reference lap', self)
        act.triggered.connect(self.load_ref_video)
        self.addAction(act)

        act = QAction('Set video alignment', self)
        act.setCheckable(True)
        act.triggered.connect(self.set_align_mode)
        self.addAction(act)

        self.layout = QGridLayout()

        self.videos = [OneVideo(self.data_view, False)]
        if self.data_view.alt_lap: self.videos.append(OneVideo(self.data_view, True))
        for c, v in enumerate(self.videos):
            self.layout.addWidget(v, 0, c, 1, 1)

        self.slider = AlignmentSlider(self.data_view)
        self.layout.addWidget(self.slider, 1, 0, 1, 1,)
        self.slider.hide()

        self.layout.setRowStretch(0, 1)
        self.setLayout(self.layout)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

    def save_state(self):
        return {'type': 'video',
                'base': self.parentWidget().save_state(),
                }

    def channels(self):
        return set()

    def updateCursor(self, old_cursor):
        if bool(self.data_view.alt_lap) != bool(len(self.videos) == 2):
            if self.data_view.alt_lap:
                self.videos.append(OneVideo(self.data_view, True))
                self.layout.addWidget(self.videos[-1], 0, 1, 1, 1)
            else:
                last = self.videos.pop()
                last.setParent(None)
                last.deleteLater()
        for v in self.videos:
            v.updateCursor(old_cursor)
        self.slider.update()

    def load_ref_video(self):
        lap = self.data_view.ref_lap
        if not lap: return
        file_name = QFileDialog.getOpenFileName(
            self, 'Open video file to associate with reference lap',
            os.path.dirname(lap.log.log.get_filename()), 'Video files (*.mp4 *.mov)')[0]
        if file_name:
            self.data_view.video_alignment[lap.log.log.get_filename()] = (file_name, 0)
            self.updateCursor(None)

    def set_align_mode(self, mode):
        self.slider.setVisible(mode)

    def play_cb(self):
        is_play = self.play_button.text() == 'Play'
        self.play_button.setText('Pause' if is_play else 'Play')
        for v in self.videos:
            if is_play:
                v.play_cb()
            else:
                v.pause_cb()

    def next_frame(self):
        self.videos[0].next_frame() # secondary will get updateCursor call

    def prev_frame(self):
        self.videos[0].prev_frame() # secondary will get updateCursor call
