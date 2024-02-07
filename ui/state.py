
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from dataclasses import dataclass, field
import math
import typing

from PySide2.QtCore import Signal
from PySide2 import QtGui
from PySide2.QtWidgets import QWidget

from data import distance

# belongs in a theme file, but ...
lap_colors = (
    QtGui.QColor(240, 0, 0),
    QtGui.QColor(223, 223, 223),
    QtGui.QColor(64, 255, 64),
    QtGui.QColor(43, 255, 255),
    QtGui.QColor(47, 151, 255),
    QtGui.QColor(186, 117, 255),
    QtGui.QColor(255, 106, 218),
    QtGui.QColor(244, 244, 0),
    QtGui.QColor(255, 160, 32),
    )

# Cursor time/dist: Offset time/dist for the reference lap
# Offset time/dist: relative to start of zoom_window
# Session time/dist: relative to the start of the session

@dataclass()
class TimeDistRef:
    time: float # seconds
    dist: float # meters?

@dataclass(eq=False)
class LogRef:
    log: object # usually data.Distance
    video_file: typing.Optional[str] = None
    video_alignment: typing.Optional[int] = None
    laps: typing.List['LapRef'] = field(default_factory=list)

    def get_channel_data(self, *args, **kwargs):
        return self.log.get_channel_data(*args, **kwargs)

@dataclass(eq=False)
class LapRef:
    log: LogRef
    num: int
    start: TimeDistRef
    end: TimeDistRef
    offset: TimeDistRef

    def lapDist2Time(self, dist):
        return self.log.log.outDist2Time(self.start.dist + dist) - self.start.time

    def lapTime2Dist(self, time):
        return self.log.log.outTime2Dist(self.start.time + time) - self.start.dist

    def offDist2Time(self, dist):
        return self.lapDist2Time(dist + self.offset.dist) - self.offset.time

    def duration(self):
        return self.end.time - self.start.time

    def get_channel_data(self, *args, **kwargs):
        return self.log.get_channel_data(*args, **kwargs)

@dataclass(eq=False)
class ChannelProperties:
    units : str
    dec_pts : int
    color : int # index into color array

@dataclass(eq=False)
class ChannelData(distance.ChannelData):
    color: int

    def __init__(self, parent, prop):
        self.timecodes = parent.timecodes
        self.distances = parent.distances
        self.values = parent.values
        self.units = parent.units
        self.dec_pts = prop.dec_pts
        self.min = parent.min
        self.max = parent.max
        self.color = prop.color

@dataclass(eq=False)
class DataView:
    ref_lap: typing.Optional[LapRef]
    alt_lap: typing.Optional[LapRef]
    extra_laps: typing.List[typing.Tuple[LapRef, QtGui.QColor]]
    cursor_time: TimeDistRef
    zoom_window: typing.Tuple[TimeDistRef, TimeDistRef] # relative to start and end of lap, respectively
    mode_time: bool # Varies per worksheet
    mode_offset: bool # Whether we have lap offsets
    log_files: typing.List[LogRef]
    active_component: typing.Optional[QWidget] # widget with current focus
    video_alignment: typing.Dict[str, typing.Tuple[str, int]] # {log_fname: (vid_fname, vid_align)}
    maps_key: typing.Optional[typing.Tuple[str, str]] # provider ('maptiler'), key

    # channel_overrides contains only those properties explicitly set by the user.
    channel_overrides: typing.Dict[str, object] # [name]
    # channel_properties should be fully populated, either from the
    # open log file(s), dynamically chosen at runtime (colors), or
    # from channel_overrides.
    channel_properties: typing.Dict[str, ChannelProperties] # [name]
    # channel_defaults should be fully populated from open log files.
    channel_defaults: typing.Dict[str, ChannelProperties] # [name]

    cursor_change: Signal # (old_cursor) when cursor position changed.  Lightest weight update
    values_change: Signal # () lap selection, lap shift, zoom window, time/dist mode.  Redraw all components, maybe more
    data_change: Signal # () focus change, channel selection, load log file.  Anything that requires dock widgets to update.

    def get_laps(self):
        return [(l, c, idx + 3)
                for idx, (l, c) in ([(-2, (self.ref_lap, lap_colors[0])),
                                     (-1, (self.alt_lap, lap_colors[1]))]
                                    + list(enumerate(self.extra_laps)))
                if l]

    def outTime2Mode(self, lapref: LapRef, time):
        return time if self.mode_time else lapref.log.log.outTime2Dist(time)

    def outMode2Dist(self, lapref: LapRef, val):
        return lapref.log.log.outTime2Dist(val) if self.mode_time else val

    def outMode2Time(self, lapref: LapRef, val):
        return val if self.mode_time else lapref.log.log.outDist2Time(val)

    def lapTime2Mode(self, lapref: LapRef, time):
        return time if self.mode_time else lapref.lapTime2Dist(time)

    def offTime2outMode(self, lapref: LapRef, time):
        return self.outTime2Mode(lapref, lapref.start.time + lapref.offset.time + time)

    def offMode2outDist(self, lapref: LapRef, val):
        return self.outMode2Dist(lapref, self.offMode2outMode(lapref, val))

    def offMode2outTime(self, lapref: LapRef, val):
        return self.outMode2Time(lapref, self.offMode2outMode(lapref, val))

    def offMode2outMode(self, lapref: LapRef, val):
        return self.getTDValue(lapref.start) + self.getTDValue(lapref.offset) + val

    def offMode2outTime(self, lapref: LapRef, val):
        return self.outMode2Time(lapref, self.offMode2outMode(lapref, val))

    def offMode2Dist(self, lapref: LapRef, val):
        return self.offMode2outDist(lapref, val) - lapref.start.dist - lapref.offset.dist

    def offMode2Time(self, lapref: LapRef, val):
        return self.offMode2outTime(lapref, val) - lapref.start.time - lapref.offset.time

    def outTime2offTime(self, lapref: LapRef, time):
        return time - lapref.start.time - lapref.offset.time

    def cursor2offDist(self, lapref: LapRef):
        return self.offMode2Dist(lapref, self.getTDValue(self.cursor_time))

    def cursor2offTime(self, lapref: LapRef):
        return self.offMode2Time(lapref, self.getTDValue(self.cursor_time))

    def cursor2outDist(self, lapref: LapRef):
        return self.offMode2outDist(lapref, self.getTDValue(self.cursor_time))

    def cursor2outTime(self, lapref: LapRef):
        return self.offMode2outTime(lapref, self.getTDValue(self.cursor_time))

    def outTime2cursor(self, lapref: LapRef, time):
        time = self.outTime2offTime(lapref, time)
        return TimeDistRef(time, lapref.lapTime2Dist(time))

    def makeTD(self, val, rel_end):
        lapref = self.ref_lap
        if rel_end:
            lapdur = lapref.duration()
            basetd = TimeDistRef(lapdur, lapref.lapTime2Dist(lapdur))
            valtd = self.makeTD(val + self.getTDValue(basetd), False)
            return TimeDistRef(valtd.time - basetd.time, valtd.dist - basetd.dist)
        else:
            if self.mode_time:
                return TimeDistRef(val, lapref.lapTime2Dist(val))
            else:
                return TimeDistRef(lapref.lapDist2Time(val), val)

    def getTDValue(self, tdref: TimeDistRef):
        return tdref.time if self.mode_time else tdref.dist

    def getLapValue(self, lapref: LapRef):
        return (self.getTDValue(lapref.start), self.getTDValue(lapref.end))

    def windowSize2Mode(self):
        lap_range = self.getLapValue(self.ref_lap)
        return (lap_range[1] - lap_range[0]
                + self.getTDValue(self.zoom_window[1]) - self.getTDValue(self.zoom_window[0]))

    def get_channel_prop(self, ch):
        if ch in self.channel_properties:
            return self.channel_properties[ch]
        else:
            return ChannelProperties(units='', dec_pts=0, color=0)

    def get_channel_data(self, ref, ch): # ref is LogRef or LapRef
        props = self.get_channel_prop(ch)
        return ChannelData(ref.get_channel_data(ch, unit=props.units), props)


# doesn't really belong here, but ....
def format_time(time_ms, sign=''): # pass '+' into sign to get +/- instead of ''/-
    return ('%' + sign + '.f:%06.3f') % (math.copysign(math.trunc(time_ms / 60000), time_ms),
                                         abs(time_ms) % 60000 / 1000)
