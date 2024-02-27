
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from configparser import ConfigParser
from dataclasses import dataclass, field
import math
import typing

from PySide2.QtCore import Signal
from PySide2 import QtGui
from PySide2.QtWidgets import QWidget

import numpy as np

from data import distance
from data import gps
from data import math_eval

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

@dataclass(eq=False)
class ChannelProperties:
    units : str
    dec_pts : int
    interpolate: bool
    color : int # index into color array

@dataclass(eq=False)
class ChannelData(distance.ChannelData):
    color: int

    @classmethod
    def derive(cls, parent, prop):
        return cls(parent.timecodes,
                   parent.distances,
                   parent.values,
                   parent.units,
                   prop.dec_pts,
                   prop.interpolate,
                   parent.min,
                   parent.max,
                   prop.color)

@dataclass(eq=False)
class MathExpr:
    name: str
    enabled: bool
    unit: str
    dec_pts: int
    interpolate: bool
    color: int
    sample_rate: int # 0 = automatic
    expr_unit: str
    expression: str
    comment: str

@dataclass(eq=False)
class MathGroup:
    enabled: bool
    # condition?
    expressions: list[MathExpr] # in the future this can include other types too
    comment: str

@dataclass(eq=False)
class Maths:
    groups: dict[str, MathGroup] = field(default_factory=dict)
    channel_map: dict[str, tuple[MathExpr, object]] = field(default_factory=dict) # computed from groups
    watcher: object = None
    finder: object = None

    def update_channel_map(self):
        cmap = {}
        for group in self.groups.values():
            if not group.enabled:
                continue
            for expr in group.expressions:
                if expr.enabled and expr.name not in cmap:
                    try:
                        cmap[expr.name] = (expr, math_eval.compile(expr.expression))
                    except (math_eval.LexError, math_eval.ParseError):
                        cmap[expr.name] = (expr, None) # placeholder, no-worky
        self.channel_map = cmap

    def get_channel_data(self, log, name, unit):
        if name not in self.channel_map:
            return None
        dummy = ChannelData([], [], [], unit, 0, False, 0, 0, 0)
        expr, eval_ = self.channel_map[name]
        if not eval_:
            return dummy # had parse issues

        # gather dependency set, bail if we find a cycle
        # XXX we should support having our own expr refer to ourselves
        seen = set()
        walk = list(eval_.depends)
        while walk:
            n = walk.pop()
            if n not in seen:
                if n == name:
                    return dummy
                seen.add(n)
                if n in self.channel_map and self.channel_map[n][1]:
                    for d in self.channel_map[n][1].depends:
                        walk.append(d)

        if expr.sample_rate:
            timecodes = np.arange(0, log.laps[-1].end.time, 1000 / expr.sample_rate,
                                  dtype=np.int32)
        else:
            timecodes = eval_.timecodes(log)
        if len(timecodes) == 0: # throw in start and end at least
            timecodes = np.linspace(0, log.laps[-1].end.time,
                                    num = int(log.laps[-1].end.time / 1000), # roughly 1/sec
                                    dtype=np.int32)
        distances = log.log.outTime2Dist(timecodes)
        try:
            values = eval_.values(log, timecodes)
        except:
            return dummy

        minval = np.min(values)
        maxval = np.max(values)
        if not np.isfinite(minval) or not np.isfinite(maxval):
            return dummy

        return ChannelData(timecodes,
                           distances,
                           values,
                           expr.unit,
                           expr.dec_pts,
                           expr.interpolate,
                           minval,
                           maxval,
                           expr.color)

# Cursor time/dist: Offset time/dist for the reference lap
# Offset time/dist: relative to start of zoom_window
# Session time/dist: relative to the start of the session

@dataclass()
class TimeDistRef:
    time: float # seconds
    dist: float # meters?

@dataclass(eq=False)
class LogRef:
    log: distance.DistanceWrapper
    video_file: typing.Optional[str] = None
    video_alignment: typing.Optional[int] = None
    laps: typing.List['LapRef'] = field(default_factory=list)
    best_lap: typing.Optional['LapRef'] = None
    math_cache: typing.Dict[str, ChannelData] = field(default_factory=dict)

    def get_channel_data(self, name, unit, maths=None):
        k = (name, unit)
        if k in self.math_cache:
            return self.math_cache[k]
        if maths:
            ret = maths.get_channel_data(self, name, unit)
            if ret:
                self.math_cache[k] = ret
                return ret
        return self.log.get_channel_data(name, unit)

    def update_laps(self):
        self.laps = [
            LapRef(self,
                   lap.num,
                   TimeDistRef(lap.start_time, self.log.outTime2Dist(lap.start_time)),
                   TimeDistRef(lap.end_time, self.log.outTime2Dist(lap.end_time)),
                   TimeDistRef(0., 0.))
            for lap in self.log.get_laps()]
        if len(self.laps) <= 2:
            self.best_lap = min(self.laps, key=lambda x: x.duration(), default=None)
        else:
            tgt_len = np.median([lap.end.dist - lap.start.dist
                                 for lap in self.laps[1:-1]]) * 0.95
            self.best_lap = min([lap for lap in self.laps[1:-1]
                                 if lap.end.dist-lap.start.dist >= tgt_len],
                                key=lambda x: x.duration())

    def math_invalidate(self):
        self.math_cache = {}

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

    def get_channel_data(self, name, unit, maths=None):
        return self.log.get_channel_data(name, unit, maths)

@dataclass(eq=False)
class Marker: # denotes end of section?
    name: str
    lat: float # in degrees
    lon: float # in degrees
    typ: str # what types do we care about? straight vs corner? left vs right? braking?  Motec allows user named markers.  AiM autochooses straight/left/right.
    _dist: float = None # in meters

@dataclass(eq=False)
class Sectors:
    name: str
    markers: list[Marker]

@dataclass(eq=False)
class Track:
    name: str
    file_name: str # not including holding directory
    coords: list[tuple[float, float, float, float]] # lat/long/alt/dist in degrees/meters.  Start finish line is first and last.
    sector_sets: dict[str, Sectors]

    def __init__(self, name, file_name, coords, sector_sets):
        self.name = name
        self.file_name = file_name
        self.coords = coords
        self.sector_sets = sector_sets
        na = np.array(coords)
        # we ignore altitude data since the markers don't have it
        # (some log formats don't include it)
        xyzd = np.column_stack(list(gps.lla2ecef(na[:,0], na[:,1], 0.)) + [na[:,3]])
        for ss in sector_sets.values():
            for m in ss.markers:
                m._dist = gps.find_crossing(xyzd, (m.lat, m.lon))[0]

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

    maths: Maths
    track: typing.Optional[Track]

    cursor_change: Signal # (old_cursor) when cursor position changed.  Lightest weight update
    values_change: Signal # () lap selection, lap shift, zoom window, time/dist mode.  Redraw all components, maybe more
    data_change: Signal # () focus change, channel selection, load log file.  Anything that requires dock widgets to update.

    config: ConfigParser

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

    def lapMode2Dist(self, lapref: LapRef, val):
        return lapref.lapTime2Dist(val) if self.mode_time else val

    def lapDist2Mode(self, lapref: LapRef, dist):
        return lapref.lapDist2Time(dist) if self.mode_time else dist

    def offTime2outMode(self, lapref: LapRef, time):
        return self.outTime2Mode(lapref, lapref.start.time + lapref.offset.time + time)

    def offMode2outDist(self, lapref: LapRef, val):
        return self.outMode2Dist(lapref, self.offMode2outMode(lapref, val))

    def offMode2outTime(self, lapref: LapRef, val):
        return self.outMode2Time(lapref, self.offMode2outMode(lapref, val))

    def offMode2outMode(self, lapref: LapRef, val):
        return self.getTDValue(lapref.start) + self.getTDValue(lapref.offset) + val

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
            return ChannelProperties(units='', dec_pts=0, interpolate=True, color=0)

    def get_channel_data(self, ref, ch): # ref is LogRef or LapRef
        props = self.get_channel_prop(ch)
        return ChannelData.derive(ref.get_channel_data(ch, unit=props.units, maths=self.maths),
                                  props)

    def math_invalidate(self):
        self.maths.update_channel_map()
        for log in self.log_files:
            log.math_invalidate()


# doesn't really belong here, but ....
def format_time(time_ms, sign=''): # pass '+' into sign to get +/- instead of ''/-
    return ('%' + sign + '.f:%06.3f') % (math.copysign(math.trunc(time_ms / 60000), time_ms),
                                         abs(time_ms) % 60000 / 1000)
