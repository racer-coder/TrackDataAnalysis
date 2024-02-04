
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from dataclasses import dataclass
import typing

from PySide2.QtCore import Signal
from PySide2.QtWidgets import QWidget

# Cursor time/dist: Offset time/dist for the reference lap
# Offset time/dist: relative to start of zoom_window
# Session time/dist: relative to the start of the session

@dataclass()
class TimeDistRef:
    time: float # seconds
    dist: float # meters?

@dataclass()
class LogRef:
    log: object # usually data.Distance
    video_file: typing.Optional[str] = None
    video_alignment: typing.Optional[int] = None

@dataclass()
class LapRef:
    log: LogRef
    lap: object
    offset: TimeDistRef

    def same_log_and_lap(self, other):
        return other and self.log == other.log and self.lap == other.lap

    def lapDist2Time(self, dist):
        return self.log.log.outDist2Time(self.log.log.outTime2Dist(self.lap.start_time) + dist) - self.lap.start_time

    def lapTime2Dist(self, time):
        p = self.log.log.outTime2Dist([self.lap.start_time + time, self.lap.start_time])
        return p[0] - p[1]

    def offDist2Time(self, dist):
        return self.lapDist2Time(dist + self.offset.dist) - self.offset.time

@dataclass()
class DataView:
    ref_lap: typing.Optional[LapRef]
    alt_lap: typing.Optional[LapRef]
    extra_laps: typing.List[LapRef]
    cursor_time: TimeDistRef
    zoom_window: typing.Tuple[TimeDistRef, TimeDistRef] # relative to start and end of lap, respectively
    mode_time: bool # Varies per worksheet
    mode_offset: bool # Whether we have lap offsets
    log_files: typing.List[LogRef]
    active_component: typing.Optional[QWidget] # widget with current focus
    video_alignment: typing.Dict[str, typing.Tuple[str, int]] # {log_fname: (vid_fname, vid_align)}
    maps_key: typing.Optional[typing.Tuple[str, str]] # provider ('maptiler'), key

    cursor_change: Signal # (old_cursor) when cursor position changed.  Lightest weight update
    values_change: Signal # () lap selection, lap shift, zoom window, time/dist mode.  Redraw all components, maybe more
    data_change: Signal # () lap selection, focus change, channel selection, load log file.  Anything that requires dock widgets to update.

    def outTime2Mode(self, lapref: LapRef, time):
        return time if self.mode_time else lapref.log.log.outTime2Dist(time)

    def outMode2Dist(self, lapref: LapRef, val):
        return lapref.log.log.outTime2Dist(val) if self.mode_time else val

    def outMode2Time(self, lapref: LapRef, val):
        return val if self.mode_time else lapref.log.log.outDist2Time(val)

    def lapTime2Mode(self, lapref: LapRef, time):
        return time if self.mode_time else lapref.lapTime2Dist(time)

    def offTime2outMode(self, lapref: LapRef, time):
        return self.outTime2Mode(lapref, lapref.lap.start_time + lapref.offset.time + time)

    def offMode2outDist(self, lapref: LapRef, val):
        return self.outMode2Dist(lapref, self.offMode2outMode(lapref, val))

    def offMode2outMode(self, lapref: LapRef, val):
        return val + self.outTime2Mode(lapref, lapref.lap.start_time + lapref.offset.time)

    def offMode2outTime(self, lapref: LapRef, val):
        return self.outMode2Time(lapref, self.offMode2outMode(lapref, val))

    def offMode2Dist(self, lapref: LapRef, val):
        return self.outMode2Dist(lapref, self.offMode2outMode(lapref, val)) - lapref.log.log.outTime2Dist(lapref.lap.start_time + lapref.offset.time)

    def offMode2Time(self, lapref: LapRef, val):
        return self.outMode2Time(lapref, self.offMode2outMode(lapref, val)) - lapref.lap.start_time - lapref.offset.time

    def outTime2offTime(self, lapref: LapRef, time):
        return time - lapref.lap.start_time - lapref.offset.time

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
            lapdur = lapref.lap.duration()
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
        p = self.outTime2Mode(lapref, [lapref.lap.start_time, lapref.lap.end_time])
        return (p[0], p[1])

    def windowSize2Mode(self):
        lap_range = self.getLapValue(self.ref_lap)
        return (lap_range[1] - lap_range[0]
                + self.getTDValue(self.zoom_window[1]) - self.getTDValue(self.zoom_window[0]))
