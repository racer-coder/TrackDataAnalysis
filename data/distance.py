
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
from dataclasses import dataclass
import typing

import numpy as np

from . import unitconv

@dataclass
class ChannelData:
    timecodes: object
    distances: object
    values: object
    units: str
    dec_pts: int
    min: float
    max: float

    def __init__(self, timecodes, distances, values, units, dec_pts):
        self.timecodes = timecodes
        self.distances = distances
        self.values = values
        self.units = units
        unitconv.check_units(units)
        self.dec_pts = dec_pts
        self.min = np.min(values) if len(values) else None
        self.max = np.max(values) if len(values) else None

    def interp(self, tc, mode_time = True):
        index = self.timecodes if mode_time else self.distances
        i = bisect.bisect_left(index, tc)
        if len(self.values) == 0: return 0
        if i == 0: return self.values[i]
        if i == len(self.values): return self.values[i - 1]
        span = index[i] - index[i-1]
        if span == 0: return self.values[i]
        return self.values[i-1] + (self.values[i] - self.values[i-1]) * (tc - index[i-1]) / span

    def change_units(self, units):
        converted = unitconv.convert(self.values,self.units, units)
        if converted is None:
            return ChannelData([], [], [], '', 0)
        return ChannelData(self.timecodes, self.distances, converted,
                           units, self.dec_pts) # ??


class DistanceWrapper:
    def __init__(self, data):
        self.data = data
        self.metadata = data.get_metadata()
        self.channel_cache = {}

        # in case we fail out
        self.dist_map_time = np.array([0.]).data
        self.dist_map_dist = self.dist_map_time

        if not self.data.get_laps() or not data.get_speed_channel():
            return

        self._update_time_dist()

    def _calc_time_dist(self, expected_len = None):
        distdata = self.data.get_channel_data(self.data.get_speed_channel())
        converted = unitconv.convert(distdata[1],
                                     self.data.get_channel_units(self.data.get_speed_channel()),
                                     'm/s')
        if len(converted) == 0:
            return np.array([0.]).data, np.array([0.]).data, 0

        # VALIDATED: AiM GPS Speed is just linear interpolation of raw
        # ECEF velocity between datapoints, modulo floating point
        # accuracy (they seem to use 32-bit floats).
        tc = np.arange(0, self.data.get_laps()[-1].end_time,
                       10, dtype=np.float64) # np.interp requires float64 arrays for performance
        gs = np.interp(tc, distdata[0], converted)

        # adjust distances of each lap to match the median, if within a certain percentage
        dividers = [int(round(l.start_time / 10)) for l in self.data.get_laps()]
        lap_len = np.add.reduceat(gs, dividers)[1:-1]  # ignore in/out laps
        if not expected_len:
            expected_len = np.median(lap_len)
        for s, e, ll in zip(dividers[1:], dividers[2:], lap_len): # ignores in/out laps
            if abs(ll - expected_len) / expected_len <= 0.05:
                gs[s:e] *= expected_len / ll

        ds = np.cumsum(gs / 100)
        return memoryview(tc), memoryview(ds), expected_len

    def _update_time_dist(self, expected_len = None):
        self.dist_map_time, self.dist_map_dist, _ = self._calc_time_dist(expected_len)
        self.channel_cache = {}

    def outDist2Time(self, dist):
        return np.interp(dist, self.dist_map_dist, self.dist_map_time)

    def outTime2Dist(self, time):
        return np.interp(time, self.dist_map_time, self.dist_map_dist)

    def get_filename(self):
        return self.data.get_filename()

    def get_laps(self):
        return self.data.get_laps()

    def get_metadata(self):
        return self.metadata

    def get_channels(self):
        return self.data.get_channels()

    # must include units, dec_pts
    def get_channel_metadata(self, name):
        return {'units': self.data.get_channel_units(name),
                'dec_pts': self.data.get_channel_dec_points(name)}

    def get_channel_data(self, *names, unit=None):
        for name in names:
            key = (name, unit)
            if key not in self.channel_cache:
                if unit:
                    converted = self.get_channel_data(name).change_units(unit)
                    if not len(converted.values): continue
                    self.channel_cache[key] = converted
                else:
                    data = self.data.get_channel_data(name)
                    if data is None: continue
                    self.channel_cache[key] = ChannelData(data[0],
                                                          self.outTime2Dist(data[0]),
                                                          data[1],
                                                          self.data.get_channel_units(name),
                                                          self.data.get_channel_dec_points(name))
            if len(self.channel_cache[key].values):
                return self.channel_cache[key]
        return ChannelData([], [], [], '', 0)


def unify_lap_distance(logs: typing.List[DistanceWrapper]):
    lens = list(filter(bool, [dw._calc_time_dist()[2] for dw in logs]))
    if not lens: return # what are we supposed to do here?
    expected_len = np.mean(lens).item()
    for dw in logs:
        dw._update_time_dist(expected_len)

