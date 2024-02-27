
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
from dataclasses import dataclass
import typing

import numpy as np

from . import base
from . import gps
from . import unitconv

@dataclass(eq=False)
class ChannelData:
    timecodes: memoryview
    distances: memoryview
    values: memoryview
    units: str
    dec_pts: int
    interpolate: bool
    min: float
    max: float

    @classmethod
    def from_data(cls, timecodes, distances, values, units, dec_pts, interpolate):
        unitconv.check_units(units)
        return cls(timecodes, distances, values, units, dec_pts, interpolate,
                   np.min(values) if len(values) else None,
                   np.max(values) if len(values) else None)

    def interp(self, tc, mode_time = True):
        index = self.timecodes if mode_time else self.distances
        # the 'not interpolate' mode requires bisect_right
        i = bisect.bisect_right(index, tc)
        if len(self.values) == 0: return 0
        if i == 0: return self.values[i]
        if i == len(self.values) or not self.interpolate: return self.values[i - 1]
        span = index[i] - index[i-1]
        if span == 0: return self.values[i]
        return self.values[i-1] + (self.values[i] - self.values[i-1]) * (tc - index[i-1]) / span

    def interp_many(self, tc, mode_time = True):
        index = self.timecodes if mode_time else self.distances
        if self.interpolate:
            return np.interp(tc, index, self.values)
        else:
            return self.values[np.minimum(np.searchsorted(index, tc, side='right'),
                                          len(index) - 1)]

    def change_units(self, units):
        converted = unitconv.convert(self.values, self.units, units)
        if converted is None:
            return ChannelData.from_data([], [], [], '', 0, True)
        return ChannelData.from_data(self.timecodes, self.distances, converted, units,
                                     self.dec_pts, self.interpolate) # what to do for dec_pts?


class DistanceWrapper:
    def __init__(self, data):
        self.data = data
        self.channel_cache = {}

        # in case we fail out
        self.dist_map_time = np.array([0.]).data
        self.dist_map_dist = self.dist_map_time

        if not data.laps or not data.key_channel_map[0]:
            return

        self.laps = data.laps

        self._update_time_dist()

    def try_gps_lap_insert(self, marker, dist):
        key_channels = self.get_key_channel_map()
        lat_ch = self.get_channel_data(key_channels[1], unit='deg')
        lon_ch = self.get_channel_data(key_channels[2], unit='deg')
        if len(lat_ch.values) and len(lon_ch.values):
            XYZ = np.column_stack(gps.lla2ecef(np.array(lat_ch.values),
                                               np.array(lon_ch.values), 0))
            lap_markers = gps.find_laps(XYZ,
                                        np.array(lat_ch.timecodes),
                                        marker)
            if lap_markers:
                lap_markers = [0] + lap_markers + [self.data.laps[-1].end_time]
                self.laps = [base.Lap(lap + self.data.laps[0].num, start_time, end_time)
                             for lap, (start_time, end_time) in enumerate(zip(lap_markers[:-1],
                                                                              lap_markers[1:]))]
        self._update_time_dist(dist)

    def _calc_time_dist(self, expected_len = None):
        distdata = self.data.channels[self.data.key_channel_map[0]]
        converted = unitconv.convert(distdata.values, distdata.units, 'm/s')
        if converted is None or len(converted) == 0:
            return np.array([0.]).data, np.array([0.]).data, 0

        # VALIDATED: AiM GPS Speed is just linear interpolation of raw
        # ECEF velocity between datapoints, modulo floating point
        # accuracy (they seem to use 32-bit floats).
        tc = np.arange(0, self.laps[-1].end_time,
                       10, dtype=np.float64) # np.interp requires float64 arrays for performance
        gs = np.interp(tc, distdata.timecodes, converted) * (1. / 100) # account for samplerate

        # adjust distances of each lap to match the median, if within a certain percentage
        dividers = [int(round(l.start_time / 10)) for l in self.laps]
        lap_len = np.add.reduceat(gs, dividers)[1:-1]  # ignore in/out laps
        if not expected_len:
            expected_len = np.median(lap_len)
        for s, e, ll in zip(dividers[1:], dividers[2:], lap_len): # ignores in/out laps
            if abs(ll - expected_len) / expected_len <= 0.05:
                gs[s:e] *= expected_len / ll

        ds = np.cumsum(gs)
        return memoryview(tc), memoryview(ds), expected_len

    def _update_time_dist(self, expected_len = None):
        self.dist_map_time, self.dist_map_dist, _ = self._calc_time_dist(expected_len)
        self.channel_cache = {}

    def outDist2Time(self, dist):
        return np.interp(dist, self.dist_map_dist, self.dist_map_time)

    def outTime2Dist(self, time):
        return np.interp(time, self.dist_map_time, self.dist_map_dist)

    def get_filename(self):
        return self.data.file_name

    def get_laps(self):
        return self.laps

    def get_metadata(self):
        return self.data.metadata

    def get_key_channel_map(self):
        return [ch if ch in self.data.channels else None for ch in self.data.key_channel_map]

    def get_channels(self):
        return self.data.channels.keys()

    # must include units, dec_pts
    def get_channel_metadata(self, name):
        ch = self.data.channels[name]
        return {'units': ch.units,
                'dec_pts': ch.dec_pts,
                'interpolate': ch.interpolate}

    def get_channel_data(self, name, unit=None):
        key = (name, unit)
        if key not in self.channel_cache:
            if unit:
                converted = self.get_channel_data(name).change_units(unit)
                if len(converted.values):
                    self.channel_cache[key] = converted
            elif name in self.data.channels:
                data = self.data.channels[name]
                self.channel_cache[key] = ChannelData.from_data(data.timecodes,
                                                                self.outTime2Dist(data.timecodes),
                                                                data.values,
                                                                data.units,
                                                                data.dec_pts,
                                                                data.interpolate)
        if key in self.channel_cache and len(self.channel_cache[key].values):
            return self.channel_cache[key]
        return ChannelData.from_data([], [], [], '', 0, True)
