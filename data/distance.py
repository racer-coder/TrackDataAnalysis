
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import bisect
from dataclasses import dataclass

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
        if i == 0: return self.values[i]
        if i == len(self.values): return self.values[i - 1]
        span = index[i] - index[i-1]
        if span == 0: return self.values[i]
        return self.values[i-1] + (self.values[i] - self.values[i-1]) * (tc - index[i-1]) / span

class DistanceWrapper:
    def __init__(self, data):
        self.data = data
        self.channel_cache = {}

        if not self.data.get_laps() or not data.get_speed_channel():
            self.dist_map_time = np.array([0.]).data
            self.dist_map_dist = self.dist_map_time
            return

        distdata = data.get_channel_data(data.get_speed_channel())

        # VALIDATED: AiM GPS Speed is just linear interpolation of raw
        # ECEF velocity between datapoints, modulo floating point
        # accuracy (they seem to use 32-bit floats).
        tc = np.arange(0, self.data.get_laps()[-1].end_time,
                       10, dtype=np.float64) # np.interp requires float64 arrays for performance
        gs = np.interp(tc, distdata[0], distdata[1])

        # adjust distances of each lap to match the median, if within a certain percentage
        dividers = [(l.start_time + 5) // 10 for l in self.data.get_laps()]
        lap_len = np.add.reduceat(gs, dividers)[1:-1]  # ignore in/out laps
        expected_len = np.median(lap_len)
        for s, e, ll in zip(dividers[1:], dividers[2:], lap_len): # ignores in/out laps
            if abs(ll - expected_len) / expected_len <= 0.05:
                gs[s:e] *= expected_len / ll

        ds = np.cumsum(gs / 100)
        self.dist_map_time = memoryview(tc)
        self.dist_map_dist = memoryview(ds)

    def outDist2Time(self, dist):
        return np.interp(dist, self.dist_map_dist, self.dist_map_time)

    def outTime2Dist(self, time):
        return np.interp(time, self.dist_map_time, self.dist_map_dist)

    def get_filename(self):
        return self.data.get_filename()

    def get_laps(self):
        return self.data.get_laps()

    def get_channels(self):
        return self.data.get_channels()

    def get_channel_data(self, *names):
        for name in names:
            if name not in self.channel_cache:
                data = self.data.get_channel_data(name)
                if data is None: data = ([], [])
                self.channel_cache[name] = ChannelData(data[0],
                                                       self.outTime2Dist(data[0]),
                                                       data[1],
                                                       self.data.get_channel_units(name),
                                                       self.data.get_channel_dec_points(name))
            if len(self.channel_cache[name].values):
                break
        return self.channel_cache[name]


