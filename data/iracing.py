
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

# Parts taken/reinterpreted from https://github.com/gmartsenkov/itelem (also MIT licensed)

import mmap
import struct
import time

import numpy as np
import yaml

from .base import Channel, Lap, LogFile

def _dec_str(s, offs, maxlen):
    s = bytes(s[offs:offs + maxlen])
    idx = s.find(b'\0')
    if idx >= 0:
        s = s[:idx]
    return s.decode('ascii')

_types = [(1, 'c'),
          (1, '?'),
          (4, 'i'),
          (4, 'I'),
          (4, 'f'),
          (8, 'd')]

def _decode_var(m, offs, timecodes, samples, nrecords, stride):
    rtype, offset, _count, _count_as_time = struct.unpack_from('<3ib', m, offs)
    name = _dec_str(m, offs + 16, 32)
    #desc = _dec_str(m, offs + 48, 112 - 48)
    unit = _dec_str(m, offs + 112, 32)
    data = np.concatenate([np.array(samples[offset + i::stride]).reshape((nrecords, 1))
                           for i in range(_types[rtype][0])],
                          axis=1)
    if rtype == 1:
        data = (data != 0)
    data = np.ascontiguousarray(data).data.cast('B').cast(_types[rtype][1])
    if unit == '%': # encoded actually as a ratio, not a percentage
        data = (np.array(data) * 100).data
    return Channel(timecodes, data, name=name, units=unit, dec_pts=2 if rtype >= 4 else 0)

def _decode(m):
    (_version, _status, tick_rate, _session_info_update, session_info_length, session_info_offset,
     num_vars, var_header_offset, num_buf, buf_len, buf_offset) = \
         struct.unpack_from('<10i12xi', m, 0)
    # not sure what to do with half these fields
    if num_buf != 1: print("Don't understand multiple buffers")

    utc, _start_time, _end_time, _lap_count, record_count = \
        struct.unpack_from('<I4xddii', m, 112)
    tm = time.localtime(utc)

    timecodes = (np.arange(record_count) * (1000 / tick_rate)).data
    samples = m[buf_offset : buf_offset+buf_len*record_count]
    channels = [_decode_var(m, var_header_offset + i * 144, timecodes, samples,
                            record_count, buf_len)
                for i in range(num_vars)]

    idata = yaml.safe_load(m[session_info_offset : session_info_offset+session_info_length].tobytes())

    return ({v.name: v for v in channels},
            {'Log Date': '%02d/%02d/%d' % (tm.tm_mon, tm.tm_mday, tm.tm_year), # Yes I'm American
             'Log Time': '%02d:%02d:%02d' % (tm.tm_hour, tm.tm_min, tm.tm_sec),
             'Driver': [d['UserName'] for d in idata['DriverInfo']['Drivers']
                        if d['UserID'] == idata['DriverInfo']['DriverUserID']][0],
             'Venue': idata['WeekendInfo']['TrackDisplayName'],
             })


def _filter_gps(data):
    # filter out lat/long of 0
    lat = data['Lat']
    lon = data['Lon']
    alt = data['Alt']
    pick = (np.array(lat.values) != 0) | (np.array(lon.values) != 0)
    tc = np.array(lat.timecodes)[pick].data
    lat.timecodes = tc
    lat.values = np.array(lat.values)[pick].data
    lon.timecodes = tc
    lon.values = np.array(lon.values)[pick].data
    alt.timecodes = tc
    alt.values = np.array(alt.values)[pick].data

def _find_laps(data):
    # LapCurrentLapTime doesn't actually reset on lap boundaries
    # (it reacts about a half second later, but doesn't react at
    # all on an out lap), so instead we rely on a combination of
    # LapDist and Speed to determine exactly when the lap change
    # occurred.
    lap = np.array(data['Lap'].values)
    lapdist = data['LapDist'].values
    speed = data['Speed']

    assert data['LapDist'].units == 'm'
    assert speed.units == 'm/s'

    lap_markers = [0]
    add_end = True
    for b in (np.nonzero(lap[1:] != lap[:-1])[0] + 1):
        if lap[b] == 0: # probably no more useful data
            lap_markers.append(speed.timecodes[b])
            add_end = False
            break
        lap_markers.append(max(speed.timecodes[b-1],
                               speed.timecodes[b] - lapdist[b] / speed.values[b] * 1000))
    if add_end:
        lap_markers.append(speed.timecodes[-1])

    return [Lap(i, s, e) for i, (s, e) in enumerate(zip(lap_markers[:-1], lap_markers[1:]))]

def IRacing(fname, progress):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            data, metadata = _decode(memoryview(m))
    _filter_gps(data)

    return LogFile(data,
                   _find_laps(data),
                   metadata,
                   ['Speed', 'Lat', 'Lon', 'Alt'],
                   fname)
