
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array

import numpy as np

from . import base
from . import gps

builtin_channels = {
    'satellites': '#',
    'latitude': 'deg', # manually convert from minutes
    'longitude': 'deg', # manually convert from minutes
    'velocity kmh': 'km/h',
    'heading': 'deg',
    'height': 'm', # assume?
    'vertical velocity m/s': 'm/s',
    'sampleperiod': 's',
    'avifileindex': '',
    'avisynctime': 's',
}

def scan_data(f):
    all_data = array('d')
    while True:
        l = f.readline().rstrip()
        if not l:
            return all_data
        all_data.extend([float(f) for f in l.split(' ')])

def VBOX(fname, progress):
    group = []
    groups = {'': group}
    with open(fname, 'rt', encoding='utf-8') as f:
        while True:
            l = f.readline().rstrip()
            if not l:
                pass
            elif l[0] == '[':
                if l[:6] == '[data]':
                    data = scan_data(f)
                    break
                group = []
                groups[l[1:l.find(']')]] = group
            else:
                group.append(l)

    data = np.asarray(data)

    nch = len(groups['header'])
    # extract timecodes
    tc_index = groups['header'].index('time')
    timecodes = data[tc_index::nch]
    # convert from HHMMSS.hhh to seconds
    timecodes = (timecodes % 100 +
                 timecodes // 100 % 100 * 60 +
                 timecodes // 10000 * 3600)
    timecodes = (timecodes - timecodes[0]) * 1000 # rebase to 0 and convert to ms

    channels = {}
    next_unit = 0
    for ch in groups['header']:
        try:
            unit = builtin_channels[ch]
        except KeyError:
            unit = groups['channel units'][next_unit]
            next_unit += 1
        if ch != 'time':
            val = data[::nch]
            if ch == 'latitude':
                val /= 60
            elif ch == 'longitude':
                val /= -60
            channels[ch] = base.Channel(name=ch,
                                        units=unit,
                                        timecodes=timecodes,
                                        values=val.copy(),
                                        dec_pts=3, # seems to be Circuit Tools default
                                        interpolate=True)
        data = data[1:]

    # Lap timing based on given marker
    marker = [float(f) for f in groups['laptiming'][0].split()[1:5]]
    marker = [(marker[1] + marker[3]) / 120,
              (marker[0] + marker[2]) / -120]
    xyz = np.column_stack(gps.lla2ecef(channels['latitude'].values,
                                       channels['longitude'].values,
                                       channels['height'].values))
    lap_markers = gps.find_laps(xyz,
                                timecodes,
                                marker)
    if lap_markers:
        lap_markers = [0] + lap_markers + [timecodes[-1]]
        laps = [base.Lap(idx, start, end)
                for idx, (start, end) in enumerate(zip(lap_markers[:-1], lap_markers[1:]))]
    else:
        laps = [base.Lap(0, 0, timecodes[-1])]

    # XXX INFER LONG/LAT/VERT G

    # XXX RENAME BUILTIN CHANNELS TO FRIENDLIER NAMES

    # No metadata is provided, but lets parse out the Utc value to determine data/time
    metadata = {}
    if groups[''][0].startswith('File created on'):
        v = groups[''][0].split(' ')
        metadata['Log Date'] = v[3] # reorder fields for US audience?
        metadata['Log Time'] = v[5]

    return base.LogFile(channels,
                        laps,
                        metadata,
                        ['velocity kmh', 'latitude', 'longitude', 'height'],
                        fname)
