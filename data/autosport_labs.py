
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
import csv
import time

import numpy as np

from . import base

def _decode_channel_hdr(ch):
    fields = list(csv.reader([ch[0]], delimiter='|'))[0]
    dec_places = 0
    for v in ch[1:]:
        if v:
            if '.' in v:
                dec_places = len(v) - v.index('.') - 1
            break
    return base.Channel(name=fields[0],
                        units=fields[1],
                        timecodes=array('d'),
                        values=array('d'),
                        dec_pts=dec_places,
                        interpolate=True)

def _decode_header(lines):
    return [_decode_channel_hdr(column)
            # group all the entries for a given column into a single list
            for column in zip(*list(csv.reader(lines, quoting=csv.QUOTE_NONE)))]

def AutosportLabs(fname, progress):
    with open(fname, 'rt', encoding='utf-8') as f:
        prefetch = 1000 # used to find # decimal places desired
        first_lines = [f.readline() for l in range(prefetch)]
        channels = _decode_header(first_lines)
        # ignore the interval column, that has special handing later on
        channels = channels[1:]
        base_timecode = float(first_lines[1].split(',')[0])
        total_len = prefetch + f.read().count(chr(10))
    with open(fname, 'rt', encoding='utf-8') as f:
        f.readline() # skip header
        update_count = 0
        for l in f:
            if (update_count & 4095) == 0 and progress:
                progress(update_count, total_len)
            update_count += 1
            l = l.rstrip().split(',')
            tc = float(l[0]) - base_timecode
            for i, v in enumerate(l[1:]):
                if v != '':
                    ch = channels[i]
                    ch.timecodes.append(tc)
                    ch.values.append(float(v))
    data = {ch.name: ch for ch in channels}

    ch = data['LapCount']
    dividers = np.array(ch.timecodes)[1:][np.array(ch.values)[1:] != np.array(ch.values)[:-1]]
    dividers = np.concatenate((np.array([0.]),
                               dividers,
                               np.array([max(np.max(ch.timecodes) for ch in channels)])))
    laps = [base.Lap(int(idx + ch.values[0]), int(start), int(end))
            for idx, (start, end) in enumerate(zip(dividers[:-1], dividers[1:]))]

    # No metadata is provided, but lets parse out the Utc value to determine data/time
    metadata = {}
    if 'Utc' in data:
        utc = data['Utc'].values[0]
        # ideally we would grab the time zone from the gps coordinates, but I'm lazy...
        tm = time.localtime(utc/1000)
        metadata['Log Date'] = '%02d/%02d/%d' % (tm.tm_mon, tm.tm_mday, tm.tm_year) # Yes I'm American
        metadata['Log Time'] = '%02d:%02d:%02d' % (tm.tm_hour, tm.tm_min, tm.tm_sec)

    return base.LogFile(data,
                        laps,
                        metadata,
                        ['Speed', 'Latitude', 'Longitude', 'Altitude'],
                        fname)
