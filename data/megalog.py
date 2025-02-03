
# Copyright 2025, Scott Smith.  MIT License (see LICENSE).

from array import array
import concurrent.futures
import mmap
import struct
import time

import numpy as np

from . import base

def _build_channel(timestamps, channel_data, name, units, disp, transform, scale):
    if transform != 0:
        channel_data = channel_data.astype(np.float32)
        channel_data += transform
    if scale != 1:
        channel_data = channel_data.astype(np.float32)
        channel_data *= scale
    return base.Channel(timestamps,
                        channel_data.copy(), # reorder data to make it efficient
                        name,
                        units,
                        disp,
                        True) # interpolate

def _decode(m):
    metadata = {}
    ftag, ver, ts, info_offset, data_offset, record_len, num_fields = struct.unpack_from(
        '>6sHIIIHH', m, 0)
    ftag = ftag.decode('utf-8').rstrip('\0')
    assert ftag == 'MLVLG'
    assert ver == 2
    if ts:
        # not sure what timezone to use, log files don't store timezones
        tm = time.localtime(ts)
        metadata['Log Date'] = '%02d/%02d/%d' % (tm.tm_mon, tm.tm_mday, tm.tm_year) # Yes I'm American
        metadata['Log Time'] = '%02d:%02d:%02d' % (tm.tm_hour, tm.tm_min, tm.tm_sec)


    # collect data
    timestamp = 0
    timestamps = array('Q')
    data = []
    decoder = struct.Struct('>xxH')
    while data_offset + 2 < len(m):
        if m[data_offset] == 0:
            # handle data block
            chunk = m[data_offset+4:data_offset+4+record_len]
            if m[data_offset + 4 + record_len] == (np.sum(chunk) & 255):
                ts16, = decoder.unpack_from(m, data_offset)
                timestamp += (ts16 - timestamp) & 65535
                timestamps.append(timestamp)
                data.append(chunk)
            data_offset += 5 + record_len
        elif m[data_offset] == 1:
            # handle marker
            data_offset += 54 # we skip it, don't know how to handle it
        else:
            print('unknown', m[data_offset])
            data_offset += 1 # keep looking?
        # XXX call progress meter, maybe every 1000 or 5000 records?
    data = memoryview(b''.join(data))

    # convert timestamps to 32-bit ms.  Yes we lose some resolution....
    timestamps = ((np.array(timestamps) + 50) // 100).astype(np.uint32)
    # XXX look for dups and remove records

    # construct channel map given the timestamps and data
    channels = {}
    type_map = 'BbHhIiqf'
    row_offset = 0
    with concurrent.futures.ThreadPoolExecutor() as worker:
        for i in range(num_fields):
            ftype, name, units, disp, scale, transform, digits, category = struct.unpack_from(
                '>B34s10sBffb34s', m, 24 + i * 89)
            assert ftype < len(type_map) # XXX we don't support bit fields (10, 11, 12) among others.  Without handling the width we can't handle any more fields
            name = name.decode('utf-8').rstrip('\0')
            units = units.decode('utf-8').rstrip('\0')
            category = category.decode('utf-8').rstrip('\0')
            channel_data = np.ndarray(shape=timestamps.shape,
                                      strides=(record_len,),
                                      dtype='>' + type_map[ftype],
                                      offset=row_offset,
                                      buffer=data)
            row_offset += channel_data.itemsize
            channels[name] = worker.submit(_build_channel, timestamps, channel_data, name, units,
                                           disp, transform, scale)
        for n in channels.keys():
            channels[n] = channels[n].result()

    return channels, metadata, timestamps

def Megalog(fname, progress):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            channels, metadata, timestamps = _decode(memoryview(m))

    return base.LogFile(
        channels,
        [base.Lap(1, int(min(timestamps)), int(max(timestamps)))],
        metadata,
        [None, None, None, None],
        fname)
