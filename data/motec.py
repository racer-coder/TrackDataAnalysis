
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import mmap
import struct

import numpy as np

from . import base

def _dec_u16(s, offs):
    return s[offs:offs+2].cast('H')[0]

def _dec_u32(s, offs):
    return s[offs:offs+4].cast('I')[0]

def _dec_str(s, offs, maxlen):
    s = bytes(s[offs:offs + maxlen])
    idx = s.find(b'\0')
    if idx >= 0:
        s = s[:idx]
    return s.decode('ascii')

def _set_if(meta, name, val, formatter=None):
    if val:
        meta[name] = formatter % val if formatter else val

def _decode_channel(s, addr):
    (data_addr, data_count, elem_type, elem_size, sample_rate,
     offset, mul, scale, dec_pts) = struct.unpack_from('<IIxxHHHHHHh', s, addr + 8)

    # cast the data into the right datatype
    data = s[data_addr : data_addr + elem_size * data_count]
    if elem_type in (0, 3, 5):
        if elem_size == 2: data = data.cast('h')
        elif elem_size == 4: data = data.cast('i')
        else: raise TypeError
    elif elem_type == 7:
        if elem_size == 4: data = data.cast('f')
        else: raise TypeError
    else: raise TypeError

    return base.Channel((np.arange(0, data_count) * (1000 / sample_rate)).data,
                        ((np.multiply(data, 1 / (scale * 10 ** dec_pts)) + offset) * mul).data,
                        dec_pts=max(dec_pts, 0),
                        name=_dec_str(s, addr+32, 32),
                        #_dec_str(s, addr+64, 8),
                        units=_dec_str(s, addr+72, 12))

def _decode(s):
    metadata = {}

    ldmarker = _dec_u32(s, 0)
    assert ldmarker == 64
    channel_meta_addr = _dec_u32(s, 8)
    #channel_data_addr = _dec_u32(s, 12)
    event_addr = _dec_u32(s, 36)
    metadata['Device Serial'] = _dec_u32(s, 70)
    metadata['Device Type'] = _dec_str(s, 74, 8)
    metadata['Device Version'] = '%.2f' % (_dec_u16(s, 82) / 100)
    num_channels = _dec_u32(s, 86)
    metadata['Log Date'] = _dec_str(s, 94, 16)
    metadata['Log Time'] = _dec_str(s, 126, 16)
    metadata['Driver'] = _dec_str(s, 158, 64)
    metadata['Vehicle'] = _dec_str(s, 222, 64)
    metadata['Venue'] = _dec_str(s, 350, 64)
    metadata['Session'] = _dec_str(s, 1508, 64)
    metadata['Short Comment'] = _dec_str(s, 1572, 64)

    # Parse more detailed event information
    # Why is this some weird linked list of structs?
    if event_addr:
        metadata['Event Name'] = _dec_str(s, event_addr, 64)
        metadata['Event Session'] = _dec_str(s, event_addr+64, 64) # how is this different from Session?
        metadata['Long Comment'] = _dec_str(s, event_addr+128, 1024)
        venue_addr = _dec_u32(s, event_addr+1152)
        if venue_addr:
            metadata['Venue Name'] = _dec_str(s, venue_addr, 64) # how is this different than Venu?
            vehicle_addr = _dec_u32(s, venue_addr+1098)
            if vehicle_addr: # 0x1f94
                metadata['Vehicle Id'] = _dec_str(s, vehicle_addr, 64) # how is this different from Vehicle?
                metadata['Vehicle Desc'] = _dec_str(s, vehicle_addr+64, 64) # Not sure on length here
                # I bet Vehicle Number is right here too
                _set_if(metadata, 'Vehicle Weight', _dec_u32(s, vehicle_addr+192))
                _set_if(metadata, 'Vehicle Type', _dec_str(s, vehicle_addr+196, 32))
                _set_if(metadata, 'Vehicle Comment', _dec_str(s, vehicle_addr+228, 32))
                _set_if(metadata, 'Diff Ratio', _dec_u16(s, vehicle_addr+260) / 1000, '%.3f') # probably?
                for gear in range(1, 10):
                    _set_if(metadata, 'Gear %d' % gear,
                            _dec_u16(s, vehicle_addr + 260 + gear*2) / 1000, '%.3f')
                _set_if(metadata, 'Vehicle Wheelbase [mm]', _dec_u16(s, vehicle_addr+284))

    channels = {}
    addr = channel_meta_addr
    for _ in range(num_channels):
        ch = _decode_channel(s, addr)
        channels[ch.name] = ch
        addr = _dec_u32(s, addr+4)
    return channels, metadata

def MOTEC(fname, progress):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            data, metadata = _decode(memoryview(m))
    laps = [0]
    seq_start_tc = None
    for tc, v in zip(data['Beacon'].timecodes, data['Beacon'].values):
        if seq_start_tc is None:
            if v < 0:
                seq_start_tc = tc
        else:
            if v >= 16384:
                last_val = int(v)
            elif v >= 0:
                if v == 100 or v == 2: # but not 56?
                    last_val &= 16383
                    if last_val >= 8192:
                        last_val -= 16384
                    laps.append(seq_start_tc - 1000 + last_val)
                seq_start_tc = None
    laps.append(max(np.max(d.timecodes) for d in data.values()))
    laps = [int(l) for l in laps]

    return base.LogFile(data,
                        [base.Lap(l, s, e) for l, (s, e) in enumerate(zip(laps[:-1], laps[1:]))],
                        metadata,
                        [ch if ch in data else None
                         for ch in ['Ground Speed', 'GPS Latitude', 'GPS Longitude', 'Altitude']],
                        fname)
