
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
from dataclasses import dataclass
import mmap
import pprint
import struct
import sys

import numpy as np

# We use array and memoryview for efficient operations, but that
# assumes the sizes we expect match the file format.  Lets assert a
# few of those assumptions here.  Our use of struct is safe since it
# has tighter control over byte order and sizing.
assert array('H').itemsize == 2
assert array('I').itemsize == 4
assert array('f').itemsize == 4
assert array('Q').itemsize == 8
assert sys.byteorder == 'little'

@dataclass
class Channel:
    timecodes: array
    values: array
    dec_places: int
    name: str
    short_name: str
    units: str

@dataclass
class Lap:
    num: int
    start_time: int
    end_time: int

    def duration(self):
        return self.end_time - self.start_time

def _dec_u16(s, offs):
    return s[offs:offs+4].cast('H')[0]

def _dec_u32(s, offs):
    return s[offs:offs+4].cast('I')[0]

def _dec_str(s, offs, maxlen):
    s = bytes(s[offs:offs + maxlen])
    idx = s.find(b'\0')
    if idx >= 0:
        s = s[:idx]
    return s.decode('ascii')

def _decode_channel(s, addr):
    (data_addr, data_count, elem_type, elem_size, sample_rate,
     offset, mul, scale, dec_places) = struct.unpack_from('<IIxxHHHHHHh', s, addr + 8)

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

    return Channel((np.arange(0, data_count) * (1000 / sample_rate)).data,
                   ((np.multiply(data, 1 / (scale * 10 ** dec_places)) + offset) * mul).data,
                   dec_places,
                   _dec_str(s, addr+32, 32),
                   _dec_str(s, addr+64, 8),
                   _dec_str(s, addr+72, 12))

def _decode(s):
    ldmarker = _dec_u32(s, 0)
    assert ldmarker == 64
    channel_meta_addr = _dec_u32(s, 8)
    #channel_data_addr = _dec_u32(s, 12)
    #event_addr = _dec_u32(s, 36)
    #device_serial = _dec_u32(s, 70)
    #device_type = _dec_str(s, 74, 8)
    #device_version = _dec_u16(s, 82)
    num_channels = _dec_u32(s, 86)
    #date_string = _dec_str(s, 94, 16)
    #time_string = _dec_str(s, 126, 16)
    #driver = _dec_str(s, 158, 64)
    #vehicleid = _dec_str(s, 222, 64)
    #venue = _dec_str(s, 350, 64)
    #session = _dec_str(s, 1508, 64)
    #short_comment = _dec_str(s, 1572, 64)

    channels = {}
    addr = channel_meta_addr
    for i in range(num_channels):
        ch = _decode_channel(s, addr)
        channels[ch.name] = ch
        addr = _dec_u32(s, addr+4)
    return channels

class MOTEC:
    def __init__(self, fname, progress):
        self.file_name = fname
        with open(fname, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                self.data = _decode(memoryview(m))
        laps = [0]
        seq_start_tc = None
        for tc, v in zip(self.data['Beacon'].timecodes, self.data['Beacon'].values):
            if seq_start_tc is None:
                if v < 0:
                    seq_start_tc = tc
            else:
                if v >= 16384:
                    last_val = v
                elif v >= 0:
                    if v == 100:
                        laps.append(seq_start_tc - 1000 + (int(last_val) & 1023))
                    seq_start_tc = None
        laps.append(max(np.max(d.timecodes) for d in self.data.values()))
        self.laps = [int(l) for l in laps]

    def get_laps(self):
        return [Lap(l, s, e) for l, (s, e) in enumerate(zip(self.laps[:-1], self.laps[1:]))]

    def get_speed_channel(self):
        return 'Ground Speed'

    def get_filename(self):
        return self.file_name

    def get_channels(self):
        return self.data.keys()

    def get_channel_units(self, name):
        return self.data[name].units if name in self.data else None

    def get_channel_dec_points(self, name):
        return max(0, self.data[name].dec_places) if name in self.data else None

    def get_channel_data(self, name):
        if name not in self.data:
            return None
        d = self.data[name]
        return (d.timecodes, d.values)
