
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
from dataclasses import dataclass
import struct
import time

import numpy as np

from . import base

# https://www.race-technology.com/wiki/index.php/General/SerialDataFormat

@dataclass
class State:
    data: bytes
    timecode: int
    samples: list[array]

class DataView:
    def __init__(self, data, timestamp_pos, gps_time):
        self.sliding_i8 = np.ndarray(buffer = data,
                                     dtype = np.int8,
                                     shape = len(data))
        self.sliding_u8 = np.ndarray(buffer = data,
                                     dtype = np.uint8,
                                     shape = len(data))
        self.sliding_i16 = np.ndarray(buffer = data,
                                      dtype = np.int16,
                                      strides = 1,
                                      shape = len(data) - 1)
        self.sliding_u16 = np.ndarray(buffer = data,
                                      dtype = np.uint16,
                                      strides = 1,
                                      shape = len(data) - 1)
        self.sliding_i32 = np.ndarray(buffer = data,
                                      dtype = np.int32,
                                      strides = 1,
                                      shape = len(data) - 3)
        self.sliding_u32 = np.ndarray(buffer = data,
                                      dtype = np.uint32,
                                      strides = 1,
                                      shape = len(data) - 3)

        # create timestamp lookup table
        self.timestamp_pos = timestamp_pos
        self.timestamp_val = (self.sliding_u32[timestamp_pos].byteswap() & 0xffffff) * 10

        # manually process #7: gps_time_storage
        # this allows other channels to map gps time->log time
        self.gps_time_map = np.column_stack(
            [self.sliding_u32[1:][gps_time].byteswap(), # gps time
             self.ts_lookup(gps_time)]) # tc

        self.lap_crossing = np.array([], dtype=np.uint32)

    def ts_lookup(self, pos):
        return self.timestamp_val[np.maximum(np.searchsorted(self.timestamp_pos, pos) - 1, 0)]

def separate_subchannels(sliding_u8, allpos):
    compound = (sliding_u8[1:][allpos].astype(np.uint64) << 32) + allpos
    compound.sort(kind='stable')
    subch = np.ndarray(buffer=compound,
                       dtype=np.uint32,
                       shape=(len(compound), 2))[:,1]
    change = np.concatenate([[0],
                             1 + np.nonzero(subch[:-1] != subch[1:])[0],
                             [len(compound)]]).data
    pos = np.ndarray(buffer=compound,
                     dtype=np.uint32,
                     shape=(len(compound), 2))[:,0]
    return {subch[start]: pos[start:end]
            for start, end in zip(change[:-1], change[1:])}

def to2s16(v):
    sign_bit = v >> 15
    return np.asarray((v ^ (sign_bit * 0x7fff)) + sign_bit,
                      dtype=np.int16)

def new_sector_time(data_view, d): # 4
    data_view.lap_crossing = data_view.sliding_u8[1:][d]
    data_view.lap_crossing = d[data_view.lap_crossing == 0]
    data_view.lap_crossing = data_view.sliding_u32[2:][data_view.lap_crossing]
    # sometimes lap_crossing will be measured in GPS time, others it
    # will be measured in logger time...
    avg_dist = np.abs(np.mean(data_view.gps_time_map, axis=0) - np.mean(data_view.lap_crossing))
    if avg_dist[0] < avg_dist[1]: # seems to be gps time
        data_view.lap_crossing = np.interp(data_view.lap_crossing,
                                           data_view.gps_time_map[:,0],
                                           data_view.gps_time_map[:,1])
    return []

def accelerations(data_view, d): # 8
    tc = data_view.ts_lookup(d)
    return [base.Channel(tc,
                         to2s16(data_view.sliding_u16[1:][d].byteswap()) * (-1/256),
                         'Lateral G', 'G', 2, True),
            base.Channel(tc,
                         to2s16(data_view.sliding_u16[3:][d].byteswap()) * (-1/256),
                         'Longitudinal G', 'G', 2, True)]

def gps_position(data_view, d): # 10
    lat = data_view.sliding_i32[5:][d].byteswap() * 1e-7
    lon = data_view.sliding_i32[1:][d].byteswap() * 1e-7
    # need to manually filter out 90, 0
    good_data = (lat != 90) | (lon != 0)
    tc = data_view.ts_lookup(d[good_data])
    return [base.Channel(tc, lon[good_data],
                         'GPS Longitude', 'deg', 7, True),
            base.Channel(tc, lat[good_data],
                         'GPS Latitude', 'deg', 7, True)]

def speed_data(data_view, d): # 11
    return [base.Channel(data_view.ts_lookup(d),
                         data_view.sliding_i32[1:][d].byteswap() * 3.6e-2,
                         'Speed', 'km/h', 1, True)]

def rpm(data_view, d): # 18
    return [base.Channel(data_view.ts_lookup(d),
                         360e6 / (data_view.sliding_u32[d].byteswap() & 0xffffff),
                         'RPM', 'rpm', 0, True)]

def analog_input(data_view, d): # 20-51
    i = data_view.sliding_u8[d[0]]
    return [base.Channel(data_view.ts_lookup(d),
                         data_view.sliding_u16[1:][d].byteswap() * 1e-3,
                         'Analog %d' % (i - 19), 'V', 3, True)]

def data_storage_channel(data_view, d): # 55
    # XXX should save timestamp to metadata
    return []
    raise ValueError
    validate(state, pos, 9)
    #print('tod %d:%02d:%02d' % (state.data[pos+3], state.data[pos+2], state.data[pos+1]))
    # Current GMT time is given by
    # Seconds = Data1
    # Minutes = Data2
    # Hours = Data3
    # Date = Data4
    # Month = Data5
    # Year = Data6 x 2^8 + Data7
    # Offset from GMT = Data8 (2's complement)
    # Offset from GMT is given by 15 minutes increments or decrements
    # For example (-22) = (- 5:30 GMT)
    return pos + 10

def external_temperature_sensor(data_view, d): # 72
    names = [None,
             'Ambient Air Temperature',
             'Inlet Pre Turbo Temp 1',
             'Inlet Pre Turbo Temp 2',
             'Inlet Post Turbo Temp 1',
             'Inlet Post Turbo Temp 2',
             'Inlet Post Intercooler Temp 1',
             'Inlet Post Intercooler Temp 2',
             'Water Temp',
             'Oil Temp',
             'Gearbox Temp',
             'Gearbox Temp Post Cooler',
             'Tyre Temp 1',
             'Tyre Temp 2',
             'Tyre Temp 3',
             'Tyre Temp 4',
             'ECU Temperature',
             'Exhaust Temp 1',
             'Exhaust Temp 2',
             'Exhaust Temp 3',
             'Exhaust Temp 4',
             'Exhaust Temp 5',
             'Exhaust Temp 6',
             'Exhaust Temp 7',
             'Exhaust Temp 8',
             'Auxiliary Temp 1']
    return [base.Channel(data_view.ts_lookup(pos),
                         data_view.sliding_i16[2:][pos] * 1e-1,
                         names[ch], 'C', 1, True)
            for ch, pos in separate_subchannels(data_view.sliding_u8, d).items()]

def external_aux_channel(data_view, d): # 74
    names = [None,
             'Throttle Position',
             'Lambda 1 Short Term Trim',
             'Lambda 2 Short Term Trim',
             'Lambda 1 Long Term Trim',
             'Lambda 2 Long Term Trim',
             'Fuel Inj 1 Pulse Width',
             'Fuel inj 2 PW',
             'Fuel inj 3 PW',
             'Fuel inj 4 PW',
             'Fuel inj 5 PW',
             'Fuel inj 6 PW',
             'Fuel inj 7 PW',
             'Fuel inj 8 PW',
             'Fuel inj 1 cut',
             'Fuel inj 2 cut',
             'Fuel inj 3 cut',
             'Fuel inj 4 cut',
             'Fuel inj 5 cut',
             'Fuel inj 6 cut',
             'Fuel inj 7 cut',
             'Fuel inj 8 cut',
             'Ignition cut',
             'ISBV 1 open',
             'ISBV 2 Open',
             'Nitrous',
             'Auxiliary 1',
             'Auxiliary 2',
             'Auxiliary 3',
             'Auxiliary 4',
             'Fuel aux temp comp',
             'Fuel aux volt comp']
    return [base.Channel(data_view.ts_lookup(pos),
                         data_view.sliding_i16[2:][pos] * 1e-1,
                         names[ch], '', 1, True)
            for ch, pos in separate_subchannels(data_view.sliding_u8, d).items()]

def external_angle_channel(data_view, d): # 93
    names = [None,
             'Throttle Angle',
             'Ignition Angle',
             'Steering Angle']
    return [base.Channel(data_view.ts_lookup(pos),
                         data_view.sliding_i16[2:][pos] * 1e-1,
                         names[ch], 'deg', 1, True)
            for ch, pos in separate_subchannels(data_view.sliding_u8, d).items()]

def external_pressure_channel(data_view, d): # 94
    names = [None,
             'Ambient_Air_Pressure',
             'Oil_Pressure',
             'Fuel_Pressure',
             'Water_Pressure',
             'Boost_Pressure']
    return [
        base.Channel(data_view.ts_lookup(pos),
                     data_view.sliding_i16[3:][pos] * 10. ** (data_view.sliding_i8[2:][pos] - 1),
                     names[ch], 'kPa', 1, True)
        for ch, pos in separate_subchannels(data_view.sliding_u8, d).items()]

def external_miscellaneous_channel(data_view, d): # 95
    names = [(None, None),
             ('Lambda 1', 'lambda'),
             ('Lambda 2', 'lambda'),
             ('Battery Voltage', 'V'),
             ('ECU speed 4', 'km/h'),
             ('ECU distance 5', 'km'),
             ('Internal battery back voltage', 'V')]
    return [base.Channel(data_view.ts_lookup(pos),
                         data_view.sliding_u16[2:][pos] * 1e-2,
                         names[ch][0], names[ch][1], 2, True)
            for ch, pos in separate_subchannels(data_view.sliding_u8, d).items()]

def ignore(data_view, d):
    return []

decoders = [ignore] * 256
decoders[4] = new_sector_time
decoders[8] = accelerations
decoders[10] = gps_position
decoders[11] = speed_data
decoders[18] = rpm
for i in range(20, 52): decoders[i] = analog_input
decoders[55] = data_storage_channel
decoders[72] = external_temperature_sensor
decoders[74] = external_aux_channel
decoders[93] = external_angle_channel
decoders[94] = external_pressure_channel
decoders[95] = external_miscellaneous_channel

g_cmdlen = [2] * 256
g_cmdlen[1] = 9
g_cmdlen[2] = 11
g_cmdlen[3] = 0
g_cmdlen[4] = 12
g_cmdlen[5] = 21
g_cmdlen[6] = 6
g_cmdlen[7] = 6
g_cmdlen[8] = 6
g_cmdlen[9] = 5
g_cmdlen[10] = 14
g_cmdlen[11] = 10
g_cmdlen[12] = 3
g_cmdlen[18] = 5
for i in range(20, 52): g_cmdlen[i] = 4
g_cmdlen[53] = 11
g_cmdlen[55] = 10
g_cmdlen[56] = 10
g_cmdlen[57] = 10
g_cmdlen[63] = 3
g_cmdlen[65] = 30
g_cmdlen[71] = 3
g_cmdlen[72] = 5
g_cmdlen[74] = 5
g_cmdlen[76] = 24
g_cmdlen[77] = 3
g_cmdlen[85] = 10
g_cmdlen[90] = 6
g_cmdlen[91] = 5
g_cmdlen[92] = 4
g_cmdlen[93] = 5
g_cmdlen[94] = 6
g_cmdlen[95] = 5
g_cmdlen[96] = 10
g_cmdlen[104] = 9
g_cmdlen[217] = 6
g_cmdlen[228] = 6

def parse(data, progress):
    crc1 = np.cumsum(np.ndarray(buffer=b'\0' + data,
                                dtype=np.uint8,
                                shape=1+len(data)),
                     dtype=np.uint8)[:-1].tobytes()
    crc2 = b'\0' + np.subtract(np.ndarray(buffer=crc1, dtype=np.uint8, shape=len(crc1)),
                               np.ndarray(buffer=data, dtype=np.uint8, shape=len(data)),
                               dtype=np.uint8).tobytes()
    cmdlen = g_cmdlen # local variables are faster than global variables

    pos = 0
    next_stop = 0
    commands = [array('I') for i in range(256)]
    next_error = 0
    while pos < len(data):
        try:
            # fast common case
            while pos < next_stop:
                cmd = data[pos]
                l = (cmdlen[cmd] or (3 + data[pos + 1])) + pos
                if crc1[pos] == crc2[l]:
                    #print('match pos=%d cmd=%d len=%d' % (pos, data[pos], l-pos))
                    commands[cmd].append(pos)
                    pos = l
                else:
                    if next_error != pos:
                        if next_error + 2 != pos:
                            print()
                    print('crc error (pos=%d, cmd=%d)' % (pos, data[pos]))
                    pos += 1
                    next_error = pos
        except IndexError:
            if next_error != pos:
                if next_error + 2 != pos:
                    print()
            print('out of bounds (pos=%d, cmd=%d)' % (pos, data[pos]))
            pos += 1
            next_error = pos
        if pos >= next_stop:
            if progress:
                progress(pos, len(data))
            next_stop = min(next_stop + 3_000_000, len(data))
    return commands

def RUN(fname, progress):
    t1 = time.perf_counter()
    with open(fname, 'rb') as f:
        data = f.read()
        commands = parse(data, progress)
        t2 = time.perf_counter()
        channels = []

        data_view = DataView(data, commands[9], commands[7])

        t3 = time.perf_counter()
        for i, d in enumerate(commands):
            if d:
                channels.extend(decoders[i](data_view, np.asarray(d)))
    lap_crossing = np.concatenate([[min(ch.timecodes[0] for ch in channels)],
                                   [l for l in data_view.lap_crossing if l],
                                   [max(ch.timecodes[-1] for ch in channels)]]).data
    t4 = time.perf_counter()
    print('decode', t2-t1, t3-t2, t4-t3)
    return base.LogFile({ch.name: ch for ch in channels},
                        [base.Lap(num, start, end)
                         for num, (start, end) in enumerate(zip(lap_crossing[:-1],
                                                                lap_crossing[1:]))],
                        {'Log Date': 'Unknown',
                         'Log Time': 'Unknown'}, # metadata
                        ['Speed', 'GPS Latitude', 'GPS Longitude', None],
                        fname)
