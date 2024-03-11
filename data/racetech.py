
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

def validate(state, pos, expected_len):
    crc = sum(state.data[pos:pos+expected_len]) & 255
    if crc != state.data[pos + expected_len]:
        print('%d: CRC error for len %d cmd %d (%d vs %d)'
              % (pos, expected_len+1, state.data[pos], crc, state.data[pos+expected_len]))
        raise ValueError

def run_information_input_1(state, pos): # 1
    validate(state, pos, 8)
    #print('run information input 1') # no more info provided
    return pos + 9

def run_status_information(state, pos): # 2
    validate(state, pos, 10)
    #print('run status information') # skip, lots of internals, maybe bias will be useful?
    return pos + 11

def raw_gps(state, pos): # 3
    l = state.data[pos + 1]
    validate(state, pos, l + 2)
    #print('raw_gps', ' '.join('%02x' % b for b in state.data[pos + 2:pos + 2 + l][::-1]))
    return pos + l + 3

def new_sector_time(state, pos): # 4
    validate(state, pos, 11)
    state.samples[4].append(state.timecode)
    state.samples[4].append(pos)
    #print('sector time', struct.unpack_from('<xBIBI', state.data, pos)) # ms
    return pos + 12

def new_lap_marker(state, pos): # 5
    validate(state, pos, 20)
    print('lap marker')
    return pos + 21

def logger_storage_channel(state, pos): # 6
    validate(state, pos, 5)
    #print('logger', struct.unpack_from('>IBB', state.data, pos + 1))
    # Serial number = Data2 + Data1 x 2^8
    # Software version = Data3
    # Booload version = Data4
    return pos + 6

def gps_time_storage(state, pos): # 7
    validate(state, pos, 5)
    state.samples[7].append(state.timecode)
    state.samples[7].append(pos)
    #print('gps time of week (ms):', struct.unpack_from('>I', state.data, pos + 1)[0])
    return pos + 6

def accelerations(state, pos): # 8
    validate(state, pos, 5)
    state.samples[8].append(state.timecode)
    state.samples[8].append(pos)
    #print('accelerations',
    #      decode16sign(state.data, pos + 1) / 256,
    #      decode16sign(state.data, pos + 3) / 256)
    return pos + 6

def timestamp(state, pos): # 9
    validate(state, pos, 4)
    state.timecode = pos # hundredths of a second
    return pos + 5

def gps_position(state, pos): # 10
    validate(state, pos, 13)
    state.samples[10].append(state.timecode)
    state.samples[10].append(pos)
    #print('gps pos',
    #      decode32sign(state.data, pos + 1) / 1e7, # long
    #      decode32sign(state.data, pos + 5) / 1e7, # lat
    #      struct.unpack_from('>I', state.data, pos + 9)[0]) # accuracy estimate, mm
    return pos + 14

def speed_data(state, pos): # 11
    validate(state, pos, 9)
    state.samples[11].append(state.timecode)
    state.samples[11].append(pos)
    #print('speed', struct.unpack_from('>II', state.data, pos + 1))
    # Speed (m/s) = (Data1 * 2^24 + Data2 * 2^16 + Data3 * 2^8 + Data4)* 0.01
    # SpeedAcc (m/s) =( Data6 * 2^16 + Data7 * 2^8 + Data8)* 0.01
    # Data source = Data5
    # 0 = raw GPS data
    # 1 = non GPS speed
    return pos + 10

def padding(state, pos): # 12
    validate(state, pos, 2)
    return pos + 3

def rpm(state, pos): # 18
    validate(state, pos, 4)
    state.samples[18].append(state.timecode)
    state.samples[18].append(pos)
    # TickPeriod = 1.66666666666667E-07
    # Frequency = 1/((Data1 * 0x10000 + Data2 * 0x100 + Data3) * TickPeriod)
    return pos + 5

def analog_input(state, pos): # 20-51
    validate(state, pos, 3)
    s = state.samples[state.data[pos]]
    s.append(state.timecode)
    s.append(pos)
    return pos + 4

def display_data_channel(state, pos): # 53
    validate(state, pos, 10)
    #print('display data channel') # internal for use by dash display?
    return pos + 11

def data_storage_channel(state, pos): # 55
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

def course_data(state, pos): # 56
    validate(state, pos, 9)
    #print('heading', struct.unpack_from('>iI', state.data, pos + 1))
    # Heading (degrees / 10^-5) = Data 1, Data 2, Data 3 and Data 4 (big endian, 2's complement signed 32bit integer).
    # Heading accuracy (degrees / 10^-5) = ( (Data5 & 0x7F) x 2^24 + Data6 x 2^16 + Data7 x 2^8 + Data8)
    # Data source = Data5 & 0x80
    # Data source values:
    # 0      raw GPS data
    # 1      processed data
    return pos + 10

def gps_altitude(state, pos): # 57
    validate(state, pos, 9)
    state.samples[57].append(state.timecode)
    state.samples[57].append(pos)
    #print('gps alt', struct.unpack_from('>II', state.data, pos + 1)) # altitude and accuracy, mm
    return pos + 10

def start_of_run(state, pos): # 63
    validate(state, pos, 2)
    #print('start_of_run')
    return pos + 3

def gear_setup(state, pos): # 65
    validate(state, pos, 29)
    #print('gear setup') # to do what exactly?
    return pos + 30

def aux_input_module(state, pos): # 71
    validate(state, pos, 2)
    return pos + 3

def external_temperature_sensor(state, pos): # 72
    validate(state, pos, 4)
    s = state.samples[0x4800 + state.data[pos + 1]]
    s.append(state.timecode)
    s.append(pos)
    # [1] = sensor type/name
    # [2-3] = little endian 2's complement in 0.1C?
    return pos + 5

def external_aux_channel(state, pos): # 74
    validate(state, pos, 4)
    s = state.samples[0x4a00 + state.data[pos + 1]]
    s.append(state.timecode)
    s.append(pos)
    # see doc
    return pos + 5

def new_lcd_data_channel(state, pos): # 76
    validate(state, pos, 23)
    return pos + 24

def new_led_data_channel(state, pos): # 77
    validate(state, pos, 2)
    #print('new led data channel') # don't care
    return pos + 3

def gradient_channel(state, pos): # 85
    validate(state, pos, 9)
    # is this hill angle?
    return pos + 10

def baseline(state, pos): # 90
    validate(state, pos, 5)
    #print('baseline', struct.unpack_from('>HH', data, pos + 1)) # ? baseline and accuracy
    return pos + 6

def unit_control_channel(state, pos): # 91
    validate(state, pos, 4)
    return pos + 5

def z_accel(state, pos): # 92
    validate(state, pos, 3)
    #print('z_accel', decode16sign(data, pos + 1) / 256)
    return pos + 4

def external_angle_channel(state, pos): # 93
    validate(state, pos, 4)
    s = state.samples[0x5d00 + state.data[pos + 1]]
    s.append(state.timecode)
    s.append(pos)
    return pos + 5

def external_pressure_channel(state, pos): # 94
    validate(state, pos, 5)
    s = state.samples[0x5e00 + state.data[pos + 1]]
    s.append(state.timecode)
    s.append(pos)

    # Sensor location = Data1
    # Default channel names are as follows but can be changed by the user as required.
    # Ambient_Air_Pressure 1
    # Oil_Pressure 2
    # Fuel_Pressure 3
    # Water_Pressure 4
    # Boost_Pressure 5
    # Scaling factor = Data2 (2's complement)
    # Pressure (mb) = (Data3 + Data4 * 2^8) * 10^Data2
    # The range on the pressure channel is 0 - 65535mb. The resolution is 1mb and the units are mb.
    return pos + 6

def external_miscellaneous_channel(state, pos): # 95
    validate(state, pos, 4)
    s = state.samples[0x5f00 + state.data[pos + 1]]
    s.append(state.timecode)
    s.append(pos)
    # [1] = sensor type/name including units
    # [2-3] = little endian unsigned?
    return pos + 5

def time_into_lap_sector(state, pos): # 96
    validate(state, pos, 9)
    # seems mostly used for dash display ('last lap time' etc)
    #sector_time_ms = state.data[pos+1] * 65536 + state.data[pos+2] * 256 + state.data[pos+3]
    #lap_time_ms = state.data[pos+4] * 65536 + state.data[pos+5] * 256 + state.data[pos+6]
    #time_slip_rate_perc = state.data[pos+7] / 5 # signed?
    #time_slip_s = state.data[pos+8] / 10 # signed?
    #print('time_into_lap_sector', sector_time_ms / 1000, lap_time_ms / 1000,
    #      time_slip_rate_perc, time_slip_s)
    return pos + 10

def general_comms_channel(state, pos): # 102
    l = state.data[pos + 1] + 3
    validate(state, pos, l - 1)
    return pos + l

def video_frame_index(state, pos): # 104
    validate(state, pos, 8)
    #vf = []
    #if data[pos + 6] & 0x80:
    #    vf.append('video_pts=%d' % struct.unpack_from('>I', data, pos + 1))
    #else:
    #    vf.append('video_frame_idx=%d' % struct.unpack_from('<I', data, pos + 1))
    #vf.append('cpu_load=%d' % (data[pos + 5] / 2))
    # ignore later details
    #print('video_frame_index', ' '.join(vf))
    return pos + 9

def unknown_217(state, pos): # 217
    validate(state, pos, 5)
    #print('unknown 217')
    return pos + 6

def unknown_228(state, pos): # 228
    validate(state, pos, 5)
    #print('unknown 228')
    return pos + 6

def error(state, pos):
    print('%d: unknown cmd %d' % (pos, state.data[pos]))
    raise ValueError

decoders = [error] * 256
decoders[1] = run_information_input_1
decoders[2] = run_status_information
decoders[3] = raw_gps
decoders[4] = new_sector_time
decoders[5] = new_lap_marker
decoders[6] = logger_storage_channel
decoders[7] = gps_time_storage
decoders[8] = accelerations
decoders[9] = timestamp
decoders[10] = gps_position
decoders[11] = speed_data
decoders[12] = padding
decoders[18] = rpm
for i in range(20, 52): decoders[i] = analog_input
decoders[53] = display_data_channel
decoders[55] = data_storage_channel
decoders[56] = course_data
decoders[57] = gps_altitude
decoders[63] = start_of_run
decoders[65] = gear_setup
decoders[71] = aux_input_module
decoders[72] = external_temperature_sensor
decoders[74] = external_aux_channel
decoders[76] = new_lcd_data_channel
decoders[77] = new_led_data_channel
decoders[85] = gradient_channel
decoders[90] = baseline
decoders[91] = unit_control_channel
decoders[92] = z_accel
decoders[93] = external_angle_channel
decoders[94] = external_pressure_channel
decoders[95] = external_miscellaneous_channel
decoders[96] = time_into_lap_sector
decoders[102] = general_comms_channel
decoders[104] = video_frame_index
decoders[217] = unknown_217
decoders[228] = unknown_228

def parse(state, progress):
    pos = 0
    next_stop = 0

    while pos < len(state.data):
        try:
            # fast common case
            while pos < next_stop:
                pos = decoders[state.data[pos]](state, pos)
        except (ValueError, IndexError):
            pos = pos + 1
            # slow roll coming back up so we can print the ok status
            while pos < len(state.data):
                try:
                    pos = decoders[state.data[pos]](state, pos)
                    print('ok')
                    break
                except (ValueError, IndexError):
                    pos = pos + 1
        if pos >= next_stop:
            if progress:
                progress(pos, len(state.data))
            next_stop += 1_000_000

def to2s16(v):
    sign_bit = v >> 15
    return np.asarray((v ^ (sign_bit * 0x7fff)) + sign_bit,
                      dtype=np.int16)

def to2s32(v):
    sign_bit = v >> 31
    return np.asarray((v ^ (sign_bit * 0x7fff_ffff)) + sign_bit,
                      dtype=np.int32)

def RUN(fname, progress):
    t1 = time.perf_counter()
    with open(fname, 'rb') as f:
        data = f.read()
        state = State(data, b'', [array('I') for i in range(65536)])
        t2 = time.perf_counter()
        parse(state, progress)
        t3 = time.perf_counter()
        channels = []
        sliding_i8 = np.ndarray(buffer = data,
                                dtype = np.int8,
                                shape = len(data))
        sliding_u8 = np.ndarray(buffer = data,
                                dtype = np.uint8,
                                shape = len(data))
        sliding_i16 = np.ndarray(buffer = data,
                                 dtype = np.int16,
                                 strides = 1,
                                 shape = len(data) - 1)
        sliding_u16 = np.ndarray(buffer = data,
                                 dtype = np.uint16,
                                 strides = 1,
                                 shape = len(data) - 1)
        sliding_i32 = np.ndarray(buffer = data,
                                 dtype = np.int32,
                                 strides = 1,
                                 shape = len(data) - 3)
        sliding_u32 = np.ndarray(buffer = data,
                                 dtype = np.uint32,
                                 strides = 1,
                                 shape = len(data) - 3)

        # first, manually process #7: gps_time_storage
        # this allows other channels to map gps time->log time
        d = state.samples[7]
        d = np.asarray(d).reshape(len(d) // 2, 2)
        gps_time_map = np.column_stack([sliding_u32[1:][d[:,1]].byteswap(), # gps time
                                        (sliding_u32[d[:,0]].byteswap() & 0xffffff) * 10]) # tc
        lap_crossing = np.array([], dtype=np.uint32)
        for i, d in enumerate(state.samples):
            if d:
                d = np.asarray(d).reshape(len(d) // 2, 2)
                tc = (sliding_u32[d[:,0]].byteswap() & 0xffffff) * 10
                if i == 4: # sector times
                    lap_crossing = sliding_u8[1:][d[:,1]]
                    lap_crossing = d[:,1][lap_crossing == 0]
                    lap_crossing = sliding_u32[2:][lap_crossing]
                    # sometimes lap_crossing will be measured in GPS
                    # time, others it will be measured in logger
                    # time...
                    avg_dist = np.abs(np.mean(gps_time_map, axis=0) - np.mean(lap_crossing))
                    if avg_dist[0] < avg_dist[1]: # seems to be gps time
                        lap_crossing = np.interp(lap_crossing,
                                                 gps_time_map[:,0],
                                                 gps_time_map[:,1])
                elif i == 7: # gps_time_storage
                    pass # handled separately
                elif i == 8: # accelerometers
                    channels.append(
                        base.Channel(tc,
                                     to2s16(sliding_u16[1:][d[:,1]].byteswap()) * 1e-2,
                                     'Lateral G', 'G', 2, True))
                    channels.append(
                        base.Channel(tc,
                                     to2s16(sliding_u16[3:][d[:,1]].byteswap()) * 1e-2,
                                     'Longitudinal G', 'G', 2, True))
                elif i == 10: # gps position
                    lat = sliding_i32[5:][d[:,1]].byteswap() * 1e-7
                    lon = sliding_i32[1:][d[:,1]].byteswap() * 1e-7
                    # need to manually filter out 90, 0
                    good_data = (lat != 90) | (lon != 0)
                    channels.append(
                        base.Channel(tc[good_data], lon[good_data],
                                     'GPS Longitude', 'deg', 7, True))
                    channels.append(
                        base.Channel(tc[good_data], lat[good_data],
                                     'GPS Latitude', 'deg', 7, True))
                elif i == 11: # speed
                    channels.append(
                        base.Channel(tc,
                                     sliding_i32[1:][d[:,1]].byteswap() * 3.6e-2,
                                     'Speed', 'km/h', 1, True))
                elif i == 18: # RPM
                    channels.append(
                        base.Channel(tc,
                                     360e6 / (sliding_u32[d[:,1]].byteswap() & 0xffffff),
                                     'RPM', 'rpm', 0, True))
                elif i >= 20 and i < 52: # analog input
                    channels.append(
                        base.Channel(tc,
                                     sliding_u16[1:][d[:,1]].byteswap() * 1e-3,
                                     'Analog %d' % (i - 19), 'V', 3, True))
                elif i == 57: # gps altitude
                    pass # skip for now, maybe read but interpolate using gps lat/lon timecodes?
                elif i // 256 == 72: # external temp sensor
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
                    channels.append(
                        base.Channel(tc,
                                     sliding_i16[2:][d[:,1]] * 1e-1,
                                     names[i & 255], 'C', 1, True))
                elif i // 256 == 74: # external aux channel
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
                    channels.append(
                        base.Channel(tc,
                                     sliding_i16[2:][d[:,1]] * 1e-1,
                                     names[i & 255], '', 1, True))
                elif i // 256 == 93: # external angle sensor
                    names = [None,
                             'Throttle Angle',
                             'Ignition Angle',
                             'Steering Angle']
                    channels.append(
                        base.Channel(tc,
                                     sliding_i16[2:][d[:,1]] * 1e-1,
                                     names[i & 255], 'deg', 1, True))
                elif i // 256 == 94: # external pressure sensor
                    names = [None,
                             'Ambient_Air_Pressure',
                             'Oil_Pressure',
                             'Fuel_Pressure',
                             'Water_Pressure',
                             'Boost_Pressure']
                    channels.append(
                        base.Channel(tc,
                                     sliding_i16[3:][d[:,1]] * 10. ** (sliding_i8[2:][d[:,1]] - 1),
                                     names[i & 255], 'kPa', 1, True))
                elif i // 256 == 95: # external misc sensor
                    names = [(None, None),
                             ('Lambda 1', 'lambda'),
                             ('Lambda 2', 'lambda'),
                             ('Battery Voltage', 'V'),
                             ('ECU speed 4', 'km/h'),
                             ('ECU distance 5', 'km'),
                             ('Internal battery back voltage', 'V')]
                    channels.append(
                        base.Channel(tc,
                                     sliding_u16[2:][d[:,1]] * 1e-2,
                                     names[i & 255][0], names[i & 255][1], 2, True))
                else:
                    print('-- unhandled %d' % i)
    lap_crossing = np.concatenate([[min(ch.timecodes[0] for ch in channels)],
                                   [l for l in lap_crossing if l],
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
