
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
from dataclasses import dataclass, field
import math
import mmap
import numpy as np
from pprint import pprint # pylint: disable=unused-import
import struct
import sys
import traceback # pylint: disable=unused-import
from typing import Dict, List, Optional

from . import gps
from . import base

# 1,2,5,10,20,25,50 Hz
# units
# dec ptr

dc_slots = {'slots': True} if sys.version_info.minor >= 10 else {}

@dataclass(**dc_slots)
class Group:
    index: int
    channels: List[int]
    samples: bytearray = field(default_factory=bytearray, repr=False)
    timecodes: Optional[array] = None
    # used during building:
    add_helper: int = 0
    last_timecode: int = -1

@dataclass(**dc_slots)
class GroupRef:
    group: Group
    offset: int

@dataclass(**dc_slots)
class Channel:
    index: int = -1
    short_name: str = ""
    long_name: str = ""
    size: int = 0
    units: str = ""
    dec_pts: int = 0
    unknown: bytes = b""
    group: Optional[GroupRef] = None
    timecodes: object = field(default=None, repr=False)
    sampledata: object = field(default=None, repr=False)
    # used during building:
    add_helper: int = 0

@dataclass(**dc_slots)
class Message:
    token: bytes
    num: int
    content: bytes

@dataclass(**dc_slots)
class DataStream:
    channels: Dict[str, Channel]
    messages: List[Message]

@dataclass(**dc_slots)
class Decoder:
    stype: str
    fixup: object = None

def _nullterm_string(s):
    zero = s.find(0)
    if zero >= 0: s = s[:zero]
    return s.decode('ascii')

_manual_decoders = {
    'Calculated_Gear': Decoder('Q', fixup=lambda a: array('I', [0 if int(x) & 0x80000 else
                                                                (int(x) >> 16) & 7 for x in a])),
    'PreCalcGear':     Decoder('Q', fixup=lambda a: array('I', [0 if int(x) & 0x80000 else
                                                                (int(x) >> 16) & 7 for x in a])),
}

_gear_table = {
    'N': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
}

_decoders = {
    0:  Decoder('i'), # Master Clock on M4GT4?
    1:  Decoder('H', fixup=lambda a: np.ndarray(buffer=a, shape=(len(a),),
                                                dtype=np.float16).astype(np.float32).data),
    3:  Decoder('i'), # Master Clock on ScottE46?
    4:  Decoder('h'),
    6:  Decoder('f'),
    11: Decoder('h'),
    12: Decoder('i'), # Predictive Time?
    13: Decoder('B'), # status field?
    15: Decoder('Q', fixup=lambda a: array('i', [_gear_table.get(chr(int(x) & 0xffff), int(x) & 0xffff) for x in a])), # ?? NdscSwitch on M4GT4
    20: Decoder('H', fixup=lambda a: np.ndarray(buffer=a, shape=(len(a),),
                                                dtype=np.float16).astype(np.float32).data),
    24: Decoder('i'), # Best Run Diff?
}

_unit_map = {
    1:  ('%', 2),
    3:  ('G', 2),
    4:  ('deg', 1),
    5:  ('deg/s', 1),
    6:  ('', 0), # number
    9:  ('Hz', 0),
    11: ('', 0), # number
    12: ('mm', 0),
    14: ('bar', 2),
    15: ('rpm', 0),
    16: ('km/h', 0),
    17: ('C', 1),
    18: ('ms', 0),
    19: ('Nm', 0),
    20: ('km/h', 0),
    21: ('V', 1), # mv?
    22: ('l', 1),
    24: ('l/s', 0), # ? rs3 displayed 1l/h
    26: ('time?', 0),
    27: ('A', 0),
    30: ('lambda', 2),
    31: ('gear', 0),
    33: ('%', 2),
}

def ndarray_from_mv(mv):
    mv = memoryview(mv) # force it
    return np.ndarray(buffer=mv, shape=(len(mv),), dtype=np.dtype(mv.format))

def _decode_sequence(s, progress=None):
    groups = {}
    channels = {}
    messages = []
    next_progress = 1_000_000
    pos = 0
    badbytes = bytearray()
    xBIH_decoder = struct.Struct('<xBIH')
    xH_decoder = struct.Struct('<8xH')
    Mms = {
        32: 20,
        64: 40,
    }
    ord_op = ord('(')
    ord_cp = ord(')')
    ord_G  = ord('G')
    ord_S  = ord('S')
    ord_M  = ord('M')
    ord_lt = ord('<')
    len_s  = len(s)
    while pos < len_s:
        try:
            while s[pos] == ord_op:
                oldpos = pos
                typ, tc, idx = xBIH_decoder.unpack_from(s, pos)
                if typ == ord_G:
                    g = groups[idx]
                    pos += g.add_helper
                    assert s[pos] == ord_cp, "%c at %x" % (s[pos], pos)
                    if tc > g.last_timecode:
                        g.samples += s[oldpos:pos]
                        g.last_timecode = tc
                    pos += 1
                elif typ == ord_S:
                    ch = channels[idx]
                    # print(hdr[1], ch)
                    pos += ch.add_helper
                    assert s[pos] == ord_cp, "%c at %x" % (s[pos], pos)
                    ch.timecodes.append(tc)
                    ch.sampledata += s[oldpos+8:pos]
                    pos += 1
                elif typ == ord_M:
                    cnt, = xH_decoder.unpack_from(s, pos)
                    ch = channels[idx]
                    pos += ch.size * cnt + 10
                    assert s[pos] == ord_cp, "%c at %x" % (s[pos], pos)
                    ms = Mms[ch.unknown[64]]
                    ch.timecodes += array('i', [tc + off for off in range(0, cnt*ms, ms)])
                    ch.sampledata += s[oldpos+10:pos]
                    pos += 1
                else:
                    assert False, "%c at %x" % (s[pos], pos)
            oldpos = pos
            if s[pos] == ord_lt:
                if pos > next_progress:
                    next_progress += 1_000_000
                    if progress:
                        progress(pos, len(s))
                pos += 1
                assert s[pos] == ord('h'), "%s at %x" % (s[pos], pos)
                pos += 1
                tok = s[pos:pos + 4]
                pos += 4
                l = struct.unpack_from('<IB', s, pos)
                pos += 5
                assert s[pos] == ord('>'), "%c at %x" % (s[pos], pos)
                pos += 1

                data = s[pos:pos + l[0]]
                bytesum = sum(data) & 0xffff
                pos += l[0]

                assert s[pos] == ord('<'), "%s at %x" % (s[pos], pos)
                pos += 1
                assert s[pos:pos + 4] == tok, "%s vs %s at %x" % (s[pos:pos + 4],
                                                                  tok, pos)
                pos += 4
                l2 = struct.unpack_from('<H', s, pos)
                pos += 2
                assert s[pos] == ord('>'), "%c at %x" % (s[pos], pos)
                pos += 1

                tok = tok.rstrip(b'\0 ').decode('ascii')

                if tok == 'CNF':
                    data = _decode_sequence(data).messages
                    #channels = {} # Replays don't necessarily contain all the original channels
                    for m in data:
                        if m.token == 'CHS':
                            if m.content.index not in channels:
                                channels[m.content.index] = m.content
                            else:
                                assert channels[m.content.index].short_name == m.content.short_name, "%s vs %s" % (channels[m.content.index].short_name, m.content.short_name)
                                assert channels[m.content.index].long_name == m.content.long_name
                        #elif t == 'CDE':
                        #    print(t, chunk)
                        elif m.token == 'GRP':
                            groups[m.content.index] = m.content
                            #print('GROUP', m.content.index, len(m.content.channels),
                            #      [(channels[ch].long_name, channels[ch].size) for ch in m.content.channels])

                            m.content.add_helper = 8 + sum(channels[ch].size
                                                           for ch in m.content.channels)
                elif tok == 'GRP':
                    data = [x[0] for x in struct.iter_unpack('<H', data)]
                    assert data[1] == len(data[2:])
                    data = Group(index = data[0], channels = data[2:])
                elif tok == 'CDE':
                    data = ['%02x' % x for x in data]
                elif tok == 'CHS':
                    dcopy = bytearray(data) # copy
                    data = Channel()
                    (data.index,
                     data.short_name,
                     data.long_name,
                     data.size) = struct.unpack('<H22x8s24s16xB39x', dcopy)
                    data.units, data.dec_pts = _unit_map[dcopy[12] & 127]
                    data.add_helper = data.size + 8

                    # [12] maybe type (lower bits) combined with scale or ??
                    # [13] decoder of some type?
                    # [20] possibly how to decode bytes
                    # [64] data rate.  32=50Hz, 64=25Hz, 80=20Hz, 160=10Hz.  What about 5Hz, 2Hz, 1Hz?
                    # [84] decoder of some type?
                    dcopy[0:2] = [0] * 2 # reset index
                    dcopy[24:32] = [0] * 8 # short name
                    dcopy[32:56] = [0] * 24 # long name
                    data.unknown = bytes(dcopy)
                    data.short_name = _nullterm_string(data.short_name)
                    data.long_name = _nullterm_string(data.long_name)
                    data.timecodes = array('i')
                    data.sampledata = bytearray()
                elif tok in ('RCR', 'VEH', 'CMP', 'VTY', 'NDV', 'TMD', 'TMT',
                             'DBUN', 'DBUT', 'DVER', 'MANL', 'MODL', 'MANI',
                             'MODI', 'HWNF', 'PDLT', 'NTE'):
                    data = _nullterm_string(data)
                elif tok == 'ENF':
                    data = _decode_sequence(data).messages
                elif tok == 'TRK':
                    data = {'name': _nullterm_string(data[:32]),
                            'sf_lat': memoryview(data).cast('i')[9] / 1e7,
                            'sf_long': memoryview(data).cast('i')[10] / 1e7}
                elif tok == 'ODO':
                    # not sure how to map fuel.
                    # Fuel Used channel claims 8.56l used (2046.0-2037.4)
                    # Fuel Used odo says 70689.
                    data = {_nullterm_string(data[i:i+16]):
                            {'time': memoryview(data[i+16:i+24]).cast('I')[0], # seconds
                             'dist': memoryview(data[i+16:i+24]).cast('I')[1]} # meters
                            for i in range(0, len(data), 64)
                            # not sure how to parse fuel, doesn't match any expected units
                            if not _nullterm_string(data[i:i+16]).startswith('Fuel')}

                assert l2[0] == bytesum, '%x vs %x at %x' % (l2[0], bytesum, pos)
                messages.append(Message(tok, l[1], data))
            else:
                assert False, "%c at %x" % (s[pos], pos)
        except Exception as _err: # pylint: disable=broad-exception-caught
            if not badbytes:
                # traceback.print_exc()
                badpos = oldpos # pylint: disable=unused-variable
            badbytes.append(s[oldpos])
            pos = oldpos + 1
        else:
            if badbytes:
                #print('Bad bytes(%d at %x):' % (len(badbytes), badpos), bytes(badbytes))
                badbytes = bytearray()
    assert pos == len(s)
    for g in groups.values():
        idx = 8
        for ch in g.channels:
            channels[ch].group = GroupRef(g, idx)
            idx += channels[ch].size
        g.samples = memoryview(g.samples)
        g.timecodes = np.ndarray(buffer=g.samples[2:], dtype=np.int32,
                                 shape=(len(g.samples) // g.add_helper),
                                 strides=(g.add_helper,)).copy().data
    for c in channels.values():
        if c.long_name in _manual_decoders:
            d = _manual_decoders[c.long_name]
        elif c.unknown[20] in _decoders:
            d = _decoders[c.unknown[20]]
        else:
            continue

        if c.group:
            grp = c.group.group
            c.timecodes = grp.timecodes
            c.sampledata = np.ndarray(buffer=grp.samples[c.group.offset:], dtype=np.dtype(d.stype),
                                      shape=(len(grp.timecodes),),
                                      strides=(grp.add_helper,)).copy().data
        else:
            tc = ndarray_from_mv(c.timecodes)
            samp = ndarray_from_mv(memoryview(c.sampledata).cast(d.stype))
            pick = np.unique(np.maximum.accumulate(tc), return_index=True)[1]
            c.timecodes = tc[pick].data
            c.sampledata = samp[pick].data

        if d.fixup:
            c.sampledata = memoryview(d.fixup(c.sampledata))

    return DataStream(
        channels={ch.long_name: ch for ch in channels.values()
                  if len(ch.sampledata) and ch.long_name not in ('StrtRec', 'MasterClk')},
        messages=messages)

def _get_metadata(msg_by_type):
    ret = {}
    for msg, name in [('RCR', 'Driver'),
                      ('VEH', 'Vehicle'),
                      ('TMD', 'Log Date'),
                      ('TMT', 'Log Time'),
                      ('VTY', 'Session'),
                      ('CMP', 'Series'),
                      ('NTE', 'Long Comment'),
                      ]:
        if msg in msg_by_type:
            ret[name] = msg_by_type[msg][-1].content
    if 'TRK' in msg_by_type:
        ret['Venue'] = msg_by_type['TRK'][-1].content['name']
        # ignore the start/finish line?
    if 'ODO' in msg_by_type:
        for name, stats in msg_by_type['ODO'][-1].content.items():
            ret['Odo/%s Distance (km)' % name] = stats['dist'] / 1000
            ret['Odo/%s Time' % name] = '%d:%02d:%02d' % (stats['time'] // 3600,
                                                          stats['time'] // 60 % 60,
                                                          stats['time'] % 60)
    return ret

def _decode_gps(channels, msg_by_type, time_offset):
    # look for either GPS or GPS1 messages
    gpsmsg = msg_by_type.get('GPS', msg_by_type.get('GPS1', None))
    if not gpsmsg: return
    alldata = memoryview(b''.join(m.content for m in gpsmsg))
    assert len(alldata) % 56 == 0
    timecodes = alldata[0:].cast('i')[::56//4]
    #itow_ms = alldata[4:].cast('I')[::56//4]
    #weekN = alldata[12:].cast('H')[::56//2]
    ecefX_cm = alldata[16:].cast('i')[::56//4]
    ecefY_cm = alldata[20:].cast('i')[::56//4]
    ecefZ_cm = alldata[24:].cast('i')[::56//4]
    #posacc_cm = alldata[28:].cast('i')[::56//4]
    ecefdX_cms = alldata[32:].cast('i')[::56//4]
    ecefdY_cms = alldata[36:].cast('i')[::56//4]
    ecefdZ_cms = alldata[40:].cast('i')[::56//4]
    #velacc_cms = alldata[44:].cast('i')[::56//4]
    #nsat = alldata[51::56]

    timecodes = memoryview(np.subtract(timecodes, time_offset))

    channels['GPS Speed'] = Channel(
        long_name='GPS Speed',
        units='m/s',
        dec_pts=1,
        timecodes=timecodes,
        sampledata=memoryview(np.sqrt(np.square(ecefdX_cms) +
                                      np.square(ecefdY_cms) +
                                      np.square(ecefdZ_cms)) / 100.))

    gpsconv = gps.ecef2lla(np.divide(ecefX_cm, 100),
                           np.divide(ecefY_cm, 100),
                           np.divide(ecefZ_cm, 100))

    channels['GPS Latitude'] = Channel(long_name='GPS Latitude',  units='deg', dec_pts=4,
                                       timecodes=timecodes, sampledata=memoryview(gpsconv.lat))
    channels['GPS Longitude'] = Channel(long_name='GPS Longitude', units='deg', dec_pts=4,
                                        timecodes=timecodes, sampledata=memoryview(gpsconv.long))
    channels['GPS Altitude'] = Channel(long_name='GPS Altitude', units='m', dec_pts=1,
                                       timecodes=timecodes, sampledata=memoryview(gpsconv.alt))

def _get_laps(channels, msg_by_type, time_offset):
    ret = []
    if 'LAP' in msg_by_type:
        for m in msg_by_type['LAP']:
            # 2nd byte is segment #, see M4GT4
            segment, lap, duration, end_time = struct.unpack('xBHIxxxxxxxxI', m.content)
            end_time -= time_offset
            if (not ret or ret[-1].num != lap) and segment == 0:
                assert not ret or ret[-1].num + 1 == lap # deal with missing data later
                ret.append(base.Lap(lap, end_time - duration, end_time))
    try:
        lat_ch = channels['GPS Latitude']
        lon_ch = channels['GPS Longitude']
    except KeyError:
        # If we don't have GPS data, just return whatever laps we found.
        return ret
    # otherwise, we do gps lap insert.

    # gps lap insert.  We assume the start finish "line" is a
    # plane containing the vector that goes through the GPS
    # coordinates sf lat/long from altitude 0 to 1000.  The normal
    # of the plane is generally in line with the direction of
    # travel, given the above constraint.

    # O, D = vehicle vector (O=origin, D=direction, [0]=O, [1]=O+D)
    # SO, SD = start finish origin, direction (plane must contain SO and SO+SD poitns)
    # SN = start finish plane normal

    # D = a*SD + SN
    # 0 = SD . SN
    # combine to get:  0 = SD . (D - a*SD)
    #                  a * (SD . SD) = SD . D
    # plug back into first eq:
    # SN = D - (SD . D) / (SD . SD) * SD
    # or to avoid division, and because length doesn't matter:
    # SN = (SD . SD) * D - (SD. D) * SD

    # now determine intersection with plane SO,SN from vector O,O+D:
    # SN . (O + tD - SO) = 0
    # t * (D . SN) + SN . (O - SO) = 0
    # t = -SN.(O-SO) / D.SN

    track = msg_by_type['TRK'][-1].content
    SO = np.array(gps.lla2ecef(track['sf_lat'], track['sf_long'], 0)).reshape((1, 3))
    SD = np.array(gps.lla2ecef(track['sf_lat'], track['sf_long'], 1000)).reshape((1, 3)) - SO

    O = np.concatenate([x.reshape((len(x), 1))
                        for x in gps.lla2ecef(np.array(lat_ch.sampledata),
                                              np.array(lon_ch.sampledata), 0)],
                       axis=1) - SO
    timecodes = np.array(lat_ch.timecodes)

    D = O[1:] - O[:-1]
    O = O[:-1]

    # Precalculate in which time periods we were traveling at least 4 m/s (~10mph)
    minspeed = np.sum(D*D, axis=1) > np.square((timecodes[1:] - timecodes[:-1]) * (4 / 1000))

    SN = (np.sum(SD * SD, axis=1).reshape((len(SD), 1)) * D
          - np.sum(SD * D, axis=1).reshape((len(D), 1)) * SD)
    t = np.maximum(-np.sum(SN * O, axis=1) / np.sum(SN * D, axis=1), 0)
    # This only works because the track is considered at altitude 0
    dist = np.sum(np.square(O + t.reshape((len(t), 1)) * D), axis=1)
    pick = (t[1:] <= 1) & (t[:-1] > 1) & (dist[1:] < 20 ** 2)

    # Now that we have a decent candidate selection of lap
    # crossings, generate a single normal vector for the
    # start/finish line to use for all lap crossings, to make the
    # lap times more accurate/consistent.  Weight the crossings by
    # velocity and add them together.  As it happens, SN is
    # already weighted by velocity...
    SN = np.sum(SN[1:][pick & minspeed[1:]], axis=0).reshape((1,3))
    # recompute t, dist, pick
    t = np.maximum(-np.sum(SN * O, axis=1) / np.sum(SN * D, axis=1), 0)
    dist = np.sum(np.square(O + t.reshape((len(t), 1)) * D), axis=1)
    pick = (t[1:] <= 1) & (t[:-1] > 1) & (dist[1:] < 20 ** 2)

    # grab the earliest seen timecode from either provided laps or channel (including GPS) data
    lap_markers = [min(([ret[0].start_time] if ret else []) +
                       [ch.timecodes[0] for ch in channels.values()
                        if len(ch.timecodes)])]
    for idx in (np.nonzero(pick)[0] + 1):
        if timecodes[idx] <= lap_markers[-1]:
            continue
        if not minspeed[idx]:
            idx = np.argmax(minspeed[idx:]) + idx
        lap_markers.append(timecodes[idx] + t[idx] * (timecodes[idx+1]-timecodes[idx]))
    # add in the latest seen timecode
    lap_markers.append(max(([ret[-1].end_time] if ret else []) +
                           [ch.timecodes[-1] for ch in channels.values()
                            if len(ch.timecodes)]))

    return [base.Lap(lap, start_time, end_time)
            for lap, (start_time, end_time) in enumerate(zip(lap_markers[:-1],
                                                             lap_markers[1:]))]


def AIMXRK(fname, progress):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            data = _decode_sequence(m, progress)
    msg_by_type = {}
    for m in data.messages:
        msg_by_type.setdefault(m.token, []).append(m)
    #pprint({k: len(v) for k, v in self.msg_by_type.items()})
    # determine time offset
    time_offset = min([ch.timecodes[0] for ch in data.channels.values() if len(ch.timecodes)] +
                      [lap.start_time for lap in _get_laps(data.channels, msg_by_type, 0)[:1]])
    for ch in data.channels.values():
        ch.timecodes = memoryview(np.subtract(ch.timecodes, time_offset))

    # decode GPS data.  Do this after determining time offset
    # since we have to fudge the data to get higher resolution
    _decode_gps(data.channels, msg_by_type, time_offset)

    return base.LogFile(
        {ch.long_name: base.Channel(ch.timecodes,
                                    ch.sampledata,
                                    ch.long_name,
                                    ch.units if ch.size != 1 else '',
                                    ch.dec_pts,
                                    interpolate = ch.size != 1)
         for ch in data.channels.values()},
        _get_laps(data.channels, msg_by_type, time_offset),
        _get_metadata(msg_by_type),
        ['GPS Speed', 'GPS Latitude', 'GPS Longitude', 'GPS Altitude'],
        fname)

#def _help_decode_channels(self, chmap):
#    pprint(chmap)
#    for i in range(len(self.data.channels[0].unknown)):
#        d = sorted([(v.unknown[i], chmap.get(v.long_name, ''), v.long_name)
#                    for v in self.data.channels
#                    if len(v.unknown) > i])
#        if len(set([x[0] for x in d])) == 1:
#            continue
#        pprint((i, d))
#    d = sorted([(len(v.sampledata), chmap.get(v.long_name, ''), v.long_name)
#                for v in self.data.channels])
#    if len(set([x[0] for x in d])) != 1:
#        pprint(('len', d))
