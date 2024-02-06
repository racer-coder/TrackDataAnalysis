
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
from dataclasses import dataclass, field
import math
import mmap
import numpy as np
import pprint
import struct
import sys
import traceback
from typing import Dict, List, Optional

from . import gps

# 1,2,5,10,20,25,50 Hz
# units
# dec ptr

pp = pprint.PrettyPrinter()

dc_slots = {'slots': True} if sys.version_info.minor >= 10 else {}

# We use array and memoryview for efficient operations, but that
# assumes the sizes we expect match the file format.  Lets assert a
# few of those assumptions here.  Our use of struct is safe since it
# has tighter control over byte order and sizing.
assert array('H').itemsize == 2
assert array('I').itemsize == 4
assert array('f').itemsize == 4
assert array('Q').itemsize == 8
assert sys.byteorder == 'little'


@dataclass(**dc_slots)
class Group:
    index: int
    channels: List[int]
    samples: bytearray = field(default_factory=bytearray, repr=False)
    # internal fields:
    _add_helper: int = 0
    _row_size: int = 0
    _last_timecode: int = -1

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
    _last_timecode: int = -1

@dataclass(**dc_slots)
class Lap:
    num: int
    start_time: int
    end_time: int

    def duration(self):
        return self.end_time - self.start_time

@dataclass(**dc_slots)
class Message:
    token: bytes
    num: int
    content: bytes

@dataclass(**dc_slots)
class DataStream:
    channels: List[Channel]
    messages: List[Message]

@dataclass(**dc_slots)
class Decoder:
    stype: str
    fixup: object = None

def _nullterm_string(s):
    zero = s.find(0)
    if zero >= 0: s = s[:zero]
    return s.decode('ascii')

def _fast_f16():
    _f16_mult = [(s * (1 / 2**25) * 2**max(e, 1)) if e != 31 else math.nan
                 for s in (1, -1)
                 for e in range(32)]
    _f16_adder = [((e != 0) - e - s) * 1024
                  for s in (0, 32)
                  for e in range(32)]

    return array('f', [(i + _f16_adder[i >> 10]) * _f16_mult[i >> 10]
                       for i in range(65536)])
_fast_f16 = _fast_f16()

# A couple examples from wikipedia
assert _fast_f16[0x0000] == 0
assert _fast_f16[0x0001] == 2**-24
assert _fast_f16[0x03ff] == 1023 * 2**-24
assert _fast_f16[0x0400] == 2**-14
assert _fast_f16[0x3c00] == 1
assert _fast_f16[0xc000] == -2


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
    1:  Decoder('H', fixup=lambda a: array('f', [_fast_f16[x] for x in a])),
    3:  Decoder('i'), # Master Clock on ScottE46?
    4:  Decoder('h'),
    6:  Decoder('f'),
    11: Decoder('h'),
    12: Decoder('i'), # Predictive Time?
    13: Decoder('B'), # status field?
    15: Decoder('Q', fixup=lambda a: array('i', [_gear_table.get(chr(int(x) & 0xffff), int(x) & 0xffff) for x in a])), # ?? NdscSwitch on M4GT4
    20: Decoder('H', fixup=lambda a: array('f', [_fast_f16[x] for x in a])),
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

def _decode_sequence(s, progress=None):
    groups = {}
    channels = {}
    messages = []
    next_progress = 1_000_000
    pos = 0
    badbytes = bytearray()
    IH_decoder = struct.Struct('<IH')
    IHH_decoder = struct.Struct('<IHH')
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
        oldpos = pos
        try:
            if s[pos] == ord_op:
                pos += 1
                if s[pos] == ord_G:
                    # print('G of', hdr[1], 'at', '%x' % pos)
                    tc, idx = IH_decoder.unpack_from(s, pos + 1)
                    g = groups[idx]
                    pos += g._add_helper
                    assert s[pos] == ord_cp, "%c at %x" % (s[pos], pos)
                    if tc > g._last_timecode:
                        g.samples += s[pos - g._row_size:pos]
                        g._last_timecode = tc
                    pos += 1
                elif s[pos] == ord_S:
                    tc, idx = IH_decoder.unpack_from(s, pos + 1)
                    ch = channels[idx]
                    # print(hdr[1], ch)
                    pos += 7 + ch.size
                    assert s[pos] == ord_cp, "%c at %x" % (s[pos], pos)
                    if tc > ch._last_timecode:
                        ch.timecodes.append(tc)
                        ch.sampledata += s[oldpos+8:pos]
                        ch._last_timecode = tc
                    pos += 1
                elif s[pos] == ord_M:
                    pos += 1
                    tc, idx, cnt = IHH_decoder.unpack_from(s, pos)
                    pos += 8
                    ch = channels[idx]
                    pos += ch.size * cnt
                    assert s[pos] == ord_cp, "%c at %x" % (s[pos], pos)
                    ms = Mms[ch.unknown[64]]
                    assert tc > ch._last_timecode # Not sure how to handle
                    ch.timecodes += array('i', [tc + off for off in range(0, cnt*ms, ms)])
                    ch.sampledata += s[oldpos+10:pos]
                    ch._last_timecode = tc + cnt * ms - ms
                    pos += 1
                else:
                    assert False, "%c at %x" % (s[pos], pos)
            elif s[pos] == ord_lt:
                if pos > next_progress and progress:
                    progress(pos, len(s))
                    next_progress += 1_000_000
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

                            align = max(4, max(channels[ch].size for ch in m.content.channels))
                            assert align == 4 or align == 8 # What else can we do?
                            m.content._add_helper = 7 + sum(channels[ch].size
                                                            for ch in m.content.channels)
                            m.content._row_size = (m.content._add_helper + align - 2) & -align
                            idx = m.content._row_size - (m.content._add_helper - 7)
                            for ch in m.content.channels:
                                channels[ch].group = GroupRef(m.content, idx)
                                idx += channels[ch].size
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
                             'MODI', 'HWNF', 'PDLT'):
                    data = _nullterm_string(data)
                elif tok == 'ENF':
                    data = _decode_sequence(data).messages

                assert l2[0] == bytesum, '%x vs %x at %x' % (l2[0], bytesum, pos)
                messages.append(Message(tok, l[1], data))
            else:
                assert False, "%c at %x" % (s[pos], pos)
        except Exception as err:
            if not badbytes:
                # traceback.print_exc()
                badpos = oldpos
            badbytes.append(s[oldpos])
            pos = oldpos + 1
        else:
            if badbytes:
                #print('Bad bytes(%d at %x):' % (len(badbytes), badpos), bytes(badbytes))
                badbytes = bytearray()
    assert pos == len(s)
    for g in groups.values():
        g.samples = memoryview(g.samples) # For more efficient access later
    for c in channels.values():
        if c.long_name in _manual_decoders:
            d = _manual_decoders[c.long_name]
        elif c.unknown[20] in _decoders:
            d = _decoders[c.unknown[20]]
        else:
            continue

        if c.group:
            idx = c.group.group.channels.index(c.index)
            tcidx = c.group.group._row_size - c.group.group._add_helper + 1
            sidx = c.group.offset
            samp = c.group.group.samples
            c.timecodes = samp[tcidx:len(samp)-(-tcidx&3)].cast('i')[::c.group.group._row_size//4]
            samp = samp[sidx:len(samp)-(-sidx & (c.size - 1))]
            c.sampledata = samp.cast(d.stype)[::c.group.group._row_size//c.size]
            c.group = None # doesn't matter anymore, we've extracted our data
        else:
            c.sampledata = memoryview(c.sampledata).cast(d.stype)

        if d.fixup:
            c.sampledata = memoryview(d.fixup(c.sampledata))

    return DataStream(
        channels=list(ch for ch in channels.values()
                      if len(ch.sampledata) and ch.long_name not in ('StrtRec', 'MasterClk')),
        messages=messages)

class AIMXRK:
    def __init__(self, fname, progress):
        self.file_name = fname
        with open(fname, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                self.data = _decode_sequence(m, progress)
        self.msg_by_type = {}
        for m in self.data.messages:
            self.msg_by_type.setdefault(m.token, []).append(m)
        #pp.pprint({k: len(v) for k, v in self.msg_by_type.items()})
        # determine time offset
        self.time_offset = 0
        laps = self.get_laps()
        if laps:
            self.time_offset = laps[0].start_time
            for ch in self.data.channels:
                ch.timecodes = memoryview(np.subtract(ch.timecodes,
                                                      self.time_offset))
        # decode GPS data.  Do this after determining time offset
        # since we have to fudge the data to get higher resolution
        self.decode_gps()

    def decode_gps(self):
        # look for either GPS or GPS1 messages
        gpsmsg = self.msg_by_type.get('GPS', self.msg_by_type.get('GPS1', None))
        if not gpsmsg: return
        alldata = memoryview(b''.join(m.content for m in gpsmsg))
        assert len(alldata) % 56 == 0
        timecodes = alldata[0:].cast('i')[::56//4]
        itow_ms = alldata[4:].cast('I')[::56//4]
        weekN = alldata[12:].cast('H')[::56//2]
        ecefX_cm = alldata[16:].cast('i')[::56//4]
        ecefY_cm = alldata[20:].cast('i')[::56//4]
        ecefZ_cm = alldata[24:].cast('i')[::56//4]
        posacc_cm = alldata[28:].cast('i')[::56//4]
        ecefdX_cms = alldata[32:].cast('i')[::56//4]
        ecefdY_cms = alldata[36:].cast('i')[::56//4]
        ecefdZ_cms = alldata[40:].cast('i')[::56//4]
        velacc_cms = alldata[44:].cast('i')[::56//4]
        nsat = alldata[51::56]

        timecodes = memoryview(np.subtract(timecodes, self.time_offset))

        self.data.channels.append(
            Channel(long_name='GPS Speed',
                    units='m/s',
                    dec_pts=1,
                    timecodes=timecodes,
                    sampledata=memoryview(np.sqrt(np.square(ecefdX_cms) +
                                                  np.square(ecefdY_cms) +
                                                  np.square(ecefdZ_cms)) / 100.)))

        gpsconv = gps.ecef2lla(np.divide(ecefX_cm, 100),
                               np.divide(ecefY_cm, 100),
                               np.divide(ecefZ_cm, 100))

        self.data.channels.append(Channel(long_name='GPS Latitude',  units='deg', dec_pts=4,
                                          timecodes=timecodes, sampledata=memoryview(gpsconv.lat)))
        self.data.channels.append(Channel(long_name='GPS Longitude', units='deg', dec_pts=4,
                                          timecodes=timecodes, sampledata=memoryview(gpsconv.long)))
        self.data.channels.append(Channel(long_name='GPS Altitude', units='m', dec_pts=1,
                                          timecodes=timecodes, sampledata=memoryview(gpsconv.alt)))

    def _help_decode_channels(self, chmap):
        pp.pprint(chmap)
        for i in range(len(self.data.channels[0].unknown)):
            d = sorted([(v.unknown[i], chmap.get(v.long_name, ''), v.long_name)
                        for v in self.data.channels
                        if len(v.unknown) > i])
            if len(set([x[0] for x in d])) == 1:
                continue
            pp.pprint((i, d))
        d = sorted([(len(v.sampledata), chmap.get(v.long_name, ''), v.long_name)
                    for v in self.data.channels])
        if len(set([x[0] for x in d])) != 1:
            pp.pprint(('len', d))

    def get_laps(self):
        ret = []
        if 'LAP' in self.msg_by_type:
            for m in self.msg_by_type['LAP']:
                # 2nd byte is segment #, see M4GT4
                segment, lap, duration, end_time = struct.unpack('xBHIxxxxxxxxI', m.content)
                end_time -= self.time_offset
                if (not ret or ret[-1].num != lap) and segment == 0:
                    assert not ret or ret[-1].num + 1 == lap # deal with missing data later
                    ret.append(Lap(lap, end_time - duration, end_time))
        return ret

    def get_speed_channel(self):
        return 'GPS Speed'

    def get_filename(self):
        return self.file_name

    def get_channels(self):
        return [ch.long_name for ch in self.data.channels]

    def get_channel_units(self, name):
        ch = [ch for ch in self.data.channels if ch.long_name == name]
        if len(ch) != 1: return None
        ch = ch[0]
        if ch.size == 1: return ''
        return ch.units

    def get_channel_dec_points(self, name):
        ch = [ch for ch in self.data.channels if ch.long_name == name]
        if len(ch) != 1: return None
        ch = ch[0]
        return ch.dec_pts

    def get_channel_data(self, name):
        ch = [ch for ch in self.data.channels if ch.long_name == name]
        if len(ch) != 1: return None
        ch = ch[0]
        if ch.long_name in ('L_BATT_VOLT', 'External Voltage', 'DTA_BATTERY', 'M800_BATTERY', 'ECU_BATTERY'):
            scale = 0.001
        else:
            scale = 1
        tc = ch.timecodes # lets store it this way
        samp = ch.sampledata
        assert len(tc) == len(samp), "%s: %d vs %d (sz=%d, tp=%d)" % (name, len(tc), len(samp), ch.size, ch.unknown[20])
        if scale != 1:
            samp = array('d', [a * scale for a in samp])
        return (tc, samp)


# CMP = championship
# ENF = ECU conf?
# HWNF = wifi
# VEH = vehicle
# VTY = session type
