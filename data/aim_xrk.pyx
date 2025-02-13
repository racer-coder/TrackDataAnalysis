
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
import concurrent.futures
import ctypes
from dataclasses import dataclass, field
import math
import mmap
import numpy as np
import os
from pprint import pprint # pylint: disable=unused-import
import struct
import sys
import time
import traceback # pylint: disable=unused-import
from typing import Dict, List, Optional

import cython
from cython.operator cimport dereference
from libcpp.vector cimport vector

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
    samples: array = field(default_factory=lambda: array('I'), repr=False)
    # used during building:
    timecodes: Optional[array] = field(default=None, repr=False)

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
    interpolate: bool = False
    unknown: bytes = b""
    group: Optional[GroupRef] = None
    timecodes: object = field(default=None, repr=False)
    sampledata: object = field(default=None, repr=False)

@dataclass(**dc_slots)
class Message:
    token: bytes
    num: int
    content: bytes

@dataclass(**dc_slots)
class DataStream:
    channels: Dict[str, Channel]
    messages: Dict[str, List[Message]]
    laps: List[base.Lap]
    time_offset: int

@dataclass(**dc_slots)
class Decoder:
    stype: str
    interpolate: bool = False
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

_gear_table = np.arange(65536, dtype=np.uint16)
_gear_table[ord('N')] = 0
_gear_table[ord('1')] = 1
_gear_table[ord('2')] = 2
_gear_table[ord('3')] = 3
_gear_table[ord('4')] = 4
_gear_table[ord('5')] = 5
_gear_table[ord('6')] = 6

_decoders = {
    0:  Decoder('i'), # Master Clock on M4GT4?
    1:  Decoder('H', interpolate=True,
                fixup=lambda a: np.ndarray(buffer=a, shape=(len(a),),
                                           dtype=np.float16).astype(np.float32).data),
    3:  Decoder('i'), # Master Clock on ScottE46?
    4:  Decoder('h'),
    6:  Decoder('f', interpolate=True),
    11: Decoder('h'),
    12: Decoder('i'), # Predictive Time?
    13: Decoder('B'), # status field?
    15: Decoder('H', fixup=lambda a: _gear_table[a]), # ?? NdscSwitch on M4GT4.  Also actual size is 8 bytes
    20: Decoder('H', interpolate=True,
                fixup=lambda a: np.ndarray(buffer=a, shape=(len(a),),
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
    43: ('kg', 3),
}

def _ndarray_from_mv(mv):
    mv = memoryview(mv) # force it
    return np.ndarray(buffer=mv, shape=(len(mv),), dtype=np.dtype(mv.format))

def _sliding_ndarray(buf, typ):
    return np.ndarray(buffer=buf, dtype=typ,
                      shape=(len(buf) - array(typ).itemsize + 1,), strides=(1,))

def _tokdec(s):
    if s: return ord(s[0]) + 256 * _tokdec(s[1:])
    return 0

def _tokenc(i):
    s = ''
    while i:
        s += chr(i & 255)
        i >>= 8
    return s

accum = cython.struct(
    last_timecode=cython.int,
    add_helper=cython.ushort,
    Mms=cython.ushort,
    data=vector[cython.uchar],
    timecodes=vector[cython.int])

cdef packed struct smsg_hdr:  # covers G, S, and M messages
    cython.ushort op
    cython.int timecode
    cython.ushort index
    cython.ushort count # for M messages only
    # data field(s) follow(s), size depends on type/group

cdef packed struct cmsg_hdr:  # covers c messages
    cython.ushort op
    cython.uchar unk1 # always 0?
    cython.ushort channel # bottom 3 bits always 4?
    cython.uchar unk3 # always 0x84?
    cython.uchar unk4 # always 6?
    cython.int timecode
    # data field follows, size depends on type

cdef packed struct hmsg_hdr:
    cython.ushort op
    cython.uint tok
    cython.int hlen
    cython.uchar ver
    cython.uchar cl

cdef packed struct hmsg_ftr:
    cython.uchar op
    cython.uint tok
    cython.ushort bytesum
    cython.uchar cl

cdef union msg_hdr:
    smsg_hdr s
    cmsg_hdr c
    hmsg_hdr h

ctypedef const cython.uchar* byte_ptr
ctypedef vector[accum] vaccum

cdef extern from '<numeric>' namespace 'std' nogil:
    T accumulate[InputIt, T](InputIt first, InputIt last, T init)

cdef _resize_vaccum(vaccum & v, int idx):
    if idx >= v.size():
        old_len = v.size()
        v.resize(idx + 1)
        for i in range(old_len, v.size()):
            v[i].last_timecode = -1
            v[i].add_helper = 1
            v[i].Mms = 0

cdef _Mms_lookup(int k):
    # Not sure how to represent 500 Hz
    if k == 8:  return 5  # 200 Hz
    if k == 16: return 10 # 100 Hz
    if k == 32: return 20 #  50 Hz
    if k == 64: return 40 #  25 Hz
    if k == 80: return 50 #  20 Hz
    # I guess 10Hz, 5Hz, 2Hz, and 1Hz don't use M messages
    return 0

@cython.wraparound(False)
def _decode_sequence(s, progress=None):
    cdef const cython.uchar[::1] sv = s
    groups = []
    channels = []
    messages = {}
    tok_GPS: cython.uint = _tokdec('GPS')
    tok_GPS1: cython.uint = _tokdec('GPS1')
    progress_interval: cython.uint = 8_000_000
    next_progress: cython.uint = progress_interval
    pos: cython.uint = 0
    oldpos: cython.uint = pos
    badbytes: cython.uint = 0
    badpos: cython.uint = 0
    ord_op: cython.int = ord('(')
    ord_cp: cython.int = ord(')')
    ord_op_G : cython.int = ord_op + 256 * ord('G')
    ord_op_S : cython.int = ord_op + 256 * ord('S')
    ord_op_M : cython.int = ord_op + 256 * ord('M')
    ord_op_c : cython.int = ord_op + 256 * ord('c')
    ord_lt: cython.int = ord('<')
    ord_lt_h : cython.int = ord_lt + 256 * ord('h')
    ord_gt: cython.int = ord('>')
    len_s = len(s)
    cdef vaccum[4] gc_data # [0]: G messages (groups) [1]: S messages (samples?) [2]: c messages (channels from expansion) [3]: M messages
    time_offset = None
    last_time = None
    t1 = time.perf_counter()
    cdef vaccum * data_cat
    cdef accum * data_p
    gpsmsg: vector[cython.uchar]
    show_all: cython.int = 0
    show_bad: cython.int = 0
    while pos < len_s:
        try:
            while True:
                oldpos = pos
                if pos + 10 >= len_s: # smallest message is 3 (frame) + 4 (tc) + 2 (idx) + 1 (data)
                    raise IndexError
                msg = <msg_hdr *>&sv[pos]
                typ: cython.int = msg.s.op
                if abs(typ - (ord_op_G + ord_op_S) // 2) == (ord_op_S - ord_op_G) // 2:
                    data_cat = &gc_data[typ == ord_op_S]
                    data_p = &dereference(data_cat)[msg.s.index]
                    if data_p >= &dereference(data_cat.end()):
                        raise IndexError
                    pos += data_p.add_helper
                    last = &sv[pos-1]
                    if last[0] != ord_cp:
                        raise ValueError("%c at %x" % (s[pos-1], pos-1))
                    if show_all:
                        print('tc=%d %c idx=%d' % (msg.s.timecode, msg.s.op >> 8, msg.s.index))
                    if msg.s.timecode > data_p.last_timecode:
                        data_p.last_timecode = msg.s.timecode
                        data_p.data.insert(data_p.data.end(),
                                           <const cython.uchar *>&msg.s.timecode, last)
                elif typ == ord_op_M:
                    data_p = &gc_data[3][msg.s.index]
                    if data_p >= &dereference(gc_data[3].end()):
                        raise IndexError
                    if data_p.Mms == 0:
                        raise ValueError('No ms understood for channel %s' %
                                         channels[msg.s.index].long_name)
                    pos += data_p.add_helper * msg.s.count + 10
                    if sv[pos] != ord_cp:
                        raise ValueError("%c at %x" % (s[pos], pos))
                    if show_all:
                        print('tc=%d M idx=%d cnt=%d ms=%d' %
                              (msg.s.timecode, msg.s.index, msg.s.count, data_p.Mms))
                    if msg.s.timecode > data_p.last_timecode:
                        data_p.last_timecode = msg.s.timecode + (msg.s.count-1) * data_p.Mms
                        m_tc : cython.int
                        for m_tc in range(msg.s.count):
                            data_p.timecodes.push_back(msg.s.timecode + m_tc * data_p.Mms)
                        data_p.data.insert(data_p.data.end(),
                                           &sv[oldpos+10], &sv[pos])
                    pos += 1
                elif typ == ord_op_c:
                    assert msg.c.unk1 == 0, '%x' % msg.c.unk1
                    assert (msg.c.channel & 7) == 4, '%x' % (msg.c.channel & 7)
                    assert msg.c.unk3 == 0x84, '%x' % msg.c.unk3
                    assert msg.c.unk4 == 6, '%x' % msg.c.unk4
                    data_cat = &gc_data[2]
                    data_p = &dereference(data_cat)[msg.c.channel >> 3]
                    if data_p >= &dereference(data_cat.end()):
                        raise IndexError
                    pos += data_p.add_helper
                    last = &sv[pos-1]
                    if last[0] != ord_cp:
                        raise ValueError("%c at %x" % (s[pos-1], pos-1))
                    if show_all:
                        print('tc=%d c idx=%d' % (msg.c.timecode, msg.c.channel >> 3))
                    if msg.c.timecode > data_p.last_timecode:
                        data_p.last_timecode = msg.c.timecode
                        data_p.data.insert(data_p.data.end(),
                                           <const cython.uchar *>&msg.c.timecode, last)
                elif typ == ord_lt_h:
                    if pos > next_progress:
                        next_progress += progress_interval
                        if progress:
                            progress(pos, len(s))
                    tok: cython.uint = msg.h.tok
                    hlen: cython.uint = msg.h.hlen
                    if hlen >= len_s:
                        raise IndexError
                    ver = msg.h.ver
                    assert msg.h.cl == ord_gt, "%c at %x" % (msg.h.cl, pos+11)
                    pos += 12

                    # get some "free" range checking here before we go walking data[]
                    assert sv[pos+hlen] == ord_lt, "%s at %x" % (s[pos+hlen], pos+hlen)

                    bytesum: cython.ushort = accumulate[byte_ptr, cython.int](
                        &sv[pos], &sv[pos+hlen], 0)
                    pos += hlen

                    msgf = <hmsg_ftr *>&sv[pos]

                    assert msgf.tok == tok, "%x vs %x at %x" % (msgf.tok, tok, pos+1)
                    assert msgf.bytesum == bytesum, '%x vs %x at %x' % (msgf.bytesum, bytesum, pos+5)
                    assert msgf.cl == ord_gt, "%c at %x" % (msgf.cl, pos+7)
                    pos += 8

                    if (tok >> 24) == 32:
                        tok -= 32 << 24 # rstrip(' ')

                    if tok == tok_GPS or tok == tok_GPS1:
                        # fast path common case
                        gpsmsg.insert(gpsmsg.end(), &sv[oldpos+12], &sv[pos-8])
                    else:
                        data = s[oldpos + 12 : pos - 8]
                        if tok == _tokdec('CNF'):
                            data = _decode_sequence(data).messages
                            #channels = {} # Replays don't necessarily contain all the original channels
                            for m in data[_tokdec('CHS')]:
                                channels += [None] * (m.content.index - len(channels) + 1)
                                if not channels[m.content.index]:
                                    channels[m.content.index] = m.content
                                    _resize_vaccum(gc_data[1], m.content.index)
                                    gc_data[1][m.content.index].add_helper = m.content.size + 9
                                    _resize_vaccum(gc_data[2], m.content.index)
                                    gc_data[2][m.content.index].add_helper = m.content.size + 12
                                    _resize_vaccum(gc_data[3], m.content.index)
                                    gc_data[3][m.content.index].add_helper = m.content.size
                                    gc_data[3][m.content.index].Mms = _Mms_lookup(
                                        m.content.unknown[64] & 127)
                                else:
                                    assert channels[m.content.index].short_name == m.content.short_name, "%s vs %s" % (channels[m.content.index].short_name, m.content.short_name)
                                    assert channels[m.content.index].long_name == m.content.long_name
                            for m in data.get(_tokdec('GRP'), []):
                                groups += [None] * (m.content.index - len(groups) + 1)
                                groups[m.content.index] = m.content
                                idx = 6
                                for ch in m.content.channels:
                                    channels[ch].group = GroupRef(m.content, idx)
                                    idx += channels[ch].size
                                if show_all:
                                    print('GROUP', m.content.index,
                                          [(ch, channels[ch].long_name, channels[ch].size)
                                           for ch in m.content.channels])

                                _resize_vaccum(gc_data[0], m.content.index)
                                gc_data[0][m.content.index].add_helper = 9 + sum(
                                    channels[ch].size for ch in m.content.channels)
                        elif tok == _tokdec('GRP'):
                            data = memoryview(data).cast('H')
                            assert data[1] == len(data[2:])
                            data = Group(index = data[0], channels = data[2:])
                        elif tok == _tokdec('CDE'):
                            data = ['%02x' % x for x in data]
                        elif tok == _tokdec('CHS'):
                            dcopy = bytearray(data) # copy
                            data = Channel()
                            (data.index,
                             data.short_name,
                             data.long_name,
                             data.size) = struct.unpack('<H22x8s24s16xB39x', dcopy)
                            try:
                                data.units, data.dec_pts = _unit_map[dcopy[12] & 127]
                            except KeyError:
                                print('Unknown units[%d] for %s' %
                                      (dcopy[12] & 127, data.long_name))
                                data.units = ''
                                data.dec_pts = 0

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
                        elif tok == _tokdec('LAP'):
                            # cache first time offset for use later
                            duration, end_time = struct.unpack('4xI8xI', data)
                            if time_offset is None:
                                time_offset = end_time - duration
                            last_time = end_time
                        elif tok in (_tokdec('RCR'), _tokdec('VEH'), _tokdec('CMP'), _tokdec('VTY'), _tokdec('NDV'), _tokdec('TMD'), _tokdec('TMT'),
                                     _tokdec('DBUN'), _tokdec('DBUT'), _tokdec('DVER'), _tokdec('MANL'), _tokdec('MODL'), _tokdec('MANI'),
                                     _tokdec('MODI'), _tokdec('HWNF'), _tokdec('PDLT'), _tokdec('NTE')):
                            data = _nullterm_string(data)
                        elif tok == _tokdec('ENF'):
                            data = _decode_sequence(data).messages
                        elif tok == _tokdec('TRK'):
                            data = {'name': _nullterm_string(data[:32]),
                                    'sf_lat': memoryview(data).cast('i')[9] / 1e7,
                                    'sf_long': memoryview(data).cast('i')[10] / 1e7}
                        elif tok == _tokdec('ODO'):
                            # not sure how to map fuel.
                            # Fuel Used channel claims 8.56l used (2046.0-2037.4)
                            # Fuel Used odo says 70689.
                            data = {_nullterm_string(data[i:i+16]):
                                    {'time': memoryview(data[i+16:i+24]).cast('I')[0], # seconds
                                     'dist': memoryview(data[i+16:i+24]).cast('I')[1]} # meters
                                    for i in range(0, len(data), 64)
                                    # not sure how to parse fuel, doesn't match any expected units
                                    if not _nullterm_string(data[i:i+16]).startswith('Fuel')}

                        try:
                            messages[tok].append(Message(tok, ver, data))
                        except KeyError:
                            messages[tok] = [Message(tok, ver, data)]
                else:
                    assert False, "%02x%02x at %x" % (s[pos], s[pos+1], pos)
        except Exception as _err: # pylint: disable=broad-exception-caught
            if oldpos != badpos + badbytes and badbytes:
                if show_bad:
                    print('Bad bytes(%d at %x):' % (badbytes, badpos),
                          ', '.join('%02x' % c for c in s[badpos:badpos + badbytes])
                          )
                badbytes = 0
            if not badbytes:
                if show_bad:
                    sys.stdout.flush()
                    traceback.print_exc()
                badpos = oldpos # pylint: disable=unused-variable
            if oldpos < len_s:
                badbytes += 1
                pos = oldpos + 1
    t2 = time.perf_counter()
    if badbytes:
        if show_bad:
            print('Bad bytes(%d at %x):' % (badbytes, badpos),
                  ', '.join('%02x' % c for c in s[badpos:badpos + badbytes])
                  )
        badbytes = 0
    assert pos == len(s)
    # quick scan through all the groups/channels for the first used timecode
    if channels:
        # int(min(time_offset, time_offset,
        time_offset = int(min(
            ([time_offset] if time_offset is not None else [])
            #XXX*[s2mv[l[0]] for l in g_indices if l.size()],
            #XXX*[s2mv[l[0]] for l in ch_indices if l.size()],
            + [c.timecodes[0] for c in channels if c and len(c.timecodes)],
            default=0))
        last_time = int(max(
            ([last_time] if last_time is not None else [])
            #XXX*[s2mv[l[l.size()-1]] for l in g_indices if l.size()],
            #XXX*[s2mv[l[l.size()-1]] for l in ch_indices if l.size()],
            + [c.timecodes[len(c.timecodes)-1] for c in channels if c and len(c.timecodes)],
            default=0))
    def process_group(g):
        g.samples = np.array([], dtype=np.int32)
        g.timecodes = g.samples.data
        if g.index < gc_data[0].size():
            data_p = &gc_data[0][g.index]
            if data_p.data.size():
                g.samples = np.asarray(<cython.uchar[:data_p.data.size()]> &data_p.data[0])
                rows = len(g.samples) // (data_p.add_helper - 3)
                g.timecodes = np.ndarray(buffer=g.samples, dtype=np.int32,
                                         shape=(rows,),
                                         strides=(data_p.add_helper-3,)) - time_offset
        for ch in g.channels:
            process_channel(channels[ch])

    def process_channel(c):
        if c.long_name in _manual_decoders:
            d = _manual_decoders[c.long_name]
        elif c.unknown[20] in _decoders:
            d = _decoders[c.unknown[20]]
        else:
            return

        c.interpolate = d.interpolate
        if c.group:
            grp = c.group.group
            c.timecodes = grp.timecodes
            c.sampledata = np.ndarray(buffer=grp.samples[c.group.offset:], dtype=d.stype,
                                      shape=grp.timecodes.shape,
                                      strides=(gc_data[0][grp.index].add_helper-3,)).copy()
        else:
            # check for S messages
            view_offset = 6
            stride_offset = 3
            data_p = &gc_data[1][c.index]
            if not data_p.data.size():
                # No? maybe c messages
                view_offset = 4
                stride_offset = 8
                data_p = &gc_data[2][c.index]
            if data_p.data.size():
                assert len(c.timecodes) == 0, "Can't have both S/c and M records for channel %s (index=%d, %d vs %d)" % (c.long_name, c.index, len(c.timecodes), data_p.data.size())

                # TREAD LIGHTLY - raw pointers here
                view = np.asarray(<cython.uchar[:data_p.data.size()]> &data_p.data[0])
                rows = len(view) // (data_p.add_helper - stride_offset)

                tc = np.ndarray(buffer=view, dtype=np.int32,
                                shape=(rows,), strides=(data_p.add_helper-stride_offset,)).copy()
                samp = np.ndarray(buffer=view[view_offset:], dtype=d.stype,
                                  shape=(rows,), strides=(data_p.add_helper-stride_offset,)).copy()
            else:
                data_p = &gc_data[3][c.index] # M messages
                if data_p.timecodes.size():
                    tc = np.asarray(<cython.int[:data_p.timecodes.size()]>
                                    &data_p.timecodes[0]).copy()
                    samp = np.ndarray(buffer=np.asarray(<cython.uchar[:data_p.data.size()]>
                                                        &data_p.data[0]),
                                      dtype=d.stype, shape=tc.shape).copy()
                else:
                    tc = _ndarray_from_mv(c.timecodes)
                    samp = _ndarray_from_mv(memoryview(c.sampledata).cast(d.stype))
            c.timecodes = (tc - time_offset).data
            c.sampledata = samp.data

        if d.fixup:
            c.sampledata = memoryview(d.fixup(c.sampledata))
        if c.units == 'V': # most are really encoded as mV, but one or two aren't....
            c.sampledata = np.divide(c.sampledata, 1000).data

    laps = None
    if not channels:
        t4 = time.perf_counter()
        pass # nothing to do
    elif progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, os.cpu_count())) as worker:
            bg_work = worker.submit(_bg_gps_laps, <cython.uchar[:gpsmsg.size()]> &gpsmsg[0],
                                    messages, time_offset, last_time)
            group_work = worker.map(process_group, [x for x in groups if x])
            channel_work = worker.map(process_channel,
                                      [x for x in channels if x and not x.group])
            gps_ch, laps = bg_work.result()
            t4 = time.perf_counter()
            for i in group_work:
                pass
            for i in channel_work:
                pass
            channels.extend(gps_ch)
    else:
        for g in groups:
            if g: process_group(g)
        for c in channels:
            if c and not c.group: process_channel(c)
        t4 = time.perf_counter()
        gps_ch, laps = _bg_gps_laps(<cython.uchar[:gpsmsg.size()]> &gpsmsg[0],
                                    messages, time_offset, last_time)
        channels.extend(gps_ch)

    t3 = time.perf_counter()
    if t3-t1 > 0.1:
        print('division: scan=%f, gps=%f, group/ch=%f more' % (t2-t1, t4-t2, t3-t4))

    return DataStream(
        channels={ch.long_name: ch for ch in channels
                  if ch and len(ch.sampledata)
                  and ch.long_name not in ('StrtRec', 'Master Clk')},
        messages=messages,
        laps=laps,
        time_offset=time_offset)

def _get_metadata(msg_by_type):
    ret = {}
    for msg, name in [(_tokdec('RCR'), 'Driver'),
                      (_tokdec('VEH'), 'Vehicle'),
                      (_tokdec('TMD'), 'Log Date'),
                      (_tokdec('TMT'), 'Log Time'),
                      (_tokdec('VTY'), 'Session'),
                      (_tokdec('CMP'), 'Series'),
                      (_tokdec('NTE'), 'Long Comment'),
                      ]:
        if msg in msg_by_type:
            ret[name] = msg_by_type[msg][-1].content
    if _tokdec('TRK') in msg_by_type:
        ret['Venue'] = msg_by_type[_tokdec('TRK')][-1].content['name']
        # ignore the start/finish line?
    if _tokdec('ODO') in msg_by_type:
        for name, stats in msg_by_type[_tokdec('ODO')][-1].content.items():
            ret['Odo/%s Distance (km)' % name] = stats['dist'] / 1000
            ret['Odo/%s Time' % name] = '%d:%02d:%02d' % (stats['time'] // 3600,
                                                          stats['time'] // 60 % 60,
                                                          stats['time'] % 60)
    return ret

def _bg_gps_laps(gpsmsg, msg_by_type, time_offset, last_time):
    channels = _decode_gps(gpsmsg, time_offset)
    lat_ch = None
    lon_ch = None
    for ch in channels:
        if ch.long_name == 'GPS Latitude': lat_ch = ch
        if ch.long_name == 'GPS Longitude': lon_ch = ch
    laps = _get_laps(lat_ch, lon_ch, msg_by_type, time_offset, last_time)
    return channels, laps

def _decode_gps(gpsmsg, time_offset):
    if not gpsmsg: return []
    alldata = memoryview(gpsmsg)
    assert len(alldata) % 56 == 0
    timecodes = np.asarray(alldata[0:].cast('i')[::56//4])
    # certain old MXP firmware (and maybe others) would periodically
    # butcher the upper 16-bits of the timecode field.  If necessary,
    # reconstruct it using only the bottom 16-bits and assuming time
    # never skips ahead too far.
    if np.any(timecodes[1:] < timecodes[:-1]):
        timecodes = (timecodes & 65535) + (timecodes[0] - (timecodes[0] & 65535))
        timecodes += 65536 * np.cumsum(np.concatenate(([0], timecodes[1:] < timecodes[:-1])))
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

    timecodes = memoryview(timecodes - time_offset)

    gpsconv = gps.ecef2lla(np.divide(ecefX_cm, 100),
                           np.divide(ecefY_cm, 100),
                           np.divide(ecefZ_cm, 100))

    return [Channel(
        long_name='GPS Speed',
        units='m/s',
        dec_pts=1,
        interpolate=True,
        timecodes=timecodes,
        sampledata=memoryview(np.sqrt(np.square(ecefdX_cms) +
                                      np.square(ecefdY_cms) +
                                      np.square(ecefdZ_cms)) / 100.)),
            Channel(long_name='GPS Latitude',  units='deg', dec_pts=4, interpolate=True,
                    timecodes=timecodes, sampledata=memoryview(gpsconv.lat)),
            Channel(long_name='GPS Longitude', units='deg', dec_pts=4, interpolate=True,
                    timecodes=timecodes, sampledata=memoryview(gpsconv.long)),
            Channel(long_name='GPS Altitude', units='m', dec_pts=1, interpolate=True,
                    timecodes=timecodes, sampledata=memoryview(gpsconv.alt))]

def _get_laps(lat_ch, lon_ch, msg_by_type, time_offset, last_time):
    if lat_ch and lon_ch:
        # If we have GPS, do gps lap insert.

        track = msg_by_type[_tokdec('TRK')][-1].content
        XYZ = np.column_stack(gps.lla2ecef(np.array(lat_ch.sampledata),
                                           np.array(lon_ch.sampledata), 0))
        lap_markers = gps.find_laps(XYZ,
                                    np.array(lat_ch.timecodes),
                                    (track['sf_lat'], track['sf_long']))

        lap_markers = [0] + lap_markers + [last_time - time_offset]

        return [base.Lap(lap, start_time, end_time)
                for lap, (start_time, end_time) in enumerate(
                        zip(lap_markers[:-1], lap_markers[1:]))]

    # otherwise, use the lap data provided.
    ret = []
    if _tokdec('LAP') in msg_by_type:
        for m in msg_by_type[_tokdec('LAP')]:
            # 2nd byte is segment #, see M4GT4
            segment, lap, duration, end_time = struct.unpack('xBHIxxxxxxxxI', m.content)
            end_time -= time_offset
            if segment:
                continue
            elif not ret:
                pass
            elif ret[-1].num == lap:
                continue
            elif ret[-1].num + 1 == lap:
                pass
            elif ret[-1].num + 2 == lap:
                # emit inferred lap
                ret.append(base.Lap(lap - 1, ret[-1].end_time,
                                    end_time - duration))
            else:
                assert False, 'Lap gap from %d to %d' % (ret[-1].num, lap)
            ret.append(base.Lap(lap, end_time - duration, end_time))
    return ret


def AIMXRK(fname, progress):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            data = _decode_sequence(m, progress)
    #pprint({k: len(v) for k, v in self.msg_by_type.items()})

    return base.LogFile(
        {ch.long_name: base.Channel(ch.timecodes,
                                    ch.sampledata,
                                    ch.long_name,
                                    ch.units if ch.size != 1 else '',
                                    ch.dec_pts,
                                    interpolate = ch.interpolate)
         for ch in data.channels.values()},
        data.laps,
        _get_metadata(data.messages),
        ['GPS Speed', 'GPS Latitude', 'GPS Longitude', 'GPS Altitude'],
        fname)

def aim_track(fname):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            data = _decode_sequence(m, None)
    return {_tokenc(k): v for k, v in data.messages.items()}

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
