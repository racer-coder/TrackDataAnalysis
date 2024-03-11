
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
import concurrent.futures
from copy import copy
from dataclasses import dataclass, field
import gzip
import struct
import sys
import time

import numpy as np

from . import base
from . import gps

def decode_string_list(data, pos):
    strings = []
    while data[pos] >= 32:
        n = data.index(0, pos)
        strings.append(data[pos:n].decode('latin-1'))
        pos = n + 1
    return strings, pos

def decode_groups(data, pos, num_groups, bases):
    groups = []
    for i in range(num_groups):
        num_ch = data[pos]
        base = bases[data[pos + 1] -1] if data[pos + 1] else ''
        n = data.index(0, pos + 2)
        groups.append((data[pos+2:n].decode('latin-1'), base, num_ch))
        pos = n + 1
    return groups, pos

@dataclass
class Channel:
    name: str = ''
    unit: str = ''
    scale: float = 1.
    adder: float = 0.
    size: int = 0
    address: int = 0
    decoder: int = 0
    timecodes: array = None
    samples: array = None
    repeat: int = None
    str_index: int = None
    str_suf: str = None

def decode_channels(data, pos, num_ch, base, units, all_channels):
    for chidx in range(num_ch):
        start = pos
        scan_str = False
        ch = Channel()
        pos += 2
        if data[start] & 0x10: # has repeat
            ch.repeat = data[pos]
            pos += 1
        if data[start+1] & 1: # has units
            ch.unit = units[data[pos]-1]
            pos += 1
        m = (data[start] * 256 + data[start+1]) & -0x10fa
        if m == 0x0004: pos += 1
        if m == 0x0006: pos += 2
        if m == 0x0106: pos += 2
        if m == 0x0204: pos += 2
        if m == 0x0206: pos += 4
        if m == 0x0302: pos += 2
        if m == 0x0306: pos += 4
        if m == 0x0506: pos += 4
        if m == 0x0606: pos += 4
        if m == 0x0706: pos += 4
        if (m >> 8) == 0: ch.size = 1
        if (m >> 8) == 1: ch.size = 1 # not sure we have any examples of this
        if (m >> 8) == 2: ch.size = 2
        if (m >> 8) == 3: ch.size = 2
        if (m >> 8) == 4: ch.size = 1
        if (m >> 8) == 5: ch.size = 4
        if (m >> 8) == 6: ch.size = 4
        if (m >> 8) == 7: ch.size = 16 # tire data
        ch.decoder = m >> 8
        if data[start+1] & 0x08: # scale/divider
            ch.scale, = struct.unpack_from('>H', data, pos)
            if data[start] == 6 and ch.scale == 0x9680:
                ch.scale = 10 ** 7
            pos += 2
        if data[start+1] & 0x10: # scale/multiplier
            ch.scale /= struct.unpack_from('>H', data, pos)[0]
            pos += 2
        if data[start+1] & 0x20: # adder
            ch.adder, = struct.unpack_from('>h', data, pos)
            pos += 2
        if data[start+1] & 0x40: # array of named values, names to appear later
            ch.str_index = data[pos + 1] * 256 + data[pos]
            scan_str = True
            pos += 2
        else:
            scan_str = False
        if data[start+1] & 0x80:
            pos += 1

        extra = data[start:pos]
        if scan_str:
            n = data.index(0, pos)
            ch.str_suf = data[pos:n].decode('ascii')
            pos = n + 1
        if (data[pos] < 32 or data[pos] >= 128) and data[pos] != 0:
            print('unknown %02x at %x' % (data[pos], pos))
            break
        n = data.index(0, pos)
        ch.name = base + data[pos:n].decode('latin-1')
        all_channels.append(ch)
        #print(chidx, '%x' % start, ch.name, ' '.join('%02x' % d for d in extra))
        pos = n + 1
        chidx += 1
    return pos

def expand_repeating_channels(all_channels, str_table):
    for ch in all_channels:
        if ch.repeat is None:
            yield ch
        else:
            for i in range(ch.repeat):
                ch_copy = copy(ch)
                if ch.str_index is None:
                    ch_copy.name = ch_copy.name.replace('$', '%d' % (i + 1))
                else:
                    ch_copy.name = str_table[ch.str_index + i] + ch.str_suf
                yield ch_copy

def assign_channel_addresses(all_channels):
    address = 0
    all_channels.sort(key=lambda ch: -ch.size)
    for ch in all_channels:
        if ch.size == 1:
            ch.address = address
            address += 1
    assert address < 0x200
    address = 0x200
    for ch in all_channels:
        if ch.size != 1:
            ch.address = address
            address += ch.size // 2
    return {ch.name: ch for ch in all_channels}

@dataclass(slots=True)
class AggregateData:
    tc: array = field(default_factory=lambda: array('I'))
    idx_off: array = field(default_factory=lambda: array('I'))
    data: bytearray = field(default_factory=bytearray)

    def result(self):
        if len(self.data) > len(self.tc):
            data = np.asarray(memoryview(self.data).cast('H'))
        else:
            data = np.array(self.data, dtype=np.uint16)

        idx_off = memoryview(self.idx_off).cast('B').cast('H')
        tc = np.asarray(self.tc) + idx_off[0::2]

        stack = np.column_stack([data,
                                 memoryview(tc).cast('B').cast('H')[0::2],
                                 memoryview(tc).cast('B').cast('H')[1::2],
                                 idx_off[1::2]])
        stack = np.ndarray(buffer=stack, dtype=np.uint64, shape=(len(stack),))

        stack.sort(kind='stable')
        ch = np.ndarray(buffer=stack, dtype=np.uint16, shape=(len(stack)*4,))[3::4]
        chboundaries = np.concatenate([[0],
                                       1 + np.nonzero(ch[1:] != ch[:-1])[0],
                                       [len(stack)]])
        return {ch[start]: (np.asarray(memoryview(stack[start:end]).cast('B')[2:-2].cast('I')[::2]),
                            np.asarray(memoryview(stack[start:end]).cast('B').cast('H')[::4]))
                for start, end in zip(chboundaries[:-1], chboundaries[1:])}

fast_ch_table = array('H', range(65536))
repeat_table = [20, 10, 5, 2, 1, 1, 1]

class SecretDecoderRing:
    def __init__(self, command, all_channel_data):
        start_ch = (command[0] << 4) | (command[1] >> 4)
        num_ch = ((command[2] & 0xf) << 8) | command[3]
        repeat = repeat_table[command[1] & 0xf]

        aci = start_ch >= 0x200
        self.count = num_ch * repeat
        self.size = self.count * (aci + 1)
        self.idx_tcoff = array('I', [(idx << 16) | off
                                     for off in range(0, 40, 40 // repeat)
                                     for idx in range(start_ch, start_ch + num_ch)])
        self.dest_idx_off = all_channel_data[aci].idx_off
        self.dest_tc = all_channel_data[aci].tc
        self.dest_data = all_channel_data[aci].data
        self.advance = 4

        if len(command) == 4:
            self.next = None
        else:
            self.next = SecretDecoderRing(command[4:], all_channel_data)
            if self.next.dest_data is self.dest_data:
                self.count += self.next.count
                self.size += self.next.size
                self.idx_tcoff.extend(self.next.idx_tcoff)
                self.advance += self.next.advance
                self.next = self.next.next

def decode_row(timestamp, data, pos, end_pos, all_channel_data, decoder_ring):
    pos += 4
    next_pos = pos + data[pos - 1] * 4
    tca = array('I', [timestamp])

    try:
        decoders = decoder_ring[data[pos:next_pos]]
    except KeyError:
        decoders = [SecretDecoderRing(data[pos:next_pos], all_channel_data)]
        while decoders[-1].next:
            decoders.append(decoders[-1].next)
        decoder_ring[data[pos:next_pos]] = decoders

    for decoder in decoders:
        pos = next_pos
        decoder.dest_idx_off.extend(decoder.idx_tcoff)
        decoder.dest_tc.extend(tca * decoder.count)
        next_pos += decoder.size
        decoder.dest_data += data[pos:next_pos]
    assert next_pos <= end_pos

def decode_rows(data, pos, progress):
    all_channel_data = [AggregateData(), AggregateData()]
    decoder_ring = {}

    first_timestamp = data[pos + 2]
    timestamp = 0
    update_pos = 8 << 20
    len_data = len(data)
    while pos + 4 < len_data:
        if pos >= update_pos:
            if progress:
                progress(pos, len_data)
            update_pos += 8 << 20
        l = data[pos+1] * 8 + 8
        if pos + l > len_data:
            break
        timestamp += (data[pos+2] - timestamp) % 25
        decode_row((timestamp - first_timestamp) * 40, data, pos, pos+l, all_channel_data,
                   decoder_ring)
        pos += l
    return all_channel_data

def assign_data(channel_map, all_channel_data):
    with concurrent.futures.ThreadPoolExecutor() as worker:
        d0 = worker.submit(all_channel_data[0].result)
        d1 = worker.submit(all_channel_data[1].result)
        data = d0.result() | d1.result()
    for ch in list(channel_map.values()):
        if ch.address not in data:
            continue
        if ch.decoder in (0, 1, 4):
            ch.timecodes, ch.samples = data[ch.address]
        if ch.decoder in (2, 3):
            ch.timecodes, ch.samples = data[ch.address]
            ch.samples = ch.samples.byteswap()
            if ch.decoder == 3:
                ch.samples = np.asarray(memoryview(ch.samples).cast('B').cast('h'))
        if ch.decoder in (5, 6):
            ch.timecodes, ch.samples = data[ch.address]
            ch.samples = np.column_stack([ch.samples, data[ch.address+1][1]])
            ch.samples = np.asarray(memoryview(ch.samples).cast('B').cast('I' if ch.decoder == 6 else 'i'))
            ch.samples = ch.samples.byteswap()
        if ch.decoder == 7:
            # array decoder
            timecodes = data[ch.address][0]
            samples = np.column_stack([data[ch.address + i][1] for i in range(8)])
            samples = np.asarray(memoryview(samples).cast('B'))
            if ch.scale != 1 or ch.adder != 0:
                samples = samples / ch.scale + ch.adder
            for i in range(16):
                chdup = copy(ch)
                chdup.timecodes = timecodes
                chdup.samples = samples[i::16]
                channel_map['%s[%d]' % (ch.name, i + 1)] = chdup
            continue
        if ch.samples is not None and (ch.scale != 1 or ch.adder != 0):
            ch.samples = ch.samples * (1 / ch.scale) + ch.adder

def decode_len_str(data, pos, num):
    ret = []
    for i in range(num):
        l = data[pos]
        ret.append(data[pos+1:pos+1+l].decode('latin-1'))
        pos += l + 1
    return ret, pos

def csv_analyze(channel_map):
    return # disable for now
    with open('2024_0309_0751.csv.gz', 'rb') as f:
        data = gzip.decompress(f.read()).decode('ascii').splitlines()
    cols = data[0].split(';')
    colpairs = [array('d') for c in cols[1:]]
    report = 0
    for l in data[21:]:
        l = l.split(';')
        tc = int(float(l[0]) * 1000 + 0.1)
        if tc >= report:
            print(tc / 60000)
            sys.stdout.flush()
            report += 30000
        for colidx, val in enumerate(l[1:]):
            if val:
                colpairs[colidx].append(tc)
                colpairs[colidx].append(float(val))
    for colidx, col in enumerate(cols[1:]):
        try:
            ch = channel_map[col]
        except KeyError:
            ch = None
        #if col != 'adu.track.startLineLatitiude': continue
        pairs = colpairs[colidx]
        if len(pairs) < 4:
            if ch and ch.timecodes is not None:
                print('%s -- no data (ch=%s)' % (col, ch))
            continue
        if not ch or ch.timecodes is None:
            print('%s -- ch has no data (%d vs %s)' % (col, len(pairs) // 2, ch))
            continue
        cht = ch.timecodes
        chs = ch.samples
        for i in range(20):
            if len(cht) and cht[0] != pairs[0]:
                cht = cht[1:]
                chs = chs[1:]
        if len(pairs) // 2 != len(cht):
            print('%s -- len mismatch (%d vs %d)' % (col, len(pairs) // 2, len(cht)))
            continue
        error = False
        for tc1, v1, tc2, v2 in zip(pairs[::2], pairs[1::2], cht, chs):
            if tc1 != tc2:
                print('tc %d/%d' % (tc1, tc2))
                error = True
            if abs(v1 - v2) > 0.001: # limits of csv accuracy
                print('v %s/%s (%s)' % (v1, v2, abs(v1-v2)))
                error = True
        if error:
            print('comparing %s (ch=%s)' % (col, ch))


def generate_laps(channel_map, last_time):
    lat = channel_map['gps.latitude']
    lon = channel_map['gps.longitude']
    sf_lat = np.median(channel_map['adu.track.startLineLatitiude'].samples) # yes typo
    sf_lon = np.median(channel_map['adu.track.startLineLongitude'].samples)
    XYZ = np.column_stack(gps.lla2ecef(lat.samples, lon.samples, 0))
    lap_markers = gps.find_laps(XYZ, lat.timecodes,
                                (sf_lat, sf_lon))
    lap_markers = [0] + lap_markers + [last_time]
    return [base.Lap(lap, start_time, end_time)
            for lap, (start_time, end_time) in enumerate(zip(lap_markers[:-1], lap_markers[1:]))]

def ECUMASTER_ADU(fname, progress):
    t0 = time.perf_counter()
    with open(fname, 'rb') as f:
        data = gzip.decompress(f.read())

    num_groups, = struct.unpack_from('<H', data, 0x20c)

    pos = 0x211
    units, pos = decode_string_list(data, pos)
    bases, pos = decode_string_list(data, pos + 1)
    groups, pos = decode_groups(data, pos, num_groups, bases)
    tot_ch = 0
    all_channels = []
    for group, base_name, num_ch in groups:
        pos = decode_channels(data, pos, num_ch, base_name, units, all_channels)
        tot_ch += num_ch
    pos = (pos + 7) & -8
    num_str, = struct.unpack_from('>H', data, pos + 10)
    str_table, pos = decode_len_str(data, pos + 12, num_str)
    all_channels = list(expand_repeating_channels(all_channels, str_table))
    pos = (pos + 7) & -8

    channel_map = assign_channel_addresses(all_channels)

    t1 = time.perf_counter()
    all_channel_data = decode_rows(data, pos, progress)

    t2 = time.perf_counter()
    assign_data(channel_map, all_channel_data)

    t3 = time.perf_counter()
    print('decoder time: %.4f %.4f %.4f' % (t1-t0, t2-t1, t3-t2))

    #csv_analyze(channel_map)

    last_time = max(ch.timecodes[-1] for ch in all_channels
                    if ch.timecodes is not None)
    try:
        laps = generate_laps(channel_map, last_time)
    except:
        laps = [base.Lap(0, 0, last_time)]

    # No metadata is provided, parse the filename instead?
    metadata = {}
    metadata['Log Date'] = 'Unknown'
    metadata['Log Time'] = 'Unknown'
    return base.LogFile({ch.name: base.Channel(ch.timecodes, # convert to float64?
                                               ch.samples,
                                               ch.name,
                                               ch.unit,
                                               int(np.ceil(np.log10(ch.scale))),
                                               True)
                         for ch in all_channels
                         if ch.timecodes is not None},
                        laps,
                        metadata,
                        ['gps.speed', 'gps.latitude', 'gps.longitude', 'gps.height'],
                        fname)
