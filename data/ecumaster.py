
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
from copy import copy
from dataclasses import dataclass, field
import gzip
import struct
import sys

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
    idx: array = field(default_factory=lambda: array('H'))
    data: bytearray = field(default_factory=bytearray)

    def result(self):
        if len(self.data) > len(self.tc):
            data = np.asarray(memoryview(self.data).cast('H'))
        else:
            data = np.array(self.data, dtype=np.uint16)

        stack = np.column_stack([data,
                                 memoryview(self.tc).cast('B').cast('H')[0::2],
                                 memoryview(self.tc).cast('B').cast('H')[1::2],
                                 self.idx])
        stack = memoryview(stack).cast('B').cast('Q')

        stack = np.sort(stack, kind='stable')
        ch = stack >> 48
        chboundaries = np.concatenate([[0],
                                       1 + np.nonzero(ch[1:] != ch[:-1])[0],
                                       [len(stack)]])
        return {ch[start]: (np.asarray(memoryview(stack[start:end]).cast('B')[2:-2].cast('I')[::2]),
                            np.asarray(memoryview(stack[start:end]).cast('B').cast('H')[::4]))
                for start, end in zip(chboundaries[:-1], chboundaries[1:])}

fast_ch_table = array('H', range(65536))
repeat_table = [20, 10, 5, 2, 1, 1, 1]

def decode_row(timestamp, data, all_channel_data):
    byte_cutoff = 0x200
    pos = 4 + data[3] * 4
    tca = array('I', [0])
    for i in struct.unpack_from('>%dI' % data[3], data, 4):
        start_ch = i >> 20
        num_ch = i & 0xfff # not sure how much to mask
        assert (start_ch < byte_cutoff) == (start_ch + num_ch <= byte_cutoff)

        repeat = repeat_table[(i >> 16) & 15]

        aci = start_ch >= byte_cutoff
        next_pos = pos + num_ch * repeat * (aci + 1)
        assert (pos & aci) == 0 # round up when dealing with 2 byte elems?

        ad = all_channel_data[aci]
        ad.idx.extend(fast_ch_table[start_ch:start_ch + num_ch] * repeat)
        for i in range(timestamp, timestamp + 40, 40 // repeat):
            tca[0] = i
            ad.tc.extend(tca * num_ch)
        ad.data += data[pos:next_pos]
        pos = next_pos
    assert pos <= len(data)

def decode_rows(data, pos, progress):
    all_channel_data = [AggregateData(), AggregateData()]

    first_timestamp = None
    timestamp = 0
    update_pos = 1 << 20
    while pos + 4 < len(data):
        if pos >= update_pos:
            if progress:
                progress(pos, len(data))
            update_pos += 1 << 20
        l = data[pos+1] * 8 + 8
        if pos + l > len(data):
            break
        timestamp += (data[pos+2] - timestamp) % 25
        if first_timestamp is None:
            first_timestamp = timestamp
        decode_row((timestamp - first_timestamp) * 40, data[pos:pos+l], all_channel_data)
        pos += l
    return all_channel_data

def assign_data(channel_map, all_channel_data):
    data = all_channel_data[0].result() | all_channel_data[1].result()
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
        if ch.samples is not None:
            ch.samples = ch.samples / ch.scale + ch.adder

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

    all_channel_data = decode_rows(data, pos, progress)

    assign_data(channel_map, all_channel_data)

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
