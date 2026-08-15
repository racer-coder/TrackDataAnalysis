
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
import concurrent.futures
from copy import copy
from dataclasses import dataclass, field, replace
import gzip
import mmap
import struct
import sys
import time
import zlib

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



class EMUPROLogReader:
    # Log file reader for EMU PRO line of ECUs.  Handles log files
    # written to by the GUI and those written by the ECU directly to
    # USB.

    # File might be gzipped (usually so if written by the GUI).  Try
    # that first.

    # Now we have a file that should start with b'\xefEML'. The header
    # seems to vary - 0x80 bytes for GUI, 0x200 bytes for ECU.  I
    # haven't really seen anything useful in it.

    # The rest of the file seems to be in chunks that have a 4-8 byte
    # header starting with a byte 0xe? which determins what the next
    # chunk is.  The big endian u16 at +2 from the header byte
    # indicates where the next section starts.

    # First section is a zlib-compressed version of the logging layout of
    # this version of firmware.  Within that you have the same section
    # format with the following data included:
    # 1. unit names
    # 2. section names (such as engine/, fuel/, drivetrain/, etc)
    # 3. parameter lists (list of values for bitfields or state vars)
    # 4. structure names
    # 5. section details (num fields, human-facing names like Engine)
    # 6. field lists for structures (which can refer to other structs)
    # 7. field lists for sections (which can refer to structs)
    # In order to get a list of fields that can be referenced later,
    # we need to sort the fields by size.  8-bit first, then 32-bit,
    # then 16-bit.  A quirk of the system is that 32-bit fields take
    # two 16-bit slots.  Some counts/indices refer to fields, some
    # refer to slots.

    # Next section describes the user/custom fields (Project Tree).
    # 1. User field units, multiplier, and offset, grouped by size (8/16/32)
    # 2. User field names and size, as ordered in the UI.

    # Next section describes the logging rate/layout for the main and
    # custom profile.
    # 1. The first part describes how the 10Hz, 5Hz, and 1Hz fields
    #    are distributed among the 25Hz refresh rate.  The 10Hz and 5Hz
    #    are divided into 5 sections (10Hz is logged AB, CD, EA, BC, DE
    #    to align 10Hz to 25Hz) while 1Hz fields are divided into 25 sections.
    # 2. Contiguous blocks of fields are listed along with their refresh rate.

    # Finally the rows themselves.  There's a 25Hz cycle field to
    # indicate which layout applies, along with a length field.
    @dataclass
    class Group:
        name: str = ''
        num_fields: int = 0
        is_group: bool = False
        fields: list = None

    @dataclass
    class Section:
        name: str = ''
        path: str = ''
        idx: int = 0
        num_fields: int = 0

    @dataclass
    class ParamList:
        name: str = ''
        dtype: str = None
        is_bitfield: bool = False
        values: dict = None

    @dataclass
    class Field:
        name: str = ''
        internal_name: str = ''
        parent: object = None
        unit: object = None
        dtype: object = None
        vmin: int = None
        vmax: int = None
        mult: int = 1
        div: int = 1
        offs: int = 0
        table_naxis: int = None
        rate_idx: int = None

    print_all = False
    rate_list = [
        '500Hz',
        '250Hz',
        '100Hz',
        '50Hz',
        '25Hz',
        '10Hz',
        '5Hz',
        '1Hz',
    ]


    def __init__(self):
        self.units = []
        self.sections = []
        self.paramlists = []
        self.structs = []

    def decode_string(self, data, pos):
        end = pos + data[pos:].index(0)
        return end + 1, data[pos:end].decode('utf-8')

    def decode_comp_field(self, fields, data, pos, parent):
        orig_pos = pos
        has_repeat = False
        has_type = False
        has_struct = False
        has_units = False
        has_decoder = False
        has_divider = False
        has_multiplier = False
        has_offset = False
        has_gaugemin = False
        has_gaugemax = False
        has_table = False
        b1 = data[pos]
        b2 = data[pos+1]
        has_human_name = b1 & 0x80
        if (b1 & 15) == 8:
            dtype = None
            has_type = True
        elif (b1 & 15) == 9:
            dtype = None
            has_struct = True
        else:
            #if (b1 & 15) == 4:
            #    print('%x' % orig_pos)
            dtype = {0: 'B',
                     1: 'b',
                     2: 'H',
                     3: 'h',
                     4: '?',
                     5: 'I',
                     6: 'i',}[b1 & 15]
        has_units = (b2 & 1)
        if b2 & 2:
            has_gaugemin = True
        if b2 & 4:
            has_gaugemax = True
        has_multiplier = (b2 & 16)
        has_divider = (b2 & 8)
        has_offset = b2 & 0x20
        has_repeat = b1 & 0x10
        has_table = b2 & 0x80
        b1 &= -160
        b2 &= -192
        if b1 == 0 and b2 == 0:
            pass
        else:
            print('unknown at %x: %s %02x %02x' % (pos, parent.name, data[pos], data[pos+1]))
            assert False
        pos += 2
        t = None
        if has_type:
            t = self.paramlists[data[pos]]
            dtype = t.dtype
            pos += 1
        if has_struct:
            assert t is None
            t = self.structs[data[pos]]
            pos += 1
        r = None
        if has_repeat:
            r = data[pos]
            pos += 1
        u = None
        if has_units:
            u = self.units[data[pos]]
            pos += 1
        gmin = None
        if has_gaugemin:
            gmin = struct.Struct('>' + dtype).unpack_from(data, pos)[0]
            pos += struct.Struct('>' + dtype).size
        gmax = None
        if has_gaugemax:
            gmax = struct.Struct('>' + dtype).unpack_from(data, pos)[0]
            pos += struct.Struct('>' + dtype).size
        d = 1
        if has_divider:
            d = data[pos] * 256 + data[pos + 1]
            pos += 2
        m = 1
        if has_multiplier:
            m = data[pos] * 256 + data[pos + 1]
            pos += 2
        o = 0
        if has_offset:
            o = struct.unpack_from('>h', data, pos)[0]
            pos += 2
        naxis = None
        if has_table:
            naxis = data[pos]
            pos += 1
        pos, internal_name = self.decode_string(data, pos)
        if has_human_name:
            pos, human_name = self.decode_string(data, pos)
        else:
            human_name = internal_name
        human_name = human_name or 'Value'
        if has_struct:
            assert r is None
            for i in t.fields:
                f = replace(i)
                f.name = human_name + ' ' + i.name
                f.internal_name = internal_name + '/' + i.internal_name
                f.parent = parent
                assert(f.dtype)
                fields.append(f)
        else:
            assert dtype
            for i in range(1 if r is None else r):
                f = self.Field()
                f.name = human_name
                f.internal_name = internal_name
                if r is not None:
                    f.name = f.name.replace('$', '%d' % (i+1))
                    f.internal_name = f.internal_name.replace('$', '%d' % (i+1))
                f.parent = parent
                f.dtype = dtype
                f.unit = u
                f.mult = m
                f.div = d
                f.offs = o
                fields.append(f)
        if self.print_all:
            print('%x' % orig_pos, '%s %s %s %d/%d+%d %s:%s %s %s %s' % ('%dx' % r if r else 'None', t, u, m, d, o, gmin, gmax, '' if naxis is None else 'T%d' % naxis,
                                                                         internal_name, human_name or 'Value'))
        return pos


    # handles compressed data that describes the base map layout
    def decode_comp(self, data):
        # decode units.  Not sure exactly where it starts or exactly how long it is...
        pos = 18
        while len(self.units) < data[4]:
            end = pos + data[pos:].index(0)
            name = data[pos:end].decode('utf-8')
            if self.print_all:
                print('UNIT', len(self.units), name)
            self.units.append(name)
            pos = end + 1

        # next comes sections?  Not sure if data[5] or data[6] better indicates how many there are?
        assert data[5] == data[6]
        while len(self.sections) < data[5]:
            end = pos + data[pos:].index(0)
            name = data[pos:end].decode('utf-8')
            if self.print_all:
                print('SECTION', len(self.sections), name)
            self.sections.append(self.Section(path=name, idx=len(self.sections)))
            pos = end + 1

        desc_map = {
            0: 'B',
            1: 'b',
            2: 'H',
        }
        while len(self.paramlists) < data[7]:
            desc = data[pos]
            pos += 1
            end = pos + data[pos:].index(0)
            name = data[pos:end].decode('utf-8')
            pos = end + 2
            assert desc < 32
            if (desc & 15) not in desc_map:
                print(desc, name)
            if self.print_all:
                print(len(self.paramlists), desc_map[desc & 15], 'bitfield' if desc & 16 else '', name)
            pl = self.ParamList()
            pl.name = name
            pl.dtype = desc_map[desc & 15]
            pl.is_bitfield = (desc & 16) != 0
            pl.values = {}
            desc = 'B' if pl.is_bitfield else desc_map[desc]
            decoder = struct.Struct('>' + desc)
            w = decoder.size
            for i in range(data[pos - 1]):
                k = decoder.unpack_from(data, pos)[0]
                pos, v = self.decode_string(data, pos + w)
                if self.print_all:
                    print('\t%d\t%s' % (k, v))
                pl.values[k] = v
            self.paramlists.append(pl)

        while len(self.structs) < data[10]: # what about data[9]
            val = data[pos]
            pos += 1
            end = pos + data[pos:].index(0)
            name = data[pos:end].decode('utf-8')
            if self.print_all:
                print('STRUCT', len(self.structs), val, name)
            self.structs.append(self.Group(name=name, num_fields = val,
                                           is_group=len(self.structs) < data[8]))
            pos = end + 1

        # back to sections?
        for idx, sect in enumerate(self.sections):
            assert idx == data[pos+1] # why?
            sect.num_fields = data[pos]
            pos += 2
            end = pos + data[pos:].index(0)
            sect.name = data[pos:end].decode('utf-8')
            if self.print_all:
                print('SECTNAME', idx, sect.num_fields, sect.name)
            pos = end + 1

        for s in self.structs:
            if self.print_all:
                print(s.name)
            s.fields = []
            for n in range(s.num_fields):
                pos = self.decode_comp_field(s.fields, data, pos, s)

        fields = []
        for s in self.sections:
            if self.print_all:
                print(s.name)
            for n in range(s.num_fields):
                pos = self.decode_comp_field(fields, data, pos, s)

        if self.print_all:
            print('%d fields, %d bytes' % (len(fields), sum(struct.Struct(f.dtype).size
                                                            for f in fields
                                                            if f.dtype)))

            print('%x' % pos)

        dsort = {
            'B': 1,
            'b': 1,
            '?': 1,
            'H': 20,
            'h': 20,
            'I': 4,
            'i': 4,
        }

        fields.sort(key=lambda x: dsort[x.dtype])

        retfields = []
        for i, f in enumerate(fields):
            if dsort[f.dtype] != 1 and i > 0 and dsort[fields[i-1].dtype] == 1:
                while i & 3:
                    retfields.append(None)
                    i += 1
            if dsort[f.dtype] == 4:
                retfields.append(None)
            retfields.append(f)

        return retfields

    def decode_user(self, data, pos, fields):
        # @0x55ea: u16 0x0c07 - length field until 0x61f0/triplets, header starts at 55e8
        # @0x55ee: u16 0x0474 - length field until 0x5a60/strings? header might start at 55e8

        # user fields description and names?

        # 0x5604 - 0x5a60: 12 byte records?

        if self.print_all:
            print('%x' % pos)

        assert data[pos] == 0xe9

        user_fields = [[], [], []]
        for f in fields:
            if f and f.parent.path == 'custom/':
                if f.internal_name.startswith('unified8Channels'):
                    user_fields[0].append(f)
                elif f.internal_name.startswith('unified16Channels'):
                    user_fields[1].append(f)
                elif f.internal_name.startswith('unified32Channels'):
                    user_fields[2].append(f)

        field_desc_pos = pos + 28
        for num, lst in zip(struct.unpack_from('>HHH', data, pos + 10), user_fields):
            for idx, f in zip(range(num), lst):
                if (data[field_desc_pos] & 0x80) == 0:
                    f.dtype = f.dtype.upper()
                # not sure what the rest of the values are...
                unit_idx, f.mult, f.offs = struct.unpack_from('>xxxBff', data, field_desc_pos)
                f.unit = self.units[unit_idx]
                field_desc_pos += 12

        for l in user_fields:
            l.reverse()

        # strings from 0x5a60 to 0x61e9
        p = field_desc_pos # this isn't reliable: pos + 4 + data[pos+6]*256 + data[pos+7]
        while p < pos + data[pos+2]*256 + data[pos+3]:
            # data[p]>>5 seems to indicate how many fields it takes
            # 0 = not a field
            # 7 = timer
            # 1,2,3 = 1,2,4 bytes
            name_len = data[p] & 0x1f
            val_type = data[p] >> 5
            if val_type == 7:
                # special timer string?
                p += 1
            p += 1
            name = data[p:p+name_len].decode('utf-8')
            p += name_len
            if val_type == 0:
                pass # not used
            elif val_type <= 3:
                user_fields[val_type-1].pop().name = name
            elif val_type == 7:
                # not sure what order to process these...
                user_fields[0].pop().name = name + ' Running'
                user_fields[2].pop().name = name + ' Elapsed'
                user_fields[0].pop().name = name + ' Value'

        pos = (pos + data[pos+2]*256 + data[pos+3] + 7) & -8
        return pos

    @dataclass
    class Decoder:
        rate_idx: int = 0
        nbytescopy: int = 0 # number of bytes to copy, should be multiple of 4
        nbytesadv: int = 0 # number of bytes to advance when scanning input buffer
        nslots: int = 0 # number of 'slots' consumed, 2 for 32-bit, 1 for 16 or 8-bit
        fields: list = field(default_factory=list) # list of Field
        timecodes: array = field(default_factory=lambda: array('i'))
        samples: bytearray = field(default_factory=bytearray) # may be shared by multiple Decoder (for >= 50Hz)

        def add_field(self, f):
            if not f.decoder_offsets:
                self.fields.append(f)
            f.decoder_offsets.append(self.nbytesadv)
            sz = struct.Struct(f.dtype).size
            self.nslots += 1 if sz < 4 else 2
            self.nbytesadv += sz
            self.nbytescopy = self.nbytesadv if self.nbytesadv < 3 else ((self.nbytesadv + 3) & -4)

        def extract_fields(self):
            tcadder = [np.arange(0, 40, 2),
                       np.arange(0, 40, 4),
                       np.arange(0, 40, 10),
                       np.arange(0, 40, 20),
                       np.array([0]),
                       np.array([0]),
                       np.array([0]),
                       np.array([0]),
                       ]
            tccache = [None] * 8
            self.timecodes = np.asarray(self.timecodes)
            for f in self.fields:
                dsize = struct.Struct(f.dtype).size
                f.data = [np.asarray(memoryview(self.samples)[o:o+((len(self.samples)-o)&-dsize)].cast(f.dtype)[::self.nbytescopy//dsize]).reshape(-1, 1)
                          for o in f.decoder_offsets]
                if len(f.data) > 1:
                    f.data = [np.concatenate(f.data, axis=1)]
                f.data = f.data[0].reshape(-1).byteswap().astype(np.float32 if dsize < 4 else np.float64)
                if f.mult != 1:
                    f.data *= f.mult
                if f.div != 1:
                    f.data /= f.div
                if f.offs != 0:
                    f.data += f.offs
                if tccache[f.rate_idx] is None:
                    tc = self.timecodes
                    tcadd = tcadder[f.rate_idx]
                    if len(tcadd) > 1:
                        gaps = np.append(tc[1:] - tc[:-1],
                                         [(tc[-1] - tc[0]) // (len(tc) - 1)])
                        gaps = gaps.reshape(len(gaps), 1) * tcadd // 40 + tc.reshape(len(self.timecodes), 1)
                        tc = gaps.reshape(-1)
                    tccache[f.rate_idx] = tc // 10
                f.tc = tccache[f.rate_idx]
                assert(f.data.shape == f.tc.shape)


    def decode_buffer_layout(self, data, pos, fields):
        orig_pos = pos

        # @0x61f2: u16 0x087C - length field for next two chunks combined, including 8 byte header starting at 0x61f0?
        # Logging rate for base profile vs custom profile?
        # logging rate 1, 5, 10, 25, 50, 100, 250, 500
        # logging for all channels, internal and user?

        # something about layout; (f[pos]*256+f[pos+1])>>4 -> position in buffer, f[pos+2] -> number of entries (NOT BYTES!), f[pos+1]&15 -> ??? usually 4, 5, 6, 7, rarely 0 or 3, list seems to be sorted by this? MAYBE LOGGING RATE, INVERTED!!
        # @0x61f8: u16 0x043A - length field
        self.layouts = []

        pos += 8
        for i in range(struct.unpack_from('>H', data, pos - 2)[0]):
            end = pos + struct.unpack_from('>H', data, pos)[0]
            layout = []
            for p in range(pos + 0x68, end, 3):
                start = data[p]*16 + (data[p+1] >> 4)
                ratecode = data[p+1] & 15
                num = data[p+2]
                if self.print_all:
                    print('LIST %d %4d-%4d %s' % (i, start, start+num-1, self.rate_list[ratecode]))
                for j in range(start, start+num):
                    if fields[j]:
                        if self.print_all:
                            print('  %s' % fields[j].name)
                        layout.append((fields[j], ratecode))
            pos = end
            self.layouts.append(layout)

        if self.print_all:
            rate_bytes = {r: 0 for r in self.rate_list}
            for f, r in self.layouts[0]:
                rate_bytes[self.rate_list[r]] += struct.Struct(f.dtype).size
            print(rate_bytes)

        for f, r in self.layouts[0]:
            f.decoder_offsets = []

        # lets start construction of decoder
        decoders = []
        for f, r in self.layouts[0]:
            if r > 4:
                break
            if not decoders or decoders[-1][0] != r:
                decoders.append((r, []))
            f.rate_idx = r
            decoders[-1][1].append(f)
        divs = [1, 2, 5, 10, 20]
        bigdecoder = self.Decoder(rate_idx=4)
        for tc in range(20):
            for r, fields in decoders:
                if tc % divs[r] == 0:
                    for f in fields:
                        bigdecoder.add_field(f)
        self.unique_decoders = [bigdecoder]
        self.decoders = [[bigdecoder] for i in range(25)]

        # 10Hz
        decoders = []
        dlimit = struct.unpack_from('>5H', data, orig_pos + 0x22)
        for f, r in self.layouts[0]:
            if r == 5:
                if not decoders or decoders[-1].nslots >= dlimit[len(decoders)-1]:
                    decoders.append(self.Decoder(rate_idx = r))
                f.rate_idx = r
                decoders[-1].add_field(f)
        assert len(decoders) <= 5
        self.unique_decoders.extend(decoders)
        for i in range(5):
            for j, d in enumerate(decoders):
                self.decoders[i*5+j//2].append(d)
            for j, d in enumerate(decoders):
                self.decoders[i*5+(5+j)//2].append(d)

        # 5Hz
        decoders = []
        dlimit = struct.unpack_from('>5H', data, orig_pos + 0x2c)
        for f, r in self.layouts[0]:
            if r == 6:
                if not decoders or decoders[-1].nslots >= dlimit[len(decoders)-1]:
                    decoders.append(self.Decoder(rate_idx = r))
                f.rate_idx = r
                decoders[-1].add_field(f)
        assert len(decoders) <= 5
        self.unique_decoders.extend(decoders)
        for i in range(5):
            for j, d in enumerate(decoders):
                self.decoders[i*5+j].append(d)

        # 1Hz
        decoders = []
        dlimit = struct.unpack_from('>25H', data, orig_pos + 0x36)
        for f, r in self.layouts[0]:
            if r == 7:
                if not decoders or decoders[-1].nslots >= dlimit[len(decoders)-1]:
                    decoders.append(self.Decoder(rate_idx = r))
                f.rate_idx = r
                decoders[-1].add_field(f)
        assert len(decoders) <= 25
        self.unique_decoders.extend(decoders)
        for j, d in enumerate(decoders):
            self.decoders[j].append(d)

        if self.print_all:
            print('DECODER SIZES')
            for i, decoders in enumerate(self.decoders):
                print(i, sum(d.nbytesadv for d in decoders), ','.join('%s' % d.nbytesadv for d in decoders))

        return (orig_pos + struct.unpack_from('>H', data, orig_pos + 2)[0] + 7) & -8


    def decode_rows(self, data, pos, progress):
        timestamp_us = 0
        last_us = None
        self.buf = bytearray()
        ltdec = struct.Struct('>HxxxxI')
        next_update = 8 << 20
        decoder_lens = [12 + sum(d.nbytesadv for d in dlist) for dlist in self.decoders]
        while pos + 8 < len(data):
            if data[pos] != 0xed:
                # Did the user change a custom variable/desc/config?
                # Well, we don't handle that yet, so just skip over it
                # and keep going.
                if data[pos] == 0xe9:
                    pos = (pos + data[pos+2]*256 + data[pos+3] + 7) & -8
                    continue
                print('End at pos %x' % pos)
                break
            if pos >= next_update:
                if progress:
                    progress(pos, len(data))
                next_update += 8 << 20
            l, tick_us = ltdec.unpack_from(data, pos + 2)
            if last_us is not None:
                timestamp_us += (tick_us - last_us) & 0xffffff
            last_us = tick_us
            if pos + l > len(data) or l < decoder_lens[data[pos+1]]:
                print('Cut short')
                break
            p = pos + 12
            timecode = timestamp_us // 100 # tenths of a ms, allows the use of 32-bit integers
            for d in self.decoders[data[pos+1]]:
                d.timecodes.append(timecode)
                d.samples.extend(data[p:p+d.nbytescopy])
                p += d.nbytesadv
            if self.print_all:
                print('%.2f' % (timestamp_us/1000000),
                      l, '%02x%02x' % (data[pos+4], data[pos+5]),
                      "%x" % (p),
                      ' '.join('%02x' % c for c in data[p:p+16]),
                      pos+l-p)
            adv = (l+7) & -8
            pos += adv

    def extract_field_data(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as worker:
            # iterate to find any failures in execution
            for r in worker.map(lambda d: d.extract_fields(), self.unique_decoders):
                pass

    def decode(self, data, progress):
        # @0x0202: u16 0x53e5 - length field until 0x55e8? matches perfectly to non null 0x0200 - 0x55e4 inclusive range
        # @0x0206: start of zlib compressed data, presumably describing data
        t1 = time.perf_counter()

        pos = data[128:2048:8].index(0xe3) * 8 + 128
        end = pos + data[pos+2]*256 + data[pos+3]
        if end > pos + 6:
            fields = self.decode_comp(zlib.decompress(data[pos+6:end]))
        pos = (end + 7) & -8

        pos = self.decode_user(data, pos, fields)

        pos = self.decode_buffer_layout(data, pos, fields)

        t2 = time.perf_counter()

        self.decode_rows(data, pos, progress)

        t3 = time.perf_counter()

        self.extract_field_data()

        t4 = time.perf_counter()

        print('Decode time %.3f / %.3f / %.3f' % (t2-t1, t3-t2, t4-t3))



def ECUMASTER_EMUPRO(fname, progress):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if m[0] != 0xef or m[1] != 0x65 or m[2] != 0x6d or m[3] != 0x6c: # search for 'eml' tag
                # probably gzip created by UI tool, rather than logged directly on USB from ECU
                m = gzip.decompress(m)
            reader = EMUPROLogReader()
            reader.decode(m, progress)

            lf = base.LogFile({f.name: base.Channel(f.tc, f.data, f.name, f.unit,
                                                    int(np.ceil(np.log10(f.div/f.mult)-0.1)), # decimal points...
                                                    True) # interpolate?
                               for d in reader.unique_decoders
                               for f in d.fields}, # name: base.Channel
                              [base.Lap(0,
                                        min(d.timecodes[0] for d in reader.unique_decoders) // 10,
                                        max(d.timecodes[-1] for d in reader.unique_decoders) // 10)], # laps
                              {'Log Date': 'Unknown',
                               'Log Time': 'Unknown'}, # metadata
                              [None, None, None, None], # special channels
                              fname)
            return lf
