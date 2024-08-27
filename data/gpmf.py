
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import datetime
import mmap
import pprint
import struct

def valid_4cc(fc):
    # also allow 0xa9 at position 0? or is it position 3?
    return all((c >= 'A' and c <= 'Z') or
               (c >= 'a' and c <= 'z') or
               (c >= '0' and c <= '9') or
               c == ' '
               for c in fc)

class BoxParser:
    def __init__(self, m):
        self._boxes = {}
        p = 0
        #print('BOX:')
        while p < len(m):
            l, fc = struct.unpack_from('>I4s', m, p)
            if l == 1:
                p += 8
                l, = struct.unpack_from('>Q', m, p)
                l -= 8
            else:
                assert l >= 8
            assert p + l <= len(m)
            fc = fc.decode('utf-8')
            assert valid_4cc(fc)
            #print('  %s %d' % (fc, l))
            if fc not in self._boxes:
                self._boxes[fc] = []
            self._boxes[fc].append(m[p+8:p+l])
            p += l

    def count(self, k):
        return len(self._boxes[k]) if k in self._boxes else 0

    def get_one(self, k):
        assert len(self._boxes[k]) == 1
        return self._boxes[k][0]

    def get_list(self, k):
        return self._boxes[k]

class Payload:
    def __init__(self, m, s, e):
        self.data = m
        self.start_time = s
        self.end_time = e

def parse_mp4(m):
    root = BoxParser(m)
    moov = BoxParser(root.get_one('moov'))
    for trak in moov.get_list('trak'):
        # Check if this is a metadata track?
        trak = BoxParser(trak)
        mdia = BoxParser(trak.get_one('mdia'))
        hdlr = struct.unpack_from('>8x4s', mdia.get_one('hdlr'), 0)
        if hdlr[0].decode('utf-8') != 'meta':
            #print('not meta - %s' % hdlr[0])
            continue

        trak_clockdemon, = struct.unpack_from('>12xI', mdia.get_one('mdhd'), 0)

        # Check if this is a GoPro metadata track?
        minf = BoxParser(mdia.get_one('minf'))
        stbl = BoxParser(minf.get_one('stbl'))
        stsd = struct.unpack_from('>12x4s', stbl.get_one('stsd'), 0)
        if stsd[0].decode('utf-8') != 'gpmd':
            #print('not gpmd - %s' % stsd[0])
            continue

        metadataoffset_clockcount = 0 # XXX edts - editlist not supported yet

        stts = stbl.get_one('stts')
        num_samples, = struct.unpack_from('>4xI', stbl.get_one('stts'), 0)
        meta_clockdemon = trak_clockdemon
        metadatalength = 0
        values = struct.unpack_from('>%dI' % (2 * num_samples), stbl.get_one('stts'), 8)
        samples = 0
        for samplecount, duration in zip(values[::2], values[1::2]):
            samples += samplecount
            metadatalength += samplecount * duration
            if samplecount > 1 or num_samples == 1 or basemetadataduration == 0:
                basemetadataduration = metadatalength / samples # really we should trac all the sample sets, but it turns out there's usually just one
        assert num_samples <= 1 # otherwise we should improve basemetadataduration support

        equal_sample_size, num_samples = struct.unpack_from('>4xII', stbl.get_one('stsz'), 0)
        assert num_samples < 5184000 # sanity check
        if equal_sample_size:
            metasizes = [equal_sample_size] * num_samples
        else:
            metasizes = struct.unpack('>%dI' % num_samples, stbl.get_one('stsz')[12:])

        if stbl.count('stsc'):
            num_x, = struct.unpack_from('>4xI', stbl.get_one('stsc'), 0)
            # (chunk_num, samples, id)
            metastsc = [struct.unpack_from('>III', stbl.get_one('stsc'), 8 + i * 12)
                        for i in range(num_x)]
        else:
            metastsc = []

        if stbl.count('stco'):
            num_samples, = struct.unpack_from('>4xI', stbl.get_one('stco'), 0)
            metaoffsets = struct.unpack('>%dI' % num_samples, stbl.get_one('stco')[8:])
        else:
            num_samples, = struct.unpack_from('>4xI', stbl.get_one('co64'), 0)
            metaoffsets = struct.unpack('>%dQ' % num_samples, stbl.get_one('co64')[8:])
        assert num_samples == len(metasizes) # need to chunk it up using metastsc, see GPMF_mp4reader.c maybe line 614?

        assert len(metaoffsets) == len(metasizes)
        payloads = [Payload(m[offs : offs+sz],
                            (index * basemetadataduration + metadataoffset_clockcount) / meta_clockdemon,
                            (min((index + 1) * basemetadataduration,
                                 metadatalength) + metadataoffset_clockcount) / meta_clockdemon)
                    for index, (offs, sz) in enumerate(zip(metaoffsets, metasizes))]
        return payloads
    print('nothing found')
    return []

def type_date(d):
    # yymmddhhmmss.sss
    try:
        d = d.decode('ascii')
        #print(d)
        return datetime.datetime(2000 + int(d[0:2]), int(d[2:4]), int(d[4:6]),
                                 int(d[6:8]), int(d[8:10]), int(d[10:12]),
                                 int(d[13:16]),
                                 tzinfo=datetime.timezone.utc)
    except:
        print('cannot parse date', d)
        return None

type_map = { # map GoPro type to python struct module type
    'b': ('b', None),
    'B': ('B', None),
    'd': ('d', None),
    'f': ('f', None),
    'F': ('4s', lambda x: x.decode('ascii')),
    'G': ('16s', None), # GUID
    'j': ('q', None),
    'J': ('Q', None),
    'l': ('i', None),
    'L': ('I', None),
    'q': ('i', lambda x: x * (2 ** -16)),
    'Q': ('q', lambda x: x * (2 ** -32)), # NOTE: precision error squeezing into 64-bit float
    's': ('h', None),
    'S': ('H', None),
    'U': ('16s', type_date),
}

class KLV_data:
    def __init__(self, qt, s, data):
        self.qt = qt
        self.s = s
        self.data = data
        self.cache = None

    def build_cache(self):
        if self.cache is not None: return

        qm = ''.join(type_map[t][0] for t in self.qt)
        qf = [type_map[t][1] for t in self.qt]
        rcount = self.s // struct.calcsize(qm)
        if self.s != struct.calcsize(qm) * rcount:
            self.cache = []
        else:
            data = struct.unpack('>' + qm * rcount * (len(self.data) // self.s), self.data)
            if data:
                data = list(zip(*[[f(d) for d in data[i::len(qf)]]
                                  if f else data[i::len(qf)]
                                  for i, f in enumerate(qf)]))
                if len(qf) == 1:
                    data = [d[0] for d in data]
                if rcount > 1:
                    data = [data[i:i+rcount] for i in range(0, len(data), rcount)]
            self.cache = data

    def __len__(self):
        self.build_cache()
        return len(self.cache)

    def __getitem__(self, idx):
        self.build_cache()
        return self.cache[idx]

    def __repr__(self):
        self.build_cache()
        return pprint.pformat(self.cache)

def KLV_parser(m):
    ret = []
    p = 0
    qtype = None
    while p < len(m):
        fc, t, s, r = struct.unpack_from('>4scBH', m, p)
        fc = fc.decode('utf-8')
        assert valid_4cc(fc)
        t = t.decode('utf-8')
        data = m[p+8:p+8+s*r]
        if t == '\0':
            data = KLV_parser(data)
        elif t == 'c':
            if s == 1:
                s = r
                r = 1
            data = [bytes(data[i*s:i*s+s]).decode('latin_1').rstrip('\0') # GoPro claims ASCII but they use some high values
                    for i in range(r)]
            if r == 1:
                data = data[0]
            if fc == 'TYPE':
                qtype = data
        else:
            data = KLV_data(qtype if t == '?' else t,
                            s, bytes(data))
        ret.append((fc, data))
        p += 8 + s * r + ((-s * r) & 3)
    return ret

def MP4_estimate_start_time(fname):
    with open(fname, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            try:
                m.madvise(mmap.MADV_RANDOM)
            except AttributeError:
                pass # Windows
            ret = []
            for p in parse_mp4(memoryview(m)):
                for fc, d in KLV_parser(p.data)[0][1]:
                    if fc == 'STRM':
                        for fc2, d2 in d:
                            if fc2 == 'GPSU':
                                if d2[0]:
                                    ret.append(d2[0] - datetime.timedelta(seconds=p.start_time))
            p = None # drop last reference to mmap
            if not ret:
                return None
            ret.sort()
            base_line = ret[len(ret) // 2]
            ret = [r - base_line
                   for r in ret
                   if abs((r - base_line).total_seconds()) < 10]
            return base_line + sum(ret, start=datetime.timedelta()) / len(ret)

if __name__ == '__main__':
    import sys
    #print(MP4_estimate_start_time(sys.argv[1]))
    #sys.exit(0)
    with open(sys.argv[1], 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            m.madvise(mmap.MADV_RANDOM)
            for i, p in enumerate(parse_mp4(memoryview(m))):
                #print('Payload %f-%f' % (p.start_time, p.end_time))
                p = KLV_parser(p.data)
                for fc, d in p[0][1]:
                    if fc == 'STRM':
                        for fc2, d2 in d:
                            if fc2 == 'GPSU':
                                print(d2)
                #pprint.pp((i, KLV_parser(p.data)))
            p = None
