
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import datetime
import mmap
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
        print('BOX:')
        while p < len(m):
            l, fc = struct.unpack_from('>I4s', m, p)
            assert l >= 8 # l==1 means 64-bit length follows?
            assert p + l <= len(m)
            fc = fc.decode('utf-8')
            assert valid_4cc(fc)
            print('  %s' % fc)
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
            print('.. not meta')
            continue

        trak_clockdemon, = struct.unpack_from('>12xI', mdia.get_one('mdhd'), 0)

        # Check if this is a GoPro metadata track?
        minf = BoxParser(mdia.get_one('minf'))
        stbl = BoxParser(minf.get_one('stbl'))
        stsd = struct.unpack_from('>12x4s', stbl.get_one('stsd'), 0)
        if stsd[0].decode('utf-8') != 'gpmd':
            print('.. not gpmd')
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

        num_samples, = struct.unpack_from('>4xI', stbl.get_one('stco'), 0)
        metaoffsets = struct.unpack('>%dI' % num_samples, stbl.get_one('stco')[8:])
        assert num_samples == len(metasizes) # need to chunk it up using metastsc, see GPMF_mp4reader.c maybe line 614?

        assert len(metaoffsets) == len(metasizes)
        payloads = [Payload(m[offs : offs+sz],
                            (index * basemetadataduration + metadataoffset_clockcount) / meta_clockdemon,
                            (min((index + 1) * basemetadataduration,
                                 metadatalength) + metadataoffset_clockcount) / meta_clockdemon)
                    for index, (offs, sz) in enumerate(zip(metaoffsets, metasizes))]
        return payloads
    return None

def type_date(d):
    # yymmddhhmmss.sss
    d = d.decode('ascii')
    return datetime.datetime(2000 + int(d[0:2]), int(d[2:4]), int(d[4:6]),
                             int(d[6:8]), int(d[8:10]), int(d[10:12]),
                             int(d[13:16]))

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
        self.cache = None # (struct decoder, qf, rcount)

    def build_cache(self):
        if self.cache: return
        qm = ''.join(type_map[t][0] for t in self.qt)
        qf = [type_map[t][1] for t in self.qt]
        rcount = self.s // struct.calcsize(qm)
        if self.s != struct.calcsize(qm) * rcount:
            self.data = b''
            self.cache = True
        else:
            self.cache = (struct.Struct('>' + qm * rcount), qf * rcount, rcount)

    def __len__(self):
        self.build_cache()
        return len(self.data) / self.s

    def __getitem__(self, idx):
        self.build_cache()
        p = idx * self.s
        self.data[p] # raise IndexError
        ret = [f(d) if f else d
               for f, d in zip(self.cache[1],
                               self.cache[0].unpack_from(self.data, p))]
        if len(self.cache[1]) != self.cache[2]: # Do we have structs?
            num = len(self.cache[1]) // self.cache[2]
            ret = [ret[i:i+num] for i in range(0, len(ret), num)]
        if self.cache[2] == 1: # If no repeat, then turn into scalar
            ret = ret[0]
        return ret

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
            data = list(KLV_data(qtype if t == '?' else t,
                            s, bytes(data)))
        ret.append((fc, data))
        p += 8 + s * r + ((-s * r) & 3)
    return ret

if __name__ == '__main__':
    import pprint
    import sys
    with open(sys.argv[1], 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            m.madvise(mmap.MADV_RANDOM)
            for i, p in enumerate(parse_mp4(memoryview(m))):
                #print('Payload %f-%f' % (p.start_time, p.end_time))
                KLV_parser(p.data)
            p = None
