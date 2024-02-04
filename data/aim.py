
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import ctypes
import os

_libxml2 = ctypes.CDLL('libxml2.so.2', mode=ctypes.RTLD_GLOBAL)
_dll = ctypes.CDLL(os.path.join(os.path.dirname(__file__),
                                'libxdrk-x86_64.so'))

# Fixup return types
_dll.get_library_date.restype = ctypes.c_char_p
_dll.get_library_time.restype = ctypes.c_char_p
_dll.get_vehicle_name.restype = ctypes.c_char_p
_dll.get_track_name.restype = ctypes.c_char_p
_dll.get_racer_name.restype = ctypes.c_char_p
_dll.get_championship_name.restype = ctypes.c_char_p
_dll.get_venue_type_name.restype = ctypes.c_char_p
_dll.get_channel_name.restype = ctypes.c_char_p
_dll.get_channel_units.restype = ctypes.c_char_p
_dll.get_GPS_channel_name.restype = ctypes.c_char_p
_dll.get_GPS_channel_units.restype = ctypes.c_char_p
_dll.get_GPS_raw_channel_name.restype = ctypes.c_char_p
_dll.get_GPS_raw_channel_units.restype = ctypes.c_char_p

# Global functions

get_library_date = _dll.get_library_date
get_library_time = _dll.get_library_time

class AIM:
    def __init__(self, fname):
        self._idx = None
        idx = _dll.open_file(str(fname).encode('ascii'))
        assert idx > 0
        self._idx = idx

    def __del__(self):
        self.close()

    def close(self):
        if self._idx is not None:
            _dll.close_file_i(self._idx)
        self._idx = None

    def get_vehicle_name(self):
        assert self._idx > 0
        return _dll.get_vehicle_name(self._idx)

    def get_track_name(self):
        assert self._idx > 0
        return _dll.get_track_name(self._idx)

    def get_racer_name(self):
        assert self._idx > 0
        return _dll.get_racer_name(self._idx)

    def get_championship_name(self):
        assert self._idx > 0
        return _dll.get_championship_name(self._idx)

    def get_venue_type_name(self):
        assert self._idx > 0
        return _dll.get_venue_type_name(self._idx)

    def get_laps_count(self):
        assert self._idx > 0
        return _dll.get_laps_count(self._idx)

    def get_lap_info(self, lap):
        assert self._idx > 0
        pstart = ctypes.c_double()
        pduration = ctypes.c_double()
        res = _dll.get_lap_info(self._idx, ctypes.c_int(lap),
                                ctypes.byref(pstart), ctypes.byref(pduration))
        assert res > 0
        return (pstart.value, pduration.value)

    def get_channels_count(self):
        assert self._idx > 0
        return _dll.get_channels_count(self._idx)

    def get_channel_name(self, ch):
        assert self._idx > 0
        return _dll.get_channel_name(self._idx, ctypes.c_int(ch)).decode('ascii')

    def get_channel_units(self, ch):
        assert self._idx > 0
        return _dll.get_channel_units(self._idx, ctypes.c_int(ch)).decode('ascii')

    def get_channel_samples_count(self, ch):
        assert self._idx > 0
        return _dll.get_channel_samples_count(self._idx, ctypes.c_int(ch))

    def get_channel_samples(self, ch):
        cnt = self.get_channel_samples_count(ch)
        assert cnt >= 0
        atype = ctypes.c_double * cnt
        ptimes = atype()
        pvalues = atype()
        if cnt != 0:
            res = _dll.get_channel_samples(self._idx, ctypes.c_int(ch),
                                           ctypes.byref(ptimes), ctypes.byref(pvalues), cnt)
        assert res > 0, "%d" % res
        return (ptimes, pvalues)

    def get_GPS_channels_count(self):
        assert self._idx > 0
        return _dll.get_GPS_channels_count(self._idx)

    def get_GPS_channel_name(self, ch):
        assert self._idx > 0
        return _dll.get_GPS_channel_name(self._idx, ctypes.c_int(ch)).decode('ascii')

    def get_GPS_channel_units(self, ch):
        assert self._idx > 0
        return _dll.get_GPS_channel_units(self._idx, ctypes.c_int(ch)).decode('ascii')

    def get_GPS_channel_samples_count(self, ch):
        assert self._idx > 0
        return _dll.get_GPS_channel_samples_count(self._idx, ctypes.c_int(ch))

    def get_GPS_channel_samples(self, ch):
        cnt = self.get_GPS_channel_samples_count(ch)
        assert cnt >= 0
        atype = ctypes.c_double * cnt
        ptimes = atype()
        pvalues = atype()
        if cnt != 0:
            res = _dll.get_GPS_channel_samples(self._idx, ctypes.c_int(ch),
                                               ctypes.byref(ptimes), ctypes.byref(pvalues), cnt)
        assert res > 0, "%d" % res
        return (ptimes, pvalues)

    def get_GPS_raw_channels_count(self):
        assert self._idx > 0
        return _dll.get_GPS_raw_channels_count(self._idx)

    def get_GPS_raw_channel_name(self, ch):
        assert self._idx > 0
        return _dll.get_GPS_raw_channel_name(self._idx, ctypes.c_int(ch)).decode('ascii')

    def get_GPS_raw_channel_units(self, ch):
        assert self._idx > 0
        return _dll.get_GPS_raw_channel_units(self._idx, ctypes.c_int(ch)).decode('ascii')

    def get_GPS_raw_channel_samples_count(self, ch):
        assert self._idx > 0
        return _dll.get_GPS_raw_channel_samples_count(self._idx, ctypes.c_int(ch))

    def get_GPS_raw_channel_samples(self, ch):
        cnt = self.get_GPS_raw_channel_samples_count(ch)
        assert cnt >= 0
        atype = ctypes.c_double * cnt
        ptimes = atype()
        pvalues = atype()
        if cnt != 0:
            res = _dll.get_GPS_raw_channel_samples(self._idx, ctypes.c_int(ch),
                                                   ctypes.byref(ptimes), ctypes.byref(pvalues),
                                                   cnt)
        assert res > 0, "%d" % res
        return (ptimes, pvalues)
