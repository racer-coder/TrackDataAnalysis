
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

import time

import numpy as np

from data import gps
from . import state

def _get_type(t):
    if t < -150: return 'Left'
    if t > 150: return 'Right'
    return 'Straight'

def select_track(data_view):
    t1 = time.perf_counter()
    data_view.track = None

    if len(data_view.log_files) != 1:
        return

    logref = data_view.log_files[0]
    lap = logref.best_lap

    # grab GPS coordinates, select roughly every 10 meters
    key_channels = logref.log.get_key_channel_map()
    gps_speed = logref.log.get_channel_data(key_channels[0], unit='m/s')
    gps_lat = logref.log.get_channel_data(key_channels[1], unit='deg')
    gps_long = logref.log.get_channel_data(key_channels[2], unit='deg')
    gps_alt = logref.log.get_channel_data(key_channels[3], unit='m')

    # NOTE: GPS coordintes from synchronized time loggers (like MoTeC)
    # which log every X Hz regardless of when data shows up may have
    # repeats or skipped GPS entries.  Be wary relying on GPS data on
    # those loggers without heavy filtering!

    if not len(gps_lat.values) or not len(gps_long.values):
        return

    marker_sample = 1 # meters
    map_sample = 10 # units of marker_sample
    dist = np.linspace(lap.start.dist, lap.end.dist,
                       num=round((lap.end.dist - lap.start.dist) / marker_sample) + 1)
    speed = gps_speed.interp_many(dist, mode_time=False)
    lat = gps_lat.interp_many(dist, mode_time=False)
    lon = gps_long.interp_many(dist, mode_time=False)
    if len(gps_alt.values):
        alt = gps_alt.interp_many(dist, mode_time=False)
    else:
        alt = np.zeros((len(dist),))
    dist -= dist[0]

    # fudge the data so we start and finish at the exact same point (if they're close)
    if np.linalg.norm(np.array(gps.lla2ecef(lat[0], lon[0], alt[0])) -
                      np.array(gps.lla2ecef(lat[-1], lon[-1], alt[-1]))) < 20:
        lat += (lat[0] - lat[-1]) * np.linspace(0, 1, num=len(lat))
        lon += (lon[0] - lon[-1]) * np.linspace(0, 1, num=len(lon))
        alt += (alt[0] - alt[-1]) * np.linspace(0, 1, num=len(alt))

    # build track
    try:
        name = logref.log.get_metadata()['Venue']
    except KeyError:
        name = 'Unknown'
    latrad = lat * (np.pi / 180)
    lonrad = lon * (np.pi /180)
    hsample = 4 # units of marker_sample.  Have > 1 to work around MoTeC GPS sampling issues
    orig_index = hsample // 2
    heading = np.arctan2(np.cos(latrad[hsample:]) * np.sin(lonrad[hsample:] - lonrad[:-hsample]),
                         np.cos(latrad[:-hsample]) * np.sin(latrad[hsample:])
                         - np.sin(latrad[:-hsample]) * np.cos(latrad[hsample:])
                         * np.cos(lonrad[hsample:] - lonrad[:-hsample])) * (180 / np.pi)

    tsample = 10 # units of marker_sample
    orig_index += tsample // 2
    turn = heading[tsample:] - heading[:-tsample]
    turn = (turn + 540) % 360 - 180 # normalize to -180 to 180
    turn *= speed[orig_index:-orig_index]

    # sliding window calculation
    window = 4
    orig_index -= 1
    turn = np.concatenate([[0.], np.cumsum(turn)])
    orig_index += window // 2
    turn = (turn[window:] - turn[:-window]) * (1./window)

    t2 = time.perf_counter()

    type_map = ['Straight', 'Right', 'Left']
    typ = (turn > 150) + 2 * (turn < -150)

    # find points where typ changes
    typidx = list(memoryview(np.concatenate([[0],
                                             np.nonzero(typ[1:] != typ[:-1])[0] + 1,
                                             [len(typ)]])))
    # check for short straights and join correctly (same turn->same turn - extend, different turn - find 0 crossing)
    min_straight = 25 # meters
    for i in range(len(typidx), 1, -1):
        # need the out of bounds check here because this loop can
        # delete more than 1 element, causing the out of bounds check
        # to fail in later iterations
        if i + 2 >= len(typidx):
            continue

        # look at the straight from [i] to [i+1], if short, merge into turn starting at [i-1] and [i+1]
        if typ[typidx[i]] == 0 and typidx[i+1] - typidx[i] < min_straight:
            if typ[typidx[i-1]] == typ[typidx[i+1]]:
                # same direction turn, just have one big turn
                typidx.pop(i)
                typidx.pop(i)
            else:
                # different direction turns, find the best zero crossing
                subset = turn[typidx[i]:typidx[i+1]]
                zero_cross = np.nonzero((subset[1:] > 0) != (subset[:-1] > 0))[0]
                if len(zero_cross):
                    newidx = int(zero_cross[0]) + typidx[i] + 1
                    typ[newidx - 1] = typ[typidx[i] - 1]
                    typ[newidx] = typ[typidx[i+1]]
                    typidx[i+1] = newidx
                    typidx.pop(i)

    # check for short turns and drop?
    min_turn = 10 # meters
    for i in range(len(typidx), 1, -1):
        if i + 2 < len(typidx) and typ[typidx[i]] != 0 and typidx[i+1] - typidx[i] < min_turn:
            typidx.pop(i)
            typidx.pop(i)

    sectors = [state.Marker('Hi', lat[i + orig_index], lon[i + orig_index], type_map[typ[i - 1]])
               for i in typidx[1:-1]]

    num = 0
    for s in sectors:
        if s.typ != 'Straight':
            num += 1
            s.name = 'Turn %d' % num
    last = num
    num = 1
    last_straight = None
    for s in sectors:
        if s.typ == 'Straight':
            s.name = 'Str %d-%d' % (last, num)
            last_straight = (s, last)
        else:
            last = num
            num += 1
    if last_straight:
        last_straight[0].name = 'Str %d-1' % last_straight[1]

    t4 = time.perf_counter()

    data_view.track = state.Track(
        name = name,
        file_name = None,
        coords = [(float(la), float(lo), float(al), float(di))
                  for la, lo, al, di in zip(*map(lambda x: x[::map_sample],
                                                 (lat, lon, alt, dist)))],
        sector_sets = {'Default': state.Sectors('Default', sectors)})


    t5 = time.perf_counter()
    print('select track: %.3f %.3f %.3f' % (t2-t1, t4-t2, t5-t4))
