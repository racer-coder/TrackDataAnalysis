#!/usr/bin/python3

from data import aim, aim_xrk, distance
import pprint
import sys

fname = 'e46n20.xrk' if len(sys.argv) < 2 else sys.argv[1]

pp = pprint.PrettyPrinter()

def test_aim():
    res = aim.AIM(fname)

    print(res.get_vehicle_name())
    print(res.get_track_name())
    print(res.get_racer_name())
    print(res.get_championship_name())
    print(res.get_venue_type_name())
    print(res.get_laps_count())
    print([res.get_lap_info(lap) for lap in range(res.get_laps_count())])
    print(res.get_channels_count())
    for ch in range(res.get_channels_count()):
        print(res.get_channel_name(ch), res.get_channel_units(ch))
    #    samp = res.get_channel_samples(ch)
    #    for t, v in zip(*samp):
    #        print('%8.3f' % t, v)
    #    t = samp[0]
    #    #print(t[0], t[-1], (t[-1] - t[0]) / (len(t) - 1), (len(t) - 1) / (t[-1] - t[0]))
    print(res.get_GPS_channels_count())
    for ch in range(res.get_GPS_channels_count()):
        samp = res.get_GPS_channel_samples(ch)
        print(res.get_GPS_channel_name(ch), res.get_GPS_channel_units(ch), len(samp[0]))
        for t, v in list(zip(*samp))[:]:
            print('%8.3f' % t, v)
    print(res.get_GPS_raw_channels_count())
    for ch in range(res.get_GPS_raw_channels_count()):
        print(res.get_GPS_raw_channel_name(ch), res.get_GPS_raw_channel_units(ch))
        samp = res.get_GPS_raw_channel_samples(ch)
        for t, v in list(zip(*samp))[:]:
            print('%8.3f' % t, v)
        t = samp[0]
        #print(t[0], t[-1], (t[-1] - t[0]) / (len(t) - 1), (len(t) - 1) / (t[-1] - t[0]))

def test_xrk():
    res = aim_xrk.AIMXRK(fname, lambda x,y: None) # dummy progress func to test parallelism

def test_xrk_and_ch():
    res = distance.DistanceWrapper(aim_xrk.AIMXRK(fname))
    for ch in res.get_channels():
        res.get_channel_data(ch)

def ch_help():
    r2 = aim_xrk.AIMXRK(fname)

    chmap = {}

    for i in ('Best Time',
              'Reset Odometer 1',
              'Reset Odometer 2',
              'Reset Odometer 3',
              'Reset Odometer 4',
              'Roll Time',
              'StrtRec',
              'Total Odometer',
              'Master Clk',
              'Predictive Time',
              'Prev Lap Diff',
              'Ref Lap Diff',
              'Best Today Diff',
              'Lap Time',
              'Best Run Diff',
            ):
        chmap[i] = 'HIDE'

    #for ch in r2.data.channels.values():
    #    chmap[ch.long_name] = (ch.long_name in aim_xrk._manual_decoders, ch.size)

    #r1 = aim.AIM(fname)
    #for ch in range(r1.get_channels_count()):
    #    chmap[r1.get_channel_name(ch)] = r1.get_channel_units(ch)

    #chmap['External Voltage'] = 'V'
    #chmap['VBattery'] = 'mV'

    r2._help_decode_channels(chmap)

def ch_compare():
    print("Loading via AiM library")
    r1 = aim.AIM(fname)
    print("Loading via python library")
    r2 = aim_xrk.AIMXRK(fname)

    errors = 0

    for ch in range(r1.get_channels_count()):
        name = r1.get_channel_name(ch)
        print(name, r1.get_channel_units(ch))
        print(r2.get_channel_units(name))
        if r1.get_channel_units(ch) != r2.get_channel_units(name):
            print("................. type mismatch!")
        d2 = r2.get_channel_data(name)
        if d2 is None:
            print("Skipping", name)
            continue
        print(name, end=' ')
        d1 = r1.get_channel_samples(ch)
        if len(d1[1]) == 0 and len(d2[1]) == 0:
            print("No data, skipping")
            continue
        toffs = d2[0][0] - int(d1[0][0] * 1000 + 0.5)
        if len(d1[1]) == len(d2[1]):
            diff = sum(1 if abs(a - b) > 0.00001 else 0
                       for a, b in zip(d1[1], d2[1], strict=True))
            tdiff = sum(int(abs(a * 1000 + toffs - b))
                        for a, b in zip(d1[0], d2[0], strict=True))
            print(diff, tdiff, toffs)
            if diff == 0 and tdiff == 0:
                continue
        else:
            print('Length mismatch, %d vs %d' % (len(d1[1]), len(d2[1])))
            print('d1: %f .. %f' % (d1[0][0], d1[0][-1]))
            print('d2: %d .. %d' % (d2[0][0], d2[0][-1]))
        errors += 1
        print_hex = True
        time_diff = True
        if print_hex:
            if all(int(x) == x for x in d2[1]):
                prev = None
                print("Hex difference:")
                for x in sorted((a, "%x" % int(b))
                                for a, b in zip(d1[1], d2[1])):
                    if x != prev:
                        pp.pprint(x)
                        prev = x
        if time_diff:
            tdiff = [('%.3f' % a, int(a * 1000 + toffs), b) for a, b in zip(d1[0], d2[0])]
            print("Time difference:")
            for i in range(len(tdiff)):
                if tdiff[i][1] != tdiff[i][2]:
                    pp.pprint(tdiff[i-5:i+10])
                    break
        alldiff = [(a,b,c,d) for a,b,c,d in zip(d1[0], d1[1], d2[0], d2[1]) if b!=d]
        print("First ten content differences")
        pp.pprint(alldiff[:10])
        print("Full comparison")
        pp.pprint(list(zip(d1[0], d1[1], d2[0], d2[1]))[:10])
        print('...')
        pp.pprint(list(zip(d1[0], d1[1], d2[0], d2[1]))[-10:])
    print('Errors:', errors)
    sys.exit(errors)

def gps_ch_compare():
    print("Loading via AiM library")
    r1 = aim.AIM(fname)
    print("Loading via python library")
    r2 = aim_xrk.AIMXRK(fname)

    ch = 'GPS Speed'
    for i in range(r1.get_GPS_channels_count()):
        if r1.get_GPS_channel_name(i) == ch:
            break
    d1 = r1.get_GPS_channel_samples(i)
    d2 = r2.get_channel_data(ch)
    for t1, s1, t2, s2 in zip(d1[0], d1[1], d2[0], d2[1]):
        if t1 != t2 or s1 != s2:
            print(t1, s1, t2, s2, (s1-s2))


#test_aim()
test_xrk()
#test_xrk_and_ch()
#ch_help()
#ch_compare()
#gps_ch_compare()
