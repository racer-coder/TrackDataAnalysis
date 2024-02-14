
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from collections import namedtuple
import numpy as np

# None of the algorithms are slow
# fastest: Fukushima 2006, but worse accuracy
# most accurate: Vermeille, also very compact code
# runner up: Osen


# Also considered:

# https://ea4eoz.blogspot.com/2015/11/simple-wgs-84-ecef-conversion-functions.html
# slower algorithm, not great accuracy

# http://wiki.gis.com/wiki/index.php/Geodetic_system
# poor height accuracy

# Olson, D. K., Converting Earth-Centered, Earth-Fixed Coordinates to
# Geodetic Coordinates, IEEE Transactions on Aerospace and Electronic
# Systems, 32 (1996) 473-476.
# difficult to vectorize (poor for numpy/python)

GPS = namedtuple('GPS', ['lat', 'long', 'alt'])

# convert lat/long/zoom to web mercator.  lat/long are degrees
# returns x,y as floats - integer component is which tile to download
def llz2web(lat, long, zoom=0):
    # wikipedia web mercator projection
    mult = 0.25 * (2 << zoom)
    return (mult * (1 + long / 180),
            mult * (1 - np.log(np.tan(np.pi/4 + np.pi/360 * lat)) / np.pi))

# returns lat/long as floats in degrees
def web2ll(x, y, zoom=0):
    mult = 1 / (0.25 * (2 << zoom))
    return (np.arctan(np.exp(np.pi - np.multiply(np.pi * mult, y))) * 360 / np.pi - 90,
            np.multiply(180 * mult, x) - 180)

# lat, long = degrees
# x, y, z, alt = meters
def lla2ecef(lat, lon, alt):
    a = 6378137
    e = 8.181919084261345e-2
    e_sq = e*e

    lat = lat*(np.pi/180)
    lon = lon*(np.pi/180)

    clat = np.cos(lat)
    slat = np.sin(lat)

    N = a/np.sqrt(1 - e_sq*slat*slat)

    x = (N+alt)*clat*np.cos(lon)
    y = (N+alt)*clat*np.sin(lon)
    z = ((1 - e_sq)*N+alt)*slat

    return x, y, z


# Karl Osen. Accurate Conversion of Earth-Fixed Earth-Centered Coordinates to Geodetic Coordinates.
# [Research Report] Norwegian University of Science and Technology. 2017. ￿hal-01704943v2
# https://hal.science/hal-01704943v2/document
# pretty accurate, reasonably fast
def ecef2lla_osen(x, y, z):
    invaa = +2.45817225764733181057e-0014 # 1/(a^2)
    l = +3.34718999507065852867e-0003 # (e^2)/2
    p1mee = +9.93305620009858682943e-0001 # 1-(e^2)
    p1meedaa = +2.44171631847341700642e-0014 # (1-(e^2))/(a^2)
    ll4 = +4.48147234524044602618e-0005 # 4*(l^2) = e^4
    ll = +1.12036808631011150655e-0005 # l^2 = (e^4)/4
    invcbrt2 = +7.93700525984099737380e-0001 # 1/(2^(1/3))
    inv3 = +3.33333333333333333333e-0001 # 1/3
    inv6 = +1.66666666666666666667e-0001 # 1/6

    w = x * x + y * y
    m = w * invaa
    w = np.sqrt(w)
    n = z * z * p1meedaa
    mpn = m + n
    p = inv6 * (mpn - ll4)
    P = p * p
    G = m * n * ll
    H = 2 * P * p + G
    p = None
    #if H < Hmin: return -1
    C = np.cbrt(H + G + 2 * np.sqrt(H * G)) * invcbrt2
    G = None
    H = None
    i = -ll - 0.5 * mpn
    beta = inv3 * i - C - P / C
    C = None
    P = None
    k = ll * (ll - mpn)
    mpn = None
    # Compute t
    t = (np.sqrt(np.sqrt(beta * beta - k) - 0.5 * (beta + i)) +
         np.sqrt(np.abs(0.5 * (beta - i))) * (2 * (m < n) - 1))
    beta = None
    # Use Newton-Raphson's method to compute t correction
    g = 2 * l * (m - n)
    m = None
    n = None
    tt = t * t
    dt = -(tt * (tt + (i + i)) + g * t + k) / (4 * t * (tt + i) + g)
    g = None
    i = None
    tt = None
    # compute latitude (range -PI/2..PI/2)
    u = t + dt + l
    v = t + dt - l
    dt = None
    zu = z * u
    wv = w * v
    # compute altitude
    invuv = 1 / (u * v)
    return GPS(np.arctan2(zu, wv) * (180/np.pi),
               np.arctan2(y, x) * (180/np.pi),
               np.sqrt(np.square(w - wv * invuv) +
                       np.square(z - zu * p1mee * invuv)) * (1 - 2 * (u < 1)))


# https://www.researchgate.net/publication/227215135_Transformation_from_Cartesian_to_Geodetic_Coordinates_Accelerated_by_Halley's_Method/link/0912f50af90e6de252000000/download
# "Fukushima 2006"
# fastest, reasonably accurate but not best
def ecef2lla_fukushima2006(x, y, z):
    a = 6378137.
    finv = 298.257222101
    f = 1./finv
    e2 = (2-f) * f
    ec2 = 1 - e2
    ec = np.sqrt(ec2)
    #b = a * ec
    c = a * e2
    #PIH = 2 * np.arctan(1.)

    lamb = np.arctan2(y, x)
    s0 = np.abs(z)
    p2 = x*x + y*y
    p = np.sqrt(p2)
    zc = ec * s0
    c0 = ec * p
    c02 = c0 * c0
    s02 = s0 * s0
    a02 = c02 + s02
    a0 = np.sqrt(a02)
    #a03 = a02 * a0
    a03 = a02
    a03 *= a0
    a02 = None
    #s1 = zc * a03 + c * (s02 * s0)
    s02 *= s0
    s02 *= c
    s1 = s02
    s1 += zc * a03
    s02 = None
    #c1 = p * a03 - c * (c02 * c0)
    c02 *= c0
    c02 *= c
    c1 = p * a03 - c02
    c02 = None
    cs0c0 = c * c0 * s0
    #b0 = 1.5 * cs0c0 * ((p*s0 - zc*c0) * a0 - cs0c0)
    zc *= c0
    b0 = cs0c0
    b0 *= 1.5 * ((p*s0 - zc) * a0 - cs0c0)
    a0 = None
    zc = None
    cs0c0 = None
    s1 = s1 * a03 - b0 * s0
    #cc = ec * (c1 * a03 - b0 * c0)
    c1 *= a03
    b0 *= c0
    c1 -= b0
    cc = c1
    cc *= ec
    c1 = None
    a03 = None
    c0 = None
    b0 = None
    phi = np.arctan2(s1, cc)
    s12 = s1 * s1
    cc2 = cc * cc
    h = (p * cc + s0*s1 - a*np.sqrt(ec2*s12 + cc2)) / np.sqrt(s12 + cc2)
    s1 = None
    cc = None
    s12 = None
    cc2 = None
    phi = np.copysign(phi, z)

    return GPS(phi * (180/np.pi), lamb * (180/np.pi), h)

# Computing geodetic coordinates from geocentric coordinates
# H. Vermeille, 2003/2004
# http://users.auth.gr/kvek/78_Vermeille.pdf
def ecef2lla_vermeille2003(x, y, z):
    a = 6378137.
    e = 8.181919084261345e-2

    p = (x*x + y*y) * (1 / (a*a))
    q = ((1-e*e)/(a*a)) * z*z
    r = (p+q-e**4) * (1 / 6)
    s = (e**4/4) * p * q / (r**3)
    p = None
    t = np.cbrt(1 + s + np.sqrt(s * (2+s)))
    s = None
    u = r * (1 + t + 1/t)
    r = None
    t = None
    v = np.sqrt(u*u + e**4 * q)
    u += v # precalc
    w = (e**2/2) * (u-q) / v
    q = None
    k = np.sqrt(u+w*w) - w
    D = k * np.sqrt(x*x + y*y) / (k + e**2)
    rtDDzz = np.sqrt(D*D + z*z)
    return GPS((180/np.pi) * 2 * np.arctan2(z, D + rtDDzz),
               (180/np.pi) * np.arctan2(y, x),
               (k + e**2 - 1) / k * rtDDzz)


ecef2lla = ecef2lla_vermeille2003

if __name__ == '__main__':
    def perf_test():
        import time
        samples = 10000000
        lat  = -90. + 180.*np.random.rand(samples, 1)
        long = -180. + 360.*np.random.rand(samples, 1)
        alt  = -11e3 + (20e3)*np.random.rand(samples, 1) # From approximately the bottom of the Mariana trench, to the top of the Everest

        print("generating x,y,z")
        x, y, z = lla2ecef(lat, long, alt)
        algos = [('osen', ecef2lla_osen),
                 ('fukushima2006', ecef2lla_fukushima2006),
                 ('vermeille2003', ecef2lla_vermeille2003)]
        stats = {name:[] for name, algo in algos}
        for _ in range(5):
            for name, algo in algos:
                start = time.time()
                ilat, ilong, ialt = algo(x, y, z)
                duration = time.time() - start
                stats[name].append(duration)
                print("algorithm %s took %.3f" % (name, duration))
                print('  avg',
                      np.sqrt(np.sum((ilat-lat) ** 2)) / len(ilat),
                      np.sqrt(np.sum((ilong-long) ** 2)) / len(ilong),
                      np.sqrt(np.sum((ialt-alt) ** 2)) / len(ialt))
                print('  max',
                      np.max(np.abs(ilat-lat)),
                      np.max(np.abs(ilong-long)),
                      np.max(np.abs(ialt-alt)))
        for name, stat in stats.items():
            print(name, ', '.join(['%.3f' % s for s in stat]))
    perf_test()
