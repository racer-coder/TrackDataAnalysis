
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).


# Using something like pint would be total overkill.  Instead, basic
# list of units and conversions.  Probably language dependent.

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Unit:
    name: str
    symbol: str
    scale: float = 1 # X base units = Y this_units.  Y = X * scale + offset
    offset: float = 0 #                              X = (Y - offset) / scale
    aliases: list[str] = field(default_factory=list)
    display: str = None # usually symbol unless special unicode characters


@dataclass
class UnitProperty:
    name: str
    units: list[Unit]

properties = [
    UnitProperty('Length & Distance',
                 [Unit('meter', 'm'),
                  Unit('millimeter', 'mm', 1000),
                  Unit('feet', 'ft', 1 / (.0254*12)),
                  Unit('mile', 'mile', 1 / (.0254*12*5280), aliases=['mi'])]),
    UnitProperty('Velocity',
                 [Unit('meter/sec', 'm/s'),
                  Unit('kilometer/hour', 'km/h', 3.6, aliases=['kph']),
                  Unit('mile/hour', 'mph', 100/2.54/12/5280*3600, aliases=['mile/h'])]),
    UnitProperty('Time',
                 [Unit('second', 's'),
                  Unit('millisecond', 'ms', 1000)]),
    UnitProperty('Angle',
                 [Unit('radian', 'rad'),
                  Unit('degree', 'deg', 180 / math.pi, display='\u00b0', aliases=['Degrees'])]),
    UnitProperty('Pressure',
                 [Unit('pascal', 'Pa'),
                  Unit('kilopascal', 'kPa', 1e-3),
                  Unit('bar', 'bar', 1e-5),
                  Unit('PSI', 'psi', 0.0001450377)]),
    UnitProperty('Temperature',
                 [Unit('kelvin', 'K'),
                  Unit('celsius', 'C', offset=-273.15, display='\u00b0C'),
                  Unit('fahrenheit', 'F', 1.8, -459.67)]),
    UnitProperty('Ratio',
                 [Unit('ratio', 'ratio'),
                  Unit('percent', '%', 100)]),
    UnitProperty('Voltage',
                 [Unit('volt', 'V'),
                  Unit('millivolt', 'mV', 1000)]),
    UnitProperty('Current',
                 [Unit('amp', 'A')]),
    UnitProperty('Volume',
                 [Unit('cubic meter', 'm3', display='m\u00b3'),
                  Unit('liter', 'l', 1000),
                  Unit('US gallon', 'USgal', 264.172052)]),
    UnitProperty('Acceleration',
                 [Unit('meter/sec squared', 'm/s/s', display='m/s\u00b2'),
                  Unit('G force', 'G', 0.10197162129779283)]),
    UnitProperty('Angular velocity',
                 [Unit('rad/sec', 'rad/s'),
                  Unit('rev/min', 'rpm', 30 / math.pi),
                  Unit('deg/sec', 'deg/s', 180 / math.pi, display='\u00b0/s')]),
    UnitProperty('Volume flow',
                 [Unit('cubic meter/sec', 'm3/s', display='m\u00b3/s'),
                  Unit('liter/sec', 'l/s', 1000)]),
]

unit_map = {name: (prop, unit)
            for prop in properties
            for unit in prop.units
            for name in [unit.name, unit.symbol] + unit.aliases}

def convert(values, from_unit, to_unit):
    try:
        old_unit = unit_map[from_unit]
        new_unit = unit_map[to_unit]
    except KeyError:
        return None
    if old_unit[0] != new_unit[0]: # Are they the same property
        return None
    if old_unit[1] == new_unit[1]:
        return values # don't waste memory on a name change
    return (np.subtract(values, old_unit[1].offset) * (new_unit[1].scale / old_unit[1].scale)
            + new_unit[1].offset)

def check_units(unit):
    return # disable check for now
    if unit and unit not in unit_map:
        print("Found unit", unit)
