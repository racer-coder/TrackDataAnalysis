
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from array import array
from dataclasses import dataclass
import sys
import typing

# We use array and memoryview for efficient operations, but that
# assumes the sizes we expect match the file format.  Lets assert a
# few of those assumptions here.  Our use of struct is safe since it
# has tighter control over byte order and sizing.
assert array('H').itemsize == 2
assert array('I').itemsize == 4
assert array('Q').itemsize == 8
assert array('f').itemsize == 4
assert array('d').itemsize == 8
assert sys.byteorder == 'little'

@dataclass(eq=False)
class Channel:
    timecodes: array
    values: array
    dec_pts: int
    name: str
    units: str

@dataclass(eq=False)
class Lap:
    num: int
    start_time: int
    end_time: int

@dataclass(eq=False)
class LogFile:
    channels: typing.Dict[str, Channel]
    laps: typing.List[Lap]
    metadata: typing.Dict[str, str]
    key_channel_map: typing.List[typing.Optional[str]] # speed, lat, long, alt
    file_name: str # move to metadata?
