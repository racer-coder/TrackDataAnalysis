# input file to dmgbuild

import platform

exec(open('version.py').read())

filename = 'dist/TrackDataAnalysis-%s-%s.dmg' % (version, platform.machine())
volume_name = 'TrackDataAnalysis'
files = [('dist/TrackDataAnalysis.app', 'TrackDataAnalysis.app')]
format = 'ULFO'
filesystem = 'APFS'
