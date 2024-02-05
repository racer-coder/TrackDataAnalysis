# input file to dmgbuild

exec(open('version.py').read())

filename = 'dist/TrackDataAnalysis-%s.dmg' % version
volume_name = 'TrackDataAnalysis'
files = [('dist/TrackDataAnalysis.app', 'TrackDataAnalysis.app')]
format = 'ULFO'
filesystem = 'APFS'
