#!/usr/bin/python3

from cx_Freeze import setup, Executable
import sys
from version import version
from Cython.Build import cythonize


# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "excludes": ["tkinter", "unittest"],
    'zip_include_packages': ['*'],
    'zip_exclude_packages': ['PySide2',
                             'glfw',
                             'numpy',
                             'numpy.libs',
                             'shiboken2',
                             'ui',
                             'yaml'],
}

if sys.platform == 'win32':
    build_exe_options['include_files'] = [('ui/libmpv-2.dll', 'lib/ui/libmpv-2.dll')]

# Currently, Mac doesn't work, but leaving this here just in case it does later.
if sys.platform == 'darwin':
    build_exe_options['include_files'] = [('/opt/homebrew/lib/libmpv.dylib', 'lib/ui/libmpv.dylib')]



setup(
    name = 'Track Data Analysis',
    version = version,
    description = 'Graphical tool to view data from race cars',
    options = {'build_exe': build_exe_options,
               'bdist_msi': {
                   'initial_target_dir': '[ProgramFilesFolder]\\TrackDataAnalysis',
                   'upgrade_code': '{7B799120-C547-4E18-BE6B-3244F39DE767}',
               },
               },
    executables = [Executable('gui.py',
                              base='Win32GUI' if sys.platform == 'win32' else None,
                              shortcut_name='Track Data Analysis',
                              shortcut_dir='StartMenuFolder',
                              )],
    ext_modules=cythonize('data/aim_xrk.pyx', language='c++'),
    include_package_data=False, # don't include .cpp file from cython
)
