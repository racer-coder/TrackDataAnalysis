#!/usr/bin/python3

from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "excludes": ["tkinter", "unittest"],
    'include_files': ['libmpv-2.dll'],
}

setup(
    name = 'Track Data Analysis',
    version = '0.1.0',
    description = 'Graphical tool to view data from race cars',
    options = {'build_exe': build_exe_options,
               'bdist_msi': {
                   'initial_target_dir': '[ProgramFilesFolder]\\TrackDataAnalysis',
                   'upgrade_code': '{7B799120-C547-4E18-BE6B-3244F39DE767}',
               },
               },
    executables = [Executable('gui.py',
                              base='Win32GUI',
                              shortcut_name='Track Data Analysis',
                              shortcut_dir='StartMenuFolder',
                              )],
)
