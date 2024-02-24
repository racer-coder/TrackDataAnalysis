# Track Data Analysis

## Motivation

Every data acquisition tool comes with their own software, so why make another tool?

- High DPI Support.  Most tools don't enable High DPI support and thus
  are scaled and look horrible on a modern display.  You can override
  this in Windows, and Motec i2 actually mostly works in this mode but
  still has some edge cases that don't render properly.  AiM
  RaceStudio 2 and 3 work very badly if you override scaling.

- Responsive UI.  Motec i2 is mostly responsive while AiM RaceStudio 2
  and 3 are quite sluggish: when dragging the cursor, the values for
  the channels update very slowly.

- Cross platform.  There should be a native tool for Windows, Linux, and Mac OSX.

- Video support.  Motec supports video files but playback was very
  choppy for a 1080p/60 video on a new 2024 laptop.  AiM only supports
  video files from SmartyCam.

### Alternatives

| Feature | MoTeC i2 | AiM RaceStudio | GEMS Data Analysis | Track Attack | Race Capture |
| --- | --- | --- | --- | --- | --- |
| High DPI support | almost | no | YES | almost | YES |
| Responsive UI | YES | no | YES | YES | YES |
| Video support | Pro (but didn't work for me) | only smartycam | Pro version | Premium version | YES |
| Cross platform | no | no | no | Windows + Mac | YES |
| Many data formats | CSV (Pro) | no | no | YES | no |

- [Aim_2_MoTeC](https://github.com/ludovicb1239/Aim_2_MoTeC) will
  convert AiM DRK and XRK files to a MoTeC LD file that enables i2 Pro
  features.  If i2 works for you this may be a more robust solution.
  It only works on Windows, but then again, i2 only works on Windows.
- [RCP2GEMS](https://github.com/autosportlabs/RCP2GEMS/) will convert
  AutoSport Labs log files (a form of CSV) to another form of CSV that
  GEMS can read.

## Features

- Native high dpi support - fonts are sized by dpi/scaling.  Most
  lines are still drawn at a single pixel though which still looks
  good on a black background.

- Responsive UI - channel search, cursor drag, switching worksheets,
  resizing all respond very quickly, even when dealing with 2 hour
  logs at 100Hz on a 10+ year old computer (i5-2500 with integrated
  graphics running 1600x1200).

- Cross platform (Windows 8+, Ubuntu 22.04+, Mac Arm)

- Opens AiM XRK, MoTeC LD, Autosport Lab LOG, and iRacing IBT files natively.

- Time/distance graph with overlaying channels, drag and drop channel
  support (for reordering the view), zoom in/out, pan, multiple laps
  with individual lap offsets, and time slip.

- Channel values report with simple, concise search (type `o p` for oil pressure).

- GPS map with optional satellite background for line comparison.

- Video from almost anything can be included.  Playback is kept
  synchronized even in distance mode by speeding up/slowing down
  playback of the alternate lap.

- Configurable workspace layout, with cut/copy/paste.

- Math channels.

- Database that caches metadata from your sessions for easy lookup.

### Future (aka missing) features

- Track editor for defining sections
- Various channel graphs (bar, steering, histogram, etc)
- Various reports (channel min/max/average, sector comparison, etc)

## Installation (from binary)

Windows x86_64 and Mac Arm images can be found at
[Releases](https://github.com/racer-coder/TrackDataAnalysis/releases).

On Windows, it will install a start menu shortcut for "Track Data
Analysis" (just press the windows key and start typing Track).

## Installation (from source)

Track Data Analysis relies on the `PySide2` package in order to
support the widest list of operating systems.  However it places some
odd constraints on how you source certain packages.  Please follow the
instructions below as closely as reasonable.

### Linux / Ubuntu

Track Data Analysis is developed and tested on Ubuntu 22.04.  You can
use the built in packages for most things except `glfw`, which you'll
have to install via `pip3`.  Later versions of Ubuntu have switched to
`python3.11` as the default, which is fine as long as the PySide2
package is available.  If you're installing PySide2 via `pip3`, then
you'll want to stick to `python3.10` or older.

Ubuntu 22.04:
```
sudo apt install python3-pyside2.qtwidgets python3-numpy libmpv1
pip3 install dacite glfw pyyaml # Ubuntu 23.04+ has apt package python3-pyglfw
```

You should then be able to run the program using either of the following:

```
./gui.py
python3.10 gui.py
```

### Windows 8+

You'll want to get Python 3.10 directly from
[python.org](https://www.python.org/downloads/release/python-31011/).
You should probably get the 'Windows installer (64-bit)'.  Do not use
the version of python from the Microsoft app store; I haven't tried it
but I have read enough that makes me think it won't work right.

Assuming that Python is in your path, you can then install the
Python package requirements with:
```
pip install -r requirements.txt
```

You will need to manually install [libmpv](https://sourceforge.net/projects/mpv-player-windows/files/libmpv/);
I typically put `libmpv-2.dll` under the `ui/` directory as that is
added to the path automatically.

You should then be able to run the program using:

```
python3 gui.py
```

### Mac Arm (and probably Intel?)

Mac is a little tricky; PySide2 is not technically supported on Mac
Arm machines, but it is available on [Homebrew](https://brew.sh/).
However, they make it a little difficult, and it will only show up
under Python3.10, so you need to make sure to specify that when
running various commands.

```
brew install pyside@2
brew link pyside@2
brew install mpv
pip3.10 install dacite glfw numpy pyyaml
```

Then you should be able to run the program with either:
```
./gui.py
python3.10 ./gui.py
```

## Basic introduction

### UI

The window is made up of components (graphs, videos, etc) and docks
(anything activated by a button on the left side of the screen).  The
dock windows can be attached to the UI on either the left side or
right side of the component area.  Docks are shown on all
worksheets, while components are specific to a worksheet.

### Components

Components have a context menu (right click), along with a top level
menu which is basically the same thing.  Components can be removed by
'Cut' and duplicated using 'Copy' / 'Paste'.  Actions such as enabling
time slip or linking video happen through component menus.  New
components can be added using the 'Add' menu.

### Worksheets / Workbooks / Workspaces

Each worksheet has it's own collection of graphs.  You can easily
switch between worksheets within a workbook by selecting the
corresponding tab on top.  You can select different workbooks using
the drop down.  Use the Layout menu to change the worksheet / workbook
organization.  The collection of workbooks is stored in the current
workspace; use the File menu to open/save workspaces.  Generally I
would assume most users will only use a single workspace, or maybe one
per car.

### Data

Log files are opened via File menu, and laps are generally selected
through the Data dock.  There should always be a single reference lap
selected.  There can be an alternate lap selected (for components that
only support two comparisons, such as video, this will be the one
given preference).  There can be a number of extra laps selected
though the UI can get pretty busy with a lot of laps selected.  Use
the data menu adjust parameters relating to laps.  NOTE that
Time/Distance is a per-worksheet parameter!

### Time/Distance Graphs

Channels can be added using either the Channels dock or the Values
dock.  Double click on the channel to add and it will by default be
added to a subgraph that has the same units, otherwise a new subgraph
will be created.  To always add to a new subgraph, hold down Ctrl
while double clicking.  Channels can be rearranged on the graph using
drag and drop: dropping onto a channel will put the dragged channel
after it.  Dropping onto the bottom half of a subgraph will insert a
new subgraph for that channel.

### Video

You can load any video file, but currently only one video file can be
associated with a log file.  Typically to align a video I will open
the first two laps, put the cursor at the start of the lap, enable
'Set video alignment' (right click on video), then drag the bottom
slider until the video lines up (the top timecode is session time, the
bottom timecode is video time).  Both videos should look roughly like
they're in the same spot.

### Satellite Maps

You'll need an API key from [MapTiler](https://www.maptiler.com/) to
get satellite maps.  A free account should work just fine and you
don't need to give them a credit card.  Once your account is created
navigate to [API Keys](https://cloud.maptiler.com/account/keys/) to
copy your key, then go to File/Preferences to set it.  You can always
disable the satellite background by right clicking on the map.

## Changelog

- 0.4.0
  - Initial math expression support
  - 4x faster AiM XRK file processing.
  - Allow specifying 'last used' instead of 'interpolate' for channels (for gear, status, etc)
- 0.3.3
  - Significantly (50%) faster AiM XRK file processing.
  - Add handling for interpolated vs previous values in graphs
- 0.3.2
  - Context menu for worksheet tabs (new/rename/delete/duplicate/etc)
  - Parse some metadata from iRacing files for use in db
  - Fix bug in parsing some iRacing files
  - Allow * in channel search
- 0.3.1
  - iRacing IBT support
  - Automatic GPS lap insert for AiM files
  - Small bug fixes
- 0.3.0
  - Add automatic scanning of directories for quick access to old log files.
  - Add ability to diplay metadata from log files (venue/driver/etc)
- 0.2.0
  - Mac/Arm support
  - Channel editor to set units, decimals places, and color
  - Basic units support.  Fixes distance calculation when speed is not m/s.
  - Fix MoTeC beacon support
  - Fix Autosport Labs lap counter
