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

- Cross platform (though only tested on Windows 10+ and Ubuntu 22.04 for now)

- Opens AiM XRK, MoTeC LD, and Autosport Lab LOG files natively.

- Time/distance graph with overlaying channels, drag and drop channel
  support (for reordering the view), zoom in/out, pan, multiple laps
  with individual lap offsets, and time slip.

- Channel values report with simple, concise search (type `o p` for oil pressure).

- GPS map with optional satellite background for line comparison.

- Video from almost anything can be included.  Playback is kept
  synchronized even in distance mode by speeding up/slowing down
  playback of the alternate lap.

- Configurable workspace layout, with cut/copy/paste.

### Future (aka missing) features

- Cached database of your files so it's easy to find them (other than walking directories).
- Channel editor for setting preferred units, decimal places, colors, etc.
- Track editor for defining sections
- Various channel graphs (bar, steering, histogram, etc)
- Various reports (channel min/max/average, sector comparison, etc)
- Math channels

## Installation (from binary)

Right now only Windows x86_64 images are being built.  Head on over to
[Releases](https://github.com/racer-coder/TrackDataAnalysis/releases)
and install the most recent.  Once installed, there will be a start
menu shortcut for "Track Data Analysis" (just press the windows key
and start typing Track).

## Installation (from source)

1. Make sure you have a proper version of Python.  PySide2 generally
only has support for up to Python 3.10, though there are patches
around to make it work with Python 3.11.  For simplicity, Python 3.10
should probably be your default choice for Windows 8+, Linux, and Mac.
Prefer to download it directly from
[python.org](https://www.python.org/downloads/) rather than the
Windows store.

2. Install python requirements.
```
pip install -r requirements.txt
```
NOTE: On MacOSX, the PyPi version of PySide2 does not support Arm CPU.  Instead, use home brew.  Something like (NOT TESTED):
```
brew install pyside@2
brew link pyside@2
```

On Ubuntu (tested with 22.04), you can install the default system packages instead:
```
apt install python3-pyside2.qtwidgets python3-numpy python3-opengl
```

3. Install video player library.

This program uses [libmpv](https://mpv.io/installation/) to handle
video.  Download the right binary and make the .so/.dll accessible
(drop it in the source directory?).

Alternatively, on Ubuntu 22.04+, you can install the defaulte system package instead:
```
apt install libmpv1
```

4. Run!
```
python gui.py
```
or maybe:
```
python3 gui.py
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
