
# Copyright 2024, Scott Smith.  MIT License (see LICENSE).

from PySide2.QtCore import QSize, Qt
from PySide2.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide2.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QSpinBox,
)

from . import state
from data import unitconv

colors = (
    QColor(255, 0, 0),
    QColor(255, 160, 32),
    QColor(64, 255, 64),
    QColor(43, 255, 255),
    QColor(47, 151, 255),
    QColor(186, 117, 255),
    QColor(255, 106, 218),
    QColor(244, 244, 0),
    )

def update_channel_properties(data_view):
    # gather channels from all the logs
    props = {}
    cidx = 0
    for log in data_view.log_files:
        for ch in log.log.get_channels():
            if ch not in props:
                if ch in data_view.channel_defaults:
                    prop = data_view.channel_defaults[ch]
                else:
                    prop = state.ChannelProperties(None, None, cidx & 7) # XXX color?
                metadata = log.log.get_channel_metadata(ch)
                prop.units = metadata['units']
                prop.dec_pts = metadata['dec_pts']
                props[ch] = prop
                cidx += 1

    data_view.channel_defaults = {n: p for n, p in props.items()} # copy before we modify it

    # apply overrides
    for ch, override in data_view.channel_overrides.items():
        if ch in props:
            p = props[ch]
            p = state.ChannelProperties(p.units, p.dec_pts, p.color) # copy to leave defaults intact
            p.units = override.get('units', p.units)
            p.dec_pts = override.get('dec_pts', p.dec_pts)
            p.color = override.get('color', p.color)
            props[ch] = p

    # set this at the end in case we fail along the way
    data_view.channel_properties = props

def channel_editor(_parent, data_view, channel):
    try:
        defaults = data_view.channel_defaults[channel]
    except KeyError:
        return # nothing to do, channel doesn't exist right now

    overrides = data_view.channel_overrides.get(channel, {})

    layout = QGridLayout()

    def adder(name, widget):
        row = layout.rowCount()
        layout.addWidget(QLabel(name), row, 0, Qt.AlignRight)
        layout.addWidget(widget, row, 2)

    unit_combo = QComboBox()
    items = [('default: %s' % defaults.units, None)] + [
        ('%s [%s]' % (name, display), name)
        for name, display in unitconv.comparable_units(defaults.units)]
    for text, data in items:
        unit_combo.addItem(text, data)
    unit_combo.setCurrentIndex([data for _, data in items].index(overrides.get('units', None)))
    adder('Units', unit_combo)

    dplace_spin = QSpinBox()
    dplace_spin.setSpecialValueText('default: %d' % defaults.dec_pts)
    dplace_spin.setMinimum(-1)
    dplace_spin.setMaximum(10)
    dplace_spin.setValue(overrides.get('dec_pts', -1))
    adder('Decimal places', dplace_spin)

    color_combo = QComboBox()
    color_combo.setIconSize(QSize(60, 6))
    pix = QPixmap(60, 6)
    painter = QPainter(pix)
    painter.fillRect(pix.rect(), colors[defaults.color])
    del painter
    ico = QIcon(pix)
    color_combo.addItem(ico, '<default>', None)
    for idx, color in enumerate(colors):
        pix = QPixmap(60, 6)
        painter = QPainter(pix)
        painter.fillRect(pix.rect(), color)
        del painter
        ico = QIcon(pix)
        color_combo.addItem(ico, '', idx)
    color_combo.setCurrentIndex(overrides.get('color', -1) + 1)
    adder('Color', color_combo)

    bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(bbox, layout.rowCount(), 0, 1, 3)

    dia = QDialog()
    dia.setWindowTitle('Channel editor for %s' % channel)
    dia.setLayout(layout)

    bbox.accepted.connect(dia.accept)
    bbox.rejected.connect(dia.reject)

    if dia.exec_():
        units = unit_combo.currentData()
        if units:
            overrides['units'] = units
        elif 'units' in overrides:
            del overrides['units']

        dec_pts = dplace_spin.value()
        if dec_pts >= 0:
            overrides['dec_pts'] = dec_pts
        elif 'dec_pts' in overrides:
            del overrides['dec_pts']

        color = color_combo.currentData()
        if color is not None:
            overrides['color'] = color
        elif 'color' in overrides:
            del overrides['color']

        if overrides:
            data_view.channel_overrides[channel] = overrides
        elif channel in data_view.channel_overrides:
            del data_view.channel_overrides[channel]

        update_channel_properties(data_view)
        data_view.values_change.emit()
