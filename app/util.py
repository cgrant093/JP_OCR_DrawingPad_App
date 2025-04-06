# '''

import math
# import pandas as pd

import pyglet
from pyglet.image import AbstractImage
from pyglet.gl import GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
from pyglet.graphics import Batch, Group
from pyglet.graphics.shader import ShaderProgram
from pyglet.gui import PushButton, TextEntry
from pyglet.shapes import Rectangle
from pyglet_new_shapes import Stroke

import os
import re
import time


class MyTablet:
    def __init__(self, app):
        self.max_int16 = 65534
        self._set_tablet_()
        self.device.open()
        self._set_controls_()
        # self.set_window_attrs(app.x_factor, app.sx, app.sy)
        self.x_factor = app.x_factor
        self.sx = app.sx
        self.sy = app.sy

    def _set_tablet_(self):
        tablets = [ 
            tablet
            for tablet in pyglet.input.get_devices()
            if 'tablet' in tablet.name.lower()
            if tablet.get_controls()
        ]
        self.device = tablets[0]

    def _set_controls_(self):
        controls = {
            control.raw_name : control
            for control in self.device.get_controls()
        }
        # AbsoluteAxis Controls
        self.P = controls['Tip Pressure']
        self.X = controls['X Axis']     
        self.Y = controls['Y Axis']
        self.THETA = controls['X Tilt']
        self.PHI = controls['Y Tilt']
        # Button Controls
        self.in_range = controls['In Range']
        self.contact = controls['Tip Switch']
        self.pen_btn = controls['Barrel Switch']
        # self.Invert = controls['Invert']      # unknown
        # self.Eraser = controls['Eraser']      # unknown

        # controls['X Axis'].set_handler('on_change', print_change)
            # x location of screen work area. ranges from 0 to 65535. left = 0.

        # controls['Y Axis'].set_handler('on_change', print_change)
            # y location of screen work area. ranges from 0 to 65535. top = 0.
            
        # controls['Tip Pressure'].set_handler('on_change', print_change)
        #     # works. ranges from 0 to 65535 (highest number for 16 bit integer)
            
        # controls['X Tilt'].set_handler('on_change', print_change)
        #     # pen's tilt angle relative to the tablet (in x direction). 
        #     # assuming the same range as above, but cannot reach outer 50%
            
        # controls['Y Tilt'].set_handler('on_change', print_change)
        #     # pen's tilt angle relative to the tablet (in y direction). 
        #     # assuming the same range as above, but cannot reach outer 50%
        
        # controls['In Range'].set_handler('on_press', print_press)
        # controls['In Range'].set_handler('on_release', print_rel)
        # controls['In Range'].set_handler('on_change', print_change)
        #     # only presses does on_change once and on_press once

        # controls['Tip Switch'].set_handler('on_press', print_press)
        # controls['Tip Switch'].set_handler('on_release', print_rel)
        # controls['Tip Switch'].set_handler('on_change', print_change)
        #     # all three work. on_change goes from True to False when on_press and on_release activate, respectively

        # controls['Barrel Switch'].set_handler('on_press', print_press)
        # controls['Barrel Switch'].set_handler('on_release', print_rel)
        # controls['Barrel Switch'].set_handler('on_change', print_change)
        #     # all three work with bottom button. on_change goes from True to False when on_press and on_release activate, respectively
        #     # nothing with top button

        # controls['Invert'].set_handler('on_press', print_press)
        # controls['Invert'].set_handler('on_release', print_rel)
        # controls['Invert'].set_handler('on_change', print_change)
        #     # doesn't do anything from what I can figure out

        # controls['Eraser'].set_handler('on_press', print_press)
        # controls['Eraser'].set_handler('on_release', print_rel)
        # controls['Eraser'].set_handler('on_change', print_change)
        #     # doesn't do anything from what I can figure out

    def get_pressure(self):
        return self.P.value/3000
    
    def get_x(self):
        return self.x_factor*self.sx*self.X.value/self.max_int16
    
    def get_y(self):
        return self.sy*(1 - self.Y.value/self.max_int16)
    
    def get_coordinates(self):
        return (self.get_x(), self.get_y())
    
    def get_data(self):
        return {
            'pressure' : self.get_pressure(),
            'coordinate' : self.get_coordinates()
        }


class MyApp:
    def __init__(self, batch):
        self.app_name = 'Japanese Character Guesser'
        self._set_app_window_()
        self.calc_draw_area(batch)
        self.create_widgets(batch)

    def _find_screen_(self):
        self.display = pyglet.display.get_display()
        screens = self.display.get_screens()
        self.x_factor = len(screens)
        self.screen = screens[0]
        if len(screens) > 1:
            self.screen = screens[1]
        self.sx = self.screen.width
        self.sy = self.screen.height

    def _set_app_window_(self):
        self._find_screen_()
        self.window = pyglet.window.Window(fullscreen=True, screen=self.screen)
        self.window.set_fullscreen(False)
        self.window.maximize()
        self.window.set_size(width=self.sx, height=self.sy)
        self.window.set_caption(self.app_name)
        
        # icon1 = pyglet.image.load('16x16.png')
        # icon2 = pyglet.image.load('32x32.png')
        # window.set_icon(icon1, icon2)

    def run(self):
        pyglet.app.run()

    def calc_draw_area(self, batch):
        border_thickness = 7
        width = 0.85*self.sy
        back_width = width + 2*border_thickness
        xb = self.sx - back_width
        yb = 0
        x0 = self.sx - border_thickness - width
        y0 = border_thickness 
        # self.draw_area.x0
        self.bigger_rectangle = Rectangle(
            x=xb, y=yb, 
            width=back_width, height=back_width,
            batch=batch, color=(164, 64, 51) 
        )
        self.draw_area = Rectangle(
            x=x0, y=y0, batch=batch,
            width=width, height=width 
        )

    def create_widgets(self, batch):
        blk_btn = pyglet.resource.image('brown_button.png')
        self.pushbutton = PushButton(
            x=10, y=0.9*self.sy, pressed=blk_btn, 
            unpressed=blk_btn, batch=batch
        )


class CalligraphyStroke:
    def __init__(self):
        self._pressures = []
        self._coordinates = []
        self._min_distance = 15

    @property
    def pressures(self) -> list:
        return self._pressures

    @property
    def coordinates(self) -> list:
        return self._coordinates
    
    def append(self, new_data):
        if self._coordinates:
            distance = math.dist(self._coordinates[-1], new_data['coordinate'])
            if distance < self._min_distance:
                return 0
        self._pressures.append(new_data['pressure']) 
        self._coordinates.append(new_data['coordinate'])

    def create_shape(self, batch, color=None):
        if not color:
            return Stroke(*self._coordinates, thicknesses=self._pressures, batch=batch)
        return Stroke(*self._coordinates, thicknesses=self._pressures, color=color, batch=batch)
    

class CalligraphyCharacter:

    def __init__(self, draw_area):
        self.max_stroke_id = 0
        self.strokes = {}
        self.shapes = {}
        self.draw_area = draw_area

    def new_stroke(self):
        self.max_stroke_id += 1
        self.strokes.update({self.max_stroke_id: CalligraphyStroke()})

    def _find_stroke_id(self, stroke_id):
        if not stroke_id:
            stroke_id = self.max_stroke_id
        return stroke_id
    
    def _stroke_origin_in_draw_area_(self, stroke_id):
        if self.strokes[stroke_id].coordinates[0] not in self.draw_area:
            return 0
        return 1
    
    def _num_coords_in_stroke_(self, stroke_id):
        return len(self.strokes[stroke_id].coordinates)

    def append_stroke_data(self, data, stroke_id=None):
        stroke_id = self._find_stroke_id(stroke_id)
        self.strokes[stroke_id].append(data)

    def draw_stroke(self, batch, color=None, stroke_id=None):
        stroke_id = self._find_stroke_id(stroke_id)
        if self._num_coords_in_stroke_(stroke_id) < 2:
            return 0
        if not self._stroke_origin_in_draw_area_(stroke_id):
            return 0
        self.shapes.update({stroke_id: self.strokes[stroke_id].create_shape(batch, color)})

    def finish_stroke(self, stroke_id=None):
        if not self.strokes:
            return 0
        stroke_id = self._find_stroke_id(stroke_id)
        if stroke_id in self.shapes.keys():
            return 0
        if not self._num_coords_in_stroke_(stroke_id):
            return 0
        if not self._stroke_origin_in_draw_area_(stroke_id):
            return 0
        self.strokes.pop(stroke_id)

    # def delete_stroke(self, stroke_id):
    #     if not self.strokes:
    #         return 0
    #     self.shapes[stroke_id].delete()
    #     self.shapes.pop(stroke_id)
    #     self.strokes.pop(stroke_id)

    def clear_all(self):
        for shape in self.shapes.values():
            shape.delete()
        self.max_stroke_id = 0
        self.strokes = {}
        self.shapes = {}
        


