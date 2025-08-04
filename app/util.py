
from math import atan2, pi
import os
from PySide6.QtGui import QColor
import sys


# to make sure pyinstaller works with the extra resources
def resource_path(relative_path):
    """Get the absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

kanakanji_app_info_path = resource_path('kanakanji_app_info.csv')
green_tea_icon_path = resource_path('green_tea_icon.png')

# app custom colors
matcha = QColor('#E4F8BA')
matcha_dark = QColor('#52681D')
sakura = QColor('#FFBAC7')
sakura_dark = QColor('#DA5286')
bamboo = QColor('#DA6304')
white = QColor('white')

# app object sizes
kanji_font = 50
button_font = 36
label_font = 21
draw_area_side_length = 500
radio_width = 140
radio_height = 140
big_button_size = 140
small_button_size = kanji_font + 20
input_spacer_height = draw_area_side_length - radio_height - big_button_size - 10
scroll_area_width = 2*1.2*big_button_size
scroll_area_height = draw_area_side_length
window_wh_ratio = 1200/720
window_height = draw_area_side_length + small_button_size + 2*label_font + 100 #720
window_width = draw_area_side_length + radio_width + scroll_area_width + 50 #1200
top_spacer_width = 0

# unit circle divisions info
class UnitCircleDivisionInfo:
    def __init__(self, directions_list, pure_division_width):
        self.directions = directions_list + [directions_list[0]]
        self.num_directions = len(self.directions)
        self.divisions_width = 2*pi/(self.num_directions-1)
        self.directions_mid_angles = [i*self.divisions_width-pi for i in range(self.num_directions)]
        self.pure_division_width = pure_division_width
        self.sub_division_boundary_angles = list({
            0,
            self.pure_division_width / 2,
            self.divisions_width / 2,
            self.divisions_width - self.pure_division_width / 2,
            self.divisions_width
        })
        self.sub_division_boundary_angles.sort()
        self.all_division_boundary_angles = [
            [   self.directions_mid_angles[division_index] + phi
                for phi in self.sub_division_boundary_angles  ]
            for division_index in range(self.num_directions-1)
        ]

    def find_direction(self, diff):
        '''
        directions key:
            u = y+ (within approx +/- 5 deg)
            d = y- (within approx +/- 5 deg)
            r = x+ (within approx +/- 5 deg)
            l = x- (within approx +/- 5 deg)
            p ~= Q1 (everything remaining after above removed)
            q ~= Q2 (everything remaining after above removed)
            s ~= Q3 (everything remaining after above removed)
            t ~= Q4 (everything remaining after above removed)
            o = starts and ends in same place
        '''

        possibilities = []
        is_primary = 1

        dx = diff.x()
        dy = -diff.y()
        theta = atan2(dy, dx) # result is element of (-pi, pi]

        # if the stroke it starts close to where if finishes
        close_upper_boundary = 25
        if abs(dx) < close_upper_boundary and abs(dy) < close_upper_boundary:
            possibilities.append(('o', is_primary))
            is_primary = 0

        # this unit circle has n divisions as opposed to 4 quadrants)
        division_index = sum([
            i if self.directions_mid_angles[i] < theta <= self.directions_mid_angles[i+1] else 0
            for i in range(self.num_directions-1)
        ])
        # this specific sub-division's boundary angles
        this_divisions_angles = self.all_division_boundary_angles[division_index]
        sub_division_index = sum([
            i if this_divisions_angles[i] < theta <= this_divisions_angles[i+1] else 0
            for i in range(len(this_divisions_angles)-1)
        ])

        # there should be only four sub-division choices
        if sub_division_index == 0:
            possibilities.append((self.directions[division_index], is_primary))
        elif sub_division_index == 1:
            possibilities.append((self.directions[division_index], is_primary))
            possibilities.append((self.directions[division_index+1], 0))
        elif sub_division_index == 2:
            possibilities.append((self.directions[division_index], 0))
            possibilities.append((self.directions[division_index+1], is_primary))
        else:
            possibilities.append((self.directions[division_index+1], is_primary))

        return possibilities