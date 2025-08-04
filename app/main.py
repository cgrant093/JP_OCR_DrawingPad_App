
import polars as pl
from PySide6.QtCore import (
    QPoint,
    QRect,
    Qt,
    Signal,
    Slot
)
from PySide6.QtGui import (
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QTabletEvent
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpacerItem,
    QVBoxLayout,
    QWidget
)
from util import *
dir_list = ['l', 's', 'd', 't', 'r', 'p', 'u', 'q']
uc_info = UnitCircleDivisionInfo(dir_list, pi / 8)



class MyScrollableButtonContainer(QWidget):
    clicked = Signal(str)

    def __init__(self):
        super().__init__()
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedSize(scroll_area_width, scroll_area_height)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Create button group
        self.button_container = QWidget()
        self.button_container.setStyleSheet(f'''
            QWidget {{
                background-color:       {matcha.name()};
            }}
            QPushButton {{
                background-color:       {sakura.name()};
                font-size:              {kanji_font}px;
            }}
        ''')
        # self.layout = QVBoxLayout(self.button_container)
        self.layout = QGridLayout(self.button_container)
        self.button_group = QButtonGroup(self.scroll_area)
        self.button_group.buttonClicked.connect(self.on_button_clicked)
        # Set the button_container as the widget for the scroll area
        self.scroll_area.setWidget(self.button_container)
        # Set group box
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)

    def add_button(self, text):
        button = QPushButton(text)
        button.setFixedSize(big_button_size, big_button_size)
        self.button_group.addButton(button)
        # self.layout.addWidget(button)
        button_id = -1 * self.button_group.id(button)
        self.layout.addWidget(button, button_id//2 - 1, button_id%2)
        self.update()

    def add_buttons(self, buttons_labels):
        for button_name in buttons_labels:
            self.add_button(button_name)

    def remove_button(self, button):
        self.button_group.removeButton(button)
        self.layout.removeWidget(button)
        button.deleteLater()
        self.update()

    def remove_all_buttons(self):
        for button in self.button_group.buttons():
            self.remove_button(button)

    @Slot(QAbstractButton)
    def on_button_clicked(self, button):
        self.clicked.emit(button.text())


class CalligraphyStroke:
    def __init__(self):
        self._pressures = []
        self._coordinates = []
        self._path = QPainterPath()

    @property
    def pressures(self) -> list:
        return self._pressures

    @property
    def coordinates(self) -> list:
        return self._coordinates

    @property
    def path(self) -> QPainterPath:
        if len(self._coordinates) > 1:
            self._path.lineTo(self._coordinates[-1])
        return self._path

    def append(self, new_data):
        self._pressures.append(new_data['pressure'])
        self._coordinates.append(new_data['coordinate'])
        if len(self._coordinates) == 1:
            self._path.moveTo(self._coordinates[0])


class CalligraphyCharacter:
    def __init__(self):
        self._strokes = {}
        self._max_stroke_id = 0
        self._empty_df = pl.DataFrame({'possibilities':'', 'primary_weight':[0]})
        self._stroke_directions = self._empty_df
        self._position_directions = self._empty_df

    @property
    def strokes(self) -> dict:
        return self._strokes

    @property
    def max_stroke_id(self) -> int:
        return self._max_stroke_id

    @property
    def stroke_directions(self) -> pl.DataFrame:
        return self._stroke_directions

    @property
    def position_directions(self) -> pl.DataFrame:
        return self._position_directions

    def new_stroke(self):
        self._max_stroke_id += 1
        self._strokes.update({self._max_stroke_id: CalligraphyStroke()})

    def update_stroke(self, new_data):
        if self._max_stroke_id == 0: return
        self._strokes[self._max_stroke_id].append(new_data)

    def _add_new_possiblities(self, data, df):
        df = df.with_columns(new_info=pl.lit(data, dtype=pl.List(pl.Struct)))
        df = df.explode('new_info')
        df = df.with_columns(
            (pl.col('possibilities') + pl.col('new_info').struct[0]),
            (pl.col('primary_weight').list.concat(pl.col('new_info').struct[1]))
        )
        return df.select(['possibilities', 'primary_weight'])

    def finish_stroke(self):
        if self._max_stroke_id == 0: return
        if not self._strokes[self._max_stroke_id].coordinates: return
        if len(self._strokes[self._max_stroke_id].coordinates) <= 1: return
        new_data = self.new_direction('stroke')
        self._stroke_directions = self._add_new_possiblities(new_data, self._stroke_directions)
        if self._max_stroke_id == 1: return
        new_data = self.new_direction('position')
        self._position_directions = self._add_new_possiblities(new_data, self._position_directions)

    def new_direction(self, direction_type='stroke'):
        curr_start = self._strokes[self._max_stroke_id].coordinates[0]
        if direction_type == 'position':
            prev_start = self._strokes[self._max_stroke_id-1].coordinates[0]
            diff = curr_start - prev_start
        else:
            curr_end = self._strokes[self._max_stroke_id].coordinates[-1]
            diff = curr_end - curr_start
        return uc_info.find_direction(diff)

    def remove_stroke(self, stroke_id=None):
        if self._max_stroke_id == 0: return
        if not stroke_id:
            stroke_id = self._max_stroke_id
            self._max_stroke_id -= 1
        self._strokes.pop(stroke_id)
        self._stroke_directions = self._stroke_directions.with_columns(
            (pl.col('possibilities').str.head(-1)),
            (pl.col('primary_weight').list.slice(0, -1))
        )
        if self._max_stroke_id == 1: return
        self._position_directions = self._position_directions.with_columns(
            (pl.col('possibilities').str.head(-1)),
            (pl.col('primary_weight').list.slice(0, -1))
        )

    def clear_all_strokes(self):
        self._strokes = {}
        self._max_stroke_id = 0
        self._stroke_directions = self._empty_df
        self._position_directions = self._empty_df


class _QWhiteboard(QWidget):
    stroke_drawn = Signal()

    def __init__(self):
        super().__init__()
        self.setTabletTracking(True)
        self.touching = 0
        self.character = CalligraphyCharacter()
        self.drawing_area = QRect(0, 0, draw_area_side_length, draw_area_side_length)  # Define your drawing area
        self.setFixedSize(self.drawing_area.size())
        self.setPalette(white)
        self.setAutoFillBackground(True)

    def _pen_in_draw_area(self, event_position):
        position = QPoint(event_position.x(), event_position.y())
        return self.drawing_area.contains(position)

    def tabletEvent(self, event):
        evt_pos = event.position()
        evt_type = event.type()
        if evt_type == QTabletEvent.TabletPress and self._pen_in_draw_area(evt_pos):
            self.touching = 1
            self.character.new_stroke()
        elif evt_type == QTabletEvent.TabletMove and self.touching == 1:
            if self._pen_in_draw_area(evt_pos):
                new_data = { 'coordinate' : evt_pos, 'pressure' : event.pressure() }
                self.character.update_stroke(new_data)
            else:
                self.touching = 0
                self.character.finish_stroke()
        elif evt_type == QTabletEvent.TabletRelease and self._pen_in_draw_area(evt_pos):
            if self.touching == 1:
                self.touching = 0
                self.character.finish_stroke()
            else:
                self.undo_last_stroke()
            self.stroke_drawn.emit()
        self.update()
        event.accept()

    def undo_last_stroke(self):
        if self.character.max_stroke_id <= 1:
            self.clear_drawing()
        self.character.remove_stroke()
        self.update()

    def clear_drawing(self):
        if self.character.max_stroke_id == 0: return
        self.character.clear_all_strokes()
        self.update()

    def paintEvent(self, event):
        if self.character.max_stroke_id == 0: return
        painter = QPainter(self)
        painter.setClipRect(self.drawing_area)
        painter.setPen(QPen(matcha_dark, 5))
        strokes = self.character.strokes.values()
        for stroke in strokes:
            painter.drawPath(stroke.path)


class MyInputWidget(QWidget):
    character_options = Signal(list)

    def __init__(self):
        super().__init__()
        self.radio_button_labels = {'漢字':'kanji', 'ひらがな':'hiragana', 'カタカナ':'katakana'}
        self._writing_system = list(self.radio_button_labels.values())[0]
        self.df = pl.read_csv(kanakanji_app_info_path)
        self.main_layout = QHBoxLayout()
        self._draw_area = _QWhiteboard()
        self._draw_area.stroke_drawn.connect(self.on_draw)
        self.main_layout.addWidget(self._draw_area)
        self._set_buttons()
        self.radio_button_group.button(-2).click()
        self.setLayout(self.main_layout)

    @property
    def writing_system(self) -> str:
        return self._writing_system

    @property
    def character(self) -> CalligraphyCharacter:
        return self._draw_area.character

    def _set_radio_button_panel(self):
        self.radio_container = QWidget()
        self.radio_layout = QFormLayout(self.radio_container)
        self.radio_button_group = QButtonGroup(self.radio_container)
        for button_name in self.radio_button_labels.keys():
            button = QRadioButton(button_name)
            self.radio_layout.addWidget(button)
            self.radio_button_group.addButton(button)
        self.radio_button_group.buttonClicked.connect(self.on_radio_change)
        self.radio_groupbox = QGroupBox('一つ選ぶ')
        self.radio_groupbox.setFixedSize(radio_width, radio_height)
        self.radio_main_layout = QVBoxLayout(self.radio_groupbox)
        self.radio_main_layout.addWidget(self.radio_container)
        self.button_layout.addWidget(self.radio_groupbox)

    def _new_drawing_button(self, label):
        button = QPushButton(label)
        button.setFixedSize(small_button_size, small_button_size)
        self.button_layout.addWidget(button)
        return button

    def _set_buttons(self):
        self.button_layout = QVBoxLayout()
        self._set_radio_button_panel()
        self.button_layout.addItem(QSpacerItem(radio_width, input_spacer_height))
        self.undo_button = self._new_drawing_button('↩')
        self.undo_button.clicked.connect(self.on_undo_clicked)
        self.clear_button = self._new_drawing_button('⮾')
        self.clear_button.clicked.connect(self.on_clear_clicked)
        self.main_layout.addLayout(self.button_layout)

    def refresh(self):
        self.update()
        self._draw_area.update()

    def _find_character_options(self):
        num_strokes = self._draw_area.character.max_stroke_id
        # print({
        #     'num_strokes' : num_strokes,
        #     'charClass' : self._writing_system,
        #     'stroke_directions' : self.character.stroke_directions,
        #     'position_directions' : self.character.position_directions,
        # })
        # print(self.character.stroke_directions)
        # print(self.character.position_directions)

        if num_strokes == 0: return []
        options = self.df.filter(
            pl.col('maxStrokeID') >= num_strokes,
            pl.col('charClass') == self._writing_system
        ).join_where(
            self.character.stroke_directions,
            pl.col('strokeDirections').str.starts_with(pl.col('possibilities'))
        )
        options = options.with_columns(
            primary_weight_stroke=pl.col('primary_weight').list.sum()
        )
        options = options.drop(['possibilities', 'primary_weight'])
        sort_by_cols = ['maxStrokeID', 'primary_weight_stroke', 'unicode']
        sort_by_desc = [False, True, False]

        if 1 < num_strokes < 6:
            options = options.join_where(
                self.character.position_directions,
                pl.col('positionDirections').str.starts_with(pl.col('possibilities'))
            )
            options = options.with_columns(
                primary_weight_position=pl.col('primary_weight').list.sum()
            )
            options = options.drop(['possibilities', 'primary_weight'])
            sort_by_cols.insert(2, 'primary_weight_position')
            sort_by_desc.insert(2, True)

        options = options.sort(sort_by_cols, descending=sort_by_desc)
        return options['kanakanji'].unique(maintain_order=True).limit(20).to_list()

    def _forward_character_options(self):
        options = self._find_character_options()
        self.character_options.emit(options)
        self.refresh()

    @Slot(QAbstractButton)
    def on_undo_clicked(self):
        self._draw_area.undo_last_stroke()
        self._forward_character_options()

    @Slot(QAbstractButton)
    def on_clear_clicked(self):
        self._draw_area.clear_drawing()
        self._forward_character_options()

    @Slot(QAbstractButton)
    def on_radio_change(self, button):
        self._writing_system = self.radio_button_labels[button.text()]
        self._forward_character_options()

    @Slot()
    def on_draw(self):
        self._forward_character_options()


class MyOutputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.main_layout = QHBoxLayout()

        self.text_output = QLineEdit()
        self.text_output.setReadOnly(True)
        self.main_layout.addWidget(self.text_output)

        self.undo_button = self._new_drawing_button('↩')
        self.undo_button.clicked.connect(self.on_undo_clicked)

        self.clear_button = self._new_drawing_button('⮾')
        self.clear_button.clicked.connect(self.on_clear_clicked)

        self.copy_button = self._new_drawing_button('⎘')
        self.copy_button.clicked.connect(self.on_copy_clicked)

        self.setLayout(self.main_layout)

    def _new_drawing_button(self, label):
        button = QPushButton(label)
        button.setFixedSize(small_button_size, small_button_size)
        self.main_layout.addWidget(button)
        return button

    def add_new_character(self, character):
        self.text_output.insert(character)
        self.update()

    @Slot(QAbstractButton)
    def on_undo_clicked(self):
        self.text_output.backspace()
        self.update()

    @Slot(QAbstractButton)
    def on_clear_clicked(self):
        self.text_output.clear()
        self.update()

    @Slot(QAbstractButton)
    def on_copy_clicked(self):
        self.text_output.selectAll()
        self.text_output.copy()
        self.text_output.end(False)
        # self.copy_output.emit(self.text_output.text())


class MyMainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setPalette(matcha)
        self.setAutoFillBackground(True)
        self.setStyleSheet(f'''
            QGroupBox {{
                color:                  black;
                font-size:              {label_font}px;
                font-weight:            bold;
            }}
            QLineEdit {{
                color:                  {matcha_dark.name()};
                background-color:       white;
                font-size:              {kanji_font}px;
            }}
            QRadioButton {{
                color:                  black;
                font-size:              {label_font}px;
            }}
            QRadioButton::indicator:checked {{
                width:                  {2*label_font/3}px;
                height:                 {2*label_font/3}px;
                border-radius:          {label_font/3+1}px;
                border:                 2px outset {sakura_dark.name()};
                background-color:       {matcha_dark.name()};
            }}
            QRadioButton::indicator:unchecked {{
                width:                  {2*label_font/3}px;
                height:                 {2*label_font/3}px;
                border-radius:          {label_font/3+1}px;
                border:                 2px outset {sakura_dark.name()};
                background-color:       {matcha.name()};
            }}
            QPushButton {{
                color:                  {matcha_dark.name()};
                background-color:       {sakura.name()};
                font-size:              {button_font}px;
            }}
            QScrollBar:vertical {{
                background:             {matcha.name()};
                width:                  30px;
            }}
        ''')

        # setup layout
        self._setup_main_layout()
        self.setLayout(self.main_layout)

        # connect to sub-widget signals
        self.input.character_options.connect(self._forward_character_options)
        self.options.clicked.connect(self._on_option_chosen)

    def _setup_decision_layout(self):
        self.decision_layout = QHBoxLayout()

        self.input_groupbox = QGroupBox('何かを描く')
        self.input_groupbox_layout = QHBoxLayout(self.input_groupbox)
        self.input = MyInputWidget()
        self.input_groupbox_layout.addWidget(self.input)
        self.decision_layout.addWidget(self.input_groupbox)

        self.decision_layout.addItem(QSpacerItem(top_spacer_width, big_button_size))

        self.scrollable_groupbox = QGroupBox('見つかった選択肢')
        self.scrollable_groupbox_layout = QVBoxLayout(self.scrollable_groupbox)
        self.options = MyScrollableButtonContainer()
        self.scrollable_groupbox_layout.addWidget(self.options)
        self.decision_layout.addWidget(self.scrollable_groupbox)

    def _setup_main_layout(self):
        self.main_layout = QVBoxLayout()

        self._setup_decision_layout()
        self.main_layout.addLayout(self.decision_layout)

        self.output_groupbox = QGroupBox('選ばれた')
        self.output_groupbox_layout = QHBoxLayout(self.output_groupbox)
        self.output = MyOutputWidget()
        self.output_groupbox_layout.addWidget(self.output)

        self.main_layout.addWidget(self.output_groupbox)

    @Slot(QAbstractButton)
    def _forward_character_options(self, character_options):
        self.options.remove_all_buttons()
        self.options.add_buttons(character_options)

    @Slot(QAbstractButton)
    def _on_option_chosen(self, selected_character):
        self.output.add_new_character(selected_character)


class MyApplication(QApplication):
    def __init__(self):
        super().__init__()
        self.setApplicationName('文字翻訳者')
        self.setWindowIcon(QIcon(green_tea_icon_path))
        self.widget = MyMainWidget()
        self.widget.resize(window_width, window_height)
        self.widget.show()


def main():
    app = MyApplication()
    app.exec()


if __name__ == '__main__':
    main()