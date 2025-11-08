from kivy.uix.button import Button
from kivy.properties import ListProperty, NumericProperty
from kivy.lang import Builder
from global_vars import *

# Define the Kivy styling inline
Builder.load_string('''
<MyDropDownButton>:
    button_color : root.button_color
    font_size: self.width * 0.04
    background_color: 0, 0, 0, 0
    background_normal: ""
    font_name: "graphics/fonts/pixel.TTF"
    canvas.before:
        Color:
            rgba: self.button_color
        RoundedRectangle:
            size: self.size
            pos: self.pos
            radius: [0]
        Color:
            rgba: root.line_button_color
        Line:
            rounded_rectangle : self.x+2, self.y, self.width-4, self.height-2, 0, 0, 0, 0, 100
            width : root.line_width
''')


class MyDropDownButton(Button):
    """A rounded button with borders for dropdown menus"""
    button_color = ListProperty(LIGHT_BLUE)
    line_button_color = ListProperty(WHITE)
    line_width = NumericProperty(2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_color = LIGHT_BLUE[:]
        self.pressed_color = DARK_BLUE[:]
        
    def on_press(self):
        """Change color when button is pressed"""
        self.button_color = self.pressed_color
        
    def on_release(self):
        """Restore original color when button is released"""
        self.button_color = self.original_color
